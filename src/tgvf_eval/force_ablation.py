from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from revisit_vlm.tgvf import FOVEATE_END, FOVEATE_START
from revisit_vlm.tgvf_capture import (
    _vision_content_items,
    build_qwen2vl_tgvf_inputs,
    capture_tgvf_single_pass,
    trim_capture_target_at_delimiters,
)
from revisit_vlm.tgvf_foveal import (
    Qwen2VLPreMergeVisualHook,
    append_answer_turn_and_open,
    append_fvt_result_and_open_answer_turn,
    continue_generation_from_state,
    finalize_tgvf_output_with_frozen_qwen_merger,
)
from revisit_vlm.tgvf_training import (
    ForcedFoveationBracketWrapper,
    ForcedFoveationWrapper,
    Qwen2VLMergedVisualHook,
)
from tgvf_eval.adapters import BenchmarkRegistry, BenchmarkSample
from tgvf_eval.config import (
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEED,
    DEFAULT_TOOLS_ROOT,
    image_budget_kwargs,
)
from tgvf_eval.progress import configure_quiet_external_progress, iter_progress
from tgvf_eval.prompts import build_direct_prompt, build_force_prompt
from tgvf_eval.results import make_run_paths, write_config_yaml

FORCE_ABLATION_MODES = (
    "direct_qwen",
    "new_user_turn_only",
    "new_user_turn_repeat_options",
    "force_target_no_D",
    "force_target_no_D_repeat_options",
    "force_correct_D",
    "force_correct_D_repeat_options",
    "force_zero_D_repeat_options",
    "force_random_D_repeat_options",
    "force_wrong_D_repeat_options",
    "force_vmerge_D_repeat_options",
    "force_correct_D_legacy_text_positions",
    "force_correct_D_native_pseudo_image",
    "force_zero_D_native_pseudo_image",
    "force_random_D_native_pseudo_image",
)

DEFAULT_FORCE_ABLATIONS = FORCE_ABLATION_MODES[:10]
NATIVE_APPEND_COMPARE_ABLATIONS = (
    "direct_qwen",
    "force_correct_D_legacy_text_positions",
    "force_correct_D_native_pseudo_image",
    "force_zero_D_native_pseudo_image",
    "force_random_D_native_pseudo_image",
)
GENERIC_TARGETS = {
    "relevant visual evidence needed to answer the question",
    "the relevant visual evidence",
    "relevant visual evidence",
    "the image",
    "image",
    "the scene",
    "scene",
    "the object",
    "object",
    "the answer",
    "answer",
    "something",
    "visual target description",
    "target color",
    "target object",
    "target region",
    "target",
}


@dataclass(frozen=True)
class AblationSpec:
    name: str
    fvt_mode: str
    repeat_options: bool
    force_target: bool
    direct: bool = False
    new_user_turn: bool = False
    fvt_append_mode: str | None = None


def validate_force_ablation_mode(value: str) -> str:
    meta_modes = {"all", "native_append_compare"}
    if value not in meta_modes and value not in FORCE_ABLATION_MODES:
        raise ValueError(
            f"Invalid force ablation mode {value!r}. Expected one of {sorted(meta_modes | set(FORCE_ABLATION_MODES))}"
        )
    return value


def ablation_spec(name: str) -> AblationSpec:
    validate_force_ablation_mode(name)
    mapping = {
        "direct_qwen": AblationSpec(name, fvt_mode="none", repeat_options=False, force_target=False, direct=True),
        "new_user_turn_only": AblationSpec(name, fvt_mode="none", repeat_options=False, force_target=False, new_user_turn=True),
        "new_user_turn_repeat_options": AblationSpec(name, fvt_mode="none", repeat_options=True, force_target=False, new_user_turn=True),
        "force_target_no_D": AblationSpec(name, fvt_mode="none", repeat_options=False, force_target=True),
        "force_target_no_D_repeat_options": AblationSpec(name, fvt_mode="none", repeat_options=True, force_target=True),
        "force_correct_D": AblationSpec(name, fvt_mode="correct", repeat_options=False, force_target=True),
        "force_correct_D_repeat_options": AblationSpec(name, fvt_mode="correct", repeat_options=True, force_target=True),
        "force_zero_D_repeat_options": AblationSpec(name, fvt_mode="zero", repeat_options=True, force_target=True),
        "force_random_D_repeat_options": AblationSpec(name, fvt_mode="random", repeat_options=True, force_target=True),
        "force_wrong_D_repeat_options": AblationSpec(name, fvt_mode="wrong", repeat_options=True, force_target=True),
        "force_vmerge_D_repeat_options": AblationSpec(name, fvt_mode="vmerge", repeat_options=True, force_target=True),
        "force_correct_D_legacy_text_positions": AblationSpec(
            name, fvt_mode="correct", repeat_options=True, force_target=True, fvt_append_mode="legacy_text_positions"
        ),
        "force_correct_D_native_pseudo_image": AblationSpec(
            name, fvt_mode="correct", repeat_options=True, force_target=True, fvt_append_mode="qwen_native_pseudo_image"
        ),
        "force_zero_D_native_pseudo_image": AblationSpec(
            name, fvt_mode="zero", repeat_options=True, force_target=True, fvt_append_mode="qwen_native_pseudo_image"
        ),
        "force_random_D_native_pseudo_image": AblationSpec(
            name, fvt_mode="random", repeat_options=True, force_target=True, fvt_append_mode="qwen_native_pseudo_image"
        ),
    }
    return mapping[name]


def target_quality_flags(
    target: str,
    *,
    force_target_mode: str = "generated",
    fixed_force_target: str | None = None,
) -> dict[str, bool]:
    text = (target or "").strip()
    norm = _normalize_target(text)
    fixed_norm = _normalize_target(fixed_force_target or "")
    is_empty = not text
    is_fixed = bool(force_target_mode == "fixed" or (fixed_norm and norm == fixed_norm))
    is_generic = is_empty or norm in GENERIC_TARGETS
    if norm.startswith("visual target description"):
        is_generic = True
    malformed = any(token in text for token in (FOVEATE_START, FOVEATE_END, "<|im_end|>", "<|endoftext|>"))
    malformed = malformed or "|" in text or len(text.split()) > 24
    return {
        "target_is_generic": bool(is_generic),
        "target_was_fixed": bool(is_fixed),
        "target_is_empty": bool(is_empty),
        "target_is_malformed": bool(malformed),
    }


def answer_distribution_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predicted = Counter(str(row.get("parsed_answer") or "") for row in rows if row.get("parsed_answer"))
    gold = Counter(str(row.get("gold_answer") or "") for row in rows if row.get("gold_answer"))
    by_pred: dict[str, list[float]] = defaultdict(list)
    by_category: dict[str, list[float]] = defaultdict(list)
    parse_fail_count = 0
    invalid_answer_count = 0
    for row in rows:
        parsed = str(row.get("parsed_answer") or "")
        score = float(row.get("score") or 0.0)
        if not parsed:
            parse_fail_count += 1
        elif parsed not in {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"}:
            invalid_answer_count += 1
        if parsed:
            by_pred[parsed].append(score)
        category = row.get("category") or (row.get("metadata") or {}).get("category")
        if category:
            by_category[str(category)].append(score)
    return {
        "predicted_letter_histogram": dict(sorted(predicted.items())),
        "gold_letter_histogram": dict(sorted(gold.items())),
        "correct_by_predicted_letter": {
            key: {
                "correct": int(sum(values)),
                "total": len(values),
                "accuracy": _mean(values),
            }
            for key, values in sorted(by_pred.items())
        },
        "accuracy_by_category": {key: _mean(values) for key, values in sorted(by_category.items())},
        "parse_fail_count": parse_fail_count,
        "invalid_answer_count": invalid_answer_count,
    }


def target_quality_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target_rows = [row for row in rows if row.get("triggered") or row.get("foveation_target")]
    if not target_rows:
        return {
            "foveation_target_unique_count": 0,
            "top_20_foveation_targets": [],
            "avg_target_length_words": None,
            "median_target_length_words": None,
            "generic_target_rate": None,
            "fixed_target_rate": None,
            "empty_target_rate": None,
            "malformed_target_rate": None,
        }
    targets = [str(row.get("foveation_target") or "").strip() for row in target_rows]
    nonempty = [target for target in targets if target]
    lengths = [len(target.split()) for target in nonempty]
    counts = Counter(targets)
    return {
        "foveation_target_unique_count": len({_normalize_target(target) for target in nonempty}),
        "top_20_foveation_targets": [
            {"target": target, "count": count} for target, count in counts.most_common(20)
        ],
        "avg_target_length_words": _mean(lengths),
        "median_target_length_words": statistics.median(lengths) if lengths else None,
        "generic_target_rate": _rate(target_rows, "target_is_generic"),
        "fixed_target_rate": _rate(target_rows, "target_was_fixed"),
        "empty_target_rate": _rate(target_rows, "target_is_empty"),
        "malformed_target_rate": _rate(target_rows, "target_is_malformed"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VSTAR force-mode TGVF ablations.")
    parser.add_argument("--benchmark", choices=["vstar_bench"], default="vstar_bench")
    parser.add_argument("--tier", choices=["light", "medium", "full"], default="light")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force-ablation-mode", choices=["all", "native_append_compare", *FORCE_ABLATION_MODES], default="all")
    parser.add_argument("--fvt-append-mode", choices=["legacy_text_positions", "qwen_native_pseudo_image"], default="qwen_native_pseudo_image")
    parser.add_argument("--repeat-options-in-continuation", type=_bool, default=False)
    parser.add_argument("--force-target-mode", choices=["generated", "fixed"], default="generated")
    parser.add_argument("--fixed-force-target", default=None)
    parser.add_argument("--force-target-max-tokens", type=int, default=32)
    parser.add_argument("--dump-target-quality-report", type=_bool, default=True)
    parser.add_argument("--dump-answer-distribution", type=_bool, default=True)
    parser.add_argument("--model-path", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--processor-path", default=None)
    parser.add_argument("--tgvf-checkpoint", required=True)
    parser.add_argument("--tgvf-variant", default="target_slot_foveal_cross_merger")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--answer-max-new-tokens", type=int, default=128)
    parser.add_argument("--capture-max-new-tokens", type=int, default=128)
    parser.add_argument("--capture-layer", type=int, default=-1)
    parser.add_argument("--image-budget", choices=["low", "mid", "high"], default="mid")
    parser.add_argument("--cot", type=_bool, default=False)
    parser.add_argument("--benchmark-root", default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--tools-root", default=DEFAULT_TOOLS_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default="vstar_force_ablation")
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--no-progress", action="store_false", dest="progress")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_vstar_force_ablation(args)


def run_vstar_force_ablation(args: argparse.Namespace) -> dict[str, Any]:
    configure_quiet_external_progress()
    if args.force_target_mode == "fixed" and not args.fixed_force_target:
        raise ValueError("--force-target-mode fixed requires --fixed-force-target")
    if args.force_ablation_mode == "all":
        modes = list(DEFAULT_FORCE_ABLATIONS)
    elif args.force_ablation_mode == "native_append_compare":
        modes = list(NATIVE_APPEND_COMPARE_ABLATIONS)
    else:
        modes = [args.force_ablation_mode]
    adapter = BenchmarkRegistry.get(
        args.benchmark,
        benchmark_root=args.benchmark_root,
        tools_root=args.tools_root,
    )
    samples = adapter.sample(tier=args.tier, limit=args.limit, seed=args.seed)
    paths = make_run_paths(args.output_root, run_id=args.run_id)
    config_payload = {
        "run_id": args.run_id,
        "benchmark": args.benchmark,
        "tier": args.tier,
        "limit": args.limit,
        "seed": args.seed,
        "force_ablation_modes": modes,
        "model": {
            "qwen_model_path": args.model_path,
            "processor_path": args.processor_path,
            "dtype": args.dtype,
            "device": args.device,
            "answer_max_new_tokens": args.answer_max_new_tokens,
            "capture_max_new_tokens": args.capture_max_new_tokens,
        },
        "tgvf": {
            "checkpoint": args.tgvf_checkpoint,
            "variant": args.tgvf_variant,
            "force_target_mode": args.force_target_mode,
            "fixed_force_target": args.fixed_force_target,
            "force_target_max_tokens": args.force_target_max_tokens,
            "fvt_append_mode": args.fvt_append_mode,
            "second_full_forward_allowed": False,
        },
        "image": {"budget": args.image_budget},
        "cot": {"enabled": bool(args.cot)},
    }
    write_config_yaml(paths.root / "config.yaml", config_payload)

    model, processor, foveal_module, device = _load_model_and_module(args)
    runner = _AblationRunner(model, processor, foveal_module, device, args)
    comparison = []
    for mode in modes:
        spec = ablation_spec(mode)
        rows = []
        previous_correct_d_cpu = None
        desc = f"vstar force ablation {mode}"
        for sample in iter_progress(samples, desc=desc, total=len(samples), enabled=args.progress):
            row, previous_correct_d_cpu = runner.run_sample(
                sample,
                spec,
                previous_correct_d_cpu=previous_correct_d_cpu,
            )
            parsed = adapter.parse_prediction(row["raw_output"], sample)
            row["parsed_answer"] = parsed
            row["predicted_letter"] = parsed
            row["score"] = 1.0 if parsed and sample.gold_answer and parsed == sample.gold_answer else 0.0
            if isinstance(row.get("debug_metadata"), dict):
                row["debug_metadata"]["predicted_letter"] = parsed
            rows.append(row)
        _write_jsonl(paths.predictions / f"{args.benchmark}__force_ablation__{mode}.jsonl", rows)
        _write_jsonl(
            paths.raw_outputs / f"{args.benchmark}__force_ablation__{mode}.jsonl",
            [
                {
                    "sample_id": row["sample_id"],
                    "ablation": row["ablation"],
                    "raw_output": row["raw_output"],
                    "error": row.get("error"),
                }
                for row in rows
            ],
        )
        summary = summarize_ablation_rows(
            rows,
            run_id=args.run_id,
            ablation_name=mode,
            benchmark=args.benchmark,
            include_answer_distribution=args.dump_answer_distribution,
            include_target_quality=args.dump_target_quality_report,
        )
        (paths.reports / f"{args.benchmark}__force_ablation__{mode}.summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
        )
        comparison.append(summary)
    comparison_payload = {
        "run_id": args.run_id,
        "benchmark": args.benchmark,
        "num_samples": len(samples),
        "ablation_summaries": comparison,
        "decision_hints": decision_hints(comparison),
    }
    (paths.reports / f"{args.benchmark}__force_ablation_comparison.json").write_text(
        json.dumps(comparison_payload, indent=2, ensure_ascii=False) + "\n"
    )
    return comparison_payload


class _AblationRunner:
    def __init__(self, model: Any, processor: Any, foveal_module: Any, device: Any, args: argparse.Namespace) -> None:
        self.model = model
        self.processor = processor
        self.foveal_module = foveal_module
        self.device = device
        self.args = args
        self.model.eval()
        self.foveal_module.eval()

    def run_sample(
        self,
        sample: BenchmarkSample,
        spec: AblationSpec,
        *,
        previous_correct_d_cpu: torch.Tensor | None,
    ) -> tuple[dict[str, Any], torch.Tensor | None]:
        start = time.perf_counter()
        try:
            if spec.direct:
                raw_output, output_tokens = self._run_direct(sample)
                debug: dict[str, Any] = {}
                foveation_target = ""
                triggered = False
                visual_token_count = 0
            elif spec.new_user_turn:
                state = self._prefill_direct_state(sample)
                appended = append_answer_turn_and_open(
                    model=self.model,
                    tokenizer_or_processor=self.processor,
                    generation_state=state,
                    benchmark_answer_format="multiple_choice",
                    option_letters=_letters(sample),
                    original_question=_question_without_choices(sample.question),
                    option_texts=sample.choices,
                    repeat_options_in_continuation=spec.repeat_options,
                    device=self.device,
                )
                continuation = continue_generation_from_state(
                    model=self.model,
                    tokenizer_or_processor=self.processor,
                    generation_state=appended,
                    max_new_tokens=self.args.answer_max_new_tokens,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
                raw_output = continuation.generated_text
                output_tokens = len(continuation.generated_ids)
                debug = dict(appended.debug_metadata)
                debug["first_generated_token_after_append"] = _first_token_debug(
                    self.processor.tokenizer, continuation.generated_ids
                )
                debug["stopped_before_generating_answer"] = _stopped_before_answer(
                    self.processor.tokenizer, continuation.generated_ids
                )
                foveation_target = ""
                triggered = False
                visual_token_count = 0
            else:
                raw_output, output_tokens, debug, foveation_target, visual_token_count, previous_correct_d_cpu = self._run_force(
                    sample,
                    spec,
                    previous_correct_d_cpu=previous_correct_d_cpu,
                )
                triggered = True
            wall = time.perf_counter() - start
            target_flags = target_quality_flags(
                foveation_target,
                force_target_mode=self.args.force_target_mode if spec.force_target else "generated",
                fixed_force_target=self.args.fixed_force_target,
            )
            row = _base_row(sample, spec)
            row.update(
                {
                    "raw_output": raw_output,
                    "gold_answer": sample.gold_answer,
                    "score": None,
                    "foveation_target": foveation_target,
                    "triggered": triggered,
                    "visual_token_count": visual_token_count,
                    "output_tokens": output_tokens,
                    "fvt_append_mode": debug.get("append_mode") or ((spec.fvt_append_mode or self.args.fvt_append_mode) if spec.fvt_mode != "none" else None),
                    "used_logits_source": debug.get("used_logits_source"),
                    "cache_preserved": bool(debug.get("cache_preserved")) if debug else False,
                    "second_full_forward_used": bool(debug.get("second_full_forward_used", False)),
                    "immediate_im_end": raw_output.strip() == "<|im_end|>",
                    "wall_time_sec": wall,
                    "debug_metadata": debug,
                    "error": None,
                    **target_flags,
                }
            )
            return row, previous_correct_d_cpu
        except Exception as exc:
            row = _base_row(sample, spec)
            row.update(
                {
                    "raw_output": "",
                    "gold_answer": sample.gold_answer,
                    "parsed_answer": "",
                    "score": 0.0,
                    "foveation_target": "",
                    "triggered": bool(spec.force_target),
                    "visual_token_count": 0,
                    "output_tokens": 0,
                    "used_logits_source": None,
                    "cache_preserved": False,
                    "second_full_forward_used": False,
                    "immediate_im_end": False,
                    "wall_time_sec": time.perf_counter() - start,
                    "debug_metadata": {},
                    "error": f"{type(exc).__name__}: {exc}",
                    **target_quality_flags(""),
                }
            )
            return row, previous_correct_d_cpu

    def _run_direct(self, sample: BenchmarkSample) -> tuple[str, int]:
        prompt = build_direct_prompt(sample.question, cot_enabled=self.args.cot)
        messages = [
            {
                "role": "user",
                "content": [
                    *_vision_content_items(sample.primary_media, self._image_kwargs()),
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = build_qwen2vl_tgvf_inputs(
            self.processor,
            image=sample.primary_media,
            question=prompt,
            messages=messages,
        )
        inputs = _move_tensors(inputs, self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.args.answer_max_new_tokens,
            do_sample=False,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        prompt_len = inputs["input_ids"].shape[-1]
        generated = outputs[0, prompt_len:].detach().cpu().tolist()
        text = self.processor.tokenizer.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return text, len(generated)

    def _prefill_direct_state(self, sample: BenchmarkSample) -> SimpleNamespace:
        prompt = build_direct_prompt(sample.question, cot_enabled=self.args.cot)
        messages = [
            {
                "role": "user",
                "content": [
                    *_vision_content_items(sample.primary_media, self._image_kwargs()),
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = build_qwen2vl_tgvf_inputs(
            self.processor,
            image=sample.primary_media,
            question=prompt,
            messages=messages,
        )
        inputs = _move_tensors(inputs, self.device)
        outputs = self.model(**inputs, use_cache=True, return_dict=True)
        model_kwargs = _resume_model_kwargs(inputs, outputs)
        return SimpleNamespace(
            past_key_values=outputs.past_key_values,
            attention_mask=inputs.get("attention_mask"),
            cache_position=inputs.get("cache_position"),
            input_ids=inputs.get("input_ids"),
            last_logits=outputs.logits,
            model_kwargs=model_kwargs,
            debug_metadata={"prefill_only": True, "second_full_forward_used": False},
        )

    def _run_force(
        self,
        sample: BenchmarkSample,
        spec: AblationSpec,
        *,
        previous_correct_d_cpu: torch.Tensor | None,
    ) -> tuple[str, int, dict[str, Any], str, int, torch.Tensor | None]:
        capture_model, forced_token_count = self._force_capture_model()
        force_prompt = build_force_prompt(sample.question, cot_enabled=self.args.cot)
        with Qwen2VLPreMergeVisualHook(self.model) as visual_hook, Qwen2VLMergedVisualHook(self.model) as merged_hook:
            capture = capture_tgvf_single_pass(
                capture_model,
                self.processor,
                image=sample.primary_media,
                question=force_prompt,
                max_new_tokens=max(forced_token_count + 4, self.args.capture_max_new_tokens if not forced_token_count else 1),
                device=self.device,
                hidden_state_index=self.args.capture_layer,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                image_kwargs=self._image_kwargs(),
            )
        if capture.capture_found:
            trim_capture_target_at_delimiters(capture, self.processor.tokenizer)
        if not capture.capture_found:
            appended = append_answer_turn_and_open(
                model=self.model,
                tokenizer_or_processor=self.processor,
                generation_state=capture,
                benchmark_answer_format="multiple_choice",
                option_letters=_letters(sample),
                original_question=_question_without_choices(sample.question),
                option_texts=sample.choices,
                repeat_options_in_continuation=spec.repeat_options,
                device=self.device,
            )
            continuation = continue_generation_from_state(
                model=self.model,
                tokenizer_or_processor=self.processor,
                generation_state=appended,
                max_new_tokens=self.args.answer_max_new_tokens,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
            debug = dict(appended.debug_metadata)
            debug.update({"capture_found": False, "capture_text": capture.generated_text})
            return continuation.generated_text, len(continuation.generated_ids), debug, "", 0, previous_correct_d_cpu

        foveation_target = capture.target_text
        correct_d = None
        if spec.fvt_mode != "none":
            if visual_hook.pre_merge_visual_tokens is None:
                raise RuntimeError("Pre-merge visual tokens were not captured")
            fvt_output = self.foveal_module(
                target_hidden_states=capture.target_hidden_states.to(self.device),
                pre_merge_visual_tokens=visual_hook.pre_merge_visual_tokens.to(self.device),
                metadata={"target_text": capture.target_text, "ablation": spec.name},
            )
            fvt_output = finalize_tgvf_output_with_frozen_qwen_merger(self.model, fvt_output)
            correct_d = fvt_output.foveated_visual_tokens
            d = correct_d
            if spec.fvt_mode == "zero":
                d = torch.zeros_like(correct_d)
            elif spec.fvt_mode == "random":
                d = torch.randn_like(correct_d)
            elif spec.fvt_mode == "wrong":
                d = (
                    previous_correct_d_cpu.to(device=correct_d.device, dtype=correct_d.dtype)
                    if previous_correct_d_cpu is not None and tuple(previous_correct_d_cpu.shape) == tuple(correct_d.shape)
                    else torch.randn_like(correct_d)
                )
            elif spec.fvt_mode == "vmerge":
                if merged_hook.merged_visual_tokens is None:
                    raise RuntimeError("Merged visual tokens were not captured")
                d = _matching_vmerge_tokens(merged_hook.merged_visual_tokens, correct_d)
            elif spec.fvt_mode != "correct":
                raise ValueError(f"Unsupported fvt_mode: {spec.fvt_mode}")
            appended = append_fvt_result_and_open_answer_turn(
                model=self.model,
                tokenizer_or_processor=self.processor,
                generation_state=capture,
                foveated_visual_tokens=d,
                target_text=capture.target_text,
                benchmark_answer_format="multiple_choice",
                option_letters=_letters(sample),
                original_question=_question_without_choices(sample.question),
                option_texts=sample.choices,
                repeat_options_in_continuation=spec.repeat_options,
                append_as_new_user_turn=True,
                fvt_append_mode=spec.fvt_append_mode or self.args.fvt_append_mode,
                image_grid_thw=(
                    capture.image_grid_thw
                    if fvt_output.debug_metadata.get("tgvf_version") == "v2"
                    or spec.fvt_mode == "vmerge"
                    else None
                ),
                device=self.device,
            )
            visual_token_count = int(d.shape[0])
            previous_correct_d_cpu = correct_d.detach().cpu()
        else:
            appended = append_answer_turn_and_open(
                model=self.model,
                tokenizer_or_processor=self.processor,
                generation_state=capture,
                benchmark_answer_format="multiple_choice",
                option_letters=_letters(sample),
                original_question=_question_without_choices(sample.question),
                option_texts=sample.choices,
                repeat_options_in_continuation=spec.repeat_options,
                device=self.device,
            )
            visual_token_count = 0
        continuation = continue_generation_from_state(
            model=self.model,
            tokenizer_or_processor=self.processor,
            generation_state=appended,
            max_new_tokens=self.args.answer_max_new_tokens,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        debug = dict(appended.debug_metadata)
        debug.update(
            {
                "capture_found": bool(capture.capture_found),
                "capture_text": capture.generated_text,
                "force_target_mode": self.args.force_target_mode,
                "fvt_mode": spec.fvt_mode,
                "first_generated_token_after_append": _first_token_debug(
                    self.processor.tokenizer, continuation.generated_ids
                ),
                "stopped_before_generating_answer": _stopped_before_answer(
                    self.processor.tokenizer, continuation.generated_ids
                ),
            }
        )
        return continuation.generated_text, len(continuation.generated_ids), debug, foveation_target, visual_token_count, previous_correct_d_cpu

    def _force_capture_model(self) -> tuple[Any, int]:
        start_ids = self.processor.tokenizer.encode(FOVEATE_START, add_special_tokens=False)
        end_ids = self.processor.tokenizer.encode(FOVEATE_END, add_special_tokens=False)
        if self.args.force_target_mode == "fixed":
            fixed_ids = self.processor.tokenizer.encode(str(self.args.fixed_force_target), add_special_tokens=False)
            forced_ids = [*start_ids, *fixed_ids, *end_ids]
            return ForcedFoveationWrapper(self.model, forced_ids), len(forced_ids)
        suppress_ids = (
            [self.processor.tokenizer.eos_token_id]
            if self.processor.tokenizer.eos_token_id is not None
            else []
        )
        wrapped = ForcedFoveationBracketWrapper(
            self.model,
            start_ids=start_ids,
            end_ids=end_ids,
            max_target_tokens=int(self.args.force_target_max_tokens),
            suppress_ids=suppress_ids,
        )
        return wrapped, len(start_ids) + int(self.args.force_target_max_tokens) + len(end_ids)

    def _image_kwargs(self) -> dict[str, int]:
        return image_budget_kwargs(self.args.image_budget)


def summarize_ablation_rows(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    ablation_name: str,
    benchmark: str,
    include_answer_distribution: bool = True,
    include_target_quality: bool = True,
) -> dict[str, Any]:
    scores = [float(row.get("score") or 0.0) for row in rows if row.get("error") is None]
    wall = [float(row.get("wall_time_sec") or 0.0) for row in rows]
    used_logits = Counter(str(row.get("used_logits_source")) for row in rows if row.get("used_logits_source"))
    append_modes = Counter(
        str((row.get("debug_metadata") or {}).get("append_mode"))
        for row in rows
        if (row.get("debug_metadata") or {}).get("append_mode")
    )
    position_sources = Counter(
        str((row.get("debug_metadata") or {}).get("position_ids_source"))
        for row in rows
        if (row.get("debug_metadata") or {}).get("position_ids_source")
    )
    image_3d_count = sum(bool((row.get("debug_metadata") or {}).get("image_position_ids_are_3d")) for row in rows)
    text_1d_count = sum(bool((row.get("debug_metadata") or {}).get("text_position_ids_are_1d")) for row in rows)
    visual_tower_called_for_fvt_count = sum(
        bool((row.get("debug_metadata") or {}).get("visual_tower_called_for_fvt")) for row in rows
    )
    summary = {
        "run_id": run_id,
        "ablation_name": ablation_name,
        "benchmark": benchmark,
        "num_samples": len(rows),
        "score": _mean(scores),
        "completion_rate": sum(_completed(row) for row in rows) / max(1, len(rows)),
        "parse_rate": sum(bool(row.get("parsed_answer")) for row in rows) / max(1, len(rows)),
        "immediate_im_end_rate": sum(bool(row.get("immediate_im_end")) for row in rows) / max(1, len(rows)),
        "trigger_rate": sum(bool(row.get("triggered")) for row in rows) / max(1, len(rows)),
        "avg_num_foveations": sum(1.0 if row.get("triggered") else 0.0 for row in rows) / max(1, len(rows)),
        "second_full_forward_used_count": sum(bool(row.get("second_full_forward_used")) for row in rows),
        "cache_preserved_count": sum(bool(row.get("cache_preserved")) for row in rows),
        "used_logits_source_counts": dict(sorted(used_logits.items())),
        "append_mode_counts": dict(sorted(append_modes.items())),
        "position_ids_source_counts": dict(sorted(position_sources.items())),
        "image_position_ids_are_3d_count": image_3d_count,
        "text_position_ids_are_1d_count": text_1d_count,
        "visual_tower_called_for_fvt_count": visual_tower_called_for_fvt_count,
        "avg_wall_time_sec": _mean(wall),
        "p50_wall_time_sec": statistics.median(wall) if wall else None,
        "p90_wall_time_sec": _quantile(wall, 0.9),
    }
    if include_answer_distribution:
        summary.update(answer_distribution_report(rows))
    if include_target_quality:
        summary.update(target_quality_report(rows))
    return summary


def decision_hints(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {summary["ablation_name"]: summary for summary in summaries}
    def score(name: str) -> float | None:
        value = by_name.get(name, {}).get("score")
        return float(value) if value is not None else None
    hints = {
        "direct_score": score("direct_qwen"),
        "new_user_turn_only_score": score("new_user_turn_only"),
        "new_user_turn_repeat_options_score": score("new_user_turn_repeat_options"),
        "force_correct_D_score": score("force_correct_D"),
        "force_correct_D_repeat_options_score": score("force_correct_D_repeat_options"),
        "force_zero_D_repeat_options_score": score("force_zero_D_repeat_options"),
        "force_random_D_repeat_options_score": score("force_random_D_repeat_options"),
        "force_wrong_D_repeat_options_score": score("force_wrong_D_repeat_options"),
    }
    direct = hints["direct_score"]
    a1 = hints["new_user_turn_only_score"]
    a2 = hints["new_user_turn_repeat_options_score"]
    a5 = hints["force_correct_D_score"]
    a6 = hints["force_correct_D_repeat_options_score"]
    if direct is not None and a1 is not None and a2 is not None and a1 + 0.1 < direct and a2 >= a1 + 0.05:
        hints["continuation_context_loss_likely"] = True
    else:
        hints["continuation_context_loss_likely"] = False
    hints["repeat_options_helped_force_correct_D"] = bool(a5 is not None and a6 is not None and a6 > a5)
    return hints


def _load_model_and_module(args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    import argparse as _argparse

    from eval.common import load_qwen_and_tgvf

    load_args = _argparse.Namespace(
        model_path=args.model_path,
        processor_path=args.processor_path,
        tgvf_checkpoint=args.tgvf_checkpoint,
        variant=args.tgvf_variant,
        device=args.device,
        dtype=args.dtype,
        attn_implementation="flash_attention_2",
        num_foveated_tokens=16,
        spatial_merge_size="auto",
    )
    model, processor, foveal_module, device, _info = load_qwen_and_tgvf(load_args)
    return model, processor, foveal_module, device


def _base_row(sample: BenchmarkSample, spec: AblationSpec) -> dict[str, Any]:
    options = {letter: text for letter, text in zip(_letters(sample), sample.choices, strict=False)}
    return {
        "sample_id": sample.sample_id,
        "category": sample.metadata.get("category"),
        "metadata": sample.metadata,
        "question": _question_without_choices(sample.question),
        "options": options,
        "gold_answer": sample.gold_answer,
        "ablation": spec.name,
        "fvt_mode": spec.fvt_mode,
        "fvt_append_mode": spec.fvt_append_mode,
        "repeat_options_in_continuation": bool(spec.repeat_options),
    }


def _matching_vmerge_tokens(merged_visual_tokens: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    merged = merged_visual_tokens.detach().to(device=like.device, dtype=like.dtype)
    if merged.ndim == 3:
        merged = merged.reshape(-1, merged.shape[-1])
    if merged.shape[-1] != like.shape[-1]:
        raise ValueError(f"merged visual hidden dim {merged.shape[-1]} != FVT dim {like.shape[-1]}")
    if merged.shape[0] >= like.shape[0]:
        return merged[: like.shape[0]].clone()
    repeats = (like.shape[0] + merged.shape[0] - 1) // max(1, merged.shape[0])
    return merged.repeat((repeats, 1))[: like.shape[0]].clone()


def _letters(sample: BenchmarkSample) -> list[str]:
    return [chr(ord("A") + index) for index in range(len(sample.choices))]


def _question_without_choices(question: str) -> str:
    lines = []
    for line in question.splitlines():
        stripped = line.strip()
        if len(stripped) > 3 and stripped[0] == "(" and stripped[2] == ")" and stripped[1].isalpha():
            continue
        if stripped.lower().startswith("answer only with"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _first_token_debug(tokenizer: Any, token_ids: list[int]) -> dict[str, Any] | None:
    if not token_ids:
        return None
    token_id = int(token_ids[0])
    return {
        "token_id": token_id,
        "token_text": tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False),
    }


def _stopped_before_answer(tokenizer: Any, token_ids: list[int]) -> bool:
    if not token_ids:
        return True
    eos_id = getattr(tokenizer, "eos_token_id", None)
    return eos_id is not None and int(token_ids[0]) == int(eos_id)


def _resume_model_kwargs(inputs: dict[str, Any], outputs: Any) -> dict[str, Any]:
    kwargs = {key: inputs[key] for key in ("image_grid_thw", "video_grid_thw", "mm_token_type_ids", "position_ids") if key in inputs}
    rope_deltas = getattr(outputs, "rope_deltas", None)
    if rope_deltas is not None:
        kwargs["rope_deltas"] = rope_deltas
    elif "rope_deltas" in inputs:
        kwargs["rope_deltas"] = inputs["rope_deltas"]
    return kwargs


def _move_tensors(payload: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in payload.items()}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _completed(row: dict[str, Any]) -> bool:
    raw = str(row.get("raw_output") or "").strip()
    return bool(raw) and raw != "<|im_end|>"


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row.get(key)) for row in rows) / max(1, len(rows))


def _mean(values: list[float] | list[int]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return float(ordered[index])


def _normalize_target(text: str) -> str:
    return " ".join(str(text or "").lower().strip().strip(".:-_").split())


def _bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


if __name__ == "__main__":
    main()
