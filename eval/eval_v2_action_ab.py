#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, StoppingCriteria, StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.common import load_eval_samples, resolve_device, write_json, write_jsonl
from revisit_vlm.models.qwen2vl import load_qwen2vl
from revisit_vlm.tgvf import FOVEATE_END, FOVEATE_START
from revisit_vlm.tgvf_capture import (
    build_qwen2vl_tgvf_inputs,
    capture_tgvf_single_pass_from_inputs,
    _decode,
)
from revisit_vlm.tgvf_training import ForcedFoveationBracketWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A/B test v2 TGVF action generation: model.generate vs custom capture loop.")
    parser.add_argument("--model-path", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--processor-path", default=None)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bf16", choices=("auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"))
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--max-action-tokens", type=int, default=64)
    parser.add_argument("--mode", choices=("free", "force_wrapper"), default="free")
    parser.add_argument("--prompt-mode", choices=("default", "force_foveate"), default="default")
    parser.add_argument("--forced-target-max-tokens", type=int, default=32)
    parser.add_argument("--capture-layer", type=int, default=-1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-key", choices=("image", "sample"), default="image")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    loaded = load_qwen2vl(
        args.model_path,
        torch_dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        device_map={"": device} if str(device).startswith("cuda") else None,
        trust_remote_code=False,
    )
    model = loaded.model
    processor = loaded.processor
    if args.processor_path:
        processor = AutoProcessor.from_pretrained(args.processor_path, trust_remote_code=False)
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.eval()
    samples = load_eval_samples(
        args.eval_jsonl,
        max_samples=args.max_samples,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        shard_key=args.shard_key,
    )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, sample in enumerate(samples, start=1):
        if args.progress and (index == 1 or index % args.log_every == 0 or index == len(samples)):
            print(json.dumps({"index": index, "total": len(samples)}, ensure_ascii=False), flush=True)
        inputs = build_inputs(processor, image=sample.image, question=sample.question, prompt_mode=args.prompt_mode)
        rows.append(run_standard_generate(model, processor, inputs, sample, index - 1, args, device))
        rows.append(run_custom_capture(model, processor, inputs, sample, index - 1, args, device))
    summary = summarize(rows)
    summary.update(
        wall_time_sec=time.perf_counter() - started,
        model_path=args.model_path,
        eval_jsonl=args.eval_jsonl,
        max_samples=args.max_samples,
        max_action_tokens=args.max_action_tokens,
        mode=args.mode,
        prompt_mode=args.prompt_mode,
        forced_target_max_tokens=args.forced_target_max_tokens,
    )
    write_jsonl(output_dir / "v2_action_ab.jsonl", rows)
    write_json(output_dir / "v2_action_ab.summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print("examples", flush=True)
    for row in rows[: min(len(rows), 16)]:
        print(json.dumps({
            "sample_index": row["sample_index"],
            "method": row["method"],
            "capture_found": row["capture_found"],
            "target": row["target_text"],
            "has_start": row["has_foveate_start"],
            "has_end": row["has_foveate_end"],
            "raw": row["raw_output"],
        }, ensure_ascii=False)[:1800], flush=True)


@torch.no_grad()
def run_standard_generate(
    model: Any,
    processor: Any,
    inputs: dict[str, Any],
    sample: Any,
    sample_index: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    model_inputs = move_tensors(dict(inputs), device)
    prompt_len = int(model_inputs["input_ids"].shape[-1])
    generation_kwargs: dict[str, Any] = {}
    if args.mode == "force_wrapper":
        start_ids, end_ids = foveation_marker_ids(processor)
        generation_kwargs["logits_processor"] = LogitsProcessorList(
            [
                ForcedFoveationBracketLogitsProcessor(
                    prompt_len=prompt_len,
                    start_ids=start_ids,
                    end_ids=end_ids,
                    max_target_tokens=args.forced_target_max_tokens,
                    suppress_ids=(
                        [processor.tokenizer.eos_token_id]
                        if processor.tokenizer.eos_token_id is not None
                        else []
                    ),
                )
            ]
        )
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [StopOnGeneratedSubsequence(end_ids, prompt_len)]
        )
    start = time.perf_counter()
    generated = model.generate(
        **model_inputs,
        max_new_tokens=args.max_action_tokens,
        do_sample=False,
        **generation_kwargs,
    )
    new_ids = generated[0, prompt_len:].detach().cpu().tolist()
    text = _decode(processor.tokenizer, new_ids)
    row = base_row(sample, sample_index=sample_index, method="standard_generate")
    row.update(parse_action_text(text))
    row.update(
        raw_output=text,
        generated_ids=new_ids,
        generated_token_count=len(new_ids),
        wall_time_sec=time.perf_counter() - start,
        second_full_forward_used=False,
    )
    return row


@torch.no_grad()
def run_custom_capture(
    model: Any,
    processor: Any,
    inputs: dict[str, Any],
    sample: Any,
    sample_index: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    start = time.perf_counter()
    capture_model = model
    if args.mode == "force_wrapper":
        start_ids, end_ids = foveation_marker_ids(processor)
        capture_model = ForcedFoveationBracketWrapper(
            model,
            start_ids=start_ids,
            end_ids=end_ids,
            max_target_tokens=args.forced_target_max_tokens,
            suppress_ids=(
                [processor.tokenizer.eos_token_id]
                if processor.tokenizer.eos_token_id is not None
                else []
            ),
        )
    capture = capture_tgvf_single_pass_from_inputs(
        capture_model,
        processor.tokenizer,
        inputs,
        max_new_tokens=args.max_action_tokens,
        device=device,
        hidden_state_index=args.capture_layer,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    row = base_row(sample, sample_index=sample_index, method="custom_single_pass_capture")
    row.update(parse_action_text(capture.generated_text))
    row.update(
        raw_output=capture.generated_text,
        generated_ids=capture.generated_ids,
        generated_token_count=len(capture.generated_ids),
        capture_found=bool(capture.capture_found),
        target_text=capture.target_text,
        target_token_count=len(capture.target_token_ids),
        H_q_shape=list(capture.target_hidden_states.shape),
        stop_reason=capture.stop_reason,
        past_key_values_present=capture.past_key_values is not None,
        wall_time_sec=time.perf_counter() - start,
        second_full_forward_used=False,
    )
    return row


def base_row(sample: Any, *, sample_index: int, method: str) -> dict[str, Any]:
    return {
        "sample_index": sample_index,
        "method": method,
        "uid": sample.uid,
        "image": sample.image,
        "image_id": sample.image_id,
        "question": sample.question,
        "expected_target": sample.target,
        "raw_output": "",
        "target_text": "",
        "capture_found": False,
        "has_foveate_start": False,
        "has_foveate_end": False,
        "malformed": False,
        "generated_ids": [],
        "generated_token_count": 0,
        "H_q_shape": None,
        "second_full_forward_used": False,
    }


def parse_action_text(text: str) -> dict[str, Any]:
    start = text.find(FOVEATE_START)
    end = text.find(FOVEATE_END, start + len(FOVEATE_START)) if start >= 0 else -1
    target = ""
    if start >= 0 and end >= 0:
        target = text[start + len(FOVEATE_START) : end].strip()
    return {
        "target_text": target,
        "capture_found": bool(start >= 0 and end >= 0 and target),
        "has_foveate_start": bool(start >= 0),
        "has_foveate_end": bool(end >= 0),
        "malformed": bool((start >= 0) != (end >= 0)),
        "target_length": len(target.split()),
    }


def build_inputs(processor: Any, *, image: str, question: str, prompt_mode: str) -> dict[str, Any]:
    if prompt_mode == "default":
        return build_qwen2vl_tgvf_inputs(processor, image=image, question=question)
    prompt = (
        "We are building a Target-Guided Visual Foveation system.\n"
        "Your task is to choose one specific local visual target in the image before answering.\n"
        "Do not answer the question yet.\n"
        "Output exactly one foveation request and stop.\n\n"
        f"Question: {question}\n\n"
        "Required format:\n"
        f"{FOVEATE_START}specific local visual target{FOVEATE_END}\n\n"
        "Rules:\n"
        "- Put only the target text between the foveation tags.\n"
        "- The target must be local and visually locatable.\n"
        "- Do not include the answer value inside the target.\n"
        "- Do not output boxes, coordinates, JSON, explanations, or the final answer.\n"
        "- Use a short noun phrase such as the small text below the barcode, "
        "the number inside the blue circle, or the logo in the upper-right corner."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return build_qwen2vl_tgvf_inputs(processor, image=image, question=question, messages=messages)


class ForcedFoveationBracketLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        *,
        prompt_len: int,
        start_ids: list[int],
        end_ids: list[int],
        max_target_tokens: int,
        suppress_ids: list[int] | None = None,
    ) -> None:
        self.prompt_len = int(prompt_len)
        self.start_ids = [int(item) for item in start_ids]
        self.end_ids = [int(item) for item in end_ids]
        self.max_target_tokens = int(max_target_tokens)
        self.suppress_ids = [int(item) for item in (suppress_ids or [])]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        generated_len = int(input_ids.shape[-1]) - self.prompt_len
        forced_id = self._forced_id_for_step(generated_len)
        if forced_id is not None:
            forced_scores = torch.full_like(scores, -1e4)
            forced_scores[:, forced_id] = 1e4
            return forced_scores
        if self._is_target_step(generated_len) and self.suppress_ids:
            scores = scores.clone()
            valid_ids = [idx for idx in self.suppress_ids if 0 <= idx < scores.shape[-1]]
            if valid_ids:
                scores[:, valid_ids] = -1e4
        return scores

    def _forced_id_for_step(self, step: int) -> int | None:
        if step < len(self.start_ids):
            return self.start_ids[step]
        end_start = len(self.start_ids) + self.max_target_tokens
        if end_start <= step < end_start + len(self.end_ids):
            return self.end_ids[step - end_start]
        return None

    def _is_target_step(self, step: int) -> bool:
        return len(self.start_ids) <= step < len(self.start_ids) + self.max_target_tokens


class StopOnGeneratedSubsequence(StoppingCriteria):
    def __init__(self, pattern: list[int], prompt_len: int) -> None:
        self.pattern = [int(item) for item in pattern]
        self.prompt_len = int(prompt_len)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> bool:
        generated = input_ids[0, self.prompt_len :].detach().cpu().tolist()
        if len(generated) < len(self.pattern):
            return False
        return generated[-len(self.pattern) :] == self.pattern


def foveation_marker_ids(processor: Any) -> tuple[list[int], list[int]]:
    tokenizer = processor.tokenizer
    return (
        tokenizer.encode(FOVEATE_START, add_special_tokens=False),
        tokenizer.encode(FOVEATE_END, add_special_tokens=False),
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method = {}
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        by_method[method] = {
            "n": len(method_rows),
            "capture_found_rate": mean(row.get("capture_found") for row in method_rows),
            "has_foveate_start_rate": mean(row.get("has_foveate_start") for row in method_rows),
            "has_foveate_end_rate": mean(row.get("has_foveate_end") for row in method_rows),
            "malformed_rate": mean(row.get("malformed") for row in method_rows),
            "avg_target_len": mean(row.get("target_length") for row in method_rows if row.get("target_length")),
            "avg_wall_time_sec": mean(row.get("wall_time_sec") for row in method_rows),
            "second_full_forward_used_any": any(bool(row.get("second_full_forward_used")) for row in method_rows),
        }
    pairs = group_by_sample(rows)
    exact_text_match = []
    exact_id_match = []
    for pair in pairs:
        standard = pair.get("standard_generate")
        custom = pair.get("custom_single_pass_capture")
        if not standard or not custom:
            continue
        exact_text_match.append(standard.get("raw_output") == custom.get("raw_output"))
        exact_id_match.append(standard.get("generated_ids") == custom.get("generated_ids"))
    return {
        "n": len(rows),
        "pairs": len(pairs),
        "exact_text_match_rate": mean(exact_text_match),
        "exact_id_match_rate": mean(exact_id_match),
        "by_method": by_method,
    }


def group_by_sample(rows: list[dict[str, Any]]) -> list[dict[str, dict[str, Any]]]:
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["sample_index"]), {})[str(row["method"])] = row
    return [grouped[key] for key in sorted(grouped)]


def mean(values: Any) -> float | None:
    vals = [float(value) for value in values if value is not None]
    return None if not vals else sum(vals) / len(vals)


def move_tensors(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


if __name__ == "__main__":
    main()
