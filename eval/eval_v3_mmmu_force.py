#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval.eval_v3_stage2_protocol import Stage2ProtocolEvaluator
from eval.eval_v3_vstar_force import append_answer_only, condition_d
from revisit_vlm.qwen3_vl_tgvf import (
    EVIDENCE_STATE_END,
    EVIDENCE_STATE_START,
    FOCUS_START,
    NEED_LOCAL_EVIDENCE,
    TGVF_PROTOCOL_CHOICES,
    generate_direct_qwen3,
    parse_v3_action,
    render_force_focus_prefix,
)
from revisit_vlm.tgvf_v3_stage2 import TGVFv3Stage2Sample
from tgvf_eval.adapters import BenchmarkRegistry, BenchmarkSample


DEFAULT_BENCHMARK_ROOT = "/nvmesv/dredvpn009/datasets/benchmarks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TGVF-v3 force diagnostic on external image benchmarks.")
    parser.add_argument("--benchmark", default="mmmu_pro", choices=BenchmarkRegistry.names())
    parser.add_argument("--benchmark-root", default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument(
        "--scoring-backend",
        choices=("project", "official", "auto"),
        default="project",
        help="Use project fallback scoring, require official scoring, or use official when wired.",
    )
    parser.add_argument(
        "--official-llm-mode",
        choices=("disabled", "optional", "required"),
        default="disabled",
        help="Controls external LLM calls in official scorers such as MathVista/MathVerse.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tier", choices=("light", "medium", "full"), default="light")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question-suffix", default="")
    parser.add_argument("--d-conditions", default="correct_D,random_D")
    parser.add_argument("--eval-mode", choices=("force", "free", "both"), default="force")
    parser.add_argument("--include-stage2-direct", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--post-tgvf-continuation",
        choices=("natural_continue", "answer_only", "evidence_then_answer", "think_then_answer"),
        default="natural_continue",
    )
    parser.add_argument("--max-action-tokens", type=int, default=64)
    parser.add_argument("--max-answer-tokens", type=int, default=32)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=8)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stage2-checkpoint", default="outputs/tgvf_v3_stage2_8b/stage2_8b_fast_8gpu_1200step_full/train/checkpoint_step_1200.pt")
    parser.add_argument("--eval-jsonl", default="data/tgvf_teacher/generated/runs/tgvf_v3_teacher_val_2k/final/tgvf_teacher_items.accepted.jsonl")
    parser.add_argument("--model-id", default="/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking")
    parser.add_argument("--processor-id", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--tgvf-protocol", choices=TGVF_PROTOCOL_CHOICES, default="legacy_v3_tags")
    parser.add_argument("--max-image-resolution", type=int, default=512)
    parser.add_argument("--fvt-position-mode", choices=("native_source_grid",), default="native_source_grid")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Stage2ProtocolEvaluator(_evaluator_args(args, output_dir / "_loader"))
    evaluator.load()
    adapter = BenchmarkRegistry.get(
        args.benchmark,
        benchmark_root=args.benchmark_root,
        scoring_backend=args.scoring_backend,
        official_llm_mode=args.official_llm_mode,
    )
    samples = adapter.sample(tier=args.tier, limit=args.limit, seed=args.seed)
    if args.question_suffix:
        for sample in samples:
            sample.question = f"{sample.question.rstrip()}\n{args.question_suffix.strip()}"
    if args.num_shards > 1:
        samples = [sample for index, sample in enumerate(samples) if index % args.num_shards == args.shard_index]
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, sample in enumerate(samples, start=1):
        if args.progress and (index == 1 or index % args.log_every == 0 or index == len(samples)):
            print(json.dumps({"index": index, "total": len(samples)}, ensure_ascii=False), flush=True)
        if args.include_stage2_direct:
            rows.append(run_direct(evaluator, adapter, sample, args))
        if args.eval_mode in {"force", "both"}:
            rows.extend(run_force_conditions(evaluator, adapter, sample, args))
        if args.eval_mode in {"free", "both"}:
            rows.extend(run_free_router_conditions(evaluator, adapter, sample, args))
    write_jsonl(output_dir / "benchmark_force_rows.jsonl", rows)
    summary = summarize(rows)
    summary.update(
        wall_time_sec=time.perf_counter() - started,
        benchmark_root=args.benchmark_root,
        tier=args.tier,
        limit=args.limit,
        seed=args.seed,
        question_suffix=args.question_suffix,
        d_conditions=args.d_conditions,
        post_tgvf_continuation=args.post_tgvf_continuation,
        eval_mode=args.eval_mode,
        include_stage2_direct=args.include_stage2_direct,
        strict_parser=True,
        benchmark=args.benchmark,
        scoring_backend=args.scoring_backend,
        official_llm_mode=args.official_llm_mode,
        official_tool_used=adapter.tool_info.official_tool_used,
        official_tool_path=adapter.tool_info.official_tool_path,
        scorer_name=adapter.tool_info.scorer_name,
        official_compatible=adapter.tool_info.official_compatible,
        llm_judge_required=adapter.tool_info.llm_judge_required,
    )
    write_json(output_dir / "benchmark_force_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print("examples", flush=True)
    for row in rows[: min(len(rows), 16)]:
        print(json.dumps({
            "id": row["id"],
            "method": row["method"],
            "gold": row["label"],
            "pred": row["pred_letter"],
            "score": row["score"],
            "focus": row.get("focus_target", ""),
            "raw": row.get("raw_output", ""),
            "final": row.get("final_raw_output", ""),
        }, ensure_ascii=False)[:1800], flush=True)


def _evaluator_args(args: argparse.Namespace, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        stage2_checkpoint=args.stage2_checkpoint,
        eval_jsonl=args.eval_jsonl,
        output_dir=str(output_dir),
        model_id=args.model_id,
        processor_id=args.processor_id,
        dtype=args.dtype,
        device=args.device,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        tgvf_protocol=args.tgvf_protocol,
        variant="tgvf_v2_bidirectional",
        num_foveated_tokens=None,
        lora_rank=64,
        lora_alpha=256,
        lora_target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        max_image_resolution=args.max_image_resolution,
        max_action_tokens=args.max_action_tokens,
        max_answer_tokens=args.max_answer_tokens,
        fvt_position_mode=args.fvt_position_mode,
        blocks="force_end2end",
        d_conditions=args.d_conditions,
        max_focus=1,
        max_no_focus=0,
        min_confidence=None,
        num_shards=1,
        shard_index=0,
        wrong_search_limit=1,
        progress=False,
        log_every=args.log_every,
    )


@torch.no_grad()
def run_direct(
    evaluator: Stage2ProtocolEvaluator,
    adapter: Any,
    sample: BenchmarkSample,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    direct = generate_direct_qwen3(
        evaluator.model,
        evaluator.processor,
        image=evaluator._image(benchmark_sample(sample)),
        question=sample.question,
        max_new_tokens=args.max_answer_tokens,
        device=evaluator.device,
        protocol=args.tgvf_protocol,
    )
    raw = direct["raw_output"]
    pred, score = parse_and_score(adapter, raw, sample)
    return base_row(sample, "direct_qwen3") | {
        "raw_output": raw,
        "parsed_answer": pred,
        "pred_letter": pred,
        "score": score,
        "answer_parse_success": bool(pred),
        "wall_time_sec": time.perf_counter() - started,
        "second_full_forward_used": False,
    }


@torch.no_grad()
def run_force_conditions(
    evaluator: Stage2ProtocolEvaluator,
    adapter: Any,
    sample: BenchmarkSample,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    tgvf_sample = benchmark_sample(sample)
    image = evaluator._image(tgvf_sample)
    capture = evaluator._capture_with_forced_text(
        image=image,
        question=tgvf_sample.prompt_question,
        forced_text=render_force_focus_prefix(protocol=args.tgvf_protocol),
    )
    parsed = parse_v3_action(capture.generated_text, protocol=args.tgvf_protocol)
    if not capture.capture_found:
        row = base_row(sample, "force_focus_failed")
        row.update(
            raw_output=capture.generated_text,
            focus_valid=False,
            malformed=True,
            focus_target="",
            pred_letter="",
            score=0.0,
            answer_parse_success=False,
            second_full_forward_used=False,
        )
        return [row]
    correct_d = evaluator._d_from_capture(tgvf_sample, capture, focus_source="mmmu_force")
    rows = []
    for condition in [part.strip() for part in args.d_conditions.split(",") if part.strip()]:
        row = base_row(sample, f"force_{condition}")
        row.update(
            raw_output=capture.generated_text,
            focus_target=capture.target_text or parsed.focus_target,
            focus_valid=True,
            malformed=False,
            H_q_shape=list(capture.target_hidden_states.shape),
            target_token_count=len(capture.target_token_ids),
            second_full_forward_used=False,
        )
        try:
            d = condition_d(condition, correct_d, evaluator)
            append_result = append_answer_only(evaluator, capture, d, mode=args.post_tgvf_continuation)
            from revisit_vlm.qwen3_vl_tgvf import continue_generation_qwen3

            continuation = continue_generation_qwen3(
                evaluator.model,
                evaluator.processor,
                append_result,
                max_new_tokens=args.max_answer_tokens,
                eos_token_id=evaluator.processor.tokenizer.eos_token_id,
            )
            pred, score = parse_and_score(adapter, continuation.generated_text, sample)
            row.update(
                final_raw_output=continuation.generated_text,
                parsed_answer=pred,
                pred_letter=pred,
                score=score,
                answer_parse_success=bool(pred),
                continuation_not_im_end=bool(continuation.generated_text.strip()),
                append_success=True,
                D_shape=None if d is None else list(d.shape),
                fvt_position_mode=args.fvt_position_mode,
                second_full_forward_used=False,
            )
        except Exception as exc:
            row.update(
                errors=[f"{type(exc).__name__}:{exc}"],
                append_success=False,
                pred_letter="",
                score=0.0,
                answer_parse_success=False,
            )
        rows.append(row)
    return rows


@torch.no_grad()
def run_free_router_conditions(
    evaluator: Stage2ProtocolEvaluator,
    adapter: Any,
    sample: BenchmarkSample,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    tgvf_sample = benchmark_sample(sample)
    capture = evaluator._capture_free_router(tgvf_sample)
    parsed = parse_v3_action(capture.generated_text, protocol=args.tgvf_protocol)
    if not capture.capture_found:
        pred, score = parse_and_score(adapter, capture.generated_text, sample)
        row = base_row(sample, "free_direct_or_miss")
        row.update(
            raw_output=capture.generated_text,
            final_raw_output=capture.generated_text,
            parsed_answer=pred,
            pred_letter=pred,
            score=score,
            answer_parse_success=bool(pred),
            focus_valid=False,
            malformed=bool(parsed.malformed),
            focus_target="",
            trigger_focus_decision=False,
            focus_miss=True,
            capture_stop_reason=capture.stop_reason,
            capture_errors=capture.errors,
            continuation_not_im_end=bool(capture.generated_text.strip()),
            second_full_forward_used=bool(capture.second_full_forward_used),
        )
        return [row]

    correct_d = evaluator._d_from_capture(tgvf_sample, capture, focus_source="benchmark_free")
    rows = []
    for condition in [part.strip() for part in args.d_conditions.split(",") if part.strip()]:
        row = base_row(sample, f"free_{condition}")
        row.update(
            raw_output=capture.generated_text,
            focus_target=capture.target_text or parsed.focus_target,
            focus_valid=True,
            malformed=bool(parsed.malformed),
            H_q_shape=list(capture.target_hidden_states.shape),
            target_token_count=len(capture.target_token_ids),
            trigger_focus_decision=True,
            focus_miss=False,
            capture_stop_reason=capture.stop_reason,
            capture_errors=capture.errors,
            second_full_forward_used=bool(capture.second_full_forward_used),
        )
        try:
            d = condition_d(condition, correct_d, evaluator)
            append_result = append_answer_only(evaluator, capture, d, mode=args.post_tgvf_continuation)
            from revisit_vlm.qwen3_vl_tgvf import continue_generation_qwen3

            continuation = continue_generation_qwen3(
                evaluator.model,
                evaluator.processor,
                append_result,
                max_new_tokens=args.max_answer_tokens,
                eos_token_id=evaluator.processor.tokenizer.eos_token_id,
            )
            pred, score = parse_and_score(adapter, continuation.generated_text, sample)
            row.update(
                final_raw_output=continuation.generated_text,
                parsed_answer=pred,
                pred_letter=pred,
                score=score,
                answer_parse_success=bool(pred),
                continuation_not_im_end=bool(continuation.generated_text.strip()),
                append_success=True,
                D_shape=None if d is None else list(d.shape),
                fvt_position_mode=args.fvt_position_mode,
                second_full_forward_used=bool(capture.second_full_forward_used),
            )
        except Exception as exc:
            row.update(
                errors=[f"{type(exc).__name__}:{exc}"],
                append_success=False,
                pred_letter="",
                score=0.0,
                answer_parse_success=False,
            )
        rows.append(row)
    return rows


def benchmark_sample(sample: BenchmarkSample) -> TGVFv3Stage2Sample:
    return TGVFv3Stage2Sample(
        image=sample.primary_media,
        question=sample.question,
        answer=sample.gold_answer or "",
        need_focus=True,
        evidence_state=NEED_LOCAL_EVIDENCE,
        trajectory_type="single_focus",
        target="",
        evidence_description="",
        image_id=sample.sample_id,
        source_dataset=sample.benchmark,
        source_profile=str(sample.metadata.get("subject") or "unknown"),
        answer_format="multiple_choice",
    )


def base_row(sample: BenchmarkSample, method: str) -> dict[str, Any]:
    return {
        "id": sample.sample_id,
        "method": method,
        "category": sample.metadata.get("subject"),
        "question": sample.question,
        "choices": sample.choices,
        "label": sample.gold_answer,
        "metadata": sample.metadata,
        "raw_output": "",
        "final_raw_output": "",
        "parsed_answer": "",
        "pred_letter": "",
        "score": None,
        "answer_parse_success": False,
        "focus_target": "",
        "focus_valid": None,
        "append_success": None,
        "second_full_forward_used": False,
    }


def parse_and_score(adapter: Any, text: str, sample: BenchmarkSample) -> tuple[str, float | None]:
    if adapter.tool_info.official_tool_used:
        parsed = adapter.parse_prediction(text, sample)
    elif sample.choices:
        parsed = extract_choice_strict(text, sample)
        if not parsed:
            parsed = adapter.parse_prediction(text, sample)
    else:
        parsed = extract_answer_text(text) or adapter.parse_prediction(text, sample)
    if sample.gold_answer is None or sample.gold_answer == "":
        return parsed, None
    payload = adapter.score_predictions(
        [
            {
                "sample_id": sample.sample_id,
                "question": sample.question,
                "raw_output": text,
                "parsed_answer": parsed,
                "gold_answer": sample.gold_answer,
                "choices": sample.choices,
                "metadata": sample.metadata,
            }
        ]
    )
    return parsed, payload.get("score")


def extract_answer_text(text: str) -> str:
    cleaned = str(text or "")
    matches = list(re.finditer(r"<ANSWER>\s*(.*?)(?:</ANSWER>|$)", cleaned, flags=re.IGNORECASE | re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return ""


def extract_choice_strict(text: str, sample: BenchmarkSample) -> str:
    cleaned = str(text or "")
    cleaned = re.sub(r"<\|[^>]+\|>", " ", cleaned)
    valid_letters = "".join(chr(ord("A") + index) for index in range(len(sample.choices)))
    valid = set(valid_letters)
    for pattern in (
        r"<ANSWER>\s*\(?\s*([A-J])\s*\)?",
        r"(?i)(?:final\s+answer|answer|option|choice)\s*(?:is|:|=)?\s*\(?\s*([A-J])\s*\)?",
        r"(?m)^\s*\(?([A-J])\)?\s*$",
    ):
        match = re.search(pattern, cleaned.strip())
        if match:
            letter = match.group(1).upper()
            if letter in valid:
                return letter
    normalized_output = normalize_choice_text(cleaned)
    option_hits = []
    for index, choice in enumerate(sample.choices):
        normalized_choice = normalize_choice_text(choice)
        if not normalized_choice:
            continue
        found = normalized_output.find(normalized_choice)
        if found >= 0:
            option_hits.append((found, chr(ord("A") + index)))
    if option_hits:
        option_hits.sort()
        return option_hits[0][1]
    return ""


def normalize_choice_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method = {}
    for method in sorted({row["method"] for row in rows}):
        rs = [row for row in rows if row["method"] == method]
        by_method[method] = {
            "n": len(rs),
            "accuracy": mean(row.get("score") for row in rs),
            "answer_parse_rate": mean(row.get("answer_parse_success") for row in rs),
            "focus_valid_rate": mean(row.get("focus_valid") for row in rs if row.get("focus_valid") is not None),
            "append_success_rate": mean(row.get("append_success") for row in rs if row.get("append_success") is not None),
            "continuation_not_im_end_rate": mean(row.get("continuation_not_im_end") for row in rs if row.get("continuation_not_im_end") is not None),
            "pred_counts": count(row.get("pred_letter") or "" for row in rs),
            "gold_counts": count(row.get("label") or "" for row in rs),
        }
    free_policy = [
        row
        for row in rows
        if row["method"] in {"free_direct_or_miss", "free_correct_D"}
    ]
    return {
        "n": len(rows),
        "by_method": by_method,
        "free_policy_correct_D": {
            "n": len(free_policy),
            "accuracy": mean(row.get("score") for row in free_policy),
            "answer_parse_rate": mean(row.get("answer_parse_success") for row in free_policy),
            "trigger_rate": mean(row.get("trigger_focus_decision") for row in free_policy),
        } if free_policy else None,
        "second_full_forward_used_any": any(bool(row.get("second_full_forward_used")) for row in rows),
    }


def mean(values: Any) -> float | None:
    vals = [float(value) for value in values if value is not None]
    return None if not vals else sum(vals) / len(vals)


def count(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return dict(sorted(out.items()))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


if __name__ == "__main__":
    main()
