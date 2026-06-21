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
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval.eval_qwen3_vstar_base_direct import generate_direct_no_think_qwen3
from revisit_vlm.qwen3_vl_tgvf import generate_direct_qwen3
from tgvf_eval.adapters import BenchmarkRegistry, BenchmarkSample


DEFAULT_BENCHMARK_ROOT = "/nvmesv/dredvpn009/datasets/benchmarks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Base Qwen3-VL direct evaluation on external image benchmarks.")
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
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=8)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-id", default="/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking")
    parser.add_argument("--processor-id", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-image-resolution", type=int, default=512)
    parser.add_argument("--max-answer-tokens", type=int, default=256)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.processor_id or args.model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=_dtype(args.dtype),
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    adapter = BenchmarkRegistry.get(
        args.benchmark,
        benchmark_root=args.benchmark_root,
        scoring_backend=args.scoring_backend,
        official_llm_mode=args.official_llm_mode,
    )
    samples = adapter.sample(tier=args.tier, limit=args.limit, seed=args.seed)
    if args.num_shards > 1:
        samples = [sample for index, sample in enumerate(samples) if index % args.num_shards == args.shard_index]
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, sample in enumerate(samples, start=1):
        if args.progress and (index == 1 or index % args.log_every == 0 or index == len(samples)):
            print(json.dumps({"index": index, "total": len(samples)}, ensure_ascii=False), flush=True)
        rows.append(run_one(model, processor, adapter, sample, args))
    summary = summarize(rows)
    summary.update(
        wall_time_sec=time.perf_counter() - started,
        benchmark_root=args.benchmark_root,
        tier=args.tier,
        limit=args.limit,
        seed=args.seed,
        checkpoint_loaded=False,
        lora_loaded=False,
        method="base_direct_qwen3",
        benchmark=args.benchmark,
        scoring_backend=args.scoring_backend,
        official_llm_mode=args.official_llm_mode,
        official_tool_used=adapter.tool_info.official_tool_used,
        official_tool_path=adapter.tool_info.official_tool_path,
        scorer_name=adapter.tool_info.scorer_name,
        official_compatible=adapter.tool_info.official_compatible,
        llm_judge_required=adapter.tool_info.llm_judge_required,
        enable_thinking=args.enable_thinking,
        max_answer_tokens=args.max_answer_tokens,
        strict_parser=True,
    )
    write_jsonl(output_dir / "benchmark_base_direct_rows.jsonl", rows)
    write_json(output_dir / "benchmark_base_direct_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


@torch.no_grad()
def run_one(model: Any, processor: Any, adapter: Any, sample: BenchmarkSample, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    image = {"type": "image", "image": sample.primary_media, "max_pixels": args.max_image_resolution * args.max_image_resolution}
    if args.enable_thinking:
        result = generate_direct_qwen3(
            model,
            processor,
            image=image,
            question=sample.question,
            max_new_tokens=args.max_answer_tokens,
            device=torch.device(args.device),
        )
    else:
        result = generate_direct_no_think_qwen3(
            model,
            processor,
            image=image,
            question=sample.question,
            max_new_tokens=args.max_answer_tokens,
            device=torch.device(args.device),
        )
    raw = result["raw_output"]
    pred, score = parse_and_score(adapter, raw, sample)
    return {
        "id": sample.sample_id,
        "method": "base_direct_qwen3",
        "category": sample.metadata.get("subject"),
        "question": sample.question,
        "choices": sample.choices,
        "label": sample.gold_answer,
        "raw_output": raw,
        "parsed_answer": pred,
        "pred_letter": pred,
        "score": score,
        "answer_parse_success": bool(pred),
        "wall_time_sec": time.perf_counter() - started,
        "checkpoint_loaded": False,
        "lora_loaded": False,
        "enable_thinking": args.enable_thinking,
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
    cleaned = re.sub(r"<\|[^>]+\|>", " ", str(text or ""))
    valid = set(chr(ord("A") + index) for index in range(len(sample.choices)))
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
    out = normalize_choice_text(cleaned)
    hits = []
    for index, choice in enumerate(sample.choices):
        normalized = normalize_choice_text(choice)
        if normalized:
            found = out.find(normalized)
            if found >= 0:
                hits.append((found, chr(ord("A") + index)))
    if hits:
        hits.sort()
        return hits[0][1]
    return ""


def normalize_choice_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "accuracy": mean(row.get("score") for row in rows),
        "answer_parse_rate": mean(row.get("answer_parse_success") for row in rows),
        "pred_counts": count(row.get("pred_letter") or "" for row in rows),
        "gold_counts": count(row.get("label") or "" for row in rows),
        "category_accuracy": category_accuracy(rows),
        "avg_wall_time_sec": mean(row.get("wall_time_sec") for row in rows),
    }


def category_accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, float]] = {}
    for row in rows:
        key = str(row.get("category") or "")
        groups.setdefault(key, {"n": 0, "score": 0.0})
        groups[key]["n"] += 1
        groups[key]["score"] += float(row.get("score") or 0)
    return {key: {"n": int(val["n"]), "accuracy": val["score"] / val["n"]} for key, val in sorted(groups.items())}


def mean(values: Any) -> float | None:
    vals = [float(value) for value in values if value is not None]
    return None if not vals else sum(vals) / len(vals)


def count(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return dict(sorted(out.items()))


def _dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


if __name__ == "__main__":
    main()
