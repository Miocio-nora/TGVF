#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_v3_vstar_force import (
    DEFAULT_VSTAR,
    load_vstar_items,
    mean,
    normalize_choice_text,
    option_map_from_question,
    write_json,
    write_jsonl,
)
from revisit_vlm.qwen3_vl_tgvf import _decode, build_direct_messages, generate_direct_qwen3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Base Qwen3-VL direct evaluation on external VSTAR.")
    parser.add_argument("--vstar-jsonl", default=DEFAULT_VSTAR)
    parser.add_argument("--vstar-root", default=None)
    parser.add_argument(
        "--scoring-backend",
        choices=("project", "official", "auto"),
        default="project",
        help="VSTAR has no standalone official scorer wrapper; official raises, auto falls back.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--category", default=None)
    parser.add_argument("--max-answer-tokens", type=int, default=32)
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
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scoring_backend == "official":
        raise RuntimeError(
            "VSTAR official_code does not provide a standalone prediction-file scorer; "
            "use --scoring-backend project, or --scoring-backend auto to fall back."
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = _dtype(args.dtype)
    processor = AutoProcessor.from_pretrained(args.processor_id or args.model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    items = load_vstar_items(args)
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for index, item in enumerate(items, start=1):
        if args.progress and (index == 1 or index % args.log_every == 0 or index == len(items)):
            print(json.dumps({"index": index, "total": len(items)}, ensure_ascii=False), flush=True)
        rows.append(run_one(model, processor, item, args))
    summary = summarize(rows)
    summary.update(
        wall_time_sec=time.perf_counter() - start,
        model_id=args.model_id,
        processor_id=args.processor_id or args.model_id,
        checkpoint_loaded=False,
        lora_loaded=False,
        method="base_direct_qwen3",
        vstar_jsonl=args.vstar_jsonl,
        max_samples=args.max_samples,
        category=args.category,
        scoring_backend=args.scoring_backend,
        official_tool_used=False,
        scorer_name="project_vstar_mc",
        enable_thinking=args.enable_thinking,
    )
    write_jsonl(output_dir / "vstar_base_direct_rows.jsonl", rows)
    write_json(output_dir / "vstar_base_direct_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


@torch.no_grad()
def run_one(model: Any, processor: Any, item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    image = {"type": "image", "image": item["image_path"], "max_pixels": args.max_image_resolution * args.max_image_resolution}
    if args.enable_thinking:
        result = generate_direct_qwen3(
            model,
            processor,
            image=image,
            question=item["text"],
            max_new_tokens=args.max_answer_tokens,
            device=torch.device(args.device),
        )
    else:
        result = generate_direct_no_think_qwen3(
            model,
            processor,
            image=image,
            question=item["text"],
            max_new_tokens=args.max_answer_tokens,
            device=torch.device(args.device),
        )
    raw = result["raw_output"]
    pred = extract_choice_strict(raw, item)
    return {
        "id": str(item.get("question_id") or item.get("image")),
        "method": "base_direct_qwen3",
        "category": item.get("category"),
        "image": item.get("image_path"),
        "question": item.get("text"),
        "label": item.get("label"),
        "raw_output": raw,
        "pred_letter": pred,
        "score": float(pred == item.get("label")),
        "answer_parse_success": bool(pred),
        "wall_time_sec": time.perf_counter() - started,
        "checkpoint_loaded": False,
        "lora_loaded": False,
        "enable_thinking": args.enable_thinking,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pred_counts: dict[str, int] = {}
    gold_counts: dict[str, int] = {}
    for row in rows:
        pred_counts[str(row.get("pred_letter") or "")] = pred_counts.get(str(row.get("pred_letter") or ""), 0) + 1
        gold_counts[str(row.get("label") or "")] = gold_counts.get(str(row.get("label") or ""), 0) + 1
    return {
        "n": len(rows),
        "accuracy": mean(row.get("score") for row in rows),
        "answer_parse_rate": mean(row.get("answer_parse_success") for row in rows),
        "pred_counts": dict(sorted(pred_counts.items())),
        "gold_counts": dict(sorted(gold_counts.items())),
        "avg_wall_time_sec": mean(row.get("wall_time_sec") for row in rows),
        "enable_thinking": rows[0].get("enable_thinking") if rows else None,
    }


@torch.no_grad()
def generate_direct_no_think_qwen3(
    model: Any,
    processor: Any,
    *,
    image: Any,
    question: str,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    messages = build_direct_messages(image, question)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    text = text + "\n</think>\n\n"
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    model_inputs = _move_tensors(dict(inputs), device)
    prompt_len = int(model_inputs["input_ids"].shape[-1])
    started = time.perf_counter()
    generated = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    wall = time.perf_counter() - started
    new_ids = generated[0, prompt_len:].detach().cpu().tolist()
    return {
        "raw_output": _decode(processor.tokenizer, new_ids),
        "generated_ids": new_ids,
        "wall_time_sec": wall,
        "output_tokens": len(new_ids),
        "no_think_prefill_used": True,
    }


def _move_tensors(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: _move_tensors(value, device) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_move_tensors(value, device) for value in obj]
    return obj


def extract_choice_strict(text: str, item: dict[str, Any]) -> str:
    cleaned = str(text or "")
    cleaned = cleaned.replace("<|im_end|>", " ")
    tag_match = __import__("re").search(r"<ANSWER>\s*\(?\s*([A-D])\s*\)?", cleaned)
    if tag_match:
        return tag_match.group(1).upper()
    answer_match = __import__("re").search(
        r"(?i)(?:final\s+answer|answer|option|choice)\s*(?:is|:|=)?\s*\(?\s*([A-D])\s*\)?",
        cleaned,
    )
    if answer_match:
        return answer_match.group(1).upper()
    single_match = __import__("re").search(r"(?m)^\s*\(?([A-D])\)?\s*$", cleaned.strip())
    if single_match:
        return single_match.group(1).upper()
    option_hits = []
    normalized_output = normalize_choice_text(cleaned)
    for letter, option_text in option_map_from_question(str(item.get("text") or "")).items():
        normalized_option = normalize_choice_text(option_text)
        if not normalized_option:
            continue
        index = normalized_output.find(normalized_option)
        if index >= 0:
            option_hits.append((index, letter))
    if option_hits:
        option_hits.sort()
        return option_hits[0][1]
    return ""


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


if __name__ == "__main__":
    main()
