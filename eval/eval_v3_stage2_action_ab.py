#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import torch

from revisit_vlm.qwen3_vl_tgvf import (
    FOCUS_START,
    NEED_LOCAL_EVIDENCE,
    SUFFICIENT_EVIDENCE,
    build_direct_messages,
    build_qwen3_inputs,
    is_generic_target,
    parse_v3_action,
    _decode,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_EVAL = ROOT / "eval" / "eval_v3_stage2_protocol.py"
spec = importlib.util.spec_from_file_location("eval_v3_stage2_protocol", PROTOCOL_EVAL)
if spec is None or spec.loader is None:
    raise RuntimeError(f"could not import {PROTOCOL_EVAL}")
protocol_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol_eval)


DEFAULT_VAL = protocol_eval.DEFAULT_VAL
DEFAULT_CKPT = protocol_eval.DEFAULT_CKPT
DEFAULT_MODEL = protocol_eval.DEFAULT_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A/B test Stage2 action generation: model.generate vs custom decode.")
    parser.add_argument("--stage2-checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--eval-jsonl", default=DEFAULT_VAL)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--processor-id", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-image-resolution", type=int, default=512)
    parser.add_argument("--max-action-tokens", type=int, default=32)
    parser.add_argument("--max-focus", type=int, default=8)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluator_args = argparse.Namespace(
        stage2_checkpoint=args.stage2_checkpoint,
        eval_jsonl=args.eval_jsonl,
        output_dir=str(out_dir / "protocol_loader"),
        model_id=args.model_id,
        processor_id=args.processor_id,
        dtype=args.dtype,
        device=args.device,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        tgvf_protocol="legacy_v3_tags",
        variant="tgvf_v2_bidirectional",
        num_foveated_tokens=None,
        lora_rank=64,
        lora_alpha=256,
        lora_target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        max_image_resolution=args.max_image_resolution,
        max_action_tokens=args.max_action_tokens,
        max_answer_tokens=64,
        fvt_position_mode="native_source_grid",
        blocks="free_router_end2end",
        d_conditions="correct_D",
        max_focus=args.max_focus,
        max_no_focus=0,
        min_confidence=None,
        num_shards=1,
        shard_index=0,
        wrong_search_limit=1,
        progress=args.progress,
        log_every=args.log_every,
    )
    evaluator = protocol_eval.Stage2ProtocolEvaluator(evaluator_args)
    evaluator.load()

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, sample in enumerate(evaluator.focus_samples, start=1):
        if args.progress and (index == 1 or index % args.log_every == 0 or index == len(evaluator.focus_samples)):
            print(json.dumps({"index": index, "total": len(evaluator.focus_samples)}, ensure_ascii=False), flush=True)
        rows.append(run_standard_generate(evaluator, sample, args))
        rows.append(run_custom_decode(evaluator, sample, args))

    write_jsonl(out_dir / "action_ab.jsonl", rows)
    summary = summarize(rows)
    summary.update(
        wall_time_sec=time.perf_counter() - started,
        prompt_mode="training_direct",
        max_focus=args.max_focus,
        max_action_tokens=args.max_action_tokens,
        model_id=args.model_id,
        stage2_checkpoint=args.stage2_checkpoint,
    )
    write_json(out_dir / "action_ab.summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print("examples", flush=True)
    for row in rows[: min(len(rows), 12)]:
        print(json.dumps({
            "sample_index": row["sample_index"],
            "method": row["method"],
            "state": row["parsed_evidence_state"],
            "focus": row["parsed_focus_target"],
            "trigger": row["trigger_focus_decision"],
            "valid": row["focus_valid"],
            "malformed": row["malformed"],
            "raw": row["raw_output"],
        }, ensure_ascii=False)[:1600], flush=True)


@torch.no_grad()
def run_standard_generate(evaluator: Any, sample: Any, args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    image = evaluator._image(sample)
    messages = build_direct_messages(image, sample.prompt_question)
    inputs = build_qwen3_inputs(evaluator.processor, messages)
    model_inputs = {
        key: value.to(evaluator.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    prompt_len = int(model_inputs["input_ids"].shape[-1])
    generated = evaluator.model.generate(
        **model_inputs,
        max_new_tokens=args.max_action_tokens,
        do_sample=False,
    )
    new_ids = generated[0, prompt_len:].detach().cpu().tolist()
    text = _decode(evaluator.processor.tokenizer, new_ids)
    parsed = parse_v3_action(text)
    row = base_row(sample, method="standard_generate", sample_index=evaluator.focus_samples.index(sample))
    update_parse_fields(row, text, parsed)
    row.update(
        generated_token_count=len(new_ids),
        wall_time_sec=time.perf_counter() - start,
        second_full_forward_used=False,
    )
    return row


@torch.no_grad()
def run_custom_decode(evaluator: Any, sample: Any, args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    capture = evaluator._capture_free_router(sample)
    parsed = parse_v3_action(capture.generated_text)
    row = base_row(sample, method="custom_single_pass_decode", sample_index=evaluator.focus_samples.index(sample))
    update_parse_fields(row, capture.generated_text, parsed)
    row.update(
        focus_valid=bool(capture.capture_found and not capture.malformed),
        H_q_shape=list(capture.target_hidden_states.shape),
        target_token_count=len(capture.target_token_ids),
        generated_token_count=len(capture.generated_ids),
        capture_stop_reason=capture.stop_reason,
        capture_errors=list(capture.errors),
        wall_time_sec=time.perf_counter() - start,
        second_full_forward_used=bool(capture.second_full_forward_used),
    )
    return row

def base_row(sample: Any, *, method: str, sample_index: int) -> dict[str, Any]:
    return {
        "sample_index": sample_index,
        "id": protocol_eval._sample_uid(sample),
        "method": method,
        "image": sample.image,
        "image_id": sample.image_id,
        "source_dataset": sample.source_dataset,
        "source_profile": sample.source_profile,
        "question": sample.prompt_question,
        "expected_target": sample.target,
        "expected_answer": sample.answer,
        "raw_output": "",
        "parsed_evidence_state": None,
        "parsed_focus_target": "",
        "parsed_answer": "",
        "focus_valid": False,
        "malformed": False,
        "trigger_focus_decision": False,
        "generic_target_flag": False,
        "target_answer_leakage_flag": False,
        "has_focus_open": False,
        "has_focus_close": False,
        "answer_parse_success": False,
        "target_length": 0,
        "second_full_forward_used": False,
        "H_q_shape": None,
    }


def update_parse_fields(row: dict[str, Any], text: str, parsed: Any) -> None:
    focus = parsed.focus_target or ""
    row.update(
        raw_output=text,
        parsed_evidence_state=parsed.evidence_state,
        parsed_focus_target=focus,
        parsed_answer=parsed.answer,
        focus_valid=bool(parsed.focus_valid),
        malformed=bool(parsed.malformed),
        trigger_focus_decision=bool(parsed.evidence_state == NEED_LOCAL_EVIDENCE or parsed.focus_valid),
        direct_sufficient_decision=bool(parsed.evidence_state == SUFFICIENT_EVIDENCE),
        has_focus_open=bool(parsed.has_focus_open or FOCUS_START in text),
        has_focus_close=bool(parsed.has_focus_close),
        answer_parse_success=bool(parsed.answer_valid),
        target_length=len(focus.split()),
        generic_target_flag=is_generic_target(focus) if focus else False,
        target_answer_leakage_flag=protocol_eval.target_leaks_answer(focus, row.get("expected_answer") or ""),
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method = {}
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        by_method[method] = {
            "n": len(method_rows),
            "trigger_rate": mean(row.get("trigger_focus_decision") for row in method_rows),
            "need_local_state_rate": mean(row.get("parsed_evidence_state") == NEED_LOCAL_EVIDENCE for row in method_rows),
            "sufficient_state_rate": mean(row.get("parsed_evidence_state") == SUFFICIENT_EVIDENCE for row in method_rows),
            "focus_valid_rate": mean(row.get("focus_valid") for row in method_rows),
            "malformed_rate": mean(row.get("malformed") for row in method_rows),
            "has_focus_open_rate": mean(row.get("has_focus_open") for row in method_rows),
            "has_focus_close_rate": mean(row.get("has_focus_close") for row in method_rows),
            "answer_parse_rate": mean(row.get("answer_parse_success") for row in method_rows),
            "generic_target_rate": mean(row.get("generic_target_flag") for row in method_rows if row.get("parsed_focus_target")),
            "avg_target_len": mean(row.get("target_length") for row in method_rows if row.get("target_length")),
            "avg_wall_time_sec": mean(row.get("wall_time_sec") for row in method_rows),
            "second_full_forward_used_any": any(bool(row.get("second_full_forward_used")) for row in method_rows),
        }
    return {"n": len(rows), "by_method": by_method}


def mean(values: Any) -> float | None:
    vals = [float(value) for value in values if value is not None]
    return None if not vals else sum(vals) / len(vals)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
