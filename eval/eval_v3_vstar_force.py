#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_v3_stage2_protocol import Stage2ProtocolEvaluator
from revisit_vlm.qwen3_vl_tgvf import (
    ANSWER_START,
    EVIDENCE_START,
    EVIDENCE_STATE_END,
    EVIDENCE_STATE_START,
    FOCUS_START,
    NEED_LOCAL_EVIDENCE,
    PROTOCOL_C_TOOL_OBSERVATION,
    PROTOCOL_C_TOOL_OBSERVATION_QWEN2_NO_THINK,
    PROTOCOL_C_THINKING_SPECIAL,
    TGVF_END,
    TGVF_PROTOCOL_CHOICES,
    TGVF_START,
    THINK_START,
    Qwen3AppendResult,
    build_direct_messages,
    build_qwen3_inputs,
    continue_generation_qwen3,
    generate_direct_qwen3,
    make_smoke_d,
    parse_v3_action,
    protocol_uses_evidence_tags,
    protocol_uses_think_tags,
    protocol_uses_tool_observation,
    render_force_focus_prefix,
    render_tgvf_prefix_suffix,
    _bracketed_visual_token_ids,
    _chunk_position_ids_1d,
    _chunk_position_ids_native_source_grid,
    _compute_qwen3_position_ids_for_sequence,
    _append_source_image_grid,
    _full_mm_token_type_ids_for_append,
    _next_position_ids_after_prefill,
    _decode,
    _decoded_tail_is_repetitive,
    _encode_text,
    _extend_attention,
    _fvt_mm_token_type_ids,
)
from revisit_vlm.tgvf_v3_stage2 import TGVFv3Stage2Sample


DEFAULT_VSTAR = "/nvmesv/dredvpn009/datasets/benchmarks/vstar_bench/snapshot/test_questions.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TGVF-v3 force diagnostic on external VSTAR benchmark.")
    parser.add_argument("--vstar-jsonl", default=DEFAULT_VSTAR)
    parser.add_argument("--vstar-root", default=None)
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Optional directory containing VSTAR images named by question_id, e.g. 5.png.",
    )
    parser.add_argument(
        "--scoring-backend",
        choices=("project", "official", "auto"),
        default="project",
        help="VSTAR has no standalone official scorer wrapper; official raises, auto falls back.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--skip-direct", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--indices",
        default="",
        help="Comma-separated VSTAR row indices/question_id values to evaluate. Applied before max-samples.",
    )
    parser.add_argument("--category", default=None)
    parser.add_argument("--d-conditions", default="correct_D,no_D,random_D")
    parser.add_argument("--eval-mode", choices=("force", "free"), default="force")
    parser.add_argument(
        "--mask-original-visual-keys-after-tgvf",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a 4D incremental attention mask so post-TGVF queries cannot attend to original image visual keys.",
    )
    parser.add_argument("--max-action-tokens", type=int, default=64)
    parser.add_argument("--max-answer-tokens", type=int, default=16)
    parser.add_argument(
        "--prompt-format",
        choices=("original", "vlmevalkit"),
        default="original",
        help="Question formatting for VSTAR local diagnostics.",
    )
    parser.add_argument(
        "--post-tgvf-continuation",
        choices=("natural_continue", "answer_only", "evidence_then_answer", "think_then_answer"),
        default="natural_continue",
    )
    parser.add_argument(
        "--post-tgvf-forward-mode",
        choices=("no_kv_full_sequence", "kv_cache"),
        default="no_kv_full_sequence",
        help="Post-D generation mode. no_kv_full_sequence recomputes the full prefix each step and is the default.",
    )
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
    parser.add_argument(
        "--force-prefix-mode",
        choices=(
            "target_hint",
            "empty_think_then_focus_start",
            "think_open_only",
            "generic_need_then_focus_start",
            "specific_region_think_open",
            "inspect_think_open",
        ),
        default="target_hint",
    )
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
    evaluator = Stage2ProtocolEvaluator(_evaluator_args(args, output_dir / "_loader"))
    evaluator.load()
    items = load_vstar_items(args)
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for index, item in enumerate(items, start=1):
        if args.progress and (index == 1 or index % args.log_every == 0 or index == len(items)):
            print(json.dumps({"index": index, "total": len(items)}, ensure_ascii=False), flush=True)
        if not args.skip_direct:
            rows.append(run_direct(evaluator, item, args))
        if args.eval_mode == "free":
            rows.extend(run_free_router_conditions(evaluator, item, args))
        else:
            rows.extend(run_force_conditions(evaluator, item, args))
    write_jsonl(output_dir / "vstar_force_rows.jsonl", rows)
    summary = summarize(rows)
    summary.update(
        wall_time_sec=time.perf_counter() - start,
        vstar_jsonl=args.vstar_jsonl,
        max_samples=args.max_samples,
        category=args.category,
        scoring_backend=args.scoring_backend,
        official_tool_used=False,
        scorer_name="project_vstar_mc",
        d_conditions=args.d_conditions,
        eval_mode=args.eval_mode,
        post_tgvf_continuation=args.post_tgvf_continuation,
        post_tgvf_forward_mode=args.post_tgvf_forward_mode,
        prompt_format=args.prompt_format,
        image_dir=args.image_dir,
        skip_direct=args.skip_direct,
        mask_original_visual_keys_after_tgvf=args.mask_original_visual_keys_after_tgvf,
    )
    write_json(output_dir / "vstar_force_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print("examples", flush=True)
    for row in rows[: min(len(rows), 16)]:
        print(json.dumps({
            "id": row["id"],
            "method": row["method"],
            "label": row["label"],
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
        force_prefix_mode=args.force_prefix_mode,
        progress=False,
        log_every=args.log_every,
    )


def load_vstar_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    jsonl = Path(args.vstar_jsonl)
    root = Path(args.vstar_root) if args.vstar_root else jsonl.parent
    selected_indices = parse_index_filter(args.indices)
    rows = []
    with jsonl.open() as f:
        for row_index, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            if args.category and item.get("category") != args.category:
                continue
            question_id = item.get("question_id")
            if selected_indices is not None and row_index not in selected_indices and question_id not in selected_indices:
                continue
            image = Path(item["image"])
            if not image.is_absolute():
                image = root / image
            if args.image_dir:
                question_id_for_image = item.get("question_id")
                if question_id_for_image is None:
                    question_id_for_image = row_index
                image = Path(args.image_dir) / f"{question_id_for_image}.png"
            item = dict(item)
            item["vstar_index"] = row_index
            item["image_path"] = str(image)
            item["prompt_text"] = format_vstar_prompt(item["text"], args.prompt_format)
            rows.append(item)
            if args.max_samples is not None and len(rows) >= args.max_samples:
                break
    if args.num_shards > 1:
        rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    return rows


def parse_index_filter(raw: str) -> set[int] | None:
    raw = str(raw or "").strip()
    if not raw:
        return None
    indices: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            indices.add(int(part))
        except ValueError as exc:
            raise ValueError(f"invalid --indices value {part!r}") from exc
    return indices


@torch.no_grad()
def run_direct(evaluator: Stage2ProtocolEvaluator, item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    direct = generate_direct_qwen3(
        evaluator.model,
        evaluator.processor,
        image={"type": "image", "image": item["image_path"], "max_pixels": args.max_image_resolution * args.max_image_resolution},
        question=item.get("prompt_text") or item["text"],
        max_new_tokens=args.max_answer_tokens,
        device=evaluator.device,
        protocol=args.tgvf_protocol,
    )
    raw = direct["raw_output"]
    pred = extract_choice(raw, item)
    return base_row(item, "direct_qwen3") | {
        "raw_output": raw,
        "pred_letter": pred,
        "score": float(pred == item.get("label")),
        "answer_parse_success": bool(pred),
        "wall_time_sec": time.perf_counter() - started,
        "second_full_forward_used": False,
    }


@torch.no_grad()
def run_force_conditions(
    evaluator: Stage2ProtocolEvaluator,
    item: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    sample = benchmark_sample(item)
    image = evaluator._image(sample)
    capture = evaluator._capture_with_forced_text(
        image=image,
        question=sample.prompt_question,
        forced_text=evaluator._force_prefix_text(sample),
    )
    rows = []
    parsed = parse_v3_action(capture.generated_text, protocol=args.tgvf_protocol)
    if not capture.capture_found:
        row = base_row(item, "force_focus_failed")
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
    correct_d = evaluator._d_from_capture(sample, capture, focus_source="vstar_force")
    for condition in [part.strip() for part in args.d_conditions.split(",") if part.strip()]:
        row = base_row(item, f"force_{condition}")
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
            condition_capture = clone_capture_for_condition(capture)
            d = condition_d(condition, correct_d, evaluator)
            append_result = append_answer_only(
                evaluator,
                condition_capture,
                d,
                mode=args.post_tgvf_continuation,
                mask_original_visual_keys_after_tgvf=args.mask_original_visual_keys_after_tgvf,
            )
            continuation = continue_after_append(
                evaluator,
                condition_capture,
                append_result,
                max_new_tokens=args.max_answer_tokens,
                mask_original_visual_keys_after_tgvf=args.mask_original_visual_keys_after_tgvf,
                forward_mode=args.post_tgvf_forward_mode,
                sample=sample,
                d=d,
            )
            pred = extract_choice(continuation.generated_text, item)
            row.update(
                final_raw_output=continuation.generated_text,
                pred_letter=pred,
                score=float(pred == item.get("label")),
                answer_parse_success=bool(pred),
                continuation_not_im_end=bool(continuation.generated_text.strip()),
                append_success=True,
                D_shape=None if d is None else list(d.shape),
                fvt_position_mode=args.fvt_position_mode,
                second_full_forward_used=False,
                post_tgvf_forward_mode=args.post_tgvf_forward_mode,
            )
            row["second_full_forward_used"] = args.post_tgvf_forward_mode == "no_kv_full_sequence"
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
    item: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    sample = benchmark_sample(item)
    capture = evaluator._capture_free_router(sample)
    parsed = parse_v3_action(capture.generated_text, protocol=args.tgvf_protocol)
    if not capture.capture_found:
        pred = extract_choice(capture.generated_text, item)
        row = base_row(item, "free_direct_or_miss")
        row.update(
            raw_output=capture.generated_text,
            final_raw_output=capture.generated_text,
            pred_letter=pred,
            score=float(pred == item.get("label")),
            answer_parse_success=bool(pred),
            focus_valid=False,
            focus_target="",
            trigger_focus_decision=False,
            focus_miss=True,
            capture_stop_reason=capture.stop_reason,
            capture_errors=capture.errors,
            continuation_not_im_end=bool(capture.generated_text.strip()),
            second_full_forward_used=bool(capture.second_full_forward_used),
        )
        return [row]

    correct_d = evaluator._d_from_capture(sample, capture, focus_source="vstar_free")
    rows = []
    for condition in [part.strip() for part in args.d_conditions.split(",") if part.strip()]:
        row = base_row(item, f"free_{condition}")
        row.update(
            raw_output=capture.generated_text,
            focus_target=capture.target_text or parsed.focus_target,
            focus_valid=True,
            malformed=False,
            H_q_shape=list(capture.target_hidden_states.shape),
            target_token_count=len(capture.target_token_ids),
            trigger_focus_decision=True,
            focus_miss=False,
            capture_stop_reason=capture.stop_reason,
            capture_errors=capture.errors,
            second_full_forward_used=bool(capture.second_full_forward_used),
        )
        try:
            condition_capture = clone_capture_for_condition(capture)
            d = condition_d(condition, correct_d, evaluator)
            append_result = append_answer_only(
                evaluator,
                condition_capture,
                d,
                mode=args.post_tgvf_continuation,
                mask_original_visual_keys_after_tgvf=args.mask_original_visual_keys_after_tgvf,
            )
            continuation = continue_after_append(
                evaluator,
                condition_capture,
                append_result,
                max_new_tokens=args.max_answer_tokens,
                mask_original_visual_keys_after_tgvf=args.mask_original_visual_keys_after_tgvf,
                forward_mode=args.post_tgvf_forward_mode,
                sample=sample,
                d=d,
            )
            pred = extract_choice(continuation.generated_text, item)
            row.update(
                final_raw_output=continuation.generated_text,
                pred_letter=pred,
                score=float(pred == item.get("label")),
                answer_parse_success=bool(pred),
                continuation_not_im_end=bool(continuation.generated_text.strip()),
                append_success=True,
                D_shape=None if d is None else list(d.shape),
                fvt_position_mode=args.fvt_position_mode,
                second_full_forward_used=bool(capture.second_full_forward_used),
                post_tgvf_forward_mode=args.post_tgvf_forward_mode,
            )
            row["second_full_forward_used"] = bool(capture.second_full_forward_used) or args.post_tgvf_forward_mode == "no_kv_full_sequence"
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


def clone_capture_for_condition(capture: Any) -> Any:
    cloned = copy.copy(capture)
    cloned.past_key_values = copy.deepcopy(capture.past_key_values)
    if getattr(capture, "attention_mask", None) is not None:
        cloned.attention_mask = capture.attention_mask.clone()
    if getattr(capture, "input_ids", None) is not None:
        cloned.input_ids = capture.input_ids.clone()
    if getattr(capture, "last_logits", None) is not None:
        cloned.last_logits = capture.last_logits.clone()
    return cloned


def append_answer_only(
    evaluator: Stage2ProtocolEvaluator,
    capture: Any,
    d: torch.Tensor | None,
    mode: str = "answer_only",
    mask_original_visual_keys_after_tgvf: bool = False,
) -> Qwen3AppendResult:
    tokenizer = evaluator.processor.tokenizer
    protocol = getattr(evaluator.args, "tgvf_protocol", "legacy_v3_tags")
    protocol_c_like = protocol in {
        PROTOCOL_C_THINKING_SPECIAL,
        PROTOCOL_C_TOOL_OBSERVATION,
        PROTOCOL_C_TOOL_OBSERVATION_QWEN2_NO_THINK,
    }
    capture_text = str(getattr(capture, "generated_text", "") or "")
    include_leading_im_end = not (
        protocol_uses_tool_observation(protocol)
        and capture_text.rstrip().endswith("<|im_end|>")
    )
    tgvf_prefix, tgvf_suffix = render_tgvf_prefix_suffix(
        protocol=protocol,
        include_leading_im_end=include_leading_im_end,
    )
    if mode == "natural_continue":
        if protocol_c_like:
            suffix_text = tgvf_suffix
            no_d_text = f"{tgvf_prefix}{tgvf_suffix}"
        else:
            suffix_text = f"\n{TGVF_END}\n"
            no_d_text = f"\n{TGVF_START}\n{TGVF_END}\n"
    elif mode == "answer_only":
        if protocol_c_like:
            answer_prompt = "Use the focused visual evidence above and answer the original multiple-choice question with only the option letter.\nAnswer:"
            suffix_text = f"{tgvf_suffix}{answer_prompt}"
            no_d_text = f"{tgvf_prefix}{tgvf_suffix}{answer_prompt}"
        else:
            suffix_text = f"\n{TGVF_END}\n{ANSWER_START}"
            no_d_text = f"\n{TGVF_START}\n{TGVF_END}\n{ANSWER_START}"
    elif mode == "evidence_then_answer":
        if protocol_uses_think_tags(protocol):
            suffix_text = f"{tgvf_suffix}{THINK_START}\n"
            no_d_text = f"{tgvf_prefix}{tgvf_suffix}{THINK_START}\n"
        elif protocol_uses_evidence_tags(protocol):
            suffix_text = f"{tgvf_suffix}<|evidence_start|>"
            no_d_text = f"{tgvf_prefix}{tgvf_suffix}<|evidence_start|>"
        else:
            suffix_text = f"\n{TGVF_END}\n{EVIDENCE_START}"
            no_d_text = f"\n{TGVF_START}\n{TGVF_END}\n{EVIDENCE_START}"
    elif mode == "think_then_answer":
        if protocol == PROTOCOL_C_TOOL_OBSERVATION_QWEN2_NO_THINK:
            suffix_text = f"{tgvf_suffix}<|evidence_start|>"
            no_d_text = f"{tgvf_prefix}{tgvf_suffix}<|evidence_start|>"
        elif protocol_c_like:
            suffix_text = f"{tgvf_suffix}{THINK_START}\n"
            no_d_text = f"{tgvf_prefix}{tgvf_suffix}{THINK_START}\n"
        else:
            suffix_text = f"\n{TGVF_END}\n<think>\n"
            no_d_text = f"\n{TGVF_START}\n{TGVF_END}\n<think>\n"
    else:
        raise ValueError(f"unsupported post-TGVF continuation mode: {mode}")
    if d is None:
        token_ids = _encode_text(tokenizer, no_d_text, evaluator.device)
        attention_mask = capture.attention_mask
        if attention_mask is not None:
            attention_mask = _extend_attention(attention_mask.to(evaluator.device), int(token_ids.shape[0]))
        model_attention_mask = attention_mask
        if mask_original_visual_keys_after_tgvf:
            model_attention_mask = build_post_tgvf_original_visual_key_mask(
                attention_mask_2d=attention_mask,
                original_image_token_indices=original_image_token_indices_from_capture(capture, evaluator.device),
                query_length=int(token_ids.shape[0]),
                past_key_values=capture.past_key_values,
                dtype=evaluator.model.get_input_embeddings().weight.dtype,
            )
        input_ids = capture.input_ids
        if input_ids is not None:
            input_ids = torch.cat([input_ids.to(evaluator.device), token_ids.view(1, -1)], dim=-1)
        outputs = evaluator.model(
            input_ids=token_ids.view(1, -1).to(evaluator.device),
            past_key_values=capture.past_key_values,
            attention_mask=model_attention_mask,
            position_ids=_chunk_position_ids_1d(
                attention_mask=attention_mask,
                chunk_length=int(token_ids.shape[0]),
                device=evaluator.device,
            ),
            use_cache=True,
            return_dict=True,
        )
        model_kwargs: dict[str, Any] = {}
        next_position_ids = _next_position_ids_after_prefill(
            _chunk_position_ids_1d(
                attention_mask=attention_mask,
                chunk_length=int(token_ids.shape[0]),
                device=evaluator.device,
            )
        )
        if next_position_ids is not None:
            model_kwargs["tgvf_next_position_ids"] = next_position_ids.detach().cpu()
        return Qwen3AppendResult(
            past_key_values=outputs.past_key_values,
            attention_mask=attention_mask,
            cache_position=None,
            input_ids=input_ids,
            last_logits=outputs.logits,
            appended_token_ids=token_ids.detach().cpu(),
            appended_inputs_embeds=torch.empty(0),
            fvt_token_start=-1,
            fvt_token_end=-1,
            model_kwargs=model_kwargs,
            debug_metadata={
                "fvt_append_path": "answer_only_no_D",
                "tgvf_protocol": protocol,
                "second_full_forward_used": False,
                "tgvf_prefix_includes_leading_im_end": include_leading_im_end,
                "mask_original_visual_keys_after_tgvf": mask_original_visual_keys_after_tgvf,
            },
        )

    source_geometry = capture.source_visual_geometry
    if source_geometry is None:
        raise ValueError("missing source visual geometry")
    prefix = tgvf_prefix
    suffix = suffix_text
    token_ids = _bracketed_visual_token_ids(
        evaluator.processor,
        evaluator.utility_model,
        num_fvt_tokens=int(d.shape[0]),
        prefix=prefix,
        suffix=suffix,
        device=evaluator.device,
    )
    prefix_ids = _encode_text(tokenizer, prefix, evaluator.device)
    fvt_token_start = int(prefix_ids.shape[0]) + 1
    fvt_token_end = fvt_token_start + int(d.shape[0])
    embed = evaluator.model.get_input_embeddings()
    embeds = embed(token_ids.view(1, -1).to(evaluator.device)).detach().clone()
    embeds[0, fvt_token_start:fvt_token_end] = d.to(device=evaluator.device, dtype=embeds.dtype)
    attention_mask = capture.attention_mask
    if attention_mask is not None:
        attention_mask = _extend_attention(attention_mask.to(evaluator.device), int(token_ids.shape[0]))
    model_attention_mask = attention_mask
    if mask_original_visual_keys_after_tgvf:
        model_attention_mask = build_post_tgvf_original_visual_key_mask(
            attention_mask_2d=attention_mask,
            original_image_token_indices=original_image_token_indices_from_capture(capture, evaluator.device),
            query_length=int(token_ids.shape[0]),
            past_key_values=capture.past_key_values,
            dtype=embeds.dtype,
        )
    input_ids = capture.input_ids
    if input_ids is not None:
        input_ids = torch.cat([input_ids.to(evaluator.device), token_ids.view(1, -1).to(evaluator.device)], dim=-1)
    mm_token_type_ids = _fvt_mm_token_type_ids(
        chunk_length=int(token_ids.shape[0]),
        fvt_token_start=fvt_token_start,
        fvt_token_end=fvt_token_end,
        device=evaluator.device,
    )
    position_ids = _chunk_position_ids_native_source_grid(
        model=evaluator.utility_model,
        capture=capture,
        token_ids=token_ids,
        attention_mask=attention_mask,
        chunk_mm_token_type_ids=mm_token_type_ids,
        fvt_token_start=fvt_token_start,
        fvt_token_end=fvt_token_end,
        source_geometry=source_geometry,
        device=evaluator.device,
    )
    outputs = evaluator.model(
        inputs_embeds=embeds,
        past_key_values=capture.past_key_values,
        attention_mask=model_attention_mask,
        position_ids=position_ids,
        mm_token_type_ids=mm_token_type_ids,
        use_cache=True,
        return_dict=True,
    )
    model_kwargs: dict[str, Any] = {}
    next_position_ids = _next_position_ids_after_prefill(position_ids)
    if next_position_ids is not None:
        model_kwargs["tgvf_next_position_ids"] = next_position_ids.detach().cpu()
    return Qwen3AppendResult(
        past_key_values=outputs.past_key_values,
        attention_mask=attention_mask,
        cache_position=None,
        input_ids=input_ids,
        last_logits=outputs.logits,
        appended_token_ids=token_ids.detach().cpu(),
        appended_inputs_embeds=embeds.detach().cpu(),
        fvt_token_start=fvt_token_start,
        fvt_token_end=fvt_token_end,
        model_kwargs=model_kwargs,
        debug_metadata={
            "fvt_append_path": "answer_only_visual_D",
            "tgvf_protocol": protocol,
            "second_full_forward_used": False,
            "tgvf_prefix_includes_leading_im_end": include_leading_im_end,
            "mask_original_visual_keys_after_tgvf": mask_original_visual_keys_after_tgvf,
        },
    )


def continue_after_append(
    evaluator: Stage2ProtocolEvaluator,
    capture: Any,
    append_result: Qwen3AppendResult,
    *,
    max_new_tokens: int,
    mask_original_visual_keys_after_tgvf: bool,
    forward_mode: str = "no_kv_full_sequence",
    sample: TGVFv3Stage2Sample | None = None,
    d: torch.Tensor | None = None,
) -> Any:
    if forward_mode == "no_kv_full_sequence":
        if mask_original_visual_keys_after_tgvf:
            raise ValueError("no_kv_full_sequence does not support post-D original-key masking")
        if sample is None:
            raise ValueError("sample is required for no_kv_full_sequence post-D continuation")
        return continue_generation_qwen3_no_kv_full_sequence(
            evaluator,
            sample,
            capture,
            append_result,
            d=d,
            max_new_tokens=max_new_tokens,
            eos_token_id=evaluator.processor.tokenizer.eos_token_id,
        )
    if forward_mode != "kv_cache":
        raise ValueError(f"unsupported post-TGVF forward mode: {forward_mode}")
    if not mask_original_visual_keys_after_tgvf:
        return continue_generation_qwen3(
            evaluator.model,
            evaluator.processor,
            append_result,
            max_new_tokens=max_new_tokens,
            eos_token_id=evaluator.processor.tokenizer.eos_token_id,
        )
    return continue_generation_qwen3_mask_original_visual_keys(
        evaluator,
        capture,
        append_result,
        max_new_tokens=max_new_tokens,
        eos_token_id=evaluator.processor.tokenizer.eos_token_id,
    )


@torch.no_grad()
def continue_generation_qwen3_no_kv_full_sequence(
    evaluator: Stage2ProtocolEvaluator,
    sample: TGVFv3Stage2Sample,
    capture: Any,
    state: Qwen3AppendResult,
    *,
    d: torch.Tensor | None,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> Any:
    tokenizer = evaluator.processor.tokenizer
    prefix_input_ids = state.input_ids
    capture_input_ids = capture.input_ids
    if prefix_input_ids is None or capture_input_ids is None:
        raise ValueError("no_kv_full_sequence requires capture and append input_ids")
    device = evaluator.device
    prefix_input_ids = prefix_input_ids.to(device=device, dtype=torch.long)
    capture_input_ids = capture_input_ids.to(device=device, dtype=torch.long)

    base_attention = torch.ones_like(prefix_input_ids, dtype=torch.long, device=device)
    if state.attention_mask is not None:
        base_attention = state.attention_mask.to(device=device, dtype=torch.long)
    base_mm_token_type_ids, image_grid_thw, video_grid_thw, visual_indices, v_merge = _full_sequence_visual_context(
        evaluator=evaluator,
        sample=sample,
        capture=capture,
        state=state,
        d=d,
        capture_input_ids=capture_input_ids,
        device=device,
    )

    generated: list[int] = []
    generated_tensor = torch.empty((1, 0), dtype=torch.long, device=device)
    logits = state.last_logits
    stop_reason = "max_new_tokens"
    for _ in range(max_new_tokens):
        full_input_ids = torch.cat([prefix_input_ids, generated_tensor], dim=-1)
        attention_mask = torch.cat(
            [
                base_attention,
                torch.ones((1, generated_tensor.shape[-1]), dtype=torch.long, device=device),
            ],
            dim=-1,
        )
        mm_token_type_ids = torch.cat(
            [
                base_mm_token_type_ids,
                torch.zeros((1, generated_tensor.shape[-1]), dtype=torch.long, device=device),
            ],
            dim=-1,
        )
        inputs_embeds = evaluator.model.get_input_embeddings()(full_input_ids).detach().clone()
        if visual_indices is not None and v_merge is not None and int(visual_indices.numel()) > 0:
            inputs_embeds[0, visual_indices] = v_merge.to(device=device, dtype=inputs_embeds.dtype)
        if d is not None:
            d_start = int(capture_input_ids.shape[-1]) + int(state.fvt_token_start)
            d_end = int(capture_input_ids.shape[-1]) + int(state.fvt_token_end)
            inputs_embeds[0, d_start:d_end] = d.to(device=device, dtype=inputs_embeds.dtype)
        position_ids = _compute_qwen3_position_ids_for_sequence(
            model=evaluator.utility_model,
            input_ids=full_input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )
        if position_ids is not None:
            position_ids = position_ids.to(device=device)
        outputs = evaluator.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        token_id = int(next_token[0, 0].detach().cpu().item())
        generated.append(token_id)
        generated_tensor = torch.cat([generated_tensor, next_token.to(device=device, dtype=torch.long)], dim=-1)
        if eos_token_id is not None and token_id == eos_token_id:
            stop_reason = "eos_token"
            break
        if _decoded_tail_is_repetitive(_decode(tokenizer, generated)):
            stop_reason = "repetition"
            break

    final_input_ids = torch.cat([prefix_input_ids, generated_tensor], dim=-1)
    final_attention = torch.cat(
        [
            base_attention,
            torch.ones((1, generated_tensor.shape[-1]), dtype=torch.long, device=device),
        ],
        dim=-1,
    )
    return type("Qwen3ContinuationLike", (), {
        "generated_ids": generated,
        "generated_text": _decode(tokenizer, generated),
        "past_key_values": None,
        "attention_mask": final_attention,
        "input_ids": final_input_ids,
        "last_logits": logits,
        "stop_reason": stop_reason,
    })()


def _full_sequence_visual_context(
    *,
    evaluator: Stage2ProtocolEvaluator,
    sample: TGVFv3Stage2Sample,
    capture: Any,
    state: Qwen3AppendResult,
    d: torch.Tensor | None,
    capture_input_ids: torch.Tensor,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    source_geometry = getattr(capture, "source_visual_geometry", None)
    appended_len = int(state.input_ids.shape[-1] - capture_input_ids.shape[-1]) if state.input_ids is not None else 0
    if d is None:
        chunk_mm_token_type_ids = torch.zeros((1, appended_len), dtype=torch.long, device=device)
        image_grid_thw = _maybe_tensor(getattr(capture, "image_grid_thw", None), device)
        video_grid_thw = _maybe_tensor(getattr(capture, "video_grid_thw", None), device)
        base_mm = _full_mm_token_type_ids_for_append(
            model=evaluator.utility_model,
            capture_input_ids=capture_input_ids,
            chunk_mm_token_type_ids=chunk_mm_token_type_ids,
            device=device,
        )
    else:
        if source_geometry is None or source_geometry.image_grid_thw is None:
            raise ValueError("no_kv_full_sequence D continuation requires source image grid geometry")
        chunk_mm_token_type_ids = _fvt_mm_token_type_ids(
            chunk_length=appended_len,
            fvt_token_start=int(state.fvt_token_start),
            fvt_token_end=int(state.fvt_token_end),
            device=device,
        )
        image_grid_thw = _append_source_image_grid(
            _maybe_tensor(getattr(capture, "image_grid_thw", None), device),
            source_geometry.image_grid_thw,
            device=device,
        )
        video_grid_thw = _maybe_tensor(getattr(capture, "video_grid_thw", None), device)
        base_mm = _full_mm_token_type_ids_for_append(
            model=evaluator.utility_model,
            capture_input_ids=capture_input_ids,
            chunk_mm_token_type_ids=chunk_mm_token_type_ids,
            device=device,
        )

    visual_indices: torch.Tensor | None = None
    v_merge: torch.Tensor | None = None
    if source_geometry is not None and source_geometry.source_visual_token_indices is not None:
        _tap, _v_pre, cached_v_merge = evaluator._vision_features(sample)
        if cached_v_merge is None:
            raise ValueError("no_kv_full_sequence could not recover source V_merge features")
        visual_indices = source_geometry.source_visual_token_indices.to(device=device, dtype=torch.long)
        v_merge = cached_v_merge.to(device=device)
        if int(visual_indices.numel()) != int(v_merge.shape[0]):
            raise ValueError(
                f"source visual token count mismatch: indices={int(visual_indices.numel())}, "
                f"v_merge={int(v_merge.shape[0])}"
            )
    return base_mm.to(device=device), image_grid_thw, video_grid_thw, visual_indices, v_merge


def _maybe_tensor(value: Any, device: torch.device | str) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.long)
    return torch.as_tensor(value, dtype=torch.long, device=device)


@torch.no_grad()
def continue_generation_qwen3_mask_original_visual_keys(
    evaluator: Stage2ProtocolEvaluator,
    capture: Any,
    state: Qwen3AppendResult,
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> Any:
    tokenizer = evaluator.processor.tokenizer
    logits = state.last_logits
    past_key_values = state.past_key_values
    attention_mask = state.attention_mask
    input_ids = state.input_ids
    next_position_ids = (state.model_kwargs or {}).get("tgvf_next_position_ids")
    generated_ids: list[int] = []
    stop_reason = "max_new_tokens"
    device = logits.device
    original_indices = original_image_token_indices_from_capture(capture, device)
    for _ in range(max_new_tokens):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        token_id = int(next_token[0, 0].detach().cpu().item())
        generated_ids.append(token_id)
        if input_ids is not None:
            input_ids = torch.cat([input_ids.to(device), next_token.to(device)], dim=-1)
        if attention_mask is not None:
            attention_mask = _extend_attention(attention_mask.to(device), 1)
        if next_position_ids is not None:
            position_ids = next_position_ids.to(device=device) + (len(generated_ids) - 1)
        else:
            position_ids = _chunk_position_ids_1d(
                attention_mask=attention_mask,
                chunk_length=1,
                device=device,
            )
        model_attention_mask = build_post_tgvf_original_visual_key_mask(
            attention_mask_2d=attention_mask,
            original_image_token_indices=original_indices,
            query_length=1,
            past_key_values=past_key_values,
            dtype=evaluator.model.get_input_embeddings().weight.dtype,
        )
        cache_position = None
        if attention_mask is not None:
            cache_position = torch.arange(
                attention_mask.shape[-1] - 1,
                attention_mask.shape[-1],
                device=device,
                dtype=torch.long,
            )
        outputs = evaluator.model(
            input_ids=next_token,
            past_key_values=past_key_values,
            attention_mask=model_attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits
        if eos_token_id is not None and token_id == eos_token_id:
            stop_reason = "eos_token"
            break
    return type("Qwen3ContinuationLike", (), {
        "generated_ids": generated_ids,
        "generated_text": _decode(tokenizer, generated_ids),
        "past_key_values": past_key_values,
        "attention_mask": attention_mask,
        "input_ids": input_ids,
        "last_logits": logits,
        "stop_reason": stop_reason,
    })()


def original_image_token_indices_from_capture(capture: Any, device: torch.device | str) -> torch.Tensor:
    geometry = getattr(capture, "source_visual_geometry", None)
    if geometry is None or geometry.source_visual_token_indices is None:
        raise ValueError("capture is missing source visual token indices")
    return geometry.source_visual_token_indices.to(device=device, dtype=torch.long)


def build_post_tgvf_original_visual_key_mask(
    *,
    attention_mask_2d: torch.Tensor | None,
    original_image_token_indices: torch.Tensor,
    query_length: int,
    past_key_values: Any,
    dtype: torch.dtype,
) -> torch.Tensor:
    if attention_mask_2d is None:
        raise ValueError("attention_mask_2d is required for masked post-TGVF generation")
    if attention_mask_2d.ndim != 2 or attention_mask_2d.shape[0] != 1:
        raise ValueError("post-TGVF original visual key mask currently supports batch size 1")
    query_length = int(query_length)
    past_length = int(past_key_values.get_seq_length()) if past_key_values is not None else int(attention_mask_2d.shape[-1]) - query_length
    key_length = past_length + query_length
    if query_length <= 0 or query_length > key_length:
        raise ValueError(f"invalid query_length={query_length} for key_length={key_length}")
    device = attention_mask_2d.device
    min_value = torch.finfo(dtype).min
    mask = torch.zeros((1, 1, query_length, key_length), dtype=dtype, device=device)
    query_absolute = torch.arange(key_length - query_length, key_length, device=device)
    key_absolute = torch.arange(key_length, device=device)
    future = key_absolute.view(1, -1) > query_absolute.view(-1, 1)
    mask = mask.masked_fill(future.view(1, 1, query_length, key_length), min_value)
    key_padding = attention_mask_2d[:, None, None, :key_length] == 0
    mask = mask.masked_fill(key_padding, min_value)
    if int(original_image_token_indices.numel()) > 0:
        original_image_token_indices = original_image_token_indices[original_image_token_indices < key_length]
        mask[:, :, :, original_image_token_indices] = min_value
    return mask


def condition_d(condition: str, correct_d: torch.Tensor, evaluator: Stage2ProtocolEvaluator) -> torch.Tensor | None:
    if condition == "correct_D":
        return correct_d
    if condition == "no_D":
        return None
    if condition == "random_D":
        return make_smoke_d(
            source="random_calibrated",
            num_fvt_tokens=int(correct_d.shape[0]),
            hidden_dim=int(correct_d.shape[-1]),
            reference=correct_d.detach().cpu(),
            device=evaluator.device,
            dtype=correct_d.dtype,
        )
    raise ValueError(f"unsupported condition: {condition}")


def benchmark_sample(item: dict[str, Any]) -> TGVFv3Stage2Sample:
    item_id = item.get("question_id")
    if item_id is None:
        item_id = item.get("image")
    return TGVFv3Stage2Sample(
        image=item["image_path"],
        question=item.get("prompt_text") or item["text"],
        answer=item.get("label", ""),
        need_focus=True,
        evidence_state=NEED_LOCAL_EVIDENCE,
        trajectory_type="single_focus",
        target="",
        evidence_description="",
        image_id=str(item_id),
        source_dataset="vstar_bench",
        source_profile=str(item.get("category") or "unknown"),
        answer_format="multiple_choice",
    )


def base_row(item: dict[str, Any], method: str) -> dict[str, Any]:
    item_id = item.get("question_id")
    if item_id is None:
        item_id = item.get("image")
    return {
        "id": str(item_id),
        "vstar_index": item.get("vstar_index"),
        "method": method,
        "category": item.get("category"),
        "image": item.get("image_path"),
        "question": item.get("prompt_text") or item.get("text"),
        "raw_question": item.get("text"),
        "label": item.get("label"),
        "raw_output": "",
        "final_raw_output": "",
        "pred_letter": "",
        "score": 0.0,
        "answer_parse_success": False,
        "focus_target": "",
        "focus_valid": None,
        "append_success": None,
        "second_full_forward_used": False,
    }


CHOICE_RE = re.compile(r"(?:<ANSWER>)?\s*\(?\b([A-D])\b\)?", re.IGNORECASE)
OPTION_RE = re.compile(r"^(?:\(([A-D])\)|([A-D])\.)\s*(.+?)\s*$", re.IGNORECASE)


def extract_choice(text: str, item: dict[str, Any]) -> str:
    cleaned = _decode_text_for_choice(text)
    answer_span = re.findall(r"<ANSWER>\s*(.*?)(?:</ANSWER>|$)", str(text or ""), flags=re.IGNORECASE | re.DOTALL)
    candidates = [answer_span[-1].strip()] if answer_span else []
    if "</think>" in cleaned:
        candidates.append(cleaned.rsplit("</think>", 1)[-1].strip())
    candidates.append(cleaned)
    for candidate in candidates:
        match = re.search(
            r"(?i)(?:final\s+answer|answer|option|choice)\s*(?:is|should be|:|=)?\s*\(?\s*([A-D])\s*\)?",
            candidate,
        )
        if match:
            return match.group(1).upper()
        line_hits = re.findall(r"(?m)^\s*\(?\s*([A-D])\s*\)?(?:[\\.:]\\s*.*)?\s*$", candidate)
        if line_hits:
            return line_hits[-1].upper()
    option_hits = []
    normalized_output = normalize_choice_text(candidates[0] if candidates else cleaned)
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
    for candidate in candidates:
        matches = re.findall(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", candidate)
        if matches:
            return matches[-1].upper()
    return ""


def _decode_text_for_choice(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"<\|[^>]+\|>", " ", text)
    text = re.sub(r"</?ANSWER>", " ", text)
    return text.strip()


def option_map_from_question(question: str) -> dict[str, str]:
    options = {}
    for line in question.splitlines():
        match = OPTION_RE.match(line.strip())
        if match:
            letter = match.group(1) or match.group(2)
            options[letter.upper()] = match.group(3).strip()
    return options


def format_vstar_prompt(text: str, prompt_format: str) -> str:
    text = str(text or "").strip()
    if prompt_format == "original":
        return text
    if prompt_format != "vlmevalkit":
        raise ValueError(f"unsupported prompt format: {prompt_format}")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    question_lines: list[str] = []
    options: list[tuple[str, str]] = []
    for line in lines:
        match = OPTION_RE.match(line)
        if match:
            letter = match.group(1) or match.group(2)
            options.append((letter.upper(), match.group(3).strip()))
        elif not line.lower().startswith("answer with"):
            question_lines.append(line)
    question = " ".join(question_lines).strip()
    option_text = "\n".join(f"{letter}. {value}" for letter, value in options)
    return (
        f"Question: {question}\n"
        "Options:\n"
        f"{option_text}\n"
        "Please select the correct answer from the options above. \n"
    )


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
        }
    return {
        "n": len(rows),
        "by_method": by_method,
        "second_full_forward_used_any": any(bool(row.get("second_full_forward_used")) for row in rows),
    }


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
