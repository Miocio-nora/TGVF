#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_v3_stage2_protocol import DEFAULT_VAL, Stage2ProtocolEvaluator
from revisit_vlm.qwen3_vl_tgvf import (
    PROTOCOL_E_EVIDENCE_START,
    PROTOCOL_E_EVIDENCE_END,
    THINK_START,
    THINK_END,
    Qwen3AppendResult,
    _append_source_image_grid,
    _compute_qwen3_position_ids_for_sequence,
    _decode,
    _encode_text,
    _extend_attention,
    _full_mm_token_type_ids_for_append,
    _fvt_mm_token_type_ids,
    _chunk_position_ids_1d,
    make_smoke_d,
    protocol_uses_think_tags,
    render_focus_readout_answer_text,
    write_json,
    write_jsonl,
)


DEFAULT_CKPT = (
    "outputs/tgvf_v3_protocol_c_qwen2vl2b/"
    "protocol_c_toolobs_qwen2_no_think_stage2_qwen2vl2b_v4data_clean_imend_open_answer_multifocus_focus_imend_from_stage1_4gpu_bs16_accum2_focus80_value1_1200step_maxres512_20260624_120501/"
    "checkpoint_step_1200.pt"
)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Stage2ProtocolEvaluator(_eval_args(args, output_dir))
    evaluator.load()
    samples = evaluator.focus_samples[: args.max_focus]
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        print(json.dumps({"index": index, "total": len(samples)}, ensure_ascii=False), flush=True)
        capture = evaluator._capture_teacher_forced(sample)
        correct_d = evaluator._d_from_capture(sample, capture, focus_source="probe_teacher_forced")
        tap, _v_pre, v_merge = evaluator._vision_features(sample)
        if v_merge is None:
            raise RuntimeError(f"missing v_merge for sample {index}: {tap.errors}")
        conditions = build_conditions(args, correct_d, v_merge)
        gold_text = gold_continuation_text(evaluator, sample)
        gold_ids = _encode_text(evaluator.processor.tokenizer, gold_text, evaluator.device).view(-1)
        close_marker = THINK_END if protocol_uses_think_tags(evaluator.args.tgvf_protocol) else PROTOCOL_E_EVIDENCE_END
        close_marker_ids = _encode_text(evaluator.processor.tokenizer, close_marker, evaluator.device).view(-1)
        evidence_end_token_id = int(close_marker_ids[0].detach().cpu().item()) if close_marker_ids.numel() == 1 else None
        for condition, d in conditions.items():
            if args.append_prefill_mode == "full_sequence":
                append_result = (
                    evaluator._append_text_only_no_d_full_sequence(sample, capture)
                    if d is None
                    else evaluator._append_visual_d_full_sequence(sample, capture, d.to(evaluator.device))
                )
            else:
                append_result = (
                    evaluator._append_text_only_no_d(capture)
                    if d is None
                    else evaluator._append_visual_d(capture, d.to(evaluator.device))
                )
            if args.continuation_forward_mode == "no_kv_full_sequence":
                probe = teacher_forced_probe_no_kv_full_sequence(
                    evaluator=evaluator,
                    sample=sample,
                    capture=capture,
                    state=append_result,
                    d=d.to(evaluator.device) if d is not None else None,
                    gold_ids=gold_ids,
                    evidence_end_token_id=evidence_end_token_id,
                )
            else:
                probe = teacher_forced_probe(
                    model=evaluator.model,
                    tokenizer=evaluator.processor.tokenizer,
                    state=append_result,
                    gold_ids=gold_ids,
                    evidence_end_token_id=evidence_end_token_id,
                )
            row = {
                "sample_index": index,
                "id": getattr(sample, "v4_uid", None) or sample.image_id or sample.image,
                "image": sample.image,
                "question": sample.prompt_question,
                "target": sample.target,
                "answer": sample.answer,
                "condition": condition,
                "gold_text": gold_text,
                "gold_first_token": _decode(evaluator.processor.tokenizer, [int(gold_ids[0].detach().cpu().item())]) if gold_ids.numel() else "",
                "gold_token_count": int(gold_ids.numel()),
                **probe,
                **d_stats(condition, d, correct_d, v_merge),
            }
            rows.append(row)
    summary = summarize(rows)
    write_jsonl(output_dir / "rows.jsonl", rows)
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def _eval_args(args: argparse.Namespace, output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        stage2_checkpoint=args.stage2_checkpoint,
        eval_jsonl=args.eval_jsonl,
        output_dir=str(output_dir / "evaluator"),
        model_id=args.model_id,
        processor_id=None,
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
        max_action_tokens=96,
        max_answer_tokens=128,
        fvt_position_mode=args.fvt_position_mode,
        append_prefill_mode=args.append_prefill_mode,
        blocks="",
        d_conditions="correct_D,no_D,zero_D,random_D",
        max_focus=args.max_focus,
        max_no_focus=0,
        min_confidence=None,
        num_shards=1,
        shard_index=0,
        wrong_search_limit=256,
        force_prefix_mode="target_hint",
        progress=False,
        log_every=25,
    )


def build_conditions(args: argparse.Namespace, correct_d: torch.Tensor, v_merge: torch.Tensor) -> dict[str, torch.Tensor | None]:
    correct_norm = correct_d.detach().float().norm(dim=-1).mean().clamp_min(1e-6)
    vmerge_norm = v_merge.detach().float().norm(dim=-1).mean().to(correct_d.device).clamp_min(1e-6)
    conditions: dict[str, torch.Tensor | None] = {
        "correct_D": correct_d,
        "correct_D_scaled_to_vmerge_norm": (correct_d.float() * (vmerge_norm / correct_norm.to(correct_d.device))).to(dtype=correct_d.dtype),
        "no_D": None,
        "zero_D": torch.zeros_like(correct_d),
        "random_calibrated_D": make_smoke_d(
            source="random_calibrated",
            num_fvt_tokens=int(correct_d.shape[0]),
            hidden_dim=int(correct_d.shape[-1]),
            reference=v_merge.detach().cpu(),
            device=correct_d.device,
            dtype=correct_d.dtype,
        ),
    }
    if args.include_random_d_from_correct:
        conditions["random_like_correct_D"] = make_smoke_d(
            source="random_calibrated",
            num_fvt_tokens=int(correct_d.shape[0]),
            hidden_dim=int(correct_d.shape[-1]),
            reference=correct_d.detach().cpu(),
            device=correct_d.device,
            dtype=correct_d.dtype,
        )
    return conditions


def gold_continuation_text(evaluator: Stage2ProtocolEvaluator, sample: Any) -> str:
    readout_text = (sample.post_focus_think or sample.evidence_description).strip()
    full = render_focus_readout_answer_text(
        evidence_description=sample.evidence_description,
        answer=sample.answer,
        protocol=evaluator.args.tgvf_protocol,
        readout_think=readout_text,
        append_im_end=False,
    )
    if full.startswith(PROTOCOL_E_EVIDENCE_START):
        return full[len(PROTOCOL_E_EVIDENCE_START) :]
    think_prefix = f"{THINK_START}\n"
    if full.startswith(think_prefix):
        return full[len(think_prefix) :]
    return full


@torch.no_grad()
def teacher_forced_probe(
    *,
    model: Any,
    tokenizer: Any,
    state: Qwen3AppendResult,
    gold_ids: torch.Tensor,
    evidence_end_token_id: int | None,
) -> dict[str, Any]:
    if gold_ids.numel() == 0:
        raise ValueError("gold_ids is empty")
    logits = state.last_logits
    past_key_values = state.past_key_values
    attention_mask = state.attention_mask
    input_ids = state.input_ids
    next_position_ids = (state.model_kwargs or {}).get("tgvf_next_position_ids")
    if logits is None:
        raise ValueError("state.last_logits is required")
    device = logits.device
    losses: list[float] = []
    ranks: list[int] = []
    probs: list[float] = []
    pred_ids: list[int] = []
    evidence_end_nll = None
    evidence_end_rank = None
    for index, token in enumerate(gold_ids.to(device).view(-1)):
        step_logits = logits[:, -1, :]
        token_id = int(token.detach().cpu().item())
        loss = F.cross_entropy(step_logits.float(), token.view(1), reduction="none")[0]
        prob = torch.softmax(step_logits.float(), dim=-1)[0, token_id]
        rank = int((step_logits[0] > step_logits[0, token_id]).sum().detach().cpu().item() + 1)
        pred_id = int(torch.argmax(step_logits[0]).detach().cpu().item())
        losses.append(float(loss.detach().cpu().item()))
        probs.append(float(prob.detach().cpu().item()))
        ranks.append(rank)
        pred_ids.append(pred_id)
        if evidence_end_token_id is not None and token_id == evidence_end_token_id and evidence_end_nll is None:
            evidence_end_nll = losses[-1]
            evidence_end_rank = rank
        next_token = token.view(1, 1)
        if input_ids is not None:
            input_ids = torch.cat([input_ids.to(device), next_token], dim=-1)
        if attention_mask is not None:
            attention_mask = _extend_attention(attention_mask.to(device), 1)
        if next_position_ids is not None:
            position_ids = next_position_ids.to(device=device) + index
        else:
            position_ids = _chunk_position_ids_1d(
                attention_mask=attention_mask,
                chunk_length=1,
                device=device,
            )
        cache_position = None
        if attention_mask is not None:
            cache_position = torch.arange(
                attention_mask.shape[-1] - 1,
                attention_mask.shape[-1],
                device=device,
                dtype=torch.long,
            )
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits
    return {
        "mean_nll": sum(losses) / len(losses),
        "sum_nll": sum(losses),
        "ppl": math.exp(min(50.0, sum(losses) / len(losses))),
        "first_token_nll": losses[0],
        "first_token_prob": probs[0],
        "first_token_rank": ranks[0],
        "first_pred_token": _decode(tokenizer, [pred_ids[0]]),
        "evidence_end_nll": evidence_end_nll,
        "evidence_end_rank": evidence_end_rank,
        "mean_first_8_nll": sum(losses[:8]) / max(1, min(8, len(losses))),
        "top_pred_prefix": _decode(tokenizer, pred_ids[: min(8, len(pred_ids))]),
    }


@torch.no_grad()
def teacher_forced_probe_no_kv_full_sequence(
    *,
    evaluator: Stage2ProtocolEvaluator,
    sample: Any,
    capture: Any,
    state: Qwen3AppendResult,
    d: torch.Tensor | None,
    gold_ids: torch.Tensor,
    evidence_end_token_id: int | None,
) -> dict[str, Any]:
    if gold_ids.numel() == 0:
        raise ValueError("gold_ids is empty")
    if state.input_ids is None:
        raise ValueError("state.input_ids is required for no-KV full-sequence probe")
    if capture.input_ids is None:
        raise ValueError("capture.input_ids is required for no-KV full-sequence probe")
    source_geometry = capture.source_visual_geometry
    if source_geometry is None or source_geometry.source_visual_token_indices is None:
        raise ValueError("capture source visual geometry is required")

    tokenizer = evaluator.processor.tokenizer
    device = evaluator.device
    prefix_ids = state.input_ids.to(device)
    generated_prefix = torch.empty((1, 0), dtype=torch.long, device=device)
    embed = evaluator.model.get_input_embeddings()
    _tap, _v_pre, v_merge = evaluator._vision_features(sample)
    if v_merge is None:
        raise RuntimeError("source merged visual features are unavailable")
    original_positions = source_geometry.source_visual_token_indices.to(device=device, dtype=torch.long)
    original_visual = v_merge.to(device=device)

    capture_input_ids = capture.input_ids.to(device)
    appended_token_count = int(prefix_ids.shape[-1]) - int(capture_input_ids.shape[-1])
    if appended_token_count < 0:
        raise ValueError("state input ids are shorter than capture input ids")
    if d is not None:
        chunk_mm = _fvt_mm_token_type_ids(
            chunk_length=appended_token_count,
            fvt_token_start=int(state.fvt_token_start),
            fvt_token_end=int(state.fvt_token_end),
            device=device,
        )
        if not isinstance(source_geometry.image_grid_thw, torch.Tensor):
            raise ValueError("source image_grid_thw is required for no-KV D probe")
        image_grid_thw = _append_source_image_grid(
            capture.image_grid_thw,
            source_geometry.image_grid_thw,
            device=device,
        )
        full_fvt_start = int(capture_input_ids.shape[-1]) + int(state.fvt_token_start)
        full_fvt_end = int(capture_input_ids.shape[-1]) + int(state.fvt_token_end)
    else:
        chunk_mm = torch.zeros((1, appended_token_count), dtype=torch.long, device=device)
        image_grid_thw = capture.image_grid_thw
        full_fvt_start = full_fvt_end = -1
    base_mm = _full_mm_token_type_ids_for_append(
        model=evaluator.utility_model,
        capture_input_ids=capture_input_ids,
        chunk_mm_token_type_ids=chunk_mm,
        device=device,
    )

    losses: list[float] = []
    ranks: list[int] = []
    probs: list[float] = []
    pred_ids: list[int] = []
    evidence_end_nll = None
    evidence_end_rank = None
    for token in gold_ids.to(device).view(-1):
        full_ids = torch.cat([prefix_ids, generated_prefix], dim=-1)
        attention_mask = torch.ones_like(full_ids)
        if generated_prefix.numel():
            gen_mm = torch.zeros((1, int(generated_prefix.shape[-1])), dtype=torch.long, device=device)
            mm_token_type_ids = torch.cat([base_mm, gen_mm], dim=-1)
        else:
            mm_token_type_ids = base_mm
        position_ids = _compute_qwen3_position_ids_for_sequence(
            model=evaluator.utility_model,
            input_ids=full_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            video_grid_thw=capture.video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )
        if position_ids is None:
            raise RuntimeError("position id computation failed for no-KV full-sequence probe")
        embeds = embed(full_ids).detach().clone()
        if int(original_positions.numel()) != int(original_visual.shape[0]):
            raise ValueError("source visual token count mismatch")
        embeds[0, original_positions, :] = original_visual.to(dtype=embeds.dtype)
        if d is not None:
            embeds[0, full_fvt_start:full_fvt_end, :] = d.to(device=device, dtype=embeds.dtype)
        outputs = evaluator.model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
            return_dict=True,
        )
        step_logits = outputs.logits[:, -1, :]
        token_id = int(token.detach().cpu().item())
        loss = F.cross_entropy(step_logits.float(), token.view(1), reduction="none")[0]
        prob = torch.softmax(step_logits.float(), dim=-1)[0, token_id]
        rank = int((step_logits[0] > step_logits[0, token_id]).sum().detach().cpu().item() + 1)
        pred_id = int(torch.argmax(step_logits[0]).detach().cpu().item())
        losses.append(float(loss.detach().cpu().item()))
        probs.append(float(prob.detach().cpu().item()))
        ranks.append(rank)
        pred_ids.append(pred_id)
        if evidence_end_token_id is not None and token_id == evidence_end_token_id and evidence_end_nll is None:
            evidence_end_nll = losses[-1]
            evidence_end_rank = rank
        generated_prefix = torch.cat([generated_prefix, token.view(1, 1)], dim=-1)

    return {
        "mean_nll": sum(losses) / len(losses),
        "sum_nll": sum(losses),
        "ppl": math.exp(min(50.0, sum(losses) / len(losses))),
        "first_token_nll": losses[0],
        "first_token_prob": probs[0],
        "first_token_rank": ranks[0],
        "first_pred_token": _decode(tokenizer, [pred_ids[0]]),
        "evidence_end_nll": evidence_end_nll,
        "evidence_end_rank": evidence_end_rank,
        "mean_first_8_nll": sum(losses[:8]) / max(1, min(8, len(losses))),
        "top_pred_prefix": _decode(tokenizer, pred_ids[: min(8, len(pred_ids))]),
        "continuation_forward_mode": "no_kv_full_sequence",
    }


def d_stats(
    condition: str,
    d: torch.Tensor | None,
    correct_d: torch.Tensor,
    v_merge: torch.Tensor,
) -> dict[str, Any]:
    if d is None:
        return {
            "d_norm_mean": None,
            "d_norm_std": None,
            "d_to_vmerge_cos_mean": None,
            "d_to_correct_cos_mean": None,
        }
    d_f = d.detach().float().cpu()
    v_f = v_merge.detach().float().cpu()
    c_f = correct_d.detach().float().cpu()
    return {
        "d_norm_mean": float(d_f.norm(dim=-1).mean().item()),
        "d_norm_std": float(d_f.norm(dim=-1).std().item()),
        "vmerge_norm_mean": float(v_f.norm(dim=-1).mean().item()),
        "vmerge_norm_std": float(v_f.norm(dim=-1).std().item()),
        "correct_d_norm_mean": float(c_f.norm(dim=-1).mean().item()),
        "d_to_vmerge_cos_mean": cosine_mean(d_f, v_f),
        "d_to_correct_cos_mean": cosine_mean(d_f, c_f) if condition != "correct_D" else 1.0,
    }


def cosine_mean(a: torch.Tensor, b: torch.Tensor) -> float | None:
    if a.shape != b.shape:
        return None
    return float(F.cosine_similarity(a, b, dim=-1).mean().item())


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cond: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cond[str(row["condition"])].append(row)
    summary: dict[str, Any] = {"n_rows": len(rows), "by_condition": {}}
    for cond, items in sorted(by_cond.items()):
        summary["by_condition"][cond] = {
            "n": len(items),
            "mean_nll": mean(items, "mean_nll"),
            "mean_first_8_nll": mean(items, "mean_first_8_nll"),
            "mean_first_token_nll": mean(items, "first_token_nll"),
            "mean_first_token_prob": mean(items, "first_token_prob"),
            "median_first_token_rank": median([row["first_token_rank"] for row in items]),
            "mean_evidence_end_nll": mean(items, "evidence_end_nll"),
            "median_evidence_end_rank": median([row["evidence_end_rank"] for row in items if row["evidence_end_rank"] is not None]),
            "mean_d_norm": mean(items, "d_norm_mean"),
            "mean_vmerge_norm": mean(items, "vmerge_norm_mean"),
            "mean_d_to_vmerge_cos": mean(items, "d_to_vmerge_cos_mean"),
            "mean_d_to_correct_cos": mean(items, "d_to_correct_cos_mean"),
        }
    return summary


def mean(items: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(item[key]) for item in items if item.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def median(vals: list[Any]) -> float | None:
    xs = sorted(float(v) for v in vals if v is not None)
    if not xs:
        return None
    mid = len(xs) // 2
    if len(xs) % 2:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Qwen2 post-D gold continuation likelihood.")
    parser.add_argument("--stage2-checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--eval-jsonl", default=DEFAULT_VAL)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--tgvf-protocol", default="protocol_c_tool_observation_qwen2_no_think")
    parser.add_argument("--max-image-resolution", type=int, default=512)
    parser.add_argument("--max-focus", type=int, default=16)
    parser.add_argument("--append-prefill-mode", choices=("kv_append", "full_sequence"), default="kv_append")
    parser.add_argument("--continuation-forward-mode", choices=("kv_cache", "no_kv_full_sequence"), default="kv_cache")
    parser.add_argument("--fvt-position-mode", choices=("native_source_grid", "inherit_source_visual_positions"), default="native_source_grid")
    parser.add_argument("--include-random-d-from-correct", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
