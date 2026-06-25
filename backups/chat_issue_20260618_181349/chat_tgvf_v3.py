#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval.eval_v3_mmmu_force import append_answer_only, condition_d  # noqa: E402
from eval.eval_v3_stage2_protocol import Stage2ProtocolEvaluator  # noqa: E402
from revisit_vlm.qwen3_vl_tgvf import (  # noqa: E402
    NEED_LOCAL_EVIDENCE,
    Qwen3FocusCapture,
    generate_direct_qwen3,
    parse_v3_action,
    render_force_focus_prefix,
    continue_generation_qwen3,
    build_qwen3_inputs,
    build_direct_messages,
    extract_qwen3_source_visual_geometry,
    protocol_focus_tokens,
    _chunk_position_ids_1d,
    _decode,
    _encode_text,
    _extend_attention,
)
from revisit_vlm.tgvf_v3_stage2 import TGVFv3Stage2Sample  # noqa: E402

DEFAULT_TOOLOBS_RUN = (
    "outputs/tgvf_v3_protocol_c/"
    "protocol_c_toolobs_stage2_v4data_clean_rowonly_gpu0_3_multifocus_focus_imend_from_clean_rowonly_focus_imend_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step_20260617"
)
DEFAULT_TOOLOBS_CKPT = f"{DEFAULT_TOOLOBS_RUN}/checkpoint_step_1200.pt"
DEFAULT_TOOLOBS_PROCESSOR = f"{DEFAULT_TOOLOBS_RUN}/processor_step_1200"
DEFAULT_TOOLOBS_PRE_IMEND_CKPT = "outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_multifocus_from_toolobs_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step/checkpoint_step_1200.pt"
DEFAULT_TOOLOBS_PRE_IMEND_PROCESSOR = "outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_multifocus_from_toolobs_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step/processor_step_1200"
DEFAULT_OLD_C_CKPT = "outputs/tgvf_v3_protocol_c/tgvf_v3_protocol_c_stage2_8b_4gpu_bs16_accum2_focus80_stage1c_tokenrows_1200step_rerun_after_reboot/checkpoint_step_1200.pt"
DEFAULT_OLD_C_PROCESSOR = "outputs/tgvf_v3_protocol_c/tgvf_v3_protocol_c_stage2_8b_4gpu_bs16_accum2_focus80_stage1c_tokenrows_1200step_rerun_after_reboot/processor_step_1200"
DEFAULT_CKPT = DEFAULT_TOOLOBS_CKPT
DEFAULT_MODEL = "/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking"
DEFAULT_EVAL_JSONL = "data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl"


@dataclass
class ChatKVState:
    image: str
    image_obj: Any
    past_key_values: Any
    attention_mask: torch.Tensor | None
    input_ids: torch.Tensor | None
    last_logits: torch.Tensor | None
    source_visual_geometry: Any
    image_grid_thw: torch.Tensor | None = None
    video_grid_thw: torch.Tensor | None = None
    ended_with_eos: bool = False
    turns: int = 0

CHAT_PROFILES: dict[str, dict[str, Any]] = {
    "toolobs": {
        "label": "Current Protocol C tool-observation, clean row-only/im_end V4 data, Stage2 1200",
        "protocol": "protocol_c_tool_observation",
        "checkpoint": DEFAULT_TOOLOBS_CKPT,
        "processor": DEFAULT_TOOLOBS_PROCESSOR,
        "post_tgvf_continuation": "evidence_then_answer",
        "force_prefix_mode": "trained_need_then_focus_start",
    },
    "current": {"alias_for": "toolobs"},
    "c": {"alias_for": "toolobs"},
    "protocol_c_toolobs": {"alias_for": "toolobs"},
    "toolobs_pre_imend": {
        "label": "Older Protocol C tool-observation before focus-action im_end refresh",
        "protocol": "protocol_c_tool_observation",
        "checkpoint": DEFAULT_TOOLOBS_PRE_IMEND_CKPT,
        "processor": DEFAULT_TOOLOBS_PRE_IMEND_PROCESSOR,
        "post_tgvf_continuation": "evidence_then_answer",
        "force_prefix_mode": "trained_need_then_focus_start",
    },
    "toolobs_old": {"alias_for": "toolobs_pre_imend"},
    "c_old": {
        "label": "Old Protocol C, Stage1-C token rows, Stage2 1200",
        "protocol": "protocol_c_thinking_special",
        "checkpoint": DEFAULT_OLD_C_CKPT,
        "processor": DEFAULT_OLD_C_PROCESSOR,
        "post_tgvf_continuation": "evidence_then_answer",
        "force_prefix_mode": "trained_need_then_focus_start",
    },
    "protocol_c": {"alias_for": "c_old"},
    "a": {
        "label": "Protocol A / legacy v3 tags, Stage2 1200",
        "protocol": "legacy_v3_tags",
        "checkpoint": "outputs/tgvf_v3_stage2_8b/stage2_8b_fast_8gpu_1200step_full/train/checkpoint_step_1200.pt",
        "post_tgvf_continuation": "evidence_then_answer",
        "force_prefix_mode": "legacy_focus_start",
    },
    "legacy": {"alias_for": "a"},
    "protocol_a": {"alias_for": "a"},
    "d": {
        "label": "Protocol D, Qwen tool-native experimental, Stage2 1200",
        "protocol": "protocol_d_qwen_tool",
        "checkpoint": "outputs/tgvf_v3_protocol_d/tgvf_v3_protocol_d_stage2_8b_4gpu_bs16_accum2_focus80_1200step_tool_native/checkpoint_step_1200.pt",
        "post_tgvf_continuation": "think_then_answer",
        "force_prefix_mode": "tool_call_target_prefix",
    },
    "protocol_d": {"alias_for": "d"},
    "e": {
        "label": "Protocol E, action/evidence special tokens experimental, Stage2 1200",
        "protocol": "protocol_e_action_evidence_special",
        "checkpoint": "outputs/tgvf_v3_protocol_e/tgvf_v3_protocol_e_stage2_8b_4gpu_bs16_accum2_focus80_stage1a_sdpa_1200step_rerun_after_reboot/checkpoint_step_1200.pt",
        "processor": "outputs/tgvf_v3_protocol_e/tgvf_v3_protocol_e_stage2_8b_4gpu_bs16_accum2_focus80_stage1a_sdpa_1200step_rerun_after_reboot/processor_step_1200",
        "post_tgvf_continuation": "evidence_then_answer",
        "force_prefix_mode": "empty_think_then_focus_start",
    },
    "protocol_e": {"alias_for": "e"},
}


def resolve_chat_profile(name: str) -> tuple[str, dict[str, Any]]:
    seen: set[str] = set()
    current = name
    while True:
        if current in seen:
            raise ValueError(f"cyclic chat profile alias: {name}")
        seen.add(current)
        profile = CHAT_PROFILES.get(current)
        if profile is None:
            raise ValueError(f"unknown chat profile: {name}")
        alias_for = profile.get("alias_for")
        if not alias_for:
            return current, profile
        current = str(alias_for)


def apply_chat_profile_defaults(args: argparse.Namespace) -> None:
    canonical, profile = resolve_chat_profile(args.chat_profile)
    args.chat_profile = canonical
    checkpoint_overridden = args.stage2_checkpoint is not None
    if args.stage2_checkpoint is None:
        args.stage2_checkpoint = profile["checkpoint"]
    if args.processor_id is None and profile.get("processor") and not checkpoint_overridden:
        args.processor_id = profile["processor"]
    if args.tgvf_protocol is None:
        args.tgvf_protocol = "legacy_v3_tags" if checkpoint_overridden else profile["protocol"]
    if args.post_tgvf_continuation is None:
        args.post_tgvf_continuation = profile["post_tgvf_continuation"]
    args.force_prefix_mode = profile.get("force_prefix_mode", "empty_think_then_focus_start")


def print_profile_list() -> None:
    printed: set[str] = set()
    for name in CHAT_PROFILES:
        canonical, profile = resolve_chat_profile(name)
        if canonical in printed:
            continue
        printed.add(canonical)
        print(f"{canonical}: {profile['label']}")
        print(f"  protocol: {profile['protocol']}")
        print(f"  checkpoint: {profile['checkpoint']}")
        if profile.get("processor"):
            print(f"  processor: {profile['processor']}")
        print(f"  continuation: {profile['post_tgvf_continuation']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive single-image chat for TGVF-v3.")
    parser.add_argument("--chat-profile", "--profile", default="toolobs", choices=tuple(CHAT_PROFILES), help="Auto-select protocol/checkpoint defaults.")
    parser.add_argument("--list-profiles", action="store_true", help="Print available chat profiles and exit.")
    parser.add_argument("--image", default=None, help="Initial image path. Can be changed with /image PATH in chat.")
    parser.add_argument("--mode", choices=("force", "free", "direct"), default="force")
    parser.add_argument("--stage2-checkpoint", default=None, help="Override profile checkpoint.")
    parser.add_argument("--eval-jsonl", default=DEFAULT_EVAL_JSONL, help="Small v3 val jsonl used only to initialize evaluator shapes.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--processor-id", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--tgvf-protocol", default=None, help="Override profile protocol.")
    parser.add_argument("--max-image-resolution", type=int, default=512)
    parser.add_argument("--max-action-tokens", type=int, default=128)
    parser.add_argument("--max-answer-tokens", type=int, default=256)
    parser.add_argument("--max-foveations", type=int, default=2, help="Maximum TGVF focus/append cycles per assistant answer.")
    parser.add_argument("--post-tgvf-continuation", choices=("answer_only", "evidence_then_answer", "think_then_answer"), default=None, help="Override profile continuation.")
    parser.add_argument("--toolobs-action-stop", choices=("im_end", "focus_end"), default="im_end", help="For protocol_c_tool_observation, stop focus action at <|im_end|> or at <|focus_end|> for old checkpoints.")
    parser.add_argument("--show-raw", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--show-focus", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--text-only-action-stop", action=argparse.BooleanOptionalAction, default=True, help="Stop text-only chat when a focus span is generated instead of letting it repeat.")
    parser.add_argument("--stateful-kv", action=argparse.BooleanOptionalAction, default=False, help="Preserve KV cache across chat turns for the same image.")
    parser.add_argument("--output-jsonl", default=None, help="Optional transcript JSONL path.")
    args = parser.parse_args()
    if args.list_profiles:
        print_profile_list()
        raise SystemExit(0)
    apply_chat_profile_defaults(args)
    os.environ["TGVF_TOOLOBS_ACTION_STOP"] = args.toolobs_action_stop
    return args


def main() -> None:
    args = parse_args()
    evaluator = Stage2ProtocolEvaluator(_evaluator_args(args))
    print(
        f"Loading chat profile={args.chat_profile} protocol={args.tgvf_protocol} checkpoint={args.stage2_checkpoint}",
        flush=True,
    )
    evaluator.load()
    image = args.image
    mode = args.mode
    if image:
        _check_image(image)
    print("Ready.", flush=True)
    print("Commands: /image PATH | /mode direct|force|free | /maxfocus N | /raw on|off | /focus on|off | /reset | /profiles | /quit", flush=True)
    print("Inline image: type `/path/to/image.jpg your question` to send/switch image in one turn.", flush=True)
    print(f"stateful_kv={args.stateful_kv}", flush=True)
    if image:
        print(f"Current image: {image}", flush=True)
    else:
        print("No image set. Text chat works now; send an image anytime with `/path/to/image.jpg question` or /image PATH.", flush=True)

    out_handle = None
    if args.output_jsonl:
        out_path = Path(args.output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_handle = out_path.open("a")

    turn = 0
    kv_state: ChatKVState | None = None
    try:
        while True:
            try:
                text = input("You> ").strip()
            except EOFError:
                break
            if not text:
                continue
            inline_image, inline_question = parse_inline_image_question(text)
            if inline_image is not None:
                _check_image(inline_image)
                if image != inline_image:
                    kv_state = None
                image = inline_image
                text = inline_question
                print(f"Current image: {image}", flush=True)
                if not text:
                    continue
            elif text.startswith("/"):
                old_image = image
                image, mode = handle_command(text, image, mode, args)
                if getattr(args, "_reset_kv", False):
                    kv_state = None
                    args._reset_kv = False
                    print("KV cache reset.", flush=True)
                elif image != old_image:
                    kv_state = None
                continue
            turn += 1
            started = time.perf_counter()
            try:
                if image and args.stateful_kv:
                    row, kv_state = run_stateful_chat_turn(
                        evaluator,
                        state=kv_state,
                        image=image,
                        question=text,
                        mode=mode,
                        args=args,
                    )
                elif image:
                    row = run_chat_turn(evaluator, image=image, question=text, mode=mode, args=args)
                else:
                    row = run_text_only_turn(evaluator, question=text, args=args)
            except Exception as exc:
                row = {
                    "turn": turn,
                    "image": image,
                    "question": text,
                    "mode": mode,
                    "error": f"{type(exc).__name__}: {exc}",
                    "wall_time_sec": time.perf_counter() - started,
                }
                print(f"Assistant> [error] {row['error']}", flush=True)
            else:
                row["turn"] = turn
                row["wall_time_sec"] = time.perf_counter() - started
                row["stateful_kv"] = bool(args.stateful_kv and image)
                if args.show_focus and row.get("focus_target"):
                    print(f"[focus] {row['focus_target']}", flush=True)
                answer = row.get("answer") or row.get("final_raw_output") or row.get("raw_output") or ""
                print(f"Assistant> {answer.strip()}", flush=True)
                if args.show_raw:
                    print(json.dumps(compact_chat_display_row(row), ensure_ascii=False, indent=2, default=str), flush=True)
            if out_handle is not None:
                out_handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                out_handle.flush()
    finally:
        if out_handle is not None:
            out_handle.close()


def _evaluator_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        stage2_checkpoint=args.stage2_checkpoint,
        eval_jsonl=args.eval_jsonl,
        output_dir="runs/tgvf_v3_chat/_loader",
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
        max_foveations=args.max_foveations,
        fvt_position_mode="native_source_grid",
        blocks="force_end2end",
        d_conditions="correct_D",
        max_focus=1,
        max_no_focus=0,
        min_confidence=None,
        num_shards=1,
        shard_index=0,
        wrong_search_limit=1,
        force_prefix_mode=args.force_prefix_mode,
        progress=False,
        log_every=8,
    )


@torch.no_grad()
def run_chat_turn(
    evaluator: Stage2ProtocolEvaluator,
    *,
    image: str,
    question: str,
    mode: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    sample = make_sample(image, question)
    img = evaluator._image(sample)
    if mode == "direct":
        direct = generate_direct_qwen3(
            evaluator.model,
            evaluator.processor,
            image=img,
            question=question,
            max_new_tokens=args.max_answer_tokens,
            device=evaluator.device,
            protocol=args.tgvf_protocol,
        )
        raw = direct["raw_output"]
        return {
            "image": image,
            "question": question,
            "mode": mode,
            "raw_output": raw,
            "final_raw_output": raw,
            "answer": clean_protocol_answer(raw),
            "triggered": False,
            "second_full_forward_used": False,
        }

    if mode == "force":
        forced_text = chat_force_prefix_text(args)
        capture = evaluator._capture_with_forced_text(image=img, question=question, forced_text=forced_text)
    elif mode == "free":
        capture = evaluator._capture_free_router(sample)
    else:
        raise ValueError(f"unsupported mode: {mode}")

    parsed = parse_v3_action(capture.generated_text, protocol=args.tgvf_protocol)
    if not capture.capture_found:
        raw = capture.generated_text
        return {
            "image": image,
            "question": question,
            "mode": mode,
            "raw_output": raw,
            "final_raw_output": raw,
            "answer": clean_protocol_answer(raw),
            "focus_target": "",
            "focus_valid": False,
            "triggered": False,
            "malformed": bool(parsed.malformed),
            "second_full_forward_used": bool(capture.second_full_forward_used),
        }

    row, _ = run_multi_focus_controller(
        evaluator,
        sample=sample,
        image_path=image,
        image_obj=img,
        initial_capture=capture,
        mode=mode,
        args=args,
        base_turns=0,
    )
    return row


@torch.no_grad()
def run_stateful_chat_turn(
    evaluator: Stage2ProtocolEvaluator,
    *,
    state: ChatKVState | None,
    image: str,
    question: str,
    mode: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], ChatKVState]:
    sample = make_sample(image, question)
    if state is None or state.image != image:
        state = prefill_stateful_image_turn(evaluator, image=image, question=question, args=args)
        kv_event = "prefill_image_question"
    else:
        state = append_user_turn_to_state(evaluator, state, question)
        kv_event = "append_user_turn"

    if mode == "direct":
        continuation = continue_generation_qwen3(
            evaluator.model,
            evaluator.processor,
            state,
            max_new_tokens=args.max_answer_tokens,
            eos_token_id=evaluator.processor.tokenizer.eos_token_id,
        )
        state = update_state_from_generation(state, continuation)
        raw = continuation.generated_text
        return (
            {
                "image": image,
                "question": question,
                "mode": mode,
                "raw_output": raw,
                "final_raw_output": raw,
                "answer": clean_protocol_answer(raw),
                "triggered": False,
                "kv_event": kv_event,
                "kv_turns": state.turns,
                "second_full_forward_used": False,
            },
            state,
        )

    forced_prefix = chat_force_prefix_text(args) if mode == "force" else None
    capture, state = capture_focus_from_state(
        evaluator,
        state,
        forced_prefix_text=forced_prefix,
        max_new_tokens=args.max_action_tokens,
        protocol=args.tgvf_protocol,
    )
    parsed = parse_v3_action(capture.generated_text, protocol=args.tgvf_protocol)
    if not capture.capture_found:
        raw = capture.generated_text
        state.turns += 1
        return (
            {
                "image": image,
                "question": question,
                "mode": mode,
                "raw_output": raw,
                "final_raw_output": raw,
                "answer": clean_protocol_answer(raw),
                "focus_target": "",
                "focus_valid": False,
                "triggered": False,
                "malformed": bool(parsed.malformed),
                "kv_event": kv_event,
                "kv_turns": state.turns,
                "second_full_forward_used": False,
            },
            state,
        )

    row, next_state = run_multi_focus_controller(
        evaluator,
        sample=sample,
        image_path=image,
        image_obj=state.image_obj,
        initial_capture=capture,
        mode=mode,
        args=args,
        base_turns=state.turns,
    )
    row["kv_event"] = kv_event
    if next_state is not None:
        state = next_state
    row["kv_turns"] = state.turns
    row["second_full_forward_used"] = False
    return row, state


@torch.no_grad()
def run_multi_focus_controller(
    evaluator: Stage2ProtocolEvaluator,
    *,
    sample: TGVFv3Stage2Sample,
    image_path: str,
    image_obj: Any,
    initial_capture: Qwen3FocusCapture,
    mode: str,
    args: argparse.Namespace,
    base_turns: int = 0,
) -> tuple[dict[str, Any], ChatKVState | None]:
    max_foveations = max(1, int(getattr(args, "max_foveations", 1)))
    current_capture = initial_capture
    final_state: ChatKVState | None = None
    focus_targets: list[str] = []
    action_outputs: list[str] = []
    continuation_outputs: list[str] = []
    hq_shapes: list[list[int]] = []
    d_shapes: list[list[int] | None] = []
    append_debug: list[dict[str, Any]] = []
    errors: list[str] = []
    second_full_forward_used = bool(current_capture.second_full_forward_used)
    conversation_events: list[dict[str, Any]] = []
    final_answer_source = ""

    for focus_index in range(max_foveations):
        target_text = current_capture.target_text or parse_v3_action(
            current_capture.generated_text,
            protocol=args.tgvf_protocol,
        ).focus_target
        focus_targets.append(target_text)
        action_outputs.append(current_capture.generated_text)
        conversation_events.append({"type": "assistant", "text": current_capture.generated_text})
        hq_shapes.append(list(current_capture.target_hidden_states.shape))
        try:
            correct_d = evaluator._d_from_capture(
                sample,
                current_capture,
                focus_source=f"chat_{mode}_focus{focus_index + 1}",
            )
            d = condition_d("correct_D", correct_d, evaluator)
            d_shapes.append(None if d is None else list(d.shape))
            append_result = append_answer_only(evaluator, current_capture, d, mode=args.post_tgvf_continuation)
            append_debug.append(getattr(append_result, "debug_metadata", {}))
        except StopIteration as exc:
            errors.append(f"focus_{focus_index + 1}_d_append_StopIteration:{exc}")
            d_shapes.append(None)
            append_result = evaluator._append_text_only_no_d(current_capture)
            debug = getattr(append_result, "debug_metadata", {}) or {}
            debug = {**debug, "fallback_reason": errors[-1]}
            append_debug.append(debug)
        except Exception as exc:
            errors.append(f"focus_{focus_index + 1}_d_append_{type(exc).__name__}:{exc}")
            d_shapes.append(None)
            append_result = evaluator._append_text_only_no_d(current_capture)
            debug = getattr(append_result, "debug_metadata", {}) or {}
            debug = {**debug, "fallback_reason": errors[-1]}
            append_debug.append(debug)
        conversation_events.append(
            {
                "type": "tool",
                "index": focus_index + 1,
                "target": target_text,
                "D_shape": d_shapes[-1],
                "debug": append_debug[-1] if append_debug else {},
            }
        )
        append_state = state_from_append_result(
            image_path=image_path,
            image_obj=image_obj,
            capture=current_capture,
            append_result=append_result,
            turns=base_turns,
        )

        if focus_index + 1 < max_foveations:
            try:
                next_capture, next_state = capture_focus_from_state(
                    evaluator,
                    append_state,
                    forced_prefix_text=None,
                    max_new_tokens=args.max_answer_tokens,
                    protocol=args.tgvf_protocol,
                )
            except StopIteration as exc:
                errors.append(f"focus_{focus_index + 1}_next_capture_StopIteration:{exc}")
                continuation = continue_generation_qwen3(
                    evaluator.model,
                    evaluator.processor,
                    append_result,
                    max_new_tokens=args.max_answer_tokens,
                    eos_token_id=evaluator.processor.tokenizer.eos_token_id,
                )
                continuation_outputs.append(continuation.generated_text)
                final_answer_source = continuation.generated_text
                conversation_events.append({"type": "assistant", "text": continuation.generated_text})
                final_state = update_state_from_generation(append_state, continuation)
                break
            except Exception as exc:
                errors.append(f"focus_{focus_index + 1}_next_capture_{type(exc).__name__}:{exc}")
                continuation = continue_generation_qwen3(
                    evaluator.model,
                    evaluator.processor,
                    append_result,
                    max_new_tokens=args.max_answer_tokens,
                    eos_token_id=evaluator.processor.tokenizer.eos_token_id,
                )
                continuation_outputs.append(continuation.generated_text)
                final_answer_source = continuation.generated_text
                conversation_events.append({"type": "assistant", "text": continuation.generated_text})
                final_state = update_state_from_generation(append_state, continuation)
                break
            continuation_outputs.append(next_capture.generated_text)
            second_full_forward_used = second_full_forward_used or bool(next_capture.second_full_forward_used)
            if next_capture.capture_found:
                current_capture = next_capture
                final_state = next_state
                continue
            final_answer_source = next_capture.generated_text
            conversation_events.append({"type": "assistant", "text": next_capture.generated_text})
            final_state = next_state
            break

        continuation = continue_generation_qwen3(
            evaluator.model,
            evaluator.processor,
            append_result,
            max_new_tokens=args.max_answer_tokens,
            eos_token_id=evaluator.processor.tokenizer.eos_token_id,
        )
        continuation_outputs.append(continuation.generated_text)
        final_answer_source = continuation.generated_text
        conversation_events.append({"type": "assistant", "text": continuation.generated_text})
        final_state = update_state_from_generation(append_state, continuation)
        break

    final_text = "\n".join(part for part in continuation_outputs if part)
    conversation_raw = render_chat_conversation_raw(conversation_events)
    if final_state is not None:
        final_state.turns = max(int(final_state.turns), int(base_turns) + 1)
    first_target = focus_targets[0] if focus_targets else ""
    return (
        {
            "image": image_path,
            "question": sample.question,
            "mode": mode,
            "raw_output": conversation_raw,
            "answer": clean_chat_answer(final_answer_source or final_text),
            "focus_target": first_target,
            "focus_targets": focus_targets,
            "num_foveations": len(focus_targets),
            "focus_valid": bool(focus_targets),
            "triggered": bool(focus_targets),
            "target_token_count": len(initial_capture.target_token_ids),
            "target_token_counts": [
                _target_token_count_from_action(action_output, args.tgvf_protocol, evaluator.processor.tokenizer)
                for action_output in action_outputs
            ],
            "H_q_shape": hq_shapes[0] if hq_shapes else None,
            "H_q_shapes": hq_shapes,
            "D_shape": d_shapes[0] if d_shapes else None,
            "D_shapes": d_shapes,
            "append_success": True,
            "continuation_not_im_end": bool(final_text.strip()),
            "second_full_forward_used": second_full_forward_used,
            "append_debug": append_debug,
            "errors": errors,
            "max_foveations": max_foveations,
        },
        final_state,
    )


def render_chat_conversation_raw(events: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for event in events:
        if event.get("type") == "assistant":
            text = str(event.get("text") or "").strip()
            if text:
                parts.append(text)
            continue
        if event.get("type") == "tool":
            shape = event.get("D_shape")
            target = str(event.get("target") or "").strip()
            debug = event.get("debug") if isinstance(event.get("debug"), dict) else {}
            leading_im_end = debug.get("tgvf_prefix_includes_leading_im_end")
            marker = f"[tool insert {event.get('index')}: TGVF visual evidence"
            if shape is not None:
                marker += f", D_shape={shape}"
            if target:
                marker += f", focus={target}"
            if leading_im_end is not None:
                marker += f", leading_im_end={bool(leading_im_end)}"
            marker += "]"
            parts.append(marker)
    return "\n".join(parts).strip()


def clean_chat_answer(text: str) -> str:
    original = text or ""
    cleaned = original.strip()
    cleaned = re.sub(r"(?:<\|im_end\|>\s*)+$", "", cleaned).strip()
    think_end = "</think>"
    last_think_end = cleaned.rfind(think_end)
    if last_think_end >= 0:
        cleaned = cleaned[last_think_end + len(think_end) :].strip()
    cleaned = re.sub(
        r"<\|focus_start\|>.*?<\|focus_end\|>(?:<\|im_end\|>)?",
        " ",
        cleaned,
        flags=re.DOTALL,
    )
    cleaned = re.sub(
        r"<\|tgvf_start\|>.*?<\|tgvf_end\|>(?:<\|im_end\|>)?",
        " ",
        cleaned,
        flags=re.DOTALL,
    )
    cleaned = cleaned.replace("<|im_end|>", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or clean_protocol_answer(original)


def compact_chat_display_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "image",
        "question",
        "mode",
        "raw_output",
        "answer",
        "focus_target",
        "focus_targets",
        "num_foveations",
        "triggered",
        "focus_valid",
        "append_success",
        "second_full_forward_used",
        "errors",
    )
    return {key: row[key] for key in keys if key in row and row.get(key) not in (None, [], "")}


def state_from_append_result(
    *,
    image_path: str,
    image_obj: Any,
    capture: Qwen3FocusCapture,
    append_result: Any,
    turns: int = 0,
) -> ChatKVState:
    return ChatKVState(
        image=image_path,
        image_obj=image_obj,
        past_key_values=append_result.past_key_values,
        attention_mask=append_result.attention_mask,
        input_ids=append_result.input_ids,
        last_logits=append_result.last_logits,
        source_visual_geometry=capture.source_visual_geometry,
        image_grid_thw=capture.image_grid_thw,
        video_grid_thw=capture.video_grid_thw,
        ended_with_eos=False,
        turns=turns,
    )


def _target_token_count_from_action(text: str, protocol: str, tokenizer: Any) -> int:
    parsed = parse_v3_action(text, protocol=protocol)
    if not parsed.focus_target:
        return 0
    return len(tokenizer.encode(parsed.focus_target, add_special_tokens=False))


@torch.no_grad()
def prefill_stateful_image_turn(
    evaluator: Stage2ProtocolEvaluator,
    *,
    image: str,
    question: str,
    args: argparse.Namespace,
) -> ChatKVState:
    img = evaluator._image(make_sample(image, question))
    messages = build_direct_messages(img, question)
    inputs = build_qwen3_inputs(evaluator.processor, messages)
    model_inputs = {
        key: value.to(evaluator.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    source_geometry = extract_qwen3_source_visual_geometry(evaluator.utility_model, model_inputs)
    outputs = evaluator.model(
        **model_inputs,
        use_cache=True,
        return_dict=True,
    )
    return ChatKVState(
        image=image,
        image_obj=img,
        past_key_values=outputs.past_key_values,
        attention_mask=model_inputs.get("attention_mask"),
        input_ids=model_inputs.get("input_ids"),
        last_logits=outputs.logits,
        source_visual_geometry=source_geometry,
        image_grid_thw=model_inputs.get("image_grid_thw"),
        video_grid_thw=model_inputs.get("video_grid_thw"),
        ended_with_eos=False,
        turns=0,
    )


@torch.no_grad()
def append_user_turn_to_state(
    evaluator: Stage2ProtocolEvaluator,
    state: ChatKVState,
    question: str,
) -> ChatKVState:
    if state.ended_with_eos:
        text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    else:
        text = f"\n<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    state, _, _ = append_text_to_state(evaluator, state, text, output_hidden_states=False)
    state.ended_with_eos = False
    return state


@torch.no_grad()
def append_text_to_state(
    evaluator: Stage2ProtocolEvaluator,
    state: ChatKVState,
    text: str,
    *,
    output_hidden_states: bool,
) -> tuple[ChatKVState, list[int], list[torch.Tensor]]:
    token_ids = _encode_text(evaluator.processor.tokenizer, text, evaluator.device)
    ids = [int(x) for x in token_ids.detach().cpu().tolist()]
    if not ids:
        return state, [], []
    attention_mask = state.attention_mask
    if attention_mask is not None:
        attention_mask = _extend_attention(attention_mask.to(evaluator.device), int(token_ids.shape[0]))
    input_ids = state.input_ids
    if input_ids is not None:
        input_ids = torch.cat([input_ids.to(evaluator.device), token_ids.view(1, -1)], dim=-1)
    outputs = evaluator.model(
        input_ids=token_ids.view(1, -1).to(evaluator.device),
        past_key_values=state.past_key_values,
        attention_mask=attention_mask,
        position_ids=_chunk_position_ids_1d(
            attention_mask=attention_mask,
            chunk_length=int(token_ids.shape[0]),
            device=evaluator.device,
        ),
        use_cache=True,
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )
    hidden: list[torch.Tensor] = []
    if output_hidden_states:
        hidden = [item.detach() for item in outputs.hidden_states[-1][0]]
    state.past_key_values = outputs.past_key_values
    state.attention_mask = attention_mask
    state.input_ids = input_ids
    state.last_logits = outputs.logits
    return state, ids, hidden


@torch.no_grad()
def capture_focus_from_state(
    evaluator: Stage2ProtocolEvaluator,
    state: ChatKVState,
    *,
    forced_prefix_text: str | None,
    max_new_tokens: int,
    protocol: str,
) -> tuple[Qwen3FocusCapture, ChatKVState]:
    tokenizer = evaluator.processor.tokenizer
    all_ids: list[int] = []
    all_hidden: list[torch.Tensor] = []
    if forced_prefix_text:
        state, prefix_ids, prefix_hidden = append_text_to_state(
            evaluator,
            state,
            forced_prefix_text,
            output_hidden_states=True,
        )
        all_ids.extend(prefix_ids)
        all_hidden.extend(prefix_hidden)
        found = completed_focus_from_ids(tokenizer, all_ids, all_hidden, protocol=protocol)
        if found is not None and focus_action_ready_for_append(tokenizer, all_ids, found, protocol=protocol):
            return make_capture_from_state(state, tokenizer, all_ids, all_hidden, found, protocol), state

    eos_id = tokenizer.eos_token_id
    stop_reason = "max_new_tokens"
    for _ in range(max_new_tokens):
        next_token = torch.argmax(state.last_logits[:, -1, :], dim=-1, keepdim=True)
        token_id = int(next_token[0, 0].detach().cpu().item())
        attention_mask = state.attention_mask
        if attention_mask is not None:
            attention_mask = _extend_attention(attention_mask.to(evaluator.device), 1)
        input_ids = state.input_ids
        if input_ids is not None:
            input_ids = torch.cat([input_ids.to(evaluator.device), next_token.to(evaluator.device)], dim=-1)
        outputs = evaluator.model(
            input_ids=next_token.to(evaluator.device),
            past_key_values=state.past_key_values,
            attention_mask=attention_mask,
            position_ids=_chunk_position_ids_1d(
                attention_mask=attention_mask,
                chunk_length=1,
                device=evaluator.device,
            ),
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        state.past_key_values = outputs.past_key_values
        state.attention_mask = attention_mask
        state.input_ids = input_ids
        state.last_logits = outputs.logits
        all_ids.append(token_id)
        all_hidden.append(outputs.hidden_states[-1][0, -1].detach())
        found = completed_focus_from_ids(tokenizer, all_ids, all_hidden, protocol=protocol)
        if found is not None and focus_action_ready_for_append(tokenizer, all_ids, found, protocol=protocol):
            return make_capture_from_state(state, tokenizer, all_ids, all_hidden, found, protocol), state
        if eos_id is not None and token_id == eos_id:
            stop_reason = "eos_token"
            state.ended_with_eos = True
            break

    text = _decode(tokenizer, all_ids)
    parsed = parse_v3_action(text, protocol=protocol)
    capture = Qwen3FocusCapture(
        target_text="",
        target_token_ids=[],
        target_hidden_states=empty_hidden_like(all_hidden, evaluator.device),
        generated_ids=all_ids,
        generated_text=text,
        generated_hidden_states=stack_hidden(all_hidden, evaluator.device),
        past_key_values=state.past_key_values,
        attention_mask=state.attention_mask,
        cache_position=None,
        input_ids=state.input_ids,
        last_logits=state.last_logits,
        model_kwargs={},
        image_grid_thw=state.image_grid_thw,
        video_grid_thw=state.video_grid_thw,
        source_visual_geometry=state.source_visual_geometry,
        stop_reason=stop_reason,
        capture_found=False,
        second_full_forward_used=False,
        malformed=bool(parsed.malformed),
        errors=list(parsed.malformed_reasons),
    )
    return capture, state


def completed_focus_from_ids(
    tokenizer: Any,
    ids: list[int],
    hidden: list[torch.Tensor],
    *,
    protocol: str,
) -> tuple[int, int, str] | None:
    del hidden
    focus_start, focus_end = protocol_focus_tokens(protocol)
    start_ids = tokenizer.encode(focus_start, add_special_tokens=False)
    end_ids = tokenizer.encode(focus_end, add_special_tokens=False)
    start = find_subsequence(ids, start_ids)
    if start < 0:
        return None
    target_start = start + len(start_ids)
    end = find_subsequence(ids[target_start:], end_ids)
    if end < 0:
        return None
    target_end = target_start + end
    target_text = _decode(tokenizer, ids[target_start:target_end]).strip()
    return target_start, target_end, target_text


def focus_action_ready_for_append(
    tokenizer: Any,
    ids: list[int],
    found: tuple[int, int, str],
    *,
    protocol: str,
) -> bool:
    if protocol != "protocol_c_tool_observation":
        return True
    if os.environ.get("TGVF_TOOLOBS_ACTION_STOP", "im_end").strip().lower() == "focus_end":
        return True
    _, target_end, _ = found
    focus_end = protocol_focus_tokens(protocol)[1]
    focus_end_ids = tokenizer.encode(focus_end, add_special_tokens=False)
    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if not focus_end_ids or not im_end_ids:
        return True
    focus_end_end = target_end + len(focus_end_ids)
    return find_subsequence(ids[focus_end_end:], im_end_ids) >= 0


def make_capture_from_state(
    state: ChatKVState,
    tokenizer: Any,
    ids: list[int],
    hidden: list[torch.Tensor],
    found: tuple[int, int, str],
    protocol: str,
) -> Qwen3FocusCapture:
    del protocol
    target_start, target_end, target_text = found
    target_ids = ids[target_start:target_end]
    target_hidden = stack_hidden(hidden[target_start:target_end], state.last_logits.device)
    return Qwen3FocusCapture(
        target_text=target_text,
        target_token_ids=target_ids,
        target_hidden_states=target_hidden,
        generated_ids=list(ids),
        generated_text=_decode(tokenizer, ids),
        generated_hidden_states=stack_hidden(hidden, state.last_logits.device),
        past_key_values=state.past_key_values,
        attention_mask=state.attention_mask,
        cache_position=None,
        input_ids=state.input_ids,
        last_logits=state.last_logits,
        model_kwargs={},
        image_grid_thw=state.image_grid_thw,
        video_grid_thw=state.video_grid_thw,
        source_visual_geometry=state.source_visual_geometry,
        target_token_start=target_start,
        target_token_end=target_end,
        stop_reason="focus_action_terminal",
        capture_found=True,
        second_full_forward_used=False,
        malformed=False,
        errors=[],
    )


def update_state_from_generation(state: ChatKVState, generation: Any) -> ChatKVState:
    state.past_key_values = generation.past_key_values
    state.attention_mask = generation.attention_mask
    state.input_ids = generation.input_ids
    state.last_logits = generation.last_logits
    eos = None
    try:
        eos = generation.generated_ids[-1]
    except Exception:
        pass
    state.ended_with_eos = getattr(generation, "stop_reason", None) == "eos_token"
    state.turns += 1
    return state


def update_state_from_append_and_generation(state: ChatKVState, append_result: Any, generation: Any) -> ChatKVState:
    del append_result
    state.past_key_values = generation.past_key_values
    state.attention_mask = generation.attention_mask
    state.input_ids = generation.input_ids
    state.last_logits = generation.last_logits
    state.ended_with_eos = getattr(generation, "stop_reason", None) == "eos_token"
    state.turns += 1
    return state


def stack_hidden(hidden: list[torch.Tensor], device: torch.device | str) -> torch.Tensor:
    if not hidden:
        return empty_hidden_like(hidden, device)
    return torch.stack([item.to(device) for item in hidden], dim=0)


def empty_hidden_like(hidden: list[torch.Tensor], device: torch.device | str) -> torch.Tensor:
    if hidden:
        return torch.empty((0, int(hidden[0].shape[-1])), device=device, dtype=hidden[0].dtype)
    return torch.empty((0, 0), device=device)


def find_subsequence(values: list[int], pattern: list[int]) -> int:
    if not pattern:
        return -1
    limit = len(values) - len(pattern) + 1
    for index in range(max(limit, 0)):
        if values[index : index + len(pattern)] == pattern:
            return index
    return -1


@torch.no_grad()
def run_text_only_turn(
    evaluator: Stage2ProtocolEvaluator,
    *,
    question: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": question}],
        }
    ]
    inputs = build_qwen3_inputs(evaluator.processor, messages)
    if args.text_only_action_stop:
        capture = capture_text_only_focus_action(
            evaluator,
            inputs=inputs,
            max_new_tokens=args.max_answer_tokens,
            protocol=args.tgvf_protocol,
        )
        parsed = parse_v3_action(capture.generated_text, protocol=args.tgvf_protocol)
        if capture.capture_found:
            return {
                "image": None,
                "question": question,
                "mode": "text_only_action_stopped",
                "raw_output": capture.generated_text,
                "final_raw_output": capture.generated_text,
                "answer": "",
                "focus_target": capture.target_text or parsed.focus_target,
                "focus_targets": [capture.target_text or parsed.focus_target],
                "focus_valid": True,
                "triggered": False,
                "text_only_focus_generated": True,
                "second_full_forward_used": False,
                "output_tokens": len(capture.generated_ids),
                "stop_reason": capture.stop_reason,
            }
        raw = capture.generated_text
        return {
            "image": None,
            "question": question,
            "mode": "text_only",
            "raw_output": raw,
            "final_raw_output": raw,
            "answer": clean_protocol_answer(raw),
            "triggered": False,
            "text_only_focus_generated": False,
            "second_full_forward_used": False,
            "output_tokens": len(capture.generated_ids),
            "stop_reason": capture.stop_reason,
        }
    device = evaluator.device
    model_inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    prompt_len = int(model_inputs["input_ids"].shape[-1])
    generated = evaluator.model.generate(
        **model_inputs,
        max_new_tokens=args.max_answer_tokens,
        do_sample=False,
    )
    new_ids = generated[0, prompt_len:].detach().cpu().tolist()
    raw = _decode(evaluator.processor.tokenizer, new_ids)
    return {
        "image": None,
        "question": question,
        "mode": "text_only",
        "raw_output": raw,
        "final_raw_output": raw,
        "answer": clean_protocol_answer(raw),
        "triggered": False,
        "second_full_forward_used": False,
        "output_tokens": len(new_ids),
    }


@torch.no_grad()
def capture_text_only_focus_action(
    evaluator: Stage2ProtocolEvaluator,
    *,
    inputs: dict[str, Any],
    max_new_tokens: int,
    protocol: str,
) -> Qwen3FocusCapture:
    model_inputs = {
        key: value.to(evaluator.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    outputs = evaluator.model(
        **model_inputs,
        use_cache=True,
        return_dict=True,
    )
    state = ChatKVState(
        image="",
        image_obj=None,
        past_key_values=outputs.past_key_values,
        attention_mask=model_inputs.get("attention_mask"),
        input_ids=model_inputs.get("input_ids"),
        last_logits=outputs.logits,
        source_visual_geometry=None,
        image_grid_thw=model_inputs.get("image_grid_thw"),
        video_grid_thw=model_inputs.get("video_grid_thw"),
        ended_with_eos=False,
        turns=0,
    )
    capture, _state = capture_focus_from_state(
        evaluator,
        state,
        forced_prefix_text=None,
        max_new_tokens=max_new_tokens,
        protocol=protocol,
    )
    return capture


def make_sample(image: str, question: str) -> TGVFv3Stage2Sample:
    return TGVFv3Stage2Sample(
        image=image,
        question=question,
        answer="",
        need_focus=True,
        evidence_state=NEED_LOCAL_EVIDENCE,
        trajectory_type="single_focus",
        target="",
        evidence_description="",
        image_id=Path(image).name,
        source_dataset="chat",
        source_profile="unknown",
        answer_format="free_text",
    )


def chat_force_prefix_text(args: argparse.Namespace) -> str:
    mode = getattr(args, "force_prefix_mode", "trained_need_then_focus_start")
    if args.tgvf_protocol not in {"protocol_c_thinking_special", "protocol_c_tool_observation", "protocol_e_action_evidence_special"}:
        return render_force_focus_prefix(protocol=args.tgvf_protocol)
    if mode == "empty_think_then_focus_start":
        return render_force_focus_prefix(protocol=args.tgvf_protocol)
    if mode == "generic_need_then_focus_start":
        return "<think>\nI need to inspect local visual evidence before answering.\n</think>\n<|focus_start|>"
    if mode == "think_open_only":
        return "<think>\n"
    # Default for interactive chat: match the supervised Protocol C focus action.
    return "<think>\nI need visual focus before answering.\n</think>\n<|focus_start|>"


def parse_inline_image_question(text: str) -> tuple[str | None, str]:
    stripped = text.strip()
    if not stripped:
        return None, ""
    # Terminal-friendly image sending: paste/drag a path, then the question.
    # Examples:
    #   /tmp/a.jpg what is the tiny text?
    #   /tmp/a.jpg
    #   "/tmp/my image.jpg" what is visible?
    if stripped[0] in {"'", '"'}:
        quote = stripped[0]
        end = stripped.find(quote, 1)
        if end > 1:
            candidate = stripped[1:end]
            if looks_like_image_path(candidate) and Path(candidate).exists():
                return candidate, stripped[end + 1 :].strip()
    parts = stripped.split(maxsplit=1)
    candidate = parts[0]
    if looks_like_image_path(candidate) and Path(candidate).exists():
        return candidate, parts[1].strip() if len(parts) > 1 else ""
    return None, stripped


def looks_like_image_path(value: str) -> bool:
    lower = value.lower()
    return lower.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")) or "/" in value


def handle_command(text: str, image: str | None, mode: str, args: argparse.Namespace) -> tuple[str | None, str]:
    parts = text.split(maxsplit=1)
    cmd = parts[0].lower()
    rest = parts[1].strip() if len(parts) > 1 else ""
    if cmd in {"/quit", "/q", "/exit"}:
        raise SystemExit(0)
    if cmd == "/image":
        if not rest:
            print("Usage: /image /path/to/image.jpg", flush=True)
            return image, mode
        _check_image(rest)
        print(f"Current image: {rest}", flush=True)
        return rest, mode
    if cmd == "/reset":
        args._reset_kv = True
        return image, mode
    if cmd == "/mode":
        if rest not in {"direct", "force", "free"}:
            print("Usage: /mode direct|force|free", flush=True)
            return image, mode
        print(f"Mode: {rest}", flush=True)
        return image, rest
    if cmd == "/maxfocus":
        try:
            value = int(rest)
        except ValueError:
            print("Usage: /maxfocus N", flush=True)
            return image, mode
        args.max_foveations = max(1, value)
        print(f"max_foveations={args.max_foveations}", flush=True)
        return image, mode
    if cmd == "/raw":
        if rest.lower() in {"on", "1", "true"}:
            args.show_raw = True
        elif rest.lower() in {"off", "0", "false"}:
            args.show_raw = False
        print(f"show_raw={args.show_raw}", flush=True)
        return image, mode
    if cmd == "/profiles":
        print_profile_list()
        return image, mode
    if cmd == "/focus":
        if rest.lower() in {"on", "1", "true"}:
            args.show_focus = True
        elif rest.lower() in {"off", "0", "false"}:
            args.show_focus = False
        print(f"show_focus={args.show_focus}", flush=True)
        return image, mode
    print("Commands: /image PATH | /mode direct|force|free | /maxfocus N | /raw on|off | /focus on|off | /reset | /profiles | /quit", flush=True)
    print("Inline image: `/path/to/image.jpg your question`", flush=True)
    return image, mode


def _check_image(path: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"image not found: {path}")


def clean_protocol_answer(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    text = re.sub(r"<\|im_end\|>.*$", "", text, flags=re.DOTALL).strip()
    legacy_answer = re.search(r"<ANSWER>(.*?)</ANSWER>", text, flags=re.DOTALL)
    if legacy_answer:
        return legacy_answer.group(1).strip()
    # Protocol C/D answer is usually after the last </think>.
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    # Protocol E answer is usually after the evidence block.
    if "<|evidence_end|>" in text:
        text = text.split("<|evidence_end|>")[-1].strip()
    text = re.sub(r"<\|[^>]+\|>", " ", text).strip()
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"</?(?:EVIDENCE|TGVF|FOCUS|EVIDENCE_STATE)>", " ", text).strip()
    return text or str(text).strip()


if __name__ == "__main__":
    main()
