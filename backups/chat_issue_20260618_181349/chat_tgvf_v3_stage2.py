#!/usr/bin/env python3
"""Interactive chat entrypoint for a TGVF-v3 Stage2 checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval.eval_v3_stage2_protocol import Stage2ProtocolEvaluator
from eval.eval_v3_vstar_force import append_answer_only
from revisit_vlm.qwen3_vl_tgvf import (
    PROTOCOL_C_TOOL_OBSERVATION,
    build_direct_messages,
    capture_focus_single_pass_qwen3,
    continue_generation_qwen3,
    parse_v3_action,
)
from revisit_vlm.tgvf_v3_stage2 import NEED_LOCAL_EVIDENCE, TGVFv3Stage2Sample


DEFAULT_CHECKPOINT = (
    "outputs/tgvf_v3_protocol_c/"
    "protocol_c_toolobs_stage2_v4data_clean_rowonly_gpu0_3_multifocus_focus_imend_from_clean_rowonly_focus_imend_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step_20260617/"
    "checkpoint_step_1200.pt"
)
DEFAULT_PROCESSOR = (
    "outputs/tgvf_v3_protocol_c/"
    "protocol_c_toolobs_stage2_v4data_clean_rowonly_gpu0_3_multifocus_focus_imend_from_clean_rowonly_focus_imend_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step_20260617/"
    "processor_step_1200"
)
DEFAULT_BOOTSTRAP_JSONL = (
    "data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/"
    "tgvf_v4_teacher_stage2_protocol_c.train.jsonl"
)
DEFAULT_MODEL_ID = "/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive TGVF-v3 Stage2 chat.")
    parser.add_argument("--stage2-checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--processor-id", default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--bootstrap-jsonl", default=DEFAULT_BOOTSTRAP_JSONL)
    parser.add_argument("--output-dir", default="outputs/chat_tgvf_v3_stage2/latest")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--tgvf-protocol", default="legacy_v3_tags")
    parser.add_argument("--max-image-resolution", type=int, default=512)
    parser.add_argument("--max-action-tokens", type=int, default=128)
    parser.add_argument("--max-answer-tokens", type=int, default=512)
    parser.add_argument("--post-tgvf-continuation", default="natural_continue")
    parser.add_argument("--force-action-prefix", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--retry-force-action-prefix", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--block-focus-in-continuation", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def evaluator_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        model_id=args.model_id,
        processor_id=args.processor_id,
        stage2_checkpoint=args.stage2_checkpoint,
        eval_jsonl=args.bootstrap_jsonl,
        output_dir=args.output_dir,
        device=args.device,
        device_map=args.device_map,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        tgvf_protocol=args.tgvf_protocol,
        min_confidence=0.0,
        max_focus=1,
        max_no_focus=0,
        max_action_tokens=args.max_action_tokens,
        max_answer_tokens=args.max_answer_tokens,
        max_image_resolution=args.max_image_resolution,
        fvt_position_mode="native_source_grid",
        blocks="free_router_end2end",
        variant="cross_attn",
        num_foveated_tokens=64,
        lora_rank=16,
        lora_alpha=32,
        lora_target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        d_conditions="correct_D",
        num_shards=1,
        shard_index=0,
        wrong_search_limit=256,
        force_prefix_mode="target_hint",
        progress=False,
        log_every=10,
    )


def make_chat_sample(image: str, question: str) -> TGVFv3Stage2Sample:
    values: dict[str, Any] = {
        "image": str(Path(image).expanduser().resolve()),
        "question": question,
        "answer": "",
        "need_focus": True,
        "evidence_state": NEED_LOCAL_EVIDENCE,
        "trajectory_type": "single_focus",
        "target": "",
        "evidence_description": "",
        "pre_focus_think": None,
        "post_focus_think": None,
        "no_focus_think": None,
        "value_span_text": None,
        "focus_steps": [],
        "confidence": 1.0,
        "metadata": {"source": "interactive_chat"},
    }
    kwargs: dict[str, Any] = {}
    for field in fields(TGVFv3Stage2Sample):
        if field.name in values:
            kwargs[field.name] = values[field.name]
        elif field.default is not MISSING:
            continue
        elif field.default_factory is not MISSING:  # type: ignore[attr-defined]
            continue
        else:
            kwargs[field.name] = None
    return TGVFv3Stage2Sample(**kwargs)


def capture_action(
    evaluator: Stage2ProtocolEvaluator,
    sample: TGVFv3Stage2Sample,
    *,
    force_action_prefix: bool,
):
    if not force_action_prefix:
        return evaluator._capture_free_router(sample)
    image = evaluator._image(sample)
    return capture_focus_single_pass_qwen3(
        evaluator.model,
        evaluator.processor,
        image=image,
        question=sample.prompt_question,
        messages=build_direct_messages(image, sample.prompt_question),
        max_new_tokens=evaluator.args.max_action_tokens,
        device=evaluator.device,
        force_action_prefix=True,
        protocol=evaluator.args.tgvf_protocol,
    )


def action_is_invalid(parsed: Any) -> bool:
    return bool(getattr(parsed, "malformed", False) or not getattr(parsed, "focus_target", None))


def run_one(evaluator: Stage2ProtocolEvaluator, sample: TGVFv3Stage2Sample, args: argparse.Namespace) -> None:
    capture = capture_action(evaluator, sample, force_action_prefix=args.force_action_prefix)
    parsed = parse_v3_action(capture.generated_text, protocol=evaluator.args.tgvf_protocol)
    print("\n[action]")
    print(capture.generated_text.strip())
    if action_is_invalid(parsed):
        print("\n[invalid action; not appending TGVF]")
        print(
            f"malformed={bool(getattr(parsed, 'malformed', False))} "
            f"focus_target={getattr(parsed, 'focus_target', None)!r}",
            flush=True,
        )
        return
    with torch.no_grad():
        d = evaluator._d_from_capture(sample, capture, focus_source="interactive_chat")
        appended = append_answer_only(evaluator, capture, d, mode=args.post_tgvf_continuation)
        continuation = continue_generation_qwen3(
            evaluator.model,
            evaluator.processor,
            appended,
            max_new_tokens=evaluator.args.max_answer_tokens,
            eos_token_id=evaluator.processor.tokenizer.eos_token_id,
        )
    print("\n[answer]")
    print(continuation.generated_text.strip())


def main() -> None:
    args = parse_args()
    if args.block_focus_in_continuation:
        os.environ.setdefault("TGVF_BLOCK_FOCUS_IN_CONTINUATION", "1")
    else:
        os.environ["TGVF_BLOCK_FOCUS_IN_CONTINUATION"] = "0"
    evaluator = Stage2ProtocolEvaluator(evaluator_args(args))
    evaluator.load()
    print("TGVF chat loaded. Enter an image path, then a question. Empty image path exits.", flush=True)
    while True:
        image = input("\nimage> ").strip()
        if not image:
            break
        question = input("question> ").strip()
        if not question:
            continue
        sample = make_chat_sample(image, question)
        try:
            run_one(evaluator, sample, args)
        except Exception as exc:
            print(f"\n[error] {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
