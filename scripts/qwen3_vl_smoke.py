#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from revisit_vlm.qwen3_vl_tgvf import (
    capture_focus_single_pass_qwen3,
    capture_to_row,
    continue_generation_qwen3,
    generate_direct_qwen3,
    llm_hidden_dim,
    load_qwen3_vl,
    make_smoke_d,
    peak_memory_gb,
    summarize_smoke,
    tap_qwen3_vision_features,
    write_json,
    write_jsonl,
    append_tgvf_visual_tokens_qwen3,
)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    loaded = load_qwen3_vl(
        args.model_id,
        processor_id=args.processor_id,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )

    start = time.perf_counter()
    row_errors: list[str] = []
    direct = None
    capture = None
    vision_tap = None
    append_result = None
    continuation = None

    try:
        if args.mode == "direct":
            direct = generate_direct_qwen3(
                loaded.model,
                loaded.processor,
                image=args.image,
                question=args.question,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
            )
        elif args.mode == "focus_force":
            capture = capture_focus_single_pass_qwen3(
                loaded.model,
                loaded.processor,
                image=args.image,
                question=args.question,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
                eos_token_id=loaded.processor.tokenizer.eos_token_id,
                force_action_prefix=args.force_action_prefix,
                scripted_target_text=args.scripted_focus_target,
            )
        elif args.mode == "capture_focus":
            capture = capture_focus_single_pass_qwen3(
                loaded.model,
                loaded.processor,
                image=args.image,
                question=args.question,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
                eos_token_id=loaded.processor.tokenizer.eos_token_id,
                force_action_prefix=args.force_action_prefix,
                scripted_target_text=args.scripted_focus_target,
            )
            if args.assert_no_second_full_forward and capture.second_full_forward_used:
                raise RuntimeError("capture path used a second full forward")
        elif args.mode == "vision_tap":
            vision_tap, _, _ = tap_qwen3_vision_features(
                loaded.model,
                loaded.processor,
                image=args.image,
                question=args.question,
                device=args.device,
            )
        elif args.mode == "append_smoke":
            capture = capture_focus_single_pass_qwen3(
                loaded.model,
                loaded.processor,
                image=args.image,
                question=args.question,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
                eos_token_id=loaded.processor.tokenizer.eos_token_id,
                force_action_prefix=args.force_action_prefix,
                scripted_target_text=args.scripted_focus_target,
            )
            if args.assert_no_second_full_forward and capture.second_full_forward_used:
                raise RuntimeError("capture path used a second full forward")
            vision_tap, v_pre, v_merge = tap_qwen3_vision_features(
                loaded.model,
                loaded.processor,
                image=args.image,
                question=args.question,
                device=args.device,
            )
            hidden_dim = llm_hidden_dim(loaded.model)
            reference = v_merge if v_merge is not None and v_merge.shape[-1] == hidden_dim else None
            if reference is None and v_pre is not None and v_pre.shape[-1] == hidden_dim:
                reference = v_pre
            source_visual_token_count = (
                capture.source_visual_geometry.source_visual_token_count
                if capture.source_visual_geometry is not None
                else 0
            )
            if source_visual_token_count <= 0:
                raise RuntimeError("capture did not expose a positive source visual token count")
            if args.num_fvt_tokens is not None and args.num_fvt_tokens != source_visual_token_count:
                raise ValueError(
                    "--num-fvt-tokens is legacy-only for append_smoke; "
                    f"D length must equal source visual token count {source_visual_token_count}"
                )
            d = make_smoke_d(
                source=args.d_source,
                num_fvt_tokens=source_visual_token_count,
                hidden_dim=hidden_dim,
                reference=reference,
                device=args.device or next(loaded.model.parameters()).device,
                dtype=next(loaded.model.parameters()).dtype,
            )
            append_result = append_tgvf_visual_tokens_qwen3(
                loaded.model,
                loaded.processor,
                capture,
                d,
                continuation_instruction=args.continuation_instruction,
                force_answer_tag=args.force_answer_tag,
                position_mode=args.fvt_position_mode,
            )
            continuation = continue_generation_qwen3(
                loaded.model,
                loaded.processor,
                append_result,
                max_new_tokens=args.answer_max_new_tokens,
                eos_token_id=loaded.processor.tokenizer.eos_token_id,
            )
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")
    except Exception as exc:
        row_errors.append(f"{type(exc).__name__}: {exc}")
        if args.raise_errors:
            raise

    wall = time.perf_counter() - start
    row = capture_to_row(
        sample_id=args.sample_id,
        image=args.image,
        question=args.question,
        mode=args.mode,
        model_id=args.model_id,
        capture=capture,
        vision_tap=vision_tap,
        append_result=append_result,
        continuation=continuation,
        direct=direct,
        errors=row_errors,
        wall_time_sec=wall,
        peak_memory_gb=peak_memory_gb(),
    )
    row.update(
        {
            "processor_id": loaded.processor_id,
            "dtype": loaded.dtype,
            "device_map": loaded.device_map,
            "attn_implementation": loaded.attn_implementation,
            "input_image_path": args.image,
            "prompt": args.question,
            "avg_output_tokens": len((direct or {}).get("generated_ids", []))
            if direct is not None
            else len(continuation.generated_ids)
            if continuation is not None
            else len(capture.generated_ids)
            if capture is not None
            else 0,
            "scripted_focus_target": args.scripted_focus_target,
        }
    )

    sample_path = out_dir / f"{args.mode}_samples.jsonl"
    summary_path = out_dir / f"{args.mode}_summary.json"
    write_jsonl(sample_path, [row])
    write_json(summary_path, summarize_smoke([row]))
    print(f"wrote {sample_path}")
    print(f"wrote {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-VL-Thinking TGVF-v3 deployment smoke.")
    parser.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Thinking")
    parser.add_argument("--processor_id", default=None)
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", default="What is visible in the image?")
    parser.add_argument(
        "--mode",
        choices=("direct", "focus_force", "capture_focus", "vision_tap", "append_smoke"),
        default="direct",
    )
    parser.add_argument("--output-dir", default="runs/qwen3_deploy_smoke")
    parser.add_argument("--sample-id", default="sample_000001")
    parser.add_argument("--dtype", default="bfloat16", choices=("auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"))
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--answer-max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--num-fvt-tokens",
        type=int,
        default=None,
        help="Legacy diagnostic override. v3 append_smoke defaults to source visual token count.",
    )
    parser.add_argument(
        "--fvt-position-mode",
        choices=("native_source_grid", "inherit_source_visual_positions"),
        default="native_source_grid",
    )
    parser.add_argument(
        "--d-source",
        choices=("zero", "random", "random_calibrated"),
        default="random_calibrated",
    )
    parser.add_argument("--continuation-instruction", default="")
    parser.add_argument("--force-answer-tag", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-action-prefix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scripted-focus-target", default=None)
    parser.add_argument("--assert-no-second-full-forward", action="store_true")
    parser.add_argument("--raise-errors", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
