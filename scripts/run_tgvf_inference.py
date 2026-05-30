from __future__ import annotations

import argparse
import json

import torch

from revisit_vlm.models.qwen2vl import load_qwen2vl
from revisit_vlm.tgvf_inference import format_tgvf_inference_debug, run_tgvf_inference
from revisit_vlm.tgvf_training import (
    TGVF_VARIANTS,
    build_tgvf_module,
    infer_tgvf_dims,
    load_tgvf_module_checkpoint,
)


def main() -> None:
    args = parse_args()
    loaded = load_qwen2vl(
        args.model_name_or_path,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        device_map={"": args.device} if args.device.startswith("cuda") else None,
        trust_remote_code=False,
    )
    model = loaded.model
    processor = loaded.processor
    model.eval()

    d_lm, d_v, inferred_merge_size = infer_tgvf_dims(model)
    spatial_merge_size = (
        inferred_merge_size
        if args.spatial_merge_size == "auto"
        else int(args.spatial_merge_size)
    )
    foveal_module = build_tgvf_module(
        variant=args.variant,
        d_lm=d_lm,
        d_v=d_v,
        num_foveated_tokens=args.num_foveated_tokens,
        spatial_merge_size=spatial_merge_size,
        attn_dim=args.attn_dim,
    ).to(args.device)
    if args.checkpoint:
        load_tgvf_module_checkpoint(foveal_module, args.checkpoint)
    foveal_module.eval()

    result = run_tgvf_inference(
        model=model,
        processor=processor,
        foveal_module=foveal_module,
        image=args.image,
        question=args.question,
        device=args.device,
        capture_max_new_tokens=args.capture_max_new_tokens,
        answer_max_new_tokens=args.answer_max_new_tokens,
    )
    print(format_tgvf_inference_debug(result))
    print(json.dumps(result.debug_metadata, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run structural TGVF inference.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument(
        "--variant",
        choices=TGVF_VARIANTS,
        default="foveal_cross_merger",
    )
    parser.add_argument("--num-foveated-tokens", type=int, default=16)
    parser.add_argument("--spatial-merge-size", default="auto")
    parser.add_argument("--attn-dim", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--capture-max-new-tokens", type=int, default=128)
    parser.add_argument("--answer-max-new-tokens", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
