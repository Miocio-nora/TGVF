#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for item in (ROOT, SRC):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from revisit_vlm.qwen3_vl_tgvf import load_qwen3_vl, peak_memory_gb
from revisit_vlm.tgvf_training import build_tgvf_module
from revisit_vlm.tgvf_v3_stage1 import (
    TGVFv3Stage1Dataset,
    collect_v3_stage1_features,
    freeze_qwen_backbone,
    infer_qwen3_stage1_dims,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test TGVF encoder-reencode @8/@16/@24 variant.")
    parser.add_argument("--model-id", default="/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking")
    parser.add_argument("--processor-id", default=None)
    parser.add_argument("--eval-jsonl", default="data/tgvf_teacher/generated/runs/tgvf_v3_teacher_val_2k/final/tgvf_teacher_items.accepted.jsonl")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--tgvf-protocol", default="legacy_v3_tags")
    parser.add_argument("--max-image-resolution", type=int, default=512)
    parser.add_argument("--encoder-adapter-layers", default="8,16,24")
    parser.add_argument("--encoder-adapter-layer-index-base", type=int, choices=(0, 1), default=0)
    parser.add_argument("--encoder-reencode-deepstack-compatible", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dataset = TGVFv3Stage1Dataset(args.eval_jsonl, focus_only=True)
    sample = dataset.samples[int(args.sample_index)]
    loaded = load_qwen3_vl(
        args.model_id,
        processor_id=args.processor_id,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )
    model = loaded.model
    processor = loaded.processor
    freeze_qwen_backbone(model)
    dims = infer_qwen3_stage1_dims(
        model=model,
        processor=processor,
        sample=sample,
        device=device,
        max_image_resolution=args.max_image_resolution,
    )
    layers = tuple(int(item.strip()) for item in args.encoder_adapter_layers.split(",") if item.strip())
    module = build_tgvf_module(
        variant="tgvf_encoder_bidir_8_16_24",
        d_lm=dims["d_lm"],
        d_v=dims["d_v"],
        num_foveated_tokens=None,
        spatial_merge_size=dims["spatial_merge_size"],
        encoder_adapter_layers=layers,
        encoder_adapter_layer_index_base=args.encoder_adapter_layer_index_base,
        encoder_adapter_gate_init=0.0,
        encoder_reencode_deepstack_compatible=args.encoder_reencode_deepstack_compatible,
    ).to(device=device, dtype=next(model.parameters()).dtype)
    gate_values = {
        name: float(adapter.alpha.detach().float().cpu().item())
        for name, adapter in module.adapters.items()
    }
    if any(abs(value) > 1e-8 for value in gate_values.values()):
        raise RuntimeError(f"nonzero initial gate values: {gate_values}")
    features = collect_v3_stage1_features(
        model=model,
        processor=processor,
        sample=sample,
        device=device,
        max_image_resolution=args.max_image_resolution,
        protocol=args.tgvf_protocol,
    )
    image_input = {
        "type": "image",
        "image": sample.image,
        "max_pixels": int(args.max_image_resolution) * int(args.max_image_resolution),
    }
    output = module(
        target_hidden_states=features.target_hidden_states.to(device),
        pre_merge_visual_tokens=features.pre_merge_visual_tokens.to(device),
        metadata={
            "qwen_model": model,
            "processor": processor,
            "image": image_input,
            "question": sample.prompt_question,
            "target": sample.target,
            "stage": "tgvf_encoder_reencode_smoke",
            "device": device,
        },
    )
    summary = {
        "variant": module.variant_name,
        "requested_layers": list(module.requested_adapter_layers),
        "actual_indices": list(module.actual_adapter_indices),
        "gate_values": gate_values,
        "H_q_shape": list(features.target_hidden_states.shape),
        "V_pre_shape": list(features.pre_merge_visual_tokens.shape),
        "D_shape": list(output.foveated_visual_tokens.shape),
        "output_deepstack_feature_count": 0 if output.deepstack_visual_embeds is None else len(output.deepstack_visual_embeds),
        "output_deepstack_feature_shapes": [] if output.deepstack_visual_embeds is None else [list(item.shape) for item in output.deepstack_visual_embeds],
        "debug_metadata": output.debug_metadata,
        "base_qwen_trainable_params": sum(int(p.numel()) for p in model.parameters() if p.requires_grad),
        "adapter_trainable_params": sum(int(p.numel()) for p in module.parameters() if p.requires_grad),
        "second_full_llm_forward": False,
        "peak_memory_gb": peak_memory_gb(),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
