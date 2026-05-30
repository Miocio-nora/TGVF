from __future__ import annotations

import argparse
import json

import torch

from revisit_vlm.models.qwen2vl import load_qwen2vl
from revisit_vlm.tgvf_foveal import finalize_tgvf_output_with_frozen_qwen_merger
from revisit_vlm.tgvf_training import (
    TGVF_VARIANTS,
    TeacherGuideDataset,
    build_tgvf_module,
    collect_training_features,
    compute_readout_lm_loss,
    freeze_qwen2vl,
    generate_readout_from_fvt,
    infer_tgvf_dims,
    load_tgvf_module_checkpoint,
    prepare_readout_inputs,
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
    freeze_qwen2vl(model)

    dataset = TeacherGuideDataset(args.sample_file)
    sample = dataset[args.sample_index]

    d_lm, d_v, inferred_merge_size = infer_tgvf_dims(model)
    spatial_merge_size = (
        inferred_merge_size if args.spatial_merge_size == "auto" else int(args.spatial_merge_size)
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

    features = collect_training_features(
        model=model,
        processor=processor,
        sample=sample,
        device=args.device,
        hidden_state_index=args.capture_layer,
    )
    with torch.no_grad():
        fvt_output = foveal_module(
            target_hidden_states=features.target_hidden_states.to(args.device),
            pre_merge_visual_tokens=features.pre_merge_visual_tokens.to(args.device),
            metadata={"target": sample.target},
        )
        fvt_output = finalize_tgvf_output_with_frozen_qwen_merger(model, fvt_output)
        readout_inputs = prepare_readout_inputs(
            model=model,
            tokenizer_or_processor=processor,
            target=sample.target,
            evidence_description=sample.evidence_description,
            foveated_visual_tokens=fvt_output.foveated_visual_tokens,
            image_grid_thw=(
                features.image_grid_thw
                if fvt_output.debug_metadata.get("tgvf_version") == "v2"
                else None
            ),
            device=args.device,
        )
        loss, _ = compute_readout_lm_loss(model=model, readout_inputs=readout_inputs)
        generated = generate_readout_from_fvt(
            model=model,
            tokenizer_or_processor=processor,
            target=sample.target,
            foveated_visual_tokens=fvt_output.foveated_visual_tokens,
            image_grid_thw=(
                features.image_grid_thw
                if fvt_output.debug_metadata.get("tgvf_version") == "v2"
                else None
            ),
            max_new_tokens=args.max_new_tokens,
            eos_token_id=processor.tokenizer.eos_token_id,
            device=args.device,
        )

    print(
        json.dumps(
            {
                "target": sample.target,
                "evidence_description": sample.evidence_description,
                "generated_readout": generated,
                "loss_gen": float(loss.detach().cpu()),
                "target_hidden_shape": list(features.target_hidden_states.shape),
                "pre_merge_visual_shape": list(features.pre_merge_visual_tokens.shape),
                "merged_visual_shape": list(features.merged_visual_tokens.shape),
                "foveated_visual_tokens_shape": list(fvt_output.foveated_visual_tokens.shape),
                "variant": args.variant,
                "second_full_forward_used": False,
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug TGVF fresh-readout training path.")
    parser.add_argument("--sample-file", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
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
    parser.add_argument("--capture-layer", type=int, default=-1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    main()
