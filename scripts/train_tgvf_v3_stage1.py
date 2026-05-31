#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from revisit_vlm.qwen3_vl_tgvf import load_qwen3_vl, peak_memory_gb
from revisit_vlm.tgvf_training import (
    TGVF_DYNAMIC_NUM_FVT_VARIANTS,
    TGVF_VARIANTS,
    LossWeights,
    TGVFModuleConfig,
    build_tgvf_module,
    save_tgvf_checkpoint,
)
from revisit_vlm.tgvf_v3_stage1 import (
    TGVFv3Stage1Dataset,
    freeze_qwen_backbone,
    infer_qwen3_stage1_dims,
    tgvf_v3_stage1_collate,
    v3_stage1_training_step,
)
from revisit_vlm.wandb_logging import WandbLogger, flatten_metrics


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = TGVFv3Stage1Dataset(
        args.train_file,
        focus_only=True,
        min_confidence=args.min_confidence,
    )
    if len(dataset) == 0:
        raise RuntimeError("No v3 Stage1 focus samples were loaded")

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
        sample=dataset[0],
        device=device,
        max_image_resolution=args.max_image_resolution,
    )
    spatial_merge_size = (
        dims["spatial_merge_size"]
        if args.spatial_merge_size == "auto"
        else int(args.spatial_merge_size)
    )
    module_config = TGVFModuleConfig(
        variant=args.variant,
        num_foveated_tokens=args.num_foveated_tokens,
        spatial_merge_size=spatial_merge_size,
        attn_dim=args.attn_dim,
    )
    loss_weights = LossWeights(
        gen=args.loss_gen,
        visual_token_manifold=args.loss_visual_token_manifold,
        same_image_negative=args.loss_same_image_negative,
        contrastive_alignment=0.0,
    )
    config = {
        "stage": "tgvf_v3_stage1",
        "model_id": args.model_id,
        "processor_id": loaded.processor_id,
        "dataset": str(args.train_file),
        "dataset_version": "tgvf_teacher_schema_v3_or_legacy_focus_normalized",
        "num_focus_samples_used": len(dataset),
        "num_rows_skipped": len(dataset.skipped_rows),
        "tgvf": asdict(module_config),
        "loss_weights": asdict(loss_weights),
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "freeze_qwen": True,
        "mask_original_image_after_tgvf": args.mask_original_image_after_tgvf,
        "attention_mask_mode": (
            "weak_strict_original_image_keys_4d"
            if args.mask_original_image_after_tgvf
            else "standard_2d_causal"
        ),
        "fvt_position_mode": args.fvt_position_mode,
        "max_image_resolution": args.max_image_resolution,
        "max_pixels": None
        if args.max_image_resolution is None
        else int(args.max_image_resolution) * int(args.max_image_resolution),
        "dims": dims,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "attn_implementation": args.attn_implementation,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or output_dir.name,
        group=args.wandb_group,
        job_type="tgvf-v3-stage1-training",
        config={
            "training": config,
            "train_file": str(args.train_file),
            "output_dir": str(output_dir),
            "dataset_size": len(dataset),
            "skipped_rows": len(dataset.skipped_rows),
        },
        mode=args.wandb_mode,
        tags=[tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()] or None,
        directory=args.wandb_dir,
    )

    train_dtype = next(model.parameters()).dtype
    foveal_module = build_tgvf_module(
        variant=args.variant,
        d_lm=dims["d_lm"],
        d_v=dims["d_v"],
        num_foveated_tokens=args.num_foveated_tokens,
        spatial_merge_size=spatial_merge_size,
        attn_dim=args.attn_dim,
    ).to(device=device, dtype=train_dtype)
    foveal_module.train()
    if args.wandb_watch and wandb_logger.enabled:
        wandb_logger._wandb.watch(
            foveal_module,
            log=args.wandb_watch,
            log_freq=max(args.log_every, 1),
        )
    optimizer = torch.optim.AdamW(foveal_module.parameters(), lr=args.learning_rate)

    if args.resume_from_checkpoint:
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        foveal_module.load_state_dict(checkpoint["tgvf_module"], strict=True)
        if checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=tgvf_v3_stage1_collate,
    )
    data_iter = iter(loader)
    debug_examples_path = output_dir / "debug_examples.jsonl"
    optimizer.zero_grad(set_to_none=True)

    for step in range(1, args.max_steps + 1):
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        try:
            samples = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            samples = next(data_iter)

        output = v3_stage1_training_step(
            qwen_model=model,
            processor=processor,
            foveal_module=foveal_module,
            samples=samples,
            loss_weights=loss_weights,
            device=device,
            hidden_state_index=args.capture_layer,
            same_image_negative_margin=args.same_image_negative_margin,
            same_image_negative_mode=args.same_image_negative_mode,
            mask_original_image_after_tgvf=args.mask_original_image_after_tgvf,
            position_mode=args.fvt_position_mode,
            max_image_resolution=args.max_image_resolution,
        )
        if not torch.isfinite(output.loss_total):
            raise RuntimeError(f"Non-finite v3 Stage1 loss at step {step}: {output.loss_total}")
        output.loss_total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(foveal_module.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step == 1 or step % args.log_every == 0 or step == args.max_steps:
            log = {
                "step": step,
                "loss_total": float(output.loss_total.detach().cpu()),
                "loss_gen": float(output.loss_gen.detach().cpu()),
                "loss_visual_token_manifold": float(output.loss_visual_token_manifold.detach().cpu()),
                "loss_same_image_negative": float(output.loss_same_image_negative.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()),
                "model_id": args.model_id,
                "dataset": str(args.train_file),
                "num_focus_samples_used": len(dataset),
                "variant": args.variant,
                "device": str(device),
                "peak_memory_gb": peak_memory_gb(),
                **output.debug,
            }
            print(json.dumps(_json_safe(log), indent=2, ensure_ascii=False))
            if wandb_logger.enabled:
                wandb_logger.log(_wandb_train_metrics(log), step=step)
            with debug_examples_path.open("a", encoding="utf-8") as handle:
                for example in output.debug.get("debug_examples", [])[: args.max_debug_examples_per_log]:
                    handle.write(json.dumps(_json_safe({"step": step, **example}), ensure_ascii=False) + "\n")

        if step % args.save_every == 0 or step == args.max_steps:
            save_tgvf_checkpoint(
                path=output_dir / f"checkpoint_step_{step}.pt",
                foveal_module=foveal_module,
                config=config,
                optimizer=optimizer,
                global_step=step,
                optimizer_step=step,
            )
            if args.wandb_log_checkpoints and wandb_logger.enabled:
                wandb_logger.log_artifact(
                    name=f"{output_dir.name}-checkpoint-{step}",
                    artifact_type="tgvf-v3-stage1-checkpoint",
                    paths=[output_dir / f"checkpoint_step_{step}.pt", output_dir / "config.json"],
                    aliases=["latest", f"step-{step}"],
                )

    wandb_logger.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TGVF-v3 Stage1 with Qwen3-VL-Thinking.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Thinking")
    parser.add_argument("--processor-id", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=("auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"))
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--variant", choices=TGVF_VARIANTS, default="tgvf_v2_bidirectional")
    parser.add_argument("--num-foveated-tokens", type=_parse_optional_positive_int, default=None)
    parser.add_argument("--spatial-merge-size", default="auto")
    parser.add_argument("--attn-dim", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--capture-layer", type=int, default=-1)
    parser.add_argument("--loss-gen", type=float, default=1.0)
    parser.add_argument("--loss-visual-token-manifold", type=float, default=0.01)
    parser.add_argument("--loss-same-image-negative", type=float, default=1.0)
    parser.add_argument("--same-image-negative-margin", type=float, default=1.0)
    parser.add_argument("--same-image-negative-mode", choices=("cyclic_margin", "matrix_ce"), default="matrix_ce")
    parser.add_argument("--mask-original-image-after-tgvf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fvt-position-mode", choices=("native_source_grid", "inherit_source_visual_positions"), default="native_source_grid")
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--max-image-resolution", type=_parse_optional_positive_int, default=512)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--max-debug-examples-per-log", type=int, default=2)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-mode", default=None, choices=("online", "offline", "disabled"))
    parser.add_argument("--wandb-dir", default=None)
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument("--wandb-watch", default=None, choices=("gradients", "parameters", "all"))
    parser.add_argument("--wandb-log-checkpoints", action="store_true")
    args = parser.parse_args()
    if args.variant not in TGVF_DYNAMIC_NUM_FVT_VARIANTS and args.num_foveated_tokens is None:
        variants = ", ".join(TGVF_DYNAMIC_NUM_FVT_VARIANTS)
        parser.error(f"--num-foveated-tokens none is supported only with --variant in: {variants}")
    return args


def _parse_optional_positive_int(value: str) -> int | None:
    if value.lower() in {"none", "null", "auto"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive, none, null, or auto")
    return parsed


def _json_safe(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _wandb_train_metrics(log: dict) -> dict:
    metrics = {
        "train/step": log.get("step"),
        "train/loss_total": log.get("loss_total"),
        "train/loss_gen": log.get("loss_gen"),
        "train/loss_visual_token_manifold": log.get("loss_visual_token_manifold"),
        "train/loss_same_image_negative": log.get("loss_same_image_negative"),
        "train/grad_norm": log.get("grad_norm"),
        "train/peak_memory_gb": log.get("peak_memory_gb"),
        "train/finite_rate": log.get("finite_rate"),
        "train/source_visual_token_count": log.get("source_visual_token_count"),
        "train/answer_token_count": log.get("answer_token_count"),
        "train/qwen_frozen": log.get("qwen_frozen"),
        "train/second_full_forward_used": log.get("second_full_forward_used"),
        "train/image_keys_blocked_for_tgvf_evidence_answer": log.get(
            "image_keys_blocked_for_tgvf_evidence_answer"
        ),
        "train/pre_tgvf_queries_keep_original_image_keys": log.get(
            "pre_tgvf_queries_keep_original_image_keys"
        ),
        "train/visual_token_manifold_active": log.get("visual_token_manifold_active"),
        "train/target_hidden_tokens": _shape_dim(log.get("target_hidden_shape"), 0),
        "train/target_hidden_dim": _shape_dim(log.get("target_hidden_shape"), 1),
        "train/pre_merge_visual_tokens": _shape_dim(log.get("pre_merge_visual_shape"), 0),
        "train/pre_merge_visual_dim": _shape_dim(log.get("pre_merge_visual_shape"), 1),
        "train/foveated_visual_tokens": _shape_dim(log.get("foveated_visual_tokens_shape"), 0),
        "train/foveated_visual_dim": _shape_dim(log.get("foveated_visual_tokens_shape"), 1),
    }
    for key, value in flatten_metrics(
        {
            "attention": log.get("attention_diagnostics") or {},
            "norm": log.get("norm_diagnostics") or {},
        }
    ).items():
        metrics[f"train/{key}"] = value
    return metrics


def _shape_dim(shape, index: int):
    if isinstance(shape, (list, tuple)) and len(shape) > index:
        return shape[index]
    return None


if __name__ == "__main__":
    main()
