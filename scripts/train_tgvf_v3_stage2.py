#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from revisit_vlm.qwen3_vl_tgvf import load_qwen3_vl, peak_memory_gb
from revisit_vlm.tgvf_training import (
    TGVF_DYNAMIC_NUM_FVT_VARIANTS,
    TGVF_VARIANTS,
    TGVFModuleConfig,
    build_tgvf_module,
)
from revisit_vlm.tgvf_v3_stage1 import freeze_qwen_backbone, infer_qwen3_stage1_dims
from revisit_vlm.tgvf_v3_stage2 import (
    Stage2LossWeights,
    TGVFv3Stage2Dataset,
    dataset_stage2_stats,
    tgvf_v3_stage2_collate,
    v3_stage2_training_step,
)
from revisit_vlm.tgvf_v3_stage2_fast import v3_stage2_batched_training_step
from revisit_vlm.wandb_logging import WandbLogger


def main() -> None:
    args = parse_args()
    ddp = setup_distributed(args)
    rank = ddp["rank"]
    local_rank = ddp["local_rank"]
    world_size = ddp["world_size"]
    is_main = rank == 0
    device = ddp["device"]
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    distributed_barrier()

    train_dataset = TGVFv3Stage2Dataset(args.train_file, min_confidence=args.min_confidence)
    if len(train_dataset) == 0:
        raise RuntimeError("No v3 Stage2 samples were loaded")
    focus_sample = next((sample for sample in train_dataset.samples if sample.need_focus), None)
    if focus_sample is None:
        raise RuntimeError("Stage2 needs at least one focus sample to infer TGVF dimensions")
    val_dataset = (
        TGVFv3Stage2Dataset(args.val_file, min_confidence=args.min_confidence)
        if args.val_file
        else None
    )

    loaded = load_qwen3_vl(
        args.model_id,
        processor_id=args.processor_id,
        dtype=args.dtype,
        device_map=ddp["device_map"],
        attn_implementation=args.attn_implementation,
    )
    model = loaded.model
    processor = loaded.processor
    if getattr(processor.tokenizer, "pad_token", None) is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    tokenizer_size_before = len(processor.tokenizer)
    freeze_qwen_backbone(model)
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False
    lora_targets = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=lora_targets,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.train()
    utility_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    tokenizer_size_after = len(processor.tokenizer)

    dims = infer_qwen3_stage1_dims(
        model=utility_model,
        processor=processor,
        sample=focus_sample,  # type: ignore[arg-type]
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
    train_dtype = next(model.parameters()).dtype
    foveal_module = build_tgvf_module(
        variant=args.variant,
        d_lm=dims["d_lm"],
        d_v=dims["d_v"],
        num_foveated_tokens=args.num_foveated_tokens,
        spatial_merge_size=spatial_merge_size,
        attn_dim=args.attn_dim,
    ).to(device=device, dtype=train_dtype)
    stage1_checkpoint = torch.load(args.stage1_checkpoint, map_location="cpu")
    foveal_module.load_state_dict(stage1_checkpoint["tgvf_module"], strict=True)
    foveal_module.train()

    loss_weights = Stage2LossWeights(
        evidence_state=args.loss_evidence_state,
        focus_target=args.loss_focus_target,
        evidence=args.loss_evidence,
        value_span=args.loss_value_span,
        answer=args.loss_answer,
        no_focus_evidence_state=args.loss_no_focus_evidence_state,
        no_focus_answer=args.loss_no_focus_answer,
        visual_token_manifold=args.loss_visual_token_manifold,
    )
    optimizer, lr_groups = build_optimizer(
        model=model,
        foveal_module=foveal_module,
        args=args,
    )
    scheduler = build_lr_scheduler(
        optimizer,
        scheduler_name=args.lr_scheduler,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps
        if args.warmup_steps is not None
        else int(math.ceil(args.max_steps * args.warmup_ratio)),
        min_lr_ratio=args.min_lr_ratio,
    )

    train_stats = dataset_stage2_stats(train_dataset)
    val_stats = dataset_stage2_stats(val_dataset) if val_dataset is not None else None
    trainable_names = trainable_parameter_names(model, prefix="qwen_lora.") + trainable_parameter_names(
        foveal_module, prefix="tgvf."
    )
    trainable_count = sum(p.numel() for p in list(model.parameters()) + list(foveal_module.parameters()) if p.requires_grad)
    frozen_count = sum(p.numel() for p in list(model.parameters()) + list(foveal_module.parameters()) if not p.requires_grad)
    config = {
        "stage": "tgvf_v3_stage2_50k",
        "model_id": args.model_id,
        "processor_id": loaded.processor_id,
        "train_file": str(args.train_file),
        "val_file": str(args.val_file) if args.val_file else None,
        "train_dataset": train_stats,
        "val_dataset": val_stats,
        "stage1_checkpoint": args.stage1_checkpoint,
        "tgvf": asdict(module_config),
        "loss_weights": asdict(loss_weights),
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "bias": args.lora_bias,
            "target_modules": lora_targets,
        },
        "lr_groups": lr_groups,
        "optimizer": {
            "name": "adamw",
            "betas": [args.adam_beta1, args.adam_beta2],
            "eps": args.adam_eps,
            "weight_decay": args.weight_decay,
        },
        "scheduler": {
            "name": args.lr_scheduler,
            "warmup_steps": args.warmup_steps,
            "warmup_ratio": args.warmup_ratio,
            "min_lr_ratio": args.min_lr_ratio,
        },
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "world_size": world_size,
        "effective_global_batch_size": args.batch_size * args.gradient_accumulation_steps * world_size,
        "max_steps": args.max_steps,
        "save_every": args.save_every,
        "eval_every": args.eval_every,
        "max_image_resolution": args.max_image_resolution,
        "max_seq_len": args.max_seq_len,
        "mask_original_image_after_tgvf": args.mask_original_image_after_tgvf,
        "fvt_position_mode": args.fvt_position_mode,
        "capture_mode": "teacher_forced",
        "train_long_cot": False,
        "special_tokens_added": False,
        "tokenizer_resized": tokenizer_size_before != tokenizer_size_after,
        "markers_are_plain_text": True,
        "fast_batched_stage2": args.fast_batched_stage2,
        "matrix_ce_enabled": False,
        "same_image_negative_enabled": False,
        "contrastive_alignment_enabled": False,
        "tgvf_trainable": True,
        "qwen_base_frozen": True,
        "vision_encoder_frozen": True,
        "lm_head_frozen": True,
        "trainable_parameter_count": trainable_count,
        "frozen_parameter_count": frozen_count,
        "dtype": args.dtype,
        "device_map": ddp["device_map"],
        "attn_implementation": args.attn_implementation,
        "dims": dims,
    }
    if is_main:
        (output_dir / "config.json").write_text(json.dumps(_json_safe(config), indent=2, ensure_ascii=False) + "\n")
        (output_dir / "trainable_params.txt").write_text("\n".join(trainable_names) + "\n")
    wandb_logger = WandbLogger(
        project=args.wandb_project if is_main else None,
        entity=args.wandb_entity,
        name=args.wandb_run_name or output_dir.name,
        group=args.wandb_group,
        job_type="tgvf-v3-stage2-training",
        config=config,
        mode=args.wandb_mode,
        tags=[tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()] or None,
        directory=args.wandb_dir,
    )
    if is_main and wandb_logger.enabled:
        wandb_logger.log(_wandb_dataset_metrics(train_stats, val_stats), step=0)

    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
    ) if world_size > 1 else None
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=tgvf_v3_stage2_collate,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    data_iter = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    optimizer_step = 0
    running_loss = 0.0
    running_loss_focus = 0.0
    running_loss_no_focus = 0.0
    running_loss_visual_token_manifold = 0.0
    running_logs: list[dict[str, Any]] = []

    while optimizer_step < args.max_steps:
        if sampler is not None and micro_step % max(len(loader), 1) == 0:
            sampler.set_epoch(micro_step // max(len(loader), 1))
        try:
            samples = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            samples = next(data_iter)
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        step_fn = v3_stage2_batched_training_step if args.fast_batched_stage2 else v3_stage2_training_step
        output = step_fn(
            qwen_model=utility_model,
            qwen_forward_model=model,
            processor=processor,
            foveal_module=foveal_module,
            samples=samples,
            loss_weights=loss_weights,
            device=device,
            hidden_state_index=args.capture_layer,
            max_image_resolution=args.max_image_resolution,
            position_mode=args.fvt_position_mode,
            mask_original_image_after_tgvf=args.mask_original_image_after_tgvf,
        )
        if not torch.isfinite(output.loss_total):
            raise RuntimeError(f"Non-finite Stage2 loss at micro step {micro_step}: {output.loss_total}")
        (output.loss_total / args.gradient_accumulation_steps).backward()
        micro_step += 1
        running_loss += float(output.loss_total.detach().cpu())
        running_loss_focus += float(output.loss_focus.detach().cpu())
        running_loss_no_focus += float(output.loss_no_focus.detach().cpu())
        running_loss_visual_token_manifold += float(output.loss_visual_token_manifold.detach().cpu())
        running_logs.append(output.debug)
        if micro_step % args.gradient_accumulation_steps != 0:
            continue

        average_gradients([model, foveal_module], world_size=world_size)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in list(model.parameters()) + list(foveal_module.parameters()) if p.requires_grad],
            args.max_grad_norm,
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1

        if is_main and (optimizer_step == 1 or optimizer_step % args.log_every == 0 or optimizer_step == args.max_steps):
            log = {
                "step": optimizer_step,
                "micro_step": micro_step,
                "loss_total": running_loss / max(len(running_logs), 1),
                "loss_focus": running_loss_focus / max(len(running_logs), 1),
                "loss_no_focus": running_loss_no_focus / max(len(running_logs), 1),
                "loss_visual_token_manifold": running_loss_visual_token_manifold / max(len(running_logs), 1),
                "grad_norm": float(grad_norm.detach().cpu()),
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
                "peak_memory_gb": peak_memory_gb(),
                "world_size": world_size,
                "effective_global_batch_size": args.batch_size * args.gradient_accumulation_steps * world_size,
                **summarize_step_debug(running_logs),
            }
            print(json.dumps(_json_safe(log), indent=2, ensure_ascii=False))
            if wandb_logger.enabled:
                wandb_logger.log(_wandb_metrics(log), step=optimizer_step)
            running_loss = 0.0
            running_loss_focus = 0.0
            running_loss_no_focus = 0.0
            running_loss_visual_token_manifold = 0.0
            running_logs = []

        if is_main and (optimizer_step % args.save_every == 0 or optimizer_step == args.max_steps):
            save_stage2_checkpoint(
                output_dir / f"checkpoint_step_{optimizer_step}.pt",
                model=model,
                foveal_module=foveal_module,
                config=config,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=optimizer_step,
                micro_step=micro_step,
            )
            model.save_pretrained(output_dir / f"lora_adapter_step_{optimizer_step}")

        if (
            is_main
            and val_dataset is not None
            and args.eval_every > 0
            and (optimizer_step % args.eval_every == 0 or optimizer_step == args.max_steps)
        ):
            val_report = validate_stage2(
                utility_model=utility_model,
                model=model,
                processor=processor,
                foveal_module=foveal_module,
                dataset=val_dataset,
                loss_weights=loss_weights,
                args=args,
                device=device,
            )
            val_path = output_dir / f"val_step_{optimizer_step}.json"
            val_path.write_text(json.dumps(_json_safe(val_report), indent=2, ensure_ascii=False) + "\n")
            if wandb_logger.enabled:
                wandb_logger.log({f"val/{k}": v for k, v in val_report["metrics"].items()}, step=optimizer_step)
        distributed_barrier()

    if is_main:
        wandb_logger.finish()
    cleanup_distributed()


def validate_stage2(
    *,
    utility_model: Any,
    model: Any,
    processor: Any,
    foveal_module: torch.nn.Module,
    dataset: TGVFv3Stage2Dataset,
    loss_weights: Stage2LossWeights,
    args: argparse.Namespace,
    device: torch.device | str,
) -> dict[str, Any]:
    model.eval()
    foveal_module.eval()
    losses = []
    focus = 0
    no_focus = 0
    value_rates = []
    mask_focus = []
    mask_no_focus = []
    with torch.no_grad():
        step_fn = v3_stage2_batched_training_step if args.fast_batched_stage2 else v3_stage2_training_step
        for sample in dataset.samples[: args.eval_max_samples]:
            output = step_fn(
                qwen_model=utility_model,
                qwen_forward_model=model,
                processor=processor,
                foveal_module=foveal_module,
                samples=[sample],
                loss_weights=loss_weights,
                device=device,
                hidden_state_index=args.capture_layer,
                max_image_resolution=args.max_image_resolution,
                position_mode=args.fvt_position_mode,
                mask_original_image_after_tgvf=args.mask_original_image_after_tgvf,
            )
            losses.append(float(output.loss_total.detach().cpu()))
            focus += output.debug["focus_count"]
            no_focus += output.debug["no_focus_count"]
            if output.debug["value_span_match_rate"] is not None:
                value_rates.append(float(output.debug["value_span_match_rate"]))
            mask_focus.append(float(output.debug["focus_sample_mask_active_rate"]))
            mask_no_focus.append(float(output.debug["no_focus_mask_active_rate"]))
    model.train()
    foveal_module.train()
    return {
        "num_samples": len(losses),
        "focus": focus,
        "no_focus": no_focus,
        "metrics": {
            "loss": sum(losses) / max(len(losses), 1),
            "focus_ratio": focus / max(focus + no_focus, 1),
            "no_focus_ratio": no_focus / max(focus + no_focus, 1),
            "focus_sample_mask_active_rate": sum(mask_focus) / max(len(mask_focus), 1),
            "no_focus_mask_active_rate": sum(mask_no_focus) / max(len(mask_no_focus), 1),
            "value_span_match_rate": sum(value_rates) / max(len(value_rates), 1) if value_rates else None,
            "matrix_ce_enabled": False,
        },
    }


def build_optimizer(*, model: Any, foveal_module: torch.nn.Module, args: argparse.Namespace):
    lora_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    tgvf_refiner = []
    tgvf_calibration = []
    for name, param in foveal_module.named_parameters():
        if not param.requires_grad:
            continue
        if "calib" in name or "calibration" in name:
            tgvf_calibration.append((name, param))
        else:
            tgvf_refiner.append((name, param))
    groups = []
    lr_groups = {}
    if lora_params:
        groups.append({"params": [p for _, p in lora_params], "lr": args.lr_lora, "name": "llm_lora"})
        lr_groups["llm_lora"] = args.lr_lora
    if tgvf_refiner:
        groups.append({"params": [p for _, p in tgvf_refiner], "lr": args.lr_tgvf, "name": "tgvf_refiner"})
        lr_groups["tgvf_refiner"] = args.lr_tgvf
    if tgvf_calibration:
        groups.append({"params": [p for _, p in tgvf_calibration], "lr": args.lr_calibration, "name": "fvt_calibration"})
        lr_groups["fvt_calibration"] = args.lr_calibration
    optimizer = torch.optim.AdamW(
        groups,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    return optimizer, lr_groups


def save_stage2_checkpoint(
    path: Path,
    *,
    model: Any,
    foveal_module: torch.nn.Module,
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    global_step: int,
    micro_step: int,
) -> None:
    checkpoint = {
        "qwen_lora": get_peft_model_state_dict(model),
        "tgvf_module": foveal_module.state_dict(),
        "config": config,
        "global_step": global_step,
        "micro_step": micro_step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(checkpoint, path)


def average_gradients(modules: list[torch.nn.Module], *, world_size: int) -> None:
    if world_size <= 1 or not dist.is_available() or not dist.is_initialized():
        return
    for module in modules:
        for param in module.parameters():
            if not param.requires_grad:
                continue
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)


def trainable_parameter_names(module: torch.nn.Module, *, prefix: str = "") -> list[str]:
    return [prefix + name for name, param in module.named_parameters() if param.requires_grad]


def summarize_step_debug(debug_logs: list[dict[str, Any]]) -> dict[str, Any]:
    focus = sum(int(item.get("focus_count", 0)) for item in debug_logs)
    no_focus = sum(int(item.get("no_focus_count", 0)) for item in debug_logs)
    value_rates = [item.get("value_span_match_rate") for item in debug_logs if item.get("value_span_match_rate") is not None]
    examples = []
    for item in debug_logs:
        examples.extend(item.get("debug_examples", []))
        if len(examples) >= 2:
            break
    return {
        "focus_count": focus,
        "no_focus_count": no_focus,
        "focus_ratio": focus / max(focus + no_focus, 1),
        "no_focus_ratio": no_focus / max(focus + no_focus, 1),
        "focus_sample_mask_active_rate": 1.0 if focus else 0.0,
        "no_focus_mask_active_rate": 0.0,
        "value_span_match_rate": sum(value_rates) / max(len(value_rates), 1) if value_rates else None,
        "special_tokens_added": False,
        "tokenizer_resized": False,
        "markers_are_plain_text": True,
        "matrix_ce_enabled": False,
        "same_image_negative_enabled": False,
        "debug_examples": examples[:2],
    }


def setup_distributed(args: argparse.Namespace) -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            device_map: str | None = f"cuda:{local_rank}"
        else:
            device = torch.device("cpu")
            device_map = None
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    else:
        device = torch.device(args.device)
        device_map = args.device_map
    return {"rank": rank, "local_rank": local_rank, "world_size": world_size, "device": device, "device_map": device_map}


def distributed_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_name: str,
    max_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(0, int(warmup_steps))
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(float(step + 1) / float(warmup_steps), 1e-8)
        if scheduler_name == "constant":
            return 1.0
        decay_steps = max(1, int(max_steps) - warmup_steps)
        progress = min(1.0, max(0.0, float(step - warmup_steps + 1) / float(decay_steps)))
        if scheduler_name == "linear":
            return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)
        if scheduler_name == "cosine":
            cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
        raise ValueError(f"Unsupported lr scheduler: {scheduler_name}")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TGVF-v3 Stage2 trajectory LoRA + TGVF.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Thinking")
    parser.add_argument("--processor-id", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=("auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"))
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--variant", choices=TGVF_VARIANTS, default="tgvf_v2_bidirectional")
    parser.add_argument("--num-foveated-tokens", type=_parse_optional_positive_int, default=None)
    parser.add_argument("--spatial-merge-size", default="auto")
    parser.add_argument("--attn-dim", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=256)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-bias", default="none", choices=("none", "all", "lora_only"))
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--lr-lora", type=float, default=2e-5)
    parser.add_argument("--lr-tgvf", type=float, default=5e-6)
    parser.add_argument("--lr-calibration", type=float, default=1e-5)
    parser.add_argument("--lr-scheduler", choices=("constant", "linear", "cosine"), default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--save-every", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--eval-max-samples", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--capture-layer", type=int, default=-1)
    parser.add_argument("--max-image-resolution", type=_parse_optional_positive_int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--fvt-position-mode", choices=("native_source_grid", "inherit_source_visual_positions"), default="native_source_grid")
    parser.add_argument("--mask-original-image-after-tgvf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast-batched-stage2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--loss-evidence-state", type=float, default=0.2)
    parser.add_argument("--loss-focus-target", type=float, default=1.5)
    parser.add_argument("--loss-evidence", type=float, default=1.0)
    parser.add_argument("--loss-value-span", type=float, default=3.0)
    parser.add_argument("--loss-answer", type=float, default=1.0)
    parser.add_argument("--loss-no-focus-evidence-state", type=float, default=0.2)
    parser.add_argument("--loss-no-focus-answer", type=float, default=1.0)
    parser.add_argument("--loss-visual-token-manifold", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-mode", default=None, choices=("online", "offline", "disabled"))
    parser.add_argument("--wandb-dir", default=None)
    parser.add_argument("--wandb-tags", default="")
    args = parser.parse_args()
    if args.variant not in TGVF_DYNAMIC_NUM_FVT_VARIANTS and args.num_foveated_tokens is None:
        parser.error("--num-foveated-tokens none is supported only for dynamic variants")
    return args


def _parse_optional_positive_int(value: str) -> int | None:
    if value.lower() in {"none", "null", "auto"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive, none, null, or auto")
    return parsed


def _json_safe(value: Any):
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


def _wandb_metrics(log: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        "train/loss_total": log.get("loss_total"),
        "train/loss_focus": log.get("loss_focus"),
        "train/loss_no_focus": log.get("loss_no_focus"),
        "train/loss_visual_token_manifold": log.get("loss_visual_token_manifold"),
        "train/grad_norm": log.get("grad_norm"),
        "train/peak_memory_gb": log.get("peak_memory_gb"),
        "train/focus_count": log.get("focus_count"),
        "train/no_focus_count": log.get("no_focus_count"),
        "train/focus_ratio": log.get("focus_ratio"),
        "train/no_focus_ratio": log.get("no_focus_ratio"),
        "train/focus_sample_mask_active_rate": log.get("focus_sample_mask_active_rate"),
        "train/no_focus_mask_active_rate": log.get("no_focus_mask_active_rate"),
        "train/value_span_match_rate": log.get("value_span_match_rate"),
        "train/focus_loss_token_weight": log.get("focus_loss_token_weight"),
        "train/no_focus_loss_token_weight": log.get("no_focus_loss_token_weight"),
        "train/effective_global_batch_size": log.get("effective_global_batch_size"),
        "train/world_size": log.get("world_size"),
        "train/special_tokens_added": float(bool(log.get("special_tokens_added"))),
        "train/tokenizer_resized": float(bool(log.get("tokenizer_resized"))),
        "train/markers_are_plain_text": float(bool(log.get("markers_are_plain_text"))),
        "train/matrix_ce_enabled": float(bool(log.get("matrix_ce_enabled"))),
        "train/same_image_negative_enabled": float(bool(log.get("same_image_negative_enabled"))),
    }
    for index, lr in enumerate(log.get("learning_rates") or []):
        metrics[f"train/lr_group_{index}"] = lr
    return metrics


def _wandb_dataset_metrics(train_stats: dict[str, Any], val_stats: dict[str, Any] | None) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for prefix, stats in (("train_dataset", train_stats), ("val_dataset", val_stats or {})):
        for key, value in stats.items():
            if isinstance(value, bool):
                metrics[f"{prefix}/{key}"] = float(value)
            elif isinstance(value, (int, float)):
                metrics[f"{prefix}/{key}"] = value
    return metrics


if __name__ == "__main__":
    main()
