from __future__ import annotations

import argparse
import json
import os
import random
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from revisit_vlm.lr_schedulers import (
    build_lr_scheduler,
    current_learning_rate,
    scheduler_config_from_args,
)
from revisit_vlm.models.qwen2vl import load_qwen2vl
from revisit_vlm.tgvf_training import (
    TGVF_DYNAMIC_NUM_FVT_VARIANTS,
    TGVF_VARIANTS,
    LossWeights,
    TeacherGuideDataset,
    TGVFModuleConfig,
    TGVFTrainingConfig,
    build_tgvf_module,
    freeze_qwen2vl,
    infer_tgvf_dims,
    save_tgvf_checkpoint,
    teacher_guide_collate,
    training_step,
)
from revisit_vlm.wandb_logging import WandbLogger, flatten_metrics


def main() -> None:
    args = parse_args()
    ddp = _init_distributed(args)
    rank = _rank()
    world_size = _world_size()
    local_rank = _local_rank()
    is_main = rank == 0
    device = _resolve_device(args.device, local_rank)

    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    if ddp:
        _distributed_barrier()

    loaded = load_qwen2vl(
        args.model_name_or_path,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        device_map={"": device} if str(device).startswith("cuda") else None,
        trust_remote_code=False,
    )
    model = loaded.model
    processor = loaded.processor
    freeze_qwen2vl(model)

    d_lm, d_v, inferred_merge_size = infer_tgvf_dims(model)
    spatial_merge_size = (
        inferred_merge_size
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
        contrastive_alignment=args.loss_contrastive_alignment,
    )
    scheduler_info = scheduler_config_from_args(args)
    config = TGVFTrainingConfig(
        model_name_or_path=args.model_name_or_path,
        tgvf=module_config,
        loss_weights=loss_weights,
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=scheduler_info["warmup_steps"],
        min_lr_ratio=args.min_lr_ratio,
        num_training_steps=args.num_training_steps,
        estimated_optimizer_steps=scheduler_info["estimated_optimizer_steps"],
        max_steps_semantics=scheduler_info["max_steps_semantics"],
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        save_every=args.save_every,
        capture_layer=args.capture_layer,
        qwen_freeze=True,
        contrastive_temperature=args.contrastive_temperature,
        same_image_negative_margin=args.same_image_negative_margin,
        same_image_negative_mode=args.same_image_negative_mode,
        readout_prompt_target_dropout=args.readout_prompt_target_dropout,
        readout_append_mode="qwen_native_pseudo_image",
    )
    if is_main:
        (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))

    wandb_logger = WandbLogger(
        project=args.wandb_project if is_main else None,
        entity=args.wandb_entity,
        name=args.wandb_run_name or output_dir.name,
        group=args.wandb_group,
        job_type="tgvf-fvt-training",
        config={
            "training": asdict(config),
            "train_file": args.train_file,
            "output_dir": str(output_dir),
            "dataset_size": len(TeacherGuideDataset(args.train_file, warn_on_leakage=False)),
            "distributed": {
                "enabled": ddp,
                "world_size": world_size,
            },
            "group_batches_by_image": args.group_batches_by_image,
            "streaming_matrix_ce_backward": args.streaming_matrix_ce_backward,
            "lr_scheduler": scheduler_info,
        },
        mode=args.wandb_mode,
        tags=[tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()] or None,
        directory=args.wandb_dir,
    )

    train_dtype = next(model.parameters()).dtype
    foveal_module = build_tgvf_module(
        variant=args.variant,
        d_lm=d_lm,
        d_v=d_v,
        num_foveated_tokens=args.num_foveated_tokens,
        spatial_merge_size=spatial_merge_size,
        attn_dim=args.attn_dim,
    ).to(device=device, dtype=train_dtype)
    foveal_module.train()
    if ddp:
        foveal_module = DistributedDataParallel(
            foveal_module,
            device_ids=[local_rank] if str(device).startswith("cuda") else None,
        )
    if args.wandb_watch and wandb_logger.enabled:
        wandb_logger._wandb.watch(
            foveal_module,
            log=args.wandb_watch,
            log_freq=max(args.log_every, 1),
        )
    optimizer = torch.optim.AdamW(foveal_module.parameters(), lr=args.learning_rate)
    scheduler = build_lr_scheduler(
        optimizer,
        lr_scheduler=args.lr_scheduler,
        num_training_steps=scheduler_info["estimated_optimizer_steps"],
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
    )
    start_batch_step = 1
    optimizer_step = 0
    if args.resume_from_checkpoint:
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        _unwrap_ddp(foveal_module).load_state_dict(checkpoint["tgvf_module"], strict=True)
        if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_batch_step = int(checkpoint.get("global_step", 0)) + 1
        optimizer_step = int(checkpoint.get("optimizer_step", 0))

    if is_main:
        startup_log = {
            "event": "training_start",
            **scheduler_info,
            "current_learning_rate": current_learning_rate(optimizer),
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "start_batch_step": start_batch_step,
            "optimizer_step": optimizer_step,
        }
        print(json.dumps(startup_log, indent=2))
        if wandb_logger.enabled:
            wandb_logger.log(_wandb_startup_metrics(startup_log), step=0)

    dataset = TeacherGuideDataset(args.train_file, warn_on_leakage=False)
    if args.group_batches_by_image and ddp:
        raise ValueError("--group-batches-by-image is currently supported only for non-DDP single-process runs")
    if args.group_batches_by_image:
        batch_sampler = SameImageBatchSampler(
            dataset,
            batch_size=args.batch_size,
            seed=args.seed,
            drop_singletons=True,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=teacher_guide_collate,
        )
        sampler = batch_sampler
    else:
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=False,
            )
            if ddp
            else None
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            collate_fn=teacher_guide_collate,
        )
    data_iter = iter(loader)
    epoch = 0
    optimizer.zero_grad(set_to_none=True)

    for step in range(start_batch_step, args.max_steps + 1):
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        try:
            samples = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            data_iter = iter(loader)
            samples = next(data_iter)

        should_step = step % args.gradient_accumulation_steps == 0 or step == args.max_steps
        sync_context = foveal_module.no_sync() if ddp and not should_step else nullcontext()
        with sync_context:
            activation_context = (
                torch.autograd.graph.save_on_cpu(pin_memory=True)
                if args.save_activations_on_cpu and str(device).startswith("cuda")
                else nullcontext()
            )
            with activation_context:
                output = training_step(
                    qwen_model=model,
                    processor=processor,
                    foveal_module=foveal_module,
                    samples=samples,
                    loss_weights=loss_weights,
                    device=device,
                    hidden_state_index=args.capture_layer,
                    contrastive_temperature=args.contrastive_temperature,
                    same_image_negative_margin=args.same_image_negative_margin,
                    same_image_negative_mode=args.same_image_negative_mode,
                    readout_prompt_target_dropout=args.readout_prompt_target_dropout,
                    backward_loss_scale=(
                        1.0 / args.gradient_accumulation_steps
                        if args.streaming_matrix_ce_backward
                        else None
                    ),
                )
                if not output.debug.get("backward_performed", False):
                    (output.loss_total / args.gradient_accumulation_steps).backward()
        if should_step:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                foveal_module.parameters(), args.max_grad_norm
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer_step += 1
            optimizer.zero_grad(set_to_none=True)
        else:
            grad_norm = torch.tensor(0.0)

        if args.progress and is_main:
            _print_progress_bar(step, args.max_steps, optimizer_step=optimizer_step)

        if step == 1 or step % args.log_every == 0 or step == args.max_steps:
            log = {
                "step": step,
                "loss_total": float(output.loss_total.detach().cpu()),
                "loss_gen": float(output.loss_gen.detach().cpu()),
                "loss_visual_token_manifold": float(
                    output.loss_visual_token_manifold.detach().cpu()
                ),
                "loss_same_image_negative": float(output.loss_same_image_negative.detach().cpu()),
                "loss_contrastive_alignment": float(
                    output.loss_contrastive_alignment.detach().cpu()
                ),
                "grad_norm": float(grad_norm.detach().cpu()),
                "learning_rate": current_learning_rate(optimizer),
                "base_learning_rate": args.learning_rate,
                "optimizer_step": optimizer_step,
                "batch_step": step,
                "lr_scheduler": args.lr_scheduler,
                "warmup_steps": scheduler_info["warmup_steps"],
                "estimated_optimizer_steps": scheduler_info["estimated_optimizer_steps"],
                "min_learning_rate": scheduler_info["min_learning_rate"],
                "max_steps_semantics": scheduler_info["max_steps_semantics"],
                "variant": args.variant,
                "distributed": ddp,
                "rank": rank,
                "world_size": world_size,
                "device": str(device),
                "batch_size_observed": len(samples),
                "group_batches_by_image": args.group_batches_by_image,
                "same_image_negative_mode": args.same_image_negative_mode,
                "readout_prompt_target_dropout": args.readout_prompt_target_dropout,
                "save_activations_on_cpu": args.save_activations_on_cpu,
                "empty_cache_every_step": args.empty_cache_every_step,
                "streaming_matrix_ce_backward": args.streaming_matrix_ce_backward,
                **output.debug,
                **_cuda_memory_log(device),
            }
            log = _average_distributed_log(log, device=device) if ddp else log
            if is_main:
                print(json.dumps(_compact_log_for_print(log), indent=2))
                if wandb_logger.enabled:
                    wandb_logger.log(_wandb_train_metrics(log), step=step)

        if is_main and (step % args.save_every == 0 or step == args.max_steps):
            checkpoint_path = output_dir / f"checkpoint_step_{step}.pt"
            save_tgvf_checkpoint(
                path=checkpoint_path,
                foveal_module=_unwrap_ddp(foveal_module),
                config=config,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=step,
                optimizer_step=optimizer_step,
            )
            if args.wandb_log_checkpoints and wandb_logger.enabled:
                wandb_logger.log_artifact(
                    name=f"{output_dir.name}-checkpoint-{step}",
                    artifact_type="tgvf-fvt-checkpoint",
                    paths=[checkpoint_path, output_dir / "config.json"],
                    aliases=["latest", f"step-{step}"],
                )

        del output, grad_norm
        if args.empty_cache_every_step and str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    if args.progress and is_main:
        print(file=sys.stderr)

    wandb_logger.finish()
    if ddp:
        _distributed_barrier()
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TGVF FVT modules from teacher-guide JSONL.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument(
        "--variant",
        choices=TGVF_VARIANTS,
        default="foveal_cross_merger",
    )
    parser.add_argument("--num-foveated-tokens", type=_parse_optional_positive_int, default=16)
    parser.add_argument("--spatial-merge-size", default="auto")
    parser.add_argument("--attn-dim", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", default="none", choices=("none", "warmup_cosine"))
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--num-training-steps", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print a compact training progress bar on the main process.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--save-activations-on-cpu",
        action="store_true",
        help="Offload autograd saved tensors to CPU during the train step to reduce CUDA peak memory.",
    )
    parser.add_argument(
        "--empty-cache-every-step",
        action="store_true",
        help="Call torch.cuda.empty_cache() after each train step to reduce CUDA reserved-memory growth.",
    )
    parser.add_argument(
        "--streaming-matrix-ce-backward",
        action="store_true",
        help=(
            "Compute readout gradients one score at a time for supported negative losses "
            "to avoid retaining all Qwen readout graphs."
        ),
    )
    parser.add_argument("--capture-layer", type=int, default=-1)
    parser.add_argument("--loss-gen", type=float, default=1.0)
    parser.add_argument("--loss-visual-token-manifold", type=float, default=0.01)
    parser.add_argument("--loss-same-image-negative", type=float, default=0.0)
    parser.add_argument("--loss-contrastive-alignment", type=float, default=0.0)
    parser.add_argument("--contrastive-temperature", type=float, default=0.07)
    parser.add_argument(
        "--readout-prompt-target-dropout",
        type=float,
        default=0.3,
        help="Probability of omitting Target from the readout prompt during training.",
    )
    parser.add_argument("--same-image-negative-margin", type=float, default=1.0)
    parser.add_argument(
        "--same-image-negative-mode",
        choices=("cyclic_margin", "matrix_ce"),
        default="cyclic_margin",
        help="Same-image negative objective: cyclic margin pair loss or full in-image matrix CE.",
    )
    parser.add_argument(
        "--group-batches-by-image",
        action="store_true",
        help="Batch samples by image_id so L_same_image_negative has same-image wrong-D pairs.",
    )
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-mode", default=None, choices=("online", "offline", "disabled"))
    parser.add_argument("--wandb-dir", default=None)
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument("--wandb-watch", default=None, choices=("gradients", "parameters", "all"))
    parser.add_argument("--wandb-log-checkpoints", action="store_true")
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Enable DistributedDataParallel. torchrun also enables this automatically.",
    )
    parser.add_argument("--ddp-backend", default="nccl")
    args = parser.parse_args()
    if args.variant not in TGVF_DYNAMIC_NUM_FVT_VARIANTS and args.num_foveated_tokens is None:
        variants = ", ".join(TGVF_DYNAMIC_NUM_FVT_VARIANTS)
        parser.error(f"--num-foveated-tokens none is supported only with --variant in: {variants}")
    if not 0.0 <= args.readout_prompt_target_dropout <= 1.0:
        parser.error("--readout-prompt-target-dropout must be in [0, 1]")
    if args.streaming_matrix_ce_backward and args.same_image_negative_mode not in {
        "matrix_ce",
        "cyclic_margin",
    }:
        parser.error(
            "--streaming-matrix-ce-backward requires --same-image-negative-mode matrix_ce or cyclic_margin"
        )
    return args


def _parse_optional_positive_int(value: str) -> int | None:
    if value.lower() in {"none", "null"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer or 'none'")
    return parsed


def _print_progress_bar(
    step: int,
    max_steps: int,
    *,
    optimizer_step: int,
    width: int = 32,
) -> None:
    ratio = min(max(step / max(max_steps, 1), 0.0), 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"\rtrain [{bar}] {step}/{max_steps} ({ratio * 100:5.1f}%) opt={optimizer_step}",
        end="",
        file=sys.stderr,
        flush=True,
    )


def _cuda_memory_log(device: torch.device | str) -> dict[str, float]:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return {}
    dev = torch.device(device)
    denom = 1024**3
    return {
        "cuda_memory_allocated_gb": torch.cuda.memory_allocated(dev) / denom,
        "cuda_memory_reserved_gb": torch.cuda.memory_reserved(dev) / denom,
        "cuda_max_memory_allocated_gb": torch.cuda.max_memory_allocated(dev) / denom,
        "cuda_max_memory_reserved_gb": torch.cuda.max_memory_reserved(dev) / denom,
    }


def _wandb_train_metrics(log: dict[str, object]) -> dict[str, object]:
    metrics = {
        "train/loss_total": log["loss_total"],
        "train/loss_gen": log["loss_gen"],
        "train/loss_visual_token_manifold": log["loss_visual_token_manifold"],
        "train/loss_same_image_negative": log["loss_same_image_negative"],
        "train/loss_contrastive_alignment": log["loss_contrastive_alignment"],
        "train/grad_norm": log["grad_norm"],
        "train/learning_rate": log["learning_rate"],
        "train/base_learning_rate": log.get("base_learning_rate"),
        "train/optimizer_step": log.get("optimizer_step"),
        "train/batch_step": log.get("batch_step"),
        "train/lr_scheduler": log.get("lr_scheduler"),
        "train/qwen_frozen": log.get("qwen_frozen"),
        "train/second_full_forward_used": log.get("second_full_forward_used"),
        "train/variant": log.get("variant"),
        "train/distributed": log.get("distributed"),
        "train/world_size": log.get("world_size"),
        "train/batch_size_observed": log.get("batch_size_observed"),
        "train/group_batches_by_image": log.get("group_batches_by_image"),
        "train/same_image_negative_mode": log.get("same_image_negative_mode"),
        "train/readout_prompt_target_dropout": log.get("readout_prompt_target_dropout"),
        "train/save_activations_on_cpu": log.get("save_activations_on_cpu"),
        "train/empty_cache_every_step": log.get("empty_cache_every_step"),
        "train/cuda_memory_allocated_gb": log.get("cuda_memory_allocated_gb"),
        "train/cuda_memory_reserved_gb": log.get("cuda_memory_reserved_gb"),
        "train/cuda_max_memory_allocated_gb": log.get("cuda_max_memory_allocated_gb"),
        "train/cuda_max_memory_reserved_gb": log.get("cuda_max_memory_reserved_gb"),
        "train/readout_append_mode": log.get("readout_append_mode"),
        "train/readout_expected_llm_image_tokens": log.get("readout_expected_llm_image_tokens"),
        "train/readout_actual_image_pad_token_count": log.get("readout_actual_image_pad_token_count"),
        "train/readout_mm_token_type_ids_present": log.get("readout_mm_token_type_ids_present"),
        "train/readout_image_pad_mm_type_is_image": log.get("readout_image_pad_mm_type_is_image"),
        "train/readout_image_position_ids_are_3d": log.get("readout_image_position_ids_are_3d"),
        "train/readout_text_position_ids_are_1d": log.get("readout_text_position_ids_are_1d"),
        "train/readout_visual_tower_called_for_fvt": log.get("readout_visual_tower_called_for_fvt"),
        "train/readout_prompt_target_dropout_observed": log.get(
            "readout_prompt_target_dropout_observed"
        ),
    }
    for key in (
        "target_hidden_shape",
        "pre_merge_visual_shape",
        "merged_visual_shape",
        "foveated_visual_tokens_shape",
    ):
        value = log.get(key)
        if isinstance(value, list):
            for index, dim in enumerate(value):
                metrics[f"train/{key}_{index}"] = dim
    metrics.update(flatten_metrics({"debug": log.get("loss_weights", {})}, prefix="train"))
    metrics.update(
        _diagnostic_scalar_metrics(
            "train/diagnostics/attention",
            log.get("attention_diagnostics"),
        )
    )
    metrics.update(
        _diagnostic_scalar_metrics(
            "train/diagnostics/norm",
            log.get("norm_diagnostics"),
        )
    )
    _add_diagnostic_histograms(
        metrics,
        "train/diagnostics/attention",
        log.get("attention_diagnostics"),
    )
    _add_diagnostic_histograms(
        metrics,
        "train/diagnostics/norm",
        log.get("norm_diagnostics"),
    )
    return metrics


def _diagnostic_scalar_metrics(prefix: str, values: object) -> dict[str, object]:
    if not isinstance(values, dict):
        return {}
    return {
        f"{prefix}/{key}": value
        for key, value in values.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _add_diagnostic_histograms(
    metrics: dict[str, object],
    prefix: str,
    values: object,
) -> None:
    if not isinstance(values, dict):
        return
    try:
        import wandb
    except ModuleNotFoundError:
        return
    for key, value in values.items():
        if not key.endswith("_values") or not isinstance(value, list) or not value:
            continue
        histogram_name = key.removesuffix("_values") + "_hist"
        metrics[f"{prefix}/{histogram_name}"] = wandb.Histogram(value)


def _compact_log_for_print(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _compact_log_for_print(item)
            for key, item in value.items()
            if not key.endswith("_values")
        }
    if isinstance(value, list):
        return [_compact_log_for_print(item) for item in value]
    return value


def _wandb_startup_metrics(log: dict[str, object]) -> dict[str, object]:
    return {
        "train/lr_scheduler": log.get("lr_scheduler"),
        "train/base_learning_rate": log.get("learning_rate"),
        "train/learning_rate": log.get("current_learning_rate"),
        "train/warmup_ratio": log.get("warmup_ratio"),
        "train/warmup_steps": log.get("warmup_steps"),
        "train/requested_warmup_steps": log.get("requested_warmup_steps"),
        "train/min_lr_ratio": log.get("min_lr_ratio"),
        "train/min_learning_rate": log.get("min_learning_rate"),
        "train/estimated_optimizer_steps": log.get("estimated_optimizer_steps"),
        "train/gradient_accumulation_steps": log.get("gradient_accumulation_steps"),
        "train/max_steps": log.get("max_steps"),
        "train/max_steps_semantics": log.get("max_steps_semantics"),
        "train/optimizer_step": log.get("optimizer_step"),
        "train/start_batch_step": log.get("start_batch_step"),
    }


class SameImageBatchSampler:
    """Yield batches whose records share image_id/stable uid for same-image negatives."""

    def __init__(
        self,
        dataset: TeacherGuideDataset,
        *,
        batch_size: int,
        seed: int,
        drop_singletons: bool = True,
    ) -> None:
        if batch_size < 2:
            raise ValueError("--group-batches-by-image requires --batch-size >= 2")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.drop_singletons = drop_singletons
        self.epoch = 0
        self.groups: list[list[int]] = []
        by_image: dict[str, list[int]] = {}
        for index, sample in enumerate(dataset.samples):
            key = sample.image_id or sample.image
            by_image.setdefault(key, []).append(index)
        for indices in by_image.values():
            if drop_singletons and len(indices) < 2:
                continue
            self.groups.append(indices)
        if not self.groups:
            raise ValueError("No same-image groups with at least two samples were found")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        groups = [list(indices) for indices in self.groups]
        rng.shuffle(groups)
        for indices in groups:
            rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if self.drop_singletons and len(batch) < 2:
                    continue
                yield batch

    def __len__(self) -> int:
        total = 0
        for indices in self.groups:
            full, remainder = divmod(len(indices), self.batch_size)
            total += full
            if remainder >= 2 or (remainder and not self.drop_singletons):
                total += 1
        return total

def _init_distributed(args: argparse.Namespace) -> bool:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    enabled = args.ddp or world_size > 1
    if not enabled:
        return False
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")
    if not dist.is_initialized():
        dist.init_process_group(backend=args.ddp_backend)
    local_rank = _local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _resolve_device(requested_device: str, local_rank: int) -> str:
    if _world_size() > 1 or requested_device == "auto":
        return f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    return requested_device


def _unwrap_ddp(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if isinstance(module, DistributedDataParallel) else module


def _distributed_barrier() -> None:
    if torch.cuda.is_available():
        dist.barrier(device_ids=[_local_rank()])
    else:
        dist.barrier()


def _average_distributed_log(log: dict[str, object], *, device: torch.device | str) -> dict[str, object]:
    averaged = dict(log)
    numeric_keys = [
        key
        for key, value in log.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and key not in {
            "step",
            "rank",
            "world_size",
            "learning_rate",
            "base_learning_rate",
            "optimizer_step",
            "batch_step",
            "batch_size_observed",
            "warmup_steps",
            "estimated_optimizer_steps",
            "min_learning_rate",
        }
    ]
    if numeric_keys:
        values = torch.tensor([float(log[key]) for key in numeric_keys], device=device)
        dist.all_reduce(values, op=dist.ReduceOp.AVG)
        for key, value in zip(numeric_keys, values.tolist(), strict=True):
            averaged[key] = value
    averaged["rank"] = 0
    averaged["world_size"] = _world_size()
    return averaged


if __name__ == "__main__":
    main()
