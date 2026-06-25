#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from revisit_vlm.qwen3_vl_tgvf import (
    PROTOCOL_C_THINKING_SPECIAL,
    PROTOCOL_C_TOOL_OBSERVATION,
    PROTOCOL_C_TOOL_OBSERVATION_QWEN2_NO_THINK,
    PROTOCOL_E_ACTION_EVIDENCE_SPECIAL,
    TGVF_PROTOCOL_CHOICES,
    ensure_tgvf_protocol_tokens,
    load_qwen3_vl,
    peak_memory_gb,
    protocol_special_token_ids,
    protocol_special_tokens,
)
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
    ddp = setup_distributed(args)
    rank = ddp["rank"]
    local_rank = ddp["local_rank"]
    world_size = ddp["world_size"]
    is_main = rank == 0
    device = ddp["device"]
    effective_device_map = ddp["device_map"]
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    distributed_barrier()

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
        device_map=effective_device_map,
        attn_implementation=args.attn_implementation,
    )
    model = loaded.model
    processor = loaded.processor
    if getattr(processor.tokenizer, "pad_token", None) is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    protocol_token_info: dict[str, Any] = {}
    token_row_protocols = {
        PROTOCOL_C_THINKING_SPECIAL,
        PROTOCOL_C_TOOL_OBSERVATION,
        PROTOCOL_C_TOOL_OBSERVATION_QWEN2_NO_THINK,
        PROTOCOL_E_ACTION_EVIDENCE_SPECIAL,
    }
    if args.tgvf_protocol in token_row_protocols:
        protocol_token_info = ensure_tgvf_protocol_tokens(
            processor.tokenizer,
            model,
            protocol=args.tgvf_protocol,
        )
    freeze_qwen_backbone(model)
    protocol_c_token_train_info: dict[str, Any] = {}
    protocol_c_token_params: list[torch.nn.Parameter] = []
    if args.tgvf_protocol in token_row_protocols:
        protocol_c_token_train_info, protocol_c_token_params = _enable_protocol_c_token_row_training(
            model=model,
            tokenizer=processor.tokenizer,
            protocol=args.tgvf_protocol,
            mode=args.protocol_token_row_mode,
        )

    reencode_model = None
    reencode_train_info: dict[str, Any] = {"enabled": False}
    reencode_trainable_params: list[torch.nn.Parameter] = []
    if args.train_reencode_vision_branch:
        reencode_loaded = load_qwen3_vl(
            args.model_id,
            processor_id=args.processor_id,
            dtype=args.dtype,
            device_map=effective_device_map,
            attn_implementation=args.attn_implementation,
        )
        reencode_model = reencode_loaded.model
        freeze_qwen_backbone(reencode_model)
        reencode_train_info, reencode_trainable_params = _enable_reencode_vision_branch_training(reencode_model)
        reencode_model.train()

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
        encoder_adapter_layers=tuple(args.encoder_adapter_layers),
        encoder_adapter_type=args.encoder_adapter_type,
        encoder_adapter_gate_init=args.encoder_adapter_gate_init,
        encoder_adapter_share_weights=args.encoder_adapter_share_weights,
        encoder_adapter_layer_index_base=args.encoder_adapter_layer_index_base,
        encoder_reencode_deepstack_compatible=args.encoder_reencode_deepstack_compatible,
        encoder_reencode=args.variant == "tgvf_encoder_bidir_8_16_24",
        preserve_llm_kv_cache=True,
        second_full_llm_forward=False,
    )
    loss_weights = LossWeights(
        gen=args.loss_gen,
        visual_token_manifold=args.loss_visual_token_manifold,
        same_image_negative=args.loss_same_image_negative,
        contrastive_alignment=0.0,
    )
    config = {
        "stage": "tgvf_v3_stage1",
        "tgvf_protocol": args.tgvf_protocol,
        "model_id": args.model_id,
        "processor_id": loaded.processor_id,
        "dataset": str(args.train_file),
        "dataset_version": "tgvf_teacher_schema_v3_or_legacy_focus_normalized",
        "num_focus_samples_used": len(dataset),
        "num_rows_skipped": len(dataset.skipped_rows),
        "tgvf": asdict(module_config),
        "loss_weights": asdict(loss_weights),
        "learning_rate": args.learning_rate,
        "lr_scheduler": args.lr_scheduler,
        "warmup_steps": args.warmup_steps,
        "min_lr_ratio": args.min_lr_ratio,
        "batch_size": args.batch_size,
        "local_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "global_batch_size": args.batch_size * world_size * args.gradient_accumulation_steps,
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
        "batch_sampling": _resolved_batch_sampling(args),
        "drop_incomplete_same_image_batches": args.drop_incomplete_same_image_batches,
        "capture_mode": args.capture_mode,
        "focus_action_im_end": args.focus_action_im_end,
        "protocol_token_row_mode": args.protocol_token_row_mode,
        "protocol_c_special_token_ids": protocol_token_info.get("protocol_c_special_token_ids"),
        "protocol_c_tokenizer_info": protocol_token_info,
        "protocol_c_token_rows_trainable": protocol_c_token_train_info,
        "readout_batch_size": args.readout_batch_size,
        "dataloader_num_workers": args.num_workers,
        "distributed": world_size > 1,
        "world_size": world_size,
        "rank": rank,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "init_tgvf_from_checkpoint": args.init_tgvf_from_checkpoint,
        "reencode_vision_branch": reencode_train_info,
        "reencode_vision_learning_rate": args.reencode_vision_learning_rate,
        "dims": dims,
        "dtype": args.dtype,
        "device_map": effective_device_map,
        "attn_implementation": args.attn_implementation,
    }
    if is_main:
        (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
        if args.tgvf_protocol in token_row_protocols:
            processor.save_pretrained(output_dir / "processor")
    wandb_logger = WandbLogger(
        project=args.wandb_project if is_main else None,
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
    raw_foveal_module = build_tgvf_module(
        variant=args.variant,
        d_lm=dims["d_lm"],
        d_v=dims["d_v"],
        num_foveated_tokens=args.num_foveated_tokens,
        spatial_merge_size=spatial_merge_size,
        attn_dim=args.attn_dim,
        encoder_adapter_layers=args.encoder_adapter_layers,
        encoder_adapter_type=args.encoder_adapter_type,
        encoder_adapter_gate_init=args.encoder_adapter_gate_init,
        encoder_adapter_share_weights=args.encoder_adapter_share_weights,
        encoder_adapter_layer_index_base=args.encoder_adapter_layer_index_base,
        encoder_reencode_deepstack_compatible=args.encoder_reencode_deepstack_compatible,
    ).to(device=device, dtype=train_dtype)
    checkpoint = None
    if args.resume_from_checkpoint:
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        if args.tgvf_protocol in token_row_protocols:
            protocol_c_token_train_info["resume_token_rows"] = _restore_protocol_c_token_rows(
                model=model,
                tokenizer=processor.tokenizer,
                protocol=args.tgvf_protocol,
                checkpoint=checkpoint,
            )
        raw_foveal_module.load_state_dict(checkpoint["tgvf_module"], strict=True)
        if reencode_model is not None:
            reencode_train_info["resume_reencode_vision_branch"] = _restore_reencode_vision_branch(
                reencode_model=reencode_model,
                checkpoint=checkpoint,
            )
    elif args.init_tgvf_from_checkpoint:
        init_checkpoint = torch.load(args.init_tgvf_from_checkpoint, map_location="cpu")
        if args.tgvf_protocol in token_row_protocols:
            protocol_c_token_train_info["init_token_rows"] = _restore_protocol_c_token_rows(
                model=model,
                tokenizer=processor.tokenizer,
                protocol=args.tgvf_protocol,
                checkpoint=init_checkpoint,
            )
        raw_foveal_module.load_state_dict(init_checkpoint["tgvf_module"], strict=True)
        if reencode_model is not None:
            reencode_train_info["init_reencode_vision_branch"] = _restore_reencode_vision_branch(
                reencode_model=reencode_model,
                checkpoint=init_checkpoint,
            )
    raw_foveal_module.train()

    if world_size > 1:
        foveal_module = DistributedDataParallel(
            raw_foveal_module,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=args.ddp_find_unused_parameters,
        )
    else:
        foveal_module = raw_foveal_module

    if args.wandb_watch and wandb_logger.enabled:
        wandb_logger._wandb.watch(
            _checkpoint_module(foveal_module),
            log=args.wandb_watch,
            log_freq=max(args.log_every, 1),
        )
    optimizer_param_groups: list[dict[str, Any]] = [
        {"params": list(foveal_module.parameters()), "lr": args.learning_rate, "name": "tgvf_module"}
    ]
    if protocol_c_token_params:
        token_group = {"params": protocol_c_token_params, "lr": args.learning_rate, "name": "protocol_c_token_rows"}
        if args.protocol_token_row_mode == "full_mask":
            token_group["weight_decay"] = 0.0
        optimizer_param_groups.append(token_group)
    if reencode_trainable_params:
        optimizer_param_groups.append(
            {
                "params": reencode_trainable_params,
                "lr": args.reencode_vision_learning_rate,
                "name": "reencode_vision_branch",
            }
        )
    optimizer = torch.optim.AdamW(optimizer_param_groups, lr=args.learning_rate)
    scheduler = build_lr_scheduler(
        optimizer,
        scheduler_name=args.lr_scheduler,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    start_step = 0
    if args.resume_from_checkpoint:
        start_step = int(checkpoint.get("global_step") or checkpoint.get("optimizer_step") or 0)
        if checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])

    batch_sampling = _resolved_batch_sampling(args)
    if batch_sampling == "same_image":
        loader = DataLoader(
            dataset,
            batch_sampler=SameImageBatchSampler(
                dataset.samples,
                batch_size=args.batch_size,
                seed=args.seed,
                rank=rank,
                world_size=world_size,
                drop_incomplete=args.drop_incomplete_same_image_batches,
            ),
            collate_fn=tgvf_v3_stage1_collate,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        )
    else:
        distributed_sampler = (
            DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed,
            )
            if world_size > 1
            else None
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=distributed_sampler is None,
            sampler=distributed_sampler,
            collate_fn=tgvf_v3_stage1_collate,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        )
    data_iter = iter(loader)
    debug_examples_path = output_dir / "debug_examples.jsonl"
    optimizer.zero_grad(set_to_none=True)

    for step in range(start_step + 1, args.max_steps + 1):
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        loss_sums = {
            "loss_total": 0.0,
            "loss_gen": 0.0,
            "loss_visual_token_manifold": 0.0,
            "loss_same_image_negative": 0.0,
        }
        output = None
        for micro_step in range(args.gradient_accumulation_steps):
            try:
                samples = next(data_iter)
            except StopIteration:
                if batch_sampling != "same_image" and isinstance(getattr(loader, "sampler", None), DistributedSampler):
                    loader.sampler.set_epoch(step)
                data_iter = iter(loader)
                samples = next(data_iter)

            sync_context = (
                foveal_module.no_sync()
                if world_size > 1
                and hasattr(foveal_module, "no_sync")
                and micro_step < args.gradient_accumulation_steps - 1
                else nullcontext()
            )
            with sync_context:
                output = v3_stage1_training_step(
                    qwen_model=model,
                    processor=processor,
                    foveal_module=foveal_module,
                    reencode_qwen_model=reencode_model,
                    samples=samples,
                    loss_weights=loss_weights,
                    device=device,
                    hidden_state_index=args.capture_layer,
                    same_image_negative_margin=args.same_image_negative_margin,
                    same_image_negative_mode=args.same_image_negative_mode,
                    mask_original_image_after_tgvf=args.mask_original_image_after_tgvf,
                    position_mode=args.fvt_position_mode,
                    max_image_resolution=args.max_image_resolution,
                    capture_mode=args.capture_mode,
                    focus_action_im_end=args.focus_action_im_end,
                    readout_batch_size=args.readout_batch_size,
                    protocol=args.tgvf_protocol,
                )
                if not torch.isfinite(output.loss_total):
                    raise RuntimeError(f"Non-finite v3 Stage1 loss at step {step}: {output.loss_total}")
                (output.loss_total / args.gradient_accumulation_steps).backward()
            loss_sums["loss_total"] += float(output.loss_total.detach().cpu())
            loss_sums["loss_gen"] += float(output.loss_gen.detach().cpu())
            loss_sums["loss_visual_token_manifold"] += float(output.loss_visual_token_manifold.detach().cpu())
            loss_sums["loss_same_image_negative"] += float(output.loss_same_image_negative.detach().cpu())
        if output is None:
            raise RuntimeError("gradient accumulation produced no Stage1 output")
        if protocol_c_token_params:
            _average_protocol_c_token_row_gradients(protocol_c_token_params, world_size=world_size)
        if reencode_trainable_params:
            _average_trainable_gradients(reencode_trainable_params, world_size=world_size)
        trainable_for_clip = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_for_clip, args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if is_main and (step == 1 or step % args.log_every == 0 or step == args.max_steps):
            log = {
                "step": step,
                "resume_start_step": start_step,
                "loss_total": loss_sums["loss_total"] / args.gradient_accumulation_steps,
                "loss_gen": loss_sums["loss_gen"] / args.gradient_accumulation_steps,
                "loss_visual_token_manifold": loss_sums["loss_visual_token_manifold"] / args.gradient_accumulation_steps,
                "loss_same_image_negative": loss_sums["loss_same_image_negative"] / args.gradient_accumulation_steps,
                "grad_norm": float(grad_norm.detach().cpu()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "model_id": args.model_id,
                "dataset": str(args.train_file),
                "num_focus_samples_used": len(dataset),
                "variant": args.variant,
                "device": str(device),
                "world_size": world_size,
                "local_batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "global_batch_size": args.batch_size * world_size * args.gradient_accumulation_steps,
                "batch_sampling": batch_sampling,
                "capture_mode": args.capture_mode,
                "readout_batch_size": args.readout_batch_size,
                "peak_memory_gb": peak_memory_gb(),
                "protocol_c_token_rows_trainable": bool(protocol_c_token_params),
                "protocol_c_token_row_param_count": sum(int(p.numel()) for p in protocol_c_token_params),
                "reencode_vision_branch_trainable": bool(reencode_trainable_params),
                "reencode_vision_branch_param_count": sum(int(p.numel()) for p in reencode_trainable_params),
                "reencode_vision_learning_rate": args.reencode_vision_learning_rate,
                **output.debug,
            }
            print(json.dumps(_json_safe(log), indent=2, ensure_ascii=False))
            if wandb_logger.enabled:
                wandb_logger.log(_wandb_train_metrics(log), step=step)
            with debug_examples_path.open("a", encoding="utf-8") as handle:
                for example in output.debug.get("debug_examples", [])[: args.max_debug_examples_per_log]:
                    handle.write(json.dumps(_json_safe({"step": step, **example}), ensure_ascii=False) + "\n")

        if is_main and (step % args.save_every == 0 or step == args.max_steps):
            checkpoint_path = output_dir / f"checkpoint_step_{step}.pt"
            save_tgvf_checkpoint(
                path=checkpoint_path,
                foveal_module=_checkpoint_module(foveal_module),
                config=config,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=step,
                optimizer_step=step,
            )
            if reencode_model is not None:
                _append_reencode_vision_branch(
                    checkpoint_path,
                    reencode_model=reencode_model,
                    train_info=reencode_train_info,
                )
            if args.tgvf_protocol in token_row_protocols:
                _append_protocol_c_token_rows(
                    checkpoint_path,
                    model=model,
                    tokenizer=processor.tokenizer,
                    protocol=args.tgvf_protocol,
                )
                processor.save_pretrained(output_dir / f"processor_step_{step}")
            if args.wandb_log_checkpoints and wandb_logger.enabled:
                wandb_logger.log_artifact(
                    name=f"{output_dir.name}-checkpoint-{step}",
                    artifact_type="tgvf-v3-stage1-checkpoint",
                    paths=[output_dir / f"checkpoint_step_{step}.pt", output_dir / "config.json"],
                    aliases=["latest", f"step-{step}"],
                )
        distributed_barrier()

    if is_main:
        wandb_logger.finish()
    cleanup_distributed()


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
    parser.add_argument("--tgvf-protocol", choices=TGVF_PROTOCOL_CHOICES, default="legacy_v3_tags")
    parser.add_argument("--variant", choices=TGVF_VARIANTS, default="tgvf_v2_bidirectional")
    parser.add_argument("--num-foveated-tokens", type=_parse_optional_positive_int, default=None)
    parser.add_argument("--spatial-merge-size", default="auto")
    parser.add_argument("--attn-dim", type=int, default=None)
    parser.add_argument("--encoder-adapter-layers", type=_parse_int_list, default=(8, 16, 24))
    parser.add_argument("--encoder-adapter-type", choices=("bidirectional", "bidirectional_film_aggressive"), default="bidirectional")
    parser.add_argument("--encoder-adapter-gate-init", type=float, default=0.0)
    parser.add_argument("--encoder-adapter-share-weights", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--encoder-adapter-layer-index-base", type=int, choices=(0, 1), default=0)
    parser.add_argument("--encoder-reencode-deepstack-compatible", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-reencode-vision-branch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reencode-vision-learning-rate", type=float, default=1e-6)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", choices=("constant", "linear", "cosine"), default="constant")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--batch-sampling", choices=("auto", "random", "same_image"), default="auto")
    parser.add_argument("--drop-incomplete-same-image-batches", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--capture-mode", choices=("teacher_forced", "decode_loop"), default="teacher_forced")
    parser.add_argument("--focus-action-im-end", action=argparse.BooleanOptionalAction, default=False, help="Append <|im_end|> after teacher-forced focus action in Stage1.")
    parser.add_argument(
        "--protocol-token-row-mode",
        choices=("row_only", "full_mask"),
        default="row_only",
        help="How to train Protocol C marker token rows: current row-only override, or old full tensor with row gradient mask.",
    )
    parser.add_argument("--readout-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--capture-layer", type=int, default=-1)
    parser.add_argument("--loss-gen", type=float, default=1.0)
    parser.add_argument("--loss-visual-token-manifold", type=float, default=0.1)
    parser.add_argument("--loss-same-image-negative", type=float, default=1.0)
    parser.add_argument("--same-image-negative-margin", type=float, default=1.0)
    parser.add_argument("--same-image-negative-mode", choices=("cyclic_margin", "matrix_ce"), default="matrix_ce")
    parser.add_argument("--mask-original-image-after-tgvf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fvt-position-mode", choices=("native_source_grid", "inherit_source_visual_positions"), default="native_source_grid")
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--max-image-resolution", type=_parse_optional_positive_int, default=512)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--init-tgvf-from-checkpoint",
        default=None,
        help="Load only tgvf_module weights from a checkpoint, without optimizer/scheduler state.",
    )
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
    parser.add_argument("--ddp-find-unused-parameters", action="store_true")
    args = parser.parse_args()
    if args.variant not in TGVF_DYNAMIC_NUM_FVT_VARIANTS and args.num_foveated_tokens is None:
        variants = ", ".join(TGVF_DYNAMIC_NUM_FVT_VARIANTS)
        parser.error(f"--num-foveated-tokens none is supported only with --variant in: {variants}")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be >= 0")
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient-accumulation-steps must be >= 1")
    if args.readout_batch_size < 1:
        parser.error("--readout-batch-size must be >= 1")
    if args.variant == "tgvf_encoder_bidir_8_16_24" and args.encoder_adapter_type not in {"bidirectional", "bidirectional_film_aggressive"}:
        parser.error("tgvf_encoder_bidir_8_16_24 requires a supported encoder adapter type")
    if args.train_reencode_vision_branch and args.variant != "tgvf_encoder_bidir_8_16_24":
        parser.error("--train-reencode-vision-branch is currently supported only for tgvf_encoder_bidir_8_16_24")
    if args.num_workers < 0:
        parser.error("--num-workers must be >= 0")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        parser.error("--min-lr-ratio must be between 0 and 1")
    return args


def _parse_int_list(value: str) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    parsed = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected comma-separated integer list")
    return parsed


def _parse_optional_positive_int(value: str) -> int | None:
    if value.lower() in {"none", "null", "auto"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive, none, null, or auto")
    return parsed


def _resolved_batch_sampling(args: argparse.Namespace) -> str:
    if args.batch_sampling != "auto":
        return args.batch_sampling
    if (
        args.batch_size > 1
        and args.loss_same_image_negative > 0
        and args.same_image_negative_mode == "matrix_ce"
    ):
        return "same_image"
    return "random"


class SameImageBatchSampler:
    def __init__(
        self,
        samples: list[Any],
        *,
        batch_size: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        drop_incomplete: bool = True,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.samples = samples
        self.batch_size = batch_size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.drop_incomplete = drop_incomplete
        self.epoch = 0
        groups: dict[str, list[int]] = {}
        for index, sample in enumerate(samples):
            group_id = str(sample.image_id or sample.image)
            if world_size > 1:
                owner = int(hashlib.sha1(group_id.encode()).hexdigest(), 16) % world_size
                if owner != rank:
                    continue
            groups.setdefault(group_id, []).append(index)
        if batch_size > 1:
            min_size = batch_size if drop_incomplete else 2
            groups = {group_id: indices for group_id, indices in groups.items() if len(indices) >= min_size}
        self.groups = groups
        self._length = 0
        for indices in self.groups.values():
            if drop_incomplete:
                self._length += len(indices) // batch_size
            else:
                self._length += max(1, (len(indices) + batch_size - 1) // batch_size)
        if not self.groups:
            raise RuntimeError("same_image batch sampling found no image groups with enough samples")

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        group_ids = list(self.groups)
        rng.shuffle(group_ids)
        for group_id in group_ids:
            indices = list(self.groups[group_id])
            rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if self.drop_incomplete and len(batch) != self.batch_size:
                    continue
                if len(batch) == 1 and self.batch_size > 1:
                    continue
                yield batch

    def __len__(self) -> int:
        return self._length


def setup_distributed(args: argparse.Namespace) -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            device_map: str | dict[str, str] | None = f"cuda:{local_rank}"
        else:
            device = torch.device("cpu")
            device_map = None
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    else:
        device = torch.device(args.device)
        device_map = args.device_map
    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
        "device_map": device_map,
    }


def distributed_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _checkpoint_module(module: torch.nn.Module) -> torch.nn.Module:
    if isinstance(module, DistributedDataParallel):
        return module.module
    return module


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_name: str,
    max_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    warmup_steps = max(0, int(warmup_steps))
    min_lr_ratio = float(min_lr_ratio)

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
            cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
        raise ValueError(f"Unsupported lr scheduler: {scheduler_name}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


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




def _visual_module_for_stage1(model: Any) -> torch.nn.Module:
    if hasattr(model, "visual"):
        return model.visual
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        return model.model.visual
    raise AttributeError("Could not find Qwen visual module")


def _enable_reencode_vision_branch_training(model: Any) -> tuple[dict[str, Any], list[torch.nn.Parameter]]:
    visual = _visual_module_for_stage1(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in visual.parameters():
        parameter.requires_grad_(True)
    params = [parameter for parameter in visual.parameters() if parameter.requires_grad]
    return {
        "enabled": True,
        "scope": "separate_qwen3_visual_branch_only",
        "original_qwen_branch_frozen": True,
        "trainable_module": "visual",
        "trainable_param_count": sum(int(parameter.numel()) for parameter in params),
        "trainable_tensor_count": len(params),
    }, params


def _average_trainable_gradients(params: list[torch.nn.Parameter], *, world_size: int) -> None:
    if world_size <= 1 or not dist.is_available() or not dist.is_initialized():
        return
    for param in params:
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world_size)


def _append_reencode_vision_branch(path: Path, *, reencode_model: Any, train_info: dict[str, Any]) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    visual = _visual_module_for_stage1(reencode_model)
    checkpoint["reencode_vision_branch"] = {
        "visual_state_dict": {key: value.detach().cpu() for key, value in visual.state_dict().items()},
        "train_info": train_info,
    }
    torch.save(checkpoint, path)


def _restore_reencode_vision_branch(*, reencode_model: Any, checkpoint: dict[str, Any]) -> dict[str, Any]:
    payload = checkpoint.get("reencode_vision_branch")
    info = {"available_in_checkpoint": payload is not None, "loaded": False}
    if payload is None:
        info["reason"] = "missing_reencode_vision_branch"
        return info
    state = payload.get("visual_state_dict")
    if state is None:
        info["reason"] = "missing_visual_state_dict"
        return info
    visual = _visual_module_for_stage1(reencode_model)
    visual.load_state_dict(state, strict=True)
    info["loaded"] = True
    info["reason"] = "loaded_from_checkpoint"
    return info


def _enable_protocol_c_token_row_training(
    *,
    model: Any,
    tokenizer: Any,
    protocol: str,
    mode: str = "row_only",
) -> tuple[dict[str, Any], list[torch.nn.Parameter]]:
    if mode == "full_mask":
        return _enable_protocol_c_token_row_training_full_mask(
            model=model,
            tokenizer=tokenizer,
            protocol=protocol,
        )
    if mode != "row_only":
        raise ValueError(f"unsupported protocol token row mode: {mode}")
    return _enable_protocol_c_token_row_training_row_only(
        model=model,
        tokenizer=tokenizer,
        protocol=protocol,
    )


def _enable_protocol_c_token_row_training_row_only(
    *,
    model: Any,
    tokenizer: Any,
    protocol: str,
) -> tuple[dict[str, Any], list[torch.nn.Parameter]]:
    tokens = protocol_special_tokens(protocol)
    token_ids = protocol_special_token_ids(tokenizer, protocol=protocol)
    ordered_ids = [int(token_ids[token]) for token in tokens]
    input_embed = model.get_input_embeddings()
    output_embed = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    output_tied = (
        output_embed is not None
        and hasattr(output_embed, "weight")
        and int(output_embed.weight.data_ptr()) == int(input_embed.weight.data_ptr())
    )
    input_override = _RowOverrideEmbedding(input_embed, ordered_ids)
    if hasattr(model, "set_input_embeddings"):
        model.set_input_embeddings(input_override)
    else:
        raise RuntimeError("model does not support set_input_embeddings; cannot train protocol token rows only")
    params: list[torch.nn.Parameter] = []
    params.append(input_override.row_values)
    output_trainable = False
    output_override = None
    if output_embed is not None and hasattr(output_embed, "weight"):
        output_override = _RowOverrideOutput(output_embed, ordered_ids, shared_row_values=input_override.row_values if output_tied else None)
        if hasattr(model, "set_output_embeddings"):
            model.set_output_embeddings(output_override)
        else:
            raise RuntimeError("model does not support set_output_embeddings; cannot train protocol output token rows only")
        output_trainable = True
        if not output_tied:
            params.append(output_override.row_values)
    return {
        "enabled": True,
        "protocol": protocol,
        "tokens": list(tokens),
        "token_ids": {token: int(token_ids[token]) for token in tokens},
        "num_token_rows": len(ordered_ids),
        "input_embeddings_trainable": True,
        "output_embeddings_trainable": output_trainable,
        "input_output_tied": output_tied,
        "row_only_parameters": True,
        "optimizer_param_tensors": len(params),
        "optimizer_param_count": sum(int(param.numel()) for param in params),
        "row_gradient_mask_active": False,
    }, params


def _enable_protocol_c_token_row_training_full_mask(
    *,
    model: Any,
    tokenizer: Any,
    protocol: str,
) -> tuple[dict[str, Any], list[torch.nn.Parameter]]:
    tokens = protocol_special_tokens(protocol)
    token_ids = protocol_special_token_ids(tokenizer, protocol=protocol)
    ordered_ids = [int(token_ids[token]) for token in tokens]
    row_ids = torch.tensor(sorted({int(row_id) for row_id in ordered_ids}), dtype=torch.long)
    input_embed = model.get_input_embeddings()
    output_embed = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    output_tied = (
        output_embed is not None
        and hasattr(output_embed, "weight")
        and int(output_embed.weight.data_ptr()) == int(input_embed.weight.data_ptr())
    )
    params: list[torch.nn.Parameter] = []
    _enable_weight_rows_with_gradient_mask(input_embed.weight, row_ids=row_ids, name="input_embeddings")
    params.append(input_embed.weight)
    output_trainable = False
    if output_embed is not None and hasattr(output_embed, "weight"):
        output_trainable = True
        if not output_tied:
            _enable_weight_rows_with_gradient_mask(output_embed.weight, row_ids=row_ids, name="output_embeddings")
            params.append(output_embed.weight)
    return {
        "enabled": True,
        "protocol": protocol,
        "tokens": list(tokens),
        "token_ids": {token: int(token_ids[token]) for token in tokens},
        "num_token_rows": len(ordered_ids),
        "input_embeddings_trainable": True,
        "output_embeddings_trainable": output_trainable,
        "input_output_tied": output_tied,
        "row_only_parameters": False,
        "optimizer_param_tensors": len(params),
        "optimizer_param_count": sum(int(param.numel()) for param in params),
        "row_gradient_mask_active": True,
        "weight_decay": 0.0,
    }, params


def _enable_weight_rows_with_gradient_mask(weight: torch.nn.Parameter, *, row_ids: torch.Tensor, name: str) -> None:
    weight.requires_grad_(True)
    setattr(weight, "_tgvf_protocol_row_ids", row_ids.detach().cpu())
    setattr(weight, "_tgvf_protocol_row_mask_name", name)
    weight.register_hook(_protocol_row_gradient_mask_hook(row_ids.detach().cpu()))


def _protocol_row_gradient_mask_hook(row_ids_cpu: torch.Tensor):
    def hook(grad: torch.Tensor) -> torch.Tensor:
        row_ids = row_ids_cpu.to(device=grad.device, dtype=torch.long)
        masked = torch.zeros_like(grad)
        masked.index_copy_(0, row_ids, grad.index_select(0, row_ids))
        return masked

    return hook


class _RowOverrideEmbedding(torch.nn.Module):
    def __init__(self, base: torch.nn.Module, row_ids: list[int]) -> None:
        super().__init__()
        self.base = base
        rows = torch.tensor(sorted({int(row_id) for row_id in row_ids}), dtype=torch.long)
        if rows.numel() == 0:
            raise ValueError("Protocol C token row ids are empty")
        self.register_buffer("row_ids", rows, persistent=False)
        self.row_values = torch.nn.Parameter(base.weight.detach()[rows].clone())
        self.num_embeddings = int(base.weight.shape[0])
        self.embedding_dim = int(base.weight.shape[1])
        self.padding_idx = getattr(base, "padding_idx", None)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.base(input_ids)
        row_ids = self.row_ids.to(device=input_ids.device)
        row_values = self.row_values.to(device=embeds.device, dtype=embeds.dtype)
        for idx in range(int(row_ids.numel())):
            mask = input_ids == row_ids[idx]
            if bool(mask.any()):
                embeds = torch.where(mask.unsqueeze(-1), row_values[idx].view(*([1] * (embeds.ndim - 1)), -1), embeds)
        return embeds

    def effective_rows(self) -> torch.Tensor:
        return self.row_values

    def set_rows(self, rows: torch.Tensor) -> None:
        with torch.no_grad():
            self.row_values.copy_(rows.to(device=self.row_values.device, dtype=self.row_values.dtype))


class _RowOverrideOutput(torch.nn.Module):
    def __init__(
        self,
        base: torch.nn.Module,
        row_ids: list[int],
        *,
        shared_row_values: torch.nn.Parameter | None = None,
    ) -> None:
        super().__init__()
        self.base = base
        rows = torch.tensor(sorted({int(row_id) for row_id in row_ids}), dtype=torch.long)
        if rows.numel() == 0:
            raise ValueError("Protocol C output token row ids are empty")
        self.register_buffer("row_ids", rows, persistent=False)
        if shared_row_values is None:
            self.row_values = torch.nn.Parameter(base.weight.detach()[rows].clone())
        else:
            self.row_values = shared_row_values
        bias = getattr(base, "bias", None)
        self.register_buffer(
            "row_bias",
            None if bias is None else bias.detach()[rows].clone(),
            persistent=False,
        )

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return getattr(self.base, "bias", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.base(hidden_states)
        row_values = self.row_values.to(device=hidden_states.device, dtype=hidden_states.dtype)
        selected_logits = torch.matmul(hidden_states, row_values.t())
        if self.row_bias is not None:
            selected_logits = selected_logits + self.row_bias.to(device=hidden_states.device, dtype=hidden_states.dtype)
        logits.index_copy_(-1, self.row_ids.to(device=logits.device), selected_logits.to(dtype=logits.dtype))
        return logits

    def effective_rows(self) -> torch.Tensor:
        return self.row_values

    def set_rows(self, rows: torch.Tensor) -> None:
        with torch.no_grad():
            self.row_values.copy_(rows.to(device=self.row_values.device, dtype=self.row_values.dtype))


def _average_protocol_c_token_row_gradients(params: list[torch.nn.Parameter], *, world_size: int) -> None:
    if world_size <= 1 or not dist.is_available() or not dist.is_initialized():
        return
    for param in params:
        if param.grad is None:
            continue
        row_ids_cpu = getattr(param, "_tgvf_protocol_row_ids", None)
        if isinstance(row_ids_cpu, torch.Tensor) and param.grad.ndim >= 2:
            row_ids = row_ids_cpu.to(device=param.grad.device, dtype=torch.long)
            row_grad = param.grad.index_select(0, row_ids).contiguous()
            dist.all_reduce(row_grad, op=dist.ReduceOp.SUM)
            row_grad.div_(world_size)
            param.grad.zero_()
            param.grad.index_copy_(0, row_ids, row_grad)
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world_size)


def _restore_protocol_c_token_rows(
    *,
    model: Any,
    tokenizer: Any,
    protocol: str,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    payload = checkpoint.get("protocol_c_token_rows")
    tokens = protocol_special_tokens(protocol)
    current_ids = protocol_special_token_ids(tokenizer, protocol=protocol)
    info: dict[str, Any] = {
        "available_in_checkpoint": payload is not None,
        "loaded": False,
        "protocol": protocol,
        "tokens": list(tokens),
        "current_token_ids": {token: int(current_ids[token]) for token in tokens},
    }
    if payload is None:
        info["reason"] = "missing_protocol_c_token_rows"
        return info
    saved_ids = payload.get("token_ids") or {}
    mismatched = {
        token: {"current": int(current_ids[token]), "saved": int(saved_ids.get(token, -1))}
        for token in tokens
        if int(saved_ids.get(token, -1)) != int(current_ids[token])
    }
    info["saved_token_ids"] = {token: int(saved_ids.get(token, -1)) for token in tokens}
    if mismatched:
        info["reason"] = "token_id_mismatch"
        info["mismatched_token_ids"] = mismatched
        return info
    ordered_ids = [int(current_ids[token]) for token in tokens]
    input_rows = payload.get("input_embeddings")
    output_rows = payload.get("output_embeddings")
    if input_rows is None:
        info["reason"] = "missing_input_embeddings"
        return info
    input_embed = model.get_input_embeddings()
    output_embed = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    with torch.no_grad():
        if hasattr(input_embed, "set_rows"):
            input_embed.set_rows(input_rows)
        else:
            input_embed.weight[ordered_ids].copy_(input_rows.to(device=input_embed.weight.device, dtype=input_embed.weight.dtype))
        if output_rows is not None and output_embed is not None and hasattr(output_embed, "weight"):
            if hasattr(output_embed, "set_rows"):
                output_embed.set_rows(output_rows)
            else:
                output_embed.weight[ordered_ids].copy_(output_rows.to(device=output_embed.weight.device, dtype=output_embed.weight.dtype))
    info["loaded"] = True
    info["reason"] = "loaded_from_checkpoint"
    info["loaded_input_rows"] = True
    info["loaded_output_rows"] = output_rows is not None
    return info

def _append_protocol_c_token_rows(path: Path, *, model: Any, tokenizer: Any, protocol: str) -> None:
    tokens = protocol_special_tokens(protocol)
    token_ids = protocol_special_token_ids(tokenizer, protocol=protocol)
    ordered_ids = [int(token_ids[token]) for token in tokens]
    input_embed = model.get_input_embeddings()
    output_embed = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    input_rows = (
        input_embed.effective_rows().detach().cpu().clone()
        if hasattr(input_embed, "effective_rows")
        else input_embed.weight.detach().cpu()[ordered_ids].clone()
    )
    payload: dict[str, Any] = {
        "protocol": protocol,
        "tokens": list(tokens),
        "token_ids": {token: int(token_ids[token]) for token in tokens},
        "input_embeddings": input_rows,
    }
    if output_embed is not None and hasattr(output_embed, "weight"):
        payload["output_embeddings"] = (
            output_embed.effective_rows().detach().cpu().clone()
            if hasattr(output_embed, "effective_rows")
            else output_embed.weight.detach().cpu()[ordered_ids].clone()
        )
    checkpoint = torch.load(path, map_location="cpu")
    checkpoint["protocol_c_token_rows"] = payload
    torch.save(checkpoint, path)

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
