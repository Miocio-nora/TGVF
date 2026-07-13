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
from revisit_vlm_clean.peft_token_rows import (
    PROTOCOL_TOKEN_TRAINING_FULL_MODULES,
    PROTOCOL_TOKEN_TRAINING_MODES,
    protocol_token_peft_kwargs,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from revisit_vlm.qwen3_vl_tgvf import (
    PROTOCOL_C_THINKING_SPECIAL,
    PROTOCOL_C_TOOL_OBSERVATION,
    PROTOCOL_C_TOOL_OBSERVATION_QWEN2_NO_THINK,
    PROTOCOL_D_QWEN_TOOL,
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
    TGVFModuleConfig,
    build_tgvf_module,
)
from revisit_vlm.tgvf_v3_stage1 import freeze_qwen_backbone, infer_qwen3_stage1_dims
from revisit_vlm.tgvf_v3_stage2 import (
    ORIGINAL_IMAGE_MASK_SCOPE_CHOICES,
    Stage2SameImageFocusBatchCursor,
    Stage2LossWeights,
    TGVFv3Stage2Dataset,
    dataset_stage2_stats,
    tgvf_v3_stage2_collate,
    v3_stage2_training_step,
)
from revisit_vlm.tgvf_v3_stage2_fast import (
    v3_stage2_batched_training_step,
    v3_stage2_same_image_matrix_ce_step,
)
from revisit_vlm.wandb_logging import WandbLogger


def main() -> None:
    args = parse_args()
    if not (0.0 <= float(args.mask_original_image_after_tgvf_prob) <= 1.0):
        raise ValueError("--mask-original-image-after-tgvf-prob must be in [0, 1]")
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
    matrix_ce_cursor = (
        Stage2SameImageFocusBatchCursor(
            train_dataset.samples,
            group_size=args.matrix_ce_group_size,
            seed=args.seed,
            rank=rank,
            world_size=world_size,
        )
        if args.matrix_ce_preservation
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
    protocol_token_info: dict[str, Any] = {}
    token_row_protocols = {
        PROTOCOL_C_THINKING_SPECIAL,
        PROTOCOL_C_TOOL_OBSERVATION,
        PROTOCOL_C_TOOL_OBSERVATION_QWEN2_NO_THINK,
        PROTOCOL_E_ACTION_EVIDENCE_SPECIAL,
    }
    if args.tgvf_protocol in token_row_protocols:
        protocol_token_info = ensure_tgvf_protocol_tokens(processor.tokenizer, model, protocol=args.tgvf_protocol)
    stage1_checkpoint = torch.load(args.stage1_checkpoint, map_location="cpu")
    if (
        args.tgvf_protocol in token_row_protocols
        and args.protocol_token_training_mode != PROTOCOL_TOKEN_TRAINING_FULL_MODULES
    ):
        _restore_protocol_c_token_rows_from_stage1(
            model=model,
            tokenizer=processor.tokenizer,
            protocol=args.tgvf_protocol,
            stage1_checkpoint=stage1_checkpoint,
        )
    freeze_qwen_backbone(model)
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False
    lora_targets = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    token_peft_kwargs = (
        protocol_token_peft_kwargs(
            mode=args.protocol_token_training_mode,
            token_ids=list(
                protocol_special_token_ids(
                    processor.tokenizer,
                    protocol=args.tgvf_protocol,
                ).values()
            ),
        )
        if args.tgvf_protocol in token_row_protocols
        else {"modules_to_save": None, "trainable_token_indices": None}
    )
    lora_modules_to_save = token_peft_kwargs["modules_to_save"]
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=lora_targets,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type="CAUSAL_LM",
        modules_to_save=lora_modules_to_save,
        trainable_token_indices=token_peft_kwargs["trainable_token_indices"],
        ensure_weight_tying=False,
    )
    model = get_peft_model(model, lora_config)
    model.train()
    utility_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    tokenizer_size_after = len(processor.tokenizer)

    stage1_tgvf_config_resolution = _apply_stage1_tgvf_config(args, stage1_checkpoint)
    if args.d_deepstack_enabled and not args.deepstack_enabled:
        raise ValueError("Stage2 D DeepStack checkpoint config requires --deepstack-enabled")

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
        encoder_adapter_layers=tuple(args.encoder_adapter_layers),
        encoder_adapter_type=args.encoder_adapter_type,
        encoder_adapter_gate_init=args.encoder_adapter_gate_init,
        encoder_adapter_share_weights=args.encoder_adapter_share_weights,
        encoder_adapter_layer_index_base=args.encoder_adapter_layer_index_base,
        encoder_reencode_deepstack_compatible=args.encoder_reencode_deepstack_compatible,
        d_deepstack_enabled=args.d_deepstack_enabled,
        d_deepstack_branch_layers=tuple(args.d_deepstack_branch_layers),
        encoder_reencode=args.variant == "tgvf_encoder_bidir_8_16_24",
        preserve_llm_kv_cache=True,
        second_full_llm_forward=False,
    )
    train_dtype = next(model.parameters()).dtype
    foveal_module = build_tgvf_module(
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
        d_deepstack_enabled=args.d_deepstack_enabled,
        d_deepstack_branch_layers=tuple(args.d_deepstack_branch_layers),
    ).to(device=device, dtype=train_dtype)
    stage1_token_row_info = {}
    if (
        args.protocol_token_training_mode == PROTOCOL_TOKEN_TRAINING_FULL_MODULES
        and args.tgvf_protocol in {
        PROTOCOL_C_THINKING_SPECIAL,
        PROTOCOL_C_TOOL_OBSERVATION,
        PROTOCOL_C_TOOL_OBSERVATION_QWEN2_NO_THINK,
        }
    ):
        stage1_token_row_info = _restore_protocol_c_token_rows_from_stage1(
            model=model,
            tokenizer=processor.tokenizer,
            protocol=args.tgvf_protocol,
            stage1_checkpoint=stage1_checkpoint,
        )
    elif args.tgvf_protocol == PROTOCOL_E_ACTION_EVIDENCE_SPECIAL:
        stage1_token_row_info = {
            "available_in_stage1_checkpoint": False,
            "loaded": False,
            "reason": "protocol_e_initializes_token_rows_from_base_model",
            "current_token_ids": {
                token: int(token_id)
                for token, token_id in protocol_special_token_ids(
                    processor.tokenizer, protocol=args.tgvf_protocol
                ).items()
            },
        }
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
        "tgvf_protocol": args.tgvf_protocol,
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
            "modules_to_save": lora_modules_to_save,
            "protocol_token_training_mode": args.protocol_token_training_mode,
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
        "mask_original_image_after_tgvf_prob": args.mask_original_image_after_tgvf_prob,
        "mask_original_image_after_tgvf_scope": args.mask_original_image_after_tgvf_scope,
        "deepstack": {
            "enabled": bool(args.deepstack_enabled),
            "original_image_scope": (
                args.mask_original_image_after_tgvf_scope if args.deepstack_enabled else "off"
            ),
            "d_features_enabled": bool(args.d_deepstack_enabled),
        },
        "fvt_position_mode": args.fvt_position_mode,
        "capture_mode": "teacher_forced",
        "train_long_cot": False,
        "special_tokens_added": bool(protocol_token_info.get("special_tokens_added", False)),
        "normal_tokens_added": bool(protocol_token_info.get("normal_tokens_added", False)),
        "protocol_c_token_registration": protocol_token_info.get("protocol_c_token_registration"),
        "tokenizer_resized": tokenizer_size_before != tokenizer_size_after,
        "markers_are_plain_text": args.tgvf_protocol not in {
            PROTOCOL_C_THINKING_SPECIAL,
            PROTOCOL_C_TOOL_OBSERVATION,
            PROTOCOL_C_TOOL_OBSERVATION_QWEN2_NO_THINK,
            PROTOCOL_D_QWEN_TOOL,
            PROTOCOL_E_ACTION_EVIDENCE_SPECIAL,
        },
        "protocol_c_special_token_ids": protocol_token_info.get("protocol_c_special_token_ids"),
        "protocol_c_token_ids": protocol_token_info.get("protocol_c_token_ids"),
        "protocol_c_tokenizer_info": protocol_token_info,
        "protocol_c_stage1_token_rows": stage1_token_row_info,
        "fast_batched_stage2": args.fast_batched_stage2,
        "matrix_ce_enabled": bool(args.matrix_ce_preservation),
        "same_image_negative_enabled": bool(args.matrix_ce_preservation),
        "matrix_ce_preservation": {
            "enabled": bool(args.matrix_ce_preservation),
            "weight": float(args.loss_same_image_matrix_ce)
            if args.matrix_ce_preservation
            else 0.0,
            "group_size": int(args.matrix_ce_group_size),
            "readout_batch_size": int(args.matrix_ce_readout_batch_size),
            "sample_scope": "same_image_single_focus",
            "score_span": "post_d_readout_before_answer",
            "candidate_swap": "d_and_d_deepstack_features",
            "original_image_mask_probability": 1.0,
            "vision_encode_count_per_group": 1,
            "sampling": matrix_ce_cursor.summary() if matrix_ce_cursor is not None else None,
        },
        "contrastive_alignment_enabled": False,
        "tgvf_trainable": True,
        "qwen_base_frozen": True,
        "vision_encoder_frozen": True,
        "lm_head_frozen": args.tgvf_protocol not in token_row_protocols,
        "input_embeddings_frozen": args.tgvf_protocol not in token_row_protocols,
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
        if args.tgvf_protocol in token_row_protocols:
            processor.save_pretrained(output_dir / "processor")
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

    sampler = None
    if args.target_focus_ratio is not None:
        target_focus_ratio = float(args.target_focus_ratio)
        focus_count = max(sum(1 for sample in train_dataset.samples if sample.need_focus), 1)
        no_focus_count = max(sum(1 for sample in train_dataset.samples if not sample.need_focus), 1)
        weights = [
            (target_focus_ratio / focus_count) if sample.need_focus else ((1.0 - target_focus_ratio) / no_focus_count)
            for sample in train_dataset.samples
        ]
        generator = torch.Generator()
        generator.manual_seed(int(args.seed) + rank)
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=math.ceil(len(train_dataset) / max(world_size, 1)),
            replacement=True,
            generator=generator,
        )
    elif world_size > 1:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
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
    running_loss_same_image_matrix_ce = 0.0
    running_logs: list[dict[str, Any]] = []

    while optimizer_step < args.max_steps:
        if sampler is not None and hasattr(sampler, "set_epoch") and micro_step % max(len(loader), 1) == 0:
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
            mask_original_image_after_tgvf_prob=args.mask_original_image_after_tgvf_prob,
            mask_original_image_after_tgvf_scope=args.mask_original_image_after_tgvf_scope,
            protocol=args.tgvf_protocol,
            deepstack_enabled=bool(args.deepstack_enabled),
        )
        if matrix_ce_cursor is not None:
            matrix_ce_batch = matrix_ce_cursor.next_batch()
            matrix_output = v3_stage2_same_image_matrix_ce_step(
                qwen_model=utility_model,
                qwen_forward_model=model,
                processor=processor,
                foveal_module=foveal_module,
                samples=matrix_ce_batch["samples"],
                loss_weights=loss_weights,
                device=device,
                hidden_state_index=args.capture_layer,
                max_image_resolution=args.max_image_resolution,
                mask_original_image_after_tgvf_scope=args.mask_original_image_after_tgvf_scope,
                protocol=args.tgvf_protocol,
                deepstack_enabled=bool(args.deepstack_enabled),
                readout_batch_size=args.matrix_ce_readout_batch_size,
            )
            output.loss_same_image_matrix_ce = matrix_output.loss
            output.loss_total = (
                output.loss_total
                + float(args.loss_same_image_matrix_ce) * matrix_output.loss
            )
            output.debug.update(matrix_output.debug)
            output.debug["matrix_ce_sample_indices"] = matrix_ce_batch["sample_indices"]
        if not torch.isfinite(output.loss_total):
            raise RuntimeError(f"Non-finite Stage2 loss at micro step {micro_step}: {output.loss_total}")
        (output.loss_total / args.gradient_accumulation_steps).backward()
        micro_step += 1
        running_loss += float(output.loss_total.detach().cpu())
        running_loss_focus += float(output.loss_focus.detach().cpu())
        running_loss_no_focus += float(output.loss_no_focus.detach().cpu())
        running_loss_visual_token_manifold += float(output.loss_visual_token_manifold.detach().cpu())
        running_loss_same_image_matrix_ce += float(
            output.loss_same_image_matrix_ce.detach().cpu()
        )
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
                "loss_same_image_matrix_ce": running_loss_same_image_matrix_ce
                / max(len(running_logs), 1),
                "grad_norm": float(grad_norm.detach().cpu()),
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
                "peak_memory_gb": peak_memory_gb(),
                "world_size": world_size,
                "effective_global_batch_size": args.batch_size * args.gradient_accumulation_steps * world_size,
                "target_focus_sampling_ratio": args.target_focus_ratio,
                **summarize_step_debug(running_logs),
            }
            log["special_tokens_added"] = config["special_tokens_added"]
            log["normal_tokens_added"] = config.get("normal_tokens_added")
            log["protocol_c_token_registration"] = config.get("protocol_c_token_registration")
            log["tokenizer_resized"] = config["tokenizer_resized"]
            log["markers_are_plain_text"] = config["markers_are_plain_text"]
            print(json.dumps(_json_safe(log), indent=2, ensure_ascii=False))
            if wandb_logger.enabled:
                wandb_logger.log(_wandb_metrics(log), step=optimizer_step)
            running_loss = 0.0
            running_loss_focus = 0.0
            running_loss_no_focus = 0.0
            running_loss_visual_token_manifold = 0.0
            running_loss_same_image_matrix_ce = 0.0
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
            if args.tgvf_protocol in token_row_protocols:
                processor.save_pretrained(output_dir / f"processor_step_{optimizer_step}")

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
    boundary_logs = []
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
                mask_original_image_after_tgvf_prob=args.mask_original_image_after_tgvf_prob,
                mask_original_image_after_tgvf_scope=args.mask_original_image_after_tgvf_scope,
                protocol=args.tgvf_protocol,
                deepstack_enabled=bool(args.deepstack_enabled),
            )
            losses.append(float(output.loss_total.detach().cpu()))
            focus += output.debug["focus_count"]
            no_focus += output.debug["no_focus_count"]
            if output.debug["value_span_match_rate"] is not None:
                value_rates.append(float(output.debug["value_span_match_rate"]))
            mask_focus.append(float(output.debug["focus_sample_mask_active_rate"]))
            mask_no_focus.append(float(output.debug["no_focus_mask_active_rate"]))
            boundary_logs.append(output.debug)
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
            "matrix_ce_enabled": bool(args.matrix_ce_preservation),
            "matrix_ce_evaluated": False,
            **_merge_boundary_debug(boundary_logs),
        },
    }



def _restore_protocol_c_token_rows_from_stage1(
    *,
    model: Any,
    tokenizer: Any,
    protocol: str,
    stage1_checkpoint: dict[str, Any],
) -> dict[str, Any]:
    payload = stage1_checkpoint.get("protocol_c_token_rows")
    tokens = protocol_special_tokens(protocol)
    current_ids = protocol_special_token_ids(tokenizer, protocol=protocol)
    info: dict[str, Any] = {
        "available_in_stage1_checkpoint": payload is not None,
        "loaded": False,
        "reason": None,
        "protocol": protocol,
        "tokens": list(tokens),
        "current_token_ids": {token: int(current_ids[token]) for token in tokens},
    }
    if payload is None:
        info["reason"] = "missing_protocol_c_token_rows_in_stage1_checkpoint"
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
        input_tensor = input_rows.to(device=input_embed.weight.device, dtype=input_embed.weight.dtype)
        input_embed.weight[ordered_ids].copy_(input_tensor)
        if output_rows is not None and output_embed is not None and hasattr(output_embed, "weight"):
            output_tensor = output_rows.to(device=output_embed.weight.device, dtype=output_embed.weight.dtype)
            output_embed.weight[ordered_ids].copy_(output_tensor)
    info["loaded"] = True
    info["reason"] = "loaded_from_stage1_checkpoint"
    info["loaded_input_rows"] = True
    info["loaded_output_rows"] = output_rows is not None
    return info

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
    focus_mask_active = sum(
        float(item.get("focus_sample_mask_active_rate", 0.0) or 0.0) * int(item.get("focus_count", 0))
        for item in debug_logs
    )
    no_focus_mask_active = sum(
        float(item.get("no_focus_mask_active_rate", 0.0) or 0.0) * int(item.get("no_focus_count", 0))
        for item in debug_logs
    )
    value_rates = [item.get("value_span_match_rate") for item in debug_logs if item.get("value_span_match_rate") is not None]
    protocol = debug_logs[0].get("tgvf_protocol", "legacy_v3_tags") if debug_logs else "legacy_v3_tags"
    mask_probs = [
        float(item.get("mask_original_image_after_tgvf_prob"))
        for item in debug_logs
        if item.get("mask_original_image_after_tgvf_prob") is not None
    ]
    mask_scopes = [
        str(item.get("mask_original_image_after_tgvf_scope"))
        for item in debug_logs
        if item.get("mask_original_image_after_tgvf_scope") is not None
    ]
    examples = []
    for item in debug_logs:
        examples.extend(item.get("debug_examples", []))
        if len(examples) >= 2:
            break
    matrix_top1 = [
        float(item["matrix_ce_top1"])
        for item in debug_logs
        if item.get("matrix_ce_top1") is not None
    ]
    matrix_margins = [
        float(item["matrix_ce_positive_negative_margin"])
        for item in debug_logs
        if item.get("matrix_ce_positive_negative_margin") is not None
    ]
    matrix_enabled = any(bool(item.get("matrix_ce_enabled")) for item in debug_logs)
    return {
        "focus_count": focus,
        "no_focus_count": no_focus,
        "focus_ratio": focus / max(focus + no_focus, 1),
        "no_focus_ratio": no_focus / max(focus + no_focus, 1),
        "focus_sample_mask_active_rate": focus_mask_active / max(focus, 1),
        "no_focus_mask_active_rate": no_focus_mask_active / max(no_focus, 1),
        "mask_original_image_after_tgvf_prob": sum(mask_probs) / max(len(mask_probs), 1) if mask_probs else None,
        "mask_original_image_after_tgvf_scope": mask_scopes[0] if mask_scopes else None,
        "value_span_match_rate": sum(value_rates) / max(len(value_rates), 1) if value_rates else None,
        "special_tokens_added": False,
        "tokenizer_resized": False,
        "markers_are_plain_text": protocol not in {
            PROTOCOL_C_THINKING_SPECIAL,
            PROTOCOL_C_TOOL_OBSERVATION,
            PROTOCOL_C_TOOL_OBSERVATION_QWEN2_NO_THINK,
            PROTOCOL_D_QWEN_TOOL,
            PROTOCOL_E_ACTION_EVIDENCE_SPECIAL,
        },
        "tgvf_protocol": protocol,
        "matrix_ce_enabled": matrix_enabled,
        "same_image_negative_enabled": matrix_enabled,
        "matrix_ce_top1": sum(matrix_top1) / len(matrix_top1) if matrix_top1 else None,
        "matrix_ce_positive_negative_margin": (
            sum(matrix_margins) / len(matrix_margins) if matrix_margins else None
        ),
        **_merge_boundary_debug(debug_logs),
        "debug_examples": examples[:2],
    }


def _merge_boundary_debug(debug_logs: list[dict[str, Any]]) -> dict[str, Any]:
    names = ("focus_start", "focus_end", "tgvf_start", "tgvf_end", "evidence_start", "evidence_end")
    merged: dict[str, Any] = {}
    acc_values = []
    total_support = 0
    for name in names:
        support_key = f"protocol_c_boundary_support_{name}"
        acc_key = f"protocol_c_boundary_acc_{name}"
        support = sum(int(item.get(support_key, 0) or 0) for item in debug_logs)
        total_support += support
        merged[support_key] = support
        if support:
            weighted = 0.0
            for item in debug_logs:
                item_support = int(item.get(support_key, 0) or 0)
                item_acc = item.get(acc_key)
                if item_support and item_acc is not None:
                    weighted += float(item_acc) * item_support
            acc = weighted / support
            merged[acc_key] = acc
            acc_values.append(acc)
        else:
            merged[acc_key] = None
    merged["protocol_c_boundary_support_total"] = total_support
    merged["protocol_c_boundary_acc_mean"] = sum(acc_values) / len(acc_values) if acc_values else None
    return merged


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


def _apply_stage1_tgvf_config(args: argparse.Namespace, stage1_checkpoint: dict) -> dict:
    """Resolve Stage2 TGVF module args from the Stage1 checkpoint config.

    Stage2 normally should instantiate the exact Stage1 TGVF module variant before
    loading `tgvf_module`. This matters for encoder-reencode checkpoints, which
    are incompatible with the legacy bidirectional module shape.
    """

    if not getattr(args, "use_stage1_tgvf_config", True):
        return {"enabled": False, "reason": "disabled_by_cli"}

    tgvf_config = ((stage1_checkpoint.get("config") or {}).get("tgvf") or {})
    if not tgvf_config:
        return {"enabled": False, "reason": "missing_checkpoint_config"}

    fields = (
        "variant",
        "num_foveated_tokens",
        "spatial_merge_size",
        "attn_dim",
        "encoder_adapter_layers",
        "encoder_adapter_type",
        "encoder_adapter_gate_init",
        "encoder_adapter_share_weights",
        "encoder_adapter_layer_index_base",
        "encoder_reencode_deepstack_compatible",
        "d_deepstack_enabled",
        "d_deepstack_branch_layers",
    )
    applied: dict[str, dict[str, object]] = {}
    for field in fields:
        if field not in tgvf_config:
            continue
        old_value = getattr(args, field, None)
        new_value = tgvf_config[field]
        if field == "encoder_adapter_layers" and new_value is not None:
            new_value = tuple(int(v) for v in new_value)
        setattr(args, field, new_value)
        if old_value != new_value:
            applied[field] = {"old": old_value, "new": new_value}
    return {"enabled": True, "source": "stage1_checkpoint.config.tgvf", "applied": applied}


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
    parser.add_argument("--tgvf-protocol", choices=TGVF_PROTOCOL_CHOICES, default="legacy_v3_tags")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--variant", choices=TGVF_VARIANTS, default="tgvf_v2_bidirectional")
    parser.add_argument("--use-stage1-tgvf-config", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-foveated-tokens", type=_parse_optional_positive_int, default=None)
    parser.add_argument("--spatial-merge-size", default="auto")
    parser.add_argument("--attn-dim", type=int, default=None)
    parser.add_argument("--encoder-adapter-layers", type=_parse_int_list, default=(8, 16, 24))
    parser.add_argument("--encoder-adapter-type", choices=("bidirectional", "bidirectional_film_aggressive"), default="bidirectional")
    parser.add_argument("--encoder-adapter-gate-init", type=float, default=0.0)
    parser.add_argument("--encoder-adapter-share-weights", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--encoder-adapter-layer-index-base", type=int, choices=(0, 1), default=0)
    parser.add_argument("--encoder-reencode-deepstack-compatible", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--d-deepstack-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--d-deepstack-branch-layers", type=_parse_int_list, default=(8, 16, 24))
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=256)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-bias", default="none", choices=("none", "all", "lora_only"))
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument(
        "--protocol-token-training-mode",
        choices=PROTOCOL_TOKEN_TRAINING_MODES,
        default=PROTOCOL_TOKEN_TRAINING_FULL_MODULES,
    )
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
    parser.add_argument("--target-focus-ratio", type=float, default=None)
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
    parser.add_argument("--mask-original-image-after-tgvf-prob", type=float, default=1.0)
    parser.add_argument("--mask-original-image-after-tgvf-scope", choices=ORIGINAL_IMAGE_MASK_SCOPE_CHOICES, default="evidence_only")
    parser.add_argument("--deepstack-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fast-batched-stage2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--loss-evidence-state", type=float, default=0.2)
    parser.add_argument("--loss-focus-target", type=float, default=1.5)
    parser.add_argument("--loss-evidence", type=float, default=1.0)
    parser.add_argument("--loss-value-span", type=float, default=1.0)
    parser.add_argument("--loss-answer", type=float, default=1.0)
    parser.add_argument("--loss-no-focus-evidence-state", type=float, default=0.2)
    parser.add_argument("--loss-no-focus-answer", type=float, default=1.0)
    parser.add_argument("--loss-visual-token-manifold", type=float, default=0.0)
    parser.add_argument(
        "--matrix-ce-preservation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--loss-same-image-matrix-ce", type=float, default=1.0)
    parser.add_argument("--matrix-ce-group-size", type=int, default=4)
    parser.add_argument("--matrix-ce-readout-batch-size", type=int, default=4)
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
    if args.target_focus_ratio is not None and not (0.0 < args.target_focus_ratio < 1.0):
        parser.error("--target-focus-ratio must be between 0 and 1")
    if args.d_deepstack_enabled and not args.deepstack_enabled:
        parser.error("--d-deepstack-enabled requires --deepstack-enabled")
    if args.deepstack_enabled and not args.fast_batched_stage2:
        parser.error("--deepstack-enabled requires --fast-batched-stage2")
    if args.d_deepstack_enabled and args.variant != "tgvf_v2_bidirectional":
        parser.error("--d-deepstack-enabled currently requires --variant tgvf_v2_bidirectional")
    if args.matrix_ce_preservation and not args.fast_batched_stage2:
        parser.error("--matrix-ce-preservation requires --fast-batched-stage2")
    if args.matrix_ce_preservation and args.loss_same_image_matrix_ce <= 0:
        parser.error("--loss-same-image-matrix-ce must be > 0 when Matrix-CE is enabled")
    if args.matrix_ce_group_size < 2:
        parser.error("--matrix-ce-group-size must be >= 2")
    if args.matrix_ce_readout_batch_size < 1:
        parser.error("--matrix-ce-readout-batch-size must be >= 1")
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
        "train/loss_same_image_matrix_ce": log.get("loss_same_image_matrix_ce"),
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
        "train/target_focus_sampling_ratio": log.get("target_focus_sampling_ratio"),
        "train/world_size": log.get("world_size"),
        "train/special_tokens_added": float(bool(log.get("special_tokens_added"))),
        "train/normal_tokens_added": float(bool(log.get("normal_tokens_added"))),
        "train/tokenizer_resized": float(bool(log.get("tokenizer_resized"))),
        "train/markers_are_plain_text": float(bool(log.get("markers_are_plain_text"))),
        "train/matrix_ce_enabled": float(bool(log.get("matrix_ce_enabled"))),
        "train/same_image_negative_enabled": float(bool(log.get("same_image_negative_enabled"))),
        "train/matrix_ce_top1": log.get("matrix_ce_top1"),
        "train/matrix_ce_positive_negative_margin": log.get(
            "matrix_ce_positive_negative_margin"
        ),
    }
    for key, value in log.items():
        if key.startswith("protocol_c_boundary_acc_") or key.startswith("protocol_c_boundary_support_"):
            metrics[f"train/{key}"] = value
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
