"""Clean-native internal diagnostics for TGVF D/readout quality."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .defaults import DEFAULT_MAX_IMAGE_RESOLUTION, DEFAULT_MODEL_ID, DEFAULT_PROTOCOL
from .schema import _to_jsonable

DIAGNOSTIC_TASKS = ("readout", "query", "distribution")
D_CONDITIONS = (
    "correct_D",
    "no_D",
    "random_D",
    "wrong_same_image_D",
    "wrong_diff_image_D",
)


@dataclass(frozen=True)
class StageDiagnosticConfig:
    run_id: str
    stage: str
    checkpoint: str
    eval_jsonl: str
    output_dir: str
    model_id: str = DEFAULT_MODEL_ID
    processor_id: str | None = None
    protocol: str = DEFAULT_PROTOCOL
    focus_action_im_end: bool = True
    variant: str = "tgvf_v2_bidirectional"
    num_foveated_tokens: str = "none"
    encoder_adapter_type: str = "bidirectional"
    max_image_resolution: int = DEFAULT_MAX_IMAGE_RESOLUTION
    fvt_position_mode: str = "native_source_grid"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    device: str = "cuda:0"
    device_map: str = "cuda:0"
    tasks: tuple[str, ...] = DIAGNOSTIC_TASKS
    readout_max_samples: int = 200
    distribution_max_samples: int = 200
    query_max_groups: int = 50
    query_require_groups: int = 0
    query_min_targets_per_image: int = 3
    eval_workers: int = 1
    use_fvt_cache: bool = False
    stage2_load_lora: bool = True
    seed: int = 20260525
    wandb_log_eval: bool = False
    wandb_project: str | None = None
    wandb_mode: str | None = None
    wandb_run_name: str | None = None
    wandb_group: str | None = None
    wandb_tags: str | None = None
    git_commit: str | None = None
    dirty_worktree: bool | None = None

    def validate(self) -> None:
        if self.stage not in {"stage1", "stage2"}:
            raise ValueError("stage must be 'stage1' or 'stage2'")
        if not self.run_id:
            raise ValueError("run_id is required")
        if not Path(self.checkpoint).is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {self.checkpoint}")
        if not Path(self.eval_jsonl).is_file():
            raise FileNotFoundError(f"eval_jsonl does not exist: {self.eval_jsonl}")
        if not self.output_dir:
            raise ValueError("output_dir is required")
        unknown = sorted(set(self.tasks) - set(DIAGNOSTIC_TASKS))
        if unknown:
            raise ValueError(f"unknown diagnostic tasks: {unknown}")
        if not self.tasks:
            raise ValueError("at least one diagnostic task is required")
        if self.eval_workers < 1:
            raise ValueError("eval_workers must be >= 1")
        if self.readout_max_samples < 1:
            raise ValueError("readout_max_samples must be >= 1")
        if self.distribution_max_samples < 1:
            raise ValueError("distribution_max_samples must be >= 1")
        if self.query_max_groups < 1:
            raise ValueError("query_max_groups must be >= 1")
        if self.query_require_groups < 0:
            raise ValueError("query_require_groups must be >= 0")
        if self.query_min_targets_per_image < 2:
            raise ValueError("query_min_targets_per_image must be >= 2")
        if self.use_fvt_cache:
            raise ValueError("clean-native stage diagnostics do not yet implement FVT cache")
        if self.wandb_log_eval:
            raise ValueError("clean-native stage diagnostics W&B upload is not implemented")

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass
class _Runtime:
    model: Any
    utility_model: Any
    processor: Any
    foveal_module: Any
    device: Any
    model_info: dict[str, Any]


@dataclass
class _DiagnosticItem:
    sample: Any
    uid: str
    target_hidden_states: Any
    pre_merge_visual_tokens: Any
    merged_visual_tokens: Any
    foveated_visual_tokens: Any
    d_deepstack_visual_embeds: Any | None
    capture: Any
    shapes: dict[str, list[int]]
    readout_metadata: dict[str, Any]


def parse_diagnostic_tasks(value: str) -> tuple[str, ...]:
    if value == "all":
        return DIAGNOSTIC_TASKS
    tasks = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(tasks) - set(DIAGNOSTIC_TASKS))
    if unknown:
        raise ValueError(f"unknown diagnostic tasks: {unknown}")
    if not tasks:
        raise ValueError("at least one diagnostic task is required")
    return tasks


def build_stage_diagnostic_plan(config: StageDiagnosticConfig) -> dict[str, Any]:
    config.validate()
    command = stage_diagnostic_command(config)
    return {
        "schema_version": "clean_stage_diagnostic_plan_v2",
        "run_id": config.run_id,
        "eval_family": "internal_diagnostic",
        "stage": config.stage,
        "diagnostic_kind": "stage1_style_fvt_readout_regression",
        "tasks": list(config.tasks),
        "d_conditions": list(D_CONDITIONS),
        "checkpoint": config.checkpoint,
        "eval_jsonl": config.eval_jsonl,
        "output_dir": config.output_dir,
        "model_id": config.model_id,
        "processor_id": config.processor_id,
        "processor_resolution": "checkpoint_config_when_missing",
        "protocol": config.protocol,
        "focus_action_im_end": config.focus_action_im_end,
        "max_image_resolution": config.max_image_resolution,
        "fvt_position_mode": config.fvt_position_mode,
        "mask_original_image_after_tgvf": True,
        "capture_mode": "teacher_forced",
        "use_fvt_cache": config.use_fvt_cache,
        "stage2_load_lora": config.stage2_load_lora,
        "wandb_log_eval": config.wandb_log_eval,
        "git_commit": config.git_commit,
        "dirty_worktree": config.dirty_worktree,
        "execution_backend": {
            "name": "clean_native_stage_diagnostics",
            "version": 1,
            "legacy_bridge": False,
            "forward_semantics": _diagnostic_forward_semantics(config),
            "metric_names": "preserved_v3_readout_query_distribution",
        },
        "command": command,
        "environment": {},
        "expected_reports": expected_stage_diagnostic_reports(config.output_dir, config.tasks),
        "config": config.to_dict(),
    }


def write_stage_diagnostic_plan(output_dir: str | Path, plan: dict[str, Any]) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "clean_stage_diagnostic_plan.json"
    command_path = out / "clean_stage_diagnostic_command.sh"
    _write_json(plan_path, plan)
    command_path.write_text(_shell_script(plan), encoding="utf-8")
    command_path.chmod(0o755)
    return {
        "schema_version": "clean_stage_diagnostic_plan_artifacts_v1",
        "plan": str(plan_path),
        "command": str(command_path),
        "output_dir": str(out),
    }


def run_stage_diagnostics(plan: dict[str, Any], *, cwd: str | Path | None = None) -> dict[str, Any]:
    del cwd
    output_dir = Path(str(plan["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "clean_stage_diagnostic_status.json"
    started = {
        "schema_version": "clean_stage_diagnostic_status_v2",
        "status": "running",
        "run_id": plan["run_id"],
        "stage": plan["stage"],
        "tasks": plan["tasks"],
        "execution_backend": plan.get("execution_backend"),
        "plan_path": str(output_dir / "clean_stage_diagnostic_plan.json"),
    }
    _write_json(status_path, started)
    try:
        execution = execute_stage_diagnostic_tasks(plan)
    except Exception as exc:
        failed = {
            **started,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "reports": report_statuses(plan),
        }
        _write_json(status_path, failed)
        raise
    finished = {
        **started,
        "status": "completed",
        "reports": report_statuses(plan),
        "execution": execution,
    }
    _write_json(status_path, finished)
    return finished


def execute_stage_diagnostic_tasks(plan: dict[str, Any]) -> dict[str, Any]:
    config = _config_from_plan(plan)
    config.validate()
    _set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = _load_samples(config)
    if not samples:
        raise RuntimeError("No focus samples found for clean stage diagnostics.")
    runtime = _load_runtime(config, example_sample=samples[0])
    _write_json(output_dir / "runtime_config.json", _runtime_config(config, runtime.model_info))

    items = _build_diagnostic_items(config, runtime, samples)
    summaries: dict[str, Any] = {
        "num_samples_loaded": len(samples),
        "num_items_built": len(items),
        "runtime": runtime.model_info,
    }
    if "readout" in config.tasks:
        summaries["readout"] = _run_readout(config, runtime, items)
    if "query" in config.tasks:
        summaries["query"] = _run_query(config, runtime, items)
    if "distribution" in config.tasks:
        summaries["distribution"] = _run_distribution(config, runtime, items)
    return summaries


def stage_diagnostic_command(config: StageDiagnosticConfig) -> list[str]:
    command = [
        "python",
        "-m",
        "revisit_vlm_clean.cli.stage_diagnostics",
        "--run-id",
        config.run_id,
        "--stage",
        config.stage,
        "--checkpoint",
        config.checkpoint,
        "--eval-jsonl",
        config.eval_jsonl,
        "--output-dir",
        config.output_dir,
        "--model-id",
        config.model_id,
        "--protocol",
        config.protocol,
        "--variant",
        config.variant,
        "--num-foveated-tokens",
        str(config.num_foveated_tokens),
        "--encoder-adapter-type",
        config.encoder_adapter_type,
        "--max-image-resolution",
        str(config.max_image_resolution),
        "--fvt-position-mode",
        config.fvt_position_mode,
        "--dtype",
        config.dtype,
        "--attn-implementation",
        config.attn_implementation,
        "--device",
        config.device,
        "--device-map",
        config.device_map,
        "--tasks",
        ",".join(config.tasks),
        "--readout-max-samples",
        str(config.readout_max_samples),
        "--distribution-max-samples",
        str(config.distribution_max_samples),
        "--query-max-groups",
        str(config.query_max_groups),
        "--query-require-groups",
        str(config.query_require_groups),
        "--query-min-targets-per-image",
        str(config.query_min_targets_per_image),
        "--eval-workers",
        str(config.eval_workers),
        "--stage2-load-lora" if config.stage2_load_lora else "--no-stage2-load-lora",
        "--seed",
        str(config.seed),
        "--execute",
    ]
    if config.processor_id:
        command.extend(["--processor-id", config.processor_id])
    command.append("--focus-action-im-end" if config.focus_action_im_end else "--no-focus-action-im-end")
    command.append("--no-use-fvt-cache")
    return command


def expected_stage_diagnostic_reports(output_dir: str | Path, tasks: tuple[str, ...]) -> dict[str, str]:
    out = Path(output_dir)
    reports: dict[str, str] = {}
    if "readout" in tasks:
        reports["readout"] = str(out / "readout" / "readout_eval_report.json")
    if "query" in tasks:
        reports["query"] = str(out / "query_sensitivity" / "query_sensitivity_report.json")
    if "distribution" in tasks:
        reports["distribution"] = str(out / "fvt_distribution" / "fvt_distribution_report.json")
    return reports


def report_statuses(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    reports = {}
    for name, path in (plan.get("expected_reports") or {}).items():
        report_path = Path(path)
        reports[name] = {
            "path": str(report_path),
            "exists": report_path.exists(),
            "size_bytes": report_path.stat().st_size if report_path.exists() else None,
        }
    return reports


def _config_from_plan(plan: dict[str, Any]) -> StageDiagnosticConfig:
    data = dict(plan.get("config") or {})
    if not data:
        data = {
            "run_id": plan["run_id"],
            "stage": plan["stage"],
            "checkpoint": plan["checkpoint"],
            "eval_jsonl": plan["eval_jsonl"],
            "output_dir": plan["output_dir"],
            "model_id": plan.get("model_id", DEFAULT_MODEL_ID),
            "processor_id": plan.get("processor_id"),
            "protocol": plan.get("protocol", DEFAULT_PROTOCOL),
            "focus_action_im_end": bool(plan.get("focus_action_im_end", True)),
            "max_image_resolution": int(plan.get("max_image_resolution", DEFAULT_MAX_IMAGE_RESOLUTION)),
            "fvt_position_mode": plan.get("fvt_position_mode", "native_source_grid"),
            "tasks": tuple(plan.get("tasks") or DIAGNOSTIC_TASKS),
        }
    data["tasks"] = tuple(data.get("tasks") or DIAGNOSTIC_TASKS)
    return StageDiagnosticConfig(**data)


def _load_samples(config: StageDiagnosticConfig) -> list[Any]:
    from revisit_vlm.tgvf_v3_stage1 import TGVFv3Stage1Dataset

    max_samples = max(
        config.readout_max_samples if {"readout", "query"} & set(config.tasks) else 0,
        config.distribution_max_samples if "distribution" in config.tasks else 0,
    )
    dataset = TGVFv3Stage1Dataset(config.eval_jsonl, focus_only=True)
    return list(dataset.samples[:max_samples])


def _load_runtime(config: StageDiagnosticConfig, *, example_sample: Any) -> _Runtime:
    import torch

    from revisit_vlm.qwen3_vl_tgvf import ensure_tgvf_protocol_tokens, load_qwen3_vl
    from revisit_vlm.tgvf_training import build_tgvf_module
    from revisit_vlm.tgvf_v3_stage1 import freeze_qwen_backbone, infer_qwen3_stage1_dims

    checkpoint = torch.load(config.checkpoint, map_location="cpu")
    checkpoint_config = checkpoint.get("config") or {}
    checkpoint_protocol = checkpoint_config.get("tgvf_protocol")
    if checkpoint_protocol and checkpoint_protocol != config.protocol:
        raise ValueError(
            "checkpoint protocol mismatch: "
            f"checkpoint={checkpoint_protocol!r} requested={config.protocol!r}"
        )
    processor_id = config.processor_id or checkpoint_config.get("processor_id")
    loaded = load_qwen3_vl(
        config.model_id,
        processor_id=processor_id,
        dtype=config.dtype,
        device_map=_resolve_device_map(config.device_map),
        attn_implementation=config.attn_implementation,
        trust_remote_code=True,
    )
    base_model = loaded.model
    processor = loaded.processor
    if getattr(processor.tokenizer, "pad_token", None) is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    protocol_token_info = ensure_tgvf_protocol_tokens(
        processor.tokenizer,
        base_model,
        protocol=config.protocol,
    )
    freeze_qwen_backbone(base_model)
    token_row_info = _restore_protocol_token_rows_from_checkpoint(
        model=base_model,
        tokenizer=processor.tokenizer,
        protocol=config.protocol,
        checkpoint=checkpoint,
    )

    model = base_model
    lora_info: dict[str, Any] = {
        "loaded": False,
        "requested": config.stage2_load_lora,
        "available_in_checkpoint": "qwen_lora" in checkpoint,
    }
    if config.stage2_load_lora and "qwen_lora" in checkpoint:
        model, lora_info = _load_stage2_lora_model(
            base_model=base_model,
            checkpoint=checkpoint,
            protocol_token_info=protocol_token_info,
        )
        lora_info["requested"] = True
    elif "qwen_lora" in checkpoint:
        lora_info["reason"] = "stage2_load_lora_disabled"
    else:
        lora_info["reason"] = "qwen_lora_absent"
    model.eval()
    utility_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    device = _resolve_runtime_device(torch, config.device)
    dims = infer_qwen3_stage1_dims(
        model=utility_model,
        processor=processor,
        sample=example_sample,
        device=device,
        max_image_resolution=config.max_image_resolution,
    )
    tgvf_cfg = _resolved_tgvf_module_config(
        checkpoint=checkpoint,
        config=config,
        dims=dims,
    )
    foveal_module = build_tgvf_module(
        variant=tgvf_cfg["variant"],
        d_lm=dims["d_lm"],
        d_v=dims["d_v"],
        num_foveated_tokens=tgvf_cfg["num_foveated_tokens"],
        spatial_merge_size=tgvf_cfg["spatial_merge_size"],
        attn_dim=tgvf_cfg["attn_dim"],
        encoder_adapter_layers=tgvf_cfg["encoder_adapter_layers"],
        encoder_adapter_type=tgvf_cfg["encoder_adapter_type"],
        encoder_adapter_gate_init=tgvf_cfg["encoder_adapter_gate_init"],
        encoder_adapter_share_weights=tgvf_cfg["encoder_adapter_share_weights"],
        encoder_adapter_layer_index_base=tgvf_cfg["encoder_adapter_layer_index_base"],
        encoder_reencode_deepstack_compatible=tgvf_cfg[
            "encoder_reencode_deepstack_compatible"
        ],
        d_deepstack_enabled=tgvf_cfg["d_deepstack_enabled"],
        d_deepstack_branch_layers=tgvf_cfg["d_deepstack_branch_layers"],
    ).to(device=device, dtype=next(model.parameters()).dtype)
    if "tgvf_module" not in checkpoint:
        raise KeyError("checkpoint is missing tgvf_module")
    foveal_module.load_state_dict(checkpoint["tgvf_module"], strict=True)
    foveal_module.eval()
    return _Runtime(
        model=model,
        utility_model=utility_model,
        processor=processor,
        foveal_module=foveal_module,
        device=device,
        model_info={
            "model_id": config.model_id,
            "processor_id": processor_id or config.model_id,
            "checkpoint_global_step": checkpoint.get("global_step"),
            "checkpoint_optimizer_step": checkpoint.get("optimizer_step"),
            "checkpoint_micro_step": checkpoint.get("micro_step"),
            "checkpoint_stage": checkpoint_config.get("stage"),
            "checkpoint_has_qwen_lora": "qwen_lora" in checkpoint,
            "qwen_lora": lora_info,
            "protocol_token_info": protocol_token_info,
            "checkpoint_protocol_token_rows": token_row_info,
            "d_lm": dims["d_lm"],
            "d_v": dims["d_v"],
            "spatial_merge_size": dims["spatial_merge_size"],
            "tgvf_module_config": tgvf_cfg,
            "forward_semantics": _diagnostic_forward_semantics(config),
        },
    )


def _load_stage2_lora_model(
    *,
    base_model: Any,
    checkpoint: dict[str, Any],
    protocol_token_info: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from revisit_vlm_clean.peft_token_rows import trainable_token_indices_from_checkpoint

    checkpoint_config = checkpoint.get("config") or {}
    lora_cfg = checkpoint_config.get("lora") or {}
    qwen_lora_state = checkpoint["qwen_lora"]
    checkpoint_has_trainable_token_adapter = any(
        "trainable_tokens" in key or "token_adapter" in key for key in qwen_lora_state
    )
    modules_to_save = lora_cfg.get("modules_to_save")
    protocol_token_ids = list(
        protocol_token_info.get("tgvf_protocol_token_ids", {}).values()
    )
    trainable_token_indices = trainable_token_indices_from_checkpoint(
        state=qwen_lora_state,
        token_ids=protocol_token_ids,
    )
    peft_config = LoraConfig(
        r=int(lora_cfg.get("rank", 64)),
        lora_alpha=int(lora_cfg.get("alpha", 256)),
        target_modules=list(
            lora_cfg.get("target_modules")
            or "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj".split(",")
        ),
        lora_dropout=float(lora_cfg.get("dropout", 0.0)),
        bias=str(lora_cfg.get("bias", "none")),
        task_type="CAUSAL_LM",
        modules_to_save=list(modules_to_save) if modules_to_save else None,
        ensure_weight_tying=False,
        trainable_token_indices=(
            trainable_token_indices
            if checkpoint_has_trainable_token_adapter and not modules_to_save
            else None
        ),
    )
    model = get_peft_model(base_model, peft_config)
    load_result = set_peft_model_state_dict(model, qwen_lora_state)
    _validate_peft_load_result(load_result)
    return model, {
        "loaded": True,
        "available_in_checkpoint": True,
        "modules_to_save": modules_to_save,
        "target_modules": sorted(peft_config.target_modules or []),
        "checkpoint_has_trainable_token_adapter": checkpoint_has_trainable_token_adapter,
    }


def _validate_peft_load_result(load_result: Any) -> None:
    unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
    missing = list(getattr(load_result, "missing_keys", []) or [])
    adapter_markers = ("lora_", "modules_to_save", "trainable_tokens_delta")
    adapter_missing = [key for key in missing if any(marker in key for marker in adapter_markers)]
    if unexpected or adapter_missing:
        raise RuntimeError(
            "Incomplete Stage2 LoRA load: "
            f"unexpected_keys={unexpected[:20]} "
            f"adapter_missing_keys={adapter_missing[:20]}"
        )


def _restore_protocol_token_rows_from_checkpoint(
    *,
    model: Any,
    tokenizer: Any,
    protocol: str,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    import torch

    from revisit_vlm.qwen3_vl_tgvf import protocol_special_token_ids, protocol_special_tokens

    tokens = protocol_special_tokens(protocol)
    current_ids = protocol_special_token_ids(tokenizer, protocol=protocol)
    info: dict[str, Any] = {
        "available_in_checkpoint": "protocol_c_token_rows" in checkpoint,
        "loaded": False,
        "protocol": protocol,
        "tokens": list(tokens),
        "current_token_ids": {token: int(current_ids[token]) for token in tokens},
    }
    payload = checkpoint.get("protocol_c_token_rows")
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
        input_embed.weight[ordered_ids].copy_(
            input_rows.to(device=input_embed.weight.device, dtype=input_embed.weight.dtype)
        )
        if output_rows is not None and output_embed is not None and hasattr(output_embed, "weight"):
            output_embed.weight[ordered_ids].copy_(
                output_rows.to(device=output_embed.weight.device, dtype=output_embed.weight.dtype)
            )
    info["loaded"] = True
    info["reason"] = "loaded_from_checkpoint"
    info["loaded_input_rows"] = True
    info["loaded_output_rows"] = output_rows is not None
    return info


def _resolved_tgvf_module_config(
    *,
    checkpoint: dict[str, Any],
    config: StageDiagnosticConfig,
    dims: dict[str, int],
) -> dict[str, Any]:
    checkpoint_config = checkpoint.get("config") or {}
    tgvf_cfg = checkpoint_config.get("tgvf") or {}
    return {
        "variant": str(tgvf_cfg.get("variant") or config.variant),
        "num_foveated_tokens": _optional_positive_int(
            tgvf_cfg.get("num_foveated_tokens", config.num_foveated_tokens)
        ),
        "spatial_merge_size": int(tgvf_cfg.get("spatial_merge_size") or dims["spatial_merge_size"]),
        "attn_dim": tgvf_cfg.get("attn_dim"),
        "encoder_adapter_layers": tuple(
            int(value) for value in (tgvf_cfg.get("encoder_adapter_layers") or (8, 16, 24))
        ),
        "encoder_adapter_type": str(
            tgvf_cfg.get("encoder_adapter_type") or config.encoder_adapter_type
        ),
        "encoder_adapter_gate_init": float(tgvf_cfg.get("encoder_adapter_gate_init", 0.0)),
        "encoder_adapter_share_weights": bool(
            tgvf_cfg.get("encoder_adapter_share_weights", False)
        ),
        "encoder_adapter_layer_index_base": int(
            tgvf_cfg.get("encoder_adapter_layer_index_base", 0)
        ),
        "encoder_reencode_deepstack_compatible": bool(
            tgvf_cfg.get("encoder_reencode_deepstack_compatible", False)
        ),
        "d_deepstack_enabled": bool(tgvf_cfg.get("d_deepstack_enabled", False)),
        "d_deepstack_branch_layers": tuple(
            int(value) for value in (tgvf_cfg.get("d_deepstack_branch_layers") or (8, 16, 24))
        ),
    }


def _build_diagnostic_items(
    config: StageDiagnosticConfig,
    runtime: _Runtime,
    samples: list[Any],
) -> list[_DiagnosticItem]:
    import torch

    from revisit_vlm.qwen3_vl_tgvf import (
        NEED_LOCAL_EVIDENCE,
        tap_qwen3_vision_features_with_deepstack_premerge,
    )
    from revisit_vlm.tgvf_foveal import finalize_tgvf_output_with_frozen_qwen_merger
    from revisit_vlm.tgvf_v3_stage1 import (
        capture_v3_stage1_focus_teacher_forced,
        prepare_v3_stage1_readout_inputs,
    )

    items: list[_DiagnosticItem] = []
    vision_cache: dict[str, tuple[Any, Any, Any, list[Any], list[Any]]] = {}
    with torch.no_grad():
        for index, sample in enumerate(samples):
            image_input = _image_input(sample.image, max_image_resolution=config.max_image_resolution)
            capture = capture_v3_stage1_focus_teacher_forced(
                model=runtime.model,
                processor=runtime.processor,
                image=image_input,
                question=sample.prompt_question,
                target=sample.target,
                device=runtime.device,
                hidden_state_index=-1,
                protocol=config.protocol,
                append_im_end=config.focus_action_im_end,
            )
            if not capture.capture_found:
                raise RuntimeError(f"forced focus span was not captured for sample {index}")
            cache_key = f"{sample.image}|{config.max_image_resolution}"
            if cache_key not in vision_cache:
                vision_cache[cache_key] = tap_qwen3_vision_features_with_deepstack_premerge(
                    runtime.utility_model,
                    runtime.processor,
                    image=image_input,
                    question=sample.prompt_question,
                    device=runtime.device,
                )
            tap, v_pre, v_merge, deepstack_pre, _deepstack_features = vision_cache[cache_key]
            if v_pre is None:
                raise RuntimeError(f"Qwen vision V_pre tap failed: {tap.errors}")
            if v_merge is None:
                raise RuntimeError(f"Qwen vision V_merge tap failed: {tap.errors}")
            output = runtime.foveal_module(
                target_hidden_states=capture.target_hidden_states.to(runtime.device),
                pre_merge_visual_tokens=v_pre.to(runtime.device),
                metadata={
                    "target": sample.target,
                    "stage": "clean_stage_diagnostics",
                    "evidence_state": NEED_LOCAL_EVIDENCE,
                    "qwen_model": runtime.utility_model,
                    "processor": runtime.processor,
                    "image": image_input,
                    "question": sample.prompt_question,
                    "device": runtime.device,
                    "deepstack_pre_merge_visual_tokens": [
                        item.to(runtime.device) for item in deepstack_pre
                    ],
                },
            )
            output = finalize_tgvf_output_with_frozen_qwen_merger(runtime.utility_model, output)
            d = output.foveated_visual_tokens.detach()
            d_deepstack = output.deepstack_visual_embeds
            readout_inputs = prepare_v3_stage1_readout_inputs(
                model=runtime.model,
                tokenizer_or_processor=runtime.processor,
                capture=capture,
                evidence_description=sample.evidence_description,
                foveated_visual_tokens=d,
                merged_visual_tokens=v_merge.to(runtime.device),
                d_deepstack_visual_embeds=d_deepstack,
                device=runtime.device,
                mask_original_image_after_tgvf=True,
                position_mode=config.fvt_position_mode,
                protocol=config.protocol,
                focus_action_im_end=config.focus_action_im_end,
            )
            items.append(
                _DiagnosticItem(
                    sample=sample,
                    uid=_sample_uid(sample),
                    target_hidden_states=capture.target_hidden_states.detach().cpu(),
                    pre_merge_visual_tokens=v_pre.detach().cpu(),
                    merged_visual_tokens=v_merge.detach().cpu(),
                    foveated_visual_tokens=d.detach().cpu(),
                    d_deepstack_visual_embeds=None
                    if not d_deepstack
                    else [item.detach().cpu() for item in d_deepstack],
                    capture=_cpu_capture(capture),
                    shapes={
                        "H_q": list(capture.target_hidden_states.shape),
                        "V_pre": list(v_pre.shape),
                        "V_merge": list(v_merge.shape),
                        "D": list(d.shape),
                        "D_deepstack": []
                        if not d_deepstack
                        else [list(item.shape) for item in d_deepstack],
                    },
                    readout_metadata={
                        "mask_mode": readout_inputs.get("mask_mode"),
                        "position_mode": readout_inputs.get("position_mode"),
                        "position_ids_source": readout_inputs.get("position_ids_source"),
                        "original_image_embeds_replaced": readout_inputs.get(
                            "original_image_embeds_replaced"
                        ),
                        "original_image_token_count": readout_inputs.get(
                            "original_image_token_count"
                        ),
                        "blocked_original_image_keys_for_post_tgvf": readout_inputs.get(
                            "blocked_original_image_keys_for_post_tgvf"
                        ),
                        "pre_tgvf_queries_keep_original_image_keys": readout_inputs.get(
                            "pre_tgvf_queries_keep_original_image_keys"
                        ),
                        "second_full_forward_used": False,
                        "tgvf_protocol": config.protocol,
                        "focus_action_im_end": bool(config.focus_action_im_end),
                        "qwen_lora_loaded": bool(
                            runtime.model_info.get("qwen_lora", {}).get("loaded")
                        ),
                        "d_deepstack_visual_embeds_used": bool(d_deepstack),
                        "d_deepstack_visual_embed_shapes": []
                        if not d_deepstack
                        else [list(item.shape) for item in d_deepstack],
                    },
                )
            )
    return items


def _run_readout(
    config: StageDiagnosticConfig,
    runtime: _Runtime,
    items: list[_DiagnosticItem],
) -> dict[str, Any]:
    import torch

    output_dir = Path(config.output_dir) / "readout"
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_items = items[: config.readout_max_samples]
    groups = _group_indices_by_image(scored_items)
    rows = []
    with torch.no_grad():
        for index, item in enumerate(scored_items):
            sample = item.sample
            d = item.foveated_visual_tokens
            same_index = _same_image_wrong_index(groups, item, index)
            diff_index = _different_image_index(scored_items, index)
            random_d = _random_d_like(d, reference=item.merged_visual_tokens, seed=config.seed + index)

            nll_correct = _compute_readout_nll(
                config=config,
                runtime=runtime,
                item=item,
                foveated_visual_tokens=d,
                d_deepstack_visual_embeds=item.d_deepstack_visual_embeds,
            )
            nll_target_only = _compute_readout_nll(
                config=config,
                runtime=runtime,
                item=item,
                foveated_visual_tokens=None,
            )
            nll_random = _compute_readout_nll(
                config=config,
                runtime=runtime,
                item=item,
                foveated_visual_tokens=random_d,
                d_deepstack_visual_embeds=None,
            )
            nll_wrong_same = None
            if same_index is not None and _can_score_fvt_for_item(
                item,
                scored_items[same_index].foveated_visual_tokens,
            ):
                nll_wrong_same = _compute_readout_nll(
                    config=config,
                    runtime=runtime,
                    item=item,
                    foveated_visual_tokens=scored_items[same_index].foveated_visual_tokens,
                    d_deepstack_visual_embeds=scored_items[
                        same_index
                    ].d_deepstack_visual_embeds,
                )
            nll_wrong_diff = None
            if diff_index is not None and _can_score_fvt_for_item(
                item,
                scored_items[diff_index].foveated_visual_tokens,
            ):
                nll_wrong_diff = _compute_readout_nll(
                    config=config,
                    runtime=runtime,
                    item=item,
                    foveated_visual_tokens=scored_items[diff_index].foveated_visual_tokens,
                    d_deepstack_visual_embeds=scored_items[
                        diff_index
                    ].d_deepstack_visual_embeds,
                )
            nlls = {
                "correct_D_plus_target": nll_correct["avg_nll"],
                "target_only_no_D": nll_target_only["avg_nll"],
                "random_D_plus_target": nll_random["avg_nll"],
                "wrong_D_same_image_plus_target": None
                if nll_wrong_same is None
                else nll_wrong_same["avg_nll"],
                "wrong_D_different_image_plus_target": None
                if nll_wrong_diff is None
                else nll_wrong_diff["avg_nll"],
            }
            rows.append(
                {
                    **_sample_metadata_row(sample),
                    "nlls": nlls,
                    "delta_correct_vs_target_only": nlls["target_only_no_D"]
                    - nlls["correct_D_plus_target"],
                    "delta_correct_vs_random": nlls["random_D_plus_target"]
                    - nlls["correct_D_plus_target"],
                    "delta_correct_vs_wrong_same": None
                    if nlls["wrong_D_same_image_plus_target"] is None
                    else nlls["wrong_D_same_image_plus_target"] - nlls["correct_D_plus_target"],
                    "delta_correct_vs_wrong_diff": None
                    if nlls["wrong_D_different_image_plus_target"] is None
                    else nlls["wrong_D_different_image_plus_target"]
                    - nlls["correct_D_plus_target"],
                    "condition_availability": {
                        "wrong_same_image": nll_wrong_same is not None,
                        "wrong_different_image": nll_wrong_diff is not None,
                    },
                    "shapes": item.shapes,
                    "readout_metadata": item.readout_metadata,
                }
            )

    metric_keys = [
        "delta_correct_vs_target_only",
        "delta_correct_vs_random",
        "delta_correct_vs_wrong_same",
        "delta_correct_vs_wrong_diff",
    ]
    report = {
        "config": _runtime_config(config, runtime.model_info),
        "num_samples_attempted": len(scored_items),
        "num_samples_evaluated": len(rows),
        "metrics": {
            "mean_nll_correct_D": _mean(row["nlls"]["correct_D_plus_target"] for row in rows),
            "median_nll_correct_D": _median(row["nlls"]["correct_D_plus_target"] for row in rows),
            "mean_nll_target_only": _mean(row["nlls"]["target_only_no_D"] for row in rows),
            "mean_nll_random_D": _mean(row["nlls"]["random_D_plus_target"] for row in rows),
            "mean_delta_correct_vs_target_only": _mean(
                row["delta_correct_vs_target_only"] for row in rows
            ),
            "mean_delta_correct_vs_random": _mean(row["delta_correct_vs_random"] for row in rows),
            "mean_delta_correct_vs_wrong_same": _mean(
                row["delta_correct_vs_wrong_same"] for row in rows
            ),
            "mean_delta_correct_vs_wrong_diff": _mean(
                row["delta_correct_vs_wrong_diff"] for row in rows
            ),
            "pct_correct_D_beats_target_only": _pct_positive(
                row["delta_correct_vs_target_only"] for row in rows
            ),
            "pct_correct_D_beats_random": _pct_positive(
                row["delta_correct_vs_random"] for row in rows
            ),
            "pct_correct_D_beats_wrong_same": _pct_positive(
                row["delta_correct_vs_wrong_same"] for row in rows
            ),
            "pct_correct_D_beats_wrong_diff": _pct_positive(
                row["delta_correct_vs_wrong_diff"] for row in rows
            ),
            "second_full_forward_used_any": False,
        },
        "metrics_by_evidence_type": _grouped_means(
            rows,
            group_key="evidence_type",
            metric_keys=metric_keys,
        ),
        "metrics_by_source_profile": _grouped_means(
            rows,
            group_key="source_profile",
            metric_keys=metric_keys,
        ),
        "metrics_by_answer_type": _grouped_means(
            rows,
            group_key="answer_type",
            metric_keys=metric_keys,
        ),
        "metrics_by_visual_difficulty": _grouped_means(
            rows,
            group_key="visual_difficulty",
            metric_keys=metric_keys,
        ),
        "position_mode": config.fvt_position_mode,
        "mask_original_image_after_tgvf": True,
    }
    _write_json(output_dir / "readout_eval_report.json", report)
    _write_jsonl(output_dir / "per_sample_results.jsonl", rows)
    _save_summary(
        output_dir / "readout_eval_summary.txt",
        "Clean TGVF Readout Evaluation",
        [
            f"samples_evaluated: {len(rows)}",
            f"mean_nll_correct_D: {report['metrics']['mean_nll_correct_D']}",
            f"mean_nll_target_only: {report['metrics']['mean_nll_target_only']}",
            f"mean_delta_correct_vs_target_only: {report['metrics']['mean_delta_correct_vs_target_only']}",
            f"mean_delta_correct_vs_wrong_same: {report['metrics']['mean_delta_correct_vs_wrong_same']}",
            f"pct_correct_D_beats_target_only: {report['metrics']['pct_correct_D_beats_target_only']}",
            f"pct_correct_D_beats_wrong_same: {report['metrics']['pct_correct_D_beats_wrong_same']}",
            f"position_mode: {config.fvt_position_mode}",
            "mask_original_image_after_tgvf: True",
            "execution_backend: clean_native_stage_diagnostics",
        ],
    )
    return {"report": str(output_dir / "readout_eval_report.json"), "num_rows": len(rows)}


def _run_query(
    config: StageDiagnosticConfig,
    runtime: _Runtime,
    items: list[_DiagnosticItem],
) -> dict[str, Any]:
    import torch

    output_dir = Path(config.output_dir) / "query_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_items = items[: config.readout_max_samples]
    groups = {
        group_id: indices
        for group_id, indices in _group_indices_by_image(scored_items).items()
        if len(indices) >= config.query_min_targets_per_image
    }
    num_eligible_groups = len(groups)
    if config.query_require_groups and num_eligible_groups < config.query_require_groups:
        raise RuntimeError(
            f"Need at least {config.query_require_groups} image groups with "
            f">= {config.query_min_targets_per_image} targets, found {num_eligible_groups}"
        )
    groups = dict(list(groups.items())[: config.query_max_groups])
    group_rows = []
    per_item_rows = []
    with torch.no_grad():
        for group_id, indices in groups.items():
            group_items = [scored_items[index] for index in indices]
            n = len(group_items)
            nll_matrix = []
            for row_item in group_items:
                row = []
                for col_item in group_items:
                    if not _can_score_fvt_for_item(row_item, col_item.foveated_visual_tokens):
                        row.append(float("inf"))
                        continue
                    nll = _compute_readout_nll(
                        config=config,
                        runtime=runtime,
                        item=row_item,
                        foveated_visual_tokens=col_item.foveated_visual_tokens,
                        d_deepstack_visual_embeds=col_item.d_deepstack_visual_embeds,
                    )
                    row.append(float(nll["avg_nll"]))
                nll_matrix.append(row)

            diagonal_ranks = []
            diagonal_gaps = []
            top1 = []
            top2 = []
            for i, row in enumerate(nll_matrix):
                ranked = sorted(range(n), key=lambda j: row[j])
                rank = ranked.index(i) + 1
                diagonal_ranks.append(rank)
                top1.append(float(rank == 1))
                top2.append(float(rank <= 2))
                wrong = [row[j] for j in range(n) if j != i and row[j] != float("inf")]
                gap = min(wrong) - row[i] if wrong and row[i] != float("inf") else None
                diagonal_gaps.append(gap)
                per_item_rows.append(
                    {
                        **_sample_metadata_row(group_items[i].sample),
                        "diagonal_rank": rank,
                        "diagonal_gap": gap,
                        "top1": float(rank == 1),
                        "top2": float(rank <= 2),
                        "group_size": n,
                        "shape_compatible_group": all(
                            _can_score_fvt_for_item(group_items[i], other.foveated_visual_tokens)
                            for other in group_items
                        ),
                    }
                )
            group_rows.append(
                {
                    "stable_image_uid": group_id,
                    "group_size": n,
                    "targets": [item.sample.target for item in group_items],
                    "evidence_descriptions": [
                        item.sample.evidence_description for item in group_items
                    ],
                    "evidence_types": [item.sample.evidence_type for item in group_items],
                    "score_type": "nll_lower_is_better",
                    "nll_matrix": nll_matrix,
                    "diagonal_ranks": diagonal_ranks,
                    "diagonal_gaps": diagonal_gaps,
                    "top1_accuracy": _mean(top1),
                    "top2_accuracy": _mean(top2),
                    "mrr": _mean(1.0 / rank for rank in diagonal_ranks),
                    "mean_diagonal_gap": _mean(diagonal_gaps),
                    "median_diagonal_gap": _median(diagonal_gaps),
                }
            )

    metric_keys = ["top1", "top2", "diagonal_gap"]
    report = {
        "config": _runtime_config(config, runtime.model_info),
        "num_samples_attempted": len(scored_items),
        "num_eligible_groups": num_eligible_groups,
        "num_groups_evaluated": len(group_rows),
        "num_items_evaluated": len(per_item_rows),
        "score_type": "nll_lower_is_better",
        "metrics": {
            "retrieval_top1": _mean(row["top1"] for row in per_item_rows),
            "retrieval_top2": _mean(row["top2"] for row in per_item_rows),
            "mrr": _mean(1.0 / row["diagonal_rank"] for row in per_item_rows),
            "mean_diagonal_gap": _mean(row["diagonal_gap"] for row in per_item_rows),
            "median_diagonal_gap": _median(row["diagonal_gap"] for row in per_item_rows),
            "second_full_forward_used_any": False,
        },
        "metrics_by_evidence_type": _grouped_means(
            per_item_rows,
            group_key="evidence_type",
            metric_keys=metric_keys,
        ),
        "metrics_by_source_profile": _grouped_means(
            per_item_rows,
            group_key="source_profile",
            metric_keys=metric_keys,
        ),
        "position_mode": config.fvt_position_mode,
        "mask_original_image_after_tgvf": True,
    }
    _write_json(output_dir / "query_sensitivity_report.json", report)
    _write_jsonl(output_dir / "per_group_results.jsonl", group_rows)
    _write_jsonl(output_dir / "per_item_results.jsonl", per_item_rows)
    _save_summary(
        output_dir / "query_sensitivity_summary.txt",
        "Clean TGVF Query Sensitivity Evaluation",
        [
            f"groups_evaluated: {len(group_rows)}",
            f"items_evaluated: {len(per_item_rows)}",
            f"retrieval_top1: {report['metrics']['retrieval_top1']}",
            f"mrr: {report['metrics']['mrr']}",
            f"mean_diagonal_gap: {report['metrics']['mean_diagonal_gap']}",
            f"position_mode: {config.fvt_position_mode}",
            "score_type: nll_lower_is_better",
            "execution_backend: clean_native_stage_diagnostics",
        ],
    )
    return {
        "report": str(output_dir / "query_sensitivity_report.json"),
        "num_groups": len(group_rows),
        "num_rows": len(per_item_rows),
    }


def _run_distribution(
    config: StageDiagnosticConfig,
    runtime: _Runtime,
    items: list[_DiagnosticItem],
) -> dict[str, Any]:
    import torch

    from revisit_vlm.tgvf_training import visual_token_manifold_loss

    output_dir = Path(config.output_dir) / "fvt_distribution"
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_items = items[: config.distribution_max_samples]
    rows = []
    for item in scored_items:
        sample = item.sample
        d = item.foveated_visual_tokens.float()
        v = item.merged_visual_tokens.float()
        d_stats = _tensor_distribution_stats(d)
        v_stats = _tensor_distribution_stats(v)
        manifold_active = d.ndim >= 2 and v.ndim >= 2 and d.shape[-1] == v.shape[-1]
        man_loss = visual_token_manifold_loss(d, v) if manifold_active else None
        mean_mse = None
        std_mse = None
        pooled_cosine = None
        if manifold_active:
            mean_mse = float(torch.nn.functional.mse_loss(d.mean(dim=0), v.mean(dim=0)).item())
            std_mse = float(
                torch.nn.functional.mse_loss(
                    d.std(dim=0, unbiased=False),
                    v.std(dim=0, unbiased=False),
                ).item()
            )
            pooled_cosine = torch.nn.functional.cosine_similarity(
                d.mean(dim=0).view(1, -1),
                v.mean(dim=0).view(1, -1),
            ).item()
        norm_ratio = None
        if v_stats["mean_token_norm"] and v_stats["mean_token_norm"] != 0:
            norm_ratio = d_stats["mean_token_norm"] / v_stats["mean_token_norm"]
        token_cos = torch.nn.functional.cosine_similarity(
            d.float(),
            d.float().mean(dim=0, keepdim=True),
            dim=-1,
        )
        rows.append(
            {
                **_sample_metadata_row(sample),
                "D_stats": d_stats,
                "V_merge_stats": v_stats,
                "manifold_active": bool(manifold_active),
                "manifold_loss": None if man_loss is None else float(man_loss.detach().cpu()),
                "mean_mse": mean_mse,
                "std_mse": std_mse,
                "norm_ratio_D_to_Vmerge": norm_ratio,
                "pooled_D_to_pooled_Vmerge_cosine": pooled_cosine,
                "finite": bool(torch.isfinite(d).all().item() and torch.isfinite(v).all().item()),
                "collapse_near_identical_tokens": float(token_cos.std(unbiased=False).item() < 1e-4),
                "shapes": item.shapes,
                "readout_metadata": item.readout_metadata,
            }
        )
    finite_rate = _mean(1.0 if row["finite"] else 0.0 for row in rows)
    collapse_rate = _mean(row["collapse_near_identical_tokens"] for row in rows)
    report = {
        "config": _runtime_config(config, runtime.model_info),
        "num_samples_attempted": len(scored_items),
        "num_samples_evaluated": len(rows),
        "metrics": {
            "avg_manifold_loss": _mean(row["manifold_loss"] for row in rows),
            "median_manifold_loss": _median(row["manifold_loss"] for row in rows),
            "avg_mean_mse": _mean(row["mean_mse"] for row in rows),
            "avg_std_mse": _mean(row["std_mse"] for row in rows),
            "avg_norm_D": _mean(row["D_stats"]["mean_token_norm"] for row in rows),
            "avg_norm_V_merge": _mean(row["V_merge_stats"]["mean_token_norm"] for row in rows),
            "norm_ratio_D_to_Vmerge": _mean(row["norm_ratio_D_to_Vmerge"] for row in rows),
            "finite_rate": finite_rate,
            "collapse_near_identical_rate": collapse_rate,
            "collapse_warning": bool((collapse_rate or 0.0) > 0.1 or (finite_rate or 0.0) < 1.0),
            "manifold_active_rate": _mean(1.0 if row["manifold_active"] else 0.0 for row in rows),
            "second_full_forward_used_any": False,
        },
        "metrics_by_evidence_type": _grouped_means(
            rows,
            group_key="evidence_type",
            metric_keys=["manifold_loss", "norm_ratio_D_to_Vmerge"],
        ),
        "metrics_by_source_profile": _grouped_means(
            rows,
            group_key="source_profile",
            metric_keys=["manifold_loss", "norm_ratio_D_to_Vmerge"],
        ),
        "position_mode": config.fvt_position_mode,
        "mask_original_image_after_tgvf": True,
    }
    _write_json(output_dir / "fvt_distribution_report.json", report)
    _write_jsonl(output_dir / "per_sample_distribution.jsonl", rows)
    _save_summary(
        output_dir / "fvt_distribution_summary.txt",
        "Clean TGVF FVT Distribution Evaluation",
        [
            f"samples_evaluated: {len(rows)}",
            f"avg_manifold_loss: {report['metrics']['avg_manifold_loss']}",
            f"manifold_active_rate: {report['metrics']['manifold_active_rate']}",
            f"norm_ratio_D_to_Vmerge: {report['metrics']['norm_ratio_D_to_Vmerge']}",
            f"finite_rate: {report['metrics']['finite_rate']}",
            f"collapse_warning: {report['metrics']['collapse_warning']}",
            f"position_mode: {config.fvt_position_mode}",
            "execution_backend: clean_native_stage_diagnostics",
        ],
    )
    return {"report": str(output_dir / "fvt_distribution_report.json"), "num_rows": len(rows)}


def _compute_readout_nll(
    *,
    config: StageDiagnosticConfig,
    runtime: _Runtime,
    item: _DiagnosticItem,
    foveated_visual_tokens: Any | None,
    d_deepstack_visual_embeds: Any | None = None,
) -> dict[str, Any]:
    from revisit_vlm.tgvf_v3_stage1 import compute_v3_stage1_lm_loss, prepare_v3_stage1_readout_inputs

    if foveated_visual_tokens is None:
        readout_inputs = _prepare_target_only_readout_inputs(
            model=runtime.model,
            tokenizer_or_processor=runtime.processor,
            capture=item.capture,
            evidence_description=item.sample.evidence_description,
            merged_visual_tokens=item.merged_visual_tokens,
            device=runtime.device,
            mask_original_image_after_tgvf=True,
            protocol=config.protocol,
        )
    else:
        readout_inputs = prepare_v3_stage1_readout_inputs(
            model=runtime.model,
            tokenizer_or_processor=runtime.processor,
            capture=item.capture,
            evidence_description=item.sample.evidence_description,
            foveated_visual_tokens=foveated_visual_tokens.to(runtime.device),
            merged_visual_tokens=item.merged_visual_tokens.to(runtime.device),
            d_deepstack_visual_embeds=None
            if d_deepstack_visual_embeds is None
            else [feature.to(runtime.device) for feature in d_deepstack_visual_embeds],
            device=runtime.device,
            mask_original_image_after_tgvf=True,
            position_mode=config.fvt_position_mode,
            protocol=config.protocol,
            focus_action_im_end=config.focus_action_im_end,
        )
    loss, log_likelihood = compute_v3_stage1_lm_loss(model=runtime.model, readout_inputs=readout_inputs)
    return {
        "avg_nll": float(loss.detach().cpu()),
        "total_nll": -float(log_likelihood.detach().cpu()),
        "log_likelihood": float(log_likelihood.detach().cpu()),
        "answer_token_count": int(readout_inputs["answer_token_count"]),
        "mask_mode": readout_inputs.get("mask_mode"),
        "position_mode": readout_inputs.get("position_mode"),
        "position_ids_source": readout_inputs.get("position_ids_source"),
        "original_image_embeds_replaced": readout_inputs.get("original_image_embeds_replaced"),
    }


def _prepare_target_only_readout_inputs(
    *,
    model: Any,
    tokenizer_or_processor: Any,
    capture: Any,
    evidence_description: str,
    merged_visual_tokens: Any | None,
    device: Any,
    mask_original_image_after_tgvf: bool = True,
    protocol: str = "legacy_v3_tags",
) -> dict[str, Any]:
    import torch

    from revisit_vlm.qwen3_vl_tgvf import (
        _compute_qwen3_position_ids_for_sequence,
        _encode_text,
        normalize_tgvf_protocol,
        render_stage1_readout_text,
    )
    from revisit_vlm.tgvf_training import IGNORE_INDEX
    from revisit_vlm.tgvf_v3_stage1 import build_weak_strict_attention_mask, summarize_weak_strict_mask

    protocol = normalize_tgvf_protocol(protocol)
    if not capture.capture_found:
        raise ValueError("capture must contain a valid focus span")
    if capture.input_ids is None:
        raise ValueError("capture input_ids are required for target-only readout")
    if capture.source_visual_geometry is None:
        raise ValueError("capture is missing source visual geometry")
    source_geometry = capture.source_visual_geometry
    original_image_indices = source_geometry.source_visual_token_indices
    if original_image_indices is None:
        raise RuntimeError("source visual token indices are unavailable")

    tokenizer = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)
    base_input_ids = capture.input_ids.to(device)
    base_len = int(base_input_ids.shape[-1])
    readout_prefix, readout_text = render_stage1_readout_text(
        evidence_description=evidence_description,
        protocol=protocol,
    )
    evidence_prefix_ids = _encode_text(tokenizer, readout_prefix, device).view(1, -1)
    evidence_ids = _encode_text(tokenizer, readout_text, device).view(1, -1)
    input_ids = torch.cat([base_input_ids, evidence_prefix_ids, evidence_ids], dim=-1)
    evidence_start = base_len + int(evidence_prefix_ids.shape[-1])

    embeds = model.get_input_embeddings()(input_ids).detach().clone()
    original_image_indices = original_image_indices.to(device=device, dtype=torch.long)
    original_image_embeds_replaced = False
    if merged_visual_tokens is not None:
        if int(merged_visual_tokens.shape[0]) != int(original_image_indices.numel()):
            raise ValueError(
                "merged visual token count mismatch for target-only readout: "
                f"positions={int(original_image_indices.numel())} "
                f"merged={int(merged_visual_tokens.shape[0])}"
            )
        embeds[:, original_image_indices, :] = merged_visual_tokens.to(
            device=device,
            dtype=embeds.dtype,
        ).unsqueeze(0)
        original_image_embeds_replaced = True
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    labels[:, evidence_start:] = input_ids[:, evidence_start:]
    attention_mask_2d = torch.ones_like(input_ids)
    mm_token_type_ids = _original_image_mm_token_type_ids(
        model=model,
        input_ids=input_ids,
        image_token_id=source_geometry.image_token_id,
        device=device,
    )
    image_grid_thw = _target_only_image_grid_thw(capture, source_geometry, device)
    position_ids = _compute_qwen3_position_ids_for_sequence(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask_2d,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None if capture.video_grid_thw is None else capture.video_grid_thw.to(device),
        mm_token_type_ids=mm_token_type_ids,
    )
    if position_ids is None:
        raise RuntimeError("Qwen position id computation is unavailable")
    if mask_original_image_after_tgvf:
        attention_mask = build_weak_strict_attention_mask(
            attention_mask_2d=attention_mask_2d,
            original_image_token_indices=original_image_indices,
            block_query_start=base_len,
            dtype=embeds.dtype,
        )
        mask_mode = "weak_strict_original_image_keys_4d_target_only"
    else:
        attention_mask = attention_mask_2d
        mask_mode = "standard_2d_causal_target_only"
    mask_summary = summarize_weak_strict_mask(
        attention_mask=attention_mask,
        original_image_token_indices=original_image_indices,
        block_query_start=base_len,
    )
    return {
        "input_ids": input_ids,
        "inputs_embeds": embeds,
        "labels": labels,
        "attention_mask": attention_mask,
        "attention_mask_2d": attention_mask_2d,
        "position_ids": position_ids,
        "image_grid_thw": image_grid_thw,
        "mm_token_type_ids": mm_token_type_ids,
        "evidence_start": evidence_start,
        "answer_token_count": int(evidence_ids.shape[-1]),
        "mask_mode": mask_mode,
        "position_mode": "target_only_no_D",
        "position_ids_source": "qwen_native_source_grid_full_trajectory",
        "original_image_embeds_replaced": original_image_embeds_replaced,
        "original_image_token_count": int(original_image_indices.numel()),
        "original_image_token_span_detection": "source_image_token_id_scan",
        "block_query_start": base_len,
        "blocked_original_image_keys_for_post_tgvf": mask_summary[
            "blocked_original_image_keys_for_post_tgvf"
        ],
        "pre_tgvf_queries_keep_original_image_keys": mask_summary[
            "pre_tgvf_queries_keep_original_image_keys"
        ],
        "mask_summary": mask_summary,
        "tgvf_protocol": protocol,
    }


def _original_image_mm_token_type_ids(
    *,
    model: Any,
    input_ids: Any,
    image_token_id: int | None,
    device: Any,
) -> Any:
    import torch

    token_type_ids = torch.zeros_like(input_ids, device=device)
    if image_token_id is None:
        image_token_id = getattr(getattr(model, "config", None), "image_token_id", None)
    if image_token_id is not None:
        token_type_ids = token_type_ids.masked_fill(input_ids.to(device) == int(image_token_id), 1)
    return token_type_ids


def _target_only_image_grid_thw(capture: Any, source_geometry: Any, device: Any) -> Any:
    image_grid_thw = source_geometry.image_grid_thw
    if image_grid_thw is None:
        image_grid_thw = capture.image_grid_thw
    return None if image_grid_thw is None else image_grid_thw.to(device)


def _cpu_capture(capture: Any) -> Any:
    import torch

    from revisit_vlm.qwen3_vl_tgvf import Qwen3FocusCapture

    return Qwen3FocusCapture(
        target_text=capture.target_text,
        target_token_ids=list(capture.target_token_ids),
        target_hidden_states=capture.target_hidden_states.detach().cpu(),
        generated_ids=list(capture.generated_ids),
        generated_text=capture.generated_text,
        generated_hidden_states=torch.empty(0),
        past_key_values=None,
        attention_mask=_detach_cpu(capture.attention_mask),
        cache_position=None,
        input_ids=_detach_cpu(capture.input_ids),
        last_logits=None,
        model_kwargs={},
        image_grid_thw=_detach_cpu(capture.image_grid_thw),
        video_grid_thw=_detach_cpu(capture.video_grid_thw),
        source_visual_geometry=capture.source_visual_geometry,
        target_token_start=capture.target_token_start,
        target_token_end=capture.target_token_end,
        stop_reason=capture.stop_reason,
        capture_found=capture.capture_found,
        second_full_forward_used=False,
        malformed=capture.malformed,
        errors=list(capture.errors),
    )


def _detach_cpu(value: Any) -> Any:
    return None if value is None else value.detach().cpu()


def _can_score_fvt_for_item(item: _DiagnosticItem, d: Any) -> bool:
    geometry = item.capture.source_visual_geometry
    if geometry is None:
        return False
    return int(d.shape[0]) == int(geometry.source_visual_token_count)


def _group_indices_by_image(items: list[_DiagnosticItem]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        groups.setdefault(_group_id(item.sample), []).append(index)
    return groups


def _same_image_wrong_index(
    groups: dict[str, list[int]],
    item: _DiagnosticItem,
    index: int,
) -> int | None:
    for other in groups.get(_group_id(item.sample), []):
        if other != index:
            return other
    return None


def _different_image_index(items: list[_DiagnosticItem], index: int) -> int | None:
    source_group = _group_id(items[index].sample)
    for other_index, item in enumerate(items):
        if other_index != index and _group_id(item.sample) != source_group:
            return other_index
    return None


def _random_d_like(d: Any, *, reference: Any | None = None, seed: int = 0) -> Any:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    base = (
        reference.detach().float().cpu()
        if reference is not None and reference.numel()
        else d.detach().float().cpu()
    )
    mean = base.mean()
    std = base.std(unbiased=False).clamp_min(1e-6)
    return torch.randn(d.shape, generator=generator, dtype=torch.float32) * std + mean


def _runtime_config(config: StageDiagnosticConfig, model_info: dict[str, Any]) -> dict[str, Any]:
    return {
        **config.to_dict(),
        "mask_original_image_after_tgvf": True,
        "capture_mode": "teacher_forced",
        "execution_backend": "clean_native_stage_diagnostics",
        "model_info": model_info,
    }


def _diagnostic_forward_semantics(config: StageDiagnosticConfig) -> str:
    if config.stage == "stage2" and not config.stage2_load_lora:
        return (
            "stage2_tgvf_module_with_base_qwen_readout; qwen_lora ignored by request"
        )
    return "capture/readout use qwen_lora when present; vision tap and merger use utility model"


def _sample_metadata_row(sample: Any) -> dict[str, Any]:
    metadata = sample.metadata or {}
    return {
        "uid": _sample_uid(sample),
        "stable_image_uid": _group_id(sample),
        "image": sample.image,
        "image_id": sample.image_id,
        "target": sample.target,
        "target_style": sample.target_style,
        "target_cues": sample.target_cues,
        "evidence_description": sample.evidence_description,
        "evidence_type": sample.evidence_type,
        "source_dataset": sample.source_dataset,
        "source_profile": sample.source_profile,
        "answer_type": metadata.get("answer_type"),
        "answer_format": sample.answer_format,
        "visual_difficulty": metadata.get("visual_difficulty"),
        "visibility": metadata.get("visibility"),
        "confidence": metadata.get("confidence"),
    }


def _group_id(sample: Any) -> str:
    return str(sample.image_id or sample.image)


def _sample_uid(sample: Any) -> str:
    metadata_uid = sample.metadata.get("uid") if sample.metadata else None
    if metadata_uid:
        return str(metadata_uid)
    key = "|".join([_group_id(sample), sample.question, sample.target, sample.evidence_description])
    return f"{_group_id(sample)}:{_short_hash(key)}"


def _short_hash(value: str, *, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def _image_input(image: str, *, max_image_resolution: int | None) -> Any:
    if max_image_resolution is None or max_image_resolution <= 0:
        return image
    return {
        "type": "image",
        "image": image,
        "max_pixels": int(max_image_resolution) * int(max_image_resolution),
    }


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null", ""}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("num_foveated_tokens must be positive or none")
    return parsed


def _resolve_device_map(device_map: str | None) -> str | dict[str, str] | None:
    if device_map is None or device_map == "" or str(device_map).lower() == "none":
        return None
    if device_map == "auto":
        return "auto"
    if str(device_map).startswith("cuda") or device_map == "cpu":
        return {"": str(device_map)}
    return device_map


def _resolve_runtime_device(torch: Any, requested: Any) -> Any:
    requested_text = str(requested or "auto")
    if requested_text == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_text)


def _set_seed(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _mean(values: Iterable[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if _finite_float(value) is not None]
    if not cleaned:
        return None
    return float(sum(cleaned) / len(cleaned))


def _median(values: Iterable[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if _finite_float(value) is not None]
    if not cleaned:
        return None
    return float(statistics.median(cleaned))


def _pct_positive(values: Iterable[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if _finite_float(value) is not None]
    if not cleaned:
        return None
    return float(sum(value > 0 for value in cleaned) / len(cleaned))


def _grouped_means(
    rows: list[dict[str, Any]],
    *,
    group_key: str,
    metric_keys: list[str],
) -> dict[str, dict[str, float | None]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_key) or "unknown")].append(row)
    return {
        group: {metric: _mean(row.get(metric) for row in group_rows) for metric in metric_keys}
        for group, group_rows in sorted(groups.items())
    }


def _tensor_distribution_stats(tensor: Any) -> dict[str, Any]:
    values = tensor.detach().float()
    finite = values.isfinite()
    flat = values[finite]
    if flat.numel() == 0:
        return {
            "shape": list(values.shape),
            "finite_rate": 0.0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "mean_token_norm": None,
        }
    token_norms = values.norm(dim=-1) if values.ndim >= 2 else values.abs()
    return {
        "shape": list(values.shape),
        "finite_rate": float(finite.float().mean().item()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "mean_token_norm": float(token_norms.float().mean().item()),
        "std_token_norm": float(token_norms.float().std(unbiased=False).item()),
    }


def _shell_script(plan: dict[str, Any]) -> str:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    for key, value in sorted((plan.get("environment") or {}).items()):
        lines.append(f"export {key}={_shell_quote(str(value))}")
    lines.append(" ".join(_shell_quote(str(item)) for item in plan["command"]))
    return "\n".join(lines) + "\n"


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")


def _save_summary(path: Path, title: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [title, "=" * len(title), *lines]
    path.write_text("\n".join(payload) + "\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [_json_safe(item) for item in sorted(value, key=lambda item: str(item))]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
        if value.numel() == 1:
            return _json_safe(value.item())
        return value.tolist()
    return _to_jsonable(value)
