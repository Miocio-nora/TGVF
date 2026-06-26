"""Clean training launch-plan contracts.

The clean training CLIs do not launch long-running jobs yet. They produce
auditable launch plans that bind dataset/checkpoint identities, batch math, mask
semantics, clean-native executor status, and the temporary historical command
mapping.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .data_generation import FileIdentity, file_identity
from .deepstack import (
    deepstack_runtime_hooks,
    deepstack_scope_contract,
    qwen3_deepstack_runtime_hook_names_for_stage2_training,
)
from .defaults import (
    DEFAULT_MAX_IMAGE_RESOLUTION,
    DEFAULT_MODEL_ID,
    DEFAULT_PROTOCOL,
    DEFAULT_STAGE1_GLOBAL_BATCH,
    DEFAULT_STAGE1_MAX_STEPS,
    DEFAULT_STAGE2_GLOBAL_BATCH,
    DEFAULT_STAGE2_MAX_STEPS,
)
from .schema import DeepStackScope, DeepStackState, StrEnum, _to_jsonable
from .tgvf_protocol import SUPPORTED_PROTOCOLS


class TrainingStage(StrEnum):
    STAGE1 = "stage1"
    STAGE2 = "stage2"


class OriginalImageMaskScope(StrEnum):
    THROUGH_ANSWER = "through_answer"
    EVIDENCE_ONLY = "evidence_only"


DEFAULT_STAGE2_SPAN_WEIGHTS = {
    "evidence_state": 0.2,
    "focus_target": 1.5,
    "evidence": 1.0,
    "value_span": 1.0,
    "answer": 1.0,
    "no_focus_evidence_state": 0.2,
    "no_focus_answer": 1.0,
}

DEFAULT_STAGE2_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

TRAINING_PLAN_SCHEMA_VERSION = "clean_training_plan_v1"


@dataclass(frozen=True)
class BatchIdentity:
    world_size: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    global_batch_size: int

    def validate(self) -> None:
        for name, value in (
            ("world_size", self.world_size),
            ("micro_batch_size", self.micro_batch_size),
            ("gradient_accumulation_steps", self.gradient_accumulation_steps),
            ("global_batch_size", self.global_batch_size),
        ):
            if int(value) < 1:
                raise ValueError(f"{name} must be >= 1")
        resolved = self.world_size * self.micro_batch_size * self.gradient_accumulation_steps
        if resolved != self.global_batch_size:
            raise ValueError(
                "global_batch_size must equal "
                "world_size * micro_batch_size * gradient_accumulation_steps "
                f"({self.world_size} * {self.micro_batch_size} * "
                f"{self.gradient_accumulation_steps} = {resolved}, "
                f"got {self.global_batch_size})"
            )

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class Stage1LaunchConfig:
    run_id: str
    train_file: str
    output_dir: str
    model_id: str = DEFAULT_MODEL_ID
    processor_id: str | None = None
    protocol: str = DEFAULT_PROTOCOL
    max_image_resolution: int = DEFAULT_MAX_IMAGE_RESOLUTION
    max_steps: int = DEFAULT_STAGE1_MAX_STEPS
    save_every: int = DEFAULT_STAGE1_MAX_STEPS
    seed: int = 20260525
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    variant: str = "tgvf_v2_bidirectional"
    token_row_mode: str = "row_only"
    capture_mode: str = "teacher_forced"
    fvt_position_mode: str = "native_source_grid"
    focus_action_im_end: bool = True
    mask_original_image_after_tgvf: bool = True
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    max_grad_norm: float = 1.0
    loss_gen: float = 1.0
    loss_visual_token_manifold: float = 0.1
    loss_same_image_negative: float = 1.0
    same_image_negative_margin: float = 1.0
    same_image_negative_mode: str = "matrix_ce"
    readout_batch_size: int = 4
    min_confidence: float | None = None
    wandb_project: str | None = None
    wandb_mode: str | None = None
    batch: BatchIdentity = field(
        default_factory=lambda: resolve_batch_identity(
            global_batch_size=DEFAULT_STAGE1_GLOBAL_BATCH,
            world_size=1,
            micro_batch_size=1,
            gradient_accumulation_steps=None,
        )
    )

    def validate(self) -> None:
        if not self.run_id:
            raise ValueError("run_id is required")
        if not self.train_file:
            raise ValueError("train_file is required")
        if not self.output_dir:
            raise ValueError("output_dir is required")
        if self.protocol not in SUPPORTED_PROTOCOLS:
            raise ValueError(f"unsupported protocol: {self.protocol}")
        if self.token_row_mode != "row_only":
            raise ValueError("clean Stage1 launcher currently keeps only token_row_mode='row_only'")
        if self.capture_mode != "teacher_forced":
            raise ValueError(
                "clean Stage1 launcher removed decode_loop; capture_mode must be teacher_forced"
            )
        if self.fvt_position_mode != "native_source_grid":
            raise ValueError(
                "clean Stage1 launcher keeps only fvt_position_mode='native_source_grid'"
            )
        if self.same_image_negative_mode not in {"matrix_ce", "cyclic_margin"}:
            raise ValueError("same_image_negative_mode must be matrix_ce or cyclic_margin")
        if float(self.same_image_negative_margin) <= 0:
            raise ValueError("same_image_negative_margin must be > 0")
        if int(self.readout_batch_size) < 1:
            raise ValueError("readout_batch_size must be >= 1")
        if self.lr_scheduler not in {"constant", "linear", "cosine"}:
            raise ValueError("lr_scheduler must be constant, linear, or cosine")
        if int(self.warmup_steps) < 0:
            raise ValueError("warmup_steps must be >= 0")
        if not 0.0 <= float(self.min_lr_ratio) <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if float(self.max_grad_norm) <= 0:
            raise ValueError("max_grad_norm must be > 0")
        self.batch.validate()


@dataclass(frozen=True)
class Stage2LaunchConfig:
    run_id: str
    train_file: str
    output_dir: str
    stage1_checkpoint: str
    val_file: str | None = None
    model_id: str = DEFAULT_MODEL_ID
    processor_id: str | None = None
    protocol: str = DEFAULT_PROTOCOL
    max_image_resolution: int = DEFAULT_MAX_IMAGE_RESOLUTION
    max_seq_len: int = 2048
    max_steps: int = DEFAULT_STAGE2_MAX_STEPS
    save_every: int = 300
    eval_every: int = 300
    seed: int = 20260525
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    variant: str = "tgvf_v2_bidirectional"
    use_stage1_tgvf_config: bool = True
    fast_batched_stage2: bool = True
    fvt_position_mode: str = "native_source_grid"
    target_focus_ratio: float | None = 0.8
    mask_original_image_after_tgvf: bool = True
    mask_original_image_after_tgvf_prob: float = 1.0
    mask_original_image_after_tgvf_scope: OriginalImageMaskScope = (
        OriginalImageMaskScope.THROUGH_ANSWER
    )
    deepstack: DeepStackState = field(default_factory=DeepStackState)
    lora_rank: int = 64
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_target_modules: tuple[str, ...] = DEFAULT_STAGE2_LORA_TARGET_MODULES
    lr_lora: float = 2e-5
    lr_tgvf: float = 5e-6
    lr_calibration: float = 1e-5
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    warmup_steps: int | None = 100
    min_lr_ratio: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    loss_visual_token_manifold: float = 0.0
    weighted_span_loss: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_STAGE2_SPAN_WEIGHTS)
    )
    min_confidence: float | None = None
    wandb_project: str | None = None
    wandb_mode: str | None = None
    batch: BatchIdentity = field(
        default_factory=lambda: resolve_batch_identity(
            global_batch_size=DEFAULT_STAGE2_GLOBAL_BATCH,
            world_size=1,
            micro_batch_size=1,
            gradient_accumulation_steps=None,
        )
    )

    def validate(self) -> None:
        if not self.run_id:
            raise ValueError("run_id is required")
        if not self.train_file:
            raise ValueError("train_file is required")
        if not self.output_dir:
            raise ValueError("output_dir is required")
        if not self.stage1_checkpoint:
            raise ValueError("stage1_checkpoint is required")
        if self.protocol not in SUPPORTED_PROTOCOLS:
            raise ValueError(f"unsupported protocol: {self.protocol}")
        if self.fvt_position_mode != "native_source_grid":
            raise ValueError(
                "clean Stage2 launcher keeps only fvt_position_mode='native_source_grid'"
            )
        if not 0.0 <= float(self.mask_original_image_after_tgvf_prob) <= 1.0:
            raise ValueError("mask_original_image_after_tgvf_prob must be in [0, 1]")
        if self.target_focus_ratio is not None and not 0.0 < float(self.target_focus_ratio) < 1.0:
            raise ValueError("target_focus_ratio must be between 0 and 1")
        if self.lr_scheduler not in {"constant", "linear", "cosine"}:
            raise ValueError("lr_scheduler must be constant, linear, or cosine")
        if self.warmup_steps is not None and int(self.warmup_steps) < 0:
            raise ValueError("warmup_steps must be >= 0")
        if int(self.lora_rank) < 1:
            raise ValueError("lora_rank must be >= 1")
        if int(self.lora_alpha) < 1:
            raise ValueError("lora_alpha must be >= 1")
        if not 0.0 <= float(self.lora_dropout) <= 1.0:
            raise ValueError("lora_dropout must be in [0, 1]")
        if self.lora_bias not in {"none", "all", "lora_only"}:
            raise ValueError("lora_bias must be none, all, or lora_only")
        if not self.lora_target_modules:
            raise ValueError("lora_target_modules must not be empty")
        if float(self.adam_eps) <= 0:
            raise ValueError("adam_eps must be > 0")
        if float(self.weight_decay) < 0:
            raise ValueError("weight_decay must be >= 0")
        if float(self.max_grad_norm) <= 0:
            raise ValueError("max_grad_norm must be > 0")
        missing_weights = set(DEFAULT_STAGE2_SPAN_WEIGHTS) - set(self.weighted_span_loss)
        if missing_weights:
            raise ValueError(f"weighted_span_loss missing keys: {sorted(missing_weights)}")
        self.deepstack.validate()
        if self.deepstack.enabled:
            expected_scope = DeepStackScope(str(self.mask_original_image_after_tgvf_scope))
            if self.deepstack.original_image_scope != expected_scope:
                raise ValueError(
                    "enabled DeepStack original_image_scope must match "
                    "mask_original_image_after_tgvf_scope"
                )
            if not self.mask_original_image_after_tgvf:
                raise ValueError("enabled DeepStack requires original-image masking semantics")
        self.batch.validate()


def resolve_batch_identity(
    *,
    global_batch_size: int,
    world_size: int,
    micro_batch_size: int | None,
    gradient_accumulation_steps: int | None,
) -> BatchIdentity:
    global_batch_size = int(global_batch_size)
    world_size = int(world_size)
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if global_batch_size < 1:
        raise ValueError("global_batch_size must be >= 1")
    if micro_batch_size is None and gradient_accumulation_steps is None:
        micro_batch_size = 1
        denom = world_size * micro_batch_size
        if global_batch_size % denom:
            raise ValueError("global_batch_size must be divisible by world_size")
        gradient_accumulation_steps = global_batch_size // denom
    elif micro_batch_size is None:
        gradient_accumulation_steps = int(gradient_accumulation_steps or 0)
        denom = world_size * gradient_accumulation_steps
        if denom < 1 or global_batch_size % denom:
            raise ValueError(
                "global_batch_size must be divisible by world_size * gradient_accumulation_steps"
            )
        micro_batch_size = global_batch_size // denom
    elif gradient_accumulation_steps is None:
        micro_batch_size = int(micro_batch_size)
        denom = world_size * micro_batch_size
        if denom < 1 or global_batch_size % denom:
            raise ValueError("global_batch_size must be divisible by world_size * micro_batch_size")
        gradient_accumulation_steps = global_batch_size // denom
    batch = BatchIdentity(
        world_size=world_size,
        micro_batch_size=int(micro_batch_size),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
        global_batch_size=global_batch_size,
    )
    batch.validate()
    return batch


def build_stage1_launch_plan(
    config: Stage1LaunchConfig,
    *,
    git_commit: str | None = None,
    dirty_worktree: bool | None = None,
) -> dict[str, Any]:
    config.validate()
    train_identity = _required_file_identity(config.train_file, label="train_file")
    command = _stage1_legacy_command(config)
    native_training = _clean_native_training_status(
        TrainingStage.STAGE1,
        world_size=config.batch.world_size,
    )
    return {
        "training_plan_schema_version": TRAINING_PLAN_SCHEMA_VERSION,
        "stage": TrainingStage.STAGE1,
        "run_id": config.run_id,
        "output_dir": config.output_dir,
        "git_commit": git_commit,
        "dirty_worktree": dirty_worktree,
        "model": {
            "model_id": config.model_id,
            "processor_id": config.processor_id,
            "dtype": config.dtype,
            "attn_implementation": config.attn_implementation,
        },
        "protocol": config.protocol,
        "dataset": {"train_file": train_identity.to_dict()},
        "batch": config.batch.to_dict(),
        "training": {
            "max_steps": config.max_steps,
            "save_every": config.save_every,
            "seed": config.seed,
            "max_image_resolution": config.max_image_resolution,
            "variant": config.variant,
            "token_row_mode": config.token_row_mode,
            "capture_mode": config.capture_mode,
            "fvt_position_mode": config.fvt_position_mode,
            "focus_action_im_end": config.focus_action_im_end,
            "mask_original_image_after_tgvf": config.mask_original_image_after_tgvf,
            "readout_batch_size": config.readout_batch_size,
            "same_image_negative_margin": config.same_image_negative_margin,
            "same_image_negative_mode": config.same_image_negative_mode,
        },
        "readout_context": _stage1_readout_context(config),
        "module_policy": _stage1_module_policy(),
        "loss": {
            "gen": config.loss_gen,
            "visual_token_manifold": config.loss_visual_token_manifold,
            "same_image_negative": config.loss_same_image_negative,
        },
        "optimizer": {
            "name": "adamw",
            "learning_rate": config.learning_rate,
            "lr_scheduler": config.lr_scheduler,
            "warmup_steps": config.warmup_steps,
            "min_lr_ratio": config.min_lr_ratio,
            "max_grad_norm": config.max_grad_norm,
        },
        "wandb": {
            "project": config.wandb_project,
            "mode": config.wandb_mode,
        },
        "clean_constraints": {
            "decode_loop_removed": True,
            "token_row_mode_whitelist": ["row_only"],
            "fvt_position_mode_whitelist": ["native_source_grid"],
        },
        "clean_native_training": native_training,
        "clean_prepare_execution_command": _clean_prepare_execution_command_payload(
            TrainingStage.STAGE1,
            output_dir=config.output_dir,
        ),
        "clean_training_command": _clean_training_command_payload(
            TrainingStage.STAGE1,
            world_size=config.batch.world_size,
            output_dir=config.output_dir,
            native_status=native_training,
        ),
        "legacy_reference_command": _legacy_reference_command_payload(command),
    }


def build_stage2_launch_plan(
    config: Stage2LaunchConfig,
    *,
    git_commit: str | None = None,
    dirty_worktree: bool | None = None,
) -> dict[str, Any]:
    config.validate()
    train_identity = _required_file_identity(config.train_file, label="train_file")
    val_identity = file_identity(config.val_file).to_dict() if config.val_file else None
    if config.val_file and not val_identity["exists"]:
        raise FileNotFoundError(f"val_file does not exist: {config.val_file}")
    checkpoint_identity = _required_file_identity(
        config.stage1_checkpoint,
        label="stage1_checkpoint",
    )
    command = _stage2_legacy_command(config) if not config.deepstack.enabled else None
    legacy_reference = (
        _legacy_reference_command_payload(command)
        if command is not None
        else _command_payload(
            [],
            executable=False,
            unavailable_reason="historical Stage2 script has no DeepStack training controls",
        )
    )
    native_training = _clean_native_training_status(
        TrainingStage.STAGE2,
        world_size=config.batch.world_size,
        deepstack_enabled=config.deepstack.enabled,
    )
    deepstack_training_plan = _deepstack_training_plan(config)
    return {
        "training_plan_schema_version": TRAINING_PLAN_SCHEMA_VERSION,
        "stage": TrainingStage.STAGE2,
        "run_id": config.run_id,
        "output_dir": config.output_dir,
        "git_commit": git_commit,
        "dirty_worktree": dirty_worktree,
        "model": {
            "model_id": config.model_id,
            "processor_id": config.processor_id,
            "dtype": config.dtype,
            "attn_implementation": config.attn_implementation,
        },
        "protocol": config.protocol,
        "dataset": {
            "train_file": train_identity.to_dict(),
            "val_file": val_identity,
            "stage1_checkpoint": checkpoint_identity.to_dict(),
        },
        "batch": config.batch.to_dict(),
        "training": {
            "max_steps": config.max_steps,
            "save_every": config.save_every,
            "eval_every": config.eval_every,
            "seed": config.seed,
            "max_image_resolution": config.max_image_resolution,
            "max_seq_len": config.max_seq_len,
            "variant": config.variant,
            "use_stage1_tgvf_config": config.use_stage1_tgvf_config,
            "fast_batched_stage2": config.fast_batched_stage2,
            "fvt_position_mode": config.fvt_position_mode,
            "target_focus_ratio": config.target_focus_ratio,
        },
        "module_policy": _stage2_module_policy(),
        "mask_policy": {
            "mask_original_image_after_tgvf": config.mask_original_image_after_tgvf,
            "mask_original_image_after_tgvf_prob": config.mask_original_image_after_tgvf_prob,
            "mask_original_image_after_tgvf_scope": str(
                config.mask_original_image_after_tgvf_scope
            ),
        },
        "deepstack": config.deepstack.to_dict(),
        "deepstack_training_plan": deepstack_training_plan,
        "lora": {
            "rank": config.lora_rank,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "bias": config.lora_bias,
            "target_modules": list(config.lora_target_modules),
        },
        "loss": {
            "weighted_span_loss": dict(config.weighted_span_loss),
            "visual_token_manifold": config.loss_visual_token_manifold,
        },
        "optimizer": {
            "name": "adamw",
            "lr_lora": config.lr_lora,
            "lr_tgvf": config.lr_tgvf,
            "lr_calibration": config.lr_calibration,
            "lr_scheduler": config.lr_scheduler,
            "warmup_ratio": config.warmup_ratio,
            "warmup_steps": config.warmup_steps,
            "min_lr_ratio": config.min_lr_ratio,
            "betas": [config.adam_beta1, config.adam_beta2],
            "eps": config.adam_eps,
            "weight_decay": config.weight_decay,
            "max_grad_norm": config.max_grad_norm,
        },
        "wandb": {
            "project": config.wandb_project,
            "mode": config.wandb_mode,
        },
        "clean_constraints": {
            "fvt_position_mode_whitelist": ["native_source_grid"],
            "deepstack_training_schema_supported": True,
            "deepstack_training_execution_supported": deepstack_training_plan[
                "execution_supported"
            ],
            "d_deepstack_features_default": False,
            "legacy_command_is_final": False,
        },
        "clean_native_training": native_training,
        "clean_prepare_execution_command": _clean_prepare_execution_command_payload(
            TrainingStage.STAGE2,
            output_dir=config.output_dir,
        ),
        "clean_training_command": _clean_training_command_payload(
            TrainingStage.STAGE2,
            world_size=config.batch.world_size,
            output_dir=config.output_dir,
            native_status=native_training,
        ),
        "legacy_reference_command": legacy_reference,
    }


def write_training_plan(output_dir: str | Path, plan: dict[str, Any]) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "training_plan.json"
    text_path = out / "training_plan.txt"
    dataset_path = out / "dataset_identity.json"
    native_status_path = out / "clean_native_training_status.json"
    prepare_command_path = out / "clean_prepare_execution_command.sh"
    clean_command_path = out / "clean_training_command.sh"
    command_path = out / "legacy_reference_command.sh"
    plan_path.write_text(
        json.dumps(_to_jsonable(plan), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    text_path.write_text(_training_plan_text(plan), encoding="utf-8")
    dataset_path.write_text(
        json.dumps(_to_jsonable(plan.get("dataset", {})), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    native_status_path.write_text(
        json.dumps(
            _to_jsonable(plan.get("clean_native_training", {})),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    prepare_command_path.write_text(
        _command_script_text(plan.get("clean_prepare_execution_command") or {}),
        encoding="utf-8",
    )
    prepare_command_path.chmod(0o755)
    clean_command_path.write_text(
        _command_script_text(plan.get("clean_training_command") or {}),
        encoding="utf-8",
    )
    if (plan.get("clean_training_command") or {}).get("executable"):
        clean_command_path.chmod(0o755)
    command_path.write_text(
        _command_script_text(plan.get("legacy_reference_command") or {}),
        encoding="utf-8",
    )
    return {
        "output_dir": str(out),
        "training_plan": str(plan_path),
        "training_plan_txt": str(text_path),
        "dataset_identity": str(dataset_path),
        "clean_native_training_status": str(native_status_path),
        "clean_prepare_execution_command": str(prepare_command_path),
        "clean_training_command": str(clean_command_path),
        "legacy_reference_command": str(command_path),
    }


def _required_file_identity(path: str, *, label: str) -> FileIdentity:
    identity = file_identity(path)
    if not identity.exists:
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return identity


def _command_payload(
    command: list[str],
    *,
    executable: bool,
    unavailable_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "executable": executable,
        "status": "temporary_legacy_reference_not_final_clean_native",
        "final_clean_native": False,
        "unavailable_reason": unavailable_reason,
        "argv": command,
        "shell": shlex.join(command) if command else "",
    }


def _legacy_reference_command_payload(command: list[str]) -> dict[str, Any]:
    return _command_payload(
        command,
        executable=False,
        unavailable_reason=(
            "historical training script is retained as an audit reference only; "
            "the final clean-native training entrypoint is clean_training_command"
        ),
    )


def _clean_training_command_payload(
    stage: TrainingStage,
    *,
    world_size: int,
    output_dir: str,
    native_status: dict[str, Any],
) -> dict[str, Any]:
    entrypoint = f"revisit_vlm_clean.training.{stage.value}_executor"
    if world_size == 1:
        command = [
            "python",
            "-m",
            entrypoint,
            "--plan",
            str(Path(output_dir) / "training_plan.json"),
            "--launch-training",
        ]
    else:
        command = [
            "torchrun",
            "--nproc-per-node",
            str(world_size),
            "-m",
            entrypoint,
            "--plan",
            str(Path(output_dir) / "training_plan.json"),
            "--launch-training",
        ]
    executable = bool(native_status.get("executable"))
    unavailable_reason = None if executable else "; ".join(
        str(item) for item in native_status.get("blocking_items") or []
    )
    return {
        "executable": executable,
        "status": native_status.get("status"),
        "final_clean_native": True,
        "planned_entrypoint": entrypoint,
        "unavailable_reason": unavailable_reason,
        "will_launch_training": executable,
        "runtime": native_status.get("runtime"),
        "argv": command,
        "shell": shlex.join(command),
    }


def _clean_prepare_execution_command_payload(
    stage: TrainingStage,
    *,
    output_dir: str,
) -> dict[str, Any]:
    entrypoint = f"revisit_vlm_clean.training.{stage.value}_executor"
    command = [
        "python",
        "-m",
        entrypoint,
        "--plan",
        str(Path(output_dir) / "training_plan.json"),
        "--prepare-execution",
    ]
    return {
        "executable": True,
        "status": "prepare_execution_supported",
        "final_clean_native": True,
        "planned_entrypoint": entrypoint,
        "artifact": "clean_training_execution_bundle.json",
        "will_launch_training": False,
        "argv": command,
        "shell": shlex.join(command),
    }


def _command_script_text(command: dict[str, Any]) -> str:
    shell = str(command.get("shell") or "").strip()
    if not shell:
        return "# unavailable\n"
    if command.get("executable"):
        return shell + "\n"
    reason = str(command.get("unavailable_reason") or "not executable").strip()
    return f"# not executable: {reason}\n# {shell}\n"


def _clean_native_training_status(
    stage: TrainingStage,
    *,
    world_size: int,
    deepstack_enabled: bool = False,
) -> dict[str, Any]:
    blockers = []
    runtime = "single_process" if world_size == 1 else "distributed_torchrun"
    executable = not blockers
    status = (
        "clean_native_single_process_launch_supported"
        if executable and world_size == 1
        else "clean_native_distributed_launch_supported"
        if executable
        else "clean_native_launch_blocked"
    )
    return {
        "stage": str(stage),
        "executable": executable,
        "status": status,
        "required_for_final_clean_project": True,
        "legacy_reference_is_final": False,
        "prepare_execution_supported": True,
        "launch_training_supported": executable,
        "runtime": runtime,
        "current_artifact": (
            "clean single-process trainer loop"
            if executable and world_size == 1
            else "clean torchrun distributed trainer loop"
            if executable
            else "auditable launch plan plus clean execution bundle handoff"
        ),
        "blocking_items": blockers,
    }


def _deepstack_training_plan(config: Stage2LaunchConfig) -> dict[str, Any]:
    state = config.deepstack
    scope = state.original_image_scope
    enabled = bool(state.enabled)
    implemented_hooks = (
        qwen3_deepstack_runtime_hook_names_for_stage2_training()
        if enabled
        else set()
    )
    runtime_hooks = deepstack_runtime_hooks(
        state,
        surface="stage2_training",
        implemented_hooks=implemented_hooks,
    )
    blocking_items = list(runtime_hooks.get("blocking_items") or [])
    scope_contract = deepstack_scope_contract(
        state,
        surface="stage2_training",
        execution_supported=not blocking_items,
        blocking_items=blocking_items,
        implemented_hooks=implemented_hooks,
    )
    return {
        "schema_version": "clean_deepstack_training_plan_v1",
        "stage": "stage2",
        "enabled": enabled,
        "requested": state.to_dict(),
        "execution_supported": not blocking_items,
        "default_enabled": False,
        "original_image_scope": str(scope),
        "original_image_deepstack": scope_contract["original_image_deepstack"],
        "d_deepstack_features": scope_contract["d_deepstack_features"],
        "fvt_visual_token_path": scope_contract["fvt_visual_token_path"],
        "scope_contract": scope_contract,
        "runtime_hooks": runtime_hooks,
        "current_training_path": {
            "uses_manual_inputs_embeds": True,
            "qwen3_deepstack_features_injected": enabled,
            "post_d_deepstack_scope_mask_applied": enabled,
            "answer_stage_restore_supported": (
                enabled and scope == DeepStackScope.EVIDENCE_ONLY
            ),
        },
        "gate_name": "apply_deepstack_training_scope_when_enabled",
        "blocking_items": blocking_items,
    }


def _stage1_module_policy() -> dict[str, Any]:
    return {
        "trainable": [
            "tgvf_module",
            "protocol_c_token_rows_row_only",
        ],
        "frozen": [
            "qwen_language_backbone",
            "qwen_vision_encoder",
            "qwen_visual_merger",
        ],
        "visual_merger": {
            "trainable": False,
            "usage": "frozen_finalize_path",
        },
        "training_runtime": {
            "use_cache": False,
            "print_trainable_parameter_names_before_launch": True,
        },
    }


def _stage1_readout_context(config: Stage1LaunchConfig) -> dict[str, Any]:
    return {
        "original_image_placeholder_embeddings": "replace_with_qwen_v_merge",
        "d_append_path": "native_qwen_visual_span",
        "position_ids": "real_qwen3_mrope_full_trajectory",
        "fvt_position_mode": config.fvt_position_mode,
        "d_token_count": "dynamic_source_image_visual_token_count",
        "visual_merger_path": "frozen_finalize_path",
        "attention_mask": {
            "mask_original_image_after_tgvf": config.mask_original_image_after_tgvf,
            "blocking": "weak_strict_original_image_key_blocking_after_tgvf_append",
            "scope": "stage1_readout_after_tgvf_append",
        },
    }


def _stage2_module_policy() -> dict[str, Any]:
    return {
        "trainable": [
            "qwen_lora_adapters",
            "tgvf_module_continued_from_stage1",
            "protocol_c_token_rows_restored_from_stage1",
        ],
        "token_row_implementation": {
            "current_peft_path": 'modules_to_save=["embed_tokens", "lm_head"]',
            "switching_requires_named_ablation": True,
        },
        "frozen": [
            "base_qwen_weights_outside_lora_and_saved_token_modules",
            "qwen_vision_encoder",
            "qwen_visual_merger",
        ],
        "visual_merger": {
            "trainable": False,
            "usage": "frozen_finalize_path",
        },
        "training_runtime": {
            "use_cache": False,
            "gradient_checkpointing": True,
            "print_trainable_parameter_names_before_launch": True,
        },
    }


def _stage1_legacy_command(config: Stage1LaunchConfig) -> list[str]:
    command = [
        "torchrun",
        "--nproc-per-node",
        str(config.batch.world_size),
        "scripts/train_tgvf_v3_stage1.py",
        "--train-file",
        config.train_file,
        "--output-dir",
        config.output_dir,
        "--model-id",
        config.model_id,
        "--tgvf-protocol",
        config.protocol,
        "--dtype",
        config.dtype,
        "--attn-implementation",
        config.attn_implementation,
        "--variant",
        config.variant,
        "--batch-size",
        str(config.batch.micro_batch_size),
        "--gradient-accumulation-steps",
        str(config.batch.gradient_accumulation_steps),
        "--max-steps",
        str(config.max_steps),
        "--save-every",
        str(config.save_every),
        "--seed",
        str(config.seed),
        "--max-image-resolution",
        str(config.max_image_resolution),
        "--protocol-token-row-mode",
        config.token_row_mode,
        "--capture-mode",
        config.capture_mode,
        "--fvt-position-mode",
        config.fvt_position_mode,
        "--learning-rate",
        str(config.learning_rate),
        "--lr-scheduler",
        config.lr_scheduler,
        "--warmup-steps",
        str(config.warmup_steps),
        "--min-lr-ratio",
        str(config.min_lr_ratio),
        "--loss-gen",
        str(config.loss_gen),
        "--loss-visual-token-manifold",
        str(config.loss_visual_token_manifold),
        "--loss-same-image-negative",
        str(config.loss_same_image_negative),
        "--same-image-negative-margin",
        str(config.same_image_negative_margin),
        "--same-image-negative-mode",
        config.same_image_negative_mode,
        "--readout-batch-size",
        str(config.readout_batch_size),
        "--max-grad-norm",
        str(config.max_grad_norm),
    ]
    _append_optional(command, "--processor-id", config.processor_id)
    _append_optional(command, "--min-confidence", config.min_confidence)
    _append_optional(command, "--wandb-project", config.wandb_project)
    _append_optional(command, "--wandb-mode", config.wandb_mode)
    command.append(
        "--focus-action-im-end" if config.focus_action_im_end else "--no-focus-action-im-end"
    )
    command.append(
        "--mask-original-image-after-tgvf"
        if config.mask_original_image_after_tgvf
        else "--no-mask-original-image-after-tgvf"
    )
    return command


def _stage2_legacy_command(config: Stage2LaunchConfig) -> list[str]:
    command = [
        "torchrun",
        "--nproc-per-node",
        str(config.batch.world_size),
        "scripts/train_tgvf_v3_stage2.py",
        "--train-file",
        config.train_file,
        "--output-dir",
        config.output_dir,
        "--stage1-checkpoint",
        config.stage1_checkpoint,
        "--model-id",
        config.model_id,
        "--tgvf-protocol",
        config.protocol,
        "--dtype",
        config.dtype,
        "--attn-implementation",
        config.attn_implementation,
        "--variant",
        config.variant,
        "--batch-size",
        str(config.batch.micro_batch_size),
        "--gradient-accumulation-steps",
        str(config.batch.gradient_accumulation_steps),
        "--max-steps",
        str(config.max_steps),
        "--save-every",
        str(config.save_every),
        "--eval-every",
        str(config.eval_every),
        "--seed",
        str(config.seed),
        "--max-image-resolution",
        str(config.max_image_resolution),
        "--max-seq-len",
        str(config.max_seq_len),
        "--fvt-position-mode",
        config.fvt_position_mode,
        "--lora-rank",
        str(config.lora_rank),
        "--lora-alpha",
        str(config.lora_alpha),
        "--lora-dropout",
        str(config.lora_dropout),
        "--lora-bias",
        config.lora_bias,
        "--lora-target-modules",
        ",".join(config.lora_target_modules),
        "--mask-original-image-after-tgvf-prob",
        str(config.mask_original_image_after_tgvf_prob),
        "--mask-original-image-after-tgvf-scope",
        str(config.mask_original_image_after_tgvf_scope),
        "--lr-lora",
        str(config.lr_lora),
        "--lr-tgvf",
        str(config.lr_tgvf),
        "--lr-calibration",
        str(config.lr_calibration),
        "--lr-scheduler",
        config.lr_scheduler,
        "--warmup-ratio",
        str(config.warmup_ratio),
        "--loss-visual-token-manifold",
        str(config.loss_visual_token_manifold),
    ]
    _append_optional(command, "--warmup-steps", config.warmup_steps)
    command.extend(
        [
            "--min-lr-ratio",
            str(config.min_lr_ratio),
            "--adam-beta1",
            str(config.adam_beta1),
            "--adam-beta2",
            str(config.adam_beta2),
            "--adam-eps",
            str(config.adam_eps),
            "--weight-decay",
            str(config.weight_decay),
            "--max-grad-norm",
            str(config.max_grad_norm),
        ]
    )
    _append_optional(command, "--processor-id", config.processor_id)
    _append_optional(command, "--val-file", config.val_file)
    _append_optional(command, "--target-focus-ratio", config.target_focus_ratio)
    _append_optional(command, "--min-confidence", config.min_confidence)
    _append_optional(command, "--wandb-project", config.wandb_project)
    _append_optional(command, "--wandb-mode", config.wandb_mode)
    command.append(
        "--use-stage1-tgvf-config"
        if config.use_stage1_tgvf_config
        else "--no-use-stage1-tgvf-config"
    )
    command.append(
        "--fast-batched-stage2" if config.fast_batched_stage2 else "--no-fast-batched-stage2"
    )
    command.append(
        "--mask-original-image-after-tgvf"
        if config.mask_original_image_after_tgvf
        else "--no-mask-original-image-after-tgvf"
    )
    for key, value in config.weighted_span_loss.items():
        command.extend([f"--loss-{key.replace('_', '-')}", str(value)])
    return command


def _append_optional(command: list[str], flag: str, value: Any | None) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def _training_plan_text(plan: dict[str, Any]) -> str:
    batch = plan["batch"]
    lines = [
        f"training_plan_schema_version: {plan.get('training_plan_schema_version')}",
        f"run_id: {plan['run_id']}",
        f"stage: {plan['stage']}",
        f"output_dir: {plan['output_dir']}",
        f"git_commit: {plan.get('git_commit')}",
        f"dirty_worktree: {plan.get('dirty_worktree')}",
        f"protocol: {plan['protocol']}",
        f"model_id: {plan['model']['model_id']}",
        f"processor_id: {plan['model'].get('processor_id')}",
        (
            "global_batch: "
            f"{batch['global_batch_size']} = {batch['world_size']} * "
            f"{batch['micro_batch_size']} * {batch['gradient_accumulation_steps']}"
        ),
    ]
    clean_command = plan.get("clean_training_command") or {}
    prepare_command = plan.get("clean_prepare_execution_command") or {}
    command = plan.get("legacy_reference_command") or {}
    native = plan.get("clean_native_training") or {}
    module_policy = plan.get("module_policy") or {}
    lines.extend(
        [
            f"clean_native_training_executable: {native.get('executable')}",
            f"clean_native_training_status: {native.get('status')}",
            f"clean_prepare_execution_command_executable: {prepare_command.get('executable')}",
            f"clean_prepare_execution_command_status: {prepare_command.get('status')}",
            f"clean_training_command_executable: {clean_command.get('executable')}",
            f"clean_training_command_status: {clean_command.get('status')}",
            f"trainable_modules: {json.dumps(module_policy.get('trainable', []))}",
            f"frozen_modules: {json.dumps(module_policy.get('frozen', []))}",
        ]
    )
    if prepare_command.get("shell"):
        lines.extend(["clean_prepare_execution_command:", prepare_command["shell"]])
    if clean_command.get("unavailable_reason"):
        lines.append(f"clean_training_unavailable_reason: {clean_command['unavailable_reason']}")
    if clean_command.get("shell"):
        lines.extend(["clean_training_command:", clean_command["shell"]])
    lines.extend(
        [
            f"legacy_reference_executable: {command.get('executable')}",
            f"legacy_reference_status: {command.get('status')}",
        ]
    )
    if command.get("unavailable_reason"):
        lines.append(f"legacy_reference_unavailable_reason: {command['unavailable_reason']}")
    if command.get("shell"):
        lines.extend(["legacy_reference_command:", command["shell"]])
    return "\n".join(lines) + "\n"
