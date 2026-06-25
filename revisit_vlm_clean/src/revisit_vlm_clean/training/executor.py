"""Fail-safe clean training executor preflight.

The real Stage1/Stage2 training loops are not ported yet. This module makes the
planned clean entrypoints importable and validates training-plan identity before
any future executor is allowed to launch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from revisit_vlm_clean.cli.common import exit_not_implemented, print_json
from revisit_vlm_clean.data_generation import file_identity
from revisit_vlm_clean.schema import _to_jsonable
from revisit_vlm_clean.training_plan import TRAINING_PLAN_SCHEMA_VERSION, TrainingStage

TRAINING_EXECUTION_BUNDLE_SCHEMA_VERSION = "clean_training_execution_bundle_v1"


def build_parser(stage: TrainingStage) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Clean {stage.value} training executor.")
    parser.add_argument("--plan", required=True, help="Path to training_plan.json.")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the plan and print executor status without launching training.",
    )
    parser.add_argument(
        "--preflight-report",
        default=None,
        help=(
            "Path for the preflight report JSON. Defaults to "
            "<plan-dir>/<stage>_training_preflight_report.json."
        ),
    )
    parser.add_argument(
        "--prepare-execution",
        action="store_true",
        help=(
            "Write executor-owned clean training execution bundle/status artifacts. "
            "This validates the handoff into the clean executor without launching training."
        ),
    )
    parser.add_argument(
        "--execution-dir",
        default=None,
        help=(
            "Directory for --prepare-execution artifacts. Defaults to "
            "<plan-dir>/clean_training_execution."
        ),
    )
    return parser


def main_for_stage(stage: TrainingStage, argv: list[str] | None = None) -> int:
    args = build_parser(stage).parse_args(argv)
    plan_path, plan = _load_training_plan(args.plan)
    _validate_training_plan(plan, expected_stage=stage)
    report = _preflight_report(plan_path, plan, expected_stage=stage)
    report_path = _preflight_report_path(
        plan_path=plan_path,
        stage=stage,
        requested=args.preflight_report,
    )
    report["preflight_report"] = str(report_path)
    _write_json(report_path, report)
    if args.prepare_execution:
        prepared = prepare_training_execution(
            plan_path,
            plan,
            expected_stage=stage,
            execution_dir=args.execution_dir,
        )
        prepared["preflight_report"] = str(report_path)
        print_json(prepared)
        return 0
    print_json(report)
    if args.preflight_only:
        return 0
    return exit_not_implemented(
        "clean-native training execution is not implemented yet; "
        "rerun with --preflight-only or --prepare-execution to validate without launching"
    )


def preflight_training_plan(path: str | Path, *, expected_stage: TrainingStage) -> dict[str, Any]:
    plan_path, plan = _load_training_plan(path)
    _validate_training_plan(plan, expected_stage=expected_stage)
    return _preflight_report(plan_path, plan, expected_stage=expected_stage)


def prepare_training_execution(
    plan_path: str | Path,
    plan: dict[str, Any] | None = None,
    *,
    expected_stage: TrainingStage,
    execution_dir: str | Path | None = None,
) -> dict[str, Any]:
    resolved_plan_path = Path(plan_path)
    if plan is None:
        resolved_plan_path, plan = _load_training_plan(resolved_plan_path)
    _validate_training_plan(plan, expected_stage=expected_stage)
    out = _execution_dir_path(plan_path=resolved_plan_path, requested=execution_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle = _build_execution_bundle(
        plan_path=resolved_plan_path,
        plan=plan,
        expected_stage=expected_stage,
        execution_dir=out,
    )
    status = _execution_status(bundle)
    bundle_path = out / "clean_training_execution_bundle.json"
    status_path = out / "clean_training_execution_status.json"
    text_path = out / "clean_training_execution_bundle.txt"
    _write_json(bundle_path, bundle)
    _write_json(status_path, status)
    text_path.write_text(_execution_bundle_text(bundle, status), encoding="utf-8")
    return {
        "execution_dir": str(out),
        "execution_bundle": str(bundle_path),
        "execution_status": str(status_path),
        "execution_bundle_txt": str(text_path),
        "stage": str(expected_stage),
        "run_id": plan.get("run_id"),
        "will_launch_training": False,
        "training_runtime_ported": False,
        "runner_status": status["runner_status"],
    }


def _load_training_plan(path: str | Path) -> tuple[Path, dict[str, Any]]:
    plan_path = Path(path)
    if not plan_path.exists():
        raise FileNotFoundError(f"training plan does not exist: {plan_path}")
    return plan_path, json.loads(plan_path.read_text(encoding="utf-8"))


def _preflight_report(
    plan_path: Path,
    plan: dict[str, Any],
    *,
    expected_stage: TrainingStage,
) -> dict[str, Any]:
    native_status = dict(plan.get("clean_native_training") or {})
    return {
        "plan_path": str(plan_path),
        "plan_identity": file_identity(plan_path).to_dict(),
        "plan_valid": True,
        "stage": str(expected_stage),
        "run_id": plan.get("run_id"),
        "output_dir": plan.get("output_dir"),
        "training_plan_schema_version": plan.get("training_plan_schema_version"),
        "clean_native_training_executable": bool(native_status.get("executable")),
        "clean_native_training_status": native_status.get("status"),
        "training_runtime_ported": False,
        "will_launch_training": False,
        "blocking_items": list(native_status.get("blocking_items") or []),
    }


def _preflight_report_path(
    *,
    plan_path: str | Path,
    stage: TrainingStage,
    requested: str | None,
) -> Path:
    if requested:
        return Path(requested)
    return Path(plan_path).resolve().parent / f"{stage.value}_training_preflight_report.json"


def _execution_dir_path(
    *,
    plan_path: str | Path,
    requested: str | Path | None,
) -> Path:
    if requested:
        return Path(requested)
    return Path(plan_path).resolve().parent / "clean_training_execution"


def _build_execution_bundle(
    *,
    plan_path: Path,
    plan: dict[str, Any],
    expected_stage: TrainingStage,
    execution_dir: Path,
) -> dict[str, Any]:
    clean_command = dict(plan.get("clean_training_command") or {})
    legacy_reference = dict(plan.get("legacy_reference_command") or {})
    native_status = dict(plan.get("clean_native_training") or {})
    bundle: dict[str, Any] = {
        "training_execution_bundle_schema_version": TRAINING_EXECUTION_BUNDLE_SCHEMA_VERSION,
        "stage": str(expected_stage),
        "run_id": plan.get("run_id"),
        "output_dir": plan.get("output_dir"),
        "execution_dir": str(execution_dir),
        "plan_path": str(plan_path),
        "plan_identity": file_identity(plan_path).to_dict(),
        "training_plan_schema_version": plan.get("training_plan_schema_version"),
        "git_commit": plan.get("git_commit"),
        "dirty_worktree": plan.get("dirty_worktree"),
        "model": plan.get("model"),
        "protocol": plan.get("protocol"),
        "dataset": plan.get("dataset"),
        "batch": plan.get("batch"),
        "training": plan.get("training"),
        "module_policy": plan.get("module_policy"),
        "loss": plan.get("loss"),
        "optimizer": plan.get("optimizer"),
        "wandb": plan.get("wandb"),
        "clean_executor": {
            "entrypoint": f"revisit_vlm_clean.training.{expected_stage.value}_executor",
            "final_clean_native": True,
            "owns_execution_bundle": True,
            "trainer_loop_ported": False,
            "will_launch_training": False,
            "status": "trainer_loop_not_ported",
            "blocking_items": list(native_status.get("blocking_items") or []),
        },
        "trainer_runtime_contract": _trainer_runtime_contract(
            expected_stage=expected_stage,
            plan=plan,
            native_status=native_status,
        ),
        "clean_training_command": {
            "planned_entrypoint": clean_command.get("planned_entrypoint"),
            "final_clean_native": clean_command.get("final_clean_native"),
            "executable": clean_command.get("executable"),
            "argv": list(clean_command.get("argv") or []),
        },
        "legacy_reference": {
            "final_clean_native": legacy_reference.get("final_clean_native"),
            "executable": legacy_reference.get("executable"),
            "allowed_for_execution": False,
        },
        "safety": {
            "will_launch_training": False,
            "legacy_reference_allowed": False,
            "legacy_reference_executable": bool(legacy_reference.get("executable")),
            "requires_clean_trainer_loop_before_launch": True,
        },
    }
    if expected_stage == TrainingStage.STAGE1:
        bundle["readout_context"] = plan.get("readout_context")
    if expected_stage == TrainingStage.STAGE2:
        bundle["mask_policy"] = plan.get("mask_policy")
        bundle["deepstack"] = plan.get("deepstack")
        bundle["lora"] = plan.get("lora")
    return bundle


def _execution_status(bundle: dict[str, Any]) -> dict[str, Any]:
    executor = bundle.get("clean_executor") or {}
    safety = bundle.get("safety") or {}
    return {
        "training_execution_bundle_schema_version": bundle.get(
            "training_execution_bundle_schema_version"
        ),
        "stage": bundle.get("stage"),
        "run_id": bundle.get("run_id"),
        "plan_identity": bundle.get("plan_identity"),
        "bundle_valid": True,
        "runner_status": executor.get("status"),
        "training_runtime_ported": bool(executor.get("trainer_loop_ported")),
        "trainer_runtime_contract_status": (
            (bundle.get("trainer_runtime_contract") or {}).get("status")
        ),
        "will_launch_training": bool(safety.get("will_launch_training")),
        "legacy_reference_allowed": bool(safety.get("legacy_reference_allowed")),
        "blocking_items": list(executor.get("blocking_items") or []),
    }


def _execution_bundle_text(bundle: dict[str, Any], status: dict[str, Any]) -> str:
    batch = bundle.get("batch") or {}
    lines = [
        f"schema: {bundle.get('training_execution_bundle_schema_version')}",
        f"run_id: {bundle.get('run_id')}",
        f"stage: {bundle.get('stage')}",
        f"plan_path: {bundle.get('plan_path')}",
        f"plan_sha256: {(bundle.get('plan_identity') or {}).get('sha256')}",
        f"output_dir: {bundle.get('output_dir')}",
        f"execution_dir: {bundle.get('execution_dir')}",
        (
            "global_batch: "
            f"{batch.get('global_batch_size')} = {batch.get('world_size')} * "
            f"{batch.get('micro_batch_size')} * {batch.get('gradient_accumulation_steps')}"
        ),
        f"runner_status: {status.get('runner_status')}",
        f"training_runtime_ported: {status.get('training_runtime_ported')}",
        f"trainer_runtime_contract_status: {status.get('trainer_runtime_contract_status')}",
        f"will_launch_training: {status.get('will_launch_training')}",
        f"legacy_reference_allowed: {status.get('legacy_reference_allowed')}",
    ]
    blocking = status.get("blocking_items") or []
    if blocking:
        lines.append("blocking_items:")
        lines.extend(f"- {item}" for item in blocking)
    return "\n".join(lines) + "\n"


def _trainer_runtime_contract(
    *,
    expected_stage: TrainingStage,
    plan: dict[str, Any],
    native_status: dict[str, Any],
) -> dict[str, Any]:
    return {
        "contract_schema_version": "clean_trainer_runtime_contract_v1",
        "stage": str(expected_stage),
        "entrypoint": f"revisit_vlm_clean.training.{expected_stage.value}_executor",
        "launch_function": "launch_training",
        "status": "not_ported",
        "launch_permitted": False,
        "global_batch_size": (plan.get("batch") or {}).get("global_batch_size"),
        "required_launch_gates": _required_launch_gates(expected_stage),
        "required_runtime_artifacts": [
            "trainable_parameters.json",
            "optimizer_groups.json",
            "first_batch_identity.json",
            "checkpoint_contract.json",
        ],
        "must_emit_before_first_optimizer_step": [
            "git_and_plan_identity",
            "dataset_file_hashes",
            "trainable_parameter_names",
            "frozen_parameter_summary",
            "global_batch_math",
        ],
        "blocking_items": list(native_status.get("blocking_items") or []),
    }


def _required_launch_gates(stage: TrainingStage) -> list[str]:
    common = [
        "load_model_and_processor",
        "ensure_protocol_token_rows",
        "set_training_use_cache_false",
        "build_dataset_loader_from_plan_identity",
        "construct_optimizer_and_scheduler_from_plan",
        "emit_trainable_parameter_audit",
        "save_checkpoint_with_clean_contract",
    ]
    if stage == TrainingStage.STAGE1:
        return [
            *common,
            "build_tgvf_module_from_stage1_plan",
            "stage1_readout_context_uses_qwen_v_merge",
            "stage1_position_ids_use_real_qwen3_mrope",
            "stage1_matrix_ce_and_manifold_losses_match_plan",
        ]
    return [
        *common,
        "load_stage1_checkpoint_tgvf_and_protocol_rows",
        "attach_lora_modules_from_plan",
        "use_fast_batched_stage2_path",
        "apply_weighted_span_losses_from_plan",
        "apply_original_image_mask_scope_from_plan",
        "apply_deepstack_training_scope_when_enabled",
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_training_plan(plan: dict[str, Any], *, expected_stage: TrainingStage) -> None:
    schema_version = plan.get("training_plan_schema_version")
    if schema_version != TRAINING_PLAN_SCHEMA_VERSION:
        raise ValueError(
            "training_plan_schema_version mismatch: "
            f"{schema_version!r} != {TRAINING_PLAN_SCHEMA_VERSION!r}"
        )
    stage = TrainingStage(str(plan.get("stage")))
    if stage != expected_stage:
        raise ValueError(f"training plan stage mismatch: {stage.value} != {expected_stage.value}")
    _validate_batch(plan.get("batch") or {})
    _validate_dataset(plan.get("dataset") or {}, stage=stage)
    if stage == TrainingStage.STAGE1:
        _validate_stage1_readout_context(plan.get("readout_context") or {})
    _validate_module_policy(plan.get("module_policy") or {}, stage=stage)
    _validate_prepare_command(plan.get("clean_prepare_execution_command") or {}, stage=stage)
    _validate_clean_command(plan.get("clean_training_command") or {}, stage=stage)
    legacy = plan.get("legacy_reference_command") or {}
    if legacy.get("final_clean_native") is not False:
        raise ValueError("legacy_reference_command must be marked final_clean_native=false")
    if legacy.get("executable") is not False:
        raise ValueError("legacy_reference_command must be non-executable in clean plans")
    native = plan.get("clean_native_training") or {}
    if native.get("required_for_final_clean_project") is not True:
        raise ValueError("clean_native_training must be required for the final clean project")


def _validate_batch(batch: dict[str, Any]) -> None:
    world_size = int(batch.get("world_size") or 0)
    micro_batch = int(batch.get("micro_batch_size") or 0)
    accum = int(batch.get("gradient_accumulation_steps") or 0)
    global_batch = int(batch.get("global_batch_size") or 0)
    if min(world_size, micro_batch, accum, global_batch) < 1:
        raise ValueError("batch identity must contain positive world/micro/accum/global values")
    resolved = world_size * micro_batch * accum
    if resolved != global_batch:
        raise ValueError(
            "batch identity mismatch: "
            f"{world_size} * {micro_batch} * {accum} = {resolved}, got {global_batch}"
        )


def _validate_dataset(dataset: dict[str, Any], *, stage: TrainingStage) -> None:
    _validate_file_identity(dataset.get("train_file") or {}, label="train_file")
    if stage == TrainingStage.STAGE2:
        _validate_file_identity(
            dataset.get("stage1_checkpoint") or {},
            label="stage1_checkpoint",
        )
        val_file = dataset.get("val_file")
        if val_file is not None:
            _validate_file_identity(val_file, label="val_file")


def _validate_file_identity(identity: dict[str, Any], *, label: str) -> None:
    if identity.get("exists") is not True:
        raise ValueError(f"{label} identity must exist")
    path = identity.get("path")
    if not path or not Path(str(path)).exists():
        raise ValueError(f"{label} path recorded in identity does not exist: {path}")
    if not identity.get("sha256"):
        raise ValueError(f"{label} identity must include sha256")
    if identity.get("size_bytes") is None:
        raise ValueError(f"{label} identity must include size_bytes")


def _validate_module_policy(policy: dict[str, Any], *, stage: TrainingStage) -> None:
    if not policy.get("trainable"):
        raise ValueError("module_policy.trainable must not be empty")
    if not policy.get("frozen"):
        raise ValueError("module_policy.frozen must not be empty")
    runtime = policy.get("training_runtime") or {}
    if runtime.get("use_cache") is not False:
        raise ValueError("training runtime must record use_cache=false")
    if runtime.get("print_trainable_parameter_names_before_launch") is not True:
        raise ValueError("training runtime must require trainable-parameter printing")
    if stage == TrainingStage.STAGE2 and runtime.get("gradient_checkpointing") is not True:
        raise ValueError("Stage2 training runtime must record gradient_checkpointing=true")


def _validate_stage1_readout_context(context: dict[str, Any]) -> None:
    expected = {
        "original_image_placeholder_embeddings": "replace_with_qwen_v_merge",
        "d_append_path": "native_qwen_visual_span",
        "position_ids": "real_qwen3_mrope_full_trajectory",
        "fvt_position_mode": "native_source_grid",
        "d_token_count": "dynamic_source_image_visual_token_count",
        "visual_merger_path": "frozen_finalize_path",
    }
    for key, value in expected.items():
        if context.get(key) != value:
            raise ValueError(f"Stage1 readout_context.{key} must be {value!r}")
    attention_mask = context.get("attention_mask") or {}
    if attention_mask.get("mask_original_image_after_tgvf") is not True:
        raise ValueError("Stage1 readout_context attention mask must enable original-image masking")
    if (
        attention_mask.get("blocking")
        != "weak_strict_original_image_key_blocking_after_tgvf_append"
    ):
        raise ValueError("Stage1 readout_context attention mask blocking identity mismatch")


def _validate_prepare_command(command: dict[str, Any], *, stage: TrainingStage) -> None:
    if command.get("final_clean_native") is not True:
        raise ValueError("clean_prepare_execution_command must be final clean-native")
    if command.get("executable") is not True:
        raise ValueError("clean_prepare_execution_command must be executable")
    if command.get("will_launch_training") is not False:
        raise ValueError("clean_prepare_execution_command must not launch training")
    if command.get("status") != "prepare_execution_supported":
        raise ValueError("clean_prepare_execution_command status mismatch")
    expected_entrypoint = f"revisit_vlm_clean.training.{stage.value}_executor"
    if command.get("planned_entrypoint") != expected_entrypoint:
        raise ValueError(
            "clean_prepare_execution_command planned_entrypoint mismatch: "
            f"{command.get('planned_entrypoint')!r} != {expected_entrypoint!r}"
        )
    argv = list(command.get("argv") or [])
    if "--prepare-execution" not in argv:
        raise ValueError("clean_prepare_execution_command must include --prepare-execution")


def _validate_clean_command(command: dict[str, Any], *, stage: TrainingStage) -> None:
    if command.get("final_clean_native") is not True:
        raise ValueError("clean_training_command must be marked final_clean_native=true")
    if command.get("executable") is not False:
        raise ValueError("clean_training_command must remain non-executable until ported")
    expected_entrypoint = f"revisit_vlm_clean.training.{stage.value}_executor"
    if command.get("planned_entrypoint") != expected_entrypoint:
        raise ValueError(
            "clean_training_command planned_entrypoint mismatch: "
            f"{command.get('planned_entrypoint')!r} != {expected_entrypoint!r}"
        )
