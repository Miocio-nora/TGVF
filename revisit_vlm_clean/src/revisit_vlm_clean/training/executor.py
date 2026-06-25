"""Fail-safe clean training executor preflight.

The real Stage1/Stage2 training loops are not ported yet. This module makes the
planned clean entrypoints importable and validates training-plan identity before
any future executor is allowed to launch.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from hashlib import sha256
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
    parser.add_argument(
        "--audit-runtime",
        action="store_true",
        help=(
            "Validate an execution bundle plus runtime artifacts and write a "
            "clean runtime audit report. This does not launch training."
        ),
    )
    parser.add_argument(
        "--bundle",
        default=None,
        help=(
            "Execution bundle to audit. Defaults to "
            "<execution-dir>/clean_training_execution_bundle.json."
        ),
    )
    parser.add_argument(
        "--runtime-audit-report",
        default=None,
        help=(
            "Path for --audit-runtime report. Defaults to "
            "<execution-dir>/clean_training_runtime_audit.json."
        ),
    )
    parser.add_argument(
        "--audit-model-parameters",
        action="store_true",
        help=(
            "During --audit-runtime, load the planned model components and replace "
            "the placeholder trainable-parameter artifact with an actual parameter audit. "
            "This can be expensive and still does not launch training."
        ),
    )
    parser.add_argument(
        "--audit-optimizer",
        action="store_true",
        help=(
            "During --audit-runtime, load model components if needed, construct the "
            "planned AdamW optimizer and scheduler, and write optimizer_runtime.json. "
            "This still does not run optimizer steps or launch training."
        ),
    )
    parser.add_argument(
        "--audit-checkpoint",
        action="store_true",
        help=(
            "During --audit-runtime, save and reload a clean checkpoint probe from "
            "loaded modules plus optimizer/scheduler state, then write "
            "checkpoint_runtime.json. This still does not launch training."
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
        if args.audit_runtime:
            audit = audit_training_runtime(
                bundle_path=prepared["execution_bundle"],
                expected_stage=stage,
                report_path=args.runtime_audit_report,
                audit_model_parameters=args.audit_model_parameters,
                audit_optimizer=args.audit_optimizer,
                audit_checkpoint=args.audit_checkpoint,
            )
            prepared["runtime_audit"] = audit["runtime_audit"]
            prepared["trainable_parameters"] = audit["trainable_parameters"]
            if audit.get("optimizer_runtime"):
                prepared["optimizer_runtime"] = audit["optimizer_runtime"]
            if audit.get("checkpoint_runtime"):
                prepared["checkpoint_runtime"] = audit["checkpoint_runtime"]
        print_json(prepared)
        return 0
    if args.audit_runtime:
        bundle_path = _execution_bundle_path(
            plan_path=plan_path,
            execution_dir=args.execution_dir,
            requested=args.bundle,
        )
        audit = audit_training_runtime(
            bundle_path=bundle_path,
            expected_stage=stage,
            report_path=args.runtime_audit_report,
            audit_model_parameters=args.audit_model_parameters,
            audit_optimizer=args.audit_optimizer,
            audit_checkpoint=args.audit_checkpoint,
        )
        audit["preflight_report"] = str(report_path)
        print_json(audit)
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
    dataset_runtime = _write_dataset_runtime_artifacts(
        execution_dir=out,
        plan=plan,
        expected_stage=expected_stage,
    )
    checkpoint_contract = _write_checkpoint_contract_artifact(
        execution_dir=out,
        plan=plan,
        expected_stage=expected_stage,
    )
    optimizer_groups = _write_optimizer_groups_artifact(
        execution_dir=out,
        plan=plan,
        expected_stage=expected_stage,
    )
    bundle["runtime_artifacts"] = {
        **dataset_runtime["paths"],
        "checkpoint_contract": checkpoint_contract["path"],
        "optimizer_groups": optimizer_groups["path"],
    }
    bundle["dataset_runtime"] = dataset_runtime["dataset_runtime"]
    bundle["first_batch_identity"] = dataset_runtime["first_batch_identity"]
    bundle["checkpoint_contract"] = checkpoint_contract["contract"]
    bundle["optimizer_groups"] = optimizer_groups["contract"]
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
        "dataset_runtime_identity": dataset_runtime["paths"]["dataset_runtime_identity"],
        "first_batch_identity": dataset_runtime["paths"]["first_batch_identity"],
        "checkpoint_contract": checkpoint_contract["path"],
        "optimizer_groups": optimizer_groups["path"],
        "runner_status": status["runner_status"],
    }


def audit_training_runtime(
    *,
    bundle_path: str | Path,
    expected_stage: TrainingStage,
    report_path: str | Path | None = None,
    audit_model_parameters: bool = False,
    audit_optimizer: bool = False,
    audit_checkpoint: bool = False,
) -> dict[str, Any]:
    bundle_file = Path(bundle_path)
    if not bundle_file.exists():
        raise FileNotFoundError(f"training execution bundle does not exist: {bundle_file}")
    bundle = json.loads(bundle_file.read_text(encoding="utf-8"))
    _validate_execution_bundle(bundle, expected_stage=expected_stage)
    execution_dir = Path(str(bundle.get("execution_dir") or bundle_file.parent))
    artifacts = _load_runtime_artifacts(bundle, expected_stage=expected_stage)
    loaded_modules = (
        _load_training_parameter_audit_modules(bundle, expected_stage=expected_stage)
        if audit_model_parameters or audit_optimizer or audit_checkpoint
        else None
    )
    trainable_parameters = (
        _write_actual_trainable_parameters_audit(
            execution_dir=execution_dir,
            bundle=bundle,
            expected_stage=expected_stage,
            loaded_modules=loaded_modules,
        )
        if audit_model_parameters or audit_optimizer or audit_checkpoint
        else _write_trainable_parameters_placeholder(
            execution_dir=execution_dir,
            bundle=bundle,
        )
    )
    optimizer_runtime = (
        _write_actual_optimizer_scheduler_audit(
            execution_dir=execution_dir,
            bundle=bundle,
            artifacts=artifacts,
            loaded_modules=loaded_modules,
            expected_stage=expected_stage,
        )
        if audit_optimizer or audit_checkpoint
        else None
    )
    checkpoint_runtime = (
        _write_actual_checkpoint_runtime_audit(
            execution_dir=execution_dir,
            bundle=bundle,
            loaded_modules=loaded_modules,
            optimizer_runtime=optimizer_runtime,
            expected_stage=expected_stage,
        )
        if audit_checkpoint
        else None
    )
    audit = _runtime_audit_report(
        bundle_path=bundle_file,
        bundle=bundle,
        artifacts=artifacts,
        trainable_parameters=trainable_parameters,
        optimizer_runtime=optimizer_runtime,
        checkpoint_runtime=checkpoint_runtime,
        expected_stage=expected_stage,
    )
    resolved_report_path = (
        Path(report_path)
        if report_path is not None
        else execution_dir / "clean_training_runtime_audit.json"
    )
    _write_json(resolved_report_path, audit)
    text_path = resolved_report_path.with_suffix(".txt")
    text_path.write_text(_runtime_audit_text(audit), encoding="utf-8")
    status_path = execution_dir / "clean_training_runtime_audit_status.json"
    _write_json(status_path, _runtime_audit_status(audit))
    result = {
        "runtime_audit": str(resolved_report_path),
        "runtime_audit_txt": str(text_path),
        "runtime_audit_status": str(status_path),
        "trainable_parameters": trainable_parameters["path"],
        "stage": str(expected_stage),
        "run_id": bundle.get("run_id"),
        "will_launch_training": False,
        "training_runtime_ported": False,
        "runner_status": audit["status"],
    }
    if optimizer_runtime is not None:
        result["optimizer_runtime"] = optimizer_runtime["path"]
    if checkpoint_runtime is not None:
        result["checkpoint_runtime"] = checkpoint_runtime["path"]
    return result


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


def _execution_bundle_path(
    *,
    plan_path: str | Path,
    execution_dir: str | Path | None,
    requested: str | Path | None,
) -> Path:
    if requested:
        return Path(requested)
    return _execution_dir_path(plan_path=plan_path, requested=execution_dir) / (
        "clean_training_execution_bundle.json"
    )


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
        "checkpoint_contract_status": (bundle.get("checkpoint_contract") or {}).get("status"),
        "optimizer_groups_status": (bundle.get("optimizer_groups") or {}).get("status"),
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
        f"checkpoint_contract_status: {status.get('checkpoint_contract_status')}",
        f"optimizer_groups_status: {status.get('optimizer_groups_status')}",
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
            "dataset_runtime_identity.json",
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


def _validate_execution_bundle(bundle: dict[str, Any], *, expected_stage: TrainingStage) -> None:
    if bundle.get("training_execution_bundle_schema_version") != (
        TRAINING_EXECUTION_BUNDLE_SCHEMA_VERSION
    ):
        raise ValueError("training execution bundle schema mismatch")
    stage = TrainingStage(str(bundle.get("stage")))
    if stage != expected_stage:
        raise ValueError(
            f"execution bundle stage mismatch: {stage.value} != {expected_stage.value}"
        )
    plan_path = Path(str(bundle.get("plan_path") or ""))
    if not plan_path.exists():
        raise ValueError(f"execution bundle plan_path does not exist: {plan_path}")
    recorded_plan_sha = (bundle.get("plan_identity") or {}).get("sha256")
    current_plan_sha = file_identity(plan_path).sha256
    if recorded_plan_sha != current_plan_sha:
        raise ValueError("execution bundle plan identity no longer matches plan_path")
    executor = bundle.get("clean_executor") or {}
    if executor.get("final_clean_native") is not True:
        raise ValueError("execution bundle clean_executor must be final clean-native")
    if executor.get("owns_execution_bundle") is not True:
        raise ValueError("execution bundle must be owned by the clean executor")
    if executor.get("will_launch_training") is not False:
        raise ValueError("execution bundle must not launch training during audit")
    safety = bundle.get("safety") or {}
    if safety.get("will_launch_training") is not False:
        raise ValueError("execution bundle safety must keep will_launch_training=false")
    if safety.get("legacy_reference_allowed") is not False:
        raise ValueError("execution bundle must not allow legacy reference execution")


def _load_runtime_artifacts(
    bundle: dict[str, Any],
    *,
    expected_stage: TrainingStage,
) -> dict[str, dict[str, Any]]:
    artifact_paths = bundle.get("runtime_artifacts") or {}
    required = {
        "dataset_runtime_identity",
        "first_batch_identity",
        "checkpoint_contract",
        "optimizer_groups",
    }
    missing = sorted(required - set(artifact_paths))
    if missing:
        raise ValueError(f"execution bundle missing runtime artifact paths: {missing}")
    artifacts: dict[str, dict[str, Any]] = {}
    for name in sorted(required):
        path = Path(str(artifact_paths[name]))
        if not path.exists():
            raise FileNotFoundError(f"runtime artifact does not exist: {name}: {path}")
        artifacts[name] = json.loads(path.read_text(encoding="utf-8"))
    _validate_runtime_artifact_payloads(artifacts, expected_stage=expected_stage)
    return artifacts


def _validate_runtime_artifact_payloads(
    artifacts: dict[str, dict[str, Any]],
    *,
    expected_stage: TrainingStage,
) -> None:
    for name in ("dataset_runtime_identity", "first_batch_identity", "optimizer_groups"):
        if artifacts[name].get("stage") != str(expected_stage):
            raise ValueError(f"{name} stage mismatch")
    checkpoint = artifacts["checkpoint_contract"]
    if checkpoint.get("stage") != str(expected_stage):
        raise ValueError("checkpoint_contract stage mismatch")
    expected_checkpoint_status = (
        "no_input_checkpoint_required"
        if expected_stage == TrainingStage.STAGE1
        else "validated"
    )
    if checkpoint.get("status") != expected_checkpoint_status:
        raise ValueError(
            "checkpoint_contract status mismatch: "
            f"{checkpoint.get('status')!r} != {expected_checkpoint_status!r}"
        )
    if artifacts["optimizer_groups"].get("status") != "validated":
        raise ValueError("optimizer_groups must be validated before runtime audit")
    dataset = artifacts["dataset_runtime_identity"]
    first_batch = artifacts["first_batch_identity"]
    if dataset.get("global_batch_size") != first_batch.get("requested_global_batch_size"):
        raise ValueError("dataset runtime and first batch global batch mismatch")
    if int(first_batch.get("materialized_batch_size") or 0) < 1:
        raise ValueError("first_batch_identity must contain at least one row")


def _write_trainable_parameters_placeholder(
    *,
    execution_dir: Path,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    module_policy = bundle.get("module_policy") or {}
    payload = {
        "schema_version": "clean_trainable_parameters_audit_v1",
        "stage": bundle.get("stage"),
        "run_id": bundle.get("run_id"),
        "status": "pending_model_load_not_actual_parameter_audit",
        "actual_model_parameters_loaded": False,
        "must_be_replaced_before_first_optimizer_step": True,
        "expected_trainable_policy": list(module_policy.get("trainable") or []),
        "expected_frozen_policy": list(module_policy.get("frozen") or []),
        "visual_merger": module_policy.get("visual_merger"),
        "training_runtime": module_policy.get("training_runtime"),
        "notes": [
            "clean runtime audit has not loaded the model",
            "the real trainer loop must overwrite this file with actual named parameters",
        ],
    }
    path = execution_dir / "trainable_parameters.json"
    _write_json(path, payload)
    return {"path": str(path), "payload": payload}


def _write_actual_checkpoint_runtime_audit(
    *,
    execution_dir: Path,
    bundle: dict[str, Any],
    loaded_modules: dict[str, Any] | None,
    optimizer_runtime: dict[str, Any] | None,
    expected_stage: TrainingStage,
) -> dict[str, Any]:
    if loaded_modules is None:
        raise ValueError("checkpoint audit requires loaded model modules")
    if optimizer_runtime is None:
        raise ValueError("checkpoint audit requires optimizer runtime construction")
    optimizer = optimizer_runtime.get("optimizer")
    scheduler = optimizer_runtime.get("scheduler")
    if optimizer is None or scheduler is None:
        raise ValueError("checkpoint audit requires optimizer and scheduler objects")
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("checkpoint audit requires torch") from exc

    modules = dict(loaded_modules.get("modules") or {})
    checkpoint_path = execution_dir / "checkpoint_runtime_probe.pt"
    checkpoint = _checkpoint_probe_payload(
        bundle=bundle,
        loaded_modules=loaded_modules,
        modules=modules,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_stage=expected_stage,
    )
    torch.save(checkpoint, checkpoint_path)
    loaded = _torch_load_checkpoint_probe(torch, checkpoint_path)
    required_keys = _checkpoint_required_output_keys(expected_stage)
    missing_keys = sorted(key for key in required_keys if key not in loaded)
    if missing_keys:
        raise ValueError(f"checkpoint runtime probe missing keys after load: {missing_keys}")
    _reload_optimizer_scheduler_probe(
        optimizer=optimizer,
        scheduler=scheduler,
        optimizer_state=loaded.get("optimizer"),
        scheduler_state=loaded.get("scheduler"),
    )
    state_checks = _checkpoint_state_checks(
        checkpoint=checkpoint,
        loaded=loaded,
        expected_stage=expected_stage,
    )
    state_checks_ok = all(check.get("ok") is True for check in state_checks.values())
    payload = {
        "schema_version": "clean_training_checkpoint_runtime_audit_v1",
        "stage": bundle.get("stage"),
        "run_id": bundle.get("run_id"),
        "status": "actual_checkpoint_save_load_audit",
        "actual_checkpoint_saved": True,
        "actual_checkpoint_loaded": True,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_identity": file_identity(checkpoint_path).to_dict(),
        "required_output_keys": required_keys,
        "saved_key_names": sorted(str(key) for key in checkpoint.keys()),
        "loaded_key_names": sorted(str(key) for key in loaded.keys()),
        "missing_required_keys": missing_keys,
        "global_step": loaded.get("global_step"),
        "micro_step": loaded.get("micro_step"),
        "optimizer_state_loaded": loaded.get("optimizer") is not None,
        "scheduler_state_loaded": loaded.get("scheduler") is not None,
        "state_checks_ok": state_checks_ok,
        "state_checks": state_checks,
        "protocol_c_token_rows": _protocol_token_rows_summary(
            loaded.get("protocol_c_token_rows")
        ),
        "notes": [
            "checkpoint probe was saved and reloaded without launching training",
            (
                "this audit does not call backward, optimizer.step, "
                "scheduler.step, or publish a training checkpoint"
            ),
        ],
    }
    path = execution_dir / "checkpoint_runtime.json"
    _write_json(path, payload)
    return {"path": str(path), "payload": payload, "checkpoint_path": str(checkpoint_path)}


def _checkpoint_probe_payload(
    *,
    bundle: dict[str, Any],
    loaded_modules: dict[str, Any],
    modules: dict[str, Any],
    optimizer: Any,
    scheduler: Any,
    expected_stage: TrainingStage,
) -> dict[str, Any]:
    tgvf_module = modules.get("tgvf")
    if tgvf_module is None or not hasattr(tgvf_module, "state_dict"):
        raise ValueError("checkpoint audit requires a tgvf module with state_dict")
    config = _checkpoint_runtime_config(bundle=bundle, expected_stage=expected_stage)
    checkpoint: dict[str, Any] = {
        "tgvf_module": tgvf_module.state_dict(),
        "config": config,
        "global_step": 0,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    if expected_stage == TrainingStage.STAGE1:
        checkpoint["optimizer_step"] = 0
        token_rows = _protocol_token_rows_payload_from_loaded(
            bundle=bundle,
            loaded_modules=loaded_modules,
            modules=modules,
        )
        if _protocol_c_rows_required(bundle) and token_rows is None:
            raise ValueError("Stage1 checkpoint audit requires protocol_c_token_rows")
        if token_rows is not None:
            checkpoint["protocol_c_token_rows"] = token_rows
        return checkpoint

    qwen_lora = modules.get("qwen_lora")
    if qwen_lora is None or not hasattr(qwen_lora, "state_dict"):
        raise ValueError("Stage2 checkpoint audit requires qwen_lora module with state_dict")
    checkpoint["qwen_lora"] = _qwen_lora_state_dict(qwen_lora)
    checkpoint["micro_step"] = 0
    return checkpoint


def _checkpoint_runtime_config(
    *,
    bundle: dict[str, Any],
    expected_stage: TrainingStage,
) -> dict[str, Any]:
    model = bundle.get("model") or {}
    training = bundle.get("training") or {}
    return {
        "stage": str(expected_stage),
        "run_id": bundle.get("run_id"),
        "model_id": model.get("model_id"),
        "processor_id": model.get("processor_id"),
        "tgvf_protocol": bundle.get("protocol"),
        "training": training,
        "tgvf": {
            "variant": training.get("variant"),
            "num_foveated_tokens": None,
        },
    }


def _checkpoint_required_output_keys(expected_stage: TrainingStage) -> list[str]:
    if expected_stage == TrainingStage.STAGE1:
        return [
            "tgvf_module",
            "config",
            "global_step",
            "optimizer_step",
            "optimizer",
            "scheduler",
        ]
    return [
        "qwen_lora",
        "tgvf_module",
        "config",
        "global_step",
        "micro_step",
        "optimizer",
        "scheduler",
    ]


def _torch_load_checkpoint_probe(torch_module: Any, checkpoint_path: Path) -> dict[str, Any]:
    try:
        loaded = torch_module.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch_module.load(checkpoint_path, map_location="cpu")
    if not isinstance(loaded, dict):
        raise ValueError("checkpoint runtime probe did not load to a mapping")
    return loaded


def _reload_optimizer_scheduler_probe(
    *,
    optimizer: Any,
    scheduler: Any,
    optimizer_state: Any,
    scheduler_state: Any,
) -> None:
    if optimizer_state is None:
        raise ValueError("checkpoint runtime probe missing optimizer state")
    if scheduler_state is None:
        raise ValueError("checkpoint runtime probe missing scheduler state")
    optimizer.load_state_dict(optimizer_state)
    scheduler.load_state_dict(scheduler_state)


def _checkpoint_state_checks(
    *,
    checkpoint: dict[str, Any],
    loaded: dict[str, Any],
    expected_stage: TrainingStage,
) -> dict[str, Any]:
    checks = {
        "tgvf_module": _state_dict_parity_check(
            checkpoint.get("tgvf_module"),
            loaded.get("tgvf_module"),
        ),
    }
    if expected_stage == TrainingStage.STAGE2:
        checks["qwen_lora"] = _state_dict_parity_check(
            checkpoint.get("qwen_lora"),
            loaded.get("qwen_lora"),
        )
    return checks


def _state_dict_parity_check(left: Any, right: Any) -> dict[str, Any]:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return {"ok": False, "reason": "state_dict_not_mapping"}
    left_keys = sorted(str(key) for key in left)
    right_keys = sorted(str(key) for key in right)
    mismatched_shapes = []
    for key in left:
        if key not in right:
            continue
        if _shape_list(left[key]) != _shape_list(right[key]):
            mismatched_shapes.append(
                {
                    "key": str(key),
                    "saved": _shape_list(left[key]),
                    "loaded": _shape_list(right[key]),
                }
            )
    return {
        "ok": left_keys == right_keys and not mismatched_shapes,
        "saved_tensor_count": len(left_keys),
        "loaded_tensor_count": len(right_keys),
        "key_sets_match": left_keys == right_keys,
        "mismatched_shapes": mismatched_shapes,
        "sample_keys": left_keys[:10],
    }


def _qwen_lora_state_dict(module: Any) -> dict[str, Any]:
    try:
        from peft import get_peft_model_state_dict
    except Exception:
        return dict(module.state_dict())
    try:
        state = get_peft_model_state_dict(module)
    except Exception:
        return dict(module.state_dict())
    return dict(state or {})


def _protocol_c_rows_required(bundle: dict[str, Any]) -> bool:
    return str(bundle.get("protocol") or "").startswith("protocol_c_")


def _protocol_token_rows_payload_from_loaded(
    *,
    bundle: dict[str, Any],
    loaded_modules: dict[str, Any],
    modules: dict[str, Any],
) -> dict[str, Any] | None:
    extras = loaded_modules.get("checkpoint_extras") or {}
    if extras.get("protocol_c_token_rows") is not None:
        return extras["protocol_c_token_rows"]
    if not _protocol_c_rows_required(bundle):
        return None
    try:
        import torch

        from revisit_vlm.qwen3_vl_tgvf import protocol_special_token_ids, protocol_special_tokens
    except Exception:
        return None
    processor = loaded_modules.get("processor")
    tokenizer = loaded_modules.get("tokenizer") or getattr(processor, "tokenizer", None)
    model = modules.get("qwen") or modules.get("qwen_lora")
    if tokenizer is None or model is None or not hasattr(model, "get_input_embeddings"):
        return None
    protocol = str(bundle.get("protocol"))
    tokens = protocol_special_tokens(protocol)
    token_ids = protocol_special_token_ids(tokenizer, protocol=protocol)
    ordered_ids = [int(token_ids[token]) for token in tokens]
    input_embed = model.get_input_embeddings()
    output_embed = (
        model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    )
    with torch.no_grad():
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
    return payload


def _write_actual_trainable_parameters_audit(
    *,
    execution_dir: Path,
    bundle: dict[str, Any],
    expected_stage: TrainingStage,
    loaded_modules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    loaded = (
        loaded_modules
        if loaded_modules is not None
        else _load_training_parameter_audit_modules(bundle, expected_stage=expected_stage)
    )
    modules = dict(loaded.get("modules") or {})
    if not modules:
        raise ValueError("model parameter audit loader returned no modules")
    module_summaries = {
        name: _parameter_module_summary(module, prefix=f"{name}.")
        for name, module in sorted(modules.items())
    }
    trainable_names = [
        parameter["name"]
        for summary in module_summaries.values()
        for parameter in summary["parameters"]
        if parameter["requires_grad"]
    ]
    frozen_names = [
        parameter["name"]
        for summary in module_summaries.values()
        for parameter in summary["parameters"]
        if not parameter["requires_grad"]
    ]
    payload = {
        "schema_version": "clean_trainable_parameters_audit_v1",
        "stage": bundle.get("stage"),
        "run_id": bundle.get("run_id"),
        "status": "actual_model_parameter_audit",
        "actual_model_parameters_loaded": True,
        "must_be_replaced_before_first_optimizer_step": False,
        "loader": loaded.get("loader") or {},
        "expected_trainable_policy": list(
            (bundle.get("module_policy") or {}).get("trainable") or []
        ),
        "expected_frozen_policy": list((bundle.get("module_policy") or {}).get("frozen") or []),
        "visual_merger": (bundle.get("module_policy") or {}).get("visual_merger"),
        "training_runtime": (bundle.get("module_policy") or {}).get("training_runtime"),
        "module_summaries": module_summaries,
        "trainable_parameter_names": trainable_names,
        "frozen_parameter_names_sample": frozen_names[:200],
        "trainable_tensor_count": len(trainable_names),
        "frozen_tensor_count": len(frozen_names),
        "trainable_numel": sum(summary["trainable_numel"] for summary in module_summaries.values()),
        "frozen_numel": sum(summary["frozen_numel"] for summary in module_summaries.values()),
        "notes": [
            "actual model components were loaded for parameter audit",
            "this audit still does not run optimizer steps or save training checkpoints",
        ],
    }
    path = execution_dir / "trainable_parameters.json"
    _write_json(path, payload)
    return {"path": str(path), "payload": payload}


def _write_actual_optimizer_scheduler_audit(
    *,
    execution_dir: Path,
    bundle: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    loaded_modules: dict[str, Any] | None,
    expected_stage: TrainingStage,
) -> dict[str, Any]:
    if loaded_modules is None:
        raise ValueError("optimizer audit requires loaded model modules")
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("optimizer audit requires torch") from exc

    optimizer_contract = artifacts["optimizer_groups"]
    group_specs = _actual_optimizer_group_specs(
        loaded_modules=loaded_modules,
        optimizer_contract=optimizer_contract,
        expected_stage=expected_stage,
    )
    constructed_specs = [spec for spec in group_specs if spec["params"]]
    if not constructed_specs:
        raise ValueError("optimizer audit found no trainable parameters for any planned group")
    optimizer_kwargs = _adamw_kwargs_from_contract(optimizer_contract)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": spec["params"],
                "lr": spec["lr"],
                "name": spec["name"],
                "weight_decay": spec["weight_decay"],
            }
            for spec in constructed_specs
        ],
        **optimizer_kwargs,
    )
    scheduler = _build_clean_lr_scheduler(
        torch_module=torch,
        optimizer=optimizer,
        scheduler_contract=optimizer_contract.get("scheduler") or {},
        max_steps=int(((bundle.get("training") or {}).get("max_steps")) or 0),
    )
    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()
    constructed_groups = [
        {
            "name": spec["name"],
            "lr": spec["lr"],
            "weight_decay": spec["weight_decay"],
            "parameter_names": spec["parameter_names"],
            "parameter_tensor_count": len(spec["params"]),
            "parameter_numel": sum(_numel(parameter) for parameter in spec["params"]),
        }
        for spec in constructed_specs
    ]
    empty_groups = [spec["name"] for spec in group_specs if not spec["params"]]
    payload = {
        "schema_version": "clean_training_optimizer_scheduler_audit_v1",
        "stage": bundle.get("stage"),
        "run_id": bundle.get("run_id"),
        "status": "actual_optimizer_scheduler_audit",
        "actual_optimizer_constructed": True,
        "actual_scheduler_constructed": True,
        "optimizer": {
            "name": "adamw",
            "param_group_count": len(optimizer.param_groups),
            "state_entry_count": len(optimizer_state.get("state") or {}),
            "betas": list(optimizer.param_groups[0].get("betas")),
            "eps": optimizer.param_groups[0].get("eps"),
            "weight_decay": optimizer.param_groups[0].get("weight_decay"),
        },
        "scheduler": {
            "name": (optimizer_contract.get("scheduler") or {}).get("name"),
            "warmup_steps": (optimizer_contract.get("scheduler") or {}).get("warmup_steps"),
            "min_lr_ratio": (optimizer_contract.get("scheduler") or {}).get("min_lr_ratio"),
            "state_dict_keys": sorted(str(key) for key in scheduler_state.keys()),
            "last_lr": list(scheduler.get_last_lr()),
        },
        "planned_group_names": list(optimizer_contract.get("group_names") or []),
        "constructed_group_names": [group["name"] for group in constructed_groups],
        "empty_planned_group_names": empty_groups,
        "constructed_groups": constructed_groups,
        "notes": [
            "actual optimizer and scheduler were constructed from loaded model parameters",
            (
                "this audit does not call backward, optimizer.step, "
                "scheduler.step, or save checkpoints"
            ),
        ],
    }
    path = execution_dir / "optimizer_runtime.json"
    _write_json(path, payload)
    return {
        "path": str(path),
        "payload": payload,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }


def _actual_optimizer_group_specs(
    *,
    loaded_modules: dict[str, Any],
    optimizer_contract: dict[str, Any],
    expected_stage: TrainingStage,
) -> list[dict[str, Any]]:
    modules = dict(loaded_modules.get("modules") or {})
    groups_by_name = {
        str(group.get("name")): group for group in optimizer_contract.get("groups") or []
    }
    if expected_stage == TrainingStage.STAGE1:
        return [
            _optimizer_group_spec(
                name="tgvf_module",
                contract=groups_by_name["tgvf_module"],
                named_parameters=_named_trainable_parameters(modules.get("tgvf"), prefix="tgvf."),
            ),
            _optimizer_group_spec(
                name="protocol_c_token_rows",
                contract=groups_by_name.get("protocol_c_token_rows"),
                named_parameters=_named_trainable_parameters(modules.get("qwen"), prefix="qwen."),
            ),
        ]
    tgvf_named = _named_trainable_parameters(modules.get("tgvf"), prefix="tgvf.")
    tgvf_refiner = [
        item for item in tgvf_named if "calib" not in item[0] and "calibration" not in item[0]
    ]
    tgvf_calibration = [
        item for item in tgvf_named if "calib" in item[0] or "calibration" in item[0]
    ]
    return [
        _optimizer_group_spec(
            name="llm_lora",
            contract=groups_by_name["llm_lora"],
            named_parameters=_named_trainable_parameters(
                modules.get("qwen_lora"),
                prefix="qwen_lora.",
            ),
        ),
        _optimizer_group_spec(
            name="tgvf_refiner",
            contract=groups_by_name["tgvf_refiner"],
            named_parameters=tgvf_refiner,
        ),
        _optimizer_group_spec(
            name="fvt_calibration",
            contract=groups_by_name["fvt_calibration"],
            named_parameters=tgvf_calibration,
        ),
    ]


def _optimizer_group_spec(
    *,
    name: str,
    contract: dict[str, Any] | None,
    named_parameters: list[tuple[str, Any]],
) -> dict[str, Any]:
    if contract is None:
        return {
            "name": name,
            "lr": None,
            "weight_decay": None,
            "params": [],
            "parameter_names": [],
        }
    return {
        "name": name,
        "lr": contract.get("lr"),
        "weight_decay": contract.get("weight_decay"),
        "params": [parameter for _, parameter in named_parameters],
        "parameter_names": [parameter_name for parameter_name, _ in named_parameters],
    }


def _named_trainable_parameters(module: Any, *, prefix: str) -> list[tuple[str, Any]]:
    if module is None:
        return []
    if not hasattr(module, "named_parameters"):
        raise TypeError(f"optimizer audit object is not a torch-style module: {type(module)!r}")
    return [
        (prefix + str(name), parameter)
        for name, parameter in module.named_parameters()
        if bool(getattr(parameter, "requires_grad", False))
    ]


def _adamw_kwargs_from_contract(optimizer_contract: dict[str, Any]) -> dict[str, Any]:
    optimizer = optimizer_contract.get("optimizer") or {}
    betas = optimizer.get("betas")
    if betas is None:
        betas = (0.9, 0.999)
    eps = optimizer.get("eps")
    if eps is None:
        eps = 1e-8
    weight_decay = optimizer.get("weight_decay")
    if weight_decay is None:
        weight_decay = 0.01
    return {
        "betas": tuple(float(value) for value in betas),
        "eps": float(eps),
        "weight_decay": float(weight_decay),
    }


def _build_clean_lr_scheduler(
    *,
    torch_module: Any,
    optimizer: Any,
    scheduler_contract: dict[str, Any],
    max_steps: int,
) -> Any:
    if max_steps <= 0:
        raise ValueError("optimizer audit requires positive training.max_steps")
    scheduler_name = str(scheduler_contract.get("name") or "cosine")
    warmup_steps = max(0, int(scheduler_contract.get("warmup_steps") or 0))
    min_lr_ratio = float(scheduler_contract.get("min_lr_ratio") or 0.0)

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

    return torch_module.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _parameter_module_summary(module: Any, *, prefix: str) -> dict[str, Any]:
    if not hasattr(module, "named_parameters"):
        raise TypeError(f"parameter audit object is not a torch-style module: {type(module)!r}")
    parameters = []
    trainable_numel = 0
    frozen_numel = 0
    for name, parameter in module.named_parameters():
        full_name = prefix + str(name)
        numel = _numel(parameter)
        requires_grad = bool(getattr(parameter, "requires_grad", False))
        if requires_grad:
            trainable_numel += numel
        else:
            frozen_numel += numel
        parameters.append(
            {
                "name": full_name,
                "shape": _shape_list(parameter),
                "numel": numel,
                "requires_grad": requires_grad,
            }
        )
    return {
        "module_type": type(module).__name__,
        "parameter_count": len(parameters),
        "trainable_tensor_count": sum(1 for parameter in parameters if parameter["requires_grad"]),
        "frozen_tensor_count": sum(1 for parameter in parameters if not parameter["requires_grad"]),
        "trainable_numel": trainable_numel,
        "frozen_numel": frozen_numel,
        "parameters": parameters,
    }


def _load_training_parameter_audit_modules(
    bundle: dict[str, Any],
    *,
    expected_stage: TrainingStage,
) -> dict[str, Any]:
    if expected_stage == TrainingStage.STAGE1:
        return _load_stage1_parameter_audit_modules(bundle)
    return _load_stage2_parameter_audit_modules(bundle)


def _load_stage1_parameter_audit_modules(bundle: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch

        from revisit_vlm.qwen3_vl_tgvf import (
            PROTOCOL_C_THINKING_SPECIAL,
            PROTOCOL_C_TOOL_OBSERVATION,
            PROTOCOL_C_TOOL_OBSERVATION_QWEN2_NO_THINK,
            PROTOCOL_E_ACTION_EVIDENCE_SPECIAL,
            ensure_tgvf_protocol_tokens,
            load_qwen3_vl,
        )
        from revisit_vlm.tgvf_training import build_tgvf_module
        from revisit_vlm.tgvf_v3_stage1 import (
            TGVFv3Stage1Dataset,
            freeze_qwen_backbone,
            infer_qwen3_stage1_dims,
        )
        from scripts.train_tgvf_v3_stage1 import _enable_protocol_c_token_row_training
    except Exception as exc:
        raise RuntimeError("Stage1 model parameter audit dependencies are unavailable") from exc
    model_cfg = bundle.get("model") or {}
    training = bundle.get("training") or {}
    dataset = bundle.get("dataset") or {}
    train_file = ((dataset.get("train_file") or {}).get("path"))
    if not train_file:
        raise ValueError("Stage1 parameter audit requires dataset.train_file.path")
    samples = TGVFv3Stage1Dataset(train_file, focus_only=True)
    if len(samples) == 0:
        raise RuntimeError("Stage1 parameter audit found no focus samples")
    loaded = load_qwen3_vl(
        model_cfg.get("model_id"),
        processor_id=model_cfg.get("processor_id"),
        dtype=str(model_cfg.get("dtype") or "bfloat16"),
        device_map=model_cfg.get("device_map") or "auto",
        attn_implementation=model_cfg.get("attn_implementation"),
    )
    model = loaded.model
    processor = loaded.processor
    if getattr(processor.tokenizer, "pad_token", None) is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    protocol = str(bundle.get("protocol"))
    token_row_protocols = {
        PROTOCOL_C_THINKING_SPECIAL,
        PROTOCOL_C_TOOL_OBSERVATION,
        PROTOCOL_C_TOOL_OBSERVATION_QWEN2_NO_THINK,
        PROTOCOL_E_ACTION_EVIDENCE_SPECIAL,
    }
    token_info: dict[str, Any] = {}
    if protocol in token_row_protocols:
        token_info = ensure_tgvf_protocol_tokens(processor.tokenizer, model, protocol=protocol)
    freeze_qwen_backbone(model)
    token_row_info: dict[str, Any] = {}
    if protocol in token_row_protocols:
        token_row_info, _ = _enable_protocol_c_token_row_training(
            model=model,
            tokenizer=processor.tokenizer,
            protocol=protocol,
            mode=str(training.get("token_row_mode") or "row_only"),
        )
    device = _parameter_audit_device(torch)
    dims = infer_qwen3_stage1_dims(
        model=model,
        processor=processor,
        sample=samples[0],
        device=device,
        max_image_resolution=training.get("max_image_resolution"),
    )
    tgvf = build_tgvf_module(
        variant=str(training.get("variant") or "tgvf_v2_bidirectional"),
        d_lm=dims["d_lm"],
        d_v=dims["d_v"],
        num_foveated_tokens=None,
        spatial_merge_size=dims["spatial_merge_size"],
    ).to(device=device, dtype=next(model.parameters()).dtype)
    return {
        "modules": {"qwen": model, "tgvf": tgvf},
        "processor": processor,
        "tokenizer": processor.tokenizer,
        "loader": {
            "backend": "stage1_qwen3_training_parameter_audit",
            "processor_id": getattr(loaded, "processor_id", model_cfg.get("processor_id")),
            "protocol_token_info": token_info,
            "protocol_token_rows": token_row_info,
            "dims": dims,
        },
    }


def _load_stage2_parameter_audit_modules(bundle: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch
        from peft import LoraConfig, get_peft_model

        from revisit_vlm.qwen3_vl_tgvf import (
            PROTOCOL_C_THINKING_SPECIAL,
            PROTOCOL_C_TOOL_OBSERVATION,
            PROTOCOL_C_TOOL_OBSERVATION_QWEN2_NO_THINK,
            PROTOCOL_E_ACTION_EVIDENCE_SPECIAL,
            ensure_tgvf_protocol_tokens,
            load_qwen3_vl,
        )
        from revisit_vlm.tgvf_training import build_tgvf_module
        from revisit_vlm.tgvf_v3_stage1 import freeze_qwen_backbone, infer_qwen3_stage1_dims
        from revisit_vlm.tgvf_v3_stage2 import TGVFv3Stage2Dataset
        from scripts.train_tgvf_v3_stage2 import _restore_protocol_c_token_rows_from_stage1
    except Exception as exc:
        raise RuntimeError("Stage2 model parameter audit dependencies are unavailable") from exc
    model_cfg = bundle.get("model") or {}
    dataset = bundle.get("dataset") or {}
    training = bundle.get("training") or {}
    lora = bundle.get("lora") or {}
    train_file = ((dataset.get("train_file") or {}).get("path"))
    checkpoint_path = ((dataset.get("stage1_checkpoint") or {}).get("path"))
    if not train_file or not checkpoint_path:
        raise ValueError("Stage2 parameter audit requires train_file and stage1_checkpoint")
    samples = TGVFv3Stage2Dataset(train_file)
    focus_sample = next((sample for sample in samples.samples if sample.need_focus), None)
    if focus_sample is None:
        raise RuntimeError("Stage2 parameter audit needs at least one focus sample")
    loaded = load_qwen3_vl(
        model_cfg.get("model_id"),
        processor_id=model_cfg.get("processor_id"),
        dtype=str(model_cfg.get("dtype") or "bfloat16"),
        device_map=model_cfg.get("device_map") or "auto",
        attn_implementation=model_cfg.get("attn_implementation"),
    )
    model = loaded.model
    processor = loaded.processor
    if getattr(processor.tokenizer, "pad_token", None) is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    protocol = str(bundle.get("protocol"))
    token_row_protocols = {
        PROTOCOL_C_THINKING_SPECIAL,
        PROTOCOL_C_TOOL_OBSERVATION,
        PROTOCOL_C_TOOL_OBSERVATION_QWEN2_NO_THINK,
        PROTOCOL_E_ACTION_EVIDENCE_SPECIAL,
    }
    token_info: dict[str, Any] = {}
    if protocol in token_row_protocols:
        token_info = ensure_tgvf_protocol_tokens(processor.tokenizer, model, protocol=protocol)
    freeze_qwen_backbone(model)
    runtime = ((bundle.get("module_policy") or {}).get("training_runtime") or {})
    if runtime.get("gradient_checkpointing") and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False
    modules_to_save = ["embed_tokens", "lm_head"] if protocol in token_row_protocols else None
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora.get("rank") or 64),
            lora_alpha=int(lora.get("alpha") or 256),
            target_modules=list(lora.get("target_modules") or []),
            lora_dropout=float(lora.get("dropout") or 0.0),
            bias=str(lora.get("bias") or "none"),
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save,
            ensure_weight_tying=False,
        ),
    )
    utility_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    stage1_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if protocol in token_row_protocols:
        _restore_protocol_c_token_rows_from_stage1(
            model=model,
            tokenizer=processor.tokenizer,
            protocol=protocol,
            stage1_checkpoint=stage1_checkpoint,
        )
    device = _parameter_audit_device(torch)
    dims = infer_qwen3_stage1_dims(
        model=utility_model,
        processor=processor,
        sample=focus_sample,  # type: ignore[arg-type]
        device=device,
        max_image_resolution=training.get("max_image_resolution"),
    )
    tgvf_cfg = (stage1_checkpoint.get("config") or {}).get("tgvf") or {}
    tgvf = build_tgvf_module(
        variant=str(tgvf_cfg.get("variant") or training.get("variant") or "tgvf_v2_bidirectional"),
        d_lm=dims["d_lm"],
        d_v=dims["d_v"],
        num_foveated_tokens=tgvf_cfg.get("num_foveated_tokens"),
        spatial_merge_size=int(tgvf_cfg.get("spatial_merge_size") or dims["spatial_merge_size"]),
        attn_dim=tgvf_cfg.get("attn_dim"),
        encoder_adapter_layers=tuple(tgvf_cfg.get("encoder_adapter_layers") or (8, 16, 24)),
        encoder_adapter_gate_init=float(tgvf_cfg.get("encoder_adapter_gate_init", 0.0)),
        encoder_adapter_share_weights=bool(tgvf_cfg.get("encoder_adapter_share_weights", False)),
        encoder_adapter_layer_index_base=int(tgvf_cfg.get("encoder_adapter_layer_index_base", 0)),
        encoder_reencode_deepstack_compatible=bool(
            tgvf_cfg.get("encoder_reencode_deepstack_compatible", False)
        ),
    ).to(device=device, dtype=next(model.parameters()).dtype)
    tgvf.load_state_dict(stage1_checkpoint["tgvf_module"], strict=True)
    return {
        "modules": {"qwen_lora": model, "tgvf": tgvf},
        "processor": processor,
        "tokenizer": processor.tokenizer,
        "loader": {
            "backend": "stage2_qwen3_lora_tgvf_parameter_audit",
            "processor_id": getattr(loaded, "processor_id", model_cfg.get("processor_id")),
            "protocol_token_info": token_info,
            "dims": dims,
            "stage1_global_step": stage1_checkpoint.get("global_step"),
            "modules_to_save": modules_to_save,
        },
    }


def _parameter_audit_device(torch_module: Any) -> Any:
    return torch_module.device("cuda:0" if torch_module.cuda.is_available() else "cpu")


def _runtime_audit_report(
    *,
    bundle_path: Path,
    bundle: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    trainable_parameters: dict[str, Any],
    optimizer_runtime: dict[str, Any] | None,
    checkpoint_runtime: dict[str, Any] | None,
    expected_stage: TrainingStage,
) -> dict[str, Any]:
    artifact_checks = _runtime_artifact_checks(
        bundle,
        artifacts,
        trainable_parameters,
        optimizer_runtime,
        checkpoint_runtime,
    )
    launch_gates = _launch_gate_audit(
        bundle,
        artifacts,
        trainable_parameters,
        optimizer_runtime,
        checkpoint_runtime,
    )
    blocking_items = [
        "native trainer loop has not been ported into revisit_vlm_clean",
        "checkpoint save/load parity must be proven before launch_permitted=true",
    ]
    if not trainable_parameters["payload"].get("actual_model_parameters_loaded"):
        blocking_items.append("actual trainable parameter audit requires loading the model")
    if not (
        optimizer_runtime
        and optimizer_runtime["payload"].get("actual_optimizer_constructed")
        and optimizer_runtime["payload"].get("actual_scheduler_constructed")
    ):
        blocking_items.append("actual optimizer/scheduler construction requires --audit-optimizer")
    if not (
        checkpoint_runtime
        and checkpoint_runtime["payload"].get("actual_checkpoint_saved")
        and checkpoint_runtime["payload"].get("actual_checkpoint_loaded")
    ):
        blocking_items.append("checkpoint save/load parity requires --audit-checkpoint")
    return {
        "schema_version": "clean_training_runtime_audit_v1",
        "stage": str(expected_stage),
        "run_id": bundle.get("run_id"),
        "status": "blocked_before_training_loop",
        "will_launch_training": False,
        "training_runtime_ported": False,
        "bundle_path": str(bundle_path),
        "bundle_identity": file_identity(bundle_path).to_dict(),
        "plan_identity": bundle.get("plan_identity"),
        "artifact_checks": artifact_checks,
        "launch_gates": launch_gates,
        "blocking_items": blocking_items,
    }


def _runtime_artifact_checks(
    bundle: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    trainable_parameters: dict[str, Any],
    optimizer_runtime: dict[str, Any] | None,
    checkpoint_runtime: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    artifact_paths = dict(bundle.get("runtime_artifacts") or {})
    artifact_paths["trainable_parameters"] = trainable_parameters["path"]
    if optimizer_runtime is not None:
        artifact_paths["optimizer_runtime"] = optimizer_runtime["path"]
    if checkpoint_runtime is not None:
        artifact_paths["checkpoint_runtime"] = checkpoint_runtime["path"]
    checks = []
    for name, path_text in sorted(artifact_paths.items()):
        path = Path(str(path_text))
        if name == "trainable_parameters":
            payload = trainable_parameters["payload"]
        elif name == "optimizer_runtime":
            payload = optimizer_runtime["payload"] if optimizer_runtime else {}
        elif name == "checkpoint_runtime":
            payload = checkpoint_runtime["payload"] if checkpoint_runtime else {}
        else:
            payload = artifacts.get(name, {})
        checks.append(
            {
                "name": name,
                "path": str(path),
                "exists": path.exists(),
                "sha256": file_identity(path).sha256 if path.exists() else None,
                "schema_version": payload.get("schema_version"),
                "status": payload.get("status"),
            }
        )
    return checks


def _launch_gate_audit(
    bundle: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    trainable_parameters: dict[str, Any],
    optimizer_runtime: dict[str, Any] | None,
    checkpoint_runtime: dict[str, Any] | None,
) -> dict[str, Any]:
    required = list(
        (bundle.get("trainer_runtime_contract") or {}).get("required_launch_gates") or []
    )
    satisfied = {"build_dataset_loader_from_plan_identity"}
    actual_parameters_loaded = bool(
        trainable_parameters["payload"].get("actual_model_parameters_loaded")
    )
    actual_optimizer_constructed = bool(
        optimizer_runtime
        and optimizer_runtime["payload"].get("actual_optimizer_constructed")
        and optimizer_runtime["payload"].get("actual_scheduler_constructed")
    )
    if actual_optimizer_constructed:
        satisfied.add("construct_optimizer_and_scheduler_from_plan")
    actual_checkpoint_validated = bool(
        checkpoint_runtime
        and checkpoint_runtime["payload"].get("actual_checkpoint_saved")
        and checkpoint_runtime["payload"].get("actual_checkpoint_loaded")
        and not checkpoint_runtime["payload"].get("missing_required_keys")
        and checkpoint_runtime["payload"].get("state_checks_ok")
    )
    if actual_checkpoint_validated:
        satisfied.add("save_checkpoint_with_clean_contract")
    if actual_parameters_loaded:
        satisfied.update(
            {
                "load_model_and_processor",
                "ensure_protocol_token_rows",
                "set_training_use_cache_false",
                "emit_trainable_parameter_audit",
            }
        )
        if bundle.get("stage") == str(TrainingStage.STAGE1):
            satisfied.add("build_tgvf_module_from_stage1_plan")
        if bundle.get("stage") == str(TrainingStage.STAGE2):
            satisfied.update(
                {
                    "load_stage1_checkpoint_tgvf_and_protocol_rows",
                    "attach_lora_modules_from_plan",
                }
            )
    pending_model_load = {
        "load_model_and_processor",
        "ensure_protocol_token_rows",
        "set_training_use_cache_false",
        "construct_optimizer_and_scheduler_from_plan",
        "emit_trainable_parameter_audit",
        "save_checkpoint_with_clean_contract",
    }
    stage_pending = {
        "build_tgvf_module_from_stage1_plan",
        "stage1_readout_context_uses_qwen_v_merge",
        "stage1_position_ids_use_real_qwen3_mrope",
        "stage1_matrix_ce_and_manifold_losses_match_plan",
        "load_stage1_checkpoint_tgvf_and_protocol_rows",
        "attach_lora_modules_from_plan",
        "use_fast_batched_stage2_path",
        "apply_weighted_span_losses_from_plan",
        "apply_original_image_mask_scope_from_plan",
        "apply_deepstack_training_scope_when_enabled",
    }
    gates = []
    for gate in required:
        if gate in satisfied:
            status = "identity_validated"
        elif gate in pending_model_load or gate in stage_pending:
            status = "pending_real_trainer_loop"
        else:
            status = "unknown_gate"
        gates.append({"name": gate, "status": status})
    return {
        "total": len(gates),
        "identity_validated": sum(1 for gate in gates if gate["status"] == "identity_validated"),
        "pending_real_trainer_loop": sum(
            1 for gate in gates if gate["status"] == "pending_real_trainer_loop"
        ),
        "unknown": sum(1 for gate in gates if gate["status"] == "unknown_gate"),
        "dataset_runtime_schema": artifacts["dataset_runtime_identity"].get("schema_version"),
        "optimizer_groups_status": artifacts["optimizer_groups"].get("status"),
        "optimizer_runtime_status": (
            optimizer_runtime["payload"].get("status") if optimizer_runtime else "not_requested"
        ),
        "checkpoint_runtime_status": (
            checkpoint_runtime["payload"].get("status") if checkpoint_runtime else "not_requested"
        ),
        "checkpoint_contract_status": artifacts["checkpoint_contract"].get("status"),
        "trainable_parameters_status": trainable_parameters["payload"].get("status"),
        "gates": gates,
    }


def _runtime_audit_status(audit: dict[str, Any]) -> dict[str, Any]:
    gates = audit.get("launch_gates") or {}
    return {
        "schema_version": "clean_training_runtime_audit_status_v1",
        "stage": audit.get("stage"),
        "run_id": audit.get("run_id"),
        "status": audit.get("status"),
        "will_launch_training": audit.get("will_launch_training"),
        "training_runtime_ported": audit.get("training_runtime_ported"),
        "identity_validated_gates": gates.get("identity_validated"),
        "pending_real_trainer_loop_gates": gates.get("pending_real_trainer_loop"),
        "blocking_items": list(audit.get("blocking_items") or []),
    }


def _runtime_audit_text(audit: dict[str, Any]) -> str:
    gates = audit.get("launch_gates") or {}
    lines = [
        f"schema: {audit.get('schema_version')}",
        f"run_id: {audit.get('run_id')}",
        f"stage: {audit.get('stage')}",
        f"status: {audit.get('status')}",
        f"will_launch_training: {audit.get('will_launch_training')}",
        f"training_runtime_ported: {audit.get('training_runtime_ported')}",
        f"identity_validated_gates: {gates.get('identity_validated')}",
        f"pending_real_trainer_loop_gates: {gates.get('pending_real_trainer_loop')}",
        "blocking_items:",
    ]
    lines.extend(f"- {item}" for item in audit.get("blocking_items") or [])
    return "\n".join(lines) + "\n"


def _write_dataset_runtime_artifacts(
    *,
    execution_dir: Path,
    plan: dict[str, Any],
    expected_stage: TrainingStage,
) -> dict[str, Any]:
    dataset = plan.get("dataset") or {}
    batch = plan.get("batch") or {}
    requested_batch = int(batch.get("global_batch_size") or 1)
    train_identity = dataset.get("train_file") or {}
    train_summary, first_batch = _scan_training_jsonl(
        train_identity,
        stage=expected_stage,
        label="train_file",
        first_batch_size=requested_batch,
    )
    val_summary = None
    if expected_stage == TrainingStage.STAGE2 and dataset.get("val_file") is not None:
        val_summary, _ = _scan_training_jsonl(
            dataset.get("val_file") or {},
            stage=expected_stage,
            label="val_file",
            first_batch_size=0,
        )
    dataset_runtime = {
        "schema_version": "clean_training_dataset_runtime_identity_v1",
        "stage": str(expected_stage),
        "train_file": train_summary,
        "val_file": val_summary,
        "global_batch_size": requested_batch,
        "plan_dataset_identity": dataset,
    }
    first_batch_identity = {
        "schema_version": "clean_training_first_batch_identity_v1",
        "stage": str(expected_stage),
        "requested_global_batch_size": requested_batch,
        "materialized_batch_size": len(first_batch),
        "row_digests": [row["row_sha256"] for row in first_batch],
        "batch_sha256": _payload_sha256(first_batch),
        "rows": first_batch,
    }
    dataset_path = execution_dir / "dataset_runtime_identity.json"
    first_batch_path = execution_dir / "first_batch_identity.json"
    _write_json(dataset_path, dataset_runtime)
    _write_json(first_batch_path, first_batch_identity)
    return {
        "paths": {
            "dataset_runtime_identity": str(dataset_path),
            "first_batch_identity": str(first_batch_path),
        },
        "dataset_runtime": dataset_runtime,
        "first_batch_identity": first_batch_identity,
    }


def _write_checkpoint_contract_artifact(
    *,
    execution_dir: Path,
    plan: dict[str, Any],
    expected_stage: TrainingStage,
) -> dict[str, Any]:
    contract = _checkpoint_contract(plan=plan, expected_stage=expected_stage)
    path = execution_dir / "checkpoint_contract.json"
    _write_json(path, contract)
    return {"path": str(path), "contract": contract}


def _write_optimizer_groups_artifact(
    *,
    execution_dir: Path,
    plan: dict[str, Any],
    expected_stage: TrainingStage,
) -> dict[str, Any]:
    contract = _optimizer_groups_contract(plan=plan, expected_stage=expected_stage)
    path = execution_dir / "optimizer_groups.json"
    _write_json(path, contract)
    return {"path": str(path), "contract": contract}


def _optimizer_groups_contract(
    *,
    plan: dict[str, Any],
    expected_stage: TrainingStage,
) -> dict[str, Any]:
    optimizer = plan.get("optimizer") or {}
    if str(optimizer.get("name") or "").lower() != "adamw":
        raise ValueError("clean training optimizer must be AdamW")
    groups = (
        _stage1_optimizer_groups(plan)
        if expected_stage == TrainingStage.STAGE1
        else _stage2_optimizer_groups(plan)
    )
    if not groups:
        raise ValueError("optimizer group contract must contain at least one group")
    return {
        "schema_version": "clean_training_optimizer_groups_v1",
        "stage": str(expected_stage),
        "status": "validated",
        "optimizer": {
            "name": "adamw",
            "betas": optimizer.get("betas"),
            "eps": optimizer.get("eps"),
            "weight_decay": optimizer.get("weight_decay"),
            "max_grad_norm": optimizer.get("max_grad_norm"),
        },
        "scheduler": {
            "name": optimizer.get("lr_scheduler"),
            "warmup_steps": optimizer.get("warmup_steps"),
            "warmup_ratio": optimizer.get("warmup_ratio"),
            "min_lr_ratio": optimizer.get("min_lr_ratio"),
        },
        "groups": groups,
        "group_names": [str(group["name"]) for group in groups],
    }


def _stage1_optimizer_groups(plan: dict[str, Any]) -> list[dict[str, Any]]:
    optimizer = plan.get("optimizer") or {}
    module_policy = plan.get("module_policy") or {}
    trainable = set(module_policy.get("trainable") or [])
    learning_rate = optimizer.get("learning_rate")
    groups = [
        {
            "name": "tgvf_module",
            "lr": learning_rate,
            "module_policy_entry": "tgvf_module",
            "weight_decay": 0.01,
            "weight_decay_source": "torch.optim.AdamW_default_in_historical_stage1",
            "expected_when_trainable": "tgvf_module" in trainable,
        }
    ]
    if "protocol_c_token_rows_row_only" in trainable:
        groups.append(
            {
                "name": "protocol_c_token_rows",
                "lr": learning_rate,
                "module_policy_entry": "protocol_c_token_rows_row_only",
                "weight_decay": 0.01,
                "weight_decay_source": "torch.optim.AdamW_default_in_historical_stage1",
                "expected_when_trainable": True,
            }
        )
    return groups


def _stage2_optimizer_groups(plan: dict[str, Any]) -> list[dict[str, Any]]:
    optimizer = plan.get("optimizer") or {}
    module_policy = plan.get("module_policy") or {}
    trainable = set(module_policy.get("trainable") or [])
    weight_decay = optimizer.get("weight_decay")
    return [
        {
            "name": "llm_lora",
            "lr": optimizer.get("lr_lora"),
            "module_policy_entry": "qwen_lora_adapters",
            "weight_decay": weight_decay,
            "expected_when_trainable": "qwen_lora_adapters" in trainable,
        },
        {
            "name": "tgvf_refiner",
            "lr": optimizer.get("lr_tgvf"),
            "module_policy_entry": "tgvf_module_continued_from_stage1",
            "weight_decay": weight_decay,
            "expected_when_trainable": "tgvf_module_continued_from_stage1" in trainable,
        },
        {
            "name": "fvt_calibration",
            "lr": optimizer.get("lr_calibration"),
            "module_policy_entry": "tgvf_module_continued_from_stage1",
            "weight_decay": weight_decay,
            "expected_when_trainable": "tgvf_module_continued_from_stage1" in trainable,
        },
    ]


def _checkpoint_contract(*, plan: dict[str, Any], expected_stage: TrainingStage) -> dict[str, Any]:
    if expected_stage == TrainingStage.STAGE1:
        return {
            "schema_version": "clean_training_checkpoint_contract_v1",
            "stage": str(expected_stage),
            "status": "no_input_checkpoint_required",
            "input_checkpoint_required": False,
            "output_checkpoint_required_keys": [
                "tgvf_module",
                "config",
                "global_step",
            ],
            "protocol_token_rows_required_for_protocol_c": True,
        }
    dataset = plan.get("dataset") or {}
    identity = dataset.get("stage1_checkpoint") or {}
    _validate_file_identity(identity, label="stage1_checkpoint")
    checkpoint = _load_torch_checkpoint(str(identity["path"]))
    if not isinstance(checkpoint, Mapping):
        raise ValueError("stage1_checkpoint must load to a mapping")
    keys = sorted(str(key) for key in checkpoint.keys())
    config = checkpoint.get("config") or {}
    if not isinstance(config, Mapping):
        raise ValueError("stage1_checkpoint config must be a mapping")
    protocol = config.get("tgvf_protocol")
    requested_protocol = plan.get("protocol")
    if protocol and requested_protocol and protocol != requested_protocol:
        raise ValueError(
            "stage1_checkpoint protocol mismatch: "
            f"{protocol!r} != requested {requested_protocol!r}"
        )
    tgvf_module = checkpoint.get("tgvf_module")
    if not isinstance(tgvf_module, Mapping) or not tgvf_module:
        raise ValueError("stage1_checkpoint missing non-empty tgvf_module state dict")
    token_rows = checkpoint.get("protocol_c_token_rows")
    protocol_c_required = str(requested_protocol or "").startswith("protocol_c_")
    if protocol_c_required:
        if not isinstance(token_rows, Mapping):
            raise ValueError("stage1_checkpoint missing protocol_c_token_rows")
        row_protocol = token_rows.get("protocol")
        if row_protocol and requested_protocol and row_protocol != requested_protocol:
            raise ValueError(
                "stage1 checkpoint protocol token rows mismatch: "
                f"{row_protocol!r} != requested {requested_protocol!r}"
            )
    return {
        "schema_version": "clean_training_checkpoint_contract_v1",
        "stage": str(expected_stage),
        "status": "validated",
        "input_checkpoint_required": True,
        "checkpoint_identity": identity,
        "checkpoint_keys": keys,
        "global_step": checkpoint.get("global_step"),
        "config": {
            "stage": config.get("stage"),
            "tgvf_protocol": protocol,
            "model_id": config.get("model_id"),
            "processor_id": config.get("processor_id"),
            "tgvf": config.get("tgvf"),
        },
        "tgvf_module": _state_dict_summary(tgvf_module),
        "protocol_c_token_rows": _protocol_token_rows_summary(token_rows),
        "protocol_c_token_rows_required": protocol_c_required,
    }


def _load_torch_checkpoint(path: str) -> Any:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch is required to validate Stage2 checkpoint contracts") from exc
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        try:
            return torch.load(path, map_location="cpu")
        except Exception as exc:
            raise ValueError(f"stage1_checkpoint is not loadable by torch: {path}") from exc
    except Exception as exc:
        raise ValueError(f"stage1_checkpoint is not loadable by torch: {path}") from exc


def _state_dict_summary(state: Mapping[Any, Any]) -> dict[str, Any]:
    items = list(state.items())
    return {
        "num_tensors": len(items),
        "total_numel": sum(_numel(value) for _, value in items),
        "sample_keys": [str(key) for key, _ in items[:10]],
        "sample_shapes": {
            str(key): _shape_list(value)
            for key, value in items[:10]
            if _shape_list(value) is not None
        },
    }


def _protocol_token_rows_summary(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    token_ids = payload.get("token_ids") or {}
    tokens = payload.get("tokens") or []
    return {
        "protocol": payload.get("protocol"),
        "num_tokens": len(tokens) if isinstance(tokens, list) else None,
        "token_ids": dict(token_ids) if isinstance(token_ids, Mapping) else None,
        "input_embeddings_shape": _shape_list(payload.get("input_embeddings")),
        "output_embeddings_shape": _shape_list(payload.get("output_embeddings")),
    }


def _shape_list(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(item) for item in shape]


def _numel(value: Any) -> int:
    try:
        return int(value.numel())
    except Exception:
        return 0


def _scan_training_jsonl(
    identity: dict[str, Any],
    *,
    stage: TrainingStage,
    label: str,
    first_batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _validate_file_identity(identity, label=label)
    path = Path(str(identity["path"]))
    required_fields = _required_dataset_fields(stage)
    line_count = 0
    missing_counts = {field: 0 for field in required_fields}
    need_focus = 0
    no_focus = 0
    missing_need_focus = 0
    answer_formats: dict[str, int] = {}
    source_datasets: dict[str, int] = {}
    malformed: list[dict[str, Any]] = []
    first_rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            line_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                malformed.append({"line": line_no, "error": str(exc)})
                continue
            for field_name in required_fields:
                if record.get(field_name) in (None, ""):
                    missing_counts[field_name] += 1
            if "need_focus" in record:
                if bool(record.get("need_focus")):
                    need_focus += 1
                else:
                    no_focus += 1
            else:
                missing_need_focus += 1
            answer_format = str(record.get("answer_format") or "unknown")
            answer_formats[answer_format] = answer_formats.get(answer_format, 0) + 1
            source_dataset = str(record.get("source_dataset") or "unknown")
            source_datasets[source_dataset] = source_datasets.get(source_dataset, 0) + 1
            if len(first_rows) < first_batch_size:
                first_rows.append(_row_runtime_identity(record, line_no=line_no))
    if malformed:
        raise ValueError(f"malformed JSONL rows in {path}: {malformed[:3]}")
    if line_count < 1:
        raise ValueError(f"training JSONL is empty: {path}")
    missing_required = {key: value for key, value in missing_counts.items() if value}
    if missing_required:
        raise ValueError(f"training JSONL missing required fields in {path}: {missing_required}")
    summary = {
        "path": str(path),
        "identity": identity,
        "line_count": line_count,
        "required_fields": required_fields,
        "missing_required_counts": missing_counts,
        "need_focus": need_focus,
        "no_focus": no_focus,
        "missing_need_focus": missing_need_focus,
        "answer_formats": answer_formats,
        "source_datasets": source_datasets,
    }
    return summary, first_rows


def _required_dataset_fields(stage: TrainingStage) -> list[str]:
    if stage == TrainingStage.STAGE1:
        return ["image", "question", "target"]
    return ["image", "question", "answer", "need_focus", "evidence_state"]


def _row_runtime_identity(record: dict[str, Any], *, line_no: int) -> dict[str, Any]:
    row_sha = _payload_sha256(record)
    return {
        "line_no": line_no,
        "row_key": str(
            record.get("uid")
            or record.get("v4_uid")
            or record.get("source_uid")
            or f"line-{line_no}-{row_sha[:12]}"
        ),
        "row_sha256": row_sha,
        "image": record.get("image"),
        "image_id": record.get("image_id") or record.get("stable_image_uid"),
        "source_dataset": record.get("source_dataset"),
        "question_sha256": _text_sha256(record.get("question")),
        "target_sha256": _text_sha256(record.get("target")),
        "answer_sha256": _text_sha256(record.get("answer")),
        "need_focus": record.get("need_focus"),
        "answer_format": record.get("answer_format"),
        "trajectory_type": record.get("trajectory_type"),
    }


def _payload_sha256(payload: Any) -> str:
    blob = json.dumps(_to_jsonable(payload), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return sha256(blob).hexdigest()


def _text_sha256(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return sha256(str(value).encode("utf-8")).hexdigest()


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
