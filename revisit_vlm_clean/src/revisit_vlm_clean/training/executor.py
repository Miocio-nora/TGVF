"""Fail-safe clean training executor preflight.

The real Stage1/Stage2 training loops are not ported yet. This module makes the
planned clean entrypoints importable and validates training-plan identity before
any future executor is allowed to launch.
"""

from __future__ import annotations

import argparse
import json
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
            )
            prepared["runtime_audit"] = audit["runtime_audit"]
            prepared["trainable_parameters"] = audit["trainable_parameters"]
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
) -> dict[str, Any]:
    bundle_file = Path(bundle_path)
    if not bundle_file.exists():
        raise FileNotFoundError(f"training execution bundle does not exist: {bundle_file}")
    bundle = json.loads(bundle_file.read_text(encoding="utf-8"))
    _validate_execution_bundle(bundle, expected_stage=expected_stage)
    execution_dir = Path(str(bundle.get("execution_dir") or bundle_file.parent))
    artifacts = _load_runtime_artifacts(bundle, expected_stage=expected_stage)
    trainable_parameters = _write_trainable_parameters_placeholder(
        execution_dir=execution_dir,
        bundle=bundle,
    )
    audit = _runtime_audit_report(
        bundle_path=bundle_file,
        bundle=bundle,
        artifacts=artifacts,
        trainable_parameters=trainable_parameters,
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
    return {
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


def _runtime_audit_report(
    *,
    bundle_path: Path,
    bundle: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    trainable_parameters: dict[str, Any],
    expected_stage: TrainingStage,
) -> dict[str, Any]:
    artifact_checks = _runtime_artifact_checks(bundle, artifacts, trainable_parameters)
    launch_gates = _launch_gate_audit(bundle, artifacts, trainable_parameters)
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
        "blocking_items": [
            "native trainer loop has not been ported into revisit_vlm_clean",
            "actual trainable parameter audit requires loading the model",
            "checkpoint save/load parity must be proven before launch_permitted=true",
        ],
    }


def _runtime_artifact_checks(
    bundle: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    trainable_parameters: dict[str, Any],
) -> list[dict[str, Any]]:
    artifact_paths = dict(bundle.get("runtime_artifacts") or {})
    artifact_paths["trainable_parameters"] = trainable_parameters["path"]
    checks = []
    for name, path_text in sorted(artifact_paths.items()):
        path = Path(str(path_text))
        payload = (
            trainable_parameters["payload"]
            if name == "trainable_parameters"
            else artifacts.get(name, {})
        )
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
) -> dict[str, Any]:
    required = list(
        (bundle.get("trainer_runtime_contract") or {}).get("required_launch_gates") or []
    )
    satisfied = {
        "build_dataset_loader_from_plan_identity",
        "construct_optimizer_and_scheduler_from_plan",
    }
    pending_model_load = {
        "load_model_and_processor",
        "ensure_protocol_token_rows",
        "set_training_use_cache_false",
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
