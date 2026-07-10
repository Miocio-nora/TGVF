"""Fail-safe clean training executor.

This module owns clean Stage1/Stage2 training preflight, execution-bundle
handoff, runtime audits, and explicit clean trainer launches.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from collections.abc import Mapping
from hashlib import sha1, sha256
from pathlib import Path
from typing import Any

from revisit_vlm_clean.cli.common import exit_not_implemented, print_json
from revisit_vlm_clean.data_generation import file_identity
from revisit_vlm_clean.deepstack import (
    DEEPSTACK_RUNTIME_HOOKS_SCHEMA_VERSION,
    DEEPSTACK_SCOPE_CONTRACT_SCHEMA_VERSION,
)
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
    parser.add_argument(
        "--audit-training-step",
        action="store_true",
        help=(
            "During --audit-runtime, run a no-backward clean training-step probe "
            "and write training_step_runtime.json."
        ),
    )
    parser.add_argument(
        "--audit-optimizer-step",
        action="store_true",
        help=(
            "During --audit-runtime, run one bounded backward/gradient-clip/"
            "optimizer.step/scheduler.step probe from the clean training-step loss "
            "and write optimizer_step_runtime.json. This still does not launch a "
            "training loop or publish checkpoints."
        ),
    )
    parser.add_argument(
        "--audit-trainer-loop",
        action="store_true",
        help=(
            "During --audit-runtime, run one bounded gradient-accumulation "
            "trainer-loop probe from the clean training-step loss and write "
            "trainer_loop_runtime.json. This still does not launch a full "
            "training run or publish checkpoints."
        ),
    )
    parser.add_argument(
        "--audit-checkpoint-publish",
        action="store_true",
        help=(
            "During --audit-runtime, run the bounded trainer-loop probe, save and "
            "reload a post-step clean checkpoint publish probe, and write "
            "training_checkpoint_publish_runtime.json. This still does not launch "
            "a full training run."
        ),
    )
    parser.add_argument(
        "--audit-checkpoint-resume",
        action="store_true",
        help=(
            "During --audit-runtime, run the checkpoint publish probe, reload fresh "
            "model/optimizer/scheduler objects from that checkpoint, and write "
            "training_checkpoint_resume_runtime.json. This still does not launch "
            "a full training run."
        ),
    )
    parser.add_argument(
        "--audit-cadence",
        action="store_true",
        help=(
            "During --audit-runtime, validate max-step, checkpoint-save, and "
            "evaluation cadence from the clean plan and write "
            "training_cadence_runtime.json. This does not load the model or launch "
            "training."
        ),
    )
    parser.add_argument(
        "--audit-launch-readiness",
        action="store_true",
        help=(
            "During --audit-runtime, run all prerequisite non-launch runtime "
            "audits needed for a clean launch-readiness summary and write "
            "training_launch_readiness.json. This still does not launch training."
        ),
    )
    parser.add_argument(
        "--launch-training",
        action="store_true",
        help=(
            "Explicitly launch the clean trainer loop from the validated plan. "
            "Single-process and torchrun-distributed clean launch paths are "
            "selected from the plan batch identity."
        ),
    )
    parser.add_argument(
        "--launch-report",
        default=None,
        help=(
            "Path for the clean training launch result JSON. Defaults to "
            "<execution-dir>/clean_training_launch_result.json."
        ),
    )
    return parser


def main_for_stage(stage: TrainingStage, argv: list[str] | None = None) -> int:
    args = build_parser(stage).parse_args(argv)
    plan_path, plan = _load_training_plan(args.plan)
    _validate_training_plan(plan, expected_stage=stage)
    launch_runtime_context = (
        _setup_launch_runtime_context({"batch": plan.get("batch") or {}})
        if args.launch_training and _torchrun_env_matches_plan(plan)
        else None
    )
    report = _preflight_report(plan_path, plan, expected_stage=stage)
    report_path = _preflight_report_path(
        plan_path=plan_path,
        stage=stage,
        requested=args.preflight_report,
    )
    report["preflight_report"] = str(report_path)
    _write_json(report_path, report)
    if args.prepare_execution:
        prepared = _prepare_training_execution_for_launch(
            plan_path=plan_path,
            plan=plan,
            expected_stage=stage,
            execution_dir=args.execution_dir,
            runtime_context=launch_runtime_context,
        ) if args.launch_training else prepare_training_execution(
            plan_path,
            plan,
            expected_stage=stage,
            execution_dir=args.execution_dir,
        )
        prepared["preflight_report"] = str(report_path)
        if args.launch_training:
            launched = launch_training(
                bundle_path=prepared["execution_bundle"],
                expected_stage=stage,
                report_path=args.launch_report,
                runtime_context=launch_runtime_context,
            )
            if launched.get("training_launch_result"):
                prepared["training_launch_result"] = launched["training_launch_result"]
            if launched.get("training_launch_status"):
                prepared["training_launch_status"] = launched["training_launch_status"]
            if launched.get("training_runtime"):
                prepared["training_runtime"] = launched["training_runtime"]
            if launched.get("single_process_training_runtime"):
                prepared["single_process_training_runtime"] = launched[
                    "single_process_training_runtime"
                ]
            if launched.get("distributed_rank_training_runtime"):
                prepared["distributed_rank_training_runtime"] = launched[
                    "distributed_rank_training_runtime"
                ]
            prepared["final_checkpoint"] = launched.get("final_checkpoint")
            print_json(prepared)
            return 0
        if args.audit_runtime:
            audit = audit_training_runtime(
                bundle_path=prepared["execution_bundle"],
                expected_stage=stage,
                report_path=args.runtime_audit_report,
                audit_model_parameters=args.audit_model_parameters,
                audit_optimizer=args.audit_optimizer,
                audit_checkpoint=args.audit_checkpoint,
                audit_training_step=args.audit_training_step,
                audit_optimizer_step=args.audit_optimizer_step,
                audit_trainer_loop=args.audit_trainer_loop,
                audit_checkpoint_publish=args.audit_checkpoint_publish,
                audit_checkpoint_resume=args.audit_checkpoint_resume,
                audit_cadence=args.audit_cadence,
                audit_launch_readiness=args.audit_launch_readiness,
            )
            prepared["runtime_audit"] = audit["runtime_audit"]
            prepared["trainable_parameters"] = audit["trainable_parameters"]
            if audit.get("optimizer_runtime"):
                prepared["optimizer_runtime"] = audit["optimizer_runtime"]
            if audit.get("checkpoint_runtime"):
                prepared["checkpoint_runtime"] = audit["checkpoint_runtime"]
            if audit.get("training_step_runtime"):
                prepared["training_step_runtime"] = audit["training_step_runtime"]
            if audit.get("optimizer_step_runtime"):
                prepared["optimizer_step_runtime"] = audit["optimizer_step_runtime"]
            if audit.get("trainer_loop_runtime"):
                prepared["trainer_loop_runtime"] = audit["trainer_loop_runtime"]
            if audit.get("training_checkpoint_publish_runtime"):
                prepared["training_checkpoint_publish_runtime"] = audit[
                    "training_checkpoint_publish_runtime"
                ]
            if audit.get("training_checkpoint_resume_runtime"):
                prepared["training_checkpoint_resume_runtime"] = audit[
                    "training_checkpoint_resume_runtime"
                ]
            if audit.get("training_cadence_runtime"):
                prepared["training_cadence_runtime"] = audit["training_cadence_runtime"]
            if audit.get("training_launch_readiness"):
                prepared["training_launch_readiness"] = audit[
                    "training_launch_readiness"
                ]
        print_json(prepared)
        return 0
    if args.launch_training:
        prepared = _prepare_training_execution_for_launch(
            plan_path=plan_path,
            plan=plan,
            expected_stage=stage,
            execution_dir=args.execution_dir,
            runtime_context=launch_runtime_context,
        )
        launched = launch_training(
            bundle_path=prepared["execution_bundle"],
            expected_stage=stage,
            report_path=args.launch_report,
            runtime_context=launch_runtime_context,
        )
        launched["preflight_report"] = str(report_path)
        print_json(launched)
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
            audit_training_step=args.audit_training_step,
            audit_optimizer_step=args.audit_optimizer_step,
            audit_trainer_loop=args.audit_trainer_loop,
            audit_checkpoint_publish=args.audit_checkpoint_publish,
            audit_checkpoint_resume=args.audit_checkpoint_resume,
            audit_cadence=args.audit_cadence,
            audit_launch_readiness=args.audit_launch_readiness,
        )
        audit["preflight_report"] = str(report_path)
        print_json(audit)
        return 0
    print_json(report)
    if args.preflight_only:
        return 0
    return exit_not_implemented(
        "explicit execution mode is required; rerun with --preflight-only, "
        "--prepare-execution, --audit-runtime, or --launch-training"
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
    deepstack_training = (
        _write_deepstack_training_plan_artifact(
            execution_dir=out,
            plan=plan,
            expected_stage=expected_stage,
        )
        if expected_stage == TrainingStage.STAGE2
        else None
    )
    bundle["runtime_artifacts"] = {
        **dataset_runtime["paths"],
        "checkpoint_contract": checkpoint_contract["path"],
        "optimizer_groups": optimizer_groups["path"],
    }
    if deepstack_training is not None:
        bundle["runtime_artifacts"]["deepstack_training_plan"] = deepstack_training["path"]
    bundle["dataset_runtime"] = dataset_runtime["dataset_runtime"]
    bundle["first_batch_identity"] = dataset_runtime["first_batch_identity"]
    bundle["checkpoint_contract"] = checkpoint_contract["contract"]
    bundle["optimizer_groups"] = optimizer_groups["contract"]
    if deepstack_training is not None:
        bundle["deepstack_training_plan"] = deepstack_training["plan"]
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
        "training_runtime_ported": status["training_runtime_ported"],
        "dataset_runtime_identity": dataset_runtime["paths"]["dataset_runtime_identity"],
        "first_batch_identity": dataset_runtime["paths"]["first_batch_identity"],
        "checkpoint_contract": checkpoint_contract["path"],
        "optimizer_groups": optimizer_groups["path"],
        "runner_status": status["runner_status"],
        **(
            {"deepstack_training_plan": deepstack_training["path"]}
            if deepstack_training is not None
            else {}
        ),
    }


def audit_training_runtime(
    *,
    bundle_path: str | Path,
    expected_stage: TrainingStage,
    report_path: str | Path | None = None,
    audit_model_parameters: bool = False,
    audit_optimizer: bool = False,
    audit_checkpoint: bool = False,
    audit_training_step: bool = False,
    audit_optimizer_step: bool = False,
    audit_trainer_loop: bool = False,
    audit_checkpoint_publish: bool = False,
    audit_checkpoint_resume: bool = False,
    audit_cadence: bool = False,
    audit_launch_readiness: bool = False,
) -> dict[str, Any]:
    if audit_launch_readiness:
        audit_checkpoint_resume = True
        audit_cadence = True
    bundle_file = Path(bundle_path)
    if not bundle_file.exists():
        raise FileNotFoundError(f"training execution bundle does not exist: {bundle_file}")
    bundle = json.loads(bundle_file.read_text(encoding="utf-8"))
    _validate_execution_bundle(bundle, expected_stage=expected_stage)
    execution_dir = Path(str(bundle.get("execution_dir") or bundle_file.parent))
    artifacts = _load_runtime_artifacts(bundle, expected_stage=expected_stage)
    loaded_modules = (
        _load_training_parameter_audit_modules(bundle, expected_stage=expected_stage)
        if (
            audit_model_parameters
            or audit_optimizer
            or audit_checkpoint
            or audit_training_step
            or audit_optimizer_step
            or audit_trainer_loop
            or audit_checkpoint_publish
            or audit_checkpoint_resume
        )
        else None
    )
    trainable_parameters = (
        _write_actual_trainable_parameters_audit(
            execution_dir=execution_dir,
            bundle=bundle,
            expected_stage=expected_stage,
            loaded_modules=loaded_modules,
        )
        if (
            audit_model_parameters
            or audit_optimizer
            or audit_checkpoint
            or audit_training_step
            or audit_optimizer_step
            or audit_trainer_loop
            or audit_checkpoint_publish
            or audit_checkpoint_resume
        )
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
        if (
            audit_optimizer
            or audit_checkpoint
            or audit_optimizer_step
            or audit_trainer_loop
            or audit_checkpoint_publish
            or audit_checkpoint_resume
        )
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
    training_step_runtime = (
        _write_actual_training_step_runtime_audit(
            execution_dir=execution_dir,
            bundle=bundle,
            artifacts=artifacts,
            loaded_modules=loaded_modules,
            expected_stage=expected_stage,
        )
        if (
            audit_training_step
            or audit_optimizer_step
            or audit_trainer_loop
            or audit_checkpoint_publish
            or audit_checkpoint_resume
        )
        else None
    )
    optimizer_step_runtime = (
        _write_actual_optimizer_step_runtime_audit(
            execution_dir=execution_dir,
            bundle=bundle,
            optimizer_runtime=optimizer_runtime,
            training_step_runtime=training_step_runtime,
        )
        if audit_optimizer_step
        else None
    )
    trainer_loop_runtime = (
        _write_actual_trainer_loop_runtime_audit(
            execution_dir=execution_dir,
            bundle=bundle,
            artifacts=artifacts,
            loaded_modules=loaded_modules,
            optimizer_runtime=optimizer_runtime,
            expected_stage=expected_stage,
        )
        if audit_trainer_loop or audit_checkpoint_publish or audit_checkpoint_resume
        else None
    )
    training_checkpoint_publish_runtime = (
        _write_training_checkpoint_publish_runtime_audit(
            execution_dir=execution_dir,
            bundle=bundle,
            loaded_modules=loaded_modules,
            optimizer_runtime=optimizer_runtime,
            trainer_loop_runtime=trainer_loop_runtime,
            expected_stage=expected_stage,
        )
        if audit_checkpoint_publish or audit_checkpoint_resume
        else None
    )
    training_checkpoint_resume_runtime = (
        _write_training_checkpoint_resume_runtime_audit(
            execution_dir=execution_dir,
            bundle=bundle,
            artifacts=artifacts,
            checkpoint_publish_runtime=training_checkpoint_publish_runtime,
            expected_stage=expected_stage,
        )
        if audit_checkpoint_resume
        else None
    )
    training_cadence_runtime = (
        _write_training_cadence_runtime_audit(
            execution_dir=execution_dir,
            bundle=bundle,
            expected_stage=expected_stage,
        )
        if audit_cadence
        else None
    )
    audit = _runtime_audit_report(
        bundle_path=bundle_file,
        bundle=bundle,
        artifacts=artifacts,
        trainable_parameters=trainable_parameters,
        optimizer_runtime=optimizer_runtime,
        checkpoint_runtime=checkpoint_runtime,
        training_step_runtime=training_step_runtime,
        optimizer_step_runtime=optimizer_step_runtime,
        trainer_loop_runtime=trainer_loop_runtime,
        training_checkpoint_publish_runtime=training_checkpoint_publish_runtime,
        training_checkpoint_resume_runtime=training_checkpoint_resume_runtime,
        training_cadence_runtime=training_cadence_runtime,
        expected_stage=expected_stage,
    )
    training_launch_readiness = (
        _write_training_launch_readiness_audit(
            execution_dir=execution_dir,
            bundle=bundle,
            audit=audit,
            expected_stage=expected_stage,
        )
        if audit_launch_readiness
        else None
    )
    if training_launch_readiness is not None:
        audit["launch_readiness"] = training_launch_readiness["payload"]
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
    if training_step_runtime is not None:
        result["training_step_runtime"] = training_step_runtime["path"]
    if optimizer_step_runtime is not None:
        result["optimizer_step_runtime"] = optimizer_step_runtime["path"]
    if trainer_loop_runtime is not None:
        result["trainer_loop_runtime"] = trainer_loop_runtime["path"]
    if training_checkpoint_publish_runtime is not None:
        result["training_checkpoint_publish_runtime"] = training_checkpoint_publish_runtime[
            "path"
        ]
    if training_checkpoint_resume_runtime is not None:
        result["training_checkpoint_resume_runtime"] = training_checkpoint_resume_runtime["path"]
    if training_cadence_runtime is not None:
        result["training_cadence_runtime"] = training_cadence_runtime["path"]
    if training_launch_readiness is not None:
        result["training_launch_readiness"] = training_launch_readiness["path"]
    return result


def launch_training(
    *,
    bundle_path: str | Path,
    expected_stage: TrainingStage,
    report_path: str | Path | None = None,
    runtime_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bundle_file = Path(bundle_path)
    if not bundle_file.exists():
        raise FileNotFoundError(f"training execution bundle does not exist: {bundle_file}")
    bundle = json.loads(bundle_file.read_text(encoding="utf-8"))
    _validate_execution_bundle(bundle, expected_stage=expected_stage)
    execution_dir = Path(str(bundle.get("execution_dir") or bundle_file.parent))
    artifacts = _load_runtime_artifacts(bundle, expected_stage=expected_stage)
    runtime_context = runtime_context or _setup_launch_runtime_context(bundle)
    bundle = _bundle_with_runtime_context(bundle, runtime_context)
    _validate_training_launch_contract(
        bundle=bundle,
        artifacts=artifacts,
        expected_stage=expected_stage,
        runtime_context=runtime_context,
    )
    try:
        rank_execution_dir = (
            execution_dir
            if runtime_context["is_main"]
            else execution_dir / f"rank_{runtime_context['rank']}"
        )
        rank_execution_dir.mkdir(parents=True, exist_ok=True)
        loaded_modules = _load_training_parameter_audit_modules(
            bundle,
            expected_stage=expected_stage,
        )
        _apply_legacy_distributed_training_semantics(
            loaded_modules=loaded_modules,
            expected_stage=expected_stage,
            runtime_context=runtime_context,
        )
        trainable_parameters = _write_actual_trainable_parameters_audit(
            execution_dir=rank_execution_dir,
            bundle=bundle,
            expected_stage=expected_stage,
            loaded_modules=loaded_modules,
        )
        optimizer_runtime = _write_actual_optimizer_scheduler_audit(
            execution_dir=rank_execution_dir,
            bundle=bundle,
            artifacts=artifacts,
            loaded_modules=loaded_modules,
            expected_stage=expected_stage,
        )
        cadence_runtime = _write_training_cadence_runtime_audit(
            execution_dir=rank_execution_dir,
            bundle=bundle,
            expected_stage=expected_stage,
        )
        training_runtime = _write_single_process_training_runtime(
            execution_dir=rank_execution_dir,
            bundle=bundle,
            artifacts=artifacts,
            loaded_modules=loaded_modules,
            optimizer_runtime=optimizer_runtime,
            cadence_runtime=cadence_runtime,
            expected_stage=expected_stage,
            runtime_context=runtime_context,
        )
        _distributed_barrier(runtime_context)
        if not runtime_context["is_main"]:
            return {
                "training_launch_result": None,
                "training_launch_status": None,
                "training_runtime": training_runtime["path"],
                "distributed_rank_training_runtime": training_runtime["path"],
                "rank": runtime_context["rank"],
                "local_rank": runtime_context["local_rank"],
                "world_size": runtime_context["world_size"],
                "stage": str(expected_stage),
                "run_id": bundle.get("run_id"),
                "will_launch_training": True,
                "training_runtime_ported": True,
                "runner_status": training_runtime["payload"].get("status"),
                "final_checkpoint": None,
            }
        launch_result = _clean_training_launch_result(
            bundle_path=bundle_file,
            bundle=bundle,
            artifacts=artifacts,
            trainable_parameters=trainable_parameters,
            optimizer_runtime=optimizer_runtime,
            cadence_runtime=cadence_runtime,
            training_runtime=training_runtime,
            expected_stage=expected_stage,
        )
        resolved_report_path = (
            Path(report_path)
            if report_path is not None
            else execution_dir / "clean_training_launch_result.json"
        )
        _write_json(resolved_report_path, launch_result)
        status_path = execution_dir / "clean_training_launch_status.json"
        _write_json(status_path, _clean_training_launch_status(launch_result))
        final_checkpoint = (
            (training_runtime["payload"].get("checkpoint_records") or [{}])[-1].get("path")
            if training_runtime["payload"].get("checkpoint_records")
            else None
        )
        return {
            "training_launch_result": str(resolved_report_path),
            "training_launch_status": str(status_path),
            "training_runtime": training_runtime["path"],
            "single_process_training_runtime": (
                training_runtime["path"] if not runtime_context["distributed"] else None
            ),
            "distributed_rank_training_runtime": (
                training_runtime["path"] if runtime_context["distributed"] else None
            ),
            "trainable_parameters": trainable_parameters["path"],
            "optimizer_runtime": optimizer_runtime["path"],
            "training_cadence_runtime": cadence_runtime["path"],
            "stage": str(expected_stage),
            "run_id": bundle.get("run_id"),
            "will_launch_training": True,
            "training_runtime_ported": True,
            "runner_status": launch_result["status"],
            "final_checkpoint": final_checkpoint,
            "rank": runtime_context["rank"],
            "local_rank": runtime_context["local_rank"],
            "world_size": runtime_context["world_size"],
        }
    finally:
        _cleanup_distributed(runtime_context)


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
        "training_runtime_ported": bool(native_status.get("launch_training_supported")),
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


def _training_plan_artifact_identities(plan_path: str | Path) -> dict[str, Any]:
    plan = Path(plan_path)
    plan_dir = plan.parent
    return {
        "training_plan": file_identity(plan).to_dict(),
        "training_plan_txt": file_identity(plan_dir / "training_plan.txt").to_dict(),
        "dataset_identity": file_identity(plan_dir / "dataset_identity.json").to_dict(),
        "clean_native_training_status": file_identity(
            plan_dir / "clean_native_training_status.json"
        ).to_dict(),
        "clean_prepare_execution_command": file_identity(
            plan_dir / "clean_prepare_execution_command.sh"
        ).to_dict(),
        "clean_training_command": file_identity(
            plan_dir / "clean_training_command.sh"
        ).to_dict(),
        "legacy_reference_command": file_identity(
            plan_dir / "legacy_reference_command.sh"
        ).to_dict(),
    }


def _torchrun_env_matches_plan(plan: dict[str, Any]) -> bool:
    plan_world_size = int(((plan.get("batch") or {}).get("world_size")) or 1)
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return plan_world_size > 1 and env_world_size == plan_world_size


def _prepare_training_execution_for_launch(
    *,
    plan_path: Path,
    plan: dict[str, Any],
    expected_stage: TrainingStage,
    execution_dir: str | Path | None,
    runtime_context: dict[str, Any] | None,
) -> dict[str, Any]:
    if runtime_context is None or not runtime_context.get("distributed"):
        return prepare_training_execution(
            plan_path,
            plan,
            expected_stage=expected_stage,
            execution_dir=execution_dir,
        )
    bundle_path = _execution_bundle_path(
        plan_path=plan_path,
        execution_dir=execution_dir,
        requested=None,
    )
    if runtime_context.get("is_main"):
        prepared = prepare_training_execution(
            plan_path,
            plan,
            expected_stage=expected_stage,
            execution_dir=execution_dir,
        )
    else:
        prepared = {
            "execution_bundle": str(bundle_path),
            "stage": str(expected_stage),
            "run_id": plan.get("run_id"),
            "distributed_prepare_owner": "rank0",
            "rank": runtime_context.get("rank"),
        }
    _distributed_barrier(runtime_context)
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"rank0 did not publish clean training execution bundle: {bundle_path}"
        )
    return prepared


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
    launch_supported = bool(native_status.get("launch_training_supported"))
    bundle: dict[str, Any] = {
        "training_execution_bundle_schema_version": TRAINING_EXECUTION_BUNDLE_SCHEMA_VERSION,
        "stage": str(expected_stage),
        "run_id": plan.get("run_id"),
        "output_dir": plan.get("output_dir"),
        "execution_dir": str(execution_dir),
        "plan_path": str(plan_path),
        "plan_identity": file_identity(plan_path).to_dict(),
        "plan_artifact_identities": _training_plan_artifact_identities(plan_path),
        "training_plan_schema_version": plan.get("training_plan_schema_version"),
        "git_commit": plan.get("git_commit"),
        "dirty_worktree": plan.get("dirty_worktree"),
        "model": plan.get("model"),
        "protocol": plan.get("protocol"),
        "dataset": plan.get("dataset"),
        "batch": plan.get("batch"),
        "tgvf": plan.get("tgvf"),
        "training": plan.get("training"),
        "module_policy": plan.get("module_policy"),
        "loss": plan.get("loss"),
        "optimizer": plan.get("optimizer"),
        "wandb": plan.get("wandb"),
        "clean_executor": {
            "entrypoint": f"revisit_vlm_clean.training.{expected_stage.value}_executor",
            "final_clean_native": True,
            "owns_execution_bundle": True,
            "trainer_loop_ported": launch_supported,
            "will_launch_training": False,
            "status": (
                "ready_for_explicit_single_process_launch"
                if launch_supported and native_status.get("runtime") == "single_process"
                else "ready_for_explicit_distributed_launch"
                if launch_supported and native_status.get("runtime") == "distributed_torchrun"
                else "trainer_loop_not_ported"
            ),
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
            "requires_explicit_launch_training_flag": True,
            "requires_clean_trainer_loop_before_launch": not launch_supported,
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
        "plan_artifact_identities": bundle.get("plan_artifact_identities"),
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
        (
            "plan_artifact_count: "
            f"{len(bundle.get('plan_artifact_identities') or {})}"
        ),
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
    launch_supported = bool(native_status.get("launch_training_supported"))
    runtime = native_status.get("runtime")
    status = (
        "single_process_launch_supported"
        if launch_supported and runtime == "single_process"
        else "distributed_torchrun_launch_supported"
        if launch_supported and runtime == "distributed_torchrun"
        else "not_ported"
    )
    return {
        "contract_schema_version": "clean_trainer_runtime_contract_v1",
        "stage": str(expected_stage),
        "entrypoint": f"revisit_vlm_clean.training.{expected_stage.value}_executor",
        "launch_function": "launch_training",
        "status": status,
        "launch_permitted": launch_supported,
        "runtime": runtime,
        "global_batch_size": (plan.get("batch") or {}).get("global_batch_size"),
        "required_launch_gates": _required_launch_gates(expected_stage),
        "required_runtime_artifacts": [
            "dataset_runtime_identity.json",
            "trainable_parameters.json",
            "optimizer_groups.json",
            "first_batch_identity.json",
            "checkpoint_contract.json",
            *(
                ["deepstack_training_plan.json"]
                if expected_stage == TrainingStage.STAGE2
                else []
            ),
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
        "validate_training_cadence_from_plan",
        "construct_optimizer_and_scheduler_from_plan",
        "run_backward_optimizer_scheduler_step_from_plan",
        "run_gradient_accumulation_loop_from_plan",
        "emit_trainable_parameter_audit",
        "save_checkpoint_with_clean_contract",
        "publish_training_checkpoint_after_trainer_loop",
        "resume_training_from_clean_checkpoint",
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


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_to_jsonable(payload), sort_keys=True) + "\n")
        handle.flush()


def _create_wandb_logger(**kwargs: Any) -> Any:
    from revisit_vlm.wandb_logging import WandbLogger

    return WandbLogger(**kwargs)


class _CleanTrainingProgressLogger:
    def __init__(
        self,
        *,
        execution_dir: Path,
        bundle: dict[str, Any],
        expected_stage: TrainingStage,
        runtime_context: dict[str, Any],
    ) -> None:
        self._is_main = bool(runtime_context.get("is_main"))
        self._started_at = time.time()
        self.records_written = 0
        self.path = execution_dir / "training_progress.jsonl" if self._is_main else None
        self._wandb_logger = None
        self._wandb_enabled = False
        self._wandb_project = None
        self._wandb_mode = None
        if not self._is_main:
            return
        wandb_config = dict(bundle.get("wandb") or {})
        self._wandb_project = wandb_config.get("project")
        self._wandb_mode = wandb_config.get("mode")
        if self._wandb_project and self._wandb_mode != "disabled":
            self._wandb_enabled = True
            self._wandb_logger = _create_wandb_logger(
                project=self._wandb_project,
                name=str(bundle.get("run_id") or ""),
                group=str(bundle.get("stage") or expected_stage.value),
                job_type="clean_training",
                mode=self._wandb_mode,
                config=_clean_training_wandb_config(bundle),
                directory=execution_dir / "wandb",
                tags=[
                    "clean-native",
                    str(expected_stage.value),
                    str(bundle.get("protocol") or ""),
                ],
            )

    def log_event(
        self,
        event: str,
        payload: dict[str, Any] | None = None,
        *,
        step: int | None = None,
        wandb_metrics: dict[str, Any] | None = None,
        stdout: str | None = None,
    ) -> None:
        if not self._is_main or self.path is None:
            return
        record = {
            "schema_version": "clean_training_progress_event_v1",
            "event": event,
            "elapsed_seconds": time.time() - self._started_at,
            **(payload or {}),
        }
        _append_jsonl(self.path, record)
        self.records_written += 1
        if stdout:
            print(stdout, flush=True)
        if self._wandb_logger is not None and wandb_metrics is not None:
            self._wandb_logger.log(wandb_metrics, step=step)

    def close(self, *, summary: dict[str, Any] | None = None) -> None:
        if self._wandb_logger is None:
            return
        if summary:
            self._wandb_logger.update_summary(summary)
        self._wandb_logger.finish()

    def summary(self) -> dict[str, Any]:
        return {
            "rank0_only": True,
            "progress_log_path": str(self.path) if self.path else None,
            "progress_records_written": self.records_written,
            "wandb_enabled": self._wandb_enabled,
            "wandb_project": self._wandb_project,
            "wandb_mode": self._wandb_mode,
        }


def _clean_training_wandb_config(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": bundle.get("run_id"),
        "stage": bundle.get("stage"),
        "protocol": bundle.get("protocol"),
        "git_commit": bundle.get("git_commit"),
        "dirty_worktree": bundle.get("dirty_worktree"),
        "model": bundle.get("model"),
        "dataset": bundle.get("dataset"),
        "batch": bundle.get("batch"),
        "training": bundle.get("training"),
        "loss": bundle.get("loss"),
        "optimizer": bundle.get("optimizer"),
        "module_policy": bundle.get("module_policy"),
    }


def _mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


def _micro_batch_sample_weight(
    *,
    sample_count: Any,
    nominal_micro_batch_size: int,
) -> float:
    nominal = max(1, int(nominal_micro_batch_size or 1))
    if isinstance(sample_count, (int, float)):
        actual = max(1, int(sample_count))
    else:
        actual = nominal
    return float(actual) / float(nominal)


def _loss_scalar_fields(result: dict[str, Any]) -> dict[str, float]:
    losses: dict[str, float] = {}
    for key, value in result.items():
        if not key.startswith("loss_") or key == "loss_tensor":
            continue
        scalar = _scalar_float(value)
        if scalar is not None and math.isfinite(scalar):
            losses[key] = scalar
    return losses


def _mean_loss_scalar_fields(micro_losses: list[dict[str, Any]]) -> dict[str, float | None]:
    values_by_key: dict[str, list[float]] = {}
    for item in micro_losses:
        for key, value in item.items():
            if not key.startswith("loss_") or not isinstance(value, (int, float)):
                continue
            scalar = float(value)
            if math.isfinite(scalar):
                values_by_key.setdefault(key, []).append(scalar)
    return {key: _mean(values) for key, values in sorted(values_by_key.items())}


def _compact_debug_payload(debug: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in debug.items():
        if key == "debug_examples":
            if isinstance(value, list):
                compact["debug_example_count"] = len(value)
            continue
        compact[key] = value
    return compact


def _weighted_mean_from_debug(
    debug_logs: list[dict[str, Any]],
    *,
    value_key: str,
    weight_key: str,
) -> float | None:
    total = 0.0
    weight_total = 0
    for item in debug_logs:
        value = item.get(value_key)
        weight = int(item.get(weight_key, 0) or 0)
        if weight <= 0 or not isinstance(value, (int, float)):
            continue
        total += float(value) * weight
        weight_total += weight
    return total / weight_total if weight_total else None


def _mean_debug_scalar(debug_logs: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(item[key])
        for item in debug_logs
        if isinstance(item.get(key), (int, float))
    ]
    return _mean(values)


def _sum_debug_int(debug_logs: list[dict[str, Any]], key: str) -> int:
    return sum(int(item.get(key, 0) or 0) for item in debug_logs)


def _merge_protocol_boundary_debug(debug_logs: list[dict[str, Any]]) -> dict[str, Any]:
    names = (
        "focus_start",
        "focus_end",
        "tgvf_start",
        "tgvf_end",
        "evidence_start",
        "evidence_end",
    )
    merged: dict[str, Any] = {}
    acc_values: list[float] = []
    total_support = 0
    for name in names:
        support_key = f"protocol_c_boundary_support_{name}"
        acc_key = f"protocol_c_boundary_acc_{name}"
        support = _sum_debug_int(debug_logs, support_key)
        total_support += support
        merged[support_key] = support
        if support <= 0:
            merged[acc_key] = None
            continue
        weighted = 0.0
        for item in debug_logs:
            item_support = int(item.get(support_key, 0) or 0)
            item_acc = item.get(acc_key)
            if item_support > 0 and isinstance(item_acc, (int, float)):
                weighted += float(item_acc) * item_support
        acc = weighted / support
        merged[acc_key] = acc
        acc_values.append(acc)
    merged["protocol_c_boundary_support_total"] = total_support
    merged["protocol_c_boundary_acc_mean"] = _mean(acc_values)
    return merged


def _summarize_training_debug(debug_logs: list[dict[str, Any]]) -> dict[str, Any]:
    if not debug_logs:
        return {}
    compact_logs = [_compact_debug_payload(item) for item in debug_logs]
    summary = dict(compact_logs[-1])
    if any("focus_count" in item or "no_focus_count" in item for item in debug_logs):
        focus = _sum_debug_int(debug_logs, "focus_count")
        no_focus = _sum_debug_int(debug_logs, "no_focus_count")
        summary.update(
            {
                "focus_count": focus,
                "single_focus_count": _sum_debug_int(debug_logs, "single_focus_count"),
                "multi_focus_count": _sum_debug_int(debug_logs, "multi_focus_count"),
                "no_focus_count": no_focus,
                "focus_ratio": focus / max(focus + no_focus, 1),
                "no_focus_ratio": no_focus / max(focus + no_focus, 1),
                "focus_sample_mask_active_rate": _weighted_mean_from_debug(
                    debug_logs,
                    value_key="focus_sample_mask_active_rate",
                    weight_key="focus_count",
                ),
                "no_focus_mask_active_rate": _weighted_mean_from_debug(
                    debug_logs,
                    value_key="no_focus_mask_active_rate",
                    weight_key="no_focus_count",
                ),
                "mask_original_image_after_tgvf_prob": _mean_debug_scalar(
                    debug_logs,
                    "mask_original_image_after_tgvf_prob",
                ),
                "value_span_match_rate": _mean_debug_scalar(
                    debug_logs,
                    "value_span_match_rate",
                ),
                "focus_loss_token_weight": _sum_debug_int(
                    debug_logs,
                    "focus_loss_token_weight",
                ),
                "no_focus_loss_token_weight": _sum_debug_int(
                    debug_logs,
                    "no_focus_loss_token_weight",
                ),
            }
        )
        scopes = [
            str(item.get("mask_original_image_after_tgvf_scope"))
            for item in debug_logs
            if item.get("mask_original_image_after_tgvf_scope") is not None
        ]
        if scopes:
            summary["mask_original_image_after_tgvf_scope"] = scopes[0]
        summary.update(_merge_protocol_boundary_debug(debug_logs))
    examples = []
    for item in debug_logs:
        item_examples = item.get("debug_examples")
        if isinstance(item_examples, list):
            examples.extend(item_examples)
        if len(examples) >= 2:
            break
    if examples:
        summary["debug_example_count"] = sum(
            len(item.get("debug_examples") or [])
            for item in debug_logs
            if isinstance(item.get("debug_examples"), list)
        )
    return summary


def _flatten_scalar_metrics(prefix: str, payload: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if isinstance(payload, bool):
        metrics[prefix] = float(payload)
    elif isinstance(payload, (int, float)) and math.isfinite(float(payload)):
        metrics[prefix] = float(payload)
    elif isinstance(payload, dict):
        for key, value in payload.items():
            nested = f"{prefix}/{key}" if prefix else str(key)
            metrics.update(_flatten_scalar_metrics(nested, value))
    elif isinstance(payload, (list, tuple)) and all(
        isinstance(item, (int, float)) for item in payload
    ):
        for index, value in enumerate(payload):
            if math.isfinite(float(value)):
                metrics[f"{prefix}/dim_{index}"] = float(value)
    return metrics


def _cuda_peak_memory_gb(torch_module: Any) -> float | None:
    try:
        if not torch_module.cuda.is_available():
            return None
        return float(torch_module.cuda.max_memory_allocated()) / float(1024**3)
    except Exception:
        return None


def _grad_total_norm(summary: dict[str, Any]) -> float | None:
    value = summary.get("total_norm")
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def _training_step_wandb_metrics(
    *,
    global_step: int,
    total_micro_steps: int,
    micro_losses: list[dict[str, Any]],
    grad_after_sync: dict[str, Any],
    grad_after_clip: dict[str, Any],
    clipped_grad_norm: float | None,
    scheduler_last_lr: list[Any],
    checkpoint_record: dict[str, Any] | None,
    validation_record: dict[str, Any] | None,
    debug_summary: dict[str, Any] | None = None,
    peak_memory_gb: float | None = None,
    effective_global_batch_size: int | None = None,
) -> dict[str, Any]:
    loss_means = _mean_loss_scalar_fields(micro_losses)
    sample_counts = [
        int(item.get("sample_count") or 0)
        for item in micro_losses
        if isinstance(item.get("sample_count"), (int, float))
    ]
    metrics: dict[str, Any] = {
        "trainer/global_step": global_step,
        "trainer/micro_steps_completed": total_micro_steps,
        "train/micro_step_count": len(micro_losses),
        "train/sample_count": sum(sample_counts),
        "train/grad_norm_after_sync": _grad_total_norm(grad_after_sync),
        "train/grad_norm_after_clip": _grad_total_norm(grad_after_clip),
        "train/clipped_grad_norm": clipped_grad_norm,
        "train/grad_norm": clipped_grad_norm,
        "train/peak_memory_gb": peak_memory_gb,
        "train/effective_global_batch_size": effective_global_batch_size,
        "checkpoint/saved": checkpoint_record is not None,
        "validation/ran": validation_record is not None,
    }
    for key, value in loss_means.items():
        metrics[f"train/{key}"] = value
    if debug_summary:
        for key, value in _flatten_scalar_metrics("train", debug_summary).items():
            metrics[key] = value
    for index, lr in enumerate(scheduler_last_lr):
        if isinstance(lr, (int, float)):
            metrics[f"train/lr_group_{index}"] = float(lr)
    if validation_record is not None:
        for key, value in _loss_scalar_fields(validation_record).items():
            metrics[f"validation/{key}"] = value
    return metrics


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
    if not isinstance(bundle.get("tgvf"), dict) or not bundle.get("tgvf"):
        raise ValueError("execution bundle must preserve the planned tgvf config")


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
    if expected_stage == TrainingStage.STAGE2:
        required.add("deepstack_training_plan")
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
    if expected_stage == TrainingStage.STAGE2:
        deepstack_plan = artifacts["deepstack_training_plan"]
        if deepstack_plan.get("schema_version") != "clean_deepstack_training_plan_v1":
            raise ValueError("deepstack_training_plan schema mismatch")
        if deepstack_plan.get("stage") != "stage2":
            raise ValueError("deepstack_training_plan stage mismatch")
        if deepstack_plan.get("gate_name") != "apply_deepstack_training_scope_when_enabled":
            raise ValueError("deepstack_training_plan gate mismatch")
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
    global_step: int = 0,
    optimizer_step: int = 0,
    micro_step: int = 0,
) -> dict[str, Any]:
    tgvf_module = _unwrap_distributed_data_parallel_module(modules.get("tgvf"))
    if tgvf_module is None or not hasattr(tgvf_module, "state_dict"):
        raise ValueError("checkpoint audit requires a tgvf module with state_dict")
    config = _checkpoint_runtime_config(
        bundle=bundle,
        loaded_modules=loaded_modules,
        expected_stage=expected_stage,
    )
    checkpoint: dict[str, Any] = {
        "tgvf_module": tgvf_module.state_dict(),
        "config": config,
        "global_step": int(global_step),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    if expected_stage == TrainingStage.STAGE1:
        checkpoint["optimizer_step"] = int(optimizer_step)
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
    checkpoint["micro_step"] = int(micro_step)
    return checkpoint


def _checkpoint_runtime_config(
    *,
    bundle: dict[str, Any],
    loaded_modules: dict[str, Any],
    expected_stage: TrainingStage,
) -> dict[str, Any]:
    model = bundle.get("model") or {}
    training = bundle.get("training") or {}
    lora = dict(bundle.get("lora") or {})
    if expected_stage == TrainingStage.STAGE2:
        token_mode = str(lora.get("protocol_token_training_mode") or "full_modules")
        lora["modules_to_save"] = (
            ["embed_tokens", "lm_head"] if token_mode == "full_modules" else None
        )
    return {
        "stage": str(expected_stage),
        "run_id": bundle.get("run_id"),
        "model_id": model.get("model_id"),
        "processor_id": model.get("processor_id"),
        "tgvf_protocol": bundle.get("protocol"),
        "training": training,
        "tgvf": _checkpoint_tgvf_config(bundle=bundle, loaded_modules=loaded_modules),
        "lora": lora if expected_stage == TrainingStage.STAGE2 else None,
    }


def _checkpoint_tgvf_config(
    *,
    bundle: dict[str, Any],
    loaded_modules: dict[str, Any],
) -> dict[str, Any]:
    loader = loaded_modules.get("loader") or {}
    resolved = loader.get("resolved_tgvf_config")
    if isinstance(resolved, Mapping) and resolved:
        return dict(resolved)
    planned = bundle.get("tgvf")
    if isinstance(planned, Mapping) and planned:
        return dict(planned)
    training = bundle.get("training") or {}
    return {
        "variant": training.get("variant"),
        "num_foveated_tokens": None,
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


def _write_actual_training_step_runtime_audit(
    *,
    execution_dir: Path,
    bundle: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    loaded_modules: dict[str, Any] | None,
    expected_stage: TrainingStage,
) -> dict[str, Any]:
    if loaded_modules is None:
        raise ValueError("training-step audit requires loaded model modules")
    result = (
        _run_stage1_training_step_probe(
            bundle=bundle,
            artifacts=artifacts,
            loaded_modules=loaded_modules,
        )
        if expected_stage == TrainingStage.STAGE1
        else _run_stage2_training_step_probe(
            bundle=bundle,
            artifacts=artifacts,
            loaded_modules=loaded_modules,
        )
    )
    debug = dict(result.get("debug") or {})
    loss_total = result.get("loss_total")
    loss_total_finite = loss_total is not None and math.isfinite(float(loss_total))
    stage_flags = (
        _stage1_training_step_flags(bundle=bundle, result=result, debug=debug)
        if expected_stage == TrainingStage.STAGE1
        else _stage2_training_step_flags(bundle=bundle, result=result, debug=debug)
    )
    payload = {
        "schema_version": "clean_training_step_runtime_audit_v1",
        "stage": bundle.get("stage"),
        "run_id": bundle.get("run_id"),
        "status": f"actual_{expected_stage.value}_training_step_forward_audit",
        "actual_training_step_forward": bool(result.get("forward_completed")),
        "backward_called": False,
        "optimizer_step_called": False,
        "scheduler_step_called": False,
        "loss_total": loss_total,
        "loss_total_finite": loss_total_finite,
        "loss_focus": result.get("loss_focus"),
        "loss_no_focus": result.get("loss_no_focus"),
        "loss_gen": result.get("loss_gen"),
        "loss_same_image_negative": result.get("loss_same_image_negative"),
        "loss_visual_token_manifold": result.get("loss_visual_token_manifold"),
        "loss_visual_token_norm": result.get("loss_visual_token_norm"),
        "sample_count": result.get("sample_count"),
        "focus_count": debug.get("focus_count"),
        "no_focus_count": debug.get("no_focus_count"),
        **stage_flags,
        "debug": debug,
        "notes": [
            f"{expected_stage.value} training-step probe ran a forward pass only",
            (
                "this audit does not call backward, optimizer.step, "
                "scheduler.step, or save checkpoints"
            ),
        ],
    }
    path = execution_dir / "training_step_runtime.json"
    _write_json(path, payload)
    return {"path": str(path), "payload": payload, "_step_result": result}


def _write_actual_optimizer_step_runtime_audit(
    *,
    execution_dir: Path,
    bundle: dict[str, Any],
    optimizer_runtime: dict[str, Any] | None,
    training_step_runtime: dict[str, Any] | None,
) -> dict[str, Any]:
    if optimizer_runtime is None:
        raise ValueError("optimizer-step audit requires optimizer runtime construction")
    if training_step_runtime is None:
        raise ValueError("optimizer-step audit requires a training-step forward probe")
    optimizer = optimizer_runtime.get("optimizer")
    scheduler = optimizer_runtime.get("scheduler")
    if optimizer is None or scheduler is None:
        raise ValueError("optimizer-step audit requires optimizer and scheduler objects")
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("optimizer-step audit requires torch") from exc

    step_result = dict(training_step_runtime.get("_step_result") or {})
    loss_tensor = step_result.get("loss_tensor")
    if loss_tensor is None:
        loss_tensor = step_result.get("_loss_total_tensor")
    if loss_tensor is None or not hasattr(loss_tensor, "backward"):
        raise ValueError("optimizer-step audit requires a differentiable loss tensor")
    if getattr(loss_tensor, "requires_grad", False) is not True:
        raise ValueError("optimizer-step audit loss tensor must require gradients")
    loss_value = _scalar_float(loss_tensor)
    if loss_value is None or not math.isfinite(loss_value):
        raise ValueError("optimizer-step audit requires a finite loss tensor")

    optimizer.zero_grad(set_to_none=True)
    loss_tensor.backward()
    grad_before_clip = _optimizer_grad_summary(optimizer)
    optimizer_contract = optimizer_runtime["payload"].get("optimizer") or {}
    max_grad_norm = _max_grad_norm_from_bundle(bundle, optimizer_runtime)
    clipped_grad_norm = None
    if max_grad_norm is not None:
        parameters = _optimizer_parameters(optimizer)
        clipped_grad_norm = _scalar_float(
            torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
        )
    grad_after_clip = _optimizer_grad_summary(optimizer)
    optimizer.step()
    scheduler.step()
    scheduler_last_lr = list(scheduler.get_last_lr())
    optimizer.zero_grad(set_to_none=True)
    grad_after_zero = _optimizer_grad_summary(optimizer)
    payload = {
        "schema_version": "clean_training_optimizer_step_runtime_audit_v1",
        "stage": bundle.get("stage"),
        "run_id": bundle.get("run_id"),
        "status": "actual_backward_optimizer_scheduler_step_audit",
        "actual_optimizer_step_probe": True,
        "backward_called": True,
        "optimizer_step_called": True,
        "scheduler_step_called": True,
        "zero_grad_called_before_backward": True,
        "zero_grad_called_after_step": True,
        "checkpoint_published": False,
        "training_loop_launched": False,
        "loss_total": loss_value,
        "loss_total_finite": True,
        "max_grad_norm": max_grad_norm,
        "clipped_grad_norm": clipped_grad_norm,
        "grad_before_clip": grad_before_clip,
        "grad_after_clip": grad_after_clip,
        "grad_after_zero": grad_after_zero,
        "optimizer": {
            "name": optimizer_contract.get("name") or "adamw",
            "param_group_count": len(optimizer.param_groups),
            "state_entry_count": len((optimizer.state_dict()).get("state") or {}),
        },
        "scheduler": {
            "name": (optimizer_runtime["payload"].get("scheduler") or {}).get("name"),
            "last_lr_after_step": scheduler_last_lr,
            "state_dict_keys": sorted(str(key) for key in scheduler.state_dict().keys()),
        },
        "notes": [
            "one bounded optimizer-step probe ran from the clean training-step loss",
            (
                "this audit does not enter an epoch loop, accumulate gradients, "
                "or publish checkpoints"
            ),
        ],
    }
    path = execution_dir / "optimizer_step_runtime.json"
    _write_json(path, payload)
    return {"path": str(path), "payload": payload}


def _write_actual_trainer_loop_runtime_audit(
    *,
    execution_dir: Path,
    bundle: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    loaded_modules: dict[str, Any] | None,
    optimizer_runtime: dict[str, Any] | None,
    expected_stage: TrainingStage,
) -> dict[str, Any]:
    if loaded_modules is None:
        raise ValueError("trainer-loop audit requires loaded model modules")
    if optimizer_runtime is None:
        raise ValueError("trainer-loop audit requires optimizer runtime construction")
    optimizer = optimizer_runtime.get("optimizer")
    scheduler = optimizer_runtime.get("scheduler")
    if optimizer is None or scheduler is None:
        raise ValueError("trainer-loop audit requires optimizer and scheduler objects")
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("trainer-loop audit requires torch") from exc

    batch = bundle.get("batch") or {}
    accumulation_steps = max(1, int(batch.get("gradient_accumulation_steps") or 1))
    nominal_micro_batch_size = max(1, int(batch.get("micro_batch_size") or 1))
    optimizer.zero_grad(set_to_none=True)
    micro_losses = []
    backward_micro_steps = 0
    for micro_step in range(accumulation_steps):
        result = _run_training_step_probe_for_stage(
            bundle=bundle,
            artifacts=artifacts,
            loaded_modules=loaded_modules,
            expected_stage=expected_stage,
        )
        loss_tensor = result.get("loss_tensor")
        if loss_tensor is None:
            loss_tensor = result.get("_loss_total_tensor")
        if loss_tensor is None or not hasattr(loss_tensor, "backward"):
            raise ValueError("trainer-loop audit requires a differentiable loss tensor")
        if getattr(loss_tensor, "requires_grad", False) is not True:
            raise ValueError("trainer-loop audit loss tensor must require gradients")
        loss_value = _scalar_float(loss_tensor)
        if loss_value is None or not math.isfinite(loss_value):
            raise ValueError("trainer-loop audit requires finite micro losses")
        sample_weight = _micro_batch_sample_weight(
            sample_count=result.get("sample_count"),
            nominal_micro_batch_size=nominal_micro_batch_size,
        )
        scaled_loss = loss_tensor * sample_weight / float(accumulation_steps)
        scaled_loss.backward()
        backward_micro_steps += 1
        micro_losses.append(
            {
                "micro_step": micro_step,
                "loss_total": loss_value,
                "scaled_loss_total": _scalar_float(scaled_loss),
                "sample_count": result.get("sample_count"),
                "nominal_micro_batch_size": nominal_micro_batch_size,
                "loss_sample_weight": sample_weight,
            }
        )

    grad_after_accumulation = _optimizer_grad_summary(optimizer)
    max_grad_norm = _max_grad_norm_from_bundle(bundle, optimizer_runtime)
    clipped_grad_norm = None
    if max_grad_norm is not None:
        clipped_grad_norm = _scalar_float(
            torch.nn.utils.clip_grad_norm_(_optimizer_parameters(optimizer), max_grad_norm)
        )
    grad_after_clip = _optimizer_grad_summary(optimizer)
    optimizer.step()
    scheduler.step()
    scheduler_last_lr = list(scheduler.get_last_lr())
    optimizer.zero_grad(set_to_none=True)
    grad_after_zero = _optimizer_grad_summary(optimizer)
    payload = {
        "schema_version": "clean_training_trainer_loop_runtime_audit_v1",
        "stage": bundle.get("stage"),
        "run_id": bundle.get("run_id"),
        "status": "actual_gradient_accumulation_trainer_loop_audit",
        "actual_trainer_loop_probe": True,
        "training_run_launched": False,
        "full_epoch_loop_entered": False,
        "checkpoint_published": False,
        "optimizer_steps_completed": 1,
        "scheduler_steps_completed": 1,
        "gradient_accumulation_steps": accumulation_steps,
        "micro_steps_run": len(micro_losses),
        "backward_micro_steps": backward_micro_steps,
        "micro_losses": micro_losses,
        "max_grad_norm": max_grad_norm,
        "clipped_grad_norm": clipped_grad_norm,
        "grad_after_accumulation": grad_after_accumulation,
        "grad_after_clip": grad_after_clip,
        "grad_after_zero": grad_after_zero,
        "scheduler": {
            "name": (optimizer_runtime["payload"].get("scheduler") or {}).get("name"),
            "last_lr_after_step": scheduler_last_lr,
        },
        "notes": [
            "one bounded gradient-accumulation trainer-loop probe ran",
            (
                "this audit proves micro-step accumulation and optimizer/scheduler "
                "ordering only; it does not launch a full training run"
            ),
        ],
    }
    path = execution_dir / "trainer_loop_runtime.json"
    _write_json(path, payload)
    return {"path": str(path), "payload": payload}


def _write_training_checkpoint_publish_runtime_audit(
    *,
    execution_dir: Path,
    bundle: dict[str, Any],
    loaded_modules: dict[str, Any] | None,
    optimizer_runtime: dict[str, Any] | None,
    trainer_loop_runtime: dict[str, Any] | None,
    expected_stage: TrainingStage,
) -> dict[str, Any]:
    if loaded_modules is None:
        raise ValueError("checkpoint-publish audit requires loaded model modules")
    if optimizer_runtime is None:
        raise ValueError("checkpoint-publish audit requires optimizer runtime construction")
    if trainer_loop_runtime is None:
        raise ValueError("checkpoint-publish audit requires trainer-loop runtime")
    trainer_payload = trainer_loop_runtime["payload"]
    if not trainer_payload.get("actual_trainer_loop_probe"):
        raise ValueError("checkpoint-publish audit requires a successful trainer-loop probe")
    optimizer = optimizer_runtime.get("optimizer")
    scheduler = optimizer_runtime.get("scheduler")
    if optimizer is None or scheduler is None:
        raise ValueError("checkpoint-publish audit requires optimizer and scheduler objects")
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("checkpoint-publish audit requires torch") from exc

    modules = dict(loaded_modules.get("modules") or {})
    global_step = int(trainer_payload.get("optimizer_steps_completed") or 0)
    micro_step = int(trainer_payload.get("micro_steps_run") or 0)
    optimizer_step = global_step
    checkpoint_path = execution_dir / "training_checkpoint_publish_probe_step_1.pt"
    checkpoint = _checkpoint_probe_payload(
        bundle=bundle,
        loaded_modules=loaded_modules,
        modules=modules,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_stage=expected_stage,
        global_step=global_step,
        optimizer_step=optimizer_step,
        micro_step=micro_step,
    )
    torch.save(checkpoint, checkpoint_path)
    loaded = _torch_load_checkpoint_probe(torch, checkpoint_path)
    required_keys = _checkpoint_required_output_keys(expected_stage)
    missing_keys = sorted(key for key in required_keys if key not in loaded)
    if missing_keys:
        raise ValueError(f"checkpoint publish probe missing keys after load: {missing_keys}")
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
    step_checks = _checkpoint_publish_step_checks(
        loaded=loaded,
        expected_stage=expected_stage,
        global_step=global_step,
        optimizer_step=optimizer_step,
        micro_step=micro_step,
    )
    step_checks_ok = all(check.get("ok") is True for check in step_checks.values())
    payload = {
        "schema_version": "clean_training_checkpoint_publish_runtime_audit_v1",
        "stage": bundle.get("stage"),
        "run_id": bundle.get("run_id"),
        "status": "actual_training_checkpoint_publish_audit",
        "actual_checkpoint_publish_probe_saved": True,
        "actual_checkpoint_publish_probe_loaded": True,
        "training_run_launched": False,
        "full_epoch_loop_entered": False,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_identity": file_identity(checkpoint_path).to_dict(),
        "required_output_keys": required_keys,
        "saved_key_names": sorted(str(key) for key in checkpoint.keys()),
        "loaded_key_names": sorted(str(key) for key in loaded.keys()),
        "missing_required_keys": missing_keys,
        "global_step": loaded.get("global_step"),
        "optimizer_step": loaded.get("optimizer_step"),
        "micro_step": loaded.get("micro_step"),
        "optimizer_state_loaded": loaded.get("optimizer") is not None,
        "scheduler_state_loaded": loaded.get("scheduler") is not None,
        "state_checks_ok": state_checks_ok,
        "state_checks": state_checks,
        "step_checks_ok": step_checks_ok,
        "step_checks": step_checks,
        "protocol_c_token_rows": _protocol_token_rows_summary(
            loaded.get("protocol_c_token_rows")
        ),
        "trainer_loop_runtime_status": trainer_payload.get("status"),
        "notes": [
            "checkpoint publish probe was saved after a bounded trainer-loop audit",
            (
                "this audit proves clean checkpoint publish semantics without "
                "launching a full training run"
            ),
        ],
    }
    path = execution_dir / "training_checkpoint_publish_runtime.json"
    _write_json(path, payload)
    return {"path": str(path), "payload": payload, "checkpoint_path": str(checkpoint_path)}


def _checkpoint_publish_step_checks(
    *,
    loaded: dict[str, Any],
    expected_stage: TrainingStage,
    global_step: int,
    optimizer_step: int,
    micro_step: int,
) -> dict[str, Any]:
    checks = {
        "global_step": {
            "ok": int(loaded.get("global_step") or -1) == int(global_step),
            "expected": int(global_step),
            "actual": loaded.get("global_step"),
        }
    }
    if expected_stage == TrainingStage.STAGE1:
        checks["optimizer_step"] = {
            "ok": int(loaded.get("optimizer_step") or -1) == int(optimizer_step),
            "expected": int(optimizer_step),
            "actual": loaded.get("optimizer_step"),
        }
    else:
        checks["micro_step"] = {
            "ok": int(loaded.get("micro_step") or -1) == int(micro_step),
            "expected": int(micro_step),
            "actual": loaded.get("micro_step"),
        }
    return checks


def _write_training_checkpoint_resume_runtime_audit(
    *,
    execution_dir: Path,
    bundle: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    checkpoint_publish_runtime: dict[str, Any] | None,
    expected_stage: TrainingStage,
) -> dict[str, Any]:
    if not _checkpoint_publish_runtime_validated(checkpoint_publish_runtime):
        raise ValueError("checkpoint-resume audit requires a valid checkpoint-publish audit")
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("checkpoint-resume audit requires torch") from exc

    publish_payload = checkpoint_publish_runtime["payload"]  # type: ignore[index]
    checkpoint_path = Path(str(publish_payload.get("checkpoint_path") or ""))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint-resume audit missing checkpoint: {checkpoint_path}")
    checkpoint = _torch_load_checkpoint_probe(torch, checkpoint_path)
    resume_loaded_modules = _load_training_parameter_audit_modules(
        bundle,
        expected_stage=expected_stage,
    )
    resume_modules = dict(resume_loaded_modules.get("modules") or {})
    model_load = _load_checkpoint_model_states_for_resume(
        checkpoint=checkpoint,
        bundle=bundle,
        loaded_modules=resume_loaded_modules,
        modules=resume_modules,
        expected_stage=expected_stage,
    )
    constructed = _construct_actual_optimizer_scheduler(
        torch_module=torch,
        bundle=bundle,
        artifacts=artifacts,
        loaded_modules=resume_loaded_modules,
        expected_stage=expected_stage,
    )
    resume_optimizer = constructed["optimizer"]
    resume_scheduler = constructed["scheduler"]
    _reload_optimizer_scheduler_probe(
        optimizer=resume_optimizer,
        scheduler=resume_scheduler,
        optimizer_state=checkpoint.get("optimizer"),
        scheduler_state=checkpoint.get("scheduler"),
    )
    state_checks = _resume_state_checks(
        checkpoint=checkpoint,
        modules=resume_modules,
        expected_stage=expected_stage,
    )
    state_checks_ok = all(check.get("ok") is True for check in state_checks.values())
    global_step = int(publish_payload.get("global_step") or 0)
    optimizer_step = int(publish_payload.get("optimizer_step") or global_step)
    micro_step = int(publish_payload.get("micro_step") or 0)
    step_checks = _checkpoint_publish_step_checks(
        loaded=checkpoint,
        expected_stage=expected_stage,
        global_step=global_step,
        optimizer_step=optimizer_step,
        micro_step=micro_step,
    )
    step_checks_ok = all(check.get("ok") is True for check in step_checks.values())
    protocol_rows = model_load.get("protocol_c_token_rows") or {}
    protocol_rows_ok = bool(protocol_rows.get("ok", True))
    payload = {
        "schema_version": "clean_training_checkpoint_resume_runtime_audit_v1",
        "stage": bundle.get("stage"),
        "run_id": bundle.get("run_id"),
        "status": "actual_training_checkpoint_resume_audit",
        "actual_checkpoint_resume_probe_loaded": True,
        "training_run_launched": False,
        "full_epoch_loop_entered": False,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_identity": file_identity(checkpoint_path).to_dict(),
        "fresh_loader": resume_loaded_modules.get("loader") or {},
        "model_state_loaded": bool(model_load.get("model_state_loaded")),
        "model_load": model_load,
        "optimizer_state_loaded": checkpoint.get("optimizer") is not None,
        "scheduler_state_loaded": checkpoint.get("scheduler") is not None,
        "optimizer": {
            "param_group_count": len(resume_optimizer.param_groups),
            "state_entry_count": len((resume_optimizer.state_dict()).get("state") or {}),
            "constructed_group_names": [
                group["name"] for group in constructed["constructed_groups"]
            ],
            "empty_planned_group_names": constructed["empty_groups"],
        },
        "scheduler": {
            "last_lr": list(resume_scheduler.get_last_lr()),
            "state_dict_keys": sorted(str(key) for key in resume_scheduler.state_dict().keys()),
        },
        "global_step": checkpoint.get("global_step"),
        "optimizer_step": checkpoint.get("optimizer_step"),
        "micro_step": checkpoint.get("micro_step"),
        "state_checks_ok": state_checks_ok,
        "state_checks": state_checks,
        "step_checks_ok": step_checks_ok,
        "step_checks": step_checks,
        "protocol_rows_ok": protocol_rows_ok,
        "notes": [
            "checkpoint resume probe loaded a fresh model/optimizer/scheduler stack",
            (
                "this audit proves clean resume wiring without launching a full "
                "training run"
            ),
        ],
    }
    path = execution_dir / "training_checkpoint_resume_runtime.json"
    _write_json(path, payload)
    return {"path": str(path), "payload": payload}


def _write_training_cadence_runtime_audit(
    *,
    execution_dir: Path,
    bundle: dict[str, Any],
    expected_stage: TrainingStage,
) -> dict[str, Any]:
    training = bundle.get("training") or {}
    dataset = bundle.get("dataset") or {}
    max_steps = int(training.get("max_steps") or 0)
    save_every = int(training.get("save_every") or 0)
    if max_steps < 1:
        raise ValueError("training cadence audit requires training.max_steps >= 1")
    if save_every < 1:
        raise ValueError("training cadence audit requires training.save_every >= 1")
    checkpoint_steps = _cadence_steps(max_steps=max_steps, every=save_every)
    val_file = dataset.get("val_file") or {}
    val_file_available = bool(val_file.get("exists")) if isinstance(val_file, Mapping) else False
    eval_every = training.get("eval_every")
    eval_enabled = expected_stage == TrainingStage.STAGE2 and val_file_available
    eval_steps: list[int] = []
    if eval_enabled:
        eval_every_int = int(eval_every or 0)
        if eval_every_int < 1:
            raise ValueError("Stage2 evaluation cadence requires training.eval_every >= 1")
        eval_steps = _cadence_steps(max_steps=max_steps, every=eval_every_int)
    payload = {
        "schema_version": "clean_training_cadence_runtime_audit_v1",
        "stage": bundle.get("stage"),
        "run_id": bundle.get("run_id"),
        "status": "actual_training_cadence_audit",
        "actual_training_cadence_validated": True,
        "training_run_launched": False,
        "max_steps": max_steps,
        "save_every": save_every,
        "checkpoint_save_steps": checkpoint_steps,
        "checkpoint_save_count": len(checkpoint_steps),
        "final_checkpoint_saved": checkpoint_steps[-1] == max_steps,
        "eval_every": eval_every,
        "eval_enabled": eval_enabled,
        "eval_disabled_reason": None if eval_enabled else _eval_disabled_reason(
            expected_stage=expected_stage,
            val_file_available=val_file_available,
        ),
        "eval_steps": eval_steps,
        "eval_count": len(eval_steps),
        "final_eval_scheduled": bool(eval_steps and eval_steps[-1] == max_steps),
        "val_file_available": val_file_available,
        "notes": [
            "training cadence was resolved from the clean plan without launching training",
            "checkpoint cadence follows step % save_every == 0 or step == max_steps",
            "Stage2 eval cadence is enabled only when a validation file is present",
        ],
    }
    path = execution_dir / "training_cadence_runtime.json"
    _write_json(path, payload)
    return {"path": str(path), "payload": payload}


def _write_training_launch_readiness_audit(
    *,
    execution_dir: Path,
    bundle: dict[str, Any],
    audit: dict[str, Any],
    expected_stage: TrainingStage,
) -> dict[str, Any]:
    gates = audit.get("launch_gates") or {}
    gate_rows = list(gates.get("gates") or [])
    pending_gates = [
        gate["name"]
        for gate in gate_rows
        if gate.get("status") == "pending_real_trainer_loop"
    ]
    unknown_gates = [
        gate["name"] for gate in gate_rows if gate.get("status") == "unknown_gate"
    ]
    blocked_gates = [
        gate["name"]
        for gate in gate_rows
        if str(gate.get("status") or "").startswith("blocked_")
    ]
    identity_validated_gates = [
        gate["name"]
        for gate in gate_rows
        if gate.get("status") == "identity_validated"
    ]
    blocking_items = list(audit.get("blocking_items") or [])
    expected_nonlaunch_blocker = (
        "native trainer loop has not been ported into revisit_vlm_clean"
    )
    unexpected_blockers = [
        item for item in blocking_items if item != expected_nonlaunch_blocker
    ]
    non_identity_gates = [
        gate["name"]
        for gate in gate_rows
        if gate.get("status") != "identity_validated"
    ]
    all_required_gates_identity_validated = bool(gate_rows) and not (
        non_identity_gates
    )
    contract_ready_for_trainer_loop = (
        all_required_gates_identity_validated and not unexpected_blockers
    )
    training_runtime_ported = bool(
        (bundle.get("clean_executor") or {}).get("trainer_loop_ported")
    )
    if contract_ready_for_trainer_loop and training_runtime_ported:
        status = "launch_contract_ready_explicit_launch_required"
        launch_permitted = True
        launch_disabled_reason = None
    elif contract_ready_for_trainer_loop:
        status = "launch_contract_ready_trainer_loop_disabled"
        launch_permitted = False
        launch_disabled_reason = "native_trainer_loop_not_enabled"
    else:
        status = "blocked_before_launch"
        launch_permitted = False
        launch_disabled_reason = "launch_contract_not_ready"
    deepstack = dict(bundle.get("deepstack") or {})
    deepstack_gate_status = next(
        (
            gate.get("status")
            for gate in gate_rows
            if gate.get("name") == "apply_deepstack_training_scope_when_enabled"
        ),
        "not_applicable",
    )
    payload = {
        "schema_version": "clean_training_launch_readiness_v1",
        "stage": str(expected_stage),
        "run_id": bundle.get("run_id"),
        "status": status,
        "will_launch_training": False,
        "training_runtime_ported": training_runtime_ported,
        "launch_permitted": launch_permitted,
        "launch_disabled_reason": launch_disabled_reason,
        "required_gates_total": int(gates.get("total") or len(gate_rows)),
        "identity_validated_gates_total": int(
            gates.get("identity_validated") or len(identity_validated_gates)
        ),
        "all_required_gates_identity_validated": all_required_gates_identity_validated,
        "contract_ready_for_trainer_loop": contract_ready_for_trainer_loop,
        "identity_validated_gates": identity_validated_gates,
        "pending_gates": pending_gates,
        "unknown_gates": unknown_gates,
        "blocked_gates": blocked_gates,
        "non_identity_gates": non_identity_gates,
        "remaining_blockers": blocking_items,
        "expected_nonlaunch_blocker": (
            expected_nonlaunch_blocker
            if expected_nonlaunch_blocker in blocking_items
            else None
        ),
        "unexpected_blockers": unexpected_blockers,
        "artifact_statuses": {
            "trainable_parameters": gates.get("trainable_parameters_status"),
            "optimizer_runtime": gates.get("optimizer_runtime_status"),
            "training_step_runtime": gates.get("training_step_runtime_status"),
            "trainer_loop_runtime": gates.get("trainer_loop_runtime_status"),
            "training_checkpoint_publish_runtime": gates.get(
                "training_checkpoint_publish_runtime_status"
            ),
            "training_checkpoint_resume_runtime": gates.get(
                "training_checkpoint_resume_runtime_status"
            ),
            "training_cadence_runtime": gates.get("training_cadence_runtime_status"),
        },
        "deepstack": {
            "enabled": bool(deepstack.get("enabled")),
            "original_image_scope": deepstack.get("original_image_scope"),
            "mask_scope": deepstack.get("mask_scope"),
            "training_scope_gate_status": deepstack_gate_status,
            "training_plan": bundle.get("deepstack_training_plan"),
        },
        "notes": [
            "readiness summarizes existing clean runtime audit gates",
            "this artifact never flips will_launch_training to true",
            "launch_permitted=true means an explicit --launch-training command is allowed",
        ],
    }
    path = execution_dir / "training_launch_readiness.json"
    _write_json(path, payload)
    return {"path": str(path), "payload": payload}


def _setup_launch_runtime_context(bundle: dict[str, Any]) -> dict[str, Any]:
    batch = bundle.get("batch") or {}
    plan_world_size = int(batch.get("world_size") or 1)
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = plan_world_size > 1
    if distributed and env_world_size != plan_world_size:
        raise ValueError(
            "distributed clean launch requires torchrun WORLD_SIZE to match "
            f"batch.world_size ({env_world_size} != {plan_world_size})"
        )
    if not distributed and env_world_size > 1:
        raise ValueError(
            "single-process clean launch cannot run under torchrun WORLD_SIZE>1"
        )
    context: dict[str, Any] = {
        "distributed": distributed,
        "rank": rank if distributed else 0,
        "local_rank": local_rank if distributed else 0,
        "world_size": plan_world_size,
        "is_main": (rank if distributed else 0) == 0,
        "device_map": None,
        "device": None,
        "backend": None,
        "process_group_initialized_by_clean_executor": False,
    }
    if not distributed:
        return context
    try:
        import torch
        import torch.distributed as dist
    except Exception as exc:
        raise RuntimeError("distributed clean launch requires torch.distributed") from exc
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        context["device"] = f"cuda:{local_rank}"
        context["device_map"] = f"cuda:{local_rank}"
        backend = "nccl"
    else:
        context["device"] = "cpu"
        context["device_map"] = None
        backend = "gloo"
    context["backend"] = backend
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
        context["process_group_initialized_by_clean_executor"] = True
    return context


def _bundle_with_runtime_context(
    bundle: dict[str, Any],
    runtime_context: dict[str, Any],
) -> dict[str, Any]:
    if not runtime_context.get("distributed"):
        return bundle
    updated = dict(bundle)
    model = dict(updated.get("model") or {})
    model["device_map"] = runtime_context.get("device_map")
    updated["model"] = model
    updated["distributed_runtime"] = {
        "enabled": True,
        "rank": runtime_context.get("rank"),
        "local_rank": runtime_context.get("local_rank"),
        "world_size": runtime_context.get("world_size"),
        "device": runtime_context.get("device"),
        "device_map": runtime_context.get("device_map"),
        "backend": runtime_context.get("backend"),
        "gradient_sync": "all_reduce_trainable_gradients_before_clip",
        "checkpoint_writer": "rank0_only",
    }
    return updated


def _distributed_barrier(runtime_context: dict[str, Any]) -> None:
    if not runtime_context.get("distributed"):
        return
    try:
        import torch.distributed as dist
    except Exception:
        return
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _cleanup_distributed(runtime_context: dict[str, Any]) -> None:
    if not runtime_context.get("process_group_initialized_by_clean_executor"):
        return
    try:
        import torch.distributed as dist
    except Exception:
        return
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _validate_training_launch_contract(
    *,
    bundle: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    expected_stage: TrainingStage,
    runtime_context: dict[str, Any],
) -> None:
    batch = bundle.get("batch") or {}
    world_size = int(batch.get("world_size") or 0)
    if world_size != int(runtime_context.get("world_size") or 0):
        raise ValueError("runtime world_size does not match batch.world_size")
    if world_size > 1 and not runtime_context.get("distributed"):
        raise ValueError("distributed clean launch requires torchrun runtime context")
    if expected_stage == TrainingStage.STAGE2:
        deepstack_plan = artifacts.get("deepstack_training_plan") or {}
        if not bool(deepstack_plan.get("execution_supported", True)):
            raise ValueError(
                "clean Stage2 launch does not yet support DeepStack training injection: "
                + "; ".join(str(item) for item in deepstack_plan.get("blocking_items") or [])
            )
    if artifacts["optimizer_groups"].get("status") != "validated":
        raise ValueError("optimizer groups must be validated before launch")


def _write_single_process_training_runtime(
    *,
    execution_dir: Path,
    bundle: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    loaded_modules: dict[str, Any],
    optimizer_runtime: dict[str, Any],
    cadence_runtime: dict[str, Any],
    expected_stage: TrainingStage,
    runtime_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("single-process training launch requires torch") from exc

    optimizer = optimizer_runtime.get("optimizer")
    scheduler = optimizer_runtime.get("scheduler")
    if optimizer is None or scheduler is None:
        raise ValueError("single-process training requires optimizer and scheduler objects")
    runtime_context = runtime_context or {
        "distributed": False,
        "rank": 0,
        "local_rank": 0,
        "world_size": 1,
        "is_main": True,
    }
    distributed = bool(runtime_context.get("distributed"))
    rank = int(runtime_context.get("rank") or 0)
    world_size = int(runtime_context.get("world_size") or 1)
    is_main = bool(runtime_context.get("is_main", rank == 0))
    modules = dict(loaded_modules.get("modules") or {})
    training = bundle.get("training") or {}
    batch = bundle.get("batch") or {}
    max_steps = int(training.get("max_steps") or 0)
    if max_steps < 1:
        raise ValueError("single-process training requires training.max_steps >= 1")
    accumulation_steps = max(1, int(batch.get("gradient_accumulation_steps") or 1))
    nominal_micro_batch_size = max(1, int(batch.get("micro_batch_size") or 1))
    checkpoint_steps = {
        int(step) for step in (cadence_runtime["payload"].get("checkpoint_save_steps") or [])
    }
    eval_steps = {
        int(step) for step in (cadence_runtime["payload"].get("eval_steps") or [])
    }
    if max_steps not in checkpoint_steps:
        raise ValueError("single-process training requires final checkpoint cadence")
    max_grad_norm = _max_grad_norm_from_bundle(bundle, optimizer_runtime)
    train_cursor = _build_single_process_sample_cursor(
        bundle=bundle,
        expected_stage=expected_stage,
        dataset_role="train",
        rank=rank,
        world_size=world_size,
    )
    validation_cursor = (
        _build_single_process_sample_cursor(
            bundle=bundle,
            expected_stage=expected_stage,
            dataset_role="val",
            rank=0,
            world_size=1,
        )
        if eval_steps and is_main
        else None
    )
    step_records = []
    checkpoint_records = []
    validation_records = []
    total_micro_steps = 0
    progress_logger = _CleanTrainingProgressLogger(
        execution_dir=execution_dir,
        bundle=bundle,
        expected_stage=expected_stage,
        runtime_context=runtime_context,
    )
    progress_logger.log_event(
        "start",
        {
            "stage": bundle.get("stage"),
            "run_id": bundle.get("run_id"),
            "rank": rank,
            "world_size": world_size,
            "max_steps": max_steps,
            "gradient_accumulation_steps": accumulation_steps,
            "ddp_enabled": distributed,
            "wandb_logging": progress_logger.summary(),
        },
        stdout=(
            f"[clean-train] start stage={bundle.get('stage')} run_id={bundle.get('run_id')} "
            f"max_steps={max_steps} accum={accumulation_steps} "
            f"wandb_enabled={progress_logger.summary()['wandb_enabled']}"
        ),
    )
    try:
        optimizer.zero_grad(set_to_none=True)
        for global_step in range(1, max_steps + 1):
            micro_losses = []
            micro_debug_logs = []
            try:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
            optimizer.zero_grad(set_to_none=True)
            for micro_index in range(accumulation_steps):
                batch_record = train_cursor.next_batch()
                with _training_micro_step_sync_context(
                    modules=modules,
                    expected_stage=expected_stage,
                    micro_index=micro_index,
                    accumulation_steps=accumulation_steps,
                ):
                    result = _run_training_step_probe_for_stage(
                        bundle=bundle,
                        artifacts=artifacts,
                        loaded_modules=loaded_modules,
                        expected_stage=expected_stage,
                        samples=batch_record["samples"],
                    )
                    loss_tensor = result.get("loss_tensor")
                    if loss_tensor is None:
                        loss_tensor = result.get("_loss_total_tensor")
                    if loss_tensor is None or not hasattr(loss_tensor, "backward"):
                        raise ValueError("single-process training requires a differentiable loss")
                    if getattr(loss_tensor, "requires_grad", False) is not True:
                        raise ValueError("single-process training loss tensor must require gradients")
                    loss_value = _scalar_float(loss_tensor)
                    if loss_value is None or not math.isfinite(loss_value):
                        raise ValueError("single-process training requires finite losses")
                    loss_scalars = _loss_scalar_fields(result)
                    loss_scalars["loss_total"] = loss_value
                    debug_payload = dict(result.get("debug") or {})
                    if debug_payload:
                        micro_debug_logs.append(debug_payload)
                    sample_weight = _micro_batch_sample_weight(
                        sample_count=result.get("sample_count"),
                        nominal_micro_batch_size=nominal_micro_batch_size,
                    )
                    scaled_loss = loss_tensor * sample_weight / float(accumulation_steps)
                    scaled_loss.backward()
                total_micro_steps += 1
                micro_losses.append(
                    {
                        "micro_step": total_micro_steps,
                        "micro_index": micro_index,
                        **loss_scalars,
                        "scaled_loss_total": _scalar_float(scaled_loss),
                        "sample_count": result.get("sample_count"),
                        "nominal_micro_batch_size": nominal_micro_batch_size,
                        "loss_sample_weight": sample_weight,
                        "debug_summary": _compact_debug_payload(debug_payload)
                        if debug_payload
                        else None,
                        "sample_trace": batch_record["sample_trace"],
                    }
                )
            grad_after_accumulation = _optimizer_grad_summary(optimizer)
            if distributed:
                _average_optimizer_gradients(
                    optimizer,
                    world_size=world_size,
                    skip_parameter_ids=_distributed_data_parallel_parameter_ids(modules),
                )
                _distributed_barrier(runtime_context)
            grad_after_sync = _optimizer_grad_summary(optimizer)
            clipped_grad_norm = None
            if max_grad_norm is not None:
                clipped_grad_norm = _scalar_float(
                    torch.nn.utils.clip_grad_norm_(
                        _optimizer_parameters(optimizer),
                        max_grad_norm,
                    )
                )
            grad_after_clip = _optimizer_grad_summary(optimizer)
            optimizer.step()
            scheduler.step()
            scheduler_last_lr = list(scheduler.get_last_lr())
            optimizer.zero_grad(set_to_none=True)
            peak_memory_gb = _cuda_peak_memory_gb(torch)
            step_records.append(
                {
                    "global_step": global_step,
                    "micro_steps": micro_losses,
                    "grad_after_accumulation": grad_after_accumulation,
                    "grad_after_sync": grad_after_sync,
                    "clipped_grad_norm": clipped_grad_norm,
                    "grad_after_clip": grad_after_clip,
                    "scheduler_last_lr": scheduler_last_lr,
                    "peak_memory_gb": peak_memory_gb,
                }
            )
            checkpoint_record = None
            if is_main and global_step in checkpoint_steps:
                checkpoint_record = _save_clean_training_checkpoint(
                    execution_dir=execution_dir,
                    bundle=bundle,
                    loaded_modules=loaded_modules,
                    modules=modules,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    expected_stage=expected_stage,
                    global_step=global_step,
                    optimizer_step=global_step,
                    micro_step=total_micro_steps,
                )
                checkpoint_records.append(checkpoint_record)
            validation_record = None
            if is_main and global_step in eval_steps:
                validation_record = _run_single_process_validation_step(
                    global_step=global_step,
                    bundle=bundle,
                    artifacts=artifacts,
                    loaded_modules=loaded_modules,
                    validation_cursor=validation_cursor,
                    expected_stage=expected_stage,
                )
                validation_records.append(validation_record)
            loss_values = [
                float(item["loss_total"])
                for item in micro_losses
                if isinstance(item.get("loss_total"), (int, float))
            ]
            mean_loss = _mean(loss_values)
            loss_means = _mean_loss_scalar_fields(micro_losses)
            debug_summary = _summarize_training_debug(micro_debug_logs)
            progress_logger.log_event(
                "optimizer_step",
                {
                    "stage": bundle.get("stage"),
                    "run_id": bundle.get("run_id"),
                    "rank": rank,
                    "world_size": world_size,
                    "global_step": global_step,
                    "max_steps": max_steps,
                    "micro_steps_completed": total_micro_steps,
                    "loss_total": mean_loss,
                    **{
                        key: value
                        for key, value in loss_means.items()
                        if key != "loss_total"
                    },
                    "micro_losses": micro_losses,
                    "grad_after_sync_total_norm": _grad_total_norm(grad_after_sync),
                    "grad_after_clip_total_norm": _grad_total_norm(grad_after_clip),
                    "clipped_grad_norm": clipped_grad_norm,
                    "grad_norm": clipped_grad_norm,
                    "peak_memory_gb": peak_memory_gb,
                    "effective_global_batch_size": int(
                        ((bundle.get("batch") or {}).get("global_batch_size") or 0)
                    )
                    or None,
                    "scheduler_last_lr": scheduler_last_lr,
                    "learning_rates": scheduler_last_lr,
                    "debug_summary": debug_summary,
                    "checkpoint_record": checkpoint_record,
                    "validation_record": validation_record,
                },
                step=global_step,
                wandb_metrics=_training_step_wandb_metrics(
                    global_step=global_step,
                    total_micro_steps=total_micro_steps,
                    micro_losses=micro_losses,
                    grad_after_sync=grad_after_sync,
                    grad_after_clip=grad_after_clip,
                    clipped_grad_norm=clipped_grad_norm,
                    scheduler_last_lr=scheduler_last_lr,
                    checkpoint_record=checkpoint_record,
                    validation_record=validation_record,
                    debug_summary=debug_summary,
                    peak_memory_gb=peak_memory_gb,
                    effective_global_batch_size=int(
                        ((bundle.get("batch") or {}).get("global_batch_size") or 0)
                    )
                    or None,
                ),
                stdout=(
                    f"[clean-train] step={global_step}/{max_steps} "
                    f"loss={mean_loss if mean_loss is not None else 'nan'} "
                    f"micro_steps={total_micro_steps} "
                    f"checkpoint={checkpoint_record is not None} "
                    f"validation={validation_record is not None}"
                ),
            )
            _distributed_barrier(runtime_context)
        progress_logger.log_event(
            "finish",
            {
                "stage": bundle.get("stage"),
                "run_id": bundle.get("run_id"),
                "rank": rank,
                "world_size": world_size,
                "optimizer_steps_completed": len(step_records),
                "micro_steps_completed": total_micro_steps,
                "checkpoint_record_count": len(checkpoint_records),
                "validation_record_count": len(validation_records),
            },
            stdout=(
                f"[clean-train] finish stage={bundle.get('stage')} "
                f"steps={len(step_records)} checkpoints={len(checkpoint_records)}"
            ),
        )
    except BaseException as exc:
        progress_logger.log_event(
            "error",
            {
                "stage": bundle.get("stage"),
                "run_id": bundle.get("run_id"),
                "rank": rank,
                "world_size": world_size,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "optimizer_steps_completed": len(step_records),
                "micro_steps_completed": total_micro_steps,
            },
            stdout=f"[clean-train] error type={type(exc).__name__} message={exc}",
        )
        raise
    finally:
        progress_logger.close(
            summary={
                "optimizer_steps_completed": len(step_records),
                "micro_steps_completed": total_micro_steps,
                "checkpoint_record_count": len(checkpoint_records),
                "validation_record_count": len(validation_records),
            }
        )

    payload = {
        "schema_version": (
            "clean_distributed_training_rank_runtime_v1"
            if distributed
            else "clean_single_process_training_runtime_v1"
        ),
        "stage": bundle.get("stage"),
        "run_id": bundle.get("run_id"),
        "status": (
            "clean_distributed_training_rank_completed"
            if distributed
            else "clean_single_process_training_completed"
        ),
        "training_run_launched": True,
        "single_process": not distributed,
        "ddp_enabled": distributed,
        "legacy_distributed_training_semantics": _legacy_distributed_training_semantics_summary(
            modules=modules,
            expected_stage=expected_stage,
            distributed=distributed,
        ),
        "rank": rank,
        "local_rank": int(runtime_context.get("local_rank") or 0),
        "is_main": is_main,
        "in_training_validation_enabled": bool(eval_steps and is_main),
        "world_size": world_size,
        "max_steps": max_steps,
        "gradient_accumulation_steps": accumulation_steps,
        "optimizer_steps_completed": len(step_records),
        "micro_steps_completed": total_micro_steps,
        "checkpoint_save_steps": sorted(checkpoint_steps),
        "validation_steps": sorted(eval_steps),
        "validation_records": validation_records,
        "train_cursor": train_cursor.summary(),
        "validation_cursor": validation_cursor.summary() if validation_cursor else None,
        "checkpoint_records": checkpoint_records,
        "step_records": step_records,
        "progress_logging": progress_logger.summary(),
        "notes": [
            (
                "clean executor ran one rank of the distributed trainer loop"
                if distributed
                else "clean executor ran the single-process trainer loop"
            ),
            (
                "rank0 writes checkpoints and validation records"
                if distributed
                else "DDP/multi-process training is not part of this runtime"
            ),
        ],
    }
    path = (
        execution_dir / f"distributed_rank_{rank}_training_runtime.json"
        if distributed
        else execution_dir / "single_process_training_runtime.json"
    )
    _write_json(path, payload)
    return {"path": str(path), "payload": payload}


def _save_clean_training_checkpoint(
    *,
    execution_dir: Path,
    bundle: dict[str, Any],
    loaded_modules: dict[str, Any],
    modules: dict[str, Any],
    optimizer: Any,
    scheduler: Any,
    expected_stage: TrainingStage,
    global_step: int,
    optimizer_step: int,
    micro_step: int,
) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("clean checkpoint save requires torch") from exc

    checkpoint_path = execution_dir / f"checkpoint_step_{global_step}.pt"
    checkpoint = _checkpoint_probe_payload(
        bundle=bundle,
        loaded_modules=loaded_modules,
        modules=modules,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_stage=expected_stage,
        global_step=global_step,
        optimizer_step=optimizer_step,
        micro_step=micro_step,
    )
    torch.save(checkpoint, checkpoint_path)
    loaded = _torch_load_checkpoint_probe(torch, checkpoint_path)
    required_keys = _checkpoint_required_output_keys(expected_stage)
    missing_keys = sorted(key for key in required_keys if key not in loaded)
    state_checks = _checkpoint_state_checks(
        checkpoint=checkpoint,
        loaded=loaded,
        expected_stage=expected_stage,
    )
    step_checks = _checkpoint_publish_step_checks(
        loaded=loaded,
        expected_stage=expected_stage,
        global_step=global_step,
        optimizer_step=optimizer_step,
        micro_step=micro_step,
    )
    state_checks_ok = all(check.get("ok") is True for check in state_checks.values())
    step_checks_ok = all(check.get("ok") is True for check in step_checks.values())
    if missing_keys or not state_checks_ok or not step_checks_ok:
        raise ValueError(
            "clean checkpoint failed post-save validation: "
            f"missing={missing_keys}, state_ok={state_checks_ok}, step_ok={step_checks_ok}"
        )
    return {
        "path": str(checkpoint_path),
        "identity": file_identity(checkpoint_path).to_dict(),
        "global_step": loaded.get("global_step"),
        "optimizer_step": loaded.get("optimizer_step"),
        "micro_step": loaded.get("micro_step"),
        "required_output_keys": required_keys,
        "missing_required_keys": missing_keys,
        "state_checks_ok": state_checks_ok,
        "step_checks_ok": step_checks_ok,
    }


class _SingleProcessSampleCursor:
    def __init__(
        self,
        *,
        samples: list[Any],
        batch_size: int,
        stage: TrainingStage,
        dataset_role: str,
        dataset_path: str,
        target_focus_ratio: float | None = None,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
    ) -> None:
        if not samples:
            raise ValueError(f"{stage.value} {dataset_role} cursor has no samples")
        self.original_sample_count = len(samples)
        self.rank = max(0, int(rank))
        self.world_size = max(1, int(world_size))
        self.seed = int(seed)
        indexed_samples = list(enumerate(samples))
        if self.world_size > 1 and stage != TrainingStage.STAGE1:
            indexed_samples = [
                item for position, item in enumerate(indexed_samples)
                if position % self.world_size == self.rank
            ]
            if not indexed_samples:
                raise ValueError(
                    f"{stage.value} {dataset_role} cursor shard rank={self.rank} "
                    f"world_size={self.world_size} has no samples"
                )
        self.indexed_samples = indexed_samples
        self.batch_size = max(1, int(batch_size))
        self.stage = stage
        self.dataset_role = dataset_role
        self.dataset_path = dataset_path
        self.target_focus_ratio = (
            max(0.0, min(1.0, float(target_focus_ratio)))
            if target_focus_ratio is not None
            else None
        )
        self.cursor = 0
        self.same_image_epoch = 0
        self.same_image_epoch_batches: list[list[tuple[int, Any]]] = []
        self.same_image_batch_cursor = 0
        self.same_image_groups = (
            self._same_image_groups() if stage == TrainingStage.STAGE1 else []
        )
        self.focus_indices = self._stage2_focus_indices() if stage == TrainingStage.STAGE2 else []
        self.no_focus_indices = (
            self._stage2_no_focus_indices() if stage == TrainingStage.STAGE2 else []
        )
        self.focus_cursor = 0
        self.no_focus_cursor = 0
        self.batch_index = 0
        self.focus_emitted = 0
        if self.same_image_groups:
            self.mode = "same_image_legacy_shuffle"
        elif self.focus_indices and self.no_focus_indices and self.target_focus_ratio is not None:
            self.mode = "target_focus_ratio_cycle"
        else:
            self.mode = "sequential_cycle"

    def next_batch(self) -> dict[str, Any]:
        if self.same_image_groups:
            if self.same_image_batch_cursor >= len(self.same_image_epoch_batches):
                self._refresh_same_image_epoch_batches()
            selected = self.same_image_epoch_batches[self.same_image_batch_cursor]
            self.same_image_batch_cursor += 1
        elif self.mode == "target_focus_ratio_cycle":
            selected = self._next_stage2_ratio_batch()
        else:
            selected = [
                self._indexed_sample((self.cursor + item_index) % len(self.indexed_samples))
                for item_index in range(self.batch_size)
            ]
            self.cursor = (self.cursor + self.batch_size) % len(self.indexed_samples)
        return {
            "samples": [sample for _, sample in selected],
            "sample_trace": [
                _sample_trace_entry(
                    index=index,
                    sample=sample,
                    dataset_role=self.dataset_role,
                )
                for index, sample in selected
            ],
        }

    def summary(self) -> dict[str, Any]:
        return {
            "stage": str(self.stage),
            "dataset_role": self.dataset_role,
            "dataset_path": self.dataset_path,
            "sample_count": len(self.indexed_samples),
            "original_sample_count": self.original_sample_count,
            "batch_size": self.batch_size,
            "rank": self.rank,
            "world_size": self.world_size,
            "mode": self.mode,
            "same_image_group_count": len(self.same_image_groups),
            "same_image_drop_incomplete": self.stage == TrainingStage.STAGE1,
            "same_image_min_batch_size": self._same_image_min_batch_size()
            if self.stage == TrainingStage.STAGE1
            else None,
            "same_image_max_batch_size": self.batch_size
            if self.stage == TrainingStage.STAGE1
            else None,
            "same_image_group_owner": "sha1(image_key)%world_size"
            if self.stage == TrainingStage.STAGE1
            else None,
            "same_image_shuffle": "legacy_group_and_group_member_shuffle"
            if self.stage == TrainingStage.STAGE1
            else None,
            "same_image_seed": self.seed if self.stage == TrainingStage.STAGE1 else None,
            "same_image_epoch": self.same_image_epoch
            if self.stage == TrainingStage.STAGE1
            else None,
            "target_focus_ratio": self.target_focus_ratio,
            "focus_sample_count": len(self.focus_indices),
            "no_focus_sample_count": len(self.no_focus_indices),
        }

    def _indexed_sample(self, index: int) -> tuple[int, Any]:
        return self.indexed_samples[index]

    def _same_image_min_batch_size(self) -> int:
        if self.stage == TrainingStage.STAGE1 and self.batch_size == 5:
            return 4
        return self.batch_size if self.batch_size > 1 else 1

    def _same_image_batch_sizes_for_group(self, group_size: int) -> list[int]:
        min_size = self._same_image_min_batch_size()
        max_size = self.batch_size
        exact_sizes: list[list[int] | None] = [None] * (max(0, group_size) + 1)
        exact_sizes[0] = []
        for used in range(1, group_size + 1):
            for size in range(max_size, min_size - 1, -1):
                if used < size:
                    continue
                prefix = exact_sizes[used - size]
                if prefix is None:
                    continue
                exact_sizes[used] = [*prefix, size]
                break
        for used in range(group_size, min_size - 1, -1):
            sizes = exact_sizes[used]
            if sizes is not None:
                return sizes
        return []

    def _same_image_groups(self) -> list[list[tuple[int, Any]]]:
        groups_by_key: dict[str, list[tuple[int, Any]]] = {}
        order: list[str] = []
        for index, sample in self.indexed_samples:
            key = str(getattr(sample, "image_id", None) or getattr(sample, "image", ""))
            if self.world_size > 1:
                owner = int(sha1(key.encode("utf-8")).hexdigest(), 16) % self.world_size
                if owner != self.rank:
                    continue
            if key not in groups_by_key:
                order.append(key)
                groups_by_key[key] = []
            groups_by_key[key].append((index, sample))
        min_size = self._same_image_min_batch_size()
        return [groups_by_key[key] for key in order if len(groups_by_key[key]) >= min_size]

    def _refresh_same_image_epoch_batches(self) -> None:
        rng = random.Random(self.seed + self.same_image_epoch)
        groups = [list(group) for group in self.same_image_groups]
        rng.shuffle(groups)
        batches: list[list[tuple[int, Any]]] = []
        for group in groups:
            rng.shuffle(group)
            start = 0
            for take in self._same_image_batch_sizes_for_group(len(group)):
                batch = group[start : start + take]
                batches.append(batch)
                start += take
        if not batches:
            raise RuntimeError("same-image Stage1 cursor produced no complete batches")
        self.same_image_epoch += 1
        self.same_image_epoch_batches = batches
        self.same_image_batch_cursor = 0

    def _stage2_focus_indices(self) -> list[int]:
        return [
            position
            for position, (_, sample) in enumerate(self.indexed_samples)
            if bool(getattr(sample, "need_focus", False))
        ]

    def _stage2_no_focus_indices(self) -> list[int]:
        return [
            position
            for position, (_, sample) in enumerate(self.indexed_samples)
            if not bool(getattr(sample, "need_focus", False))
        ]

    def _next_stage2_ratio_batch(self) -> list[tuple[int, Any]]:
        self.batch_index += 1
        desired_focus_total = round(
            self.batch_index * self.batch_size * float(self.target_focus_ratio or 0.0)
        )
        focus_count = desired_focus_total - self.focus_emitted
        focus_count = max(0, min(self.batch_size, focus_count))
        no_focus_count = self.batch_size - focus_count
        selected: list[tuple[int, Any]] = []
        for _ in range(focus_count):
            index = self.focus_indices[self.focus_cursor % len(self.focus_indices)]
            selected.append(self._indexed_sample(index))
            self.focus_cursor += 1
            self.focus_emitted += 1
        for _ in range(no_focus_count):
            index = self.no_focus_indices[self.no_focus_cursor % len(self.no_focus_indices)]
            selected.append(self._indexed_sample(index))
            self.no_focus_cursor += 1
        return selected


def _build_single_process_sample_cursor(
    *,
    bundle: dict[str, Any],
    expected_stage: TrainingStage,
    dataset_role: str,
    rank: int = 0,
    world_size: int = 1,
) -> _SingleProcessSampleCursor:
    dataset = bundle.get("dataset") or {}
    if dataset_role == "train":
        identity = dataset.get("train_file") or {}
    elif dataset_role == "val":
        identity = dataset.get("val_file") or {}
    else:
        raise ValueError(f"unknown dataset_role: {dataset_role}")
    path = str((identity or {}).get("path") or "")
    if not path:
        raise ValueError(f"{expected_stage.value} {dataset_role} cursor requires a dataset path")
    try:
        if expected_stage == TrainingStage.STAGE1:
            from revisit_vlm.tgvf_v3_stage1 import TGVFv3Stage1Dataset

            dataset_obj = TGVFv3Stage1Dataset(path, focus_only=True)
        else:
            from revisit_vlm.tgvf_v3_stage2 import TGVFv3Stage2Dataset

            dataset_obj = TGVFv3Stage2Dataset(path)
    except Exception as exc:
        raise RuntimeError(
            f"{expected_stage.value} {dataset_role} cursor dataset load failed"
        ) from exc
    return _SingleProcessSampleCursor(
        samples=list(dataset_obj.samples),
        batch_size=int(((bundle.get("batch") or {}).get("micro_batch_size")) or 1),
        stage=expected_stage,
        dataset_role=dataset_role,
        dataset_path=path,
        rank=rank,
        world_size=world_size,
        seed=int((bundle.get("training") or {}).get("seed") or 0),
        target_focus_ratio=(
            (bundle.get("training") or {}).get("target_focus_ratio")
            if expected_stage == TrainingStage.STAGE2 and dataset_role == "train"
            else None
        ),
    )


def _sample_trace_entry(
    *,
    index: int,
    sample: Any,
    dataset_role: str,
) -> dict[str, Any]:
    image = getattr(sample, "image", None)
    question = getattr(sample, "question", None)
    return {
        "dataset_role": dataset_role,
        "sample_index": int(index),
        "image_sha256": _text_sha256(image),
        "question_sha256": _text_sha256(question),
        "need_focus": getattr(sample, "need_focus", None),
        "trajectory_type": getattr(sample, "trajectory_type", None),
        "image_id": getattr(sample, "image_id", None),
    }


def _run_single_process_validation_step(
    *,
    global_step: int,
    bundle: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    loaded_modules: dict[str, Any],
    validation_cursor: _SingleProcessSampleCursor | None,
    expected_stage: TrainingStage,
) -> dict[str, Any]:
    if expected_stage != TrainingStage.STAGE2:
        raise ValueError("in-training validation is currently defined for Stage2 only")
    if validation_cursor is None:
        raise ValueError("in-training validation requires a validation cursor")
    batch_record = validation_cursor.next_batch()
    modules = dict(loaded_modules.get("modules") or {})
    previous_training_states = _set_module_training_mode(modules, training=False)
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("single-process validation requires torch") from exc
    try:
        with torch.no_grad():
            result = _run_training_step_probe_for_stage(
                bundle=bundle,
                artifacts=artifacts,
                loaded_modules=loaded_modules,
                expected_stage=expected_stage,
                samples=batch_record["samples"],
            )
    finally:
        _restore_module_training_mode(modules, previous_training_states)
    debug = dict(result.get("debug") or {})
    return {
        "global_step": int(global_step),
        "dataset_path": validation_cursor.dataset_path,
        "loss_total": result.get("loss_total"),
        "loss_focus": result.get("loss_focus"),
        "loss_no_focus": result.get("loss_no_focus"),
        "loss_visual_token_manifold": result.get("loss_visual_token_manifold"),
        "loss_visual_token_norm": result.get("loss_visual_token_norm"),
        "sample_count": result.get("sample_count"),
        "sample_trace": batch_record["sample_trace"],
        "focus_count": debug.get("focus_count"),
        "no_focus_count": debug.get("no_focus_count"),
        "forward_completed": bool(result.get("forward_completed")),
        "backward_called": False,
        "optimizer_step_called": False,
    }


def _set_module_training_mode(
    modules: dict[str, Any],
    *,
    training: bool,
) -> dict[str, bool]:
    previous: dict[str, bool] = {}
    for name, module in modules.items():
        if hasattr(module, "training"):
            previous[name] = bool(module.training)
        if training and hasattr(module, "train"):
            module.train()
        elif not training and hasattr(module, "eval"):
            module.eval()
    return previous


def _restore_module_training_mode(
    modules: dict[str, Any],
    previous: dict[str, bool],
) -> None:
    for name, was_training in previous.items():
        module = modules.get(name)
        if module is None or not hasattr(module, "train"):
            continue
        module.train(was_training)


def _clean_training_launch_result(
    *,
    bundle_path: Path,
    bundle: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    trainable_parameters: dict[str, Any],
    optimizer_runtime: dict[str, Any],
    cadence_runtime: dict[str, Any],
    training_runtime: dict[str, Any],
    expected_stage: TrainingStage,
) -> dict[str, Any]:
    runtime_payload = training_runtime["payload"]
    ddp_enabled = bool(runtime_payload.get("ddp_enabled"))
    checkpoint_records = list(runtime_payload.get("checkpoint_records") or [])
    final_checkpoint = checkpoint_records[-1] if checkpoint_records else None
    unsupported_runtime_features: list[str] = []
    deepstack_plan = artifacts.get("deepstack_training_plan") or {}
    if (
        expected_stage == TrainingStage.STAGE2
        and not bool(deepstack_plan.get("execution_supported", True))
    ):
        unsupported_runtime_features.append("stage2_deepstack_training")
    status = (
        "clean_distributed_training_completed"
        if ddp_enabled
        else "clean_single_process_training_completed"
    )
    return {
        "schema_version": "clean_training_launch_result_v1",
        "stage": str(expected_stage),
        "run_id": bundle.get("run_id"),
        "status": status,
        "will_launch_training": True,
        "training_runtime_ported": True,
        "single_process": not ddp_enabled,
        "ddp_enabled": ddp_enabled,
        "rank": runtime_payload.get("rank"),
        "world_size": runtime_payload.get("world_size"),
        "bundle_path": str(bundle_path),
        "bundle_identity": file_identity(bundle_path).to_dict(),
        "plan_path": bundle.get("plan_path"),
        "plan_identity": bundle.get("plan_identity"),
        "plan_artifact_identities": bundle.get("plan_artifact_identities"),
        "git_commit": bundle.get("git_commit"),
        "dirty_worktree": bundle.get("dirty_worktree"),
        "dataset_identity": _launch_dataset_identity(artifacts),
        "input_checkpoint_contract": _launch_input_checkpoint_contract(artifacts),
        "runtime_artifact_identities": _launch_runtime_artifact_identities(
            bundle=bundle,
            trainable_parameters=trainable_parameters,
            optimizer_runtime=optimizer_runtime,
            cadence_runtime=cadence_runtime,
            training_runtime=training_runtime,
        ),
        "trainable_parameters": trainable_parameters["path"],
        "optimizer_runtime": optimizer_runtime["path"],
        "training_cadence_runtime": cadence_runtime["path"],
        "training_runtime": training_runtime["path"],
        "single_process_training_runtime": (
            training_runtime["path"] if not ddp_enabled else None
        ),
        "distributed_rank_training_runtime": (
            training_runtime["path"] if ddp_enabled else None
        ),
        "optimizer_steps_completed": runtime_payload.get("optimizer_steps_completed"),
        "micro_steps_completed": runtime_payload.get("micro_steps_completed"),
        "in_training_validation_enabled": runtime_payload.get(
            "in_training_validation_enabled"
        ),
        "validation_steps": list(runtime_payload.get("validation_steps") or []),
        "validation_record_count": len(runtime_payload.get("validation_records") or []),
        "checkpoint_records": checkpoint_records,
        "final_checkpoint": (final_checkpoint or {}).get("path"),
        "final_checkpoint_identity": (final_checkpoint or {}).get("identity"),
        "unsupported_runtime_features": unsupported_runtime_features,
    }


def _clean_training_launch_status(result: dict[str, Any]) -> dict[str, Any]:
    checkpoint_records = list(result.get("checkpoint_records") or [])
    return {
        "schema_version": "clean_training_launch_status_v1",
        "stage": result.get("stage"),
        "run_id": result.get("run_id"),
        "status": result.get("status"),
        "will_launch_training": result.get("will_launch_training"),
        "training_runtime_ported": result.get("training_runtime_ported"),
        "plan_identity": result.get("plan_identity"),
        "plan_artifact_identities": result.get("plan_artifact_identities"),
        "bundle_identity": result.get("bundle_identity"),
        "dataset_identity": result.get("dataset_identity"),
        "input_checkpoint_contract": result.get("input_checkpoint_contract"),
        "single_process": result.get("single_process"),
        "ddp_enabled": result.get("ddp_enabled"),
        "rank": result.get("rank"),
        "world_size": result.get("world_size"),
        "in_training_validation_enabled": result.get("in_training_validation_enabled"),
        "validation_record_count": result.get("validation_record_count"),
        "optimizer_steps_completed": result.get("optimizer_steps_completed"),
        "micro_steps_completed": result.get("micro_steps_completed"),
        "checkpoint_count": len(checkpoint_records),
        "final_checkpoint": checkpoint_records[-1]["path"] if checkpoint_records else None,
        "final_checkpoint_identity": (
            checkpoint_records[-1].get("identity") if checkpoint_records else None
        ),
        "runtime_artifact_identity_count": len(
            result.get("runtime_artifact_identities") or {}
        ),
        "unsupported_runtime_features": list(
            result.get("unsupported_runtime_features") or []
        ),
    }


def _launch_dataset_identity(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    dataset = artifacts.get("dataset_runtime_identity") or {}
    first_batch = artifacts.get("first_batch_identity") or {}
    return {
        "train_file": dataset.get("train_file"),
        "val_file": dataset.get("val_file"),
        "global_batch_size": dataset.get("global_batch_size"),
        "first_batch": {
            "materialized_batch_size": first_batch.get("materialized_batch_size"),
            "batch_sha256": first_batch.get("batch_sha256"),
            "row_digests": list(first_batch.get("row_digests") or []),
        },
    }


def _launch_input_checkpoint_contract(
    artifacts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    checkpoint = artifacts.get("checkpoint_contract") or {}
    return {
        "status": checkpoint.get("status"),
        "input_checkpoint_required": checkpoint.get("input_checkpoint_required"),
        "checkpoint_identity": checkpoint.get("checkpoint_identity"),
        "global_step": checkpoint.get("global_step"),
        "protocol_c_token_rows_required": checkpoint.get(
            "protocol_c_token_rows_required"
        ),
    }


def _launch_runtime_artifact_identities(
    *,
    bundle: dict[str, Any],
    trainable_parameters: dict[str, Any],
    optimizer_runtime: dict[str, Any],
    cadence_runtime: dict[str, Any],
    training_runtime: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    artifact_paths = dict(bundle.get("runtime_artifacts") or {})
    artifact_paths.update(
        {
            "trainable_parameters": trainable_parameters["path"],
            "optimizer_runtime": optimizer_runtime["path"],
            "training_cadence_runtime": cadence_runtime["path"],
            "training_runtime": training_runtime["path"],
        }
    )
    return {
        name: file_identity(path).to_dict()
        for name, path in sorted(artifact_paths.items())
    }


def _cadence_steps(*, max_steps: int, every: int) -> list[int]:
    steps = set(range(every, max_steps + 1, every))
    steps.add(max_steps)
    return sorted(steps)


def _eval_disabled_reason(
    *,
    expected_stage: TrainingStage,
    val_file_available: bool,
) -> str:
    if expected_stage != TrainingStage.STAGE2:
        return "stage_has_no_eval_cadence"
    if not val_file_available:
        return "missing_val_file"
    return "not_disabled"


def _load_checkpoint_model_states_for_resume(
    *,
    checkpoint: dict[str, Any],
    bundle: dict[str, Any],
    loaded_modules: dict[str, Any],
    modules: dict[str, Any],
    expected_stage: TrainingStage,
) -> dict[str, Any]:
    tgvf_module = _unwrap_distributed_data_parallel_module(modules.get("tgvf"))
    if tgvf_module is None or not hasattr(tgvf_module, "load_state_dict"):
        raise ValueError("checkpoint-resume audit requires a tgvf module")
    tgvf_module.load_state_dict(checkpoint["tgvf_module"], strict=True)
    payload = {
        "model_state_loaded": True,
        "tgvf_module_loaded": True,
    }
    if expected_stage == TrainingStage.STAGE1:
        payload["protocol_c_token_rows"] = _restore_protocol_rows_for_resume(
            checkpoint=checkpoint,
            bundle=bundle,
            loaded_modules=loaded_modules,
            modules=modules,
        )
        return payload

    qwen_lora = modules.get("qwen_lora")
    if qwen_lora is None:
        raise ValueError("checkpoint-resume audit requires qwen_lora module")
    payload["qwen_lora"] = _load_qwen_lora_resume_state(
        module=qwen_lora,
        state=checkpoint.get("qwen_lora"),
    )
    return payload


def _load_qwen_lora_resume_state(*, module: Any, state: Any) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint-resume audit requires qwen_lora state mapping")
    try:
        from peft import set_peft_model_state_dict
    except Exception:
        set_peft_model_state_dict = None
    if set_peft_model_state_dict is not None:
        try:
            result = set_peft_model_state_dict(module, dict(state))
            return {
                "loaded": True,
                "loader": "peft.set_peft_model_state_dict",
                "result_type": type(result).__name__,
            }
        except Exception:
            pass
    incompatible = module.load_state_dict(dict(state), strict=False)
    return {
        "loaded": True,
        "loader": "module.load_state_dict(strict=False)",
        "missing_keys": list(getattr(incompatible, "missing_keys", []) or []),
        "unexpected_keys": list(getattr(incompatible, "unexpected_keys", []) or []),
    }


def _restore_protocol_rows_for_resume(
    *,
    checkpoint: dict[str, Any],
    bundle: dict[str, Any],
    loaded_modules: dict[str, Any],
    modules: dict[str, Any],
) -> dict[str, Any]:
    required = _protocol_c_rows_required(bundle)
    available = checkpoint.get("protocol_c_token_rows") is not None
    info = {"required": required, "available_in_checkpoint": available, "ok": True}
    if not required or not available:
        return info
    try:
        from scripts.train_tgvf_v3_stage1 import _restore_protocol_c_token_rows
    except Exception as exc:
        return {**info, "ok": False, "loaded": False, "reason": f"import_failed:{exc}"}
    processor = loaded_modules.get("processor")
    tokenizer = loaded_modules.get("tokenizer") or getattr(processor, "tokenizer", None)
    model = modules.get("qwen") or modules.get("qwen_lora")
    if tokenizer is None or model is None:
        return {**info, "ok": False, "loaded": False, "reason": "missing_tokenizer_or_model"}
    restored = _restore_protocol_c_token_rows(
        model=model,
        tokenizer=tokenizer,
        protocol=str(bundle.get("protocol")),
        checkpoint=checkpoint,
    )
    return {**info, **restored, "ok": restored.get("loaded") is True}


def _resume_state_checks(
    *,
    checkpoint: dict[str, Any],
    modules: dict[str, Any],
    expected_stage: TrainingStage,
) -> dict[str, Any]:
    checks = {
        "tgvf_module": _state_dict_parity_check(
            checkpoint.get("tgvf_module"),
            _unwrap_distributed_data_parallel_module(modules.get("tgvf")).state_dict()
            if modules.get("tgvf") is not None
            else None,
        )
    }
    if expected_stage == TrainingStage.STAGE2:
        qwen_lora = modules.get("qwen_lora")
        checks["qwen_lora"] = _state_dict_parity_check(
            checkpoint.get("qwen_lora"),
            _qwen_lora_state_dict(qwen_lora) if qwen_lora is not None else None,
        )
    return checks


def _run_training_step_probe_for_stage(
    *,
    bundle: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    loaded_modules: dict[str, Any],
    expected_stage: TrainingStage,
    samples: list[Any] | None = None,
) -> dict[str, Any]:
    kwargs = {
        "bundle": bundle,
        "artifacts": artifacts,
        "loaded_modules": loaded_modules,
    }
    if samples is not None:
        kwargs["samples"] = samples
    if expected_stage == TrainingStage.STAGE1:
        return _run_stage1_training_step_probe(**kwargs)
    return _run_stage2_training_step_probe(**kwargs)


def _max_grad_norm_from_bundle(
    bundle: dict[str, Any],
    optimizer_runtime: dict[str, Any],
) -> float | None:
    optimizer = optimizer_runtime["payload"].get("optimizer") or {}
    contract_norm = optimizer.get("max_grad_norm")
    if contract_norm is None:
        contract_norm = ((bundle.get("optimizer") or {}).get("max_grad_norm"))
    if contract_norm is None:
        return None
    value = float(contract_norm)
    return value if value > 0 else None


def _optimizer_parameters(optimizer: Any) -> list[Any]:
    parameters: list[Any] = []
    for group in optimizer.param_groups:
        parameters.extend(list(group.get("params") or []))
    return parameters


def _average_optimizer_gradients(
    optimizer: Any,
    *,
    world_size: int,
    skip_parameter_ids: set[int] | None = None,
) -> None:
    if world_size <= 1:
        return
    try:
        import torch.distributed as dist
    except Exception as exc:
        raise RuntimeError("distributed gradient averaging requires torch.distributed") from exc
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("distributed gradient averaging requires an initialized process group")
    skip_parameter_ids = skip_parameter_ids or set()
    for parameter in _optimizer_parameters(optimizer):
        if id(parameter) in skip_parameter_ids:
            continue
        grad = getattr(parameter, "grad", None)
        if grad is None:
            continue
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        grad.div_(float(world_size))


def _apply_legacy_distributed_training_semantics(
    *,
    loaded_modules: dict[str, Any],
    expected_stage: TrainingStage,
    runtime_context: dict[str, Any],
) -> None:
    """Match historical Stage1 distributed semantics before optimizer creation."""
    if expected_stage != TrainingStage.STAGE1 or not runtime_context.get("distributed"):
        return
    modules = dict(loaded_modules.get("modules") or {})
    tgvf = modules.get("tgvf")
    if tgvf is None or _is_distributed_data_parallel_module(tgvf):
        return
    try:
        from torch.nn.parallel import DistributedDataParallel
    except Exception as exc:
        raise RuntimeError("distributed Stage1 launch requires DistributedDataParallel") from exc
    if hasattr(tgvf, "train"):
        tgvf.train()
    local_rank = int(runtime_context.get("local_rank") or 0)
    device = str(runtime_context.get("device") or "")
    wrapped = DistributedDataParallel(
        tgvf,
        device_ids=[local_rank] if device.startswith("cuda") else None,
        output_device=local_rank if device.startswith("cuda") else None,
        find_unused_parameters=False,
    )
    modules["tgvf"] = wrapped
    loaded_modules["modules"] = modules
    loader = dict(loaded_modules.get("loader") or {})
    loader["distributed_training"] = {
        "stage1_tgvf_wrapped_with_ddp": True,
        "ddp_broadcast_initial_parameters": True,
        "gradient_sync": "legacy_ddp_tgvf_plus_manual_protocol_rows",
        "gradient_accumulation": "ddp_no_sync_until_final_micro_step",
    }
    loaded_modules["loader"] = loader


def _training_micro_step_sync_context(
    *,
    modules: dict[str, Any],
    expected_stage: TrainingStage,
    micro_index: int,
    accumulation_steps: int,
) -> Any:
    if (
        expected_stage == TrainingStage.STAGE1
        and micro_index < accumulation_steps - 1
        and _is_distributed_data_parallel_module(modules.get("tgvf"))
    ):
        return modules["tgvf"].no_sync()
    return nullcontext()


def _legacy_distributed_training_semantics_summary(
    *,
    modules: dict[str, Any],
    expected_stage: TrainingStage,
    distributed: bool,
) -> dict[str, Any]:
    tgvf_ddp = _is_distributed_data_parallel_module(modules.get("tgvf"))
    return {
        "stage": str(expected_stage),
        "distributed": bool(distributed),
        "stage1_tgvf_wrapped_with_ddp": bool(
            expected_stage == TrainingStage.STAGE1 and tgvf_ddp
        ),
        "manual_gradient_average_skips_ddp_parameters": bool(tgvf_ddp),
        "non_ddp_trainables_manually_averaged": bool(distributed),
    }


def _distributed_data_parallel_parameter_ids(modules: dict[str, Any]) -> set[int]:
    parameter_ids: set[int] = set()
    for module in modules.values():
        if not _is_distributed_data_parallel_module(module) or not hasattr(module, "parameters"):
            continue
        parameter_ids.update(id(parameter) for parameter in module.parameters())
    return parameter_ids


def _is_distributed_data_parallel_module(module: Any) -> bool:
    try:
        from torch.nn.parallel import DistributedDataParallel
    except Exception:
        return False
    return isinstance(module, DistributedDataParallel)


def _unwrap_distributed_data_parallel_module(module: Any) -> Any:
    if _is_distributed_data_parallel_module(module):
        return module.module
    return module


def _optimizer_grad_summary(optimizer: Any) -> dict[str, Any]:
    parameters = _optimizer_parameters(optimizer)
    tensors_with_grad = 0
    grad_numel = 0
    squared_norm = 0.0
    max_abs = 0.0
    nonfinite_tensors = 0
    for parameter in parameters:
        grad = getattr(parameter, "grad", None)
        if grad is None:
            continue
        tensors_with_grad += 1
        grad_numel += _numel(grad)
        try:
            detached = grad.detach()
            finite = bool(detached.isfinite().all().item())
            if not finite:
                nonfinite_tensors += 1
            squared_norm += float(detached.float().norm(2).item()) ** 2
            max_abs = max(max_abs, float(detached.float().abs().max().item()))
        except Exception:
            nonfinite_tensors += 1
    return {
        "parameter_tensor_count": len(parameters),
        "tensors_with_grad": tensors_with_grad,
        "grad_numel": grad_numel,
        "total_norm": math.sqrt(squared_norm),
        "max_abs": max_abs,
        "nonfinite_tensors": nonfinite_tensors,
    }


def _stage1_training_step_flags(
    *,
    bundle: dict[str, Any],
    result: dict[str, Any],
    debug: dict[str, Any],
) -> dict[str, Any]:
    loss = bundle.get("loss") or {}
    expected_mode = str((bundle.get("training") or {}).get("same_image_negative_mode"))
    observed_loss_weights = dict(debug.get("loss_weights") or {})
    readout_context_applied = (
        debug.get("readout_append_mode") == "qwen3_visual_special_tokens_embedding_replace"
    )
    position_ids_applied = (
        debug.get("position_ids_source") == "qwen3_native_source_grid_full_trajectory"
    )
    matrix_ce_and_manifold_applied = bool(
        result.get("forward_completed")
        and debug.get("same_image_negative_mode") == expected_mode
        and observed_loss_weights.get("same_image_negative") == loss.get("same_image_negative")
        and observed_loss_weights.get("visual_token_manifold")
        == loss.get("visual_token_manifold")
        and observed_loss_weights.get("visual_token_norm") == loss.get("visual_token_norm")
        and debug.get("visual_token_manifold_active") is True
        and debug.get("visual_token_norm_active") is True
    )
    return {
        "stage1_readout_context_applied": readout_context_applied,
        "stage1_position_ids_applied": position_ids_applied,
        "stage1_matrix_ce_and_manifold_losses_applied": matrix_ce_and_manifold_applied,
        "expected_stage1_loss": loss,
        "observed_stage1_loss_weights": observed_loss_weights,
        "expected_same_image_negative_mode": expected_mode,
        "observed_same_image_negative_mode": debug.get("same_image_negative_mode"),
        "observed_readout_context": {
            "readout_append_mode": debug.get("readout_append_mode"),
            "attention_mask_mode": debug.get("attention_mask_mode"),
            "position_ids_source": debug.get("position_ids_source"),
            "image_keys_blocked_for_tgvf_evidence_answer": debug.get(
                "image_keys_blocked_for_tgvf_evidence_answer"
            ),
        },
    }


def _stage2_training_step_flags(
    *,
    bundle: dict[str, Any],
    result: dict[str, Any],
    debug: dict[str, Any],
) -> dict[str, Any]:
    expected_weights = dict((bundle.get("loss") or {}).get("weighted_span_loss") or {})
    mask_policy = bundle.get("mask_policy") or {}
    fast_path_used = debug.get("fast_batched_stage2") is True
    weighted_span_loss_applied = bool(
        result.get("forward_completed")
        and expected_weights
        and debug.get("focus_loss_token_weight") is not None
        and debug.get("no_focus_loss_token_weight") is not None
    )
    mask_scope_matches = (
        bool(mask_policy.get("mask_original_image_after_tgvf"))
        == bool(result.get("mask_original_image_after_tgvf"))
        and float(mask_policy.get("mask_original_image_after_tgvf_prob") or 0.0)
        == float(debug.get("mask_original_image_after_tgvf_prob") or 0.0)
        and str(mask_policy.get("mask_original_image_after_tgvf_scope"))
        == str(debug.get("mask_original_image_after_tgvf_scope"))
    )
    return {
        "fast_batched_stage2_used": fast_path_used,
        "weighted_span_loss_applied": weighted_span_loss_applied,
        "mask_scope_applied": mask_scope_matches,
        "expected_weighted_span_loss": expected_weights,
        "observed_loss_token_weights": {
            "focus": debug.get("focus_loss_token_weight"),
            "no_focus": debug.get("no_focus_loss_token_weight"),
        },
        "mask_policy": mask_policy,
        "observed_mask_policy": {
            "mask_original_image_after_tgvf": result.get("mask_original_image_after_tgvf"),
            "mask_original_image_after_tgvf_prob": debug.get(
                "mask_original_image_after_tgvf_prob"
            ),
            "mask_original_image_after_tgvf_scope": debug.get(
                "mask_original_image_after_tgvf_scope"
            ),
            "focus_sample_mask_active_rate": debug.get("focus_sample_mask_active_rate"),
            "no_focus_mask_active_rate": debug.get("no_focus_mask_active_rate"),
        },
    }


def _run_stage1_training_step_probe(
    *,
    bundle: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    loaded_modules: dict[str, Any],
    samples: list[Any] | None = None,
) -> dict[str, Any]:
    try:
        import torch

        from revisit_vlm.tgvf_training import LossWeights
        from revisit_vlm.tgvf_v3_stage1 import TGVFv3Stage1Dataset, v3_stage1_training_step
    except Exception as exc:
        raise RuntimeError("Stage1 training-step audit dependencies are unavailable") from exc
    modules = dict(loaded_modules.get("modules") or {})
    qwen_model = modules.get("qwen")
    if qwen_model is None:
        raise ValueError("Stage1 training-step audit requires qwen module")
    processor = loaded_modules.get("processor")
    if processor is None:
        raise ValueError("Stage1 training-step audit requires processor")
    foveal_module = modules.get("tgvf")
    if foveal_module is None:
        raise ValueError("Stage1 training-step audit requires tgvf module")
    if samples is None:
        train_file = (((bundle.get("dataset") or {}).get("train_file") or {}).get("path"))
        if not train_file:
            raise ValueError("Stage1 training-step audit requires dataset.train_file.path")
        dataset = TGVFv3Stage1Dataset(train_file, focus_only=True)
        samples = _select_stage1_step_probe_samples(
            dataset.samples,
            requested_count=max(
                1,
                int(((bundle.get("batch") or {}).get("micro_batch_size")) or 1),
            ),
        )
    if not samples:
        raise ValueError("Stage1 training-step audit found no usable focus samples")
    loss = bundle.get("loss") or {}
    training = bundle.get("training") or {}
    output = v3_stage1_training_step(
        qwen_model=qwen_model,
        processor=processor,
        foveal_module=foveal_module,
        reencode_qwen_model=None,
        samples=samples,
        loss_weights=LossWeights(
            gen=float(loss.get("gen") or 0.0),
            visual_token_manifold=float(loss.get("visual_token_manifold") or 0.0),
            visual_token_norm=float(loss.get("visual_token_norm") or 0.0),
            same_image_negative=float(loss.get("same_image_negative") or 0.0),
            contrastive_alignment=float(loss.get("contrastive_alignment") or 0.0),
        ),
        device=_parameter_audit_device(torch),
        hidden_state_index=int(training.get("capture_layer") or -1),
        same_image_negative_margin=float(
            training.get("same_image_negative_margin") or 1.0
        ),
        same_image_negative_mode=str(training.get("same_image_negative_mode") or "matrix_ce"),
        mask_original_image_after_tgvf=bool(
            training.get("mask_original_image_after_tgvf")
        ),
        position_mode=str(training.get("fvt_position_mode") or "native_source_grid"),
        max_image_resolution=training.get("max_image_resolution"),
        capture_mode=str(training.get("capture_mode") or "teacher_forced"),
        protocol=str(bundle.get("protocol")),
        focus_action_im_end=bool(training.get("focus_action_im_end")),
        readout_batch_size=int(training.get("readout_batch_size") or 4),
    )
    return {
        "forward_completed": True,
        "sample_count": len(samples),
        "loss_total": _scalar_float(output.loss_total),
        "loss_tensor": output.loss_total,
        "loss_gen": _scalar_float(output.loss_gen),
        "loss_visual_token_manifold": _scalar_float(output.loss_visual_token_manifold),
        "loss_visual_token_norm": _scalar_float(output.loss_visual_token_norm),
        "loss_same_image_negative": _scalar_float(output.loss_same_image_negative),
        "debug": output.debug,
    }


def _run_stage2_training_step_probe(
    *,
    bundle: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    loaded_modules: dict[str, Any],
    samples: list[Any] | None = None,
) -> dict[str, Any]:
    try:
        import torch

        from revisit_vlm.tgvf_v3_stage2 import Stage2LossWeights, TGVFv3Stage2Dataset
        from revisit_vlm.tgvf_v3_stage2_fast import v3_stage2_batched_training_step
    except Exception as exc:
        raise RuntimeError("Stage2 training-step audit dependencies are unavailable") from exc
    modules = dict(loaded_modules.get("modules") or {})
    qwen_forward_model = modules.get("qwen_lora")
    if qwen_forward_model is None:
        raise ValueError("Stage2 training-step audit requires qwen_lora module")
    qwen_model = loaded_modules.get("qwen_model")
    if qwen_model is None and hasattr(qwen_forward_model, "get_base_model"):
        qwen_model = qwen_forward_model.get_base_model()
    if qwen_model is None:
        qwen_model = qwen_forward_model
    processor = loaded_modules.get("processor")
    if processor is None:
        raise ValueError("Stage2 training-step audit requires processor")
    foveal_module = modules.get("tgvf")
    if foveal_module is None:
        raise ValueError("Stage2 training-step audit requires tgvf module")
    if samples is None:
        train_file = (((bundle.get("dataset") or {}).get("train_file") or {}).get("path"))
        if not train_file:
            raise ValueError("Stage2 training-step audit requires dataset.train_file.path")
        dataset = TGVFv3Stage2Dataset(train_file)
        samples = _select_stage2_step_probe_samples(
            dataset.samples,
            requested_count=max(
                1,
                int(((bundle.get("batch") or {}).get("micro_batch_size")) or 1),
            ),
        )
    if not samples:
        raise ValueError("Stage2 training-step audit found no usable samples")
    loss = bundle.get("loss") or {}
    span_weights = dict(loss.get("weighted_span_loss") or {})
    output = v3_stage2_batched_training_step(
        qwen_model=qwen_model,
        qwen_forward_model=qwen_forward_model,
        processor=processor,
        foveal_module=foveal_module,
        samples=samples,
        loss_weights=Stage2LossWeights(
            **span_weights,
            visual_token_manifold=float(loss.get("visual_token_manifold") or 0.0),
        ),
        device=_parameter_audit_device(torch),
        hidden_state_index=int(((bundle.get("training") or {}).get("capture_layer")) or -1),
        max_image_resolution=(bundle.get("training") or {}).get("max_image_resolution"),
        position_mode=str((bundle.get("training") or {}).get("fvt_position_mode")),
        mask_original_image_after_tgvf=bool(
            (bundle.get("mask_policy") or {}).get("mask_original_image_after_tgvf")
        ),
        mask_original_image_after_tgvf_prob=float(
            (bundle.get("mask_policy") or {}).get("mask_original_image_after_tgvf_prob")
        ),
        mask_original_image_after_tgvf_scope=str(
            (bundle.get("mask_policy") or {}).get("mask_original_image_after_tgvf_scope")
        ),
        protocol=str(bundle.get("protocol")),
        deepstack_enabled=bool((bundle.get("deepstack") or {}).get("enabled")),
    )
    return {
        "forward_completed": True,
        "sample_count": len(samples),
        "loss_total": _scalar_float(output.loss_total),
        "loss_tensor": output.loss_total,
        "loss_focus": _scalar_float(output.loss_focus),
        "loss_no_focus": _scalar_float(output.loss_no_focus),
        "loss_visual_token_manifold": _scalar_float(output.loss_visual_token_manifold),
        "mask_original_image_after_tgvf": bool(
            (bundle.get("mask_policy") or {}).get("mask_original_image_after_tgvf")
        ),
        "debug": output.debug,
    }


def _select_stage1_step_probe_samples(samples: list[Any], *, requested_count: int) -> list[Any]:
    by_image: dict[str, list[Any]] = {}
    for sample in samples:
        image_key = str(getattr(sample, "image_id", None) or getattr(sample, "image", ""))
        by_image.setdefault(image_key, []).append(sample)
    same_image_group = next(
        (group for group in by_image.values() if len(group) > 1),
        None,
    )
    if same_image_group is not None:
        return same_image_group[: max(2, requested_count)]
    return samples[:requested_count]


def _select_stage2_step_probe_samples(samples: list[Any], *, requested_count: int) -> list[Any]:
    selected: list[Any] = []
    focus = next((sample for sample in samples if getattr(sample, "need_focus", False)), None)
    no_focus = next(
        (sample for sample in samples if not getattr(sample, "need_focus", False)),
        None,
    )
    if focus is not None:
        selected.append(focus)
    if no_focus is not None and len(selected) < requested_count:
        selected.append(no_focus)
    for sample in samples:
        if len(selected) >= requested_count:
            break
        if sample not in selected:
            selected.append(sample)
    return selected


def _scalar_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            return float(value.detach().cpu())
        return float(value)
    except Exception:
        return None


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

    constructed = _construct_actual_optimizer_scheduler(
        torch_module=torch,
        bundle=bundle,
        artifacts=artifacts,
        loaded_modules=loaded_modules,
        expected_stage=expected_stage,
    )
    optimizer_contract = constructed["optimizer_contract"]
    optimizer = constructed["optimizer"]
    scheduler = constructed["scheduler"]
    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()
    constructed_groups = constructed["constructed_groups"]
    empty_groups = constructed["empty_groups"]
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
            "max_grad_norm": (optimizer_contract.get("optimizer") or {}).get(
                "max_grad_norm"
            ),
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


def _construct_actual_optimizer_scheduler(
    *,
    torch_module: Any,
    bundle: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    loaded_modules: dict[str, Any],
    expected_stage: TrainingStage,
) -> dict[str, Any]:
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
    optimizer = torch_module.optim.AdamW(
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
        torch_module=torch_module,
        optimizer=optimizer,
        scheduler_contract=optimizer_contract.get("scheduler") or {},
        max_steps=int(((bundle.get("training") or {}).get("max_steps")) or 0),
    )
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
    return {
        "optimizer_contract": optimizer_contract,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "constructed_groups": constructed_groups,
        "empty_groups": [spec["name"] for spec in group_specs if not spec["params"]],
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
        device_map=model_cfg.get("device_map") if "device_map" in model_cfg else "auto",
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
    tgvf_cfg = _resolved_stage1_tgvf_config(bundle=bundle, dims=dims)
    tgvf = build_tgvf_module(
        variant=str(tgvf_cfg.get("variant") or "tgvf_v2_bidirectional"),
        d_lm=dims["d_lm"],
        d_v=dims["d_v"],
        num_foveated_tokens=tgvf_cfg.get("num_foveated_tokens"),
        spatial_merge_size=int(tgvf_cfg.get("spatial_merge_size") or dims["spatial_merge_size"]),
        attn_dim=tgvf_cfg.get("attn_dim"),
        encoder_adapter_layers=tuple(tgvf_cfg.get("encoder_adapter_layers") or (8, 16, 24)),
        encoder_adapter_gate_init=float(tgvf_cfg.get("encoder_adapter_gate_init", 0.0)),
        encoder_adapter_type=str(tgvf_cfg.get("encoder_adapter_type") or "bidirectional"),
        encoder_adapter_share_weights=bool(
            tgvf_cfg.get("encoder_adapter_share_weights", False)
        ),
        encoder_adapter_layer_index_base=int(
            tgvf_cfg.get("encoder_adapter_layer_index_base", 0)
        ),
        encoder_reencode_deepstack_compatible=bool(
            tgvf_cfg.get("encoder_reencode_deepstack_compatible", False)
        ),
        d_deepstack_enabled=bool(tgvf_cfg.get("d_deepstack_enabled", False)),
        d_deepstack_branch_layers=tuple(
            tgvf_cfg.get("d_deepstack_branch_layers") or (8, 16, 24)
        ),
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
            "resolved_tgvf_config": tgvf_cfg,
        },
    }


def _resolved_stage1_tgvf_config(
    *,
    bundle: dict[str, Any],
    dims: dict[str, Any],
) -> dict[str, Any]:
    training = bundle.get("training") or {}
    planned = dict(bundle.get("tgvf") or {})
    spatial_merge_size = planned.get("spatial_merge_size")
    if spatial_merge_size in (None, "auto"):
        spatial_merge_size = dims["spatial_merge_size"]
    return {
        "variant": str(
            planned.get("variant")
            or training.get("variant")
            or "tgvf_v2_bidirectional"
        ),
        "num_foveated_tokens": planned.get("num_foveated_tokens"),
        "spatial_merge_size": int(spatial_merge_size),
        "attn_dim": planned.get("attn_dim"),
        "encoder_adapter_layers": [
            int(layer)
            for layer in (planned.get("encoder_adapter_layers") or (8, 16, 24))
        ],
        "encoder_adapter_type": str(
            planned.get("encoder_adapter_type") or "bidirectional"
        ),
        "encoder_adapter_gate_init": float(
            planned.get("encoder_adapter_gate_init", 0.0)
        ),
        "encoder_adapter_share_weights": bool(
            planned.get("encoder_adapter_share_weights", False)
        ),
        "encoder_adapter_layer_index_base": int(
            planned.get("encoder_adapter_layer_index_base", 0)
        ),
        "encoder_reencode_deepstack_compatible": bool(
            planned.get("encoder_reencode_deepstack_compatible", False)
        ),
        "d_deepstack_enabled": bool(planned.get("d_deepstack_enabled", False)),
        "d_deepstack_branch_layers": [
            int(layer) for layer in (planned.get("d_deepstack_branch_layers") or (8, 16, 24))
        ],
        "d_deepstack_adapter_type": planned.get("d_deepstack_adapter_type"),
        "d_deepstack_independent_branch_adapters": bool(
            planned.get("d_deepstack_independent_branch_adapters", False)
        ),
        "encoder_reencode": bool(
            planned.get("encoder_reencode")
            if planned.get("encoder_reencode") is not None
            else str(
                planned.get("variant")
                or training.get("variant")
                or "tgvf_v2_bidirectional"
            )
            == "tgvf_encoder_bidir_8_16_24"
        ),
        "preserve_llm_kv_cache": bool(planned.get("preserve_llm_kv_cache", True)),
        "second_full_llm_forward": bool(
            planned.get("second_full_llm_forward", False)
        ),
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
            protocol_special_token_ids,
        )
        from revisit_vlm.tgvf_training import build_tgvf_module
        from revisit_vlm.tgvf_v3_stage1 import freeze_qwen_backbone, infer_qwen3_stage1_dims
        from revisit_vlm.tgvf_v3_stage2 import TGVFv3Stage2Dataset
        from scripts.train_tgvf_v3_stage2 import _restore_protocol_c_token_rows_from_stage1
        from revisit_vlm_clean.peft_token_rows import (
            PROTOCOL_TOKEN_TRAINING_FULL_MODULES,
            protocol_token_peft_kwargs,
        )
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
        device_map=model_cfg.get("device_map") if "device_map" in model_cfg else "auto",
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
    stage1_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    token_mode = str(
        lora.get("protocol_token_training_mode")
        or PROTOCOL_TOKEN_TRAINING_FULL_MODULES
    )
    if protocol in token_row_protocols and token_mode != PROTOCOL_TOKEN_TRAINING_FULL_MODULES:
        _restore_protocol_c_token_rows_from_stage1(
            model=model,
            tokenizer=processor.tokenizer,
            protocol=protocol,
            stage1_checkpoint=stage1_checkpoint,
        )
    freeze_qwen_backbone(model)
    runtime = ((bundle.get("module_policy") or {}).get("training_runtime") or {})
    if runtime.get("gradient_checkpointing") and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False
    token_peft_kwargs = (
        protocol_token_peft_kwargs(
            mode=token_mode,
            token_ids=list(protocol_special_token_ids(processor.tokenizer, protocol=protocol).values()),
        )
        if protocol in token_row_protocols
        else {"modules_to_save": None, "trainable_token_indices": None}
    )
    modules_to_save = token_peft_kwargs["modules_to_save"]
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
            trainable_token_indices=token_peft_kwargs["trainable_token_indices"],
            ensure_weight_tying=False,
        ),
    )
    utility_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    if protocol in token_row_protocols and token_mode == PROTOCOL_TOKEN_TRAINING_FULL_MODULES:
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
    tgvf_cfg = _resolved_tgvf_config_from_checkpoint(
        checkpoint=stage1_checkpoint,
        training=training,
        dims=dims,
    )
    tgvf = build_tgvf_module(
        variant=str(tgvf_cfg.get("variant") or training.get("variant") or "tgvf_v2_bidirectional"),
        d_lm=dims["d_lm"],
        d_v=dims["d_v"],
        num_foveated_tokens=tgvf_cfg.get("num_foveated_tokens"),
        spatial_merge_size=int(tgvf_cfg.get("spatial_merge_size") or dims["spatial_merge_size"]),
        attn_dim=tgvf_cfg.get("attn_dim"),
        encoder_adapter_layers=tuple(tgvf_cfg.get("encoder_adapter_layers") or (8, 16, 24)),
        encoder_adapter_gate_init=float(tgvf_cfg.get("encoder_adapter_gate_init", 0.0)),
        encoder_adapter_type=str(tgvf_cfg.get("encoder_adapter_type") or "bidirectional"),
        encoder_adapter_share_weights=bool(tgvf_cfg.get("encoder_adapter_share_weights", False)),
        encoder_adapter_layer_index_base=int(tgvf_cfg.get("encoder_adapter_layer_index_base", 0)),
        encoder_reencode_deepstack_compatible=bool(
            tgvf_cfg.get("encoder_reencode_deepstack_compatible", False)
        ),
        d_deepstack_enabled=bool(tgvf_cfg.get("d_deepstack_enabled", False)),
        d_deepstack_branch_layers=tuple(
            tgvf_cfg.get("d_deepstack_branch_layers") or (8, 16, 24)
        ),
    ).to(device=device, dtype=next(model.parameters()).dtype)
    tgvf.load_state_dict(stage1_checkpoint["tgvf_module"], strict=True)
    return {
        "modules": {"qwen_lora": model, "tgvf": tgvf},
        "qwen_model": utility_model,
        "qwen_forward_model": model,
        "processor": processor,
        "tokenizer": processor.tokenizer,
        "loader": {
            "backend": "stage2_qwen3_lora_tgvf_parameter_audit",
            "processor_id": getattr(loaded, "processor_id", model_cfg.get("processor_id")),
            "protocol_token_info": token_info,
            "dims": dims,
            "stage1_global_step": stage1_checkpoint.get("global_step"),
            "modules_to_save": modules_to_save,
            "protocol_token_training_mode": token_mode,
            "trainable_token_indices": token_peft_kwargs["trainable_token_indices"],
            "resolved_tgvf_config": tgvf_cfg,
        },
    }


def _resolved_tgvf_config_from_checkpoint(
    *,
    checkpoint: dict[str, Any],
    training: dict[str, Any],
    dims: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_config = checkpoint.get("config") or {}
    raw = dict(checkpoint_config.get("tgvf") or {})
    variant = str(raw.get("variant") or training.get("variant") or "tgvf_v2_bidirectional")
    spatial_merge_size = raw.get("spatial_merge_size")
    if spatial_merge_size in (None, "auto"):
        spatial_merge_size = dims["spatial_merge_size"]
    return {
        "variant": variant,
        "num_foveated_tokens": raw.get("num_foveated_tokens"),
        "spatial_merge_size": int(spatial_merge_size),
        "attn_dim": raw.get("attn_dim"),
        "encoder_adapter_layers": [
            int(layer) for layer in (raw.get("encoder_adapter_layers") or (8, 16, 24))
        ],
        "encoder_adapter_type": str(raw.get("encoder_adapter_type") or "bidirectional"),
        "encoder_adapter_gate_init": float(raw.get("encoder_adapter_gate_init", 0.0)),
        "encoder_adapter_share_weights": bool(raw.get("encoder_adapter_share_weights", False)),
        "encoder_adapter_layer_index_base": int(
            raw.get("encoder_adapter_layer_index_base", 0)
        ),
        "encoder_reencode_deepstack_compatible": bool(
            raw.get("encoder_reencode_deepstack_compatible", False)
        ),
        "d_deepstack_enabled": bool(raw.get("d_deepstack_enabled", False)),
        "d_deepstack_branch_layers": [
            int(layer) for layer in (raw.get("d_deepstack_branch_layers") or (8, 16, 24))
        ],
        "d_deepstack_adapter_type": raw.get("d_deepstack_adapter_type"),
        "d_deepstack_independent_branch_adapters": bool(
            raw.get("d_deepstack_independent_branch_adapters", False)
        ),
        "encoder_reencode": bool(
            raw.get("encoder_reencode")
            if raw.get("encoder_reencode") is not None
            else variant == "tgvf_encoder_bidir_8_16_24"
        ),
        "preserve_llm_kv_cache": bool(raw.get("preserve_llm_kv_cache", True)),
        "second_full_llm_forward": bool(raw.get("second_full_llm_forward", False)),
    }


def _parameter_audit_device(torch_module: Any) -> Any:
    if torch_module.cuda.is_available():
        return torch_module.device(f"cuda:{torch_module.cuda.current_device()}")
    return torch_module.device("cpu")


def _runtime_audit_report(
    *,
    bundle_path: Path,
    bundle: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    trainable_parameters: dict[str, Any],
    optimizer_runtime: dict[str, Any] | None,
    checkpoint_runtime: dict[str, Any] | None,
    training_step_runtime: dict[str, Any] | None,
    optimizer_step_runtime: dict[str, Any] | None,
    trainer_loop_runtime: dict[str, Any] | None,
    training_checkpoint_publish_runtime: dict[str, Any] | None,
    training_checkpoint_resume_runtime: dict[str, Any] | None,
    training_cadence_runtime: dict[str, Any] | None,
    expected_stage: TrainingStage,
) -> dict[str, Any]:
    artifact_checks = _runtime_artifact_checks(
        bundle,
        artifacts,
        trainable_parameters,
        optimizer_runtime,
        checkpoint_runtime,
        training_step_runtime,
        optimizer_step_runtime,
        trainer_loop_runtime,
        training_checkpoint_publish_runtime,
        training_checkpoint_resume_runtime,
        training_cadence_runtime,
    )
    launch_gates = _launch_gate_audit(
        bundle,
        artifacts,
        trainable_parameters,
        optimizer_runtime,
        checkpoint_runtime,
        training_step_runtime,
        optimizer_step_runtime,
        trainer_loop_runtime,
        training_checkpoint_publish_runtime,
        training_checkpoint_resume_runtime,
        training_cadence_runtime,
    )
    trainer_loop_ported = bool(
        (bundle.get("clean_executor") or {}).get("trainer_loop_ported")
    )
    blocking_items = (
        []
        if trainer_loop_ported
        else ["native trainer loop has not been ported into revisit_vlm_clean"]
    )
    if not trainable_parameters["payload"].get("actual_model_parameters_loaded"):
        blocking_items.append("actual trainable parameter audit requires loading the model")
    if not (
        optimizer_runtime
        and optimizer_runtime["payload"].get("actual_optimizer_constructed")
        and optimizer_runtime["payload"].get("actual_scheduler_constructed")
    ):
        blocking_items.append("actual optimizer/scheduler construction requires --audit-optimizer")
    if not _checkpoint_contract_runtime_validated(
        checkpoint_runtime=checkpoint_runtime,
        training_checkpoint_publish_runtime=training_checkpoint_publish_runtime,
    ):
        blocking_items.append(
            "checkpoint save/load parity requires --audit-checkpoint or "
            "--audit-checkpoint-publish"
        )
    if not (
        training_step_runtime
        and training_step_runtime["payload"].get("actual_training_step_forward")
    ):
        blocking_items.append(
            f"{expected_stage.value} no-backward training-step forward requires "
            "--audit-training-step"
        )
    if not (
        (
            optimizer_step_runtime
            and optimizer_step_runtime["payload"].get("actual_optimizer_step_probe")
            and optimizer_step_runtime["payload"].get("backward_called")
            and optimizer_step_runtime["payload"].get("optimizer_step_called")
            and optimizer_step_runtime["payload"].get("scheduler_step_called")
        )
        or (
            trainer_loop_runtime
            and trainer_loop_runtime["payload"].get("actual_trainer_loop_probe")
            and trainer_loop_runtime["payload"].get("optimizer_steps_completed") == 1
            and trainer_loop_runtime["payload"].get("scheduler_steps_completed") == 1
        )
    ):
        blocking_items.append(
            f"{expected_stage.value} backward/optimizer/scheduler step probe requires "
            "--audit-optimizer-step or --audit-trainer-loop"
        )
    if not (
        trainer_loop_runtime
        and trainer_loop_runtime["payload"].get("actual_trainer_loop_probe")
        and trainer_loop_runtime["payload"].get("optimizer_steps_completed") == 1
        and trainer_loop_runtime["payload"].get("scheduler_steps_completed") == 1
    ):
        blocking_items.append(
            f"{expected_stage.value} gradient-accumulation trainer-loop probe requires "
            "--audit-trainer-loop"
        )
    if not _checkpoint_publish_runtime_validated(training_checkpoint_publish_runtime):
        blocking_items.append(
            f"{expected_stage.value} post-trainer-loop checkpoint publish requires "
            "--audit-checkpoint-publish"
        )
    if not _checkpoint_resume_runtime_validated(training_checkpoint_resume_runtime):
        blocking_items.append(
            f"{expected_stage.value} checkpoint resume requires --audit-checkpoint-resume"
        )
    if not _training_cadence_runtime_validated(training_cadence_runtime):
        blocking_items.append(
            f"{expected_stage.value} training cadence requires --audit-cadence"
        )
    if expected_stage == TrainingStage.STAGE2:
        deepstack_plan = artifacts.get("deepstack_training_plan") or {}
        if not bool(deepstack_plan.get("execution_supported", True)):
            blocking_items.extend(
                str(item) for item in deepstack_plan.get("blocking_items") or []
            )
    status = "ready_for_explicit_launch" if not blocking_items else "blocked_before_training_loop"
    return {
        "schema_version": "clean_training_runtime_audit_v1",
        "stage": str(expected_stage),
        "run_id": bundle.get("run_id"),
        "status": status,
        "will_launch_training": False,
        "training_runtime_ported": trainer_loop_ported,
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
    training_step_runtime: dict[str, Any] | None,
    optimizer_step_runtime: dict[str, Any] | None,
    trainer_loop_runtime: dict[str, Any] | None,
    training_checkpoint_publish_runtime: dict[str, Any] | None,
    training_checkpoint_resume_runtime: dict[str, Any] | None,
    training_cadence_runtime: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    artifact_paths = dict(bundle.get("runtime_artifacts") or {})
    artifact_paths["trainable_parameters"] = trainable_parameters["path"]
    if optimizer_runtime is not None:
        artifact_paths["optimizer_runtime"] = optimizer_runtime["path"]
    if checkpoint_runtime is not None:
        artifact_paths["checkpoint_runtime"] = checkpoint_runtime["path"]
    if training_step_runtime is not None:
        artifact_paths["training_step_runtime"] = training_step_runtime["path"]
    if optimizer_step_runtime is not None:
        artifact_paths["optimizer_step_runtime"] = optimizer_step_runtime["path"]
    if trainer_loop_runtime is not None:
        artifact_paths["trainer_loop_runtime"] = trainer_loop_runtime["path"]
    if training_checkpoint_publish_runtime is not None:
        artifact_paths["training_checkpoint_publish_runtime"] = (
            training_checkpoint_publish_runtime["path"]
        )
    if training_checkpoint_resume_runtime is not None:
        artifact_paths["training_checkpoint_resume_runtime"] = (
            training_checkpoint_resume_runtime["path"]
        )
    if training_cadence_runtime is not None:
        artifact_paths["training_cadence_runtime"] = training_cadence_runtime["path"]
    checks = []
    for name, path_text in sorted(artifact_paths.items()):
        path = Path(str(path_text))
        if name == "trainable_parameters":
            payload = trainable_parameters["payload"]
        elif name == "optimizer_runtime":
            payload = optimizer_runtime["payload"] if optimizer_runtime else {}
        elif name == "checkpoint_runtime":
            payload = checkpoint_runtime["payload"] if checkpoint_runtime else {}
        elif name == "training_step_runtime":
            payload = training_step_runtime["payload"] if training_step_runtime else {}
        elif name == "optimizer_step_runtime":
            payload = optimizer_step_runtime["payload"] if optimizer_step_runtime else {}
        elif name == "trainer_loop_runtime":
            payload = trainer_loop_runtime["payload"] if trainer_loop_runtime else {}
        elif name == "training_checkpoint_publish_runtime":
            payload = (
                training_checkpoint_publish_runtime["payload"]
                if training_checkpoint_publish_runtime
                else {}
            )
        elif name == "training_checkpoint_resume_runtime":
            payload = (
                training_checkpoint_resume_runtime["payload"]
                if training_checkpoint_resume_runtime
                else {}
            )
        elif name == "training_cadence_runtime":
            payload = training_cadence_runtime["payload"] if training_cadence_runtime else {}
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
    training_step_runtime: dict[str, Any] | None,
    optimizer_step_runtime: dict[str, Any] | None,
    trainer_loop_runtime: dict[str, Any] | None,
    training_checkpoint_publish_runtime: dict[str, Any] | None,
    training_checkpoint_resume_runtime: dict[str, Any] | None,
    training_cadence_runtime: dict[str, Any] | None,
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
    if _training_cadence_runtime_validated(training_cadence_runtime):
        satisfied.add("validate_training_cadence_from_plan")
    if actual_optimizer_constructed:
        satisfied.add("construct_optimizer_and_scheduler_from_plan")
    actual_checkpoint_validated = _checkpoint_contract_runtime_validated(
        checkpoint_runtime=checkpoint_runtime,
        training_checkpoint_publish_runtime=training_checkpoint_publish_runtime,
    )
    if actual_checkpoint_validated:
        satisfied.add("save_checkpoint_with_clean_contract")
    actual_checkpoint_publish_validated = _checkpoint_publish_runtime_validated(
        training_checkpoint_publish_runtime
    )
    if actual_checkpoint_publish_validated:
        satisfied.add("publish_training_checkpoint_after_trainer_loop")
    actual_checkpoint_resume_validated = _checkpoint_resume_runtime_validated(
        training_checkpoint_resume_runtime
    )
    if actual_checkpoint_resume_validated:
        satisfied.add("resume_training_from_clean_checkpoint")
    actual_trainer_loop_validated = bool(
        trainer_loop_runtime
        and trainer_loop_runtime["payload"].get("actual_trainer_loop_probe")
        and trainer_loop_runtime["payload"].get("optimizer_steps_completed") == 1
        and trainer_loop_runtime["payload"].get("scheduler_steps_completed") == 1
        and trainer_loop_runtime["payload"].get("micro_steps_run")
        == trainer_loop_runtime["payload"].get("gradient_accumulation_steps")
        and trainer_loop_runtime["payload"].get("backward_micro_steps")
        == trainer_loop_runtime["payload"].get("gradient_accumulation_steps")
        and not trainer_loop_runtime["payload"].get("checkpoint_published")
        and not trainer_loop_runtime["payload"].get("training_run_launched")
    )
    if actual_trainer_loop_validated:
        satisfied.add("run_gradient_accumulation_loop_from_plan")
    actual_optimizer_step_validated = bool(
        (
            optimizer_step_runtime
            and optimizer_step_runtime["payload"].get("actual_optimizer_step_probe")
            and optimizer_step_runtime["payload"].get("backward_called")
            and optimizer_step_runtime["payload"].get("optimizer_step_called")
            and optimizer_step_runtime["payload"].get("scheduler_step_called")
            and not optimizer_step_runtime["payload"].get("checkpoint_published")
            and not optimizer_step_runtime["payload"].get("training_loop_launched")
        )
        or actual_trainer_loop_validated
    )
    if actual_optimizer_step_validated:
        satisfied.add("run_backward_optimizer_scheduler_step_from_plan")
    if training_step_runtime is not None:
        step_payload = training_step_runtime["payload"]
        if step_payload.get("stage1_readout_context_applied"):
            satisfied.add("stage1_readout_context_uses_qwen_v_merge")
        if step_payload.get("stage1_position_ids_applied"):
            satisfied.add("stage1_position_ids_use_real_qwen3_mrope")
        if step_payload.get("stage1_matrix_ce_and_manifold_losses_applied"):
            satisfied.add("stage1_matrix_ce_and_manifold_losses_match_plan")
        if step_payload.get("fast_batched_stage2_used"):
            satisfied.add("use_fast_batched_stage2_path")
        if step_payload.get("weighted_span_loss_applied"):
            satisfied.add("apply_weighted_span_losses_from_plan")
        if step_payload.get("mask_scope_applied"):
            satisfied.add("apply_original_image_mask_scope_from_plan")
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
    if bundle.get("stage") == str(TrainingStage.STAGE2):
        deepstack_plan = artifacts.get("deepstack_training_plan") or {}
        if bool(deepstack_plan.get("execution_supported", True)):
            satisfied.add("apply_deepstack_training_scope_when_enabled")
    pending_model_load = {
        "load_model_and_processor",
        "ensure_protocol_token_rows",
        "set_training_use_cache_false",
        "validate_training_cadence_from_plan",
        "construct_optimizer_and_scheduler_from_plan",
        "run_backward_optimizer_scheduler_step_from_plan",
        "run_gradient_accumulation_loop_from_plan",
        "emit_trainable_parameter_audit",
        "save_checkpoint_with_clean_contract",
        "publish_training_checkpoint_after_trainer_loop",
        "resume_training_from_clean_checkpoint",
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
        elif (
            gate == "apply_deepstack_training_scope_when_enabled"
            and bundle.get("stage") == str(TrainingStage.STAGE2)
            and not bool(
                (artifacts.get("deepstack_training_plan") or {}).get(
                    "execution_supported",
                    True,
                )
            )
        ):
            status = "blocked_unimplemented_deepstack_training"
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
        "training_step_runtime_status": (
            training_step_runtime["payload"].get("status")
            if training_step_runtime
            else "not_requested"
        ),
        "optimizer_step_runtime_status": (
            optimizer_step_runtime["payload"].get("status")
            if optimizer_step_runtime
            else "not_requested"
        ),
        "trainer_loop_runtime_status": (
            trainer_loop_runtime["payload"].get("status")
            if trainer_loop_runtime
            else "not_requested"
        ),
        "training_checkpoint_publish_runtime_status": (
            training_checkpoint_publish_runtime["payload"].get("status")
            if training_checkpoint_publish_runtime
            else "not_requested"
        ),
        "training_checkpoint_resume_runtime_status": (
            training_checkpoint_resume_runtime["payload"].get("status")
            if training_checkpoint_resume_runtime
            else "not_requested"
        ),
        "training_cadence_runtime_status": (
            training_cadence_runtime["payload"].get("status")
            if training_cadence_runtime
            else "not_requested"
        ),
        "checkpoint_contract_status": artifacts["checkpoint_contract"].get("status"),
        "trainable_parameters_status": trainable_parameters["payload"].get("status"),
        "gates": gates,
    }


def _checkpoint_contract_runtime_validated(
    *,
    checkpoint_runtime: dict[str, Any] | None,
    training_checkpoint_publish_runtime: dict[str, Any] | None,
) -> bool:
    return bool(
        (
            checkpoint_runtime
            and checkpoint_runtime["payload"].get("actual_checkpoint_saved")
            and checkpoint_runtime["payload"].get("actual_checkpoint_loaded")
            and not checkpoint_runtime["payload"].get("missing_required_keys")
            and checkpoint_runtime["payload"].get("state_checks_ok")
        )
        or _checkpoint_publish_runtime_validated(training_checkpoint_publish_runtime)
    )


def _checkpoint_publish_runtime_validated(runtime: dict[str, Any] | None) -> bool:
    return bool(
        runtime
        and runtime["payload"].get("actual_checkpoint_publish_probe_saved")
        and runtime["payload"].get("actual_checkpoint_publish_probe_loaded")
        and not runtime["payload"].get("missing_required_keys")
        and runtime["payload"].get("state_checks_ok")
        and runtime["payload"].get("step_checks_ok")
        and not runtime["payload"].get("training_run_launched")
        and not runtime["payload"].get("full_epoch_loop_entered")
    )


def _checkpoint_resume_runtime_validated(runtime: dict[str, Any] | None) -> bool:
    return bool(
        runtime
        and runtime["payload"].get("actual_checkpoint_resume_probe_loaded")
        and runtime["payload"].get("model_state_loaded")
        and runtime["payload"].get("optimizer_state_loaded")
        and runtime["payload"].get("scheduler_state_loaded")
        and runtime["payload"].get("state_checks_ok")
        and runtime["payload"].get("step_checks_ok")
        and runtime["payload"].get("protocol_rows_ok")
        and not runtime["payload"].get("training_run_launched")
        and not runtime["payload"].get("full_epoch_loop_entered")
    )


def _training_cadence_runtime_validated(runtime: dict[str, Any] | None) -> bool:
    if not runtime:
        return False
    payload = runtime["payload"]
    max_steps = int(payload.get("max_steps") or 0)
    checkpoint_steps = list(payload.get("checkpoint_save_steps") or [])
    eval_steps = list(payload.get("eval_steps") or [])
    return bool(
        payload.get("actual_training_cadence_validated")
        and not payload.get("training_run_launched")
        and max_steps >= 1
        and checkpoint_steps
        and int(checkpoint_steps[-1]) == max_steps
        and (
            not payload.get("eval_enabled")
            or (eval_steps and int(eval_steps[-1]) == max_steps)
        )
    )


def _runtime_audit_status(audit: dict[str, Any]) -> dict[str, Any]:
    gates = audit.get("launch_gates") or {}
    launch_readiness = audit.get("launch_readiness") or {}
    return {
        "schema_version": "clean_training_runtime_audit_status_v1",
        "stage": audit.get("stage"),
        "run_id": audit.get("run_id"),
        "status": audit.get("status"),
        "will_launch_training": audit.get("will_launch_training"),
        "training_runtime_ported": audit.get("training_runtime_ported"),
        "identity_validated_gates": gates.get("identity_validated"),
        "pending_real_trainer_loop_gates": gates.get("pending_real_trainer_loop"),
        "optimizer_step_runtime_status": gates.get("optimizer_step_runtime_status"),
        "trainer_loop_runtime_status": gates.get("trainer_loop_runtime_status"),
        "training_checkpoint_publish_runtime_status": gates.get(
            "training_checkpoint_publish_runtime_status"
        ),
        "training_checkpoint_resume_runtime_status": gates.get(
            "training_checkpoint_resume_runtime_status"
        ),
        "training_cadence_runtime_status": gates.get("training_cadence_runtime_status"),
        "training_launch_readiness_status": launch_readiness.get("status"),
        "launch_contract_ready_for_trainer_loop": launch_readiness.get(
            "contract_ready_for_trainer_loop"
        ),
        "launch_permitted": launch_readiness.get("launch_permitted"),
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
    ]
    if audit.get("launch_readiness"):
        launch_readiness = audit["launch_readiness"]
        lines.extend(
            [
                f"launch_readiness_status: {launch_readiness.get('status')}",
                "launch_contract_ready_for_trainer_loop: "
                f"{launch_readiness.get('contract_ready_for_trainer_loop')}",
                f"launch_permitted: {launch_readiness.get('launch_permitted')}",
            ]
        )
    lines.append("blocking_items:")
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
    if expected_stage == TrainingStage.STAGE1:
        first_batch = _stage1_materialized_first_batch_rows(
            plan=plan,
            train_path=str(train_identity["path"]),
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


def _write_deepstack_training_plan_artifact(
    *,
    execution_dir: Path,
    plan: dict[str, Any],
    expected_stage: TrainingStage,
) -> dict[str, Any]:
    if expected_stage != TrainingStage.STAGE2:
        raise ValueError("deepstack training plan is Stage2-only")
    payload = dict(plan.get("deepstack_training_plan") or {})
    if payload.get("schema_version") != "clean_deepstack_training_plan_v1":
        raise ValueError("Stage2 plan missing clean deepstack_training_plan")
    path = execution_dir / "deepstack_training_plan.json"
    _write_json(path, payload)
    return {"path": str(path), "plan": payload}


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
        return ["image", "question", "target", "evidence_description"]
    return ["image", "question", "answer", "need_focus", "evidence_state"]


def _stage1_materialized_first_batch_rows(
    *,
    plan: dict[str, Any],
    train_path: str,
) -> list[dict[str, Any]]:
    try:
        from revisit_vlm.qwen3_vl_tgvf import NEED_LOCAL_EVIDENCE
        from revisit_vlm.tgvf_v3_stage1 import TGVFv3Stage1Dataset
    except Exception as exc:
        raise RuntimeError(
            "Stage1 first-batch materialization dependencies unavailable"
        ) from exc
    batch = plan.get("batch") or {}
    training = plan.get("training") or {}
    world_size = max(1, int(batch.get("world_size") or 1))
    micro_batch_size = max(1, int(batch.get("micro_batch_size") or 1))
    accumulation_steps = max(1, int(batch.get("gradient_accumulation_steps") or 1))
    seed = int(training.get("seed") or 0)
    dataset = TGVFv3Stage1Dataset(train_path, focus_only=True)
    row_identities = _stage1_focus_row_identities(
        train_path,
        need_local_evidence=str(NEED_LOCAL_EVIDENCE),
    )
    if len(row_identities) != len(dataset.samples):
        raise ValueError(
            "Stage1 first-batch row identity count does not match parsed samples: "
            f"{len(row_identities)} != {len(dataset.samples)}"
        )
    materialized: list[dict[str, Any]] = []
    for rank in range(world_size):
        cursor = _SingleProcessSampleCursor(
            samples=list(dataset.samples),
            batch_size=micro_batch_size,
            stage=TrainingStage.STAGE1,
            dataset_role="train",
            dataset_path=train_path,
            rank=rank,
            world_size=world_size,
            seed=seed,
        )
        for micro_index in range(accumulation_steps):
            batch_record = cursor.next_batch()
            for trace in batch_record["sample_trace"]:
                sample_index = int(trace["sample_index"])
                row = dict(row_identities[sample_index])
                row.update(
                    {
                        "rank": rank,
                        "micro_index": micro_index,
                        "sample_index": sample_index,
                        "sampler_mode": cursor.summary().get("mode"),
                        "sampler_group_owner": cursor.summary().get(
                            "same_image_group_owner"
                        ),
                    }
                )
                materialized.append(row)
    return materialized


def _stage1_focus_row_identities(
    train_path: str,
    *,
    need_local_evidence: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = Path(train_path)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            need_focus = bool(record.get("need_focus", True))
            trajectory_type = record.get("trajectory_type", "single_focus")
            evidence_state = record.get("evidence_state", need_local_evidence)
            if (
                not need_focus
                or trajectory_type != "single_focus"
                or evidence_state != need_local_evidence
            ):
                continue
            rows.append(_row_runtime_identity(record, line_no=line_no))
    return rows


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
        _validate_stage1_tgvf_config(plan.get("tgvf") or {})
        _validate_stage1_readout_context(plan.get("readout_context") or {})
    if stage == TrainingStage.STAGE2:
        _validate_stage2_deepstack_training_plan(plan)
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
    if stage == TrainingStage.STAGE1:
        if dataset.get("batch_sampling") != "same_image":
            raise ValueError("Stage1 dataset.batch_sampling must be same_image")
        if dataset.get("drop_incomplete_same_image_batches") is not True:
            raise ValueError("Stage1 must drop incomplete same-image batches")
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


def _validate_stage1_tgvf_config(config: dict[str, Any]) -> None:
    if not config.get("variant"):
        raise ValueError("Stage1 tgvf.variant is required")
    spatial_merge_size = config.get("spatial_merge_size")
    if spatial_merge_size is None:
        raise ValueError("Stage1 tgvf.spatial_merge_size is required")
    if spatial_merge_size != "auto" and int(spatial_merge_size) < 1:
        raise ValueError("Stage1 tgvf.spatial_merge_size must be auto or positive")
    if config.get("encoder_adapter_layers") is None:
        raise ValueError("Stage1 tgvf.encoder_adapter_layers is required")
    if config.get("encoder_adapter_type") not in {
        "bidirectional",
        "bidirectional_film_aggressive",
    }:
        raise ValueError("Stage1 tgvf.encoder_adapter_type is invalid")


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


def _validate_stage2_deepstack_training_plan(plan: dict[str, Any]) -> None:
    deepstack = plan.get("deepstack") or {}
    deepstack_plan = plan.get("deepstack_training_plan") or {}
    if deepstack_plan.get("schema_version") != "clean_deepstack_training_plan_v1":
        raise ValueError("Stage2 plan must include clean deepstack_training_plan")
    enabled = bool(deepstack.get("enabled"))
    if bool(deepstack_plan.get("enabled")) != enabled:
        raise ValueError("Stage2 deepstack_training_plan enabled mismatch")
    if deepstack_plan.get("gate_name") != "apply_deepstack_training_scope_when_enabled":
        raise ValueError("Stage2 deepstack_training_plan gate mismatch")
    scope_contract = deepstack_plan.get("scope_contract") or {}
    if scope_contract.get("schema_version") != DEEPSTACK_SCOPE_CONTRACT_SCHEMA_VERSION:
        raise ValueError("Stage2 deepstack_training_plan scope_contract schema mismatch")
    if scope_contract.get("surface") != "stage2_training":
        raise ValueError("Stage2 deepstack_training_plan scope_contract surface mismatch")
    if bool(scope_contract.get("enabled")) != enabled:
        raise ValueError("Stage2 deepstack_training_plan scope_contract enabled mismatch")
    if scope_contract.get("original_image_scope") != deepstack_plan.get(
        "original_image_scope"
    ):
        raise ValueError("Stage2 deepstack_training_plan scope_contract scope mismatch")
    if (
        scope_contract.get("original_image_deepstack")
        != deepstack_plan.get("original_image_deepstack")
    ):
        raise ValueError(
            "Stage2 deepstack_training_plan original-image scope contract mismatch"
        )
    runtime_hooks = deepstack_plan.get("runtime_hooks") or {}
    if runtime_hooks.get("schema_version") != DEEPSTACK_RUNTIME_HOOKS_SCHEMA_VERSION:
        raise ValueError("Stage2 deepstack_training_plan runtime_hooks schema mismatch")
    if runtime_hooks.get("surface") != "stage2_training":
        raise ValueError("Stage2 deepstack_training_plan runtime_hooks surface mismatch")
    if bool(runtime_hooks.get("enabled")) != enabled:
        raise ValueError("Stage2 deepstack_training_plan runtime_hooks enabled mismatch")
    if runtime_hooks.get("original_image_scope") != deepstack_plan.get(
        "original_image_scope"
    ):
        raise ValueError("Stage2 deepstack_training_plan runtime_hooks scope mismatch")
    blocking_items = list(deepstack_plan.get("blocking_items") or [])
    runtime_hook_blockers = list(runtime_hooks.get("blocking_items") or [])
    if blocking_items != runtime_hook_blockers:
        raise ValueError("Stage2 deepstack_training_plan runtime_hooks blocker mismatch")
    if enabled:
        scope = deepstack.get("original_image_scope")
        if deepstack_plan.get("original_image_scope") != scope:
            raise ValueError("Stage2 deepstack_training_plan scope mismatch")
        if deepstack_plan.get("execution_supported") is not True:
            raise ValueError("enabled Stage2 DeepStack must be execution-supported")
        if blocking_items:
            raise ValueError("enabled Stage2 DeepStack plan must not record blocking items")
    else:
        if deepstack_plan.get("execution_supported") is not True:
            raise ValueError("disabled Stage2 DeepStack plan must be execution-supported")
        if blocking_items:
            raise ValueError("disabled Stage2 DeepStack plan must not record blockers")


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
    expected_entrypoint = f"revisit_vlm_clean.training.{stage.value}_executor"
    if command.get("planned_entrypoint") != expected_entrypoint:
        raise ValueError(
            "clean_training_command planned_entrypoint mismatch: "
            f"{command.get('planned_entrypoint')!r} != {expected_entrypoint!r}"
        )
    argv = list(command.get("argv") or [])
    if "--launch-training" not in argv:
        raise ValueError("clean_training_command must include --launch-training")
    executable = bool(command.get("executable"))
    if executable:
        runtime = command.get("runtime")
        expected_status = (
            "clean_native_single_process_launch_supported"
            if runtime == "single_process"
            else "clean_native_distributed_launch_supported"
            if runtime == "distributed_torchrun"
            else None
        )
        if command.get("status") != expected_status:
            raise ValueError("executable clean_training_command status mismatch")
        if command.get("will_launch_training") is not True:
            raise ValueError("executable clean_training_command must launch training")
        if runtime not in {"single_process", "distributed_torchrun"}:
            raise ValueError(
                "executable clean_training_command must be single_process "
                "or distributed_torchrun"
            )
    else:
        if command.get("will_launch_training") is not False:
            raise ValueError("blocked clean_training_command must not launch training")
        if not command.get("unavailable_reason"):
            raise ValueError("blocked clean_training_command must explain unavailable_reason")
