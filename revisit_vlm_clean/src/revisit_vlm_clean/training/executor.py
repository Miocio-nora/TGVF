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
from revisit_vlm_clean.schema import _to_jsonable
from revisit_vlm_clean.training_plan import TRAINING_PLAN_SCHEMA_VERSION, TrainingStage


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
    return parser


def main_for_stage(stage: TrainingStage, argv: list[str] | None = None) -> int:
    args = build_parser(stage).parse_args(argv)
    report = preflight_training_plan(args.plan, expected_stage=stage)
    report_path = _preflight_report_path(
        plan_path=args.plan,
        stage=stage,
        requested=args.preflight_report,
    )
    report["preflight_report"] = str(report_path)
    _write_json(report_path, report)
    print_json(report)
    if args.preflight_only:
        return 0
    return exit_not_implemented(
        "clean-native training execution is not implemented yet; "
        "rerun with --preflight-only to validate the plan without launching"
    )


def preflight_training_plan(path: str | Path, *, expected_stage: TrainingStage) -> dict[str, Any]:
    plan_path = Path(path)
    if not plan_path.exists():
        raise FileNotFoundError(f"training plan does not exist: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _validate_training_plan(plan, expected_stage=expected_stage)
    native_status = dict(plan.get("clean_native_training") or {})
    return {
        "plan_path": str(plan_path),
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
