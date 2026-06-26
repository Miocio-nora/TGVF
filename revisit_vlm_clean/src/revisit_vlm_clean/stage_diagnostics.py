"""Clean internal diagnostics for Stage1-style TGVF readout quality."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .defaults import DEFAULT_MAX_IMAGE_RESOLUTION, DEFAULT_MODEL_ID, DEFAULT_PROTOCOL
from .schema import _to_jsonable

DIAGNOSTIC_TASKS = ("readout", "query", "distribution")


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
    use_fvt_cache: bool = True
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
        if self.wandb_log_eval and not self.wandb_project:
            raise ValueError("--wandb-log-eval requires --wandb-project")

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


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
    env = stage_diagnostic_environment(config)
    command = stage_diagnostic_command(config)
    return {
        "schema_version": "clean_stage_diagnostic_plan_v1",
        "run_id": config.run_id,
        "eval_family": "internal_diagnostic",
        "stage": config.stage,
        "diagnostic_kind": "stage1_style_fvt_readout_regression",
        "tasks": list(config.tasks),
        "checkpoint": config.checkpoint,
        "eval_jsonl": config.eval_jsonl,
        "output_dir": config.output_dir,
        "model_id": config.model_id,
        "processor_id": config.processor_id,
        "protocol": config.protocol,
        "focus_action_im_end": config.focus_action_im_end,
        "max_image_resolution": config.max_image_resolution,
        "fvt_position_mode": config.fvt_position_mode,
        "use_fvt_cache": config.use_fvt_cache,
        "wandb_log_eval": config.wandb_log_eval,
        "git_commit": config.git_commit,
        "dirty_worktree": config.dirty_worktree,
        "legacy_eval_suite": {
            "script": "eval/run_tgvf_v3_eval_suite.sh",
            "reason": (
                "readout/query/distribution are preserved legacy internal "
                "diagnostics with established metric names"
            ),
        },
        "command": command,
        "environment": env,
        "expected_reports": expected_stage_diagnostic_reports(config.output_dir, config.tasks),
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
    output_dir = Path(str(plan["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "clean_stage_diagnostic_status.json"
    started = {
        "schema_version": "clean_stage_diagnostic_status_v1",
        "status": "running",
        "run_id": plan["run_id"],
        "stage": plan["stage"],
        "tasks": plan["tasks"],
        "plan_path": str(output_dir / "clean_stage_diagnostic_plan.json"),
    }
    _write_json(status_path, started)
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in (plan.get("environment") or {}).items()})
    command = [str(item) for item in plan["command"]]
    result = subprocess.run(command, cwd=str(cwd or _repo_root()), env=env, check=False)
    finished = {
        **started,
        "status": "completed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "reports": report_statuses(plan),
    }
    _write_json(status_path, finished)
    if result.returncode != 0:
        raise RuntimeError(f"stage diagnostics failed with return code {result.returncode}")
    return finished


def stage_diagnostic_command(config: StageDiagnosticConfig) -> list[str]:
    tasks = ",".join(config.tasks)
    return ["bash", "eval/run_tgvf_v3_eval_suite.sh", config.checkpoint, tasks]


def stage_diagnostic_environment(config: StageDiagnosticConfig) -> dict[str, str]:
    env = {
        "RUN_ID": config.run_id,
        "EVAL_JSONL": config.eval_jsonl,
        "MODEL_ID": config.model_id,
        "TGVF_PROTOCOL": config.protocol,
        "FOCUS_ACTION_IM_END": "1" if config.focus_action_im_end else "0",
        "DEVICE": config.device,
        "DEVICE_MAP": config.device_map,
        "DTYPE": config.dtype,
        "ATTN_IMPL": config.attn_implementation,
        "VARIANT": config.variant,
        "NUM_FVT": str(config.num_foveated_tokens),
        "ENCODER_ADAPTER_TYPE": config.encoder_adapter_type,
        "MAX_IMAGE_RESOLUTION": str(config.max_image_resolution),
        "FVT_POSITION_MODE": config.fvt_position_mode,
        "OUT_ROOT": config.output_dir,
        "READOUT_MAX_SAMPLES": str(config.readout_max_samples),
        "DISTRIBUTION_MAX_SAMPLES": str(config.distribution_max_samples),
        "QUERY_MAX_GROUPS": str(config.query_max_groups),
        "QUERY_REQUIRE_GROUPS": str(config.query_require_groups),
        "QUERY_MIN_TARGETS_PER_IMAGE": str(config.query_min_targets_per_image),
        "EVAL_WORKERS": str(config.eval_workers),
        "USE_FVT_CACHE": "1" if config.use_fvt_cache else "0",
        "SEED": str(config.seed),
        "WANDB_LOG_EVAL": "1" if config.wandb_log_eval else "0",
    }
    optional = {
        "PROCESSOR_ID": config.processor_id,
        "WANDB_PROJECT": config.wandb_project,
        "WANDB_MODE": config.wandb_mode,
        "WANDB_RUN_NAME": config.wandb_run_name,
        "WANDB_GROUP": config.wandb_group,
        "WANDB_TAGS": config.wandb_tags,
    }
    for key, value in optional.items():
        if value:
            env[key] = str(value)
    return env


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
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]
