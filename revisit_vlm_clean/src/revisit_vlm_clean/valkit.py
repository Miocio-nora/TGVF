"""Clean ValKit/VLMEvalKit run-plan contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data_generation import file_identity
from .defaults import DEFAULT_MAX_IMAGE_RESOLUTION, DEFAULT_MODEL_ID, DEFAULT_PROTOCOL
from .schema import EvalFamily, EvalMode, ForwardMode, StrEnum, _to_jsonable
from .tgvf_protocol import SUPPORTED_PROTOCOLS

VALKIT_PLAN_SCHEMA_VERSION = "clean_valkit_plan_v1"


class ValKitRunnerStatus(StrEnum):
    PREFLIGHT_ONLY = "preflight_only"
    NOT_IMPLEMENTED = "not_implemented"


@dataclass(frozen=True)
class ValKitRunConfig:
    run_id: str
    output_dir: str
    checkpoint_path: str
    benchmarks: tuple[str, ...]
    model_id: str = DEFAULT_MODEL_ID
    processor_id: str | None = None
    mode: EvalMode = EvalMode.ORIGINAL
    max_image_resolution: int = DEFAULT_MAX_IMAGE_RESOLUTION
    tgvf_protocol: str = DEFAULT_PROTOCOL
    post_tgvf_forward_mode: ForwardMode = ForwardMode.KV_CACHE
    stage2_checkpoint: str | None = None
    valkit_root: str | None = None
    work_dir: str | None = None
    git_commit: str | None = None
    dirty_worktree: bool | None = None

    def validate(self) -> None:
        if not self.run_id:
            raise ValueError("run_id is required")
        if not self.output_dir:
            raise ValueError("output_dir is required")
        if not self.checkpoint_path:
            raise ValueError("checkpoint_path is required")
        if not self.benchmarks:
            raise ValueError("at least one ValKit benchmark name is required")
        if self.tgvf_protocol not in SUPPORTED_PROTOCOLS:
            raise ValueError(f"unsupported protocol: {self.tgvf_protocol}")
        if self.mode != EvalMode.ORIGINAL and not self.stage2_checkpoint:
            raise ValueError("TGVF ValKit modes require --stage2-checkpoint")
        if int(self.max_image_resolution) < 1:
            raise ValueError("max_image_resolution must be positive")

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


def build_valkit_plan(config: ValKitRunConfig) -> dict[str, Any]:
    config.validate()
    checkpoint = file_identity(config.checkpoint_path)
    if not checkpoint.exists:
        raise FileNotFoundError(f"checkpoint_path does not exist: {config.checkpoint_path}")
    stage2_checkpoint = None
    if config.stage2_checkpoint:
        stage2_checkpoint = file_identity(config.stage2_checkpoint)
        if not stage2_checkpoint.exists:
            raise FileNotFoundError(f"stage2_checkpoint does not exist: {config.stage2_checkpoint}")
    return {
        "valkit_plan_schema_version": VALKIT_PLAN_SCHEMA_VERSION,
        "eval_family": EvalFamily.VALKIT.value,
        "run_id": config.run_id,
        "output_dir": config.output_dir,
        "git_commit": config.git_commit,
        "dirty_worktree": config.dirty_worktree,
        "model": {
            "model_id": config.model_id,
            "processor_id": config.processor_id,
            "checkpoint": checkpoint.to_dict(),
            "stage2_checkpoint": None if stage2_checkpoint is None else stage2_checkpoint.to_dict(),
        },
        "run": {
            "benchmarks": list(config.benchmarks),
            "mode": config.mode.value,
            "max_image_resolution": config.max_image_resolution,
            "tgvf_protocol": config.tgvf_protocol,
            "post_tgvf_forward_mode": config.post_tgvf_forward_mode.value,
            "valkit_root": _path_identity(config.valkit_root),
            "work_dir": config.work_dir,
        },
        "runner": {
            "status": ValKitRunnerStatus.NOT_IMPLEMENTED.value,
            "executable": False,
            "final_clean_surface": True,
            "legacy_shell_wrapper_allowed": False,
            "unavailable_reason": (
                "clean ValKit execution is not ported yet; this command records "
                "identity only and must not shell out to historical wrappers"
            ),
        },
    }


def build_valkit_preflight_report(plan: dict[str, Any]) -> dict[str, Any]:
    _validate_valkit_plan(plan)
    runner = plan.get("runner") or {}
    return {
        "valkit_plan_schema_version": plan.get("valkit_plan_schema_version"),
        "eval_family": plan.get("eval_family"),
        "run_id": plan.get("run_id"),
        "output_dir": plan.get("output_dir"),
        "plan_valid": True,
        "runner_executable": bool(runner.get("executable")),
        "runner_status": runner.get("status"),
        "will_launch_valkit": False,
        "benchmarks": list((plan.get("run") or {}).get("benchmarks") or []),
        "mode": (plan.get("run") or {}).get("mode"),
        "unavailable_reason": runner.get("unavailable_reason"),
    }


def write_valkit_plan(output_dir: str | Path, plan: dict[str, Any]) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "valkit_plan.json"
    text_path = out / "valkit_plan.txt"
    report_path = out / "valkit_preflight_report.json"
    report = build_valkit_preflight_report(plan)
    _write_json(plan_path, plan)
    _write_json(report_path, report)
    text_path.write_text(_valkit_plan_text(plan, report), encoding="utf-8")
    return {
        "output_dir": str(out),
        "valkit_plan": str(plan_path),
        "valkit_plan_txt": str(text_path),
        "valkit_preflight_report": str(report_path),
    }


def _validate_valkit_plan(plan: dict[str, Any]) -> None:
    if plan.get("valkit_plan_schema_version") != VALKIT_PLAN_SCHEMA_VERSION:
        raise ValueError("invalid ValKit plan schema version")
    if plan.get("eval_family") != EvalFamily.VALKIT.value:
        raise ValueError("ValKit plan must use eval_family='valkit'")
    model = plan.get("model") or {}
    checkpoint = model.get("checkpoint") or {}
    if checkpoint.get("exists") is not True:
        raise ValueError("ValKit checkpoint identity must exist")
    runner = plan.get("runner") or {}
    if runner.get("legacy_shell_wrapper_allowed") is not False:
        raise ValueError("ValKit clean plan must not allow historical shell wrappers")
    if runner.get("final_clean_surface") is not True:
        raise ValueError("ValKit clean plan must be marked final_clean_surface=true")
    benchmarks = (plan.get("run") or {}).get("benchmarks") or []
    if not benchmarks:
        raise ValueError("ValKit plan must include at least one benchmark")


def _path_identity(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = Path(path)
    return {
        "path": str(resolved),
        "exists": resolved.exists(),
        "kind": "directory" if resolved.is_dir() else "file" if resolved.is_file() else "missing",
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _valkit_plan_text(plan: dict[str, Any], report: dict[str, Any]) -> str:
    run = plan.get("run") or {}
    model = plan.get("model") or {}
    runner = plan.get("runner") or {}
    lines = [
        f"run_id: {plan.get('run_id')}",
        f"schema: {plan.get('valkit_plan_schema_version')}",
        f"eval_family: {plan.get('eval_family')}",
        f"model_id: {model.get('model_id')}",
        f"checkpoint: {(model.get('checkpoint') or {}).get('path')}",
        f"mode: {run.get('mode')}",
        f"benchmarks: {', '.join(run.get('benchmarks') or [])}",
        f"runner_executable: {runner.get('executable')}",
        f"runner_status: {runner.get('status')}",
        f"will_launch_valkit: {report.get('will_launch_valkit')}",
    ]
    if runner.get("unavailable_reason"):
        lines.append(f"unavailable_reason: {runner['unavailable_reason']}")
    return "\n".join(lines) + "\n"
