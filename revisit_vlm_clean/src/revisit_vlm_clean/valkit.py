"""Clean ValKit/VLMEvalKit run-plan contracts."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .data_generation import file_identity
from .defaults import DEFAULT_MAX_IMAGE_RESOLUTION, DEFAULT_MODEL_ID, DEFAULT_PROTOCOL
from .schema import EvalFamily, EvalMode, ForwardMode, StrEnum, _to_jsonable
from .tgvf_protocol import SUPPORTED_PROTOCOLS

VALKIT_PLAN_SCHEMA_VERSION = "clean_valkit_plan_v1"
VALKIT_EXECUTION_BUNDLE_SCHEMA_VERSION = "clean_valkit_execution_bundle_v1"


class ValKitRunnerStatus(StrEnum):
    PREFLIGHT_ONLY = "preflight_only"
    NOT_IMPLEMENTED = "not_implemented"
    HANDOFF_SUPPORTED_RUNNER_NOT_PORTED = "handoff_supported_valkit_runner_not_ported"


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
            "prepare_execution_supported": True,
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
    command_path = out / "valkit_prepare_execution_command.sh"
    report = build_valkit_preflight_report(plan)
    _write_json(plan_path, plan)
    _write_json(report_path, report)
    text_path.write_text(_valkit_plan_text(plan, report), encoding="utf-8")
    command_path.write_text(_valkit_prepare_execution_command_text(plan), encoding="utf-8")
    command_path.chmod(0o755)
    return {
        "output_dir": str(out),
        "valkit_plan": str(plan_path),
        "valkit_plan_txt": str(text_path),
        "valkit_preflight_report": str(report_path),
        "valkit_prepare_execution_command": str(command_path),
    }


def build_valkit_execution_bundle(plan: dict[str, Any]) -> dict[str, Any]:
    _validate_valkit_plan(plan)
    runner = plan.get("runner") or {}
    return {
        "valkit_execution_bundle_schema_version": VALKIT_EXECUTION_BUNDLE_SCHEMA_VERSION,
        "valkit_plan_schema_version": plan.get("valkit_plan_schema_version"),
        "eval_family": plan.get("eval_family"),
        "run_id": plan.get("run_id"),
        "output_dir": plan.get("output_dir"),
        "plan_sha256": _payload_sha256(plan),
        "git_commit": plan.get("git_commit"),
        "dirty_worktree": plan.get("dirty_worktree"),
        "model": plan.get("model"),
        "run": plan.get("run"),
        "runner": {
            "status": ValKitRunnerStatus.HANDOFF_SUPPORTED_RUNNER_NOT_PORTED.value,
            "plan_runner_status": runner.get("status"),
            "will_launch_valkit": False,
            "valkit_runtime_ported": False,
            "legacy_shell_wrapper_allowed": False,
            "final_clean_surface": True,
            "unavailable_reason": runner.get("unavailable_reason"),
        },
    }


def build_valkit_execution_status(bundle: dict[str, Any]) -> dict[str, Any]:
    runner = bundle.get("runner") or {}
    run = bundle.get("run") or {}
    return {
        "valkit_execution_bundle_schema_version": bundle.get(
            "valkit_execution_bundle_schema_version"
        ),
        "eval_family": bundle.get("eval_family"),
        "run_id": bundle.get("run_id"),
        "plan_sha256": bundle.get("plan_sha256"),
        "runner_status": runner.get("status"),
        "will_launch_valkit": runner.get("will_launch_valkit"),
        "valkit_runtime_ported": runner.get("valkit_runtime_ported"),
        "legacy_shell_wrapper_allowed": runner.get("legacy_shell_wrapper_allowed"),
        "benchmarks": list(run.get("benchmarks") or []),
        "mode": run.get("mode"),
        "unavailable_reason": runner.get("unavailable_reason"),
    }


def write_valkit_execution_bundle(
    output_dir: str | Path, plan: dict[str, Any]
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle = build_valkit_execution_bundle(plan)
    status = build_valkit_execution_status(bundle)
    bundle_path = out / "valkit_execution_bundle.json"
    status_path = out / "valkit_execution_status.json"
    text_path = out / "valkit_execution_bundle.txt"
    _write_json(bundle_path, bundle)
    _write_json(status_path, status)
    text_path.write_text(_valkit_execution_bundle_text(bundle, status), encoding="utf-8")
    return {
        "execution_dir": str(out),
        "valkit_execution_bundle": str(bundle_path),
        "valkit_execution_status": str(status_path),
        "valkit_execution_bundle_txt": str(text_path),
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
    if runner.get("prepare_execution_supported") is not True:
        raise ValueError("ValKit clean plan must support prepare-execution handoff")
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


def _payload_sha256(payload: Any) -> str:
    blob = json.dumps(_to_jsonable(payload), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return sha256(blob).hexdigest()


def _valkit_prepare_execution_command_text(plan: dict[str, Any]) -> str:
    run = plan.get("run") or {}
    model = plan.get("model") or {}
    checkpoint = (model.get("checkpoint") or {}).get("path")
    stage2_checkpoint = model.get("stage2_checkpoint")
    argv = [
        "python",
        "-m",
        "revisit_vlm_clean.cli.valkit",
        "--run-id",
        str(plan.get("run_id")),
        "--checkpoint-path",
        str(checkpoint),
        "--output-dir",
        str(plan.get("output_dir")),
        "--model-id",
        str(model.get("model_id")),
        "--mode",
        str(run.get("mode")),
        "--max-image-resolution",
        str(run.get("max_image_resolution")),
        "--tgvf-protocol",
        str(run.get("tgvf_protocol")),
        "--post-tgvf-forward-mode",
        str(run.get("post_tgvf_forward_mode")),
    ]
    if model.get("processor_id"):
        argv.extend(["--processor-id", str(model["processor_id"])])
    if stage2_checkpoint:
        argv.extend(["--stage2-checkpoint", str(stage2_checkpoint.get("path"))])
    valkit_root = run.get("valkit_root")
    if valkit_root:
        argv.extend(["--valkit-root", str(valkit_root.get("path"))])
    if run.get("work_dir"):
        argv.extend(["--work-dir", str(run["work_dir"])])
    for benchmark in run.get("benchmarks") or []:
        argv.extend(["--benchmark", str(benchmark)])
    argv.append("--prepare-execution")
    command = " ".join(shlex.quote(item) for item in argv)
    return "#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n"


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


def _valkit_execution_bundle_text(bundle: dict[str, Any], status: dict[str, Any]) -> str:
    run = bundle.get("run") or {}
    runner = bundle.get("runner") or {}
    lines = [
        f"run_id: {bundle.get('run_id')}",
        f"schema: {bundle.get('valkit_execution_bundle_schema_version')}",
        f"eval_family: {bundle.get('eval_family')}",
        f"plan_sha256: {bundle.get('plan_sha256')}",
        f"mode: {run.get('mode')}",
        f"benchmarks: {', '.join(run.get('benchmarks') or [])}",
        f"runner_status: {runner.get('status')}",
        f"will_launch_valkit: {status.get('will_launch_valkit')}",
        f"legacy_shell_wrapper_allowed: {runner.get('legacy_shell_wrapper_allowed')}",
    ]
    if runner.get("unavailable_reason"):
        lines.append(f"unavailable_reason: {runner['unavailable_reason']}")
    return "\n".join(lines) + "\n"
