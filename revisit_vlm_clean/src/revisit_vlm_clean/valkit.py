"""Clean ValKit/VLMEvalKit run-plan contracts."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
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
    READY_TO_EXECUTE = "clean_valkit_ready_to_execute"
    COMPLETED = "clean_valkit_execution_completed"
    FAILED = "clean_valkit_execution_failed"


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
    valkit_model_name: str | None = None
    valkit_run_mode: str = "all"
    reuse: bool = False
    verbose: bool = False
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
        if self.valkit_run_mode not in {"all", "infer", "eval"}:
            raise ValueError("valkit_run_mode must be one of: all, infer, eval")

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
    root_identity = _path_identity(config.valkit_root)
    runner_executable = bool(
        root_identity
        and root_identity.get("exists") is True
        and root_identity.get("kind") == "directory"
        and config.valkit_model_name
    )
    runner_status = (
        ValKitRunnerStatus.READY_TO_EXECUTE.value
        if runner_executable
        else ValKitRunnerStatus.NOT_IMPLEMENTED.value
    )
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
            "valkit_root": root_identity,
            "valkit_model_name": config.valkit_model_name,
            "valkit_run_mode": config.valkit_run_mode,
            "reuse": config.reuse,
            "verbose": config.verbose,
            "work_dir": config.work_dir,
        },
        "runner": {
            "status": runner_status,
            "executable": runner_executable,
            "final_clean_surface": True,
            "legacy_shell_wrapper_allowed": False,
            "prepare_execution_supported": True,
            "execute_supported": True,
            "unavailable_reason": (
                None
                if runner_executable
                else "clean ValKit execution requires --valkit-root and --valkit-model-name; "
                "it must not shell out to historical wrappers"
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


def build_valkit_execution_bundle(
    plan: dict[str, Any],
    *,
    launch_command: dict[str, Any] | None = None,
    will_launch: bool = False,
) -> dict[str, Any]:
    _validate_valkit_plan(plan)
    runner = plan.get("runner") or {}
    runner_status = (
        ValKitRunnerStatus.READY_TO_EXECUTE.value
        if will_launch
        else ValKitRunnerStatus.HANDOFF_SUPPORTED_RUNNER_NOT_PORTED.value
    )
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
        "launch_command": launch_command,
        "runner": {
            "status": runner_status,
            "plan_runner_status": runner.get("status"),
            "will_launch_valkit": will_launch,
            "valkit_runtime_ported": will_launch,
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
        "returncode": runner.get("returncode"),
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


def execute_valkit_plan(output_dir: str | Path, plan: dict[str, Any]) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    launch_command = build_valkit_launch_command(plan)
    bundle = build_valkit_execution_bundle(plan, launch_command=launch_command, will_launch=True)
    bundle_path = out / "valkit_execution_bundle.json"
    status_path = out / "valkit_execution_status.json"
    text_path = out / "valkit_execution_bundle.txt"
    command_path = out / "valkit_launch_command.sh"
    stdout_path = out / "valkit_stdout.log"
    stderr_path = out / "valkit_stderr.log"
    result_path = out / "valkit_execution_result.json"
    command_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + launch_command["shell"] + "\n",
        encoding="utf-8",
    )
    command_path.chmod(0o755)
    started_at = time.time()
    completed = subprocess.run(
        launch_command["argv"],
        cwd=launch_command["cwd"],
        env=_valkit_subprocess_env(launch_command),
        text=True,
        capture_output=True,
        check=False,
    )
    finished_at = time.time()
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    result = {
        "schema_version": "clean_valkit_execution_result_v1",
        "run_id": plan.get("run_id"),
        "returncode": completed.returncode,
        "started_at_unix": started_at,
        "finished_at_unix": finished_at,
        "elapsed_sec": finished_at - started_at,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "launch_command": launch_command,
    }
    _write_json(result_path, result)
    runner = bundle["runner"]
    runner["status"] = (
        ValKitRunnerStatus.COMPLETED.value
        if completed.returncode == 0
        else ValKitRunnerStatus.FAILED.value
    )
    runner["returncode"] = completed.returncode
    runner["stdout_log"] = str(stdout_path)
    runner["stderr_log"] = str(stderr_path)
    runner["result"] = str(result_path)
    status = build_valkit_execution_status(bundle)
    _write_json(bundle_path, bundle)
    _write_json(status_path, status)
    text_path.write_text(_valkit_execution_bundle_text(bundle, status), encoding="utf-8")
    return {
        "execution_dir": str(out),
        "valkit_execution_bundle": str(bundle_path),
        "valkit_execution_status": str(status_path),
        "valkit_execution_bundle_txt": str(text_path),
        "valkit_launch_command": str(command_path),
        "valkit_execution_result": str(result_path),
        "valkit_stdout": str(stdout_path),
        "valkit_stderr": str(stderr_path),
        "returncode": completed.returncode,
        "runner_status": runner["status"],
        "will_launch_valkit": True,
    }


def build_valkit_launch_command(plan: dict[str, Any]) -> dict[str, Any]:
    _validate_valkit_plan(plan)
    run = plan.get("run") or {}
    root = _validated_valkit_root(run)
    model_name = run.get("valkit_model_name")
    if not model_name:
        raise ValueError("--valkit-model-name is required for --execute")
    benchmarks = [str(item) for item in run.get("benchmarks") or []]
    work_dir = str(run.get("work_dir") or (Path(str(plan.get("output_dir"))) / "valkit_work"))
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable,
        str(root / "run.py"),
        "--data",
        *benchmarks,
        "--model",
        str(model_name),
        "--work-dir",
        work_dir,
        "--mode",
        str(run.get("valkit_run_mode") or "all"),
    ]
    if run.get("reuse"):
        argv.append("--reuse")
    if run.get("verbose"):
        argv.append("--verbose")
    return {
        "schema_version": "clean_valkit_launch_command_v1",
        "argv": argv,
        "shell": shlex.join(argv),
        "cwd": str(root),
        "env_delta": {
            "PYTHONPATH_prepend": str(root),
        },
        "legacy_shell_wrapper_allowed": False,
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


def _validated_valkit_root(run: dict[str, Any]) -> Path:
    root_identity = run.get("valkit_root") or {}
    root_path = root_identity.get("path")
    if not root_path:
        raise ValueError("--valkit-root is required for --execute")
    root = Path(str(root_path))
    if not root.exists() or not root.is_dir():
        raise ValueError(f"ValKit root does not exist or is not a directory: {root}")
    run_py = root / "run.py"
    if not run_py.exists() or not run_py.is_file():
        raise ValueError(f"ValKit root is missing run.py: {run_py}")
    return root


def _valkit_subprocess_env(launch_command: dict[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    prepend = (launch_command.get("env_delta") or {}).get("PYTHONPATH_prepend")
    if prepend:
        current = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(prepend) if not current else f"{prepend}{os.pathsep}{current}"
    return env


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
    if run.get("valkit_model_name"):
        argv.extend(["--valkit-model-name", str(run["valkit_model_name"])])
    if run.get("valkit_run_mode"):
        argv.extend(["--valkit-run-mode", str(run["valkit_run_mode"])])
    if run.get("reuse"):
        argv.append("--reuse")
    if run.get("verbose"):
        argv.append("--verbose")
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
        f"valkit_runtime_ported: {status.get('valkit_runtime_ported')}",
        f"returncode: {status.get('returncode')}",
        f"legacy_shell_wrapper_allowed: {runner.get('legacy_shell_wrapper_allowed')}",
    ]
    if runner.get("unavailable_reason"):
        lines.append(f"unavailable_reason: {runner['unavailable_reason']}")
    return "\n".join(lines) + "\n"
