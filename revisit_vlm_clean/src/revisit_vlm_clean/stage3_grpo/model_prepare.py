"""Prepare local Stage3 GRPO judge model checkpoints."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from revisit_vlm_clean.data_generation import file_identity
from revisit_vlm_clean.schema import _to_jsonable

from .judge_runner import (
    DEFAULT_MODEL_ROOT,
    MODEL_PRESETS,
    local_model_identity,
    model_download_commands,
    resolve_local_or_remote_model_id,
)
from .schemas import now_iso, write_json

JUDGE_MODEL_PREPARE_SCHEMA_VERSION = "stage3_grpo_judge_model_prepare_v0"

JUDGE_MODEL_TARGET_SETS = {
    "stage3_qwen3_recommended": (
        "qwen3_vl_32b_thinking",
        "qwen3_vl_235b_a22b_thinking_fp8",
    ),
    "stage3_qwen3_bf16": (
        "qwen3_vl_32b_thinking",
        "qwen3_vl_235b_a22b_thinking",
    ),
    "stage3_72b_fallback": (
        "qwen3_vl_32b_thinking",
        "qwen25_vl_72b_instruct",
    ),
    "local_smoke": ("qwen3_vl_8b_thinking",),
}


@dataclass(frozen=True)
class JudgeModelPrepareConfig:
    output_dir: str
    model_root: str = DEFAULT_MODEL_ROOT
    targets: tuple[str, ...] = field(default_factory=tuple)
    target_sets: tuple[str, ...] = ("stage3_qwen3_recommended",)
    revision: str | None = None
    max_workers: int = 8
    execute_download: bool = False
    allow_large_download: bool = False
    local_files_only: bool = False
    remote_metadata: bool = False

    def validate(self) -> None:
        if not self.output_dir:
            raise ValueError("output_dir is required")
        if int(self.max_workers) < 1:
            raise ValueError("max_workers must be >= 1")
        for target_set in self.target_sets:
            if target_set not in JUDGE_MODEL_TARGET_SETS:
                raise ValueError(f"unknown judge model target set: {target_set}")
        if self.execute_download and not self.allow_large_download:
            raise ValueError("--execute-download requires --allow-large-download")

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


def run_judge_model_prepare(
    config: JudgeModelPrepareConfig,
    *,
    preflight_only: bool = False,
    write_plan_only: bool = False,
) -> dict[str, Any]:
    config.validate()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = expand_model_targets(config.targets, config.target_sets)
    plan = build_judge_model_prepare_plan(config, targets=targets)
    plan_path = output_dir / "stage3_grpo_judge_model_prepare_plan.json"
    report_path = output_dir / "stage3_grpo_judge_model_prepare_report.json"
    ledger_path = output_dir / "stage3_grpo_judge_model_prepare_ledger.jsonl"
    write_json(plan_path, plan)
    append_ledger(
        ledger_path,
        {
            "event": "plan_written",
            "targets": targets,
            "plan_path": str(plan_path),
            "created_at": now_iso(),
        },
    )
    if write_plan_only or preflight_only:
        report = {
            **plan,
            "status": "passed" if not plan["missing_targets"] else "missing_models",
            "preflight_only": preflight_only,
            "write_plan_only": write_plan_only,
            "plan_path": str(plan_path),
            "report_path": str(report_path),
            "ledger_path": str(ledger_path),
        }
        write_json(report_path, report)
        return report
    downloads: list[dict[str, Any]] = []
    if config.execute_download:
        for target in plan["targets"]:
            if target["status"] == "ready":
                continue
            downloads.append(download_target(config, target, ledger_path=ledger_path))
    final_plan = build_judge_model_prepare_plan(config, targets=targets)
    report = {
        **final_plan,
        "status": "passed" if not final_plan["missing_targets"] else "missing_models",
        "plan_path": str(plan_path),
        "report_path": str(report_path),
        "ledger_path": str(ledger_path),
        "downloads": downloads,
    }
    write_json(report_path, report)
    return report


def build_judge_model_prepare_plan(
    config: JudgeModelPrepareConfig,
    *,
    targets: list[str],
) -> dict[str, Any]:
    rows = [target_status(target, config=config) for target in targets]
    missing = [row for row in rows if row["status"] != "ready"]
    return {
        "schema_version": JUDGE_MODEL_PREPARE_SCHEMA_VERSION,
        "created_at": now_iso(),
        "config": config.to_dict(),
        "model_root_identity": path_identity(config.model_root),
        "targets": rows,
        "ready_targets": [row["target"] for row in rows if row["status"] == "ready"],
        "missing_targets": [row["target"] for row in missing],
        "target_count": len(rows),
        "ready_count": len(rows) - len(missing),
        "missing_count": len(missing),
        "notes": [
            "Qwen3-VL currently has 32B and 235B-A22B large judge options; "
            "Qwen2.5-VL/QVQ 72B presets are kept as 70B-class fallbacks.",
            "No large checkpoint is downloaded unless execute_download and allow_large_download are both true.",
        ],
    }


def target_status(target: str, *, config: JudgeModelPrepareConfig) -> dict[str, Any]:
    hf_id = MODEL_PRESETS.get(target, target)
    resolved = resolve_local_or_remote_model_id(
        hf_id,
        model_root=config.model_root,
        require_local_model=True,
    )
    local_dir = local_dir_for_model(hf_id, model_root=config.model_root)
    row = {
        "target": target,
        "hf_id": hf_id,
        "status": "ready" if resolved["status"] == "ready" else "missing",
        "resolved_model_id": resolved.get("resolved_model_id"),
        "local_dir": str(local_dir),
        "local_identity": local_model_identity(local_dir),
        "errors": list(resolved.get("errors") or []),
    }
    row["download_commands"] = (
        [] if row["status"] == "ready" else model_download_commands(hf_id, model_root=config.model_root)
    )
    if config.remote_metadata:
        row["remote_metadata"] = remote_model_metadata(hf_id)
    return row


def download_target(
    config: JudgeModelPrepareConfig,
    target: dict[str, Any],
    *,
    ledger_path: Path,
) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    hf_id = str(target["hf_id"])
    local_dir = Path(str(target["local_dir"]))
    append_ledger(
        ledger_path,
        {"event": "download_started", "hf_id": hf_id, "local_dir": str(local_dir), "created_at": now_iso()},
    )
    started = time.perf_counter()
    kwargs: dict[str, Any] = {
        "repo_id": hf_id,
        "local_dir": str(local_dir),
        "revision": config.revision,
        "local_files_only": config.local_files_only,
        "max_workers": int(config.max_workers),
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    try:
        downloaded = snapshot_download(**kwargs)
        result = {
            "status": "downloaded",
            "hf_id": hf_id,
            "local_dir": str(local_dir),
            "snapshot_path": str(downloaded),
            "wall_time_sec": time.perf_counter() - started,
        }
        append_ledger(ledger_path, {"event": "download_finished", **result, "created_at": now_iso()})
        return result
    except Exception as exc:
        result = {
            "status": "failed",
            "hf_id": hf_id,
            "local_dir": str(local_dir),
            "error": f"{type(exc).__name__}: {exc}",
            "wall_time_sec": time.perf_counter() - started,
        }
        append_ledger(ledger_path, {"event": "download_failed", **result, "created_at": now_iso()})
        return result


def expand_model_targets(targets: tuple[str, ...], target_sets: tuple[str, ...]) -> list[str]:
    ordered: list[str] = []
    for target_set in target_sets:
        ordered.extend(JUDGE_MODEL_TARGET_SETS[target_set])
    ordered.extend(targets)
    out: list[str] = []
    seen: set[str] = set()
    for target in ordered:
        if target not in seen:
            out.append(target)
            seen.add(target)
    return out


def local_dir_for_model(model_id: str, *, model_root: str) -> Path:
    hf_id = MODEL_PRESETS.get(model_id, model_id)
    path = Path(hf_id).expanduser()
    if path.is_absolute():
        return path
    return Path(model_root).expanduser() / hf_id.split("/")[-1]


def remote_model_metadata(model_id: str) -> dict[str, Any]:
    from huggingface_hub import HfApi

    hf_id = MODEL_PRESETS.get(model_id, model_id)
    try:
        info = HfApi().model_info(hf_id, files_metadata=True)
        siblings = getattr(info, "siblings", None) or []
        size = 0
        file_count = 0
        for sibling in siblings:
            file_count += 1
            size += int(getattr(sibling, "size", 0) or 0)
        return {
            "status": "available",
            "hf_id": hf_id,
            "sha": getattr(info, "sha", None),
            "last_modified": str(getattr(info, "last_modified", "") or ""),
            "file_count": file_count,
            "total_size_bytes": size,
        }
    except Exception as exc:
        return {"status": "unavailable", "hf_id": hf_id, "error": f"{type(exc).__name__}: {exc}"}


def append_ledger(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_to_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")


def path_identity(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"exists": False, "path": str(path), "type": "missing"}
    if path.is_file():
        payload = file_identity(path).to_dict()
        payload["type"] = "file"
        return payload
    children = list(path.iterdir()) if path.is_dir() else []
    return {
        "exists": True,
        "path": str(path),
        "type": "directory" if path.is_dir() else "other",
        "child_count": len(children),
    }
