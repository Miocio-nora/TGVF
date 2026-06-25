"""Clean data-generation identity contracts.

This module intentionally starts with identity-only planning. It records the
inputs and intended transforms for teacher/Stage1/Stage2 data generation before
the heavy generators are ported into the clean tree.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .defaults import DEFAULT_PROTOCOL
from .schema import StrEnum, _to_jsonable
from .tgvf_protocol import SUPPORTED_PROTOCOLS


class DataGenerationStage(StrEnum):
    TEACHER_TRAJECTORY = "teacher_trajectory"
    STAGE1_PROTOCOL_C_FOCUS = "stage1_protocol_c_focus"
    STAGE2_PROTOCOL_C = "stage2_protocol_c"
    CLEAN_SPLITS = "clean_splits"
    CHOICE_TO_OPEN_ANSWER = "choice_to_open_answer"


class DataGenerationTransform(StrEnum):
    NONE = "none"
    V4_TO_PROTOCOL_C = "v4_to_protocol_c"
    CLEAN_IMEND = "clean_imend"
    CHOICE_TO_OPEN_ANSWER = "choice_to_open_answer"


@dataclass(frozen=True)
class FileIdentity:
    path: str
    exists: bool
    size_bytes: int | None = None
    sha256: str | None = None
    line_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class DataGenerationConfig:
    run_id: str
    stage: DataGenerationStage
    output_dir: str
    input_root: str = "."
    input_files: tuple[str, ...] = ()
    protocol: str = DEFAULT_PROTOCOL
    transform: DataGenerationTransform = DataGenerationTransform.NONE
    source_manifest_path: str | None = None
    source_manifest_hash: str | None = None
    source_run_id: str | None = None
    split_policy: str = "preserve_input"
    field_weights: dict[str, float] = field(default_factory=dict)
    mask_policy: dict[str, Any] = field(default_factory=dict)
    git_commit: str | None = None
    dirty_worktree: bool | None = None

    def validate(self) -> None:
        if not self.run_id:
            raise ValueError("run_id is required")
        if not self.output_dir:
            raise ValueError("output_dir is required")
        if self.protocol not in SUPPORTED_PROTOCOLS:
            raise ValueError(f"unsupported protocol: {self.protocol}")
        if self.source_manifest_hash and not self.source_manifest_path:
            raise ValueError("source_manifest_hash requires source_manifest_path")

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


def build_data_generation_plan(config: DataGenerationConfig) -> dict[str, Any]:
    config.validate()
    identities = [
        file_identity(_resolve_input_path(config.input_root, rel))
        for rel in config.input_files
    ]
    missing = [item.path for item in identities if not item.exists]
    if missing:
        raise FileNotFoundError(f"missing data-generation input file(s): {missing}")
    manifest_identity = None
    if config.source_manifest_path:
        manifest_identity = file_identity(Path(config.source_manifest_path))
        if not manifest_identity.exists:
            raise FileNotFoundError(f"source_manifest_path does not exist: {config.source_manifest_path}")
    return {
        "config": config.to_dict(),
        "input_files": [item.to_dict() for item in identities],
        "source_manifest": None if manifest_identity is None else manifest_identity.to_dict(),
        "summary": {
            "stage": str(config.stage),
            "transform": str(config.transform),
            "n_input_files": len(identities),
            "total_input_lines": sum(item.line_count or 0 for item in identities),
            "identity_only": True,
            "generated_data_written": False,
        },
    }


def write_data_generation_plan(output_dir: str | Path, *, config: DataGenerationConfig) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan = build_data_generation_plan(config)

    config_path = out / "data_generation_config.json"
    config_txt_path = out / "data_generation_config.txt"
    inputs_path = out / "input_files.json"
    report_path = out / "data_generation_report.json"

    _write_json(config_path, plan["config"])
    _write_json(inputs_path, plan["input_files"])
    _write_json(report_path, plan)
    config_txt_path.write_text(_config_text(plan), encoding="utf-8")
    return {
        "output_dir": str(out),
        "data_generation_config": str(config_path),
        "data_generation_config_txt": str(config_txt_path),
        "input_files": str(inputs_path),
        "report": str(report_path),
    }


def file_identity(path: str | Path) -> FileIdentity:
    resolved = Path(path)
    if not resolved.exists():
        return FileIdentity(path=str(resolved), exists=False)
    digest = hashlib.sha256()
    line_count = 0
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            line_count += chunk.count(b"\n")
    return FileIdentity(
        path=str(resolved),
        exists=True,
        size_bytes=resolved.stat().st_size,
        sha256=digest.hexdigest(),
        line_count=line_count,
    )


def _resolve_input_path(input_root: str, rel: str) -> Path:
    path = Path(rel)
    return path if path.is_absolute() else Path(input_root) / path


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _config_text(plan: dict[str, Any]) -> str:
    config = plan["config"]
    summary = plan["summary"]
    lines = [
        f"run_id: {config['run_id']}",
        f"stage: {config['stage']}",
        f"transform: {config['transform']}",
        f"protocol: {config['protocol']}",
        f"source_run_id: {config.get('source_run_id')}",
        f"source_manifest_path: {config.get('source_manifest_path')}",
        f"source_manifest_hash: {config.get('source_manifest_hash')}",
        f"input_root: {config['input_root']}",
        f"input_files: {', '.join(config['input_files'])}",
        f"split_policy: {config['split_policy']}",
        f"total_input_lines: {summary['total_input_lines']}",
        "identity_only: true",
        "generated_data_written: false",
    ]
    return "\n".join(lines) + "\n"
