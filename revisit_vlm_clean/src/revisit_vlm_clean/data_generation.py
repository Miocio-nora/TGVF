"""Clean data-generation identity contracts.

This module intentionally starts with identity-only planning. It records the
inputs and intended transforms for teacher/Stage1/Stage2 data generation before
the heavy generators are ported into the clean tree.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
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


TEXT_KEYS_FOR_CLEANING = {
    "question",
    "answer",
    "short_answer",
    "value_span_text",
    "target",
    "evidence_description",
    "pre_focus_think",
    "post_focus_think",
    "no_focus_think",
}

BAD_TEXT_SUBSTRINGS = (
    "'}],",
    '"}],',
    "'}]}",
    '"}]}',
    "'choices':",
    '"choices":',
    "'question':",
    '"question":',
    "'answer':",
    '"answer":',
    "metadata:",
)


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


def execute_data_generation(config: DataGenerationConfig) -> dict[str, str]:
    config.validate()
    if config.transform == DataGenerationTransform.NONE:
        raise NotImplementedError("data generation execution requires a concrete --transform")
    if config.transform == DataGenerationTransform.V4_TO_PROTOCOL_C:
        raise NotImplementedError("v4_to_protocol_c execution is not ported into the clean CLI yet")

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan = build_data_generation_plan(config)
    generated_files: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    for rel in config.input_files:
        input_path = _resolve_input_path(config.input_root, rel)
        output_path = out / rel
        if config.transform == DataGenerationTransform.CHOICE_TO_OPEN_ANSWER:
            report = _convert_choice_to_open_answer_file(input_path, output_path)
        elif config.transform == DataGenerationTransform.CLEAN_IMEND:
            report = _clean_protocol_split_file(input_path, output_path)
        else:
            raise NotImplementedError(f"transform execution is not ported: {config.transform}")
        reports[rel] = report
        generated_files.append(
            {
                "input": str(input_path),
                "output": str(output_path),
                "report": report,
            }
        )

    paths = write_data_generation_plan(out, config=config)
    generated_files_path = out / "generated_files.json"
    transform_report_path = out / "transform_report.json"
    report_path = out / "data_generation_report.json"
    _write_json(generated_files_path, generated_files)
    _write_json(transform_report_path, reports)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["summary"]["identity_only"] = False
    report["summary"]["generated_data_written"] = True
    report["summary"]["n_generated_files"] = len(generated_files)
    report["generated_files"] = generated_files
    report["transform_report"] = reports
    _write_json(report_path, report)
    return {
        **paths,
        "generated_files": str(generated_files_path),
        "transform_report": str(transform_report_path),
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


def _clean_protocol_split_file(input_path: Path, output_path: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            counts["input"] += 1
            row = json.loads(line)
            reason = _bad_text_reason(row)
            if reason:
                counts["dropped"] += 1
                counts[f"dropped_{reason}"] += 1
                continue
            counts["kept"] += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    return dict(counts)


def _bad_text_reason(value: Any, *, path: str = "") -> str:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            reason = _bad_text_reason(child, path=child_path)
            if reason:
                return reason
        return ""
    if isinstance(value, list):
        for index, child in enumerate(value):
            reason = _bad_text_reason(child, path=f"{path}[{index}]")
            if reason:
                return reason
        return ""
    if not isinstance(value, str):
        return ""
    if path.split(".")[-1] not in TEXT_KEYS_FOR_CLEANING and not path.endswith(".text") and not path.endswith(".focus_text"):
        return ""
    text = value.strip()
    if not text:
        return ""
    for bad in BAD_TEXT_SUBSTRINGS:
        if bad in text:
            return f"bad_text_{path or 'text'}"
    return ""


def _convert_choice_to_open_answer_file(input_path: Path, output_path: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            counts["input"] += 1
            record = json.loads(line)
            before_format = str(record.get("answer_format") or "unknown")
            counts[f"input_answer_format_{before_format}"] += 1
            if _is_choice_record(record):
                row = _convert_choice_record(record, line_no=line_no)
                counts["converted_choice"] += 1
            else:
                row = dict(record)
                counts["kept_non_choice"] += 1
            after_format = str(row.get("answer_format") or "unknown")
            counts[f"output_answer_format_{after_format}"] += 1
            if _is_choice_record(row):
                counts["output_choice_like"] += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["output"] += 1
    counts["input_choice_ratio"] = counts["converted_choice"] / counts["input"] if counts["input"] else 0.0
    counts["output_choice_ratio"] = counts["output_choice_like"] / counts["output"] if counts["output"] else 0.0
    return dict(counts)


def _convert_choice_record(record: dict[str, Any], *, line_no: int) -> dict[str, Any]:
    short_answer = _open_answer_text(record)
    if not short_answer:
        raise ValueError(f"{line_no}: choice record has no recoverable open answer")

    row = dict(record)
    original_answer = row.get("answer")
    original_choices = row.get("choices") or []
    original_answer_format = row.get("answer_format")
    original_value_span = row.get("value_span_text")

    row["question"] = _strip_answer_choices(str(row.get("question") or ""))
    row["choices"] = []
    row["answer"] = short_answer
    row["short_answer"] = short_answer
    row["answer_format"] = "short_text"
    row["value_span_text"] = short_answer
    row["metadata"] = {
        **dict(row.get("metadata") or {}),
        "choice_to_open_answer": {
            "original_answer": original_answer,
            "original_answer_format": original_answer_format,
            "original_choices": original_choices,
            "original_value_span_text": original_value_span,
        },
    }
    for step in row.get("focus_steps") or []:
        if isinstance(step, dict):
            step["value_span_text"] = short_answer
    return row


def _is_choice_record(record: dict[str, Any]) -> bool:
    return bool(record.get("choices")) or record.get("answer_format") == "multiple_choice"


def _open_answer_text(record: dict[str, Any]) -> str:
    short = str(record.get("short_answer") or "").strip()
    if short:
        return _strip_choice_letter_prefix(short)
    value_span = str(record.get("value_span_text") or "").strip()
    if value_span:
        return _strip_choice_letter_prefix(value_span)
    answer = str(record.get("answer") or "").strip()
    choices = _choice_texts(record.get("choices") or [])
    letter = _answer_choice_letter(answer)
    if letter and choices:
        index = ord(letter) - ord("A")
        if 0 <= index < len(choices):
            return choices[index].strip()
    return _strip_choice_letter_prefix(answer)


def _answer_choice_letter(answer: str) -> str:
    match = re.match(r"^\s*([A-Z])(?:[.)]|:)\s+", answer.strip(), flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""


def _strip_choice_letter_prefix(text: str) -> str:
    return re.sub(r"^\s*[A-Z](?:[.)]|:)\s+", "", text.strip(), count=1, flags=re.IGNORECASE).strip()


def _choice_texts(choices: Any) -> list[str]:
    out: list[str] = []
    for choice in choices or []:
        if isinstance(choice, dict):
            text = choice.get("text") or choice.get("answer") or choice.get("label") or ""
        else:
            text = str(choice)
        out.append(str(text).strip())
    return out


def _strip_answer_choices(question: str) -> str:
    lines = []
    for line in question.splitlines():
        stripped = line.strip()
        if len(stripped) > 3 and stripped[0] == "(" and stripped[2] == ")" and stripped[1].isalpha():
            continue
        if len(stripped) > 2 and stripped[0].isalpha() and stripped[1] == ".":
            continue
        lowered = stripped.lower()
        if lowered.startswith("answer with") or lowered.startswith("answer only with"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()
