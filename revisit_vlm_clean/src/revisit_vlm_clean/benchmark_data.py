"""Benchmark sample materialization from clean manifests."""

from __future__ import annotations

import ast
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import _to_jsonable

IMAGE_KEYS = {"image", "decoded_image", "image_path", "img", "images", "media"}
VIDEO_KEYS = {"video", "video_path"}


@dataclass(frozen=True)
class BenchmarkSample:
    sample_id: str
    benchmark: str
    population_id: str
    source_file: str
    question: str
    media: tuple[dict[str, Any], ...] = ()
    choices: tuple[str, ...] = ()
    gold_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def primary_media(self) -> dict[str, Any] | None:
        return self.media[0] if self.media else None

    def to_materialized_row(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "benchmark": self.benchmark,
            "population_id": self.population_id,
            "source_file": self.source_file,
            "question": self.question,
            "choices": list(self.choices),
            "gold_answer": self.gold_answer,
            "media_count": len(self.media),
            "media": [_media_summary(item) for item in self.media],
            "metadata": _safe_json(self.metadata),
        }


def load_manifest_payload(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError(f"invalid sample manifest: {path}")
    return payload


def materialize_samples_from_manifest_path(
    manifest_path: str | Path,
    *,
    benchmark_root: str | Path,
    metadata_only: bool = True,
) -> list[BenchmarkSample]:
    return materialize_samples_from_manifest_payload(
        load_manifest_payload(manifest_path),
        benchmark_root=benchmark_root,
        metadata_only=metadata_only,
    )


def materialize_samples_from_manifest_payload(
    manifest_payload: dict[str, Any],
    *,
    benchmark_root: str | Path,
    metadata_only: bool = True,
) -> list[BenchmarkSample]:
    root = Path(benchmark_root)
    if not root.exists():
        raise FileNotFoundError(f"benchmark root does not exist: {root}")

    refs = [dict(item) for item in manifest_payload.get("samples", [])]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for ref in refs:
        source_file = str(ref.get("source_file") or "")
        if not source_file:
            raise ValueError(f"sample missing source_file: {ref.get('sample_id')}")
        grouped.setdefault(source_file, []).append(ref)

    records_by_source: dict[str, dict[int, dict[str, Any]]] = {}
    for source_file, source_refs in grouped.items():
        path = root / source_file
        indices = sorted(_row_index(ref) for ref in source_refs)
        records_by_source[source_file] = _read_records_at_indices(
            path,
            indices,
            metadata_only=metadata_only,
        )

    samples: list[BenchmarkSample] = []
    for ref in refs:
        source_file = str(ref["source_file"])
        row_index = _row_index(ref)
        record = records_by_source[source_file].get(row_index)
        if record is None:
            raise IndexError(f"{source_file} did not contain manifest row_index={row_index}")
        samples.append(_record_to_sample(ref, record, root=root))
    return samples


def materialized_rows(samples: Iterable[BenchmarkSample]) -> list[dict[str, Any]]:
    return [sample.to_materialized_row() for sample in samples]


def _record_to_sample(
    ref: dict[str, Any], record: dict[str, Any], *, root: Path
) -> BenchmarkSample:
    benchmark = str(ref.get("benchmark") or "")
    population_id = str(ref.get("population_id") or "")
    source_file = str(ref.get("source_file") or "")
    source_path = root / source_file
    benchmark_dir = root / benchmark
    question = _extract_question(record)
    choices = _extract_choices(record)
    question = _question_with_choices(question, choices)
    gold = _extract_gold(record)
    metadata = _metadata_for_sample(ref, record)
    if choices:
        metadata.setdefault("choices", list(choices))
    media = _extract_media(
        record,
        base_dir=source_path.parent,
        benchmark_dir=benchmark_dir,
        benchmark=benchmark,
    )
    return BenchmarkSample(
        sample_id=str(ref.get("sample_id") or ""),
        benchmark=benchmark,
        population_id=population_id,
        source_file=source_file,
        question=question,
        media=tuple(media),
        choices=tuple(choices),
        gold_answer=gold,
        metadata=metadata,
        raw=record,
    )


def _read_records_at_indices(
    path: Path,
    indices: list[int],
    *,
    metadata_only: bool,
) -> dict[int, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"manifest source file does not exist: {path}")
    wanted = set(indices)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return {row_index: record for row_index, record in _iter_jsonl(path) if row_index in wanted}
    if suffix == ".json":
        return {
            row_index: record
            for row_index, record in enumerate(_read_json_records(path))
            if row_index in wanted
        }
    if suffix == ".parquet":
        return _read_parquet_records_at_indices(path, wanted, metadata_only=metadata_only)
    raise ValueError(f"unsupported manifest source file: {path}")


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open() as handle:
        for row_index, line in enumerate(handle):
            if line.strip():
                yield row_index, json.loads(line)


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, dict):
        for key in ("data", "questions", "annotations", "examples"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value]
        records = []
        for key, value in payload.items():
            if isinstance(value, dict):
                record = dict(value)
                record.setdefault("id", key)
                records.append(record)
        return records
    raise ValueError(f"unsupported json payload at {path}")


def _read_parquet_records_at_indices(
    path: Path,
    wanted: set[int],
    *,
    metadata_only: bool,
) -> dict[int, dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
        import pyarrow.types as patypes
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("pyarrow is required for parquet-backed benchmark samples") from exc

    parquet = pq.ParquetFile(path)
    if metadata_only:
        top_level_columns: list[str] = []
        image_struct_columns: list[str] = []
        for field in parquet.schema_arrow:
            name = field.name
            if _is_image_payload_column(name, field.type, patypes):
                if patypes.is_struct(field.type) and any(
                    child.name == "path" for child in field.type
                ):
                    image_struct_columns.append(name)
                continue
            top_level_columns.append(name)
        table = pq.read_table(path, columns=top_level_columns)
        rows = table.to_pylist()
        for image_column in image_struct_columns:
            path_rows = pq.read_table(path, columns=[f"{image_column}.path"]).to_pylist()
            for row_index, path_row in enumerate(path_rows):
                path_hint = path_row.get("path")
                if path_hint:
                    rows[row_index][image_column] = {
                        "bytes": None,
                        "path": path_hint,
                        "__payload_omitted__": True,
                    }
    else:
        rows = pq.read_table(path).to_pylist()
    return {row_index: dict(rows[row_index]) for row_index in wanted if row_index < len(rows)}


def _is_image_payload_column(name: str, arrow_type: Any, patypes: Any) -> bool:
    if name in {"decoded_image"} or name.startswith("image_"):
        return True
    if name == "image" and not patypes.is_string(arrow_type):
        return True
    return False


def _extract_question(record: dict[str, Any]) -> str:
    for key in ("question", "text", "query", "prompt"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _extract_choices(record: dict[str, Any]) -> list[str]:
    for key in ("choices", "options", "answer_options"):
        value = record.get(key)
        decoded = _decode_choice_value(value)
        if decoded:
            return decoded
    letter_choices: list[str] = []
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        value = record.get(letter)
        if value not in (None, ""):
            letter_choices.append(str(value))
        elif letter_choices:
            break
    if letter_choices:
        return letter_choices
    return _choices_from_question(_extract_question(record))


def _decode_choice_value(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        for parser in (json.loads, ast.literal_eval):
            try:
                decoded = parser(value)
            except Exception:
                continue
            if isinstance(decoded, list):
                return [str(item) for item in decoded]
    return []


def _question_with_choices(question: str, choices: list[str]) -> str:
    if not choices or _choices_from_question(question):
        return question
    lines = [question.rstrip(), ""]
    for index, choice in enumerate(choices):
        lines.append(f"({chr(ord('A') + index)}) {choice}")
    lines.append("Answer only with the option letter.")
    return "\n".join(lines).strip()


def _choices_from_question(question: str) -> list[str]:
    choices: list[str] = []
    for line in str(question or "").splitlines():
        stripped = line.strip()
        if len(stripped) > 3 and stripped[0] == "(" and stripped[2] == ")":
            choices.append(stripped[3:].strip())
            continue
        if len(stripped) > 2 and stripped[0].isalpha() and stripped[1] in {":", "."}:
            choices.append(stripped[2:].strip())
    return choices


def _extract_gold(record: dict[str, Any]) -> str | None:
    for key in ("label", "answer", "gold", "gold_answer", "correct_answer", "target"):
        value = record.get(key)
        if value not in (None, ""):
            text = str(value).strip()
            return None if text.lower() == "hidden" else text
    answers = record.get("answers")
    if isinstance(answers, list) and answers:
        return str(answers[0]).strip()
    return None


def _metadata_for_sample(ref: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if isinstance(ref.get("metadata"), dict):
        metadata.update(ref["metadata"])
    embedded = record.get("metadata")
    if isinstance(embedded, dict):
        metadata.update(embedded)
    for key in (
        "category",
        "task",
        "task_type",
        "ability",
        "answer",
        "answers",
        "bbox",
        "content",
        "eval",
        "id",
        "image_shape",
        "options",
        "pid",
        "precision",
        "problem_index",
        "subject",
        "mode",
        "subset",
        "version",
        "source",
        "split",
        "sub_task",
        "subdomain",
        "sample_index",
        "dataset_name",
        "type",
        "question_type",
        "answer_type",
        "problem_version",
        "cycle_category",
        "query",
        "raw_text",
        "unit",
        "uid",
    ):
        if key in record and record[key] not in (None, ""):
            metadata.setdefault(key, record[key])
    return _safe_json(metadata)


def _extract_media(
    record: dict[str, Any],
    *,
    base_dir: Path,
    benchmark_dir: Path,
    benchmark: str,
) -> list[dict[str, Any]]:
    values: list[tuple[str, Any]] = []
    for key, value in record.items():
        if value in (None, "", [], {}):
            continue
        if key in IMAGE_KEYS or key in VIDEO_KEYS or key.startswith("image_"):
            if isinstance(value, list):
                values.extend((key, item) for item in value if item not in (None, ""))
            else:
                values.append((key, value))
    media: list[dict[str, Any]] = []
    for key, value in values:
        media.extend(
            _media_refs_for_value(
                key, value, base_dir=base_dir, benchmark_dir=benchmark_dir, benchmark=benchmark
            )
        )
    return _dedupe_media_refs(media)


def _media_refs_for_value(
    key: str,
    value: Any,
    *,
    base_dir: Path,
    benchmark_dir: Path,
    benchmark: str,
) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        path_hint = value.get("path")
        bytes_value = value.get("bytes")
        omitted = bool(value.get("__payload_omitted__"))
        if path_hint:
            resolved = _resolve_media_path(
                str(path_hint), base_dir, benchmark_dir / "snapshot", benchmark_dir
            )
            return [
                {
                    "kind": "embedded_image_struct" if omitted else "image_struct",
                    "source_key": key,
                    "path_hint": str(path_hint),
                    "path": resolved,
                    "exists": Path(resolved).exists(),
                    "payload_loaded": bytes_value is not None,
                    "byte_length": len(bytes_value)
                    if isinstance(bytes_value, (bytes, bytearray))
                    else None,
                    "bytes": bytes_value,
                }
            ]
        return [
            {
                "kind": "image_struct",
                "source_key": key,
                "payload_loaded": bytes_value is not None,
                "byte_length": len(bytes_value)
                if isinstance(bytes_value, (bytes, bytearray))
                else None,
                "bytes": bytes_value,
            }
        ]
    if isinstance(value, (bytes, bytearray)):
        return [
            {
                "kind": "embedded_bytes",
                "source_key": key,
                "payload_loaded": True,
                "byte_length": len(value),
                "bytes": bytes(value),
            }
        ]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("http://", "https://")):
            return [{"kind": "url", "source_key": key, "url": stripped}]
        if _looks_like_base64_image(stripped):
            return [
                {
                    "kind": "embedded_base64",
                    "source_key": key,
                    "payload_loaded": True,
                    "char_length": len(stripped),
                    "value": stripped,
                }
            ]
        roots = _media_roots(base_dir=base_dir, benchmark_dir=benchmark_dir, benchmark=benchmark)
        resolved = _resolve_media_path(stripped, *roots)
        return [
            {
                "kind": "path",
                "source_key": key,
                "path": resolved,
                "exists": Path(resolved).exists(),
                "path_hint": stripped,
            }
        ]
    return [{"kind": "raw_media_value", "source_key": key, "value_type": type(value).__name__}]


def _media_roots(*, base_dir: Path, benchmark_dir: Path, benchmark: str) -> tuple[Path, ...]:
    snapshot = benchmark_dir / "snapshot"
    roots = [base_dir, snapshot, snapshot / "images", benchmark_dir]
    if benchmark == "mathverse":
        roots.insert(1, snapshot / "images")
    return tuple(dict.fromkeys(roots))


def _dedupe_media_refs(media: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing_path_basenames = {
        Path(str(item.get("path"))).name
        for item in media
        if item.get("kind") == "path" and item.get("exists") is True and item.get("path")
    }
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in media:
        path = str(item.get("path") or "")
        path_hint = str(item.get("path_hint") or "")
        if (
            item.get("kind") == "embedded_image_struct"
            and not item.get("payload_loaded")
            and item.get("exists") is False
            and Path(path_hint or path).name in existing_path_basenames
        ):
            continue
        key = (str(item.get("kind") or ""), path or path_hint or str(item.get("source_key") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _resolve_media_path(value: str, *roots: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    for root in roots:
        candidate = root / value
        if candidate.exists():
            return str(candidate)
    return str(roots[0] / value) if roots else value


def _looks_like_base64_image(value: str) -> bool:
    if value.startswith("data:image") and "," in value:
        return True
    return value.startswith(("/9j/", "iVBOR", "R0lGOD", "UklGR"))


def _row_index(ref: dict[str, Any]) -> int:
    metadata = ref.get("metadata")
    if not isinstance(metadata, dict) or "row_index" not in metadata:
        raise ValueError(f"sample missing metadata.row_index: {ref.get('sample_id')}")
    return int(metadata["row_index"])


def _media_summary(media: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for key, value in media.items():
        if key == "bytes":
            summary["byte_length"] = (
                len(value) if isinstance(value, (bytes, bytearray)) else media.get("byte_length")
            )
            summary["payload_loaded"] = value is not None
            continue
        if key == "value" and isinstance(value, str) and _looks_like_base64_image(value):
            summary["char_length"] = len(value)
            summary["payload_loaded"] = True
            continue
        summary[key] = _safe_json(value)
    return summary


def _safe_json(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return {"type": "bytes", "length": len(value)}
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    try:
        return _to_jsonable(value)
    except TypeError:
        return str(value)
