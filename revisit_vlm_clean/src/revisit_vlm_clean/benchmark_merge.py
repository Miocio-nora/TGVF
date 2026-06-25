"""Deterministic merge helpers for clean benchmark shard outputs."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from .schema import EvalSummary, RunConfig, _to_jsonable


def merge_benchmark_shards(
    shard_dirs: list[str | Path],
    *,
    output_dir: str | Path,
    run_id: str | None = None,
    expected_num_shards: int | None = None,
    expected_source_manifest_hash: str | None = None,
) -> dict[str, Any]:
    if not shard_dirs:
        raise ValueError("at least one shard directory is required")

    shard_payloads = [_read_shard(Path(path)) for path in shard_dirs]
    shard_payloads.sort(key=lambda item: int(item["config"].shard_index))
    num_shards = int(shard_payloads[0]["config"].num_shards)
    if expected_num_shards is not None and int(expected_num_shards) != num_shards:
        raise ValueError(
            f"expected_num_shards={expected_num_shards} does not match shard config {num_shards}"
        )
    _validate_shard_set(shard_payloads, num_shards=num_shards)
    source_manifest_hash = _source_manifest_hash(shard_payloads)
    if (
        expected_source_manifest_hash is not None
        and expected_source_manifest_hash != source_manifest_hash
    ):
        raise ValueError(
            "expected_source_manifest_hash does not match shard source manifest: "
            f"{expected_source_manifest_hash} != {source_manifest_hash}"
        )

    rows = _interleave_rows(shard_payloads, num_shards=num_shards)
    samples = _interleave_samples(shard_payloads, num_shards=num_shards)
    _validate_merged_rows(rows, samples)

    base_config = shard_payloads[0]["config"]
    merged_run_id = run_id or f"{base_config.run_id}_merged"
    merged_config = replace(
        base_config,
        run_id=merged_run_id,
        manifest_hash=source_manifest_hash,
        shard_index=0,
        num_shards=num_shards,
    )
    summary = _summarize_rows(
        rows,
        run_id=merged_run_id,
        manifest_hash=source_manifest_hash,
    )
    metadata = _merge_metadata(shard_payloads, len(rows), source_manifest_hash)
    summary["merge_metadata"] = metadata
    manifest = _merged_manifest(
        shard_payloads,
        samples=samples,
        source_manifest_hash=source_manifest_hash,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    summary_path = out / "summary.json"
    run_config_path = out / "run_config.json"
    manifest_path = out / "sample_manifest.json"
    metadata_path = out / "merge_metadata.json"

    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_to_jsonable(row), sort_keys=True) + "\n")
    _write_json(summary_path, summary)
    _write_json(run_config_path, merged_config)
    _write_json(manifest_path, manifest)
    _write_json(metadata_path, metadata)

    return {
        "output_dir": str(out),
        "rows": str(rows_path),
        "summary": str(summary_path),
        "run_config": str(run_config_path),
        "sample_manifest": str(manifest_path),
        "merge_metadata": str(metadata_path),
    }


def _read_shard(path: Path) -> dict[str, Any]:
    run_config_path = path / "run_config.json"
    manifest_path = path / "sample_manifest.json"
    rows_path = path / "rows.jsonl"
    summary_path = path / "summary.json"
    for required in (run_config_path, manifest_path, rows_path, summary_path):
        if not required.exists():
            raise FileNotFoundError(f"missing shard artifact: {required}")
    config = RunConfig.from_json(run_config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "path": path,
        "config": config,
        "manifest": manifest,
        "summary": summary,
        "rows": rows,
    }


def _validate_shard_set(shards: list[dict[str, Any]], *, num_shards: int) -> None:
    indices = [int(item["config"].shard_index) for item in shards]
    expected = list(range(num_shards))
    if indices != expected:
        raise ValueError(f"shard indices must cover {expected}, got {indices}")
    for item in shards:
        config: RunConfig = item["config"]
        if int(config.num_shards) != num_shards:
            raise ValueError("all shard configs must use the same num_shards")
        manifest = item["manifest"]
        if int(manifest.get("num_shards", num_shards)) != num_shards:
            raise ValueError("all shard manifests must use the same num_shards")
        if int(manifest.get("shard_index", config.shard_index)) != int(config.shard_index):
            raise ValueError("shard manifest index does not match run config")
        samples = list(manifest.get("samples") or [])
        if len(samples) != len(item["rows"]):
            raise ValueError(
                f"shard {config.shard_index} row count {len(item['rows'])} "
                f"does not match manifest sample count {len(samples)}"
            )
        for row, sample in zip(item["rows"], samples, strict=True):
            if row.get("sample_id") != sample.get("sample_id"):
                raise ValueError(
                    f"shard {config.shard_index} row/sample id mismatch: "
                    f"{row.get('sample_id')} != {sample.get('sample_id')}"
                )
            if int(row.get("shard_index", config.shard_index)) != int(config.shard_index):
                raise ValueError("row shard_index does not match run config")


def _source_manifest_hash(shards: list[dict[str, Any]]) -> str:
    hashes = {
        item["manifest"].get("source_manifest_hash") or item["manifest"].get("manifest_hash")
        for item in shards
    }
    if len(hashes) != 1:
        raise ValueError(f"all shards must share one source manifest hash, got {sorted(hashes)}")
    value = hashes.pop()
    if not value:
        raise ValueError("source manifest hash is missing")
    return str(value)


def _interleave_rows(shards: list[dict[str, Any]], *, num_shards: int) -> list[dict[str, Any]]:
    row_groups = [list(item["rows"]) for item in shards]
    return _interleave(row_groups, num_shards=num_shards)


def _interleave_samples(shards: list[dict[str, Any]], *, num_shards: int) -> list[dict[str, Any]]:
    sample_groups = [list(item["manifest"].get("samples") or []) for item in shards]
    return _interleave(sample_groups, num_shards=num_shards)


def _interleave(groups: list[list[dict[str, Any]]], *, num_shards: int) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    max_len = max((len(group) for group in groups), default=0)
    for offset in range(max_len):
        for shard_index in range(num_shards):
            group = groups[shard_index]
            if offset < len(group):
                merged.append(group[offset])
    return merged


def _validate_merged_rows(rows: list[dict[str, Any]], samples: list[dict[str, Any]]) -> None:
    if len(rows) != len(samples):
        raise ValueError("merged row count does not match merged sample count")
    sample_ids = [sample.get("sample_id") for sample in samples]
    row_ids = [row.get("sample_id") for row in rows]
    if row_ids != sample_ids:
        raise ValueError("merged row order does not match merged sample manifest order")
    duplicates = sorted(sample_id for sample_id, count in Counter(row_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate merged sample ids: {duplicates}")


def _summarize_rows(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    manifest_hash: str,
) -> dict[str, Any]:
    scored = [float(row["score"]) for row in rows if row.get("score") is not None]
    parse_values = [bool(row.get("answer_parse_success")) for row in rows]
    malformed = [bool(row.get("malformed")) for row in rows]
    trigger_values = [
        bool(row.get("trigger_focus_decision"))
        for row in rows
        if row.get("trigger_focus_decision") is not None
    ]
    focus_values = [
        bool(row.get("focus_valid")) for row in rows if row.get("focus_valid") is not None
    ]
    append_values = [
        bool(row.get("append_success")) for row in rows if row.get("append_success") is not None
    ]
    summary = EvalSummary(
        run_id=run_id,
        n_rows=len(rows),
        n_scored=len(scored),
        accuracy=_mean(scored),
        answer_parse_rate=_mean_bool(parse_values),
        malformed_rate=_mean_bool(malformed),
        trigger_rate=_mean_bool(trigger_values) if trigger_values else None,
        focus_valid_rate=_mean_bool(focus_values) if focus_values else None,
        append_success_rate=_mean_bool(append_values) if append_values else None,
        manifest_hash=manifest_hash,
        comparable=True,
        comparability_note="deterministically merged clean benchmark shards",
    ).to_dict()
    summary["merged_shards"] = True
    return summary


def _merged_manifest(
    shards: list[dict[str, Any]],
    *,
    samples: list[dict[str, Any]],
    source_manifest_hash: str,
) -> dict[str, Any]:
    first = shards[0]["manifest"]
    source_manifest_id = first.get("source_manifest_id") or first.get("manifest_id")
    return {
        "manifest_id": source_manifest_id,
        "manifest_hash": source_manifest_hash,
        "samples": samples,
        "source_population_ids": first.get("source_population_ids"),
        "merged_from_shards": _merge_metadata(shards, len(samples), source_manifest_hash),
    }


def _merge_metadata(
    shards: list[dict[str, Any]],
    merged_row_count: int,
    source_manifest_hash: str,
) -> dict[str, Any]:
    return {
        "num_shards": int(shards[0]["config"].num_shards),
        "shard_indices": [int(item["config"].shard_index) for item in shards],
        "source_manifest_hash": source_manifest_hash,
        "merged_row_count": merged_row_count,
        "shard_dirs": [str(item["path"]) for item in shards],
        "merge_order": "source_manifest_order_modulo",
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _mean_bool(values: list[bool]) -> float | None:
    return sum(1.0 if value else 0.0 for value in values) / len(values) if values else None


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True) + "\n")
