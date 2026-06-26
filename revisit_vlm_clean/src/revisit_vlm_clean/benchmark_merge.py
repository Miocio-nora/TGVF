"""Deterministic merge helpers for clean benchmark shard outputs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from .data_generation import file_identity
from .outputs import (
    BENCHMARK_SOURCES_FILENAME,
    benchmark_source_manifest_reference,
    build_comparability_flags_payload,
    build_manifest_verification_payload,
    write_run_config_text,
)
from .runner import (
    build_deepstack_execution_plan,
    summarize_deepstack_execution,
    summarize_result_breakdowns,
)
from .schema import EvalSummary, RunConfig, _to_jsonable

SHARD_RUN_CONFIG_IDENTITY_FIELDS = (
    "output_schema_version",
    "eval_family",
    "mode",
    "checkpoint_path",
    "model_id",
    "processor_id",
    "population_id",
    "subset_id",
    "manifest_path",
    "benchmark_root",
    "max_image_resolution",
    "max_action_tokens",
    "max_answer_tokens",
    "tgvf_protocol",
    "post_tgvf_forward_mode",
    "post_tgvf_continuation",
    "prompt_suffix",
    "softforce_prompt_text",
    "deepstack",
    "parser_scorer",
    "git_commit",
    "dirty_worktree",
)


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
    _validate_shard_run_config_identity(shard_payloads)
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
    source_manifest = _merged_benchmark_source_manifest(
        shard_payloads,
        source_manifest_hash=source_manifest_hash,
    )
    merged_config = replace(
        base_config,
        run_id=merged_run_id,
        manifest_hash=source_manifest_hash,
        shard_index=0,
        num_shards=num_shards,
        benchmark_source_manifest=benchmark_source_manifest_reference(source_manifest),
        execution_backend=_merged_execution_backend(shard_payloads, rows),
    )
    metadata = _merge_metadata(shard_payloads, len(rows), source_manifest_hash)
    manifest = _merged_manifest(
        shard_payloads,
        samples=samples,
        source_manifest_hash=source_manifest_hash,
    )
    summary = _summarize_rows(
        rows,
        run_id=merged_run_id,
        manifest_hash=source_manifest_hash,
    )
    summary["benchmark_source_manifest"] = benchmark_source_manifest_reference(source_manifest)
    summary["merge_metadata"] = metadata
    summary["manifest_verification"] = build_manifest_verification_payload(
        config=merged_config,
        manifest=manifest,
        rows=rows,
        source_manifest=source_manifest,
        merge_metadata=metadata,
    )
    summary["comparability"] = build_comparability_flags_payload(
        config=merged_config,
        manifest=manifest,
        rows=rows,
        comparable=bool(summary.get("comparable")),
        comparability_note=str(summary.get("comparability_note") or ""),
        merge_metadata=metadata,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    summary_path = out / "summary.json"
    run_config_path = out / "run_config.json"
    run_config_txt_path = out / "run_config.txt"
    manifest_path = out / "sample_manifest.json"
    metadata_path = out / "merge_metadata.json"
    sources_path = out / BENCHMARK_SOURCES_FILENAME

    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_to_jsonable(row), sort_keys=True) + "\n")
    _write_json(summary_path, summary)
    _write_json(run_config_path, merged_config)
    write_run_config_text(
        run_config_txt_path,
        config=merged_config,
        manifest=manifest,
    )
    _write_json(manifest_path, manifest)
    _write_json(metadata_path, metadata)
    _write_json(sources_path, source_manifest)

    return {
        "output_dir": str(out),
        "rows": str(rows_path),
        "summary": str(summary_path),
        "run_config": str(run_config_path),
        "run_config_txt": str(run_config_txt_path),
        "sample_manifest": str(manifest_path),
        "merge_metadata": str(metadata_path),
        "benchmark_sources": str(sources_path),
    }


def _read_shard(path: Path) -> dict[str, Any]:
    run_config_path = path / "run_config.json"
    run_config_txt_path = path / "run_config.txt"
    manifest_path = path / "sample_manifest.json"
    rows_path = path / "rows.jsonl"
    summary_path = path / "summary.json"
    sources_path = path / BENCHMARK_SOURCES_FILENAME
    artifact_paths = {
        "run_config": run_config_path,
        "run_config_txt": run_config_txt_path,
        "sample_manifest": manifest_path,
        "rows": rows_path,
        "summary": summary_path,
        "benchmark_sources": sources_path,
    }
    for required in artifact_paths.values():
        if not required.exists():
            raise FileNotFoundError(f"missing shard artifact: {required}")
    config = RunConfig.from_json(run_config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
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
        "benchmark_sources": sources,
        "rows": rows,
        "artifact_identities": {
            name: file_identity(artifact_path).to_dict()
            for name, artifact_path in artifact_paths.items()
        },
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
        source_manifest = item["benchmark_sources"]
        if source_manifest.get("schema_version") != "clean_benchmark_source_manifest_v1":
            raise ValueError("shard benchmark_sources schema mismatch")
        if int(source_manifest.get("sample_count", -1)) != len(item["rows"]):
            raise ValueError("shard benchmark_sources sample_count does not match rows")
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
            _validate_row_identity_against_shard_config(
                row,
                sample=sample,
                config=config,
            )
            if int(row.get("shard_index", config.shard_index)) != int(config.shard_index):
                raise ValueError("row shard_index does not match run config")


def _validate_row_identity_against_shard_config(
    row: dict[str, Any],
    *,
    sample: dict[str, Any],
    config: RunConfig,
) -> None:
    execution_backend = config.execution_backend or {}
    expected = {
        "benchmark": sample.get("benchmark"),
        "population_id": sample.get("population_id"),
        "source_file": sample.get("source_file"),
        "num_shards": int(config.num_shards),
        "shard_index": int(config.shard_index),
        "method": config.mode.value,
        "eval_family": config.eval_family.value,
        "tgvf_protocol": config.tgvf_protocol,
        "post_tgvf_continuation": config.post_tgvf_continuation.value,
        "post_tgvf_forward_mode": config.post_tgvf_forward_mode.value,
        "deepstack": config.deepstack.to_dict(),
        "parser_scorer": config.parser_scorer.to_dict(),
        "subset_id": config.subset_id,
        "runner_backend": execution_backend.get("backend"),
        "resolved_runner_backend": execution_backend.get("resolved_backend"),
        "runner_backend_final_clean": execution_backend.get("final_clean_backend"),
        "runner_backend_diagnostic_bridge": execution_backend.get("diagnostic_bridge"),
        "runner_backend_deprecated_alias": execution_backend.get("deprecated_alias"),
        "runner_backend_stage2_generic_alias": execution_backend.get("stage2_generic_alias"),
        "runner_backend_alias_target": execution_backend.get("alias_target"),
        "d_condition": (
            (execution_backend.get("stage2") or {}).get("d_condition")
            if isinstance(execution_backend.get("stage2"), dict)
            else None
        ),
    }
    for field, expected_value in expected.items():
        current = row.get(field)
        if not _json_equal(current, expected_value):
            raise ValueError(
                "row identity does not match shard run config: "
                f"shard={config.shard_index} sample_id={row.get('sample_id')} "
                f"field={field} expected={expected_value!r} current={current!r}"
            )
    _validate_nested_row_identity(
        row,
        config=config,
        execution_backend=execution_backend,
    )


def _validate_nested_row_identity(
    row: dict[str, Any],
    *,
    config: RunConfig,
    execution_backend: dict[str, Any],
) -> None:
    stage2 = execution_backend.get("stage2") if isinstance(execution_backend, dict) else None
    resolved_backend = str(
        execution_backend.get("resolved_backend") or execution_backend.get("backend") or ""
    )
    deepstack_plan = build_deepstack_execution_plan(config, backend=resolved_backend)
    expected_blocks = {
        "trigger_policy": _expected_trigger_policy(config),
        "continuation_metadata": {
            "schema_version": "clean_benchmark_continuation_metadata_v1",
            "post_tgvf_continuation": config.post_tgvf_continuation.value,
            "post_tgvf_forward_mode": config.post_tgvf_forward_mode.value,
            "stage2_append_forward_mode": (
                stage2.get("append_forward_mode") if isinstance(stage2, dict) else None
            ),
            "runner_backend": execution_backend.get("backend"),
            "resolved_runner_backend": execution_backend.get("resolved_backend"),
        },
        "deepstack_execution": {
            "schema_version": "clean_deepstack_execution_row_v1",
            "requested": config.deepstack.to_dict(),
            "requested_enabled": bool(config.deepstack.enabled),
            "requested_scope": str(config.deepstack.original_image_scope),
            "execution_supported_for_requested_state": deepstack_plan["execution_supported"],
            "backend": execution_backend.get("backend"),
            "resolved_backend": execution_backend.get("resolved_backend"),
            "notes": (
                ["DeepStack was requested but this clean execution plan rejects it before rows"]
                if config.deepstack.enabled and not deepstack_plan["execution_supported"]
                else []
            ),
        },
    }
    for block_name, expected_fields in expected_blocks.items():
        block = row.get(block_name)
        if not isinstance(block, dict):
            raise ValueError(
                "row identity block is missing or invalid: "
                f"shard={config.shard_index} sample_id={row.get('sample_id')} "
                f"field={block_name}"
            )
        for field, expected_value in expected_fields.items():
            current = block.get(field)
            if not _json_equal(current, expected_value):
                raise ValueError(
                    "row identity does not match shard run config: "
                    f"shard={config.shard_index} sample_id={row.get('sample_id')} "
                    f"field={block_name}.{field} expected={expected_value!r} "
                    f"current={current!r}"
                )


def _expected_trigger_policy(config: RunConfig) -> dict[str, Any]:
    mode = config.mode.value
    if mode == "original":
        policy = "none_original"
    elif mode == "tgvf_force":
        policy = "forced_focus_prefix"
    elif mode == "tgvf_softforce":
        policy = "softforce_prompted_router"
    else:
        policy = "free_router"
    return {
        "schema_version": "clean_benchmark_trigger_policy_v1",
        "policy": policy,
        "mode": mode,
        "requires_focus": mode == "tgvf_force",
        "allows_no_focus": mode in {
            "original",
            "tgvf_free",
            "tgvf_softforce",
        },
        "softforce_prompt_text": config.softforce_prompt_text,
    }


def _validate_shard_run_config_identity(shards: list[dict[str, Any]]) -> None:
    if not shards:
        return
    baseline = _run_config_merge_identity(shards[0]["config"])
    baseline_index = int(shards[0]["config"].shard_index)
    for item in shards[1:]:
        config = item["config"]
        current = _run_config_merge_identity(config)
        mismatched = [
            key
            for key in baseline
            if not _json_equal(baseline[key], current.get(key))
        ]
        if mismatched:
            shard_index = int(config.shard_index)
            first_key = mismatched[0]
            raise ValueError(
                "shard run config identity mismatch: "
                f"baseline_shard={baseline_index} shard={shard_index} field={first_key} "
                f"baseline={baseline[first_key]!r} current={current.get(first_key)!r}"
            )


def _run_config_merge_identity(config: RunConfig) -> dict[str, Any]:
    payload = {
        field: _to_jsonable(getattr(config, field))
        for field in SHARD_RUN_CONFIG_IDENTITY_FIELDS
    }
    payload["execution_backend"] = _semantic_execution_backend(config.execution_backend)
    return payload


def _semantic_execution_backend(value: Any) -> Any:
    if not isinstance(value, dict):
        return _to_jsonable(value)
    cleaned = dict(value)
    cleaned.pop("device", None)
    cleaned.pop("device_map", None)
    return _to_jsonable(cleaned)


def _json_equal(left: Any, right: Any) -> bool:
    return json.dumps(_to_jsonable(left), sort_keys=True, separators=(",", ":")) == json.dumps(
        _to_jsonable(right),
        sort_keys=True,
        separators=(",", ":"),
    )


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
    summary["deepstack_execution"] = summarize_deepstack_execution(rows)
    summary["result_breakdowns"] = summarize_result_breakdowns(rows)
    summary["parser_scorer"] = _single_json_field(rows, "parser_scorer")
    summary["deepstack"] = _single_json_field(rows, "deepstack")
    summary["post_tgvf_forward_mode"] = _single_scalar_field(rows, "post_tgvf_forward_mode")
    summary["post_tgvf_continuation"] = _single_scalar_field(rows, "post_tgvf_continuation")
    summary["eval_family"] = _single_scalar_field(rows, "eval_family")
    summary["tgvf_protocol"] = _single_scalar_field(rows, "tgvf_protocol")
    summary["runner_backend"] = _merged_runner_backend_summary(rows)
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


def _merged_benchmark_source_manifest(
    shards: list[dict[str, Any]],
    *,
    source_manifest_hash: str,
) -> dict[str, Any]:
    first = shards[0]["benchmark_sources"]
    benchmark_root = first.get("benchmark_root")
    manifest_id = first.get("source_manifest_id") or first.get("manifest_id")
    grouped: dict[str, dict[str, Any]] = {}
    for shard in shards:
        sources = shard["benchmark_sources"]
        if sources.get("benchmark_root") != benchmark_root:
            raise ValueError("all shard benchmark_sources must share one benchmark_root")
        for item in sources.get("source_files") or []:
            source_file = str(item.get("source_file") or "")
            group = grouped.setdefault(
                source_file,
                {
                    "template": item,
                    "sample_count": 0,
                    "row_indices": [],
                    "benchmarks": set(),
                    "population_ids": set(),
                },
            )
            template = group["template"]
            for key in ("path", "exists", "byte_size", "sha256"):
                if template.get(key) != item.get(key):
                    raise ValueError(f"source file identity mismatch for {source_file}: {key}")
            group["sample_count"] += int(item.get("sample_count") or 0)
            group["row_indices"].extend(int(value) for value in item.get("row_indices") or [])
            group["benchmarks"].update(str(value) for value in item.get("benchmarks") or [])
            group["population_ids"].update(str(value) for value in item.get("population_ids") or [])

    source_files = []
    for source_file, group in sorted(grouped.items()):
        template = group["template"]
        row_indices = sorted(group["row_indices"])
        source_files.append(
            {
                "source_file": source_file,
                "path": template.get("path"),
                "exists": template.get("exists"),
                "byte_size": template.get("byte_size"),
                "sha256": template.get("sha256"),
                "benchmarks": sorted(group["benchmarks"]),
                "population_ids": sorted(group["population_ids"]),
                "sample_count": group["sample_count"],
                "row_index_count": len(row_indices),
                "row_index_min": min(row_indices) if row_indices else None,
                "row_index_max": max(row_indices) if row_indices else None,
                "row_indices": row_indices,
                "row_indices_sha256": _stable_json_hash(row_indices),
            }
        )

    return {
        "schema_version": "clean_benchmark_source_manifest_v1",
        "benchmark_root": benchmark_root,
        "manifest_id": manifest_id,
        "manifest_hash": source_manifest_hash,
        "source_manifest_id": manifest_id,
        "source_manifest_hash": source_manifest_hash,
        "source_file_count": len(source_files),
        "sample_count": sum(int(item["sample_count"]) for item in source_files),
        "all_files_exist": all(bool(item["exists"]) for item in source_files),
        "source_files": source_files,
        "merged_from_shards": _merge_metadata(
            shards,
            sum(int(item["sample_count"]) for item in source_files),
            source_manifest_hash,
        ),
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
        "shard_artifacts": [
            {
                "shard_index": int(item["config"].shard_index),
                "path": str(item["path"]),
                "artifacts": item["artifact_identities"],
            }
            for item in shards
        ],
        "merge_order": "source_manifest_order_modulo",
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _mean_bool(values: list[bool]) -> float | None:
    return sum(1.0 if value else 0.0 for value in values) / len(values) if values else None


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _stable_json_hash(value: Any) -> str:
    encoded = json.dumps(_to_jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _single_json_field(rows: list[dict[str, Any]], key: str) -> Any:
    values = {}
    for row in rows:
        value = row.get(key)
        encoded = json.dumps(_to_jsonable(value), sort_keys=True, separators=(",", ":"))
        values.setdefault(encoded, value)
    if len(values) != 1:
        raise ValueError(f"merged rows disagree on {key}: {sorted(values)}")
    return next(iter(values.values()))


def _single_scalar_field(rows: list[dict[str, Any]], key: str) -> Any:
    values = sorted({row.get(key) for row in rows})
    if len(values) != 1:
        raise ValueError(f"merged rows disagree on {key}: {values}")
    return values[0]


def _merged_runner_backend_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    backend_counts = Counter(str(row.get("runner_backend") or "none") for row in rows)
    resolved_counts = Counter(str(row.get("resolved_runner_backend") or "none") for row in rows)
    alias_targets = sorted(
        {
            str(row.get("runner_backend_alias_target"))
            for row in rows
            if row.get("runner_backend_alias_target")
        }
    )
    return {
        "schema_version": "clean_merged_runner_backend_summary_v1",
        "backend_counts": dict(sorted(backend_counts.items())),
        "resolved_backend_counts": dict(sorted(resolved_counts.items())),
        "final_clean_backend_rows": sum(
            bool(row.get("runner_backend_final_clean")) for row in rows
        ),
        "diagnostic_bridge_rows": sum(
            bool(row.get("runner_backend_diagnostic_bridge")) for row in rows
        ),
        "stage2_generic_alias_rows": sum(
            bool(row.get("runner_backend_stage2_generic_alias")) for row in rows
        ),
        "deprecated_alias_rows": sum(
            bool(row.get("runner_backend_deprecated_alias")) for row in rows
        ),
        "alias_targets": alias_targets,
    }


def _merged_execution_backend(
    shards: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    encoded_backends: dict[str, Any] = {}
    missing_shards = []
    for shard in shards:
        config: RunConfig = shard["config"]
        backend = config.execution_backend
        if backend is None:
            missing_shards.append(int(config.shard_index))
            continue
        encoded = json.dumps(_to_jsonable(backend), sort_keys=True, separators=(",", ":"))
        encoded_backends.setdefault(encoded, backend)
    return {
        "schema_version": "clean_merged_execution_backend_v1",
        "shard_count": len(shards),
        "source_execution_backend_count": len(encoded_backends),
        "source_execution_backend_hashes": sorted(
            hashlib.sha256(encoded.encode()).hexdigest() for encoded in encoded_backends
        ),
        "source_execution_backends": [
            encoded_backends[key] for key in sorted(encoded_backends)
        ],
        "missing_source_execution_backend_shards": missing_shards,
        "runner_backend": _merged_runner_backend_summary(rows),
    }
