"""Output helpers for clean benchmark runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .benchmark_data import BenchmarkSample
from .manifest import SampleManifest, manifest_payload
from .populations import CORE_FULL_N, CORE_POPULATIONS, SUBSETS
from .rendering import RenderedBenchmarkInput
from .runner import BackendConfig, summarize_deepstack_execution, summarize_result_breakdowns
from .schema import EvalSummary, RunConfig, _to_jsonable

BENCHMARK_SOURCES_FILENAME = "benchmark_sources.json"


def write_empty_benchmark_output(
    output_dir: str | Path,
    *,
    config: RunConfig,
    manifest: SampleManifest | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    run_config_path = out / "run_config.json"
    run_config_txt_path = out / "run_config.txt"
    rows_path = out / "rows.jsonl"
    summary_path = out / "summary.json"
    manifest_path = out / "sample_manifest.json"
    sources_path = out / BENCHMARK_SOURCES_FILENAME

    source_manifest = write_benchmark_source_manifest(
        sources_path,
        config=config,
        manifest=manifest,
    )
    runtime_config = replace(
        config,
        benchmark_source_manifest=benchmark_source_manifest_reference(source_manifest),
    )

    _write_json(run_config_path, runtime_config)
    rows_path.write_text("")
    if manifest is not None:
        _write_json(manifest_path, manifest_payload(manifest))
        manifest_hash = manifest.stable_hash()
    else:
        _write_json(
            manifest_path,
            {
                "manifest_id": config.subset_id or config.population_id,
                "manifest_path": config.manifest_path,
                "manifest_hash": config.manifest_hash,
                "status": "not_loaded",
            },
        )
        manifest_hash = config.manifest_hash

    _write_run_config_text(run_config_txt_path, config=runtime_config, manifest=manifest)
    summary = EvalSummary(
        run_id=runtime_config.run_id,
        n_rows=0,
        n_scored=0,
        accuracy=None,
        answer_parse_rate=None,
        malformed_rate=None,
        manifest_hash=manifest_hash,
        comparable=False,
        comparability_note="schema smoke output; no model inference executed",
    )
    summary_payload = summary.to_dict()
    summary_payload["parser_scorer"] = config.parser_scorer.to_dict()
    summary_payload["deepstack"] = config.deepstack.to_dict()
    summary_payload["post_tgvf_forward_mode"] = str(config.post_tgvf_forward_mode)
    summary_payload["post_tgvf_continuation"] = str(config.post_tgvf_continuation)
    summary_payload["benchmark_source_manifest"] = runtime_config.benchmark_source_manifest
    _write_json(summary_path, summary_payload)

    return {
        "output_dir": str(out),
        "run_config": str(run_config_path),
        "run_config_txt": str(run_config_txt_path),
        "rows": str(rows_path),
        "summary": str(summary_path),
        "sample_manifest": str(manifest_path),
        "benchmark_sources": str(sources_path),
    }


def write_materialized_sample_output(
    output_dir: str | Path,
    *,
    config: RunConfig,
    manifest: dict[str, Any],
    samples: list[BenchmarkSample],
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    run_config_path = out / "run_config.json"
    run_config_txt_path = out / "run_config.txt"
    materialized_rows_path = out / "materialized_rows.jsonl"
    summary_path = out / "summary.json"
    manifest_path = out / "sample_manifest.json"
    sources_path = out / BENCHMARK_SOURCES_FILENAME

    source_manifest = write_benchmark_source_manifest(
        sources_path,
        config=config,
        manifest=manifest,
    )
    runtime_config = replace(
        config,
        benchmark_source_manifest=benchmark_source_manifest_reference(source_manifest),
    )

    _write_json(run_config_path, runtime_config)
    _write_json(manifest_path, manifest)
    _write_run_config_text(run_config_txt_path, config=runtime_config, manifest=manifest)
    with materialized_rows_path.open("w") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_materialized_row(), sort_keys=True) + "\n")

    manifest_hash = manifest.get("manifest_hash")
    summary = EvalSummary(
        run_id=config.run_id,
        n_rows=len(samples),
        n_scored=0,
        accuracy=None,
        answer_parse_rate=None,
        malformed_rate=None,
        manifest_hash=manifest_hash,
        comparable=False,
        comparability_note="sample materialization smoke output; no model inference executed",
    )
    summary_payload = summary.to_dict()
    summary_payload["parser_scorer"] = config.parser_scorer.to_dict()
    summary_payload["deepstack"] = config.deepstack.to_dict()
    summary_payload["post_tgvf_forward_mode"] = str(config.post_tgvf_forward_mode)
    summary_payload["post_tgvf_continuation"] = str(config.post_tgvf_continuation)
    summary_payload["materialized_sample_count"] = len(samples)
    summary_payload["benchmark_source_manifest"] = runtime_config.benchmark_source_manifest
    _write_json(summary_path, summary_payload)

    return {
        "output_dir": str(out),
        "run_config": str(run_config_path),
        "run_config_txt": str(run_config_txt_path),
        "materialized_rows": str(materialized_rows_path),
        "summary": str(summary_path),
        "sample_manifest": str(manifest_path),
        "benchmark_sources": str(sources_path),
    }


def write_rendered_input_output(
    output_dir: str | Path,
    *,
    config: RunConfig,
    manifest: dict[str, Any],
    rendered_inputs: list[RenderedBenchmarkInput],
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    run_config_path = out / "run_config.json"
    run_config_txt_path = out / "run_config.txt"
    rendered_inputs_path = out / "rendered_inputs.jsonl"
    summary_path = out / "summary.json"
    manifest_path = out / "sample_manifest.json"
    sources_path = out / BENCHMARK_SOURCES_FILENAME

    source_manifest = write_benchmark_source_manifest(
        sources_path,
        config=config,
        manifest=manifest,
    )
    runtime_config = replace(
        config,
        benchmark_source_manifest=benchmark_source_manifest_reference(source_manifest),
    )

    _write_json(run_config_path, runtime_config)
    _write_json(manifest_path, manifest)
    _write_run_config_text(run_config_txt_path, config=runtime_config, manifest=manifest)
    with rendered_inputs_path.open("w") as handle:
        for rendered in rendered_inputs:
            handle.write(json.dumps(rendered.to_row(), sort_keys=True) + "\n")

    manifest_hash = manifest.get("manifest_hash")
    summary = EvalSummary(
        run_id=config.run_id,
        n_rows=len(rendered_inputs),
        n_scored=0,
        accuracy=None,
        answer_parse_rate=None,
        malformed_rate=None,
        manifest_hash=manifest_hash,
        comparable=False,
        comparability_note="rendered input smoke output; no model inference executed",
    )
    summary_payload = summary.to_dict()
    summary_payload["parser_scorer"] = config.parser_scorer.to_dict()
    summary_payload["deepstack"] = config.deepstack.to_dict()
    summary_payload["post_tgvf_forward_mode"] = str(config.post_tgvf_forward_mode)
    summary_payload["post_tgvf_continuation"] = str(config.post_tgvf_continuation)
    summary_payload["rendered_input_count"] = len(rendered_inputs)
    summary_payload["benchmark_source_manifest"] = runtime_config.benchmark_source_manifest
    _write_json(summary_path, summary_payload)

    return {
        "output_dir": str(out),
        "run_config": str(run_config_path),
        "run_config_txt": str(run_config_txt_path),
        "rendered_inputs": str(rendered_inputs_path),
        "summary": str(summary_path),
        "sample_manifest": str(manifest_path),
        "benchmark_sources": str(sources_path),
    }


def write_executed_benchmark_output(
    output_dir: str | Path,
    *,
    config: RunConfig,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    summary: EvalSummary,
    backend_config: BackendConfig,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    run_config_path = out / "run_config.json"
    run_config_txt_path = out / "run_config.txt"
    rows_path = out / "rows.jsonl"
    summary_path = out / "summary.json"
    manifest_path = out / "sample_manifest.json"
    sources_path = out / BENCHMARK_SOURCES_FILENAME

    source_manifest = write_benchmark_source_manifest(
        sources_path,
        config=config,
        manifest=manifest,
    )
    runtime_config = replace(
        config,
        benchmark_source_manifest=benchmark_source_manifest_reference(source_manifest),
        execution_backend=backend_config.to_dict(),
    )

    _write_json(run_config_path, runtime_config)
    _write_json(manifest_path, manifest)
    _write_run_config_text(
        run_config_txt_path,
        config=runtime_config,
        manifest=manifest,
        backend_config=backend_config,
    )
    with rows_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(_to_jsonable(row), sort_keys=True) + "\n")

    summary_payload = summary.to_dict()
    summary_payload["parser_scorer"] = config.parser_scorer.to_dict()
    summary_payload["deepstack"] = config.deepstack.to_dict()
    summary_payload["deepstack_execution"] = summarize_deepstack_execution(rows)
    summary_payload["result_breakdowns"] = summarize_result_breakdowns(rows)
    summary_payload["post_tgvf_forward_mode"] = str(config.post_tgvf_forward_mode)
    summary_payload["post_tgvf_continuation"] = str(config.post_tgvf_continuation)
    summary_payload["runner_backend"] = backend_config.to_dict()
    summary_payload["benchmark_source_manifest"] = runtime_config.benchmark_source_manifest
    summary_payload["manifest_verification"] = build_manifest_verification_payload(
        config=runtime_config,
        manifest=manifest,
        rows=rows,
        source_manifest=source_manifest,
    )
    summary_payload["comparability"] = build_comparability_flags_payload(
        config=runtime_config,
        manifest=manifest,
        rows=rows,
        comparable=summary.comparable,
        comparability_note=summary.comparability_note,
    )
    _write_json(summary_path, summary_payload)

    return {
        "output_dir": str(out),
        "run_config": str(run_config_path),
        "run_config_txt": str(run_config_txt_path),
        "rows": str(rows_path),
        "summary": str(summary_path),
        "sample_manifest": str(manifest_path),
        "benchmark_sources": str(sources_path),
    }


def write_benchmark_source_manifest(
    path: str | Path,
    *,
    config: RunConfig,
    manifest: dict[str, Any] | SampleManifest | None,
) -> dict[str, Any]:
    payload = build_benchmark_source_manifest_payload(config=config, manifest=manifest)
    _write_json(Path(path), payload)
    return payload


def write_run_config_text(
    path: str | Path,
    *,
    config: RunConfig,
    manifest: dict[str, Any] | SampleManifest | None,
    backend_config: BackendConfig | None = None,
) -> None:
    _write_run_config_text(
        Path(path),
        config=config,
        manifest=manifest,
        backend_config=backend_config,
    )


def build_benchmark_source_manifest_payload(
    *,
    config: RunConfig,
    manifest: dict[str, Any] | SampleManifest | None,
) -> dict[str, Any]:
    manifest_info = _manifest_dict(manifest)
    root = Path(config.benchmark_root)
    samples = list(manifest_info.get("samples") or [])
    grouped: dict[str, dict[str, Any]] = {}
    for sample in samples:
        source_file = str(sample.get("source_file") or "")
        if not source_file:
            continue
        group = grouped.setdefault(
            source_file,
            {
                "source_file": source_file,
                "benchmarks": set(),
                "population_ids": set(),
                "row_indices": [],
                "sample_count": 0,
            },
        )
        if sample.get("benchmark"):
            group["benchmarks"].add(str(sample.get("benchmark")))
        if sample.get("population_id"):
            group["population_ids"].add(str(sample.get("population_id")))
        row_index = _sample_row_index(sample)
        if row_index is not None:
            group["row_indices"].append(row_index)
        group["sample_count"] += 1

    source_files = []
    for source_file, group in sorted(grouped.items()):
        path = root / source_file
        row_indices = sorted(group["row_indices"])
        source_files.append(
            {
                "source_file": source_file,
                "path": str(path),
                "exists": path.is_file(),
                "byte_size": path.stat().st_size if path.is_file() else None,
                "sha256": _sha256_file(path) if path.is_file() else None,
                "benchmarks": sorted(group["benchmarks"]),
                "population_ids": sorted(group["population_ids"]),
                "sample_count": int(group["sample_count"]),
                "row_index_count": len(row_indices),
                "row_index_min": min(row_indices) if row_indices else None,
                "row_index_max": max(row_indices) if row_indices else None,
                "row_indices": row_indices,
                "row_indices_sha256": _sha256_json(row_indices),
            }
        )

    return {
        "schema_version": "clean_benchmark_source_manifest_v1",
        "benchmark_root": str(root),
        "manifest_id": manifest_info.get("manifest_id"),
        "manifest_hash": manifest_info.get("manifest_hash") or config.manifest_hash,
        "source_manifest_id": manifest_info.get("source_manifest_id"),
        "source_manifest_hash": manifest_info.get("source_manifest_hash"),
        "source_file_count": len(source_files),
        "sample_count": len(samples),
        "all_files_exist": all(item["exists"] for item in source_files),
        "source_files": source_files,
    }


def benchmark_source_manifest_reference(
    payload: dict[str, Any],
    *,
    artifact_path: str = BENCHMARK_SOURCES_FILENAME,
) -> dict[str, Any]:
    return {
        "schema_version": payload.get("schema_version"),
        "artifact_path": artifact_path,
        "manifest_hash": payload.get("manifest_hash"),
        "source_manifest_hash": payload.get("source_manifest_hash"),
        "source_file_count": payload.get("source_file_count"),
        "sample_count": payload.get("sample_count"),
        "all_files_exist": payload.get("all_files_exist"),
        "source_files_sha256": _sha256_json(payload.get("source_files") or []),
    }


def build_manifest_verification_payload(
    *,
    config: RunConfig,
    manifest: dict[str, Any] | SampleManifest,
    rows: list[dict[str, Any]],
    source_manifest: dict[str, Any],
    merge_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_info = _manifest_dict(manifest)
    samples = list(manifest_info.get("samples") or [])
    sample_ids = [str(sample.get("sample_id") or "") for sample in samples]
    row_ids = [str(row.get("sample_id") or "") for row in rows]
    row_id_set = set(row_ids)
    sample_id_set = set(sample_ids)
    source_sample_count = int(source_manifest.get("sample_count") or 0)
    return {
        "schema_version": "clean_benchmark_manifest_verification_v1",
        "manifest_path": config.manifest_path,
        "manifest_id": manifest_info.get("manifest_id"),
        "manifest_hash": manifest_info.get("manifest_hash") or config.manifest_hash,
        "source_manifest_id": manifest_info.get("source_manifest_id"),
        "source_manifest_hash": manifest_info.get("source_manifest_hash"),
        "sample_manifest_sample_count": len(samples),
        "rows_count": len(rows),
        "row_count_matches_sample_manifest": len(rows) == len(samples),
        "row_sample_id_order_matches_manifest": row_ids == sample_ids,
        "missing_row_sample_ids": sorted(sample_id_set - row_id_set),
        "extra_row_sample_ids": sorted(row_id_set - sample_id_set),
        "source_manifest_sample_count": source_sample_count,
        "source_manifest_matches_sample_manifest": source_sample_count == len(samples),
        "source_file_count": source_manifest.get("source_file_count"),
        "all_source_files_exist": bool(source_manifest.get("all_files_exist")),
        "num_shards": config.num_shards,
        "shard_index": config.shard_index,
        "merged_from_shards": merge_metadata is not None,
        "merge_metadata": merge_metadata,
    }


def build_comparability_flags_payload(
    *,
    config: RunConfig,
    manifest: dict[str, Any] | SampleManifest,
    rows: list[dict[str, Any]],
    comparable: bool,
    comparability_note: str,
    merge_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_info = _manifest_dict(manifest)
    samples = list(manifest_info.get("samples") or [])
    source_population_ids = _source_population_ids(manifest_info)
    unknown_population_ids = sorted(
        population_id
        for population_id in source_population_ids
        if population_id not in CORE_POPULATIONS
    )
    subset_id = config.subset_id or (
        str(manifest_info.get("manifest_id"))
        if manifest_info.get("manifest_id") in SUBSETS
        else None
    )
    subset_spec = SUBSETS.get(str(subset_id)) if subset_id else None
    population_id = config.population_id
    population_spec = CORE_POPULATIONS.get(str(population_id)) if population_id else None
    clean_core = bool(source_population_ids) and not unknown_population_ids
    subset_expected_n = subset_spec.n if subset_spec is not None else None
    population_expected_n = population_spec.n if population_spec is not None else None
    sample_count = len(samples)
    full_clean_core = clean_core and sample_count == CORE_FULL_N
    diagnostic_subset = bool(subset_id and str(subset_id).startswith("diagnostic_"))
    return {
        "schema_version": "clean_benchmark_comparability_v1",
        "comparable": bool(comparable),
        "comparability_note": comparability_note,
        "eval_family": str(config.eval_family),
        "mode": str(config.mode),
        "manifest_id": manifest_info.get("manifest_id"),
        "manifest_hash": manifest_info.get("manifest_hash") or config.manifest_hash,
        "source_manifest_hash": manifest_info.get("source_manifest_hash"),
        "clean_core": clean_core,
        "clean_core_population_ids": [
            population_id
            for population_id in source_population_ids
            if population_id in CORE_POPULATIONS
        ],
        "non_core_population_ids": unknown_population_ids,
        "subset_run": subset_id is not None,
        "subset_id": subset_id,
        "subset_short_name": subset_spec.short_name if subset_spec is not None else None,
        "subset_expected_n": subset_expected_n,
        "subset_expected_n_matches": (
            sample_count == subset_expected_n if subset_expected_n is not None else None
        ),
        "diagnostic_subset": diagnostic_subset,
        "population_run": population_id is not None,
        "population_id": population_id,
        "population_expected_n": population_expected_n,
        "population_expected_n_matches": (
            sample_count == population_expected_n if population_expected_n is not None else None
        ),
        "full_clean_core": full_clean_core,
        "sample_count": sample_count,
        "rows_count": len(rows),
        "side_result": False,
        "side_result_reason": None,
        "invalid_for_baseline": False,
        "invalid_baseline_id": None,
        "invalid_reason": None,
        "merged_from_shards": merge_metadata is not None,
        "comparison_scope": _comparison_scope(
            clean_core=clean_core,
            full_clean_core=full_clean_core,
            subset_id=subset_id,
            diagnostic_subset=diagnostic_subset,
            population_id=population_id,
            population_expected_n_matches=(
                sample_count == population_expected_n
                if population_expected_n is not None
                else None
            ),
        ),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _manifest_dict(manifest: dict[str, Any] | SampleManifest | None) -> dict[str, Any]:
    if isinstance(manifest, SampleManifest):
        return manifest_payload(manifest)
    if manifest is None:
        return {"status": "not_loaded", "samples": []}
    return dict(manifest)


def _sample_row_index(sample: dict[str, Any]) -> int | None:
    metadata = sample.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("row_index") is None:
        return None
    try:
        return int(metadata["row_index"])
    except (TypeError, ValueError):
        return None


def _source_population_ids(manifest_info: dict[str, Any]) -> list[str]:
    values = manifest_info.get("source_population_ids")
    if isinstance(values, list) and values:
        return sorted({str(value) for value in values if value not in (None, "")})
    return sorted(
        {
            str(sample.get("population_id"))
            for sample in manifest_info.get("samples", [])
            if sample.get("population_id") not in (None, "")
        }
    )


def _comparison_scope(
    *,
    clean_core: bool,
    full_clean_core: bool,
    subset_id: str | None,
    diagnostic_subset: bool,
    population_id: str | None,
    population_expected_n_matches: bool | None,
) -> str:
    if diagnostic_subset:
        return "diagnostic_subset"
    if full_clean_core:
        return "clean_core_full"
    if subset_id:
        return "clean_core_subset" if clean_core else "custom_subset"
    if population_id and population_expected_n_matches:
        return "clean_core_population_full" if clean_core else "custom_population_full"
    if population_id:
        return "clean_core_population_subset" if clean_core else "custom_population_subset"
    return "custom_manifest"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(_to_jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_run_config_text(
    path: Path,
    *,
    config: RunConfig,
    manifest: dict[str, Any] | SampleManifest | None,
    backend_config: BackendConfig | None = None,
) -> None:
    if isinstance(manifest, SampleManifest):
        manifest_info = manifest_payload(manifest)
    elif manifest is None:
        manifest_info = {
            "manifest_id": config.subset_id or config.population_id,
            "manifest_path": config.manifest_path,
            "manifest_hash": config.manifest_hash,
            "status": "not_loaded",
        }
    else:
        manifest_info = manifest

    lines = [
        f"run_id: {config.run_id}",
        f"output_schema_version: {config.output_schema_version}",
        f"started_at: {config.started_at}",
        f"num_shards: {config.num_shards}",
        f"shard_index: {config.shard_index}",
        f"checkpoint_path: {config.checkpoint_path}",
        f"model_id: {config.model_id}",
        f"processor_id: {config.processor_id}",
        f"mode: {config.mode}",
        f"post_tgvf_forward_mode: {config.post_tgvf_forward_mode}",
        f"post_tgvf_continuation: {config.post_tgvf_continuation}",
        f"tgvf_protocol: {config.tgvf_protocol}",
        f"population_id: {config.population_id}",
        f"subset_id: {config.subset_id}",
        f"manifest_path: {config.manifest_path}",
        f"manifest_hash: {manifest_info.get('manifest_hash') or config.manifest_hash}",
        f"benchmark_root: {config.benchmark_root}",
        f"max_image_resolution: {config.max_image_resolution}",
        f"max_action_tokens: {config.max_action_tokens}",
        f"max_answer_tokens: {config.max_answer_tokens}",
        f"deepstack: {json.dumps(config.deepstack.to_dict(), sort_keys=True)}",
        f"parser_scorer: {json.dumps(config.parser_scorer.to_dict(), sort_keys=True)}",
        "benchmark_source_manifest: "
        f"{json.dumps(config.benchmark_source_manifest, sort_keys=True)}",
        f"execution_backend: {json.dumps(config.execution_backend, sort_keys=True)}",
        f"dirty_worktree: {config.dirty_worktree}",
        f"git_commit: {config.git_commit}",
    ]
    if backend_config is not None:
        lines.append(f"runner_backend: {json.dumps(backend_config.to_dict(), sort_keys=True)}")
    lines.extend(
        [
            "",
            "[run_config_json]",
            json.dumps(config.to_dict(), indent=2, sort_keys=True),
            "",
            "[sample_manifest_json]",
            json.dumps(_to_jsonable(manifest_info), indent=2, sort_keys=True),
        ]
    )
    if backend_config is not None:
        lines.extend(
            [
                "",
                "[runner_backend_json]",
                json.dumps(_to_jsonable(backend_config.to_dict()), indent=2, sort_keys=True),
            ]
        )
    path.write_text("\n".join(lines) + "\n")
