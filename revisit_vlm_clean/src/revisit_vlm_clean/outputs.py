"""Output helpers for clean benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .benchmark_data import BenchmarkSample
from .manifest import SampleManifest, manifest_payload
from .rendering import RenderedBenchmarkInput
from .runner import BackendConfig
from .schema import EvalSummary, RunConfig, _to_jsonable


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

    _write_json(run_config_path, config)
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

    _write_run_config_text(run_config_txt_path, config=config, manifest=manifest)
    summary = EvalSummary(
        run_id=config.run_id,
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
    _write_json(summary_path, summary_payload)

    return {
        "output_dir": str(out),
        "run_config": str(run_config_path),
        "run_config_txt": str(run_config_txt_path),
        "rows": str(rows_path),
        "summary": str(summary_path),
        "sample_manifest": str(manifest_path),
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

    _write_json(run_config_path, config)
    _write_json(manifest_path, manifest)
    _write_run_config_text(run_config_txt_path, config=config, manifest=manifest)
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
    _write_json(summary_path, summary_payload)

    return {
        "output_dir": str(out),
        "run_config": str(run_config_path),
        "run_config_txt": str(run_config_txt_path),
        "materialized_rows": str(materialized_rows_path),
        "summary": str(summary_path),
        "sample_manifest": str(manifest_path),
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

    _write_json(run_config_path, config)
    _write_json(manifest_path, manifest)
    _write_run_config_text(run_config_txt_path, config=config, manifest=manifest)
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
    _write_json(summary_path, summary_payload)

    return {
        "output_dir": str(out),
        "run_config": str(run_config_path),
        "run_config_txt": str(run_config_txt_path),
        "rendered_inputs": str(rendered_inputs_path),
        "summary": str(summary_path),
        "sample_manifest": str(manifest_path),
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

    _write_json(run_config_path, config)
    _write_json(manifest_path, manifest)
    _write_run_config_text(
        run_config_txt_path,
        config=config,
        manifest=manifest,
        backend_config=backend_config,
    )
    with rows_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(_to_jsonable(row), sort_keys=True) + "\n")

    summary_payload = summary.to_dict()
    summary_payload["parser_scorer"] = config.parser_scorer.to_dict()
    summary_payload["deepstack"] = config.deepstack.to_dict()
    summary_payload["post_tgvf_forward_mode"] = str(config.post_tgvf_forward_mode)
    summary_payload["post_tgvf_continuation"] = str(config.post_tgvf_continuation)
    summary_payload["runner_backend"] = backend_config.to_dict()
    _write_json(summary_path, summary_payload)

    return {
        "output_dir": str(out),
        "run_config": str(run_config_path),
        "run_config_txt": str(run_config_txt_path),
        "rows": str(rows_path),
        "summary": str(summary_path),
        "sample_manifest": str(manifest_path),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True) + "\n")


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
        f"max_image_resolution: {config.max_image_resolution}",
        f"max_action_tokens: {config.max_action_tokens}",
        f"max_answer_tokens: {config.max_answer_tokens}",
        f"deepstack: {json.dumps(config.deepstack.to_dict(), sort_keys=True)}",
        f"parser_scorer: {json.dumps(config.parser_scorer.to_dict(), sort_keys=True)}",
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
