"""Output helpers for clean benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .manifest import SampleManifest, manifest_payload
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
        "rows": str(rows_path),
        "summary": str(summary_path),
        "sample_manifest": str(manifest_path),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True) + "\n")
