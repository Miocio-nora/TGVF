from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class RunPaths:
    root: Path
    predictions: Path
    raw_outputs: Path
    scores: Path
    reports: Path
    logs: Path
    artifacts: Path


def make_run_paths(output_root: str | Path, run_id: str | None = None) -> RunPaths:
    run = run_id or datetime.now(timezone.utc).strftime("eval_%Y%m%d_%H%M%S")
    root = Path(output_root) / "runs" / run
    paths = RunPaths(
        root=root,
        predictions=root / "predictions",
        raw_outputs=root / "raw_outputs",
        scores=root / "scores",
        reports=root / "reports",
        logs=root / "logs",
        artifacts=root / "artifacts",
    )
    for path in asdict(paths).values():
        Path(path).mkdir(parents=True, exist_ok=True)
    return paths


def write_config_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    # JSON is valid YAML 1.2 and avoids adding a hard PyYAML dependency.
    Path(path).write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n")


class ResultWriter:
    def __init__(self, paths: RunPaths, *, benchmark: str, method: str) -> None:
        self.paths = paths
        self.benchmark = benchmark
        self.method = method
        self.prediction_path = paths.predictions / f"{benchmark}__{method}.jsonl"
        self.raw_path = paths.raw_outputs / f"{benchmark}__{method}.jsonl"
        self.summary_path = paths.reports / f"{benchmark}__{method}.summary.json"

    def completed_ids(self) -> set[str]:
        if not self.prediction_path.exists():
            return set()
        ids = set()
        with self.prediction_path.open() as handle:
            for line in handle:
                if line.strip():
                    try:
                        row = json.loads(line)
                        ids.add(str(row.get("sample_id")))
                    except json.JSONDecodeError:
                        continue
        return ids

    def append_row(self, row: dict[str, Any]) -> None:
        payload = json.dumps(_jsonable(row), ensure_ascii=False)
        with self.prediction_path.open("a") as handle:
            handle.write(payload + "\n")
        with self.raw_path.open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "benchmark": row.get("benchmark"),
                        "sample_id": row.get("sample_id"),
                        "method": row.get("method"),
                        "raw_output": row.get("raw_output"),
                        "error": row.get("error"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def read_rows(self) -> list[dict[str, Any]]:
        if not self.prediction_path.exists():
            return []
        rows = []
        with self.prediction_path.open() as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def write_summary(self, summary: dict[str, Any]) -> None:
        self.summary_path.write_text(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False) + "\n")


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    benchmark: str,
    tier: str,
    method: str,
    cot_enabled: bool,
    scorer_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scored = [float(row["score"]) for row in rows if row.get("score") is not None]
    triggered = [row for row in rows if row.get("triggered")]
    not_triggered = [row for row in rows if not row.get("triggered")]
    wall_times = [float(row.get("wall_time_sec") or 0.0) for row in rows]
    visual_tokens = [float(row.get("visual_token_count") or 0.0) for row in rows]
    foveations = [float(row.get("num_foveations") or 0.0) for row in rows]
    summary = {
        "run_id": run_id,
        "benchmark": benchmark,
        "tier": tier,
        "method": method,
        "num_samples": len(rows),
        "score": _mean(scored),
        "trigger_rate": len(triggered) / max(1, len(rows)),
        "accuracy_triggered_only": _mean([float(row["score"]) for row in triggered if row.get("score") is not None]),
        "accuracy_not_triggered": _mean([float(row["score"]) for row in not_triggered if row.get("score") is not None]),
        "avg_num_foveations": _mean(foveations),
        "avg_wall_time_sec": _mean(wall_times),
        "p50_wall_time_sec": _quantile(wall_times, 0.5),
        "p90_wall_time_sec": _quantile(wall_times, 0.9),
        "avg_visual_tokens": _mean(visual_tokens),
        "cot_enabled": cot_enabled,
        "second_full_forward_used": any(bool(row.get("second_full_forward_used")) for row in rows),
    }
    if scorer_payload:
        summary["scoring"] = scorer_payload
        if scorer_payload.get("score") is not None:
            summary["score"] = scorer_payload["score"]
    return summary


def _looks_like_image(value: Any) -> bool:
    if not hasattr(value, "size"):
        return False
    module = type(value).__module__
    return module.startswith("PIL.") or hasattr(value, "filename")


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return float(statistics.quantiles(values, n=100)[max(0, min(99, int(q * 100) - 1))])


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if _looks_like_image(value):
        filename = getattr(value, "filename", "") or "<embedded_image>"
        return {"type": "image", "filename": filename, "size": list(getattr(value, "size", ())) }
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)
