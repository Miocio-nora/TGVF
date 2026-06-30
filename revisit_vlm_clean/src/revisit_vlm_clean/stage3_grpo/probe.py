"""Forced probe cache and ToolDecision labels for Stage3 GRPO."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import STAGE3_GRPO_PROBE_SCHEMA_VERSION, Stage3Sample


@dataclass(frozen=True)
class ToolDecisionLabel:
    label: str
    delta_tool: float | None
    source: str
    weight: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "delta_tool": self.delta_tool,
            "source": self.source,
            "weight": self.weight,
        }


class ProbeCache:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = None if path is None else Path(path)
        self.rows: dict[str, dict[str, Any]] = {}
        if self.path is not None and self.path.exists():
            with self.path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    sample_id = str(row.get("sample_id") or "")
                    if sample_id:
                        self.rows[sample_id] = row

    def get(self, sample_id: str) -> dict[str, Any] | None:
        return self.rows.get(str(sample_id))

    def label_for(
        self,
        sample: Stage3Sample,
        *,
        tau: float,
        missing_policy: str,
        hint_label_weight: float,
    ) -> ToolDecisionLabel:
        row = self.get(sample.sample_id)
        if row is not None:
            delta = row.get("delta_tool")
            delta_float = float(delta) if isinstance(delta, (int, float)) else None
            if delta_float is not None and delta_float > float(tau):
                return ToolDecisionLabel("tool_needed", delta_float, "forced_probe", 1.0)
            if delta_float is not None and delta_float < -float(tau):
                return ToolDecisionLabel("tool_unnecessary", delta_float, "forced_probe", 1.0)
            off = row.get("mean_correct_off")
            if delta_float is not None and abs(delta_float) <= float(tau) and _good_off(off):
                return ToolDecisionLabel("tool_unnecessary", delta_float, "forced_probe", 1.0)
            return ToolDecisionLabel("unknown", delta_float, "forced_probe", 0.0)
        if missing_policy == "teacher_hint":
            return label_from_tool_hint(sample.tool_need_hint, weight=hint_label_weight)
        return ToolDecisionLabel("unknown", None, "missing_probe", 0.0)

    def write_rows(self, path: str | Path, rows: list[dict[str, Any]]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                payload = {
                    "schema_version": STAGE3_GRPO_PROBE_SCHEMA_VERSION,
                    **row,
                }
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def label_from_tool_hint(hint: str | None, *, weight: float) -> ToolDecisionLabel:
    value = str(hint or "").strip().lower()
    if value in {"likely_required", "useful_tool"}:
        return ToolDecisionLabel("tool_needed", None, "teacher_hint", float(weight))
    if value == "optional_tool":
        return ToolDecisionLabel("tool_optional", None, "teacher_hint", float(weight))
    if value == "no_tool":
        return ToolDecisionLabel("tool_unnecessary", None, "teacher_hint", float(weight))
    return ToolDecisionLabel("unknown", None, "teacher_hint", 0.0)


def _good_off(value: Any) -> bool:
    return isinstance(value, (int, float)) and float(value) >= 0.75
