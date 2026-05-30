from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OfficialToolInfo:
    official_tool_used: bool
    official_tool_path: str | None
    scorer_name: str
    prompt_source: str
    note: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "official_tool_used": self.official_tool_used,
            "official_tool_path": self.official_tool_path,
            "scorer_name": self.scorer_name,
            "prompt_source": self.prompt_source,
            "note": self.note,
        }


def resolve_official_tool(
    benchmark: str,
    *,
    benchmark_root: str | Path,
    tools_root: str | Path,
) -> OfficialToolInfo:
    root = Path(benchmark_root)
    tools = Path(tools_root)
    candidates = [
        tools / benchmark,
        tools / f"{benchmark}.py",
        root / benchmark / "official_code",
        root / benchmark / "snapshot" / "eval.py",
        root / benchmark / "snapshot" / "evaluate.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return OfficialToolInfo(
                official_tool_used=False,
                official_tool_path=str(candidate),
                scorer_name="fallback",
                prompt_source="project",
                note="official tool path found; adapter-specific scorer invocation is not wired in v0",
            )
    manifest = tools / "benchmark_manifest.json"
    if manifest.exists():
        try:
            entries = json.loads(manifest.read_text())
            if any(entry.get("name") == benchmark for entry in entries):
                return OfficialToolInfo(
                    official_tool_used=False,
                    official_tool_path=None,
                    scorer_name="fallback",
                    prompt_source="project",
                    note="benchmark is in manifest, but no local official scorer path was found",
                )
        except Exception as exc:  # pragma: no cover - diagnostic branch
            return OfficialToolInfo(
                official_tool_used=False,
                official_tool_path=None,
                scorer_name="fallback",
                prompt_source="project",
                note=f"could not parse tool manifest: {exc}",
            )
    return OfficialToolInfo(
        official_tool_used=False,
        official_tool_path=None,
        scorer_name="fallback",
        prompt_source="project",
        note="no official tool found",
    )
