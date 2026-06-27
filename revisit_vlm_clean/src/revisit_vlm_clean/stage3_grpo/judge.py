"""Offline/cache-first judge wrapper for Stage3 GRPO rewards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .schemas import STAGE3_GRPO_JUDGE_SCHEMA_VERSION, JudgeConfig, RolloutRecord, Stage3Sample


class JudgeCache:
    def __init__(self, *, kind: str, config: JudgeConfig, path: str | Path | None) -> None:
        self.kind = kind
        self.config = config
        self.path = None if path is None else Path(path)
        self.rows: dict[str, dict[str, Any]] = {}
        if self.path is not None and self.path.exists():
            with self.path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    key = str(row.get("cache_key") or "")
                    if key:
                        self.rows[key] = row

    def lookup(self, sample: Stage3Sample, rollout: RolloutRecord) -> dict[str, Any] | None:
        return self.rows.get(judge_cache_key(self.kind, self.config, sample, rollout))

    def pending_payload(self, sample: Stage3Sample, rollout: RolloutRecord) -> dict[str, Any]:
        return {
            "schema_version": STAGE3_GRPO_JUDGE_SCHEMA_VERSION,
            "kind": self.kind,
            "cache_key": judge_cache_key(self.kind, self.config, sample, rollout),
            "judge_model": self.config.model,
            "prompt_version": self.config.prompt_version,
            "sample_id": sample.sample_id,
            "image_path": sample.image_path,
            "image_sha256": sample.image_sha256,
            "question": sample.question,
            "choices": list(sample.choices),
            "target": rollout.targets[0] if rollout.targets else "",
            "post_tool_reasoning": rollout.post_tool_reasoning,
            "final_answer": rollout.final_answer,
            "status": "pending",
        }


class JudgeBundle:
    def __init__(self, config: JudgeConfig, *, output_dir: str | Path | None = None) -> None:
        self.config = config
        self.focus = JudgeCache(kind="focus", config=config, path=config.focus_cache_path)
        self.ground = JudgeCache(kind="grounding", config=config, path=config.grounding_cache_path)
        if config.pending_path:
            self.pending_path = Path(config.pending_path)
        elif output_dir is not None:
            self.pending_path = Path(output_dir) / "judge_pending.jsonl"
        else:
            self.pending_path = None
        self._pending_keys: set[str] = set()

    def focus_score(self, sample: Stage3Sample, rollout: RolloutRecord) -> dict[str, Any] | None:
        return self._lookup_or_pending(self.focus, sample, rollout)

    def grounding_score(self, sample: Stage3Sample, rollout: RolloutRecord) -> dict[str, Any] | None:
        return self._lookup_or_pending(self.ground, sample, rollout)

    def _lookup_or_pending(
        self,
        cache: JudgeCache,
        sample: Stage3Sample,
        rollout: RolloutRecord,
    ) -> dict[str, Any] | None:
        if not self.config.enabled or self.config.mode == "disabled":
            return None
        row = cache.lookup(sample, rollout)
        if row is not None:
            return row
        if self.config.mode in {"cache_only", "log_only", "offline"}:
            payload = cache.pending_payload(sample, rollout)
            key = str(payload["cache_key"])
            if self.pending_path is not None and key not in self._pending_keys:
                self.pending_path.parent.mkdir(parents=True, exist_ok=True)
                with self.pending_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                self._pending_keys.add(key)
        return None


def judge_cache_key(
    kind: str,
    config: JudgeConfig,
    sample: Stage3Sample,
    rollout: RolloutRecord,
) -> str:
    payload = {
        "kind": kind,
        "model": config.model,
        "prompt_version": config.prompt_version,
        "image_sha256": sample.image_sha256,
        "image_path": sample.image_path,
        "question": sample.question,
        "target": rollout.targets[0] if rollout.targets else "",
        "post_tool_reasoning": rollout.post_tool_reasoning,
        "final_answer": rollout.final_answer,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def judge_score_value(row: dict[str, Any] | None, *, field: str, zero_reward: float, missing_reward: float) -> float:
    if row is None:
        return float(missing_reward)
    score = row.get(field)
    if score is None:
        score = row.get("score")
    try:
        score_int = int(score)
    except Exception:
        return float(missing_reward)
    if score_int >= 2:
        return 1.0
    if score_int == 1:
        return 0.5
    return float(zero_reward)
