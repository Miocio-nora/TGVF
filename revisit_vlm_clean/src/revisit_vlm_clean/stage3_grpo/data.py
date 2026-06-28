"""Stage3 GRPO data loading and deterministic sampling."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from revisit_vlm_clean.data_generation import file_identity

from .schemas import Stage3Sample

STAGE3_GRPO_SAMPLE_SCHEDULE_SCHEMA_VERSION = "stage3_grpo_sample_schedule_v0"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_stage3_samples(
    path: str | Path,
    *,
    limit: int | None = None,
    require_train_eligible: bool = True,
) -> list[Stage3Sample]:
    samples: list[Stage3Sample] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if require_train_eligible:
                metadata = record.get("rl_metadata") or {}
                if metadata.get("is_rl_train_eligible") is False:
                    continue
                if metadata.get("is_validation_reserved") is True:
                    continue
            sample = Stage3Sample.from_record(record)
            if not sample.sample_id or not sample.question:
                continue
            samples.append(sample)
            if limit is not None and len(samples) >= int(limit):
                break
    return samples


def dataset_identity(path: str | Path, *, max_rows: int = 5) -> dict[str, Any]:
    samples = load_stage3_samples(path, limit=None)
    return {
        "path": str(path),
        "file": file_identity(path).to_dict(),
        "rows": len(samples),
        "source_dataset": dict(Counter(sample.source_dataset for sample in samples)),
        "evidence_type": dict(Counter(sample.evidence_type for sample in samples)),
        "answer_type": dict(Counter(sample.answer_type for sample in samples)),
        "difficulty": dict(Counter(sample.difficulty for sample in samples)),
        "tool_need_hint": dict(Counter(sample.tool_need_hint or "unknown" for sample in samples)),
        "examples": [
            {
                "sample_id": sample.sample_id,
                "source_dataset": sample.source_dataset,
                "question": sample.question,
                "gold_answer": sample.gold_answer,
                "tool_need_hint": sample.tool_need_hint,
            }
            for sample in samples[:max_rows]
        ],
    }


def load_stage3_sample_schedule(path: str | Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    normalized: list[dict[str, Any]] = []
    seen_slots: set[tuple[int, int, int, int]] = set()
    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise ValueError(f"sample schedule row {index} missing sample_id")
        try:
            global_step = int(row.get("global_step"))
            rank = int(row.get("rank"))
            accumulation_index = int(row.get("accumulation_index", 0))
            prompt_index = int(row.get("prompt_index", 0))
        except Exception as exc:
            raise ValueError(f"sample schedule row {index} has invalid integer fields") from exc
        if global_step < 1:
            raise ValueError(f"sample schedule row {index} global_step must be >= 1")
        if rank < 0 or accumulation_index < 0 or prompt_index < 0:
            raise ValueError(f"sample schedule row {index} rank/accumulation/prompt must be >= 0")
        slot = (global_step, rank, accumulation_index, prompt_index)
        if slot in seen_slots:
            raise ValueError(f"sample schedule has duplicate slot: {slot}")
        seen_slots.add(slot)
        normalized.append(
            {
                **row,
                "schema_version": str(
                    row.get("schema_version") or STAGE3_GRPO_SAMPLE_SCHEDULE_SCHEMA_VERSION
                ),
                "global_step": global_step,
                "rank": rank,
                "accumulation_index": accumulation_index,
                "prompt_index": prompt_index,
                "sample_id": sample_id,
            }
        )
    return normalized


def sample_schedule_identity(
    path: str | Path,
    *,
    samples: list[Stage3Sample] | None = None,
    max_rows: int = 5,
) -> dict[str, Any]:
    rows = load_stage3_sample_schedule(path)
    sample_ids = [str(row["sample_id"]) for row in rows]
    image_uids = [str(row.get("stable_image_uid") or "") for row in rows if row.get("stable_image_uid")]
    known_sample_ids = {sample.sample_id for sample in samples or []}
    missing_sample_ids = (
        sorted(set(sample_ids) - known_sample_ids)[:20]
        if samples is not None
        else []
    )
    return {
        "path": str(path),
        "file": file_identity(path).to_dict(),
        "rows": len(rows),
        "step_count": len({int(row["global_step"]) for row in rows}),
        "rank_count": len({int(row["rank"]) for row in rows}),
        "duplicate_sample_ids": _duplicate_count(sample_ids),
        "duplicate_image_uids": _duplicate_count(image_uids),
        "missing_sample_ids": missing_sample_ids,
        "source_dataset": dict(Counter(str(row.get("source_dataset") or "unknown") for row in rows)),
        "tool_bucket": dict(Counter(str(row.get("tool_bucket") or "unknown") for row in rows)),
        "tool_need_hint": dict(Counter(str(row.get("tool_need_hint") or "unknown") for row in rows)),
        "examples": rows[:max_rows],
    }


def _duplicate_count(values: list[str]) -> int:
    counts = Counter(value for value in values if value)
    return sum(count - 1 for count in counts.values() if count > 1)


class BalancedPromptSampler:
    """Small deterministic sampler for Stage3 prompts.

    The first version balances primarily by tool bucket while preserving source
    diversity inside each bucket. It intentionally records bucket names instead
    of silently pretending the teacher hint is a forced-probe label.
    """

    def __init__(
        self,
        samples: list[Stage3Sample],
        *,
        seed: int,
        tool_bucket_weights: dict[str, float] | None = None,
    ) -> None:
        if not samples:
            raise ValueError("BalancedPromptSampler requires at least one sample")
        self.samples = list(samples)
        self.rng = random.Random(int(seed))
        self.weights = tool_bucket_weights or {
            "tool_helpful": 0.5,
            "tool_unnecessary": 0.3,
            "uncertain": 0.2,
        }
        buckets: dict[str, list[Stage3Sample]] = defaultdict(list)
        for sample in self.samples:
            buckets[tool_bucket_from_hint(sample.tool_need_hint)].append(sample)
        self.bucket_queues: dict[str, deque[Stage3Sample]] = {}
        for name, items in buckets.items():
            shuffled = list(items)
            self.rng.shuffle(shuffled)
            self.bucket_queues[name] = deque(shuffled)

    def next_batch(self, batch_size: int) -> list[Stage3Sample]:
        if int(batch_size) < 1:
            raise ValueError("batch_size must be >= 1")
        batch: list[Stage3Sample] = []
        bucket_names = list(self.weights)
        for _ in range(int(batch_size)):
            available = [name for name in bucket_names if self.bucket_queues.get(name)]
            if not available:
                self._reshuffle_all()
                available = [name for name in bucket_names if self.bucket_queues.get(name)]
            if not available:
                raise ValueError("sampler has no available buckets")
            weights = [max(0.0, float(self.weights.get(name, 0.0))) for name in available]
            if sum(weights) <= 0:
                weights = [1.0] * len(available)
            chosen = self.rng.choices(available, weights=weights, k=1)[0]
            batch.append(self.bucket_queues[chosen].popleft())
        return batch

    def summary(self) -> dict[str, Any]:
        return {
            "sample_count": len(self.samples),
            "weights": dict(self.weights),
            "bucket_counts": {
                name: len(queue)
                for name, queue in sorted(self.bucket_queues.items())
            },
        }

    def _reshuffle_all(self) -> None:
        buckets: dict[str, list[Stage3Sample]] = defaultdict(list)
        for sample in self.samples:
            buckets[tool_bucket_from_hint(sample.tool_need_hint)].append(sample)
        self.bucket_queues = {}
        for name, items in buckets.items():
            shuffled = list(items)
            self.rng.shuffle(shuffled)
            self.bucket_queues[name] = deque(shuffled)


def tool_bucket_from_hint(hint: str | None) -> str:
    value = str(hint or "").strip().lower()
    if value in {"likely_required", "useful_tool"}:
        return "tool_helpful"
    if value == "no_tool":
        return "tool_unnecessary"
    return "uncertain"
