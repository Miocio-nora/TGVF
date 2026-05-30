from __future__ import annotations

import random
from collections import defaultdict
from typing import Protocol


class HasSampleId(Protocol):
    sample_id: str
    metadata: dict[str, object]


TIER_COUNTS: dict[str, dict[str, int | None]] = {
    "light": {
        "vstar_bench": 50,
        "hr_bench_4k": 50,
        "ocrbench_v2": 500,
        "blink": 280,
        "mmmu_pro": 300,
        "mathvista": 200,
        "mathverse": 300,
        "ovo_bench": 90,
    },
    "medium": {
        "vstar_bench": 191,
        "hr_bench_4k": 200,
        "ocrbench_v2": 2000,
        "blink": 700,
        "mmmu_pro": 900,
        "mathvista": 1000,
        "mathverse": 1200,
        "ovo_bench": 450,
    },
    "full": {
        "vstar_bench": None,
        "hr_bench_4k": None,
        "ocrbench_v2": None,
        "blink": None,
        "mmmu_pro": None,
        "mathvista": 1000,
        "mathverse": None,
        "ovo_bench": None,
    },
}


DEFAULT_STRATIFY_KEYS = (
    "category",
    "task",
    "task_type",
    "ability",
    "subject",
    "mode",
    "version",
    "subset",
)


def tier_limit(benchmark: str, tier: str, explicit_limit: int | None = None) -> int | None:
    if explicit_limit is not None:
        return explicit_limit
    if tier not in TIER_COUNTS:
        raise ValueError("tier must be one of light, medium, full")
    return TIER_COUNTS[tier].get(benchmark)


def deterministic_sample(
    samples: list[HasSampleId],
    *,
    benchmark: str,
    tier: str = "light",
    limit: int | None = None,
    seed: int = 20260525,
    stratify: bool = True,
) -> list[HasSampleId]:
    count = tier_limit(benchmark, tier, limit)
    ordered = sorted(samples, key=lambda sample: sample.sample_id)
    if count is None or count >= len(ordered):
        return ordered
    rng = random.Random(seed)
    if not stratify:
        shuffled = ordered[:]
        rng.shuffle(shuffled)
        return sorted(shuffled[:count], key=lambda sample: sample.sample_id)

    groups: dict[str, list[HasSampleId]] = defaultdict(list)
    for sample in ordered:
        groups[_stratum(sample)].append(sample)
    for group in groups.values():
        rng.shuffle(group)

    picked: list[HasSampleId] = []
    keys = sorted(groups)
    while len(picked) < count and keys:
        next_keys = []
        for key in keys:
            group = groups[key]
            if group and len(picked) < count:
                picked.append(group.pop())
            if group:
                next_keys.append(key)
        keys = next_keys
    return sorted(picked, key=lambda sample: sample.sample_id)


def _stratum(sample: HasSampleId) -> str:
    for key in DEFAULT_STRATIFY_KEYS:
        value = sample.metadata.get(key)
        if value:
            return f"{key}:{value}"
    return "all"
