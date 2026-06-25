"""Sample manifest interfaces for clean benchmark runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .populations import SubsetSpec, get_subset
from .schema import _to_jsonable


@dataclass(frozen=True)
class SampleRef:
    sample_id: str
    benchmark: str
    population_id: str
    source_file: str
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class SampleManifest:
    manifest_id: str
    seed: int
    samples: tuple[SampleRef, ...]
    source_population_ids: tuple[str, ...]
    stratification: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)

    def stable_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


def describe_subset(subset_id: str) -> SubsetSpec:
    return get_subset(subset_id)


def build_manifest(*, subset_id: str, benchmark_root: str) -> SampleManifest:
    _ = describe_subset(subset_id)
    _ = benchmark_root
    raise NotImplementedError("manifest generation is phase 2; this phase defines the interface only")
