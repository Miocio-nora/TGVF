"""Sample manifest generation for clean benchmark runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .populations import PopulationSpec, SubsetSpec, get_population, get_subset
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
    subset = describe_subset(subset_id)
    root = Path(benchmark_root)
    if not root.exists():
        raise FileNotFoundError(f"benchmark root does not exist: {root}")

    selected: list[SampleRef] = []
    stratification: dict[str, Any] = {
        "subset_id": subset.subset_id,
        "short_name": subset.short_name,
        "seed": 20260625,
        "allocations": [],
    }
    for allocation in subset.allocations:
        population = get_population(allocation.population_id)
        samples = load_population_samples(population.population_id, benchmark_root=root)
        if len(samples) != population.n:
            raise ValueError(
                f"{population.population_id} expected n={population.n}, loaded n={len(samples)}"
            )
        chosen = _select_samples(
            samples,
            n=allocation.n,
            population_id=population.population_id,
            seed=20260625,
        )
        selected.extend(chosen)
        stratification["allocations"].append(
            {
                "population_id": population.population_id,
                "requested_n": allocation.n,
                "selected_n": len(chosen),
                "stratification_rule": allocation.stratification_rule,
                "counts_by_stratum": _counts_by_stratum(
                    chosen, _stratification_keys(population.population_id)
                ),
                "source_files": [
                    str(path.relative_to(root))
                    for path in _resolve_population_files(population, root)
                ],
            }
        )

    if len(selected) != subset.n:
        raise ValueError(f"{subset.subset_id} expected n={subset.n}, selected n={len(selected)}")

    return SampleManifest(
        manifest_id=subset.subset_id,
        seed=20260625,
        samples=tuple(selected),
        source_population_ids=tuple(allocation.population_id for allocation in subset.allocations),
        stratification=stratification,
    )


def load_population_samples(population_id: str, *, benchmark_root: str | Path) -> list[SampleRef]:
    root = Path(benchmark_root)
    population = get_population(population_id)
    files = _resolve_population_files(population, root)
    samples: list[SampleRef] = []
    for path in files:
        rel = str(path.relative_to(root))
        for row_index, record in enumerate(
            _read_records(path, columns=_read_columns(population_id))
        ):
            metadata = _metadata_for_record(population_id, record, path)
            metadata["row_index"] = row_index
            metadata["raw_id"] = _raw_id(record, row_index)
            sample_id = _sample_id(population_id, rel, metadata["raw_id"], row_index)
            samples.append(
                SampleRef(
                    sample_id=sample_id,
                    benchmark=population.benchmark,
                    population_id=population.population_id,
                    source_file=rel,
                    metadata=metadata,
                )
            )
    return samples


def manifest_payload(manifest: SampleManifest) -> dict[str, Any]:
    payload = manifest.to_dict()
    payload["manifest_hash"] = manifest.stable_hash()
    return payload


def write_manifest(path: str | Path, manifest: SampleManifest) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(manifest_payload(manifest), indent=2, sort_keys=True) + "\n")


def _resolve_population_files(population: PopulationSpec, root: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in population.source_files:
        matches = sorted(root.glob(pattern))
        if not matches:
            candidate = root / pattern
            if candidate.exists():
                matches = [candidate]
        files.extend(path for path in matches if path.is_file())
    files = sorted(dict.fromkeys(files))
    if not files:
        raise FileNotFoundError(f"no files resolved for {population.population_id}")
    return files


def _read_records(path: Path, *, columns: tuple[str, ...]) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = []
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        return records
    if suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            return [dict(item) for item in payload]
        if isinstance(payload, dict):
            for key in ("data", "questions", "annotations", "examples"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [dict(item) for item in value]
            return [
                dict(value, id=key) if isinstance(value, dict) else {"id": key, "value": value}
                for key, value in payload.items()
            ]
        raise ValueError(f"unsupported json payload at {path}")
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except Exception as exc:  # pragma: no cover - environment-specific
            raise RuntimeError("pyarrow is required to build parquet-backed manifests") from exc
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema.names)
        selected = [column for column in columns if column in available]
        table = pq.read_table(path, columns=selected or None)
        return [dict(row) for row in table.to_pylist()]
    raise ValueError(f"unsupported manifest source file: {path}")


def _read_columns(population_id: str) -> tuple[str, ...]:
    return {
        "vstar_test_questions_191": ("question_id", "category", "label"),
        "hr_bench_4k_800": ("index", "category", "cycle_category", "answer"),
        "blink_val_all_subtasks_1901": ("idx", "answer"),
        "ocrbench_v2_data_test_10000": ("id", "dataset_name", "type"),
        "mmmu_pro_standard10_test_1730": ("id", "subject", "answer"),
        "mathvista_testmini_1000": ("pid", "question_type", "answer_type", "answer"),
        "mathverse_testmini_3940": (
            "sample_index",
            "problem_index",
            "problem_version",
            "question_type",
            "answer",
        ),
    }[population_id]


def _metadata_for_record(population_id: str, record: dict[str, Any], path: Path) -> dict[str, Any]:
    keys = _read_columns(population_id)
    metadata = {key: _json_scalar(record.get(key)) for key in keys if key in record}
    if population_id == "blink_val_all_subtasks_1901":
        metadata["sub_task"] = path.parent.name
    return metadata


def _stratification_keys(population_id: str) -> tuple[str, ...]:
    return {
        "vstar_test_questions_191": ("category", "label"),
        "hr_bench_4k_800": ("category", "cycle_category", "answer"),
        "blink_val_all_subtasks_1901": ("sub_task",),
        "ocrbench_v2_data_test_10000": ("type",),
        "mmmu_pro_standard10_test_1730": ("subject",),
        "mathvista_testmini_1000": ("question_type", "answer_type"),
        "mathverse_testmini_3940": ("problem_version", "question_type"),
    }[population_id]


def _select_samples(
    samples: list[SampleRef],
    *,
    n: int,
    population_id: str,
    seed: int,
) -> list[SampleRef]:
    if n >= len(samples):
        return list(samples)
    keys = _stratification_keys(population_id)
    groups: dict[tuple[str, ...], list[SampleRef]] = {}
    for sample in samples:
        metadata = sample.metadata or {}
        group_key = tuple(str(metadata.get(key, "")) for key in keys)
        groups.setdefault(group_key, []).append(sample)
    for group_samples in groups.values():
        group_samples.sort(key=lambda sample: _stable_key(seed, sample.sample_id))
    selected: list[SampleRef] = []
    group_keys = sorted(groups)
    while len(selected) < n and group_keys:
        next_group_keys: list[tuple[str, ...]] = []
        for group_key in group_keys:
            group = groups[group_key]
            if group and len(selected) < n:
                selected.append(group.pop(0))
            if group:
                next_group_keys.append(group_key)
        group_keys = next_group_keys
    return selected


def _counts_by_stratum(samples: list[SampleRef], keys: tuple[str, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        metadata = sample.metadata or {}
        key = "|".join(str(metadata.get(item, "")) for item in keys)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _raw_id(record: dict[str, Any], row_index: int) -> str:
    for key in ("question_id", "id", "uid", "sample_id", "idx", "pid", "sample_index", "index"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return str(row_index)


def _sample_id(population_id: str, source_file: str, raw_id: str, row_index: int) -> str:
    return f"{population_id}/{_slug(source_file)}/{_slug(raw_id)}_{row_index:06d}"


def _slug(value: Any) -> str:
    text = str(value)
    chars = [char if char.isalnum() else "_" for char in text]
    return "_".join("".join(chars).split("_")).strip("_")


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
