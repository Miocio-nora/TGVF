"""Clean benchmark population and subset contracts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .defaults import DEFAULT_CORE_DEV_SHORT_NAME, DEFAULT_CORE_DEV_SUBSET_ID


@dataclass(frozen=True)
class PopulationSpec:
    population_id: str
    benchmark: str
    source_files: tuple[str, ...]
    n: int
    status: str = "clean_core"


@dataclass(frozen=True)
class SubsetAllocation:
    population_id: str
    n: int
    stratification_rule: str


@dataclass(frozen=True)
class SubsetSpec:
    subset_id: str
    short_name: str
    n: int
    purpose: str
    allocations: tuple[SubsetAllocation, ...] = ()


CORE_POPULATIONS: dict[str, PopulationSpec] = {
    "vstar_test_questions_191": PopulationSpec(
        population_id="vstar_test_questions_191",
        benchmark="vstar_bench",
        source_files=("vstar_bench/snapshot/test_questions.jsonl",),
        n=191,
    ),
    "hr_bench_4k_800": PopulationSpec(
        population_id="hr_bench_4k_800",
        benchmark="hr_bench_4k",
        source_files=("hr_bench_4k/snapshot/hr_bench_4k.parquet",),
        n=800,
    ),
    "blink_val_all_subtasks_1901": PopulationSpec(
        population_id="blink_val_all_subtasks_1901",
        benchmark="blink",
        source_files=("blink/snapshot/*/val-*.parquet",),
        n=1901,
    ),
    "ocrbench_v2_data_test_10000": PopulationSpec(
        population_id="ocrbench_v2_data_test_10000",
        benchmark="ocrbench_v2",
        source_files=("ocrbench_v2/snapshot/data/test-*.parquet",),
        n=10000,
    ),
    "mmmu_pro_standard10_test_1730": PopulationSpec(
        population_id="mmmu_pro_standard10_test_1730",
        benchmark="mmmu_pro",
        source_files=("mmmu_pro/snapshot/standard (10 options)/test-*.parquet",),
        n=1730,
    ),
    "mathvista_testmini_1000": PopulationSpec(
        population_id="mathvista_testmini_1000",
        benchmark="mathvista",
        source_files=("mathvista/snapshot/data/testmini-00000-of-00001-725687bf7a18d64b.parquet",),
        n=1000,
    ),
    "mathverse_testmini_3940": PopulationSpec(
        population_id="mathverse_testmini_3940",
        benchmark="mathverse",
        source_files=("mathverse/snapshot/testmini.json",),
        n=3940,
    ),
}


CORE_FULL_N = sum(population.n for population in CORE_POPULATIONS.values())


SUBSETS: dict[str, SubsetSpec] = {
    "core_smoke_256_seed20260625": SubsetSpec(
        subset_id="core_smoke_256_seed20260625",
        short_name="CoreSmoke-256",
        n=256,
        purpose="Fast code/parser/scorer/DeepStack field smoke. Not for effect conclusions.",
        allocations=(
            SubsetAllocation("vstar_test_questions_191", 32, "stratified smoke slice"),
            SubsetAllocation("hr_bench_4k_800", 32, "stratified smoke slice"),
            SubsetAllocation("blink_val_all_subtasks_1901", 48, "stratified smoke slice"),
            SubsetAllocation("ocrbench_v2_data_test_10000", 48, "stratified smoke slice"),
            SubsetAllocation("mmmu_pro_standard10_test_1730", 32, "stratified smoke slice"),
            SubsetAllocation("mathvista_testmini_1000", 32, "stratified smoke slice"),
            SubsetAllocation("mathverse_testmini_3940", 32, "stratified smoke slice"),
        ),
    ),
    DEFAULT_CORE_DEV_SUBSET_ID: SubsetSpec(
        subset_id=DEFAULT_CORE_DEV_SUBSET_ID,
        short_name=DEFAULT_CORE_DEV_SHORT_NAME,
        n=2511,
        purpose="Main fast experimental comparison subset with balanced benchmark/task coverage.",
        allocations=(
            SubsetAllocation("vstar_test_questions_191", 191, "full population"),
            SubsetAllocation(
                "hr_bench_4k_800",
                200,
                "balance category, cycle_category, and answer",
            ),
            SubsetAllocation(
                "blink_val_all_subtasks_1901",
                420,
                "14 subtasks, 30 rows per subtask",
            ),
            SubsetAllocation(
                "ocrbench_v2_data_test_10000",
                600,
                "30 type values, 20 rows per type",
            ),
            SubsetAllocation(
                "mmmu_pro_standard10_test_1730",
                300,
                "30 subjects, 10 rows per subject",
            ),
            SubsetAllocation(
                "mathvista_testmini_1000",
                300,
                "balance question_type and answer_type",
            ),
            SubsetAllocation(
                "mathverse_testmini_3940",
                500,
                "5 problem_version values, 100 rows each, balanced across question_type "
                "where possible",
            ),
        ),
    ),
    "core_full_19562": SubsetSpec(
        subset_id="core_full_19562",
        short_name="CoreFull-19562",
        n=CORE_FULL_N,
        purpose="Final full clean image-core confirmation.",
        allocations=tuple(
            SubsetAllocation(population.population_id, population.n, "full population")
            for population in CORE_POPULATIONS.values()
        ),
    ),
    "diagnostic_vstar_first_1_20260626": SubsetSpec(
        subset_id="diagnostic_vstar_first_1_20260626",
        short_name="DiagVStarFirst-1",
        n=1,
        purpose="One fixed VStar row for runner/backend smoke validation only.",
        allocations=(
            SubsetAllocation(
                "vstar_test_questions_191",
                1,
                "first vstar sample from committed CoreSmoke manifest",
            ),
        ),
    ),
    "diagnostic_vstar_core_smoke_first_8_20260626": SubsetSpec(
        subset_id="diagnostic_vstar_core_smoke_first_8_20260626",
        short_name="DiagVStarCoreSmokeFirst-8",
        n=8,
        purpose="Eight fixed VStar rows for runner/backend smoke validation only.",
        allocations=(
            SubsetAllocation(
                "vstar_test_questions_191",
                8,
                "first eight VStar samples from committed CoreSmoke manifest",
            ),
        ),
    ),
    "diagnostic_vstar_core_smoke_32_20260626": SubsetSpec(
        subset_id="diagnostic_vstar_core_smoke_32_20260626",
        short_name="DiagVStarCoreSmoke-32",
        n=32,
        purpose="Thirty-two fixed VStar rows for runner/backend smoke validation only.",
        allocations=(
            SubsetAllocation(
                "vstar_test_questions_191",
                32,
                "all VStar rows from committed CoreSmoke manifest",
            ),
        ),
    ),
}


def iter_core_populations() -> Iterable[PopulationSpec]:
    return CORE_POPULATIONS.values()


def iter_subsets() -> Iterable[SubsetSpec]:
    return SUBSETS.values()


def get_population(population_id: str) -> PopulationSpec:
    try:
        return CORE_POPULATIONS[population_id]
    except KeyError as exc:
        raise ValueError(f"unknown clean population id: {population_id}") from exc


def get_subset(subset_id: str) -> SubsetSpec:
    try:
        return SUBSETS[subset_id]
    except KeyError as exc:
        raise ValueError(f"unknown clean subset id: {subset_id}") from exc
