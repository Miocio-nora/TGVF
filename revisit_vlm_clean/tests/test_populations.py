from revisit_vlm_clean.populations import CORE_FULL_N, get_subset, iter_core_populations


def test_core_full_count_matches_population_sum() -> None:
    assert CORE_FULL_N == 19562
    assert sum(population.n for population in iter_core_populations()) == CORE_FULL_N


def test_coredev_2511_allocation() -> None:
    subset = get_subset("core_balanced_dev_2511_seed20260625")
    assert subset.short_name == "CoreDev-2511"
    assert subset.n == 2511
    assert sum(item.n for item in subset.allocations) == subset.n


def test_diagnostic_vstar_subsets_are_registered() -> None:
    first = get_subset("diagnostic_vstar_first_1_20260626")
    first_8 = get_subset("diagnostic_vstar_core_smoke_first_8_20260626")
    smoke_32 = get_subset("diagnostic_vstar_core_smoke_32_20260626")

    assert first.short_name == "DiagVStarFirst-1"
    assert first.n == 1
    assert first.allocations[0].population_id == "vstar_test_questions_191"
    assert first_8.n == 8
    assert smoke_32.n == 32
    assert "smoke validation only" in smoke_32.purpose
