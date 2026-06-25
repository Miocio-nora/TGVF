from revisit_vlm_clean.populations import CORE_FULL_N, get_subset, iter_core_populations


def test_core_full_count_matches_population_sum() -> None:
    assert CORE_FULL_N == 19562
    assert sum(population.n for population in iter_core_populations()) == CORE_FULL_N


def test_coredev_2511_allocation() -> None:
    subset = get_subset("core_balanced_dev_2511_seed20260625")
    assert subset.short_name == "CoreDev-2511"
    assert subset.n == 2511
    assert sum(item.n for item in subset.allocations) == subset.n
