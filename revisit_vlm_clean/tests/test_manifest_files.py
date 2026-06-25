import json
from pathlib import Path


MANIFEST_DIR = Path(__file__).resolve().parents[1] / "benchmark_manifests"


def test_committed_manifest_counts_and_hashes() -> None:
    expected = {
        "core_smoke_256_seed20260625.json": (
            256,
            "7da4963199c7d75baee224e52049625129a0f9335d156dd84efd652b2df0c036",
        ),
        "core_balanced_dev_2511_seed20260625.json": (
            2511,
            "a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579",
        ),
        "core_full_19562.json": (
            19562,
            "1b2942590ff4eada644b51507acfde461f6049e87a96486d061d71a3f1de0352",
        ),
        "diagnostic_vstar_first_1_20260626.json": (
            1,
            "851e301ea0730ee90086c83565135fa5c5fcc5d33b962fd074444d85241d6995",
        ),
        "diagnostic_vstar_core_smoke_first_8_20260626.json": (
            8,
            "3a6020b145c1543be3fc8a5541c17d00b4fc5d27aeddbecb90a10417d25c84f4",
        ),
    }
    for name, (count, digest) in expected.items():
        payload = json.loads((MANIFEST_DIR / name).read_text())
        assert len(payload["samples"]) == count
        assert payload["manifest_hash"] == digest


def test_coredev_distribution_contract() -> None:
    payload = json.loads((MANIFEST_DIR / "core_balanced_dev_2511_seed20260625.json").read_text())
    allocations = {
        item["population_id"]: item["counts_by_stratum"]
        for item in payload["stratification"]["allocations"]
    }
    assert set(allocations["blink_val_all_subtasks_1901"].values()) == {30}
    assert set(allocations["ocrbench_v2_data_test_10000"].values()) == {20}
    assert set(allocations["mmmu_pro_standard10_test_1730"].values()) == {10}
    assert set(allocations["mathverse_testmini_3940"].values()) == {50}
