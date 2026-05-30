from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tgvf_data.prepare import (
    build_image_pool_manifest,
    default_manifest_path,
    default_registry,
    default_registry_path,
    select_image_pool,
    summarize_manifest,
    validate_images,
    verify_roots,
)


def test_verify_roots_writes_registry_and_preparation_layout(tmp_path: Path) -> None:
    dataset_root = tmp_path / "datasets"
    project_root = tmp_path / "project"
    (dataset_root / "pascal_context").mkdir(parents=True)

    report = verify_roots(dataset_root=dataset_root, project_root=project_root)

    assert report["dataset_root_exists"] is True
    assert default_registry_path(project_root).exists()
    assert (project_root / "data" / "tgvf_teacher" / "preparation" / "reports").exists()
    assert report["datasets"]["pascal_context"]["exists"] is True
    assert report["datasets"]["benchmarks"]["enabled"] is False


def test_default_registry_keeps_benchmarks_disabled(tmp_path: Path) -> None:
    registry = default_registry(tmp_path / "datasets")

    assert registry["datasets"]["benchmarks"]["enabled"] is False
    assert registry["datasets"]["benchmarks"]["use_for_train_generation"] is False
    assert registry["datasets"]["visual_genome"]["source_profile"] == "natural_image"
    assert registry["datasets"]["textvqa"]["source_profile"] == "scene_text"


def test_build_manifest_is_append_only_and_records_validation_status(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "datasets"
    project_root = tmp_path / "project"
    registry_path = _write_test_registry(dataset_root, project_root)
    _image(dataset_root / "pascal_context" / "PASCAL_MT" / "natural.jpg")
    _image(dataset_root / "textvqa" / "train_images" / "text.jpg")
    _image(dataset_root / "textvqa" / "train_images" / "tiny.jpg", size=(32, 32))

    first = build_image_pool_manifest(
        dataset_root=dataset_root,
        project_root=project_root,
        registry_path=registry_path,
        inspect_images=True,
    )

    assert first.latest_manifest == default_manifest_path(project_root)
    assert first.versioned_manifest.name == "image_pool_manifest.v000.jsonl"
    assert len(first.records) == 3
    assert {record["status"] for record in first.records} == {
        "available",
        "invalid_too_small",
    }
    old_records = [dict(record) for record in first.records]

    _image(dataset_root / "pascal_context" / "PASCAL_MT" / "new_natural.jpg")
    second = build_image_pool_manifest(
        dataset_root=dataset_root,
        project_root=project_root,
        registry_path=registry_path,
        inspect_images=True,
    )

    assert second.versioned_manifest.name == "image_pool_manifest.v001.jsonl"
    assert len(second.added_records) == 1
    assert second.records[: len(old_records)] == old_records
    assert len(second.records) == 4


def test_validate_and_select_image_pool_are_deterministic(tmp_path: Path) -> None:
    dataset_root = tmp_path / "datasets"
    project_root = tmp_path / "project"
    registry_path = _write_test_registry(dataset_root, project_root)
    for idx in range(4):
        _image(dataset_root / "visual_genome" / "VG_100K" / f"vg_{idx}.jpg")
    for idx in range(2):
        _image(dataset_root / "textvqa" / "train_images" / f"textvqa_{idx}.jpg")
    _image(dataset_root / "textocr" / "train_images" / "textocr_0.jpg")
    for idx in range(2):
        _image(dataset_root / "docvqa" / "train" / f"doc_{idx}.png")
    _image(dataset_root / "chartqa" / "train" / "png" / "chart_0.png")

    result = build_image_pool_manifest(
        dataset_root=dataset_root,
        project_root=project_root,
        registry_path=registry_path,
        inspect_images=True,
    )
    validation_report = validate_images(
        manifest=result.latest_manifest,
        report=project_root
        / "data"
        / "tgvf_teacher"
        / "preparation"
        / "reports"
        / "image_validation_report.json",
    )
    selection_path = (
        project_root
        / "data"
        / "tgvf_teacher"
        / "preparation"
        / "selections"
        / "selected_images_20k_samples_v0.jsonl"
    )
    report_a = select_image_pool(
        manifest=result.latest_manifest,
        output=selection_path,
        target_samples=20_000,
        selected_images=10,
        seed=20260525,
    )
    selected_a = selection_path.read_text()
    report_b = select_image_pool(
        manifest=result.latest_manifest,
        output=selection_path,
        target_samples=20_000,
        selected_images=10,
        seed=20260525,
    )
    selected_b = selection_path.read_text()
    summary = summarize_manifest(manifest=result.latest_manifest, selection=selection_path)

    assert validation_report["status_counts"] == {"available": 10}
    assert report_a["selected_images"] == 10
    assert report_a["source_mix_counts"] == {
        "visual_genome": 4,
        "textvqa_textocr": 3,
        "docvqa": 2,
        "chartqa": 1,
    }
    assert report_a["selection_strategy"] == "source_dataset_mix_v0"
    assert report_b["profile_counts"] == report_a["profile_counts"]
    assert selected_a == selected_b
    assert summary["total_records"] == 10
    assert summary["selection"]["total_records"] == 10
    for line in selected_a.splitlines():
        record = json.loads(line)
        assert record["selection_run_id"] == "selection_20k_v0"
        assert record["planned_teacher_run_id"] == "teacher_run_000001"
        assert record["status"] == "selected"


def _write_test_registry(dataset_root: Path, project_root: Path) -> Path:
    registry = default_registry(dataset_root)
    keep = {
        "pascal_context",
        "visual_genome",
        "textvqa",
        "textocr",
        "docvqa",
        "chartqa",
        "benchmarks",
    }
    registry["datasets"] = {
        name: entry for name, entry in registry["datasets"].items() if name in keep
    }
    registry_path = default_registry_path(project_root)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry))
    return registry_path


def _image(path: Path, size: tuple[int, int] = (160, 160)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(40, 90, 140)).save(path)
