from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

try:  # pragma: no cover - fallback exists for minimal environments.
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

DATASET_ROOT = Path("/home/dredvpn009/Flash_Storage/datasets")
MANIFEST_VERSION = "tgvf_image_pool_v0"
REGISTRY_VERSION = "tgvf_dataset_registry_v0"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_SEED = 20260525
DEFAULT_TARGET_SAMPLES = 20_000
DEFAULT_SELECTED_IMAGES = 7_000
DEFAULT_SELECTION_RUN_ID = "selection_20k_v0"
DEFAULT_TEACHER_RUN_ID = "teacher_run_000001"
DEFAULT_SOURCE_MIX = {
    "visual_genome": 0.40,
    "textvqa_textocr": 0.30,
    "docvqa": 0.20,
    "chartqa": 0.10,
}
SOURCE_MIX_DATASETS = {
    "visual_genome": {"visual_genome"},
    "textvqa_textocr": {"textvqa", "textocr"},
    "docvqa": {"docvqa"},
    "chartqa": {"chartqa"},
}
SKIP_DIR_NAMES = {
    ".git",
    "__MACOSX",
    "__pycache__",
    "_archives",
    "_tools",
}


@dataclass
class BuildManifestResult:
    records: list[dict[str, Any]]
    added_records: list[dict[str, Any]]
    report: dict[str, Any]
    latest_manifest: Path
    versioned_manifest: Path


def default_preparation_root(project_root: str | Path) -> Path:
    return Path(project_root) / "data" / "tgvf_teacher" / "preparation"


def default_generated_root(project_root: str | Path) -> Path:
    return Path(project_root) / "data" / "tgvf_teacher" / "generated"


def ensure_preparation_layout(project_root: str | Path) -> dict[str, Path]:
    preparation_root = default_preparation_root(project_root)
    paths = {
        "preparation": preparation_root,
        "manifests": preparation_root / "manifests",
        "selections": preparation_root / "selections",
        "reports": preparation_root / "reports",
        "logs": preparation_root / "logs",
        "generated": default_generated_root(project_root),
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def default_registry_path(project_root: str | Path) -> Path:
    return default_preparation_root(project_root) / "dataset_registry.yaml"


def default_manifest_path(project_root: str | Path) -> Path:
    return default_preparation_root(project_root) / "image_pool_manifest.latest.jsonl"


def default_registry(dataset_root: str | Path = DATASET_ROOT) -> dict[str, Any]:
    root = Path(dataset_root)
    return {
        "registry_version": REGISTRY_VERSION,
        "dataset_root": str(root),
        "datasets": {
            "visual_genome": _entry(
                root,
                "visual_genome",
                "natural_image",
                "high",
                ["VG_100K", "VG_100K_2", "images"],
                ["region_descriptions.json", "objects.json", "attributes.json"],
                "May require manual download depending on mirror.",
                "prefer train images; avoid official evaluation splits if present",
                train=True,
            ),
            "coco": _entry(
                root,
                "coco",
                "natural_image",
                "high",
                ["train2017", "train2014", "images/train2017", "val2017", "val2014"],
                ["annotations/instances_train2017.json", "annotations/captions_train2017.json"],
                "COCO license applies; keep raw data under DATASET_ROOT.",
                "prefer train splits for teacher generation",
                train=True,
            ),
            "textvqa": _entry(
                root,
                "textvqa",
                "scene_text",
                "high",
                ["train_images", "images/train", "images"],
                ["TextVQA_0.5.1_train.json", "TextVQA_Rosetta_OCR_v0.2_train.json"],
                "May require manual download from official source.",
                "prefer train split",
                train=True,
            ),
            "textocr": _entry(
                root,
                "textocr",
                "scene_text",
                "high",
                ["train_images", "images/train", "images"],
                ["TextOCR_0.1_train.json"],
                "May require manual download from official source.",
                "prefer train split",
                train=True,
            ),
            "docvqa": _entry(
                root,
                "docvqa",
                "document",
                "high",
                ["train", "images/train", "documents/train", "images"],
                ["train_v1.0.json", "spdocvqa_train_v1.0.json"],
                "Requires official access agreement for some variants.",
                "prefer train split",
                train=True,
            ),
            "chartqa": _entry(
                root,
                "chartqa",
                "chart",
                "medium",
                ["train/png", "train/images", "images/train", "images"],
                ["train/train_augmented.json", "train/train_human.json"],
                "May require manual download from dataset host.",
                "prefer train split",
                train=True,
            ),
            "ocr_vqa": _entry(
                root,
                "ocr_vqa",
                "scene_text",
                "medium",
                ["images", "train"],
                ["dataset.json"],
                "Optional OCR-heavy source; verify license before use.",
                "prefer train split if present",
                train=True,
            ),
            "pascal_context": _entry(
                root,
                "pascal_context",
                "natural_image",
                "medium",
                ["PASCAL_MT", "JPEGImages", "images"],
                ["PASCAL_MT/pascal_context_train.json"],
                "Existing local source for objects, parts, and contextual scenes.",
                "prefer train/unknown split",
                train=True,
            ),
            "image_net": _entry(
                root,
                "image_net",
                "natural_image",
                "low",
                ["train", "val"],
                [],
                "Object-centric source; lower priority for local evidence.",
                "prefer train; avoid val if reserved for evaluation",
                train=True,
            ),
            "co3d": _entry(
                root,
                "co3d",
                "mixed",
                "low",
                ["data", "data_full", "co3d"],
                ["annotation"],
                "Optional multi-view/object-centric source; not needed for first run.",
                "unknown; enable selectively",
                train=False,
            ),
            "realestate10k": _entry(
                root,
                "realestate10k",
                "mixed",
                "low",
                ["train", "images", "."],
                [],
                "Optional indoor/spatial-layout source; inspect before heavy use.",
                "unknown",
                train=False,
            ),
            "olmoe-mix": _entry(
                root,
                "olmoe-mix",
                "unknown",
                "low",
                ["."],
                [],
                "Inspect before use; only image files are indexed if enabled.",
                "unknown",
                enabled=False,
                train=False,
            ),
            "benchmarks": _entry(
                root,
                "benchmarks",
                "mixed",
                "low",
                ["."],
                [],
                "Disabled by default to avoid contaminating evaluation benchmarks.",
                "do not use for train generation unless explicitly configured",
                enabled=False,
                train=False,
            ),
        },
    }


def write_default_registry(path: str | Path, dataset_root: str | Path = DATASET_ROOT) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_yaml(path, default_registry(dataset_root))
    return path


def load_registry(
    registry_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    create_if_missing: bool = False,
) -> dict[str, Any]:
    path = Path(registry_path)
    if not path.exists():
        if not create_if_missing:
            raise FileNotFoundError(path)
        write_default_registry(path, dataset_root or DATASET_ROOT)
    registry = _read_yaml(path)
    if dataset_root is not None:
        registry["dataset_root"] = str(dataset_root)
        for name, entry in registry.get("datasets", {}).items():
            if _is_under_dataset_root(entry.get("dataset_root"), registry.get("dataset_root")):
                entry["dataset_root"] = str(Path(dataset_root) / name)
    return registry


def verify_roots(
    *,
    dataset_root: str | Path,
    project_root: str | Path,
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    paths = ensure_preparation_layout(project_root)
    registry_file = Path(registry_path) if registry_path else default_registry_path(project_root)
    registry = load_registry(
        registry_file,
        dataset_root=dataset_root,
        create_if_missing=True,
    )
    datasets = {}
    for name, entry in registry["datasets"].items():
        root = Path(entry["dataset_root"])
        datasets[name] = {
            "dataset_root": str(root),
            "exists": root.exists(),
            "enabled": bool(entry.get("enabled", True)),
            "source_profile": entry.get("source_profile", "unknown"),
            "use_for_train_generation": bool(entry.get("use_for_train_generation", False)),
        }
    report = {
        "dataset_root": str(dataset_root),
        "dataset_root_exists": Path(dataset_root).exists(),
        "project_root": str(project_root),
        "preparation_root": str(paths["preparation"]),
        "registry_path": str(registry_file),
        "datasets": datasets,
        "created_at": _now(),
    }
    _write_json(paths["reports"] / "dataset_scan_report.json", report)
    _log(paths["logs"] / "prepare_datasets.log", "verify-roots completed")
    return report


def build_image_pool_manifest(
    *,
    dataset_root: str | Path,
    project_root: str | Path,
    registry_path: str | Path | None = None,
    output: str | Path | None = None,
    append_only: bool = True,
    max_images_per_dataset: int | None = None,
    inspect_images: bool = True,
    hash_images: bool = False,
    min_width: int = 128,
    min_height: int = 128,
) -> BuildManifestResult:
    paths = ensure_preparation_layout(project_root)
    registry_file = Path(registry_path) if registry_path else default_registry_path(project_root)
    registry = load_registry(
        registry_file,
        dataset_root=dataset_root,
        create_if_missing=True,
    )
    latest_manifest = Path(output) if output else default_manifest_path(project_root)
    existing_records = (
        read_jsonl(latest_manifest) if append_only and latest_manifest.exists() else []
    )
    existing_uids = {record["stable_image_uid"] for record in existing_records}
    existing_hashes = {
        record["sha1"]: record["stable_image_uid"]
        for record in existing_records
        if record.get("sha1")
    }

    added_records: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "registry_path": str(registry_file),
        "append_only": append_only,
        "max_images_per_dataset": max_images_per_dataset,
        "inspect_images": inspect_images,
        "hash_images": hash_images,
        "existing_records": len(existing_records),
        "datasets": {},
        "created_at": _now(),
    }

    for dataset_name, entry in registry["datasets"].items():
        dataset_report = _empty_dataset_report(entry)
        report["datasets"][dataset_name] = dataset_report
        if not entry.get("enabled", True):
            dataset_report["status"] = "disabled"
            continue
        root = Path(entry["dataset_root"])
        if not root.exists():
            dataset_report["status"] = "missing_manual_download_required"
            continue

        count_for_dataset = 0
        seen_new_uids: set[str] = set()
        for image_path in iter_image_files(entry):
            if max_images_per_dataset is not None and count_for_dataset >= max_images_per_dataset:
                dataset_report["stopped_at_max_images"] = True
                break
            record = build_image_record(
                dataset_name=dataset_name,
                entry=entry,
                image_path=image_path,
                inspect_images=inspect_images,
                hash_images=hash_images,
                min_width=min_width,
                min_height=min_height,
            )
            dataset_report["scanned_images"] += 1
            uid = record["stable_image_uid"]
            if uid in existing_uids or uid in seen_new_uids:
                dataset_report["skipped_duplicate_uid"] += 1
                continue
            content_hash = record.get("sha1")
            if content_hash and content_hash in existing_hashes:
                dataset_report["skipped_duplicate_sha1"] += 1
                dataset_report["duplicate_aliases"].append(
                    {
                        "stable_image_uid": uid,
                        "duplicate_of": existing_hashes[content_hash],
                    }
                )
                continue
            added_records.append(record)
            existing_uids.add(uid)
            seen_new_uids.add(uid)
            if content_hash:
                existing_hashes[content_hash] = uid
            dataset_report["added_images"] += 1
            dataset_report["status_counts"][record["status"]] += 1
            count_for_dataset += 1
        dataset_report["status"] = "scanned"

    records = [*existing_records, *added_records]
    versioned_manifest = write_versioned_manifest(
        records,
        preparation_root=paths["preparation"],
        latest_manifest=latest_manifest,
    )
    report["added_records"] = len(added_records)
    report["total_records"] = len(records)
    report["latest_manifest"] = str(latest_manifest)
    report["versioned_manifest"] = str(versioned_manifest)
    _write_json(paths["reports"] / "dataset_scan_report.json", _jsonable(report))
    _log(
        paths["logs"] / "prepare_datasets.log",
        f"build-manifest added={len(added_records)} total={len(records)}",
    )
    return BuildManifestResult(
        records=records,
        added_records=added_records,
        report=_jsonable(report),
        latest_manifest=latest_manifest,
        versioned_manifest=versioned_manifest,
    )


def build_image_record(
    *,
    dataset_name: str,
    entry: dict[str, Any],
    image_path: str | Path,
    inspect_images: bool = True,
    hash_images: bool = False,
    min_width: int = 128,
    min_height: int = 128,
) -> dict[str, Any]:
    image_path = Path(image_path)
    dataset_root = Path(entry["dataset_root"])
    relative_path = _safe_relative(image_path, dataset_root)
    source_image_id = infer_source_image_id(dataset_name, relative_path)
    status = "available"
    width = None
    height = None
    validation_error = None
    if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        status = "invalid_unsupported_extension"
    elif inspect_images:
        image_info = inspect_image(image_path, min_width=min_width, min_height=min_height)
        status = image_info["status"]
        width = image_info["width"]
        height = image_info["height"]
        validation_error = image_info.get("error")
    file_size = image_path.stat().st_size if image_path.exists() else None
    content_hash = sha1_file(image_path) if hash_images and image_path.exists() else None
    annotation_path = first_existing_annotation_path(entry)
    return {
        "stable_image_uid": f"{dataset_name}:{source_image_id}",
        "source_dataset": dataset_name,
        "source_profile": entry.get("source_profile", "unknown"),
        "image_path": str(image_path),
        "annotation_path": str(annotation_path) if annotation_path else None,
        "split": infer_split(relative_path),
        "width": width,
        "height": height,
        "file_size": file_size,
        "sha1": content_hash,
        "source_image_id": source_image_id,
        "metadata": {
            "relative_image_path": relative_path,
            "priority": entry.get("priority", "medium"),
            "has_region_annotations": _has_region_annotations(dataset_name, entry),
            "has_ocr_annotations": _has_ocr_annotations(dataset_name, entry),
            "has_qa_annotations": _has_qa_annotations(dataset_name),
            "use_for_train_generation": bool(entry.get("use_for_train_generation", True)),
            "use_for_validation_generation": bool(
                entry.get("use_for_validation_generation", False)
            ),
            "validation_error": validation_error,
            "aliases": [],
        },
        "status": status,
        "created_at": _now(),
        "manifest_version": MANIFEST_VERSION,
    }


def iter_image_files(entry: dict[str, Any]) -> Iterator[Path]:
    scan_roots = _scan_roots(entry)
    seen_paths: set[str] = set()
    for root in sorted(scan_roots, key=lambda path: path.as_posix()):
        if root.is_file():
            if root.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                resolved = root.resolve().as_posix()
                if resolved not in seen_paths:
                    seen_paths.add(resolved)
                    yield root
            continue
        for dirpath_str, dirnames, filenames in os.walk(root):
            dirpath = Path(dirpath_str)
            dirnames[:] = [
                dirname for dirname in dirnames if dirname not in SKIP_DIR_NAMES
            ]
            dirnames.sort()
            for filename in sorted(filenames):
                path = dirpath / filename
                if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                    continue
                resolved = path.resolve().as_posix()
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                yield path


def validate_images(
    *,
    manifest: str | Path,
    report: str | Path,
    min_width: int = 128,
    min_height: int = 128,
    hash_images: bool = False,
) -> dict[str, Any]:
    records = read_jsonl(manifest)
    status_counts: Counter[str] = Counter()
    failures = []
    sha1_counts: Counter[str] = Counter()
    for record in records:
        path = Path(record["image_path"])
        if not path.exists():
            status = "missing_file"
            info = {"width": None, "height": None, "error": "missing_file"}
        elif path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            status = "invalid_unsupported_extension"
            info = {"width": None, "height": None, "error": "unsupported_extension"}
        else:
            info = inspect_image(path, min_width=min_width, min_height=min_height)
            status = info["status"]
        status_counts[status] += 1
        if hash_images and path.exists():
            sha1_counts[sha1_file(path)] += 1
        if status != "available":
            failures.append(
                {
                    "stable_image_uid": record["stable_image_uid"],
                    "image_path": record["image_path"],
                    "status": status,
                    "error": info.get("error"),
                }
            )
    duplicate_sha1 = sum(count - 1 for count in sha1_counts.values() if count > 1)
    validation_report = {
        "manifest": str(manifest),
        "total_records": len(records),
        "status_counts": dict(status_counts),
        "failure_count": len(failures),
        "failures": failures,
        "hash_images": hash_images,
        "duplicate_sha1_count": duplicate_sha1,
        "created_at": _now(),
    }
    _write_json(report, validation_report)
    return validation_report


def select_image_pool(
    *,
    manifest: str | Path,
    output: str | Path,
    target_samples: int = DEFAULT_TARGET_SAMPLES,
    selected_images: int = DEFAULT_SELECTED_IMAGES,
    seed: int = DEFAULT_SEED,
    selection_run_id: str = DEFAULT_SELECTION_RUN_ID,
    planned_teacher_run_id: str = DEFAULT_TEACHER_RUN_ID,
    teacher_ledger: str | Path | None = None,
    allow_used_images: bool = False,
    train_splits_only: bool = True,
) -> dict[str, Any]:
    records = read_jsonl(manifest)
    used_uids = set()
    if teacher_ledger and Path(teacher_ledger).exists() and not allow_used_images:
        used_uids = read_used_image_uids(teacher_ledger)

    eligible = [
        record
        for record in records
        if _eligible_for_selection(record, used_uids, train_splits_only=train_splits_only)
    ]
    selected, selection_debug = _stratified_select(eligible, selected_images, seed)
    budget_hint = max(1, round(target_samples / max(len(selected), 1)))
    selected_records = [
        {
            "stable_image_uid": record["stable_image_uid"],
            "source_dataset": record["source_dataset"],
            "source_profile": record["source_profile"],
            "image_path": record["image_path"],
            "source_image_id": record.get("source_image_id"),
            "selection_run_id": selection_run_id,
            "planned_teacher_run_id": planned_teacher_run_id,
            "target_sample_budget_hint": budget_hint,
            "status": "selected",
            "manifest_version": MANIFEST_VERSION,
            "metadata": {
                "split": record.get("split", "unknown"),
                "source_manifest": str(manifest),
            },
        }
        for record in selected
    ]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(selected_records, output)
    report = {
        "manifest": str(manifest),
        "output": str(output),
        "target_teacher_samples": target_samples,
        "requested_selected_images": selected_images,
        "selected_images": len(selected_records),
        "seed": seed,
        "selection_run_id": selection_run_id,
        "planned_teacher_run_id": planned_teacher_run_id,
        "profile_counts": dict(Counter(record["source_profile"] for record in selected)),
        "dataset_counts": dict(Counter(record["source_dataset"] for record in selected)),
        "source_mix_counts": dict(
            Counter(_source_mix_bucket(record["source_dataset"]) for record in selected)
        ),
        "source_mix_target_fractions": DEFAULT_SOURCE_MIX,
        "source_mix_quota_shortfalls": selection_debug["quota_shortfalls"],
        "selection_strategy": "source_dataset_mix_v0",
        "excluded_used_images": len(used_uids),
        "created_at": _now(),
    }
    _write_json(output.parent.parent / "reports" / "selection_report.json", report)
    return report


def summarize_manifest(
    *,
    manifest: str | Path,
    selection: str | Path | None = None,
) -> dict[str, Any]:
    records = read_jsonl(manifest)
    summary = {
        "manifest": str(manifest),
        "total_records": len(records),
        "status_counts": dict(Counter(record.get("status", "unknown") for record in records)),
        "profile_counts": dict(
            Counter(record.get("source_profile", "unknown") for record in records)
        ),
        "dataset_counts": dict(
            Counter(record.get("source_dataset", "unknown") for record in records)
        ),
        "split_counts": dict(Counter(record.get("split", "unknown") for record in records)),
    }
    if selection:
        selected = read_jsonl(selection)
        summary["selection"] = {
            "path": str(selection),
            "total_records": len(selected),
            "profile_counts": dict(
                Counter(record.get("source_profile", "unknown") for record in selected)
            ),
            "dataset_counts": dict(
                Counter(record.get("source_dataset", "unknown") for record in selected)
            ),
        }
    return summary


def write_versioned_manifest(
    records: list[dict[str, Any]],
    *,
    preparation_root: str | Path,
    latest_manifest: str | Path,
) -> Path:
    preparation_root = Path(preparation_root)
    manifests_dir = preparation_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    version = _next_manifest_version(manifests_dir)
    versioned = manifests_dir / f"image_pool_manifest.v{version:03d}.jsonl"
    write_jsonl(records, versioned)
    latest_manifest = Path(latest_manifest)
    latest_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(versioned, latest_manifest)
    return versioned


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} invalid JSONL record") from exc
    return records


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")


def inspect_image(path: str | Path, *, min_width: int, min_height: int) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"status": "missing_file", "width": None, "height": None, "error": "missing_file"}
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
    except Exception as exc:  # noqa: BLE001 - this is a validation boundary.
        return {
            "status": "invalid_unreadable",
            "width": None,
            "height": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if width < min_width or height < min_height:
        return {
            "status": "invalid_too_small",
            "width": width,
            "height": height,
            "error": f"below_min_size_{min_width}x{min_height}",
        }
    return {"status": "available", "width": width, "height": height, "error": None}


def infer_source_image_id(dataset_name: str, relative_path: str) -> str:
    path = Path(relative_path)
    stem = path.stem
    if dataset_name in {
        "visual_genome",
        "coco",
        "textvqa",
        "textocr",
        "docvqa",
        "chartqa",
        "ocr_vqa",
    } and stem:
        return stem
    return f"path_{sha1_text(relative_path)[:16]}"


def infer_split(relative_path: str) -> str:
    parts = {part.lower() for part in Path(relative_path).parts}
    if parts & {"train", "training", "train2014", "train2017"}:
        return "train"
    if parts & {"val", "valid", "validation", "val2014", "val2017"}:
        return "validation"
    if parts & {"test", "testing"}:
        return "test"
    return "unknown"


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def sha1_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_used_image_uids(ledger: str | Path) -> set[str]:
    return {
        record["stable_image_uid"]
        for record in read_jsonl(ledger)
        if record.get("stable_image_uid")
    }


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "verify-roots":
        report = verify_roots(
            dataset_root=args.dataset_root,
            project_root=args.project_root,
            registry_path=args.registry,
        )
        print(json.dumps(_jsonable(report), indent=2))
    elif args.command == "build-manifest":
        result = build_image_pool_manifest(
            dataset_root=args.dataset_root,
            project_root=args.project_root,
            registry_path=args.registry,
            output=args.output,
            append_only=args.append_only,
            max_images_per_dataset=args.max_images_per_dataset,
            inspect_images=not args.no_inspect_images,
            hash_images=args.hash_images,
            min_width=args.min_width,
            min_height=args.min_height,
        )
        print(json.dumps(_jsonable(result.report), indent=2))
    elif args.command == "validate-images":
        report = validate_images(
            manifest=args.manifest,
            report=args.report,
            min_width=args.min_width,
            min_height=args.min_height,
            hash_images=args.hash_images,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "select":
        report = select_image_pool(
            manifest=args.manifest,
            output=args.output,
            target_samples=args.target_samples,
            selected_images=args.selected_images,
            seed=args.seed,
            selection_run_id=args.selection_run_id,
            planned_teacher_run_id=args.planned_teacher_run_id,
            teacher_ledger=args.teacher_ledger,
            allow_used_images=args.allow_used_images,
            train_splits_only=not args.include_validation_or_test_splits,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "summarize":
        print(
            json.dumps(
                summarize_manifest(manifest=args.manifest, selection=args.selection),
                indent=2,
            )
        )
    else:  # pragma: no cover
        parser.error(f"Unknown command: {args.command}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare TGVF image-source manifests.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify-roots")
    _add_root_args(verify)

    build = subparsers.add_parser("build-manifest")
    _add_root_args(build)
    build.add_argument("--output", default=None)
    build.add_argument("--append-only", action=argparse.BooleanOptionalAction, default=True)
    build.add_argument("--max-images-per-dataset", type=int, default=None)
    build.add_argument("--no-inspect-images", action="store_true")
    build.add_argument("--hash-images", action="store_true")
    build.add_argument("--min-width", type=int, default=128)
    build.add_argument("--min-height", type=int, default=128)

    validate = subparsers.add_parser("validate-images")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--report", required=True)
    validate.add_argument("--min-width", type=int, default=128)
    validate.add_argument("--min-height", type=int, default=128)
    validate.add_argument("--hash-images", action="store_true")

    select = subparsers.add_parser("select")
    select.add_argument("--manifest", required=True)
    select.add_argument("--target-samples", type=int, default=DEFAULT_TARGET_SAMPLES)
    select.add_argument("--selected-images", type=int, default=DEFAULT_SELECTED_IMAGES)
    select.add_argument("--output", required=True)
    select.add_argument("--seed", type=int, default=DEFAULT_SEED)
    select.add_argument("--selection-run-id", default=DEFAULT_SELECTION_RUN_ID)
    select.add_argument("--planned-teacher-run-id", default=DEFAULT_TEACHER_RUN_ID)
    select.add_argument("--teacher-ledger", default=None)
    select.add_argument("--allow-used-images", action="store_true")
    select.add_argument("--include-validation-or-test-splits", action="store_true")

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--manifest", required=True)
    summarize.add_argument("--selection", default=None)
    return parser


def _add_root_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT))
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--registry", default=None)


def _entry(
    dataset_root: Path,
    name: str,
    source_profile: str,
    priority: str,
    expected_image_dirs: list[str],
    expected_annotation_files: list[str],
    license_or_access_note: str,
    split_policy: str,
    *,
    enabled: bool = True,
    train: bool,
) -> dict[str, Any]:
    return {
        "dataset_name": name,
        "dataset_root": str(dataset_root / name),
        "enabled": enabled,
        "source_profile": source_profile,
        "priority": priority,
        "expected_image_dirs": expected_image_dirs,
        "expected_annotation_files": expected_annotation_files,
        "license_or_access_note": license_or_access_note,
        "split_policy": split_policy,
        "use_for_train_generation": train,
        "use_for_validation_generation": False,
        "notes": "",
    }


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    if yaml is None:
        path.write_text(json.dumps(data, indent=2))
        return
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _read_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if yaml is not None:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def _write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True))


def _scan_roots(entry: dict[str, Any]) -> list[Path]:
    dataset_root = Path(entry["dataset_root"])
    roots = []
    for rel in entry.get("expected_image_dirs", []):
        candidate = dataset_root / rel
        if candidate.exists():
            roots.append(candidate)
    return roots or [dataset_root]


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _empty_dataset_report(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_root": entry.get("dataset_root"),
        "enabled": bool(entry.get("enabled", True)),
        "source_profile": entry.get("source_profile", "unknown"),
        "status": "pending",
        "scanned_images": 0,
        "added_images": 0,
        "skipped_duplicate_uid": 0,
        "skipped_duplicate_sha1": 0,
        "duplicate_aliases": [],
        "status_counts": Counter(),
    }


def _next_manifest_version(manifests_dir: Path) -> int:
    versions = []
    for path in manifests_dir.glob("image_pool_manifest.v*.jsonl"):
        try:
            versions.append(int(path.stem.rsplit(".v", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(versions, default=-1) + 1


def _is_under_dataset_root(path: str | None, dataset_root: str | None) -> bool:
    if not path or not dataset_root:
        return True
    try:
        Path(path).relative_to(Path(dataset_root))
        return True
    except ValueError:
        return False


def first_existing_annotation_path(entry: dict[str, Any]) -> Path | None:
    root = Path(entry["dataset_root"])
    for rel in entry.get("expected_annotation_files", []):
        path = root / rel
        if path.exists():
            return path
    return None


def _has_region_annotations(dataset_name: str, entry: dict[str, Any]) -> bool:
    return dataset_name in {"visual_genome", "coco", "pascal_context"} or bool(
        entry.get("expected_annotation_files")
        and entry.get("source_profile") == "natural_image"
    )


def _has_ocr_annotations(dataset_name: str, entry: dict[str, Any]) -> bool:
    return dataset_name in {"textvqa", "textocr", "ocr_vqa"} or entry.get(
        "source_profile"
    ) in {"scene_text", "document"}


def _has_qa_annotations(dataset_name: str) -> bool:
    return dataset_name in {"textvqa", "docvqa", "chartqa", "ocr_vqa"}


def _eligible_for_selection(
    record: dict[str, Any],
    used_uids: set[str],
    *,
    train_splits_only: bool,
) -> bool:
    if record.get("status") != "available":
        return False
    if record["stable_image_uid"] in used_uids:
        return False
    metadata = record.get("metadata") or {}
    if metadata.get("use_for_train_generation") is False:
        return False
    if train_splits_only and record.get("split") in {"validation", "test"}:
        return False
    return True


def _stratified_select(
    records: list[dict[str, Any]], selected_images: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        bucket = _source_mix_bucket(record.get("source_dataset", "unknown"))
        if bucket is not None:
            buckets[bucket].append(record)
    for bucket_records in buckets.values():
        bucket_records.sort(key=lambda item: item["stable_image_uid"])
        rng.shuffle(bucket_records)

    quotas = _selection_quotas(selected_images)
    chosen_by_bucket: dict[str, list[dict[str, Any]]] = {}
    selected_uids: set[str] = set()
    quota_shortfalls: dict[str, int] = {}
    for bucket, quota in quotas.items():
        chosen = buckets.get(bucket, [])[:quota]
        chosen_by_bucket[bucket] = chosen
        selected_uids.update(record["stable_image_uid"] for record in chosen)
        if len(chosen) < quota:
            quota_shortfalls[bucket] = quota - len(chosen)

    selected = _interleave_source_mix(chosen_by_bucket, selected_images)

    if len(selected) < selected_images:
        leftovers = [
            record
            for bucket in ("visual_genome", "textvqa_textocr", "docvqa", "chartqa")
            for record in buckets.get(bucket, [])
            if record["stable_image_uid"] not in selected_uids
        ]
        selected.extend(leftovers[: selected_images - len(selected)])

    return selected[:selected_images], {"quota_shortfalls": quota_shortfalls}


def _interleave_source_mix(
    records_by_bucket: dict[str, list[dict[str, Any]]],
    total_count: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    indices = {bucket: 0 for bucket in DEFAULT_SOURCE_MIX}
    targets = _selection_quotas(total_count)
    while len(selected) < total_count:
        best_bucket = None
        best_deficit = None
        next_position = len(selected) + 1
        for bucket, fraction in DEFAULT_SOURCE_MIX.items():
            bucket_records = records_by_bucket.get(bucket, [])
            if indices[bucket] >= len(bucket_records):
                continue
            if indices[bucket] >= targets.get(bucket, 0):
                continue
            expected = next_position * fraction
            deficit = expected - indices[bucket]
            if best_deficit is None or deficit > best_deficit:
                best_bucket = bucket
                best_deficit = deficit
        if best_bucket is None:
            break
        selected.append(records_by_bucket[best_bucket][indices[best_bucket]])
        indices[best_bucket] += 1
    return selected


def _source_mix_bucket(source_dataset: str) -> str | None:
    for bucket, dataset_names in SOURCE_MIX_DATASETS.items():
        if source_dataset in dataset_names:
            return bucket
    return None


def _selection_quotas(selected_images: int) -> dict[str, int]:
    raw = {
        bucket: selected_images * fraction
        for bucket, fraction in DEFAULT_SOURCE_MIX.items()
    }
    quotas = {bucket: int(value) for bucket, value in raw.items()}
    remainder = selected_images - sum(quotas.values())
    for bucket, _value in sorted(
        raw.items(), key=lambda item: item[1] - int(item[1]), reverse=True
    )[:remainder]:
        quotas[bucket] += 1
    return quotas


def _jsonable(value: Any) -> Any:
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=path, level=logging.INFO)
    logging.info("%s %s", _now(), message)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
