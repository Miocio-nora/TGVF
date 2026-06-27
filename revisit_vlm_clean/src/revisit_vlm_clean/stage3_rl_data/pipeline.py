"""Stage3 RL source-pool planning and execution."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from revisit_vlm_clean.schema import _to_jsonable

from .schemas import (
    BALANCE_VERSION,
    FILTER_VERSION,
    GENERATOR_VERSION,
    IMAGE_BUNDLE_SCHEMA_VERSION,
    PLAN_SCHEMA_VERSION,
    QA_PROMPT_SCHEMA_VERSION,
    SUPPORTED_ANSWER_TYPES,
    SUPPORTED_EVIDENCE_TYPES,
    SUPPORTED_EVAL_METRICS,
    build_rule_target_spec,
    contains_sensitive_identifier,
    now_iso,
    normalize_text,
    qa_prompt_from_bundle_item,
    stable_hash,
    validate_target_spec,
)
from .sources import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_SOURCE_MIX,
    SourceConfig,
    adapter_for_config,
    load_source_configs,
)
from .teacher import (
    DEFAULT_STAGE3_RL_TEACHER_MODEL,
    STAGE3_RL_TEACHER_VERSION,
    build_stage3_rl_teacher_requests,
    summarize_teacher_requests,
)

DEFAULT_OUTPUT_ROOT = "revisit_vlm_clean/data/stage3_rl/v0_20k"
DEFAULT_TARGET_ACCEPTED_PROMPTS = 20_000
DEFAULT_MAX_QA_PER_IMAGE = 4
DEFAULT_SEED = 42
DEFAULT_SPLIT_ALLOWLIST = ("train", "training", "unknown")
DEFAULT_SPLIT_BLOCKLIST = ("val", "validation", "dev", "test")
SUPPORTED_QA_GENERATION_MODES = ("source_qa", "teacher_triage")
TEACHER_REQUEST_MODES = {"teacher_triage"}
DEFAULT_SFT_EXCLUDE_MANIFESTS = (
    "data/tgvf_teacher/preparation/selections/selected_images_v3_50k_v0.jsonl",
    "data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k/final/tgvf_teacher_items.accepted.jsonl",
    "data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.train.jsonl",
    "data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl",
)


@dataclass(frozen=True)
class Stage3RLBuildOptions:
    project_root: str
    dataset_root: str = DEFAULT_DATASET_ROOT
    output_root: str = DEFAULT_OUTPUT_ROOT
    target_accepted_prompts: int = DEFAULT_TARGET_ACCEPTED_PROMPTS
    seed: int = DEFAULT_SEED
    source_config_path: str | None = None
    exclude_manifests: tuple[str, ...] = ()
    source_split_allowlist: tuple[str, ...] = DEFAULT_SPLIT_ALLOWLIST
    source_split_blocklist: tuple[str, ...] = DEFAULT_SPLIT_BLOCKLIST
    max_new_images: int | None = None
    max_new_prompts: int | None = None
    max_qa_per_image: int = DEFAULT_MAX_QA_PER_IMAGE
    extend_from: str | None = None
    qa_generation_mode: str = "source_qa"
    teacher_backend: str | None = None
    hash_images: bool = True
    require_target_spec: bool = True
    run_id: str | None = None


@dataclass
class ExclusionSet:
    stable_image_uids: set[str] = field(default_factory=set)
    image_ids: set[str] = field(default_factory=set)
    image_paths: set[str] = field(default_factory=set)
    source_ids: set[str] = field(default_factory=set)
    image_sha256: set[str] = field(default_factory=set)
    rows: int = 0

    def counts(self) -> dict[str, int]:
        return {
            "stable_image_uids": len(self.stable_image_uids),
            "image_ids": len(self.image_ids),
            "image_paths": len(self.image_paths),
            "source_ids": len(self.source_ids),
            "image_sha256": len(self.image_sha256),
            "rows": self.rows,
        }


def build_stage3_rl_plan(options: Stage3RLBuildOptions) -> dict[str, Any]:
    if options.qa_generation_mode not in SUPPORTED_QA_GENERATION_MODES:
        raise ValueError(
            "unsupported Stage3 RL qa_generation_mode: "
            f"{options.qa_generation_mode}; expected one of {SUPPORTED_QA_GENERATION_MODES}"
        )
    project_root = Path(options.project_root).resolve()
    dataset_root = Path(options.dataset_root).expanduser()
    output_root = Path(options.output_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root

    source_configs = [
        item.to_dict()
        for item in load_source_configs(options.source_config_path, dataset_root=dataset_root)
        if item.enabled
    ]
    excluded = _default_exclude_manifests(project_root)
    excluded.extend(_resolve_manifest_paths(project_root, options.exclude_manifests))
    max_new_images = options.max_new_images
    if max_new_images is None:
        max_new_images = max(1000, math.ceil(options.target_accepted_prompts / options.max_qa_per_image * 1.6))
    run_id = options.run_id or f"stage3_rl_data_{now_iso().replace(':', '').replace('-', '')[:15]}"
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "run_id": run_id,
        "data_plan_id": stable_hash("stage3_rl_plan", run_id, str(output_root), options.seed),
        "created_at": now_iso(),
        "project_root": str(project_root),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "target_accepted_prompts": int(options.target_accepted_prompts),
        "random_seed": int(options.seed),
        "source_configs": source_configs,
        "source_config_path": options.source_config_path,
        "excluded_manifests": excluded,
        "source_split_allowlist": list(options.source_split_allowlist),
        "source_split_blocklist": list(options.source_split_blocklist),
        "source_dataset_paths": {
            item["name"]: item["path"] for item in source_configs if item.get("enabled", True)
        },
        "hash_method": "sha256_file" if options.hash_images else "disabled",
        "dedup_method": "stable_image_uid|image_path|image_sha256|sample_id",
        "filter_config": {
            "filter_version": FILTER_VERSION,
            "require_target_spec": bool(options.require_target_spec),
            "deterministic_only": True,
            "reject_answer_leakage_in_target": True,
            "reject_generic_target": True,
            "reject_sensitive_personal_info": True,
        },
        "balance_config": {
            "balance_version": BALANCE_VERSION,
            "source_weights": dict(DEFAULT_SOURCE_MIX),
            "max_qa_per_image": int(options.max_qa_per_image),
            "source_dominance_warning_threshold": 0.55,
            "answer_type_dominance_warning_threshold": 0.60,
        },
        "prompt_generation_mode": options.qa_generation_mode,
        "teacher_backend": options.teacher_backend,
        "teacher_config": {
            "teacher_version": STAGE3_RL_TEACHER_VERSION,
            "model": options.teacher_backend or DEFAULT_STAGE3_RL_TEACHER_MODEL,
            "image_detail": "original",
            "temperature": 0.2,
            "max_output_tokens": 3000,
            "source_qa_role": "candidate_anchor",
            "allowed_provenance": [
                "source_qa_kept",
                "source_qa_rewritten",
                "teacher_generated_legacy_style",
            ],
            "difficulty_labels": [
                "direct_easy",
                "local_easy",
                "local_medium",
                "local_hard",
                "reasoning_hard",
            ],
            "tool_need_hints": [
                "no_tool",
                "optional_tool",
                "useful_tool",
                "likely_required",
            ],
        },
        "max_new_images": int(max_new_images) if max_new_images is not None else None,
        "max_new_prompts": (
            int(options.max_new_prompts) if options.max_new_prompts is not None else None
        ),
        "extend_from": options.extend_from,
        "output_files": _output_files(str(output_root)),
        "generator_version": GENERATOR_VERSION,
    }
    plan["config_hash"] = _config_hash(plan)
    return plan


def write_stage3_rl_plan(output_root: str | Path, *, plan: dict[str, Any]) -> dict[str, str]:
    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "stage3_rl_data_plan.json"
    _write_json(plan_path, plan)
    _append_ledger(out / "generation_ledger.jsonl", "plan_created", plan, {"plan_path": str(plan_path)})
    return {"output_root": str(out), "stage3_rl_data_plan": str(plan_path)}


def load_stage3_rl_plan(path: str | Path) -> dict[str, Any]:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError(f"unsupported Stage3 RL plan schema: {plan.get('schema_version')}")
    return plan


def preflight_stage3_rl_plan(
    plan_or_path: dict[str, Any] | str | Path,
    *,
    write_report: bool = True,
) -> dict[str, Any]:
    plan = load_stage3_rl_plan(plan_or_path) if not isinstance(plan_or_path, dict) else plan_or_path
    output_root = Path(plan["output_root"])
    errors: list[str] = []
    warnings: list[str] = []
    source_reports: list[dict[str, Any]] = []

    dataset_root = Path(plan["dataset_root"])
    if not dataset_root.exists():
        errors.append(f"dataset_root_missing:{dataset_root}")
    output_writable = _check_output_writable(output_root)
    if not output_writable:
        errors.append(f"output_root_not_writable:{output_root}")

    total_estimated_qa = 0
    for raw_config in plan.get("source_configs") or []:
        config = SourceConfig.from_dict(raw_config)
        path = adapter_for_config(config, dataset_root=dataset_root).annotation_path()
        report = {
            "name": config.name,
            "adapter": config.adapter,
            "path": str(path),
            "exists": path.exists(),
            "estimated_qa": 0,
            "status": "ok",
        }
        if not path.exists():
            report["status"] = "missing"
            errors.append(f"source_path_missing:{config.name}:{path}")
        elif _looks_like_benchmark_path(path):
            report["status"] = "blocked_benchmark_path"
            errors.append(f"source_path_looks_like_benchmark:{config.name}:{path}")
        else:
            try:
                report["estimated_qa"] = adapter_for_config(
                    config, dataset_root=dataset_root
                ).estimate_candidate_qa()
                total_estimated_qa += int(report["estimated_qa"])
            except Exception as exc:  # pragma: no cover - exact source errors vary.
                report["status"] = "source_error"
                report["error"] = str(exc)
                errors.append(f"source_error:{config.name}:{exc}")
        source_reports.append(report)

    for manifest in plan.get("excluded_manifests") or []:
        if not Path(manifest).exists():
            errors.append(f"exclude_manifest_missing:{manifest}")

    if total_estimated_qa < int(plan["target_accepted_prompts"]):
        errors.append(
            f"estimated_candidate_qa_below_target:{total_estimated_qa}<"
            f"{plan['target_accepted_prompts']}"
        )

    report = {
        "schema_version": "stage3_rl_preflight_report_v0",
        "run_id": plan["run_id"],
        "plan_valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "output_root_writable": output_writable,
        "source_reports": source_reports,
        "estimated_candidate_qa": total_estimated_qa,
        "target_accepted_prompts": plan["target_accepted_prompts"],
        "schema_import_ok": True,
        "hash_method": plan.get("hash_method"),
        "created_at": now_iso(),
    }
    if write_report:
        output_root.mkdir(parents=True, exist_ok=True)
        _write_json(output_root / "preflight_report.json", report)
        event = "preflight_passed" if report["plan_valid"] else "failed_with_error"
        _append_ledger(output_root / "generation_ledger.jsonl", event, plan, report)
    return report


def execute_stage3_rl_plan(
    plan_or_path: dict[str, Any] | str | Path,
    *,
    dry_run: bool = False,
    limit_images: int | None = None,
    experiment_ledger_path: str | Path | None = None,
) -> dict[str, Any]:
    plan = load_stage3_rl_plan(plan_or_path) if not isinstance(plan_or_path, dict) else dict(plan_or_path)
    base_output_root = Path(plan["output_root"])
    output_root = base_output_root / "dry_run" if dry_run else base_output_root
    plan = dict(plan)
    plan["output_root"] = str(output_root)
    plan["output_files"] = _output_files(str(output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / "stage3_rl_data_plan.json"
    _write_json(plan_path, plan)
    ledger_path = output_root / "generation_ledger.jsonl"
    if not ledger_path.exists():
        _append_ledger(ledger_path, "plan_created", plan, {"plan_path": str(plan_path)})

    preflight = preflight_stage3_rl_plan(plan, write_report=True)
    if not preflight["plan_valid"]:
        raise RuntimeError(f"Stage3 RL preflight failed: {preflight['errors']}")

    prior = _load_extension_state(plan.get("extend_from"))
    exclusion_set = load_exclusion_manifests(plan.get("excluded_manifests") or [])
    raw_bundles, dedup_report = _load_and_select_bundles(
        plan,
        exclusion_set=exclusion_set,
        prior_stable_image_uids=prior["stable_image_uids"],
        limit_images=limit_images,
    )
    _append_ledger(
        output_root / "generation_ledger.jsonl",
        "image_manifest_built",
        plan,
        {
            "candidate_images": len(raw_bundles),
            "exclude_counts": exclusion_set.counts(),
            "dedup": dedup_report,
        },
    )

    materialized = [
        _materialize_bundle(bundle, hash_images=plan.get("hash_method") == "sha256_file")
        for bundle in raw_bundles
    ]
    teacher_request_report: dict[str, Any] | None = None
    if plan.get("prompt_generation_mode") in TEACHER_REQUEST_MODES:
        teacher_requests = build_stage3_rl_teacher_requests(materialized, plan=plan)
        teacher_request_report = summarize_teacher_requests(teacher_requests)
        files = plan["output_files"]
        _write_jsonl(Path(files["image_bundles"]), materialized)
        _write_jsonl(Path(files["teacher_requests"]), teacher_requests)
        _write_json(Path(files["teacher_request_summary"]), teacher_request_report)
        _append_ledger(
            output_root / "generation_ledger.jsonl",
            "teacher_requests_written",
            plan,
            {
                "teacher_requests": len(teacher_requests),
                "source_qa_candidates": teacher_request_report.get("source_qa_candidates"),
                "output": files["teacher_requests"],
            },
        )
        if plan.get("prompt_generation_mode") == "teacher_triage":
            empty_filter_report = _empty_filter_report()
            empty_balance_report = _empty_balance_report(plan)
            _write_jsonl(Path(files["qa_candidates"]), [])
            _write_jsonl(Path(files["accepted_rl_prompts"]), [])
            _write_jsonl(Path(files["rejected_rl_prompts"]), [])
            _write_json(Path(files["dedup_report"]), dedup_report)
            _write_json(Path(files["filter_report"]), empty_filter_report)
            _write_json(Path(files["balance_report"]), empty_balance_report)
            summary = _manifest_summary(
                plan=plan,
                bundles=materialized,
                candidates=[],
                accepted=[],
                rejected=[],
                dedup_report=dedup_report,
                filter_report=empty_filter_report,
            )
            summary["teacher_request_summary"] = teacher_request_report
            _write_json(Path(files["manifest_summary"]), summary)
            if not dry_run:
                _write_latest_pointer(output_root)
            if not dry_run and experiment_ledger_path:
                _append_experiment_ledger(Path(experiment_ledger_path), plan, summary)
            return {
                "schema_version": "stage3_rl_execute_result_v0",
                "dry_run": dry_run,
                "output_root": str(output_root),
                "files": files,
                "summary": summary,
            }
    candidates, filter_rejected, filter_report = _build_candidates_and_filter(
        materialized,
        plan,
        prior_sample_ids=prior["sample_ids"],
    )
    _append_ledger(
        output_root / "generation_ledger.jsonl",
        "qa_candidates_built",
        plan,
        {"qa_candidates": len(candidates) + len(filter_rejected)},
    )
    _append_ledger(
        output_root / "generation_ledger.jsonl",
        "filters_applied",
        plan,
        filter_report,
    )
    dedup_report["unresolved_missing_images"] = int(
        (filter_report.get("rejected_counts") or {}).get("missing_image", 0)
    )

    accepted_new, balance_rejected, balance_report = _balance_candidates(
        candidates,
        prior_accepted=prior["accepted"],
        plan=plan,
    )
    accepted_all = [*prior["accepted"], *accepted_new]
    rejected_all = [*prior["rejected"], *filter_rejected, *balance_rejected]
    candidates_all = [*prior["candidates"], *candidates, *filter_rejected]
    bundles_all = [*prior["bundles"], *materialized]

    _append_ledger(output_root / "generation_ledger.jsonl", "balancing_done", plan, balance_report)

    files = plan["output_files"]
    _write_jsonl(Path(files["image_bundles"]), bundles_all)
    _write_jsonl(Path(files["qa_candidates"]), candidates_all)
    _write_jsonl(Path(files["accepted_rl_prompts"]), accepted_all)
    _write_jsonl(Path(files["rejected_rl_prompts"]), rejected_all)
    _write_json(Path(files["dedup_report"]), dedup_report)
    _write_json(Path(files["filter_report"]), filter_report)
    _write_json(Path(files["balance_report"]), balance_report)
    summary = _manifest_summary(
        plan=plan,
        bundles=bundles_all,
        candidates=candidates_all,
        accepted=accepted_all,
        rejected=rejected_all,
        dedup_report=dedup_report,
        filter_report=filter_report,
    )
    if teacher_request_report is not None:
        summary["teacher_request_summary"] = teacher_request_report
    _write_json(Path(files["manifest_summary"]), summary)
    if not dry_run:
        _write_latest_pointer(output_root)

    _append_ledger(
        output_root / "generation_ledger.jsonl",
        "accepted_written",
        plan,
        {"accepted_qa": len(accepted_all), "output": files["accepted_rl_prompts"]},
    )
    _append_ledger(
        output_root / "generation_ledger.jsonl",
        "rejected_written",
        plan,
        {"rejected_qa": len(rejected_all), "output": files["rejected_rl_prompts"]},
    )
    if plan.get("extend_from"):
        _append_ledger(
            output_root / "generation_ledger.jsonl",
            "extension_done",
            plan,
            {
                "extend_from": plan.get("extend_from"),
                "prior_accepted": len(prior["accepted"]),
                "new_accepted": len(accepted_new),
            },
        )
    if not dry_run and experiment_ledger_path:
        _append_experiment_ledger(Path(experiment_ledger_path), plan, summary)

    return {
        "schema_version": "stage3_rl_execute_result_v0",
        "dry_run": dry_run,
        "output_root": str(output_root),
        "files": files,
        "summary": summary,
    }


def finalize_teacher_outputs_stage3_rl_plan(
    plan_or_path: dict[str, Any] | str | Path,
    *,
    experiment_ledger_path: str | Path | None = None,
) -> dict[str, Any]:
    plan = load_stage3_rl_plan(plan_or_path) if not isinstance(plan_or_path, dict) else dict(plan_or_path)
    output_root = Path(plan["output_root"])
    files = plan["output_files"]
    bundles = _read_jsonl(Path(files["image_bundles"]))
    archived_custom_ids = _load_archived_teacher_custom_ids(output_root)
    teacher_outputs = [
        row
        for row in _read_jsonl(output_root / "teacher_outputs.jsonl")
        if str(row.get("custom_id") or "") not in archived_custom_ids
    ]
    if not teacher_outputs:
        raise RuntimeError(f"no parsed teacher outputs found under {output_root}")

    bundle_by_uid = {str(bundle.get("stable_image_uid")): bundle for bundle in bundles}
    request_by_custom_id = {
        str(row.get("custom_id")): row for row in _read_jsonl(output_root / "teacher_requests.jsonl")
    }
    synthetic_bundles_by_uid: dict[str, dict[str, Any]] = {}
    for output_row in teacher_outputs:
        custom_id = str(output_row.get("custom_id") or "")
        request = request_by_custom_id.get(custom_id) or {}
        uid = str(request.get("stable_image_uid") or "")
        if not uid or uid not in bundle_by_uid:
            continue
        bundle = synthetic_bundles_by_uid.setdefault(
            uid,
            {**bundle_by_uid[uid], "qa_items": []},
        )
        source_qas = {
            str(item.get("qa_id")): item
            for item in (bundle_by_uid[uid].get("qa_items") or [])
            if item.get("qa_id")
        }
        image_output = output_row.get("output") or {}
        for index, item in enumerate(image_output.get("items") or []):
            qa_item = _teacher_item_to_qa_item(
                item,
                source_qas=source_qas,
                custom_id=custom_id,
                item_index=index,
                response_id=output_row.get("response_id"),
            )
            if qa_item is not None:
                bundle["qa_items"].append(qa_item)

    synthetic_bundles = [
        bundle for bundle in synthetic_bundles_by_uid.values() if bundle.get("qa_items")
    ]
    candidates, filter_rejected, filter_report = _build_candidates_and_filter(
        synthetic_bundles,
        plan,
        prior_sample_ids=set(),
    )
    accepted, balance_rejected, balance_report = _balance_candidates(
        candidates,
        prior_accepted=[],
        plan=plan,
    )
    rejected = [*filter_rejected, *balance_rejected]
    dedup_report = _read_json(Path(files["dedup_report"]))
    _write_jsonl(Path(files["qa_candidates"]), [*candidates, *filter_rejected])
    _write_jsonl(Path(files["accepted_rl_prompts"]), accepted)
    _write_jsonl(Path(files["rejected_rl_prompts"]), rejected)
    _write_json(Path(files["filter_report"]), filter_report)
    _write_json(Path(files["balance_report"]), balance_report)
    summary = _manifest_summary(
        plan=plan,
        bundles=bundles,
        candidates=[*candidates, *filter_rejected],
        accepted=accepted,
        rejected=rejected,
        dedup_report=dedup_report,
        filter_report=filter_report,
    )
    summary["teacher_outputs"] = len(teacher_outputs)
    summary["archived_teacher_outputs_skipped"] = len(archived_custom_ids)
    summary["teacher_output_items"] = sum(
        len((row.get("output") or {}).get("items") or []) for row in teacher_outputs
    )
    _write_json(Path(files["manifest_summary"]), summary)
    _append_ledger(
        output_root / "generation_ledger.jsonl",
        "teacher_outputs_finalized",
        plan,
        {
            "teacher_outputs": len(teacher_outputs),
            "archived_teacher_outputs_skipped": len(archived_custom_ids),
            "candidate_qa": len(candidates) + len(filter_rejected),
            "accepted_qa": len(accepted),
            "rejected_qa": len(rejected),
        },
    )
    if experiment_ledger_path:
        _append_experiment_ledger(Path(experiment_ledger_path), plan, summary)
    return {
        "schema_version": "stage3_rl_finalize_teacher_outputs_result_v0",
        "output_root": str(output_root),
        "files": files,
        "summary": summary,
    }


def load_exclusion_manifests(paths: Iterable[str | Path]) -> ExclusionSet:
    exclusion = ExclusionSet()
    for path_like in paths:
        path = Path(path_like)
        rows = _read_records(path)
        for row in rows:
            exclusion.rows += 1
            keys = _exclusion_keys(row)
            exclusion.stable_image_uids.update(keys["stable_image_uids"])
            exclusion.image_ids.update(keys["image_ids"])
            exclusion.image_paths.update(keys["image_paths"])
            exclusion.source_ids.update(keys["source_ids"])
            exclusion.image_sha256.update(keys["image_sha256"])
    return exclusion


def _load_archived_teacher_custom_ids(output_root: Path) -> set[str]:
    path = output_root / "api_archived_custom_ids.jsonl"
    return {str(row.get("custom_id") or "") for row in _read_jsonl(path) if row.get("custom_id")}


def _load_and_select_bundles(
    plan: dict[str, Any],
    *,
    exclusion_set: ExclusionSet,
    prior_stable_image_uids: set[str],
    limit_images: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_root = Path(plan["dataset_root"])
    all_bundles: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    duplicate_by_uid = 0
    duplicate_by_path = 0
    excluded = 0
    seen_uids: set[str] = set()
    seen_paths: set[str] = set()
    source_errors: dict[str, str] = {}
    max_images = limit_images or plan.get("max_new_images")
    weights = (plan.get("balance_config") or {}).get("source_weights") or DEFAULT_SOURCE_MIX
    source_limits = _source_selection_limits(
        [SourceConfig.from_dict(item).name for item in plan.get("source_configs") or []],
        max_images=max_images,
        weights=weights,
    )

    for raw_config in plan.get("source_configs") or []:
        config = SourceConfig.from_dict(raw_config)
        try:
            adapter = adapter_for_config(config, dataset_root=dataset_root)
            for bundle in adapter.select_bundles(
                max_bundles=source_limits.get(config.name),
                seed=int(plan["random_seed"]),
            ):
                counts[f"source_sampled_{config.name}"] += 1
                uid = str(bundle.get("stable_image_uid") or "")
                path_key = _path_key(bundle.get("image_path"))
                if uid in prior_stable_image_uids:
                    excluded += 1
                    counts["prior_extension_image_skip"] += 1
                    continue
                if _bundle_excluded(bundle, exclusion_set):
                    excluded += 1
                    counts["cross_exclude_matches"] += 1
                    continue
                if uid in seen_uids:
                    duplicate_by_uid += 1
                    continue
                if path_key and path_key in seen_paths:
                    duplicate_by_path += 1
                    continue
                seen_uids.add(uid)
                if path_key:
                    seen_paths.add(path_key)
                all_bundles.append(bundle)
        except Exception as exc:  # pragma: no cover - exact source failures vary.
            source_errors[config.name] = str(exc)

    selected = _stratified_select_images(
        all_bundles,
        max_images=max_images,
        seed=int(plan["random_seed"]),
        weights=weights,
    )
    report = {
        "schema_version": "stage3_rl_dedup_report_v0",
        "by_source_id": int(duplicate_by_uid),
        "by_sha256": 0,
        "by_phash": 0,
        "by_path": int(duplicate_by_path),
        "cross_exclude_matches": int(excluded),
        "unresolved_missing_images": 0,
        "source_errors": source_errors,
        "source_seen_counts": dict(counts),
        "source_selection_limits": source_limits,
        "candidate_images_after_dedup": len(all_bundles),
        "candidate_images_selected": len(selected),
        "exclude_key_counts": exclusion_set.counts(),
        "created_at": now_iso(),
    }
    return selected, report


def _materialize_bundle(bundle: dict[str, Any], *, hash_images: bool) -> dict[str, Any]:
    image_path = Path(str(bundle.get("image_path") or ""))
    exists = image_path.exists()
    image_sha = sha256_file(image_path) if exists and hash_images else None
    width, height = image_size(image_path) if exists else (0, 0)
    return {
        "schema_version": IMAGE_BUNDLE_SCHEMA_VERSION,
        "bundle_id": bundle["bundle_id"],
        "stable_image_uid": bundle["stable_image_uid"],
        "image_id": bundle["image_id"],
        "image_path": str(image_path),
        "image_sha256": image_sha,
        "image_phash": None,
        "width": width,
        "height": height,
        "source_dataset": bundle["source_dataset"],
        "source_split": bundle["source_split"],
        "source_profile": bundle["source_profile"],
        "source_record_id": bundle["source_record_id"],
        "metadata": bundle.get("metadata") or {},
        "qa_items": bundle.get("qa_items") or [],
    }


def _build_candidates_and_filter(
    bundles: list[dict[str, Any]],
    plan: dict[str, Any],
    *,
    prior_sample_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen_sample_ids = set(prior_sample_ids)
    allowlist = {normalize_text(item) for item in plan.get("source_split_allowlist") or []}
    blocklist = {normalize_text(item) for item in plan.get("source_split_blocklist") or []}
    require_target = bool((plan.get("filter_config") or {}).get("require_target_spec", True))
    created_at = now_iso()

    for bundle in bundles:
        image_exists = Path(bundle["image_path"]).exists()
        split_norm = normalize_text(bundle.get("source_split"))
        seen_questions: set[str] = set()
        for qa_item in bundle.get("qa_items") or []:
            qa_item = _ensure_target_spec(qa_item, bundle)
            candidate = qa_prompt_from_bundle_item(
                bundle=bundle,
                qa_item=qa_item,
                data_plan_id=plan["data_plan_id"],
                created_at=created_at,
            )
            reasons: list[str] = []
            if not image_exists:
                reasons.append("missing_image")
            if not str(candidate.get("question") or "").strip():
                reasons.append("missing_question")
            if not str(candidate.get("gold_answer") or "").strip():
                reasons.append("missing_answer")
            if split_norm in blocklist or (allowlist and split_norm not in allowlist):
                reasons.append("blocked_split")
            if candidate.get("eval_metric") not in SUPPORTED_EVAL_METRICS:
                reasons.append("unsupported_metric")
            if candidate.get("answer_type") not in SUPPORTED_ANSWER_TYPES:
                reasons.append("unsupported_answer_type")
            if candidate.get("evidence_type") not in SUPPORTED_EVIDENCE_TYPES:
                reasons.append("unsupported_evidence_type")
            qa_metadata = (candidate.get("source_metadata") or {}).get("qa_metadata") or {}
            if "teacher_quality" in qa_metadata:
                reasons.extend(_teacher_quality_rejection_reasons(qa_metadata.get("teacher_quality") or {}))
            question_key = normalize_text(candidate.get("question"))
            if question_key in seen_questions:
                reasons.append("duplicate_question")
            seen_questions.add(question_key)
            if candidate["sample_id"] in seen_sample_ids:
                reasons.append("duplicate")
            else:
                seen_sample_ids.add(candidate["sample_id"])
            if require_target or candidate.get("target_spec"):
                reasons.extend(
                    validate_target_spec(
                        candidate.get("target_spec"),
                        answer=candidate.get("gold_answer"),
                        answer_aliases=candidate.get("answer_aliases") or (),
                        choices=candidate.get("original_choices") or candidate.get("choices") or (),
                    )
                )
            if contains_sensitive_identifier(
                " ".join(
                    [
                        str(candidate.get("question") or ""),
                        str(candidate.get("gold_answer") or ""),
                    ]
                )
            ):
                reasons.append("sensitive_personal_info")
            reasons = sorted(set(reasons))
            if reasons:
                for reason in reasons:
                    counts[reason] += 1
                rejected.append(_reject(candidate, reasons))
            else:
                valid.append(candidate)

    report = {
        "schema_version": "stage3_rl_filter_report_v0",
        "filter_version": FILTER_VERSION,
        "input_qa": len(valid) + len(rejected),
        "valid_qa": len(valid),
        "rejected_qa": len(rejected),
        "rejected_counts": _filter_reason_report(counts),
        "created_at": now_iso(),
    }
    return valid, rejected, report


def _ensure_target_spec(qa_item: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    if qa_item.get("target_spec"):
        return qa_item
    spec = build_rule_target_spec(
        question=str(qa_item.get("question") or ""),
        answer=qa_item.get("gold_answer"),
        answer_aliases=qa_item.get("answer_aliases") or (),
        choices=qa_item.get("original_choices") or qa_item.get("choices") or (),
        source_dataset=str(bundle.get("source_dataset") or ""),
        source_profile=str(bundle.get("source_profile") or ""),
        evidence_type=str(qa_item.get("evidence_type") or "unknown"),
    )
    out = dict(qa_item)
    out["target_spec"] = None if spec is None else spec.to_dict()
    out["reference_target"] = None if spec is None else spec.target_text
    return out


def _teacher_item_to_qa_item(
    item: dict[str, Any],
    *,
    source_qas: dict[str, dict[str, Any]],
    custom_id: str,
    item_index: int,
    response_id: Any,
) -> dict[str, Any] | None:
    question = str(item.get("question") or "").strip()
    answer = str(item.get("gold_answer") or "").strip()
    if not question and not answer:
        return None
    source_qa_id = None if item.get("source_qa_id") is None else str(item.get("source_qa_id"))
    source_qa = source_qas.get(source_qa_id or "")
    answer_aliases = [str(value).strip() for value in (item.get("answer_aliases") or []) if str(value).strip()]
    if answer and answer not in answer_aliases:
        answer_aliases.insert(0, answer)
    evidence_type = _normalize_teacher_evidence_type(str(item.get("evidence_type") or "unknown"))
    target_spec = item.get("target_spec") if isinstance(item.get("target_spec"), dict) else None
    if target_spec is not None:
        target_spec = {
            "target_text": str(target_spec.get("target_text") or ""),
            "focus_type": _normalize_teacher_focus_type(str(target_spec.get("focus_type") or "other")),
            "entities": list(target_spec.get("entities") or []),
            "attribute_type": target_spec.get("attribute_type"),
            "relation_type": target_spec.get("relation_type"),
            "source": "teacher",
            "confidence": target_spec.get("confidence"),
        }
    qa_id = stable_hash(
        "stage3rl_teacher_qa",
        custom_id,
        item.get("item_id") or item_index,
        item.get("provenance"),
        source_qa_id,
        question,
        answer,
    )
    metadata = {
        "stage3_provenance": item.get("provenance") or "teacher_generated_legacy_style",
        "teacher_decision": item.get("source_decision"),
        "tool_need_hint": item.get("tool_need_hint"),
        "answer_source": item.get("answer_source"),
        "source_qa_id": source_qa_id,
        "original_question": item.get("original_question"),
        "teacher_quality": item.get("quality") or {},
        "teacher_response_id": response_id,
        "teacher_custom_id": custom_id,
        "teacher_item_index": item_index,
    }
    original_choices = list((source_qa or {}).get("original_choices") or [])
    if source_qa is not None:
        metadata["source_qa"] = {
            "question": source_qa.get("question"),
            "gold_answer": source_qa.get("gold_answer"),
            "answer_aliases": source_qa.get("answer_aliases") or [],
            "original_choices": original_choices,
        }
    return {
        "qa_id": qa_id,
        "question": question,
        "choices": [],
        "original_choices": original_choices,
        "gold_answer": answer,
        "answer_aliases": answer_aliases,
        "answer_format": "short_text",
        "answer_type": str(item.get("answer_type") or "short_text"),
        "eval_metric": str(item.get("eval_metric") or "normalized_exact_match"),
        "evidence_type": evidence_type,
        "difficulty": str(item.get("difficulty_label") or "unknown"),
        "question_family": str(item.get("question_family") or evidence_type),
        "reference_target": None if target_spec is None else target_spec.get("target_text"),
        "target_spec": target_spec,
        "metadata": metadata,
    }


def _normalize_teacher_evidence_type(value: str) -> str:
    aliases = {
        "color": "color_attribute",
        "number": "ocr_text",
        "diagram_detail": "diagram_visual_fact",
        "math_visual_fact": "math_reasoning",
        "other": "unknown",
    }
    return aliases.get(value, value or "unknown")


def _normalize_teacher_focus_type(value: str) -> str:
    aliases = {
        "relative_position": "relative_position",
        "spatial_relation": "relative_position",
        "ocr_text": "ocr",
        "number": "ocr",
        "color_attribute": "attribute",
        "pattern": "texture_material",
        "shape_boundary": "object_part",
    }
    return aliases.get(value, value or "other")


def _balance_candidates(
    candidates: list[dict[str, Any]],
    *,
    prior_accepted: list[dict[str, Any]],
    plan: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    target_total = int(plan["target_accepted_prompts"])
    needed = max(0, target_total - len(prior_accepted))
    if plan.get("max_new_prompts") is not None:
        needed = min(needed, int(plan["max_new_prompts"]))
    max_qa_per_image = int((plan.get("balance_config") or {}).get("max_qa_per_image", 4))
    weights = (plan.get("balance_config") or {}).get("source_weights") or DEFAULT_SOURCE_MIX
    rng = random.Random(int(plan["random_seed"]))
    per_image_counts = Counter(row.get("stable_image_uid") for row in prior_accepted)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    rejected: list[dict[str, Any]] = []

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_source[str(row.get("source_dataset") or "unknown")].append(row)
    for rows in by_source.values():
        rows.sort(key=lambda item: item["sample_id"])
        rng.shuffle(rows)

    quotas = _quotas(needed, weights)
    for source, quota in quotas.items():
        for row in by_source.get(source, []):
            if len([item for item in selected if item.get("source_dataset") == source]) >= quota:
                break
            if _can_accept(row, per_image_counts, max_qa_per_image):
                selected.append(row)
                selected_ids.add(row["sample_id"])
                per_image_counts[row["stable_image_uid"]] += 1

    leftovers = [row for rows in by_source.values() for row in rows if row["sample_id"] not in selected_ids]
    leftovers.sort(key=lambda item: item["sample_id"])
    rng.shuffle(leftovers)
    for row in leftovers:
        if len(selected) >= needed:
            break
        if _can_accept(row, per_image_counts, max_qa_per_image):
            selected.append(row)
            selected_ids.add(row["sample_id"])
            per_image_counts[row["stable_image_uid"]] += 1

    for row in candidates:
        if row["sample_id"] in selected_ids:
            continue
        reason = "target_count_reached" if len(selected) >= needed else "max_qa_per_image"
        rejected.append(_reject(row, [reason]))

    accepted_all = [*prior_accepted, *selected]
    report = {
        "schema_version": "stage3_rl_balance_report_v0",
        "balance_version": BALANCE_VERSION,
        "target_accepted_prompts": target_total,
        "prior_accepted_prompts": len(prior_accepted),
        "new_accepted_prompts": len(selected),
        "actual_accepted_prompts": len(accepted_all),
        "bucket_counts": _bucket_counts(accepted_all),
        "accepted_qa_per_image": dict(Counter(per_image_counts.values())),
        "rejected_by_balance": len(rejected),
        "warnings": _balance_warnings(accepted_all, per_image_counts, plan),
        "created_at": now_iso(),
    }
    return selected, rejected, report


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_size(path: str | Path) -> tuple[int, int]:
    path = Path(path)
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as image:
            width, height = image.size
            return int(width), int(height)
    except Exception:
        pass
    try:
        with path.open("rb") as handle:
            header = handle.read(24)
        if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
            return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")
    except Exception:
        return 0, 0
    return 0, 0


def _load_extension_state(path_like: str | None) -> dict[str, Any]:
    if not path_like:
        return {
            "accepted": [],
            "rejected": [],
            "candidates": [],
            "bundles": [],
            "sample_ids": set(),
            "stable_image_uids": set(),
        }
    root = Path(path_like)
    accepted = _read_jsonl(root / "accepted_rl_prompts.jsonl")
    rejected = _read_jsonl(root / "rejected_rl_prompts.jsonl")
    candidates = _read_jsonl(root / "qa_candidates.jsonl")
    bundles = _read_jsonl(root / "image_bundles.jsonl")
    return {
        "accepted": accepted,
        "rejected": rejected,
        "candidates": candidates,
        "bundles": bundles,
        "sample_ids": {str(row.get("sample_id")) for row in [*accepted, *candidates] if row.get("sample_id")},
        "stable_image_uids": {
            str(row.get("stable_image_uid")) for row in [*accepted, *bundles] if row.get("stable_image_uid")
        },
    }


def _manifest_summary(
    *,
    plan: dict[str, Any],
    bundles: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    dedup_report: dict[str, Any],
    filter_report: dict[str, Any],
) -> dict[str, Any]:
    accepted_images = {row.get("stable_image_uid") for row in accepted}
    return {
        "schema_version": "stage3_rl_manifest_summary_v0",
        "run_id": plan["run_id"],
        "output_root": plan["output_root"],
        "dataset_root": plan["dataset_root"],
        "target_accepted_prompts": plan["target_accepted_prompts"],
        "actual_accepted_prompts": len(accepted),
        "candidate_images": len(bundles),
        "candidate_qa": len(candidates),
        "accepted_images": len(accepted_images),
        "accepted_qa": len(accepted),
        "rejected_qa": len(rejected),
        "source_distribution": _distribution(accepted, "source_dataset"),
        "source_profile_distribution": _distribution(accepted, "source_profile"),
        "answer_type_distribution": _distribution(accepted, "answer_type"),
        "evidence_type_distribution": _distribution(accepted, "evidence_type"),
        "difficulty_distribution": _distribution(accepted, "difficulty"),
        "provenance_distribution": _distribution(accepted, "provenance"),
        "tool_need_hint_distribution": _distribution(accepted, "tool_need_hint"),
        "target_spec_availability": dict(
            Counter("available" if row.get("target_spec") else "missing" for row in accepted)
        ),
        "duplicates_removed": {
            "by_source_id": dedup_report.get("by_source_id", 0),
            "by_path": dedup_report.get("by_path", 0),
            "by_sha256": dedup_report.get("by_sha256", 0),
        },
        "excluded_images_removed": dedup_report.get("cross_exclude_matches", 0),
        "filter_rejected_counts": filter_report.get("rejected_counts", {}),
        "created_at": now_iso(),
        "code_commit": _git_commit(Path(plan["project_root"])),
        "config_hash": plan.get("config_hash"),
    }


def _stratified_select_images(
    bundles: list[dict[str, Any]],
    *,
    max_images: int | None,
    seed: int,
    weights: dict[str, float],
) -> list[dict[str, Any]]:
    if max_images is None or max_images >= len(bundles):
        return sorted(bundles, key=lambda item: item["stable_image_uid"])
    rng = random.Random(seed)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for bundle in bundles:
        by_source[str(bundle.get("source_dataset") or "unknown")].append(bundle)
    for rows in by_source.values():
        rows.sort(key=lambda item: item["stable_image_uid"])
        rng.shuffle(rows)
    selected: list[dict[str, Any]] = []
    selected_uids: set[str] = set()
    for source, quota in _quotas(max_images, weights).items():
        for row in by_source.get(source, [])[:quota]:
            selected.append(row)
            selected_uids.add(row["stable_image_uid"])
    leftovers = [row for row in bundles if row["stable_image_uid"] not in selected_uids]
    leftovers.sort(key=lambda item: item["stable_image_uid"])
    rng.shuffle(leftovers)
    selected.extend(leftovers[: max_images - len(selected)])
    return selected[:max_images]


def _source_selection_limits(
    source_names: list[str],
    *,
    max_images: int | None,
    weights: dict[str, float],
) -> dict[str, int | None]:
    if max_images is None:
        return {name: None for name in source_names}
    quotas = _quotas(max_images, weights)
    fallback = max(1, math.ceil(max_images / max(len(source_names), 1)))
    limits: dict[str, int | None] = {}
    for name in source_names:
        quota = quotas.get(name, fallback)
        limits[name] = max(quota * 3, quota + 10)
    return limits


def _quotas(total: int, weights: dict[str, float]) -> dict[str, int]:
    if total <= 0:
        return {source: 0 for source in weights}
    raw = {source: total * float(weight) for source, weight in weights.items()}
    quotas = {source: int(value) for source, value in raw.items()}
    remainder = total - sum(quotas.values())
    for source, _value in sorted(
        raw.items(), key=lambda item: item[1] - int(item[1]), reverse=True
    )[:remainder]:
        quotas[source] += 1
    return quotas


def _can_accept(row: dict[str, Any], per_image_counts: Counter[str], max_qa_per_image: int) -> bool:
    return per_image_counts[row["stable_image_uid"]] < max_qa_per_image


def _reject(row: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    rejected = dict(row)
    rejected["rejection_reasons"] = sorted(set(reasons))
    metadata = dict(rejected.get("rl_metadata") or {})
    metadata["is_rl_train_eligible"] = False
    metadata["notes"] = sorted(set([*(metadata.get("notes") or []), *reasons]))
    rejected["rl_metadata"] = metadata
    return rejected


def _filter_reason_report(counts: Counter[str]) -> dict[str, int]:
    known = [
        "missing_image",
        "missing_question",
        "missing_answer",
        "unsupported_metric",
        "unsupported_answer_type",
        "unsupported_evidence_type",
        "low_teacher_confidence",
        "teacher_not_visually_verifiable",
        "teacher_answer_not_visible",
        "teacher_too_easy",
        "answer_leakage",
        "target_leakage",
        "generic_target",
        "sensitive_personal_info",
        "blocked_split",
        "duplicate",
        "duplicate_question",
        "source_error",
        "other",
    ]
    report = {key: int(counts.get(key, 0)) for key in known}
    for key, value in counts.items():
        if key not in report:
            report[key] = int(value)
    return report


def _teacher_quality_rejection_reasons(quality: dict[str, Any]) -> list[str]:
    if not quality:
        return ["missing_teacher_quality"]
    reasons: list[str] = []
    confidence = quality.get("confidence")
    if not isinstance(confidence, int | float) or float(confidence) < 0.75:
        reasons.append("low_teacher_confidence")
    if quality.get("visually_verifiable") is not True:
        reasons.append("teacher_not_visually_verifiable")
    if quality.get("answer_visible") is not True:
        reasons.append("teacher_answer_not_visible")
    if quality.get("too_easy") is True:
        reasons.append("teacher_too_easy")
    if str(quality.get("target_leakage_risk") or "").lower() in {"medium", "high"}:
        reasons.append("target_leakage")
    if quality.get("target_generic") is True:
        reasons.append("generic_target")
    return reasons


def _empty_filter_report() -> dict[str, Any]:
    return {
        "schema_version": "stage3_rl_filter_report_v0",
        "filter_version": FILTER_VERSION,
        "input_qa": 0,
        "valid_qa": 0,
        "rejected_qa": 0,
        "rejected_counts": _filter_reason_report(Counter()),
        "created_at": now_iso(),
    }


def _empty_balance_report(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "stage3_rl_balance_report_v0",
        "balance_version": BALANCE_VERSION,
        "target_accepted_prompts": plan["target_accepted_prompts"],
        "prior_accepted_prompts": 0,
        "new_accepted_prompts": 0,
        "actual_accepted_prompts": 0,
        "bucket_counts": _bucket_counts([]),
        "accepted_qa_per_image": {},
        "rejected_by_balance": 0,
        "warnings": ["teacher_triage_requests_only_no_accepted_prompts"],
        "created_at": now_iso(),
    }


def _bucket_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    per_image = Counter(row.get("stable_image_uid") for row in rows)
    return {
        "source_dataset": _distribution(rows, "source_dataset"),
        "source_profile": _distribution(rows, "source_profile"),
        "answer_type": _distribution(rows, "answer_type"),
        "evidence_type": _distribution(rows, "evidence_type"),
        "difficulty": _distribution(rows, "difficulty"),
        "provenance": _distribution(rows, "provenance"),
        "tool_need_hint": _distribution(rows, "tool_need_hint"),
        "target_spec_availability": dict(
            Counter("available" if row.get("target_spec") else "missing" for row in rows)
        ),
        "image_bundle_size": dict(Counter(per_image.values())),
    }


def _balance_warnings(
    accepted: list[dict[str, Any]],
    per_image_counts: Counter[str],
    plan: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    if not accepted:
        return ["no_accepted_prompts"]
    source_threshold = float(
        (plan.get("balance_config") or {}).get("source_dominance_warning_threshold", 0.55)
    )
    answer_threshold = float(
        (plan.get("balance_config") or {}).get("answer_type_dominance_warning_threshold", 0.60)
    )
    total = len(accepted)
    if max(_distribution(accepted, "source_dataset").values() or [0]) / total > source_threshold:
        warnings.append("one_source_dataset_above_threshold")
    if max(_distribution(accepted, "answer_type").values() or [0]) / total > answer_threshold:
        warnings.append("one_answer_type_above_threshold")
    multi_images = sum(1 for count in per_image_counts.values() if count > 1)
    if per_image_counts and multi_images / len(per_image_counts) < 0.20:
        warnings.append("too_few_multi_qa_images")
    target_available = sum(1 for row in accepted if row.get("target_spec"))
    if target_available / total < 0.90:
        warnings.append("too_few_target_spec_samples")
    easy = _distribution(accepted, "difficulty").get("easy", 0)
    if easy / total > 0.75:
        warnings.append("too_many_direct_or_easy_samples")
    verifiable = sum(1 for row in accepted if row.get("gold_answer") and row.get("eval_metric"))
    if verifiable / total < 0.95:
        warnings.append("too_few_verifiable_samples")
    return warnings


def _distribution(rows: list[dict[str, Any]], field_name: str) -> dict[str, int]:
    return dict(Counter(str(row.get(field_name) or "unknown") for row in rows))


def _bundle_excluded(bundle: dict[str, Any], exclusion_set: ExclusionSet) -> bool:
    uid = str(bundle.get("stable_image_uid") or "")
    image_id = str(bundle.get("image_id") or "")
    path = _path_key(bundle.get("image_path"))
    return (
        uid in exclusion_set.stable_image_uids
        or uid in exclusion_set.image_ids
        or image_id in exclusion_set.image_ids
        or image_id in exclusion_set.stable_image_uids
        or (path and path in exclusion_set.image_paths)
    )


def _exclusion_keys(row: dict[str, Any]) -> dict[str, set[str]]:
    stable_image_uids: set[str] = set()
    image_ids: set[str] = set()
    image_paths: set[str] = set()
    source_ids: set[str] = set()
    image_sha256: set[str] = set()

    def add_path(value: Any) -> None:
        key = _path_key(value)
        if key:
            image_paths.add(key)

    for key in ("stable_image_uid", "image_id", "source_image_id"):
        value = row.get(key)
        if value:
            image_ids.add(str(value))
            if ":" in str(value):
                stable_image_uids.add(str(value))
    for key in ("uid", "sample_id", "source_id", "item_content_hash"):
        if row.get(key):
            source_ids.add(str(row[key]))
    for key in ("image", "image_path", "path"):
        add_path(row.get(key))
    for key in ("image_sha256", "sha256", "image_hash"):
        if row.get(key):
            image_sha256.add(str(row[key]))
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    for key in ("stable_image_uid", "image_id", "source_id"):
        value = source.get(key)
        if value:
            image_ids.add(str(value))
            if key == "stable_image_uid" or ":" in str(value):
                stable_image_uids.add(str(value))
            if key == "source_id":
                source_ids.add(str(value))
    if source.get("path"):
        add_path(source.get("path"))
    media = row.get("media") if isinstance(row.get("media"), dict) else {}
    for image in media.get("images") or []:
        add_path(image)
    return {
        "stable_image_uids": stable_image_uids,
        "image_ids": image_ids,
        "image_paths": image_paths,
        "source_ids": source_ids,
        "image_sha256": image_sha256,
    }


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"manifest does not exist: {path}")
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("records", "samples", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [payload]
    return []


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_to_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")


def _append_ledger(path: Path, event: str, plan: dict[str, Any], payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": "stage3_rl_generation_ledger_v0",
        "event": event,
        "run_id": plan.get("run_id"),
        "data_plan_id": plan.get("data_plan_id"),
        "created_at": now_iso(),
        "payload": payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_to_jsonable(record), ensure_ascii=False, sort_keys=True) + "\n")


def _append_experiment_ledger(path: Path, plan: dict[str, Any], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = f"""
### EXP-{now_iso().replace('-', '').replace(':', '')[:13]}-stage3-rl-data

- Status: DONE.
- Question: Build clean Stage3 RL source-QA pool manifest and reports.
- Baseline anchor: Current 50k SFT image pool excluded by manifest.
- Intended diff: Source-QA Stage3 RL pool, image-disjoint from SFT images; no training/eval/judge execution.
- Allowed changed variables: Data source selection, filtering, balancing, TargetSpec rule construction.
- Not allowed to change: SFT 50k data, benchmark eval data, GRPO/reward code, D generation.
- Code commit / worktree: `{summary.get('code_commit')}`, dirty status not recorded by data builder.
- Train data: `{plan.get('output_files', {}).get('accepted_rl_prompts')}`.
- Validation data: none.
- Benchmark output: none.
- Script / command: `tgvf_generate_data stage3-rl --plan {Path(plan['output_root']) / 'stage3_rl_data_plan.json'} --execute`.
- GPUs: none.
- Started: {summary.get('created_at')}.
- Finished: {now_iso()}.
- Metrics: accepted_qa={summary.get('accepted_qa')}, accepted_images={summary.get('accepted_images')}, rejected_qa={summary.get('rejected_qa')}.
- Analysis: Data generation only; reports are under `{plan.get('output_root')}`.
- Conclusion: Stage3 RL source pool generated; reward/judge/training remain non-goals.
- Comparable to baseline: Not a benchmark or training result.
- Follow-up: Run forced probes and GRPO reward design on this source pool separately.
"""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(entry)


def _output_files(output_root: str) -> dict[str, str]:
    root = Path(output_root)
    return {
        "stage3_rl_data_plan": str(root / "stage3_rl_data_plan.json"),
        "image_bundles": str(root / "image_bundles.jsonl"),
        "qa_candidates": str(root / "qa_candidates.jsonl"),
        "teacher_requests": str(root / "teacher_requests.jsonl"),
        "teacher_request_summary": str(root / "teacher_request_summary.json"),
        "accepted_rl_prompts": str(root / "accepted_rl_prompts.jsonl"),
        "rejected_rl_prompts": str(root / "rejected_rl_prompts.jsonl"),
        "dedup_report": str(root / "dedup_report.json"),
        "filter_report": str(root / "filter_report.json"),
        "balance_report": str(root / "balance_report.json"),
        "generation_ledger": str(root / "generation_ledger.jsonl"),
        "manifest_summary": str(root / "manifest_summary.json"),
    }


def _default_exclude_manifests(project_root: Path) -> list[str]:
    return [
        str(project_root / rel)
        for rel in DEFAULT_SFT_EXCLUDE_MANIFESTS
        if (project_root / rel).exists()
    ]


def _resolve_manifest_paths(project_root: Path, paths: Iterable[str]) -> list[str]:
    resolved = []
    for raw in paths:
        path = Path(raw)
        resolved.append(str(path if path.is_absolute() else project_root / path))
    return resolved


def _config_hash(plan: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in plan.items()
        if key not in {"created_at", "config_hash", "output_files"}
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _check_output_writable(output_root: Path) -> bool:
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        probe = output_root / ".write_probe"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
        return True
    except Exception:
        return False


def _looks_like_benchmark_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return "benchmarks" in parts or "benchmark_manifests" in parts


def _path_key(value: Any) -> str:
    if value is None:
        return ""
    raw = str(value)
    if not raw:
        return ""
    path = Path(raw)
    try:
        return str(path.resolve()) if path.exists() else str(path)
    except OSError:
        return str(path)


def _write_latest_pointer(output_root: Path) -> None:
    try:
        (output_root.parent / "LATEST_VERSION.txt").write_text(output_root.name + "\n", encoding="utf-8")
    except Exception:
        pass


def _git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None
