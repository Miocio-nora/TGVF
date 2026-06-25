#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "tgvf_rl_dataset_v0"
DEFAULT_SFT_SOURCE = "data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k/final/tgvf_teacher_items.accepted.jsonl"
DEFAULT_RL_PROMPT_SOURCE = "data/tgvf_teacher/generated/runs/tgvf_v3_teacher_val_2k/final/tgvf_teacher_items.accepted.jsonl"
DEFAULT_OUTPUT_ROOT = "data/tgvf_rl/v0"
DEFAULT_ROLLOUTS = (
    "outputs/full_benchmarks_oldmaskprob075_throughanswer_correct_stage1_4gpu_20260623_005426/"
    "hr_bench_4k/free/merged_rows.jsonl",
    "outputs/full_benchmarks_oldmaskprob075_throughanswer_correct_stage1_4gpu_20260623_005426/"
    "hr_bench_4k/free_softforce_focus/merged_rows.jsonl",
    "outputs/full_benchmarks_oldmaskprob075_throughanswer_correct_stage1_4gpu_20260623_005426/"
    "blink/free/merged_rows.jsonl",
    "outputs/full_benchmarks_oldmaskprob075_throughanswer_correct_stage1_4gpu_20260623_005426/"
    "blink/free_softforce_focus/merged_rows.jsonl",
    "outputs/full_benchmarks_oldmaskprob075_throughanswer_correct_stage1_4gpu_20260623_005426/"
    "ocrbench_v2/free/merged_rows.jsonl",
    "outputs/full_benchmarks_oldmaskprob075_throughanswer_correct_stage1_4gpu_20260623_005426/"
    "ocrbench_v2/free_softforce_focus/merged_rows.jsonl",
)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sft_source": {},
        "rl_prompts": {},
        "rollouts": {},
        "disjoint_checks": {},
        "notes": [
            "Existing 50k SFT data is referenced as an exclusion set, not copied.",
            "RL prompt rows must be disjoint from SFT source IDs and image IDs.",
            "Rollout files are intended as reward/eval diagnostics unless explicitly promoted.",
            "Benchmark-derived samples must stay split from final benchmark reporting.",
        ],
    }

    sft_source = Path(args.sft_source)
    sft_keys = collect_exclusion_keys_from_raw(sft_source)
    manifest["sft_source"] = summarize_sft_source(sft_source, sft_keys)

    rl_prompt_out = output_root / "rl_prompt_seed.heldout.jsonl"
    manifest["rl_prompts"]["heldout"] = convert_teacher_file(
        Path(args.rl_prompt_source),
        rl_prompt_out,
        split="heldout",
        dataset_role="rl_prompt_seed",
        protocol=args.protocol,
        max_records=args.max_rl_prompt_records,
        exclude_keys=sft_keys,
    )
    manifest["disjoint_checks"]["sft_50k_vs_rl_prompt_heldout"] = verify_disjoint_against_exclusion(
        sft_keys,
        rl_prompt_out,
        left_name="tgvf_v4_teacher_50k_sft",
        right_name="rl_prompt_seed.heldout",
    )

    rollout_paths = [Path(item) for item in args.rollout]
    rollout_out = output_root / "benchmark_rollout_eval.jsonl"
    manifest["rollouts"]["eval"] = convert_rollout_files(
        rollout_paths,
        rollout_out,
        dataset_role="benchmark_rollout_eval",
        protocol=args.protocol,
        max_records_per_file=args.max_rollout_records_per_file,
    )

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "manifest": str(manifest_path), **compact_manifest(manifest)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a unified TGVF RL dataset v0.")
    parser.add_argument("--sft-source", default=DEFAULT_SFT_SOURCE)
    parser.add_argument("--rl-prompt-source", default=DEFAULT_RL_PROMPT_SOURCE)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--protocol", default="protocol_c_tool_observation")
    parser.add_argument("--max-rl-prompt-records", type=int, default=None)
    parser.add_argument("--max-rollout-records-per-file", type=int, default=None)
    parser.add_argument("--rollout", action="append", default=list(DEFAULT_ROLLOUTS))
    return parser.parse_args()


def convert_teacher_file(
    input_path: Path,
    output_path: Path,
    *,
    split: str,
    dataset_role: str,
    protocol: str,
    max_records: int | None,
    exclude_keys: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    stats = new_stats(input_path=input_path, output_path=output_path)
    stats["excluded_overlap_rows"] = 0
    with input_path.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as sink:
        for index, record in enumerate(read_jsonl(source)):
            if max_records is not None and index >= max_records:
                break
            if exclude_keys is not None and raw_record_overlaps_exclusion(record, exclude_keys):
                stats["excluded_overlap_rows"] += 1
                continue
            sample = teacher_to_rl_sample(
                record,
                input_path=input_path,
                row_index=index,
                split=split,
                dataset_role=dataset_role,
                protocol=protocol,
            )
            sink.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
            update_stats(stats, sample)
    finish_stats(stats, output_path)
    return stats


def convert_rollout_files(
    input_paths: list[Path],
    output_path: Path,
    *,
    dataset_role: str,
    protocol: str,
    max_records_per_file: int | None,
) -> dict[str, Any]:
    stats = new_stats(input_path=None, output_path=output_path)
    stats["input_paths"] = []
    with output_path.open("w", encoding="utf-8") as sink:
        for input_path in input_paths:
            if not input_path.exists():
                stats.setdefault("missing_inputs", []).append(str(input_path))
                continue
            stats["input_paths"].append(str(input_path))
            benchmark, rollout_split = infer_benchmark_and_split(input_path)
            with input_path.open(encoding="utf-8") as source:
                for index, record in enumerate(read_jsonl(source)):
                    if max_records_per_file is not None and index >= max_records_per_file:
                        break
                    sample = rollout_to_rl_sample(
                        record,
                        input_path=input_path,
                        row_index=index,
                        benchmark=benchmark,
                        split=rollout_split,
                        dataset_role=dataset_role,
                        protocol=protocol,
                    )
                    sink.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
                    update_stats(stats, sample)
    finish_stats(stats, output_path)
    return stats


def teacher_to_rl_sample(
    record: dict[str, Any],
    *,
    input_path: Path,
    row_index: int,
    split: str,
    dataset_role: str,
    protocol: str,
) -> dict[str, Any]:
    source_id = str(record.get("v4_uid") or record.get("uid") or record.get("source_uid") or row_index)
    sample_id = stable_id("teacher", split, source_id)
    choices = normalize_choices(record.get("choices"))
    original_choice = ((record.get("metadata") or {}).get("choice_to_open_answer") or {})
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_role": dataset_role,
        "sample_id": sample_id,
        "source": {
            "kind": "teacher",
            "path": str(input_path),
            "split": split,
            "row_index": row_index,
            "source_id": source_id,
            "image_id": record.get("image_id"),
            "stable_image_uid": record.get("stable_image_uid"),
            "source_dataset": record.get("source_dataset"),
            "source_profile": record.get("source_profile"),
            "teacher_prompt_version": record.get("teacher_prompt_version"),
            "teacher_schema_version": record.get("schema_version"),
        },
        "media": {
            "images": [record["image"]] if record.get("image") else [],
            "videos": [],
        },
        "prompt": {
            "protocol": protocol,
            "question": record.get("question") or "",
            "choices": choices,
            "original_choices": original_choice.get("original_choices") or choices,
        },
        "supervision": {
            "gold_answer": record.get("answer") or record.get("short_answer") or "",
            "short_answer": record.get("short_answer"),
            "answer_format": record.get("answer_format"),
            "need_focus": bool(record.get("need_focus")),
            "evidence_state": record.get("evidence_state"),
            "trajectory_type": record.get("trajectory_type"),
            "target": record.get("target") or "",
            "focus_steps": record.get("focus_steps") or [],
            "evidence_description": record.get("evidence_description") or "",
            "pre_focus_think": record.get("pre_focus_think"),
            "post_focus_think": record.get("post_focus_think"),
            "no_focus_think": record.get("no_focus_think"),
            "value_span_text": record.get("value_span_text"),
        },
        "reward": {
            "available": False,
            "score": None,
            "components": {},
        },
        "metadata": {
            "confidence": record.get("confidence"),
            "question_type": record.get("question_type"),
            "focus_category": record.get("focus_category"),
            "evidence_type": record.get("evidence_type"),
            "target_style": record.get("target_style"),
            "target_cues": record.get("target_cues") or [],
            "target_leakage_risk": record.get("target_leakage_risk"),
            "evidence_specificity": record.get("evidence_specificity"),
            "choice_to_open_answer": original_choice or None,
            "v4_item_type": record.get("v4_item_type"),
        },
    }


def rollout_to_rl_sample(
    record: dict[str, Any],
    *,
    input_path: Path,
    row_index: int,
    benchmark: str,
    split: str,
    dataset_role: str,
    protocol: str,
) -> dict[str, Any]:
    source_id = str(record.get("id") or row_index)
    method = str(record.get("method") or "unknown")
    sample_id = stable_id("rollout", benchmark, split, method, source_id, str(row_index))
    score = record.get("score")
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_role": dataset_role,
        "sample_id": sample_id,
        "source": {
            "kind": "benchmark_rollout",
            "path": str(input_path),
            "split": split,
            "row_index": row_index,
            "source_id": source_id,
            "benchmark": benchmark,
            "method": method,
        },
        "media": {
            "images": [],
            "videos": [],
        },
        "prompt": {
            "protocol": protocol,
            "question": record.get("question") or "",
            "choices": normalize_choices(record.get("choices")),
        },
        "supervision": {
            "gold_answer": record.get("label"),
            "parsed_answer": record.get("parsed_answer") or record.get("pred_letter"),
            "candidate_output": record.get("final_raw_output") or record.get("raw_output") or "",
            "raw_output": record.get("raw_output") or "",
            "final_raw_output": record.get("final_raw_output"),
            "focus_target": record.get("focus_target") or "",
        },
        "reward": {
            "available": score is not None,
            "score": score,
            "components": {
                "answer_correct": score,
                "answer_parse_success": record.get("answer_parse_success"),
                "focus_valid": record.get("focus_valid"),
                "focus_miss": record.get("focus_miss"),
                "trigger_focus_decision": record.get("trigger_focus_decision"),
                "append_success": record.get("append_success"),
                "malformed": record.get("malformed"),
                "continuation_not_im_end": record.get("continuation_not_im_end"),
            },
        },
        "metadata": {
            "benchmark_metadata": record.get("metadata") or {},
            "category": record.get("category"),
            "capture_errors": record.get("capture_errors") or [],
            "capture_stop_reason": record.get("capture_stop_reason"),
            "D_shape": record.get("D_shape"),
            "H_q_shape": record.get("H_q_shape"),
            "target_token_count": record.get("target_token_count"),
            "fvt_position_mode": record.get("fvt_position_mode"),
            "second_full_forward_used": record.get("second_full_forward_used"),
        },
    }


def normalize_choices(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def infer_benchmark_and_split(path: Path) -> tuple[str, str]:
    parts = path.parts
    for name in ("vstar_bench", "hr_bench_4k", "ocrbench_v2", "blink", "mmmu_pro", "mathvista", "mathverse"):
        if name in parts:
            idx = parts.index(name)
            split = parts[idx + 1] if idx + 1 < len(parts) else "unknown"
            return name, split
    if len(parts) >= 3:
        return parts[-3], parts[-2]
    return "unknown", "unknown"


def read_jsonl(handle: Any) -> Any:
    for line in handle:
        line = line.strip()
        if line:
            yield json.loads(line)


def stable_id(*parts: str) -> str:
    payload = "\x1f".join(parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:24]


def new_stats(*, input_path: Path | None, output_path: Path) -> dict[str, Any]:
    return {
        "input_path": None if input_path is None else str(input_path),
        "output_path": str(output_path),
        "rows": 0,
        "by_role": Counter(),
        "by_source_dataset": Counter(),
        "by_source_profile": Counter(),
        "by_benchmark": Counter(),
        "by_method": Counter(),
        "by_split": Counter(),
        "by_need_focus": Counter(),
        "by_trajectory_type": Counter(),
        "by_answer_format": Counter(),
        "reward_available": 0,
        "reward_score_sum": 0.0,
        "reward_score_count": 0,
    }


def update_stats(stats: dict[str, Any], sample: dict[str, Any]) -> None:
    stats["rows"] += 1
    stats["by_role"][sample.get("dataset_role")] += 1
    source = sample.get("source") or {}
    stats["by_source_dataset"][source.get("source_dataset")] += 1
    stats["by_source_profile"][source.get("source_profile")] += 1
    stats["by_benchmark"][source.get("benchmark")] += 1
    stats["by_method"][source.get("method")] += 1
    stats["by_split"][source.get("split")] += 1
    supervision = sample.get("supervision") or {}
    stats["by_need_focus"][str(supervision.get("need_focus"))] += 1
    stats["by_trajectory_type"][supervision.get("trajectory_type")] += 1
    stats["by_answer_format"][supervision.get("answer_format")] += 1
    reward = sample.get("reward") or {}
    if reward.get("available"):
        stats["reward_available"] += 1
        score = reward.get("score")
        if isinstance(score, int | float):
            stats["reward_score_sum"] += float(score)
            stats["reward_score_count"] += 1


def finish_stats(stats: dict[str, Any], output_path: Path) -> None:
    for key, value in list(stats.items()):
        if isinstance(value, Counter):
            stats[key] = {
                str(k): v
                for k, v in sorted(value.items(), key=lambda item: (str(item[0]), item[1]))
                if k is not None
            }
    count = stats.pop("reward_score_count")
    score_sum = stats.pop("reward_score_sum")
    stats["reward_score_mean"] = None if count == 0 else score_sum / count
    stats["sha256"] = file_sha256(output_path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group, payload in manifest.items():
        if group not in {"rl_prompts", "rollouts"}:
            continue
        out[group] = {}
        for name, stats in payload.items():
            out[group][name] = {
                "rows": stats.get("rows"),
                "output_path": stats.get("output_path"),
                "reward_score_mean": stats.get("reward_score_mean"),
            }
    return out


def summarize_sft_source(path: Path, keys: dict[str, set[str]]) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "path": str(path),
        "rows": 0,
        "sft_text_rows": 0,
        "by_source_dataset": Counter(),
        "by_need_focus": Counter(),
        "by_item_type": Counter(),
    }
    with path.open(encoding="utf-8") as handle:
        for record in read_jsonl(handle):
            stats["rows"] += 1
            if record.get("sft_text"):
                stats["sft_text_rows"] += 1
            stats["by_source_dataset"][record.get("source_dataset")] += 1
            stats["by_need_focus"][str(record.get("need_focus"))] += 1
            stats["by_item_type"][record.get("item_type")] += 1
    for key, value in list(stats.items()):
        if isinstance(value, Counter):
            stats[key] = {str(k): v for k, v in sorted(value.items(), key=lambda item: str(item[0])) if k is not None}
    stats["exclusion_key_counts"] = {name: len(values) for name, values in keys.items()}
    return stats


def verify_disjoint_against_exclusion(
    left_keys: dict[str, set[str]],
    right_path: Path,
    *,
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    right_keys = collect_exclusion_keys_from_rl(right_path)
    overlap: dict[str, list[str]] = {}
    for name, values in right_keys.items():
        hits = values & left_keys.get(name, set())
        if hits:
            overlap[name] = sorted(hits)[:20]
    result = {
        "left": left_name,
        "right": right_name,
        "left_key_counts": {name: len(values) for name, values in left_keys.items()},
        "right_key_counts": {name: len(values) for name, values in right_keys.items()},
        "overlap_count": sum(len(values) for values in overlap.values()),
        "passed": not overlap,
    }
    if overlap:
        result["overlap_examples"] = overlap
        raise RuntimeError(f"{left_name} and {right_name} are not disjoint: {result['overlap_count']} overlapping keys")
    return result


def collect_exclusion_keys_from_raw(path: Path) -> dict[str, set[str]]:
    keys = empty_exclusion_keys()
    with path.open(encoding="utf-8") as handle:
        for record in read_jsonl(handle):
            add_raw_exclusion_keys(keys, record)
    return keys


def collect_exclusion_keys_from_rl(path: Path) -> dict[str, set[str]]:
    keys = empty_exclusion_keys()
    with path.open(encoding="utf-8") as handle:
        for record in read_jsonl(handle):
            source = record.get("source") or {}
            for name, source_key in (
                ("source_ids", "source_id"),
                ("image_ids", "image_id"),
                ("stable_image_uids", "stable_image_uid"),
            ):
                value = source.get(source_key)
                if value:
                    keys[name].add(str(value))
    return keys


def empty_exclusion_keys() -> dict[str, set[str]]:
    return {"source_ids": set(), "image_ids": set(), "stable_image_uids": set()}


def add_raw_exclusion_keys(keys: dict[str, set[str]], record: dict[str, Any]) -> None:
    for key in ("uid", "v4_uid", "source_uid", "item_id"):
        value = record.get(key)
        if value:
            keys["source_ids"].add(str(value))
    if record.get("image_id"):
        keys["image_ids"].add(str(record["image_id"]))
    if record.get("stable_image_uid"):
        keys["stable_image_uids"].add(str(record["stable_image_uid"]))


def raw_record_overlaps_exclusion(record: dict[str, Any], keys: dict[str, set[str]]) -> bool:
    candidate = empty_exclusion_keys()
    add_raw_exclusion_keys(candidate, record)
    return any(candidate[name] & keys.get(name, set()) for name in candidate)


if __name__ == "__main__":
    main()
