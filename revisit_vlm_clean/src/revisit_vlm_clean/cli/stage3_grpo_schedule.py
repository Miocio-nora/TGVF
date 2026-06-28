"""Build deterministic Stage3 GRPO sample schedules."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from revisit_vlm_clean.cli.common import print_json
from revisit_vlm_clean.data_generation import file_identity
from revisit_vlm_clean.stage3_grpo.data import (
    BalancedPromptSampler,
    STAGE3_GRPO_SAMPLE_SCHEDULE_SCHEMA_VERSION,
    load_stage3_samples,
    sample_schedule_identity,
    tool_bucket_from_hint,
    write_jsonl,
)
from revisit_vlm_clean.stage3_grpo.schemas import now_iso, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a deterministic Stage3 GRPO sample schedule.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--rl-data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--per-device-prompt-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--unique-image", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-schedule", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not (args.dry_run or args.write_schedule):
        raise SystemExit("use --dry-run or --write-schedule")
    rows, summary = build_stage3_grpo_sample_schedule(
        run_id=args.run_id,
        rl_data_path=args.rl_data_path,
        steps=args.steps,
        world_size=args.world_size,
        per_device_prompt_batch_size=args.per_device_prompt_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        unique_image=args.unique_image,
    )
    if args.dry_run:
        print_json(summary)
        return 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = output_dir / "stage3_grpo_sample_schedule.jsonl"
    summary_path = output_dir / "stage3_grpo_sample_schedule_summary.json"
    state_path = output_dir / "stage3_grpo_stepwise_state.json"
    write_jsonl(schedule_path, rows)
    identity = sample_schedule_identity(schedule_path, samples=load_stage3_samples(args.rl_data_path))
    summary = {**summary, "schedule_path": str(schedule_path), "schedule_identity": identity}
    write_json(summary_path, summary)
    write_json(
        state_path,
        {
            "schema_version": "stage3_grpo_stepwise_state_v0",
            "created_at": now_iso(),
            "run_id": args.run_id,
            "status": "planned",
            "target_steps": int(args.steps),
            "completed_steps": [],
            "failed_steps": [],
            "next_step": 1,
            "current_checkpoint": None,
            "schedule_path": str(schedule_path),
            "used_sample_ids": [],
            "used_image_uids": [],
        },
    )
    print_json(
        {
            "status": "sample_schedule_written",
            "schedule_path": str(schedule_path),
            "summary_path": str(summary_path),
            "state_path": str(state_path),
            "rows": len(rows),
        }
    )
    return 0


def build_stage3_grpo_sample_schedule(
    *,
    run_id: str,
    rl_data_path: str | Path,
    steps: int,
    world_size: int,
    per_device_prompt_batch_size: int,
    gradient_accumulation_steps: int,
    seed: int,
    unique_image: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if int(steps) < 1:
        raise ValueError("steps must be >= 1")
    if int(world_size) < 1:
        raise ValueError("world_size must be >= 1")
    if int(per_device_prompt_batch_size) < 1:
        raise ValueError("per_device_prompt_batch_size must be >= 1")
    if int(gradient_accumulation_steps) < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    samples = load_stage3_samples(rl_data_path)
    required = (
        int(steps)
        * int(world_size)
        * int(per_device_prompt_batch_size)
        * int(gradient_accumulation_steps)
    )
    if required > len(samples):
        raise ValueError(f"schedule requires {required} samples but dataset only has {len(samples)}")
    sampler = BalancedPromptSampler(samples, seed=int(seed))
    ordered = _unique_sampler_order(sampler, len(samples))
    chosen = _choose_schedule_samples(ordered, required=required, unique_image=unique_image)
    rows: list[dict[str, Any]] = []
    cursor = 0
    for global_step in range(1, int(steps) + 1):
        for rank in range(int(world_size)):
            for accumulation_index in range(int(gradient_accumulation_steps)):
                for prompt_index in range(int(per_device_prompt_batch_size)):
                    sample = chosen[cursor]
                    cursor += 1
                    rows.append(
                        {
                            "schema_version": STAGE3_GRPO_SAMPLE_SCHEDULE_SCHEMA_VERSION,
                            "created_at": now_iso(),
                            "run_id": run_id,
                            "global_step": global_step,
                            "rank": rank,
                            "accumulation_index": accumulation_index,
                            "prompt_index": prompt_index,
                            "sample_id": sample.sample_id,
                            "stable_image_uid": sample.stable_image_uid,
                            "image_sha256": sample.image_sha256,
                            "source_dataset": sample.source_dataset,
                            "source_profile": sample.source_profile,
                            "source_split": sample.source_split,
                            "answer_type": sample.answer_type,
                            "evidence_type": sample.evidence_type,
                            "difficulty": sample.difficulty,
                            "tool_need_hint": sample.tool_need_hint,
                            "tool_bucket": tool_bucket_from_hint(sample.tool_need_hint),
                        }
                    )
    sample_ids = [row["sample_id"] for row in rows]
    image_uids = [row["stable_image_uid"] for row in rows if row.get("stable_image_uid")]
    summary = {
        "schema_version": "stage3_grpo_sample_schedule_summary_v0",
        "created_at": now_iso(),
        "run_id": run_id,
        "rl_data_path": str(rl_data_path),
        "rl_data_identity": file_identity(rl_data_path).to_dict(),
        "dataset_rows": len(samples),
        "steps": int(steps),
        "world_size": int(world_size),
        "per_device_prompt_batch_size": int(per_device_prompt_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "required_rows": required,
        "schedule_rows": len(rows),
        "seed": int(seed),
        "unique_image": bool(unique_image),
        "duplicate_sample_ids": _duplicate_count(sample_ids),
        "duplicate_image_uids": _duplicate_count(image_uids),
        "source_dataset": dict(Counter(str(row.get("source_dataset") or "unknown") for row in rows)),
        "evidence_type": dict(Counter(str(row.get("evidence_type") or "unknown") for row in rows)),
        "answer_type": dict(Counter(str(row.get("answer_type") or "unknown") for row in rows)),
        "tool_need_hint": dict(Counter(str(row.get("tool_need_hint") or "unknown") for row in rows)),
        "tool_bucket": dict(Counter(str(row.get("tool_bucket") or "unknown") for row in rows)),
        "examples": rows[:8],
    }
    if summary["duplicate_sample_ids"]:
        raise ValueError(f"sample schedule produced duplicate samples: {summary['duplicate_sample_ids']}")
    if unique_image and summary["duplicate_image_uids"]:
        raise ValueError(f"sample schedule produced duplicate images: {summary['duplicate_image_uids']}")
    return rows, summary


def _unique_sampler_order(
    sampler: BalancedPromptSampler,
    sample_count: int,
) -> list[Any]:
    ordered = []
    seen: set[str] = set()
    while len(seen) < int(sample_count):
        sample = sampler.next_batch(1)[0]
        if sample.sample_id in seen:
            break
        seen.add(sample.sample_id)
        ordered.append(sample)
    return ordered


def _choose_schedule_samples(
    ordered: list[Any],
    *,
    required: int,
    unique_image: bool,
) -> list[Any]:
    chosen = []
    used_images: set[str] = set()
    for sample in ordered:
        image_uid = str(sample.stable_image_uid or sample.image_sha256 or sample.image_path)
        if unique_image and image_uid in used_images:
            continue
        chosen.append(sample)
        if image_uid:
            used_images.add(image_uid)
        if len(chosen) >= int(required):
            return chosen
    raise ValueError(
        f"not enough samples after unique_image={unique_image}: "
        f"required={required} available={len(chosen)}"
    )


def _duplicate_count(values: list[str]) -> int:
    counts = Counter(value for value in values if value)
    return sum(count - 1 for count in counts.values() if count > 1)


if __name__ == "__main__":
    raise SystemExit(main())
