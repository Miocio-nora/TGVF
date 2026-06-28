"""Clean Stage3 RL-GRPO executor.

This executor validates the Stage3 plan handoff and launches configured Stage3
GRPO actions. The fake backend validates lightweight plumbing; native_single_focus
loads the Stage2 checkpoint and runs real sampled rollout/reward/replay/GRPO
training updates.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from revisit_vlm_clean.cli.common import exit_not_implemented, print_json
from revisit_vlm_clean.data_generation import file_identity
from revisit_vlm_clean.stage3_grpo.data import (
    dataset_identity,
    load_stage3_samples,
    sample_schedule_identity,
    write_jsonl,
)
from revisit_vlm_clean.stage3_grpo.probe import ProbeCache
from revisit_vlm_clean.stage3_grpo.reward import answer_is_correct
from revisit_vlm_clean.stage3_grpo.rollout import build_rollout_engine
from revisit_vlm_clean.stage3_grpo.schemas import (
    STAGE3_GRPO_EXECUTION_SCHEMA_VERSION,
    STAGE3_GRPO_PLAN_SCHEMA_VERSION,
    STAGE3_GRPO_PROBE_SCHEMA_VERSION,
    Stage3GRPOConfig,
    now_iso,
    read_json,
    write_json,
)
from revisit_vlm_clean.stage3_grpo.trainer import Stage3GRPOTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean Stage3 RL-GRPO executor.")
    parser.add_argument("--plan", required=True, help="Path to stage3_grpo_training_plan.json.")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-report", default=None)
    parser.add_argument("--prepare-execution", action="store_true")
    parser.add_argument("--execution-dir", default=None)
    parser.add_argument("--precompute-probes", action="store_true")
    parser.add_argument("--probe-output", default=None)
    parser.add_argument("--limit-prompts", type=int, default=None)
    parser.add_argument("--rollout-only", action="store_true")
    parser.add_argument("--launch-training", action="store_true")
    parser.add_argument("--launch-report", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan_path, plan, config = load_stage3_grpo_plan(args.plan)
    report = preflight_stage3_grpo_plan(plan_path, plan, config)
    report_path = _preflight_report_path(plan_path, requested=args.preflight_report)
    report["preflight_report"] = str(report_path)
    write_json(report_path, report)
    if report["status"] != "passed":
        print_json(report)
        return 1
    result: dict[str, Any] = {"preflight_report": str(report_path)}
    if args.prepare_execution:
        prepared = prepare_stage3_grpo_execution(
            plan_path=plan_path,
            plan=plan,
            config=config,
            execution_dir=args.execution_dir,
        )
        result.update(prepared)
    if args.precompute_probes:
        result["probe_cache"] = precompute_forced_probes(
            config,
            output_path=args.probe_output,
            limit_prompts=args.limit_prompts,
        )
    if args.rollout_only:
        result["rollout_only"] = run_rollout_only(config)
    if args.launch_training:
        launched = launch_stage3_grpo_training(config, report_path=args.launch_report)
        result["training_launch_result"] = launched
    if result.keys() != {"preflight_report"}:
        print_json(result)
        if (result.get("training_launch_result") or {}).get("status") == "blocked":
            return 2
        return 0
    print_json(report)
    if args.preflight_only:
        return 0
    return exit_not_implemented(
        "explicit execution mode is required; rerun with --preflight-only, "
        "--prepare-execution, --precompute-probes, --rollout-only, or --launch-training"
    )


def load_stage3_grpo_plan(
    path: str | Path,
) -> tuple[Path, dict[str, Any], Stage3GRPOConfig]:
    plan_path = Path(path)
    plan = read_json(plan_path)
    if not isinstance(plan, dict):
        raise ValueError("Stage3 GRPO plan must be a JSON object")
    config_payload = dict(plan.get("config") or plan)
    config = Stage3GRPOConfig.from_dict(config_payload)
    return plan_path, plan, config


def preflight_stage3_grpo_plan(
    plan_path: str | Path,
    plan: dict[str, Any],
    config: Stage3GRPOConfig,
) -> dict[str, Any]:
    config.validate()
    errors: list[str] = []
    warnings: list[str] = []
    data_id = dataset_identity(config.rl_data_path) if Path(config.rl_data_path).exists() else None
    checkpoint_id = file_identity(config.policy_checkpoint).to_dict()
    if not Path(plan_path).exists():
        errors.append(f"plan does not exist: {plan_path}")
    if plan.get("schema_version") != STAGE3_GRPO_PLAN_SCHEMA_VERSION:
        warnings.append(
            f"unexpected plan schema_version: {plan.get('schema_version')!r}"
        )
    if data_id is None or not data_id["file"]["exists"]:
        errors.append(f"rl_data_path does not exist: {config.rl_data_path}")
    elif int(data_id["rows"]) < 1:
        errors.append(f"rl_data_path contains no trainable samples: {config.rl_data_path}")
    if not checkpoint_id["exists"]:
        errors.append(f"policy_checkpoint does not exist: {config.policy_checkpoint}")
    stage2_checkpoint = None
    if config.rollout.runtime_backend == "native_single_focus" and checkpoint_id["exists"]:
        stage2_checkpoint = _native_stage2_checkpoint_preflight(config.policy_checkpoint)
        errors.extend(stage2_checkpoint.get("errors") or [])
    output_parent = Path(config.output_dir).parent
    if not output_parent.exists():
        warnings.append(f"output parent will be created: {output_parent}")
    if config.rollout.runtime_backend == "native_single_focus":
        warnings.append(
            "native_single_focus launch will load the Stage2 checkpoint and run real sampled rollouts"
        )
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if int(config.train.world_size) > 1 and env_world_size not in {1, int(config.train.world_size)}:
        errors.append(
            "train.world_size does not match torchrun WORLD_SIZE: "
            f"{config.train.world_size} != {env_world_size}"
        )
    judge_cache_status = _judge_cache_preflight(config)
    warnings.extend(judge_cache_status.get("warnings") or [])
    if config.judge.enabled and config.judge.mode == "cache_only":
        warnings.append("judge cache misses will be logged and receive configured cache-miss reward")
    schedule_status = None
    if config.sample_schedule_path:
        if not Path(config.sample_schedule_path).exists():
            errors.append(f"sample_schedule_path does not exist: {config.sample_schedule_path}")
        elif data_id is not None:
            samples = load_stage3_samples(config.rl_data_path)
            schedule_status = sample_schedule_identity(
                config.sample_schedule_path,
                samples=samples,
            )
            if schedule_status["missing_sample_ids"]:
                errors.append(
                    "sample schedule references missing sample ids: "
                    + ",".join(schedule_status["missing_sample_ids"][:5])
                )
            if int(schedule_status["duplicate_sample_ids"]) > 0:
                warnings.append(
                    f"sample schedule has duplicate sample ids: {schedule_status['duplicate_sample_ids']}"
                )
            if int(schedule_status["duplicate_image_uids"]) > 0:
                warnings.append(
                    f"sample schedule has duplicate image uids: {schedule_status['duplicate_image_uids']}"
                )
    return {
        "schema_version": "stage3_grpo_preflight_report_v0",
        "created_at": now_iso(),
        "status": "failed" if errors else "passed",
        "errors": errors,
        "warnings": warnings,
        "plan_path": str(plan_path),
        "plan_identity": file_identity(plan_path).to_dict(),
        "run_id": config.run_id,
        "runtime_backend": config.rollout.runtime_backend,
        "rl_data_path": config.rl_data_path,
        "dataset_identity": data_id,
        "policy_checkpoint_identity": checkpoint_id,
        "stage2_checkpoint": stage2_checkpoint,
        "processor_id": config.processor_id,
        "output_dir": config.output_dir,
        "group_size": config.rollout.group_size,
        "world_size": config.train.world_size,
        "global_rollouts_per_step": (
            int(config.train.world_size)
            * int(config.train.per_device_prompt_batch_size)
            * int(config.train.gradient_accumulation_steps)
            * int(config.rollout.group_size)
        ),
        "max_tool_calls": config.rollout.max_tool_calls,
        "judge": config.judge.to_dict(),
        "judge_cache_status": judge_cache_status,
        "probe": config.probe.to_dict(),
        "sample_schedule": schedule_status,
    }


def prepare_stage3_grpo_execution(
    *,
    plan_path: str | Path,
    plan: dict[str, Any],
    config: Stage3GRPOConfig,
    execution_dir: str | Path | None,
) -> dict[str, Any]:
    exec_dir = _execution_dir(Path(plan_path), config=config, requested=execution_dir)
    exec_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = exec_dir / "stage3_grpo_execution_bundle.json"
    status_path = exec_dir / "stage3_grpo_execution_status.json"
    bundle = {
        "schema_version": STAGE3_GRPO_EXECUTION_SCHEMA_VERSION,
        "created_at": now_iso(),
        "plan_path": str(plan_path),
        "plan_identity": file_identity(plan_path).to_dict(),
        "run_id": config.run_id,
        "config": config.to_dict(),
        "dataset_identity": dataset_identity(config.rl_data_path),
        "policy_checkpoint_identity": file_identity(config.policy_checkpoint).to_dict(),
        "launchable_actions": {
            "fake": ["precompute_probes", "rollout_only", "launch_training"],
            "native_single_focus": [
                "precompute_probes",
                "rollout_only",
                "launch_training",
            ],
        },
        "source_plan_summary": plan.get("summary"),
    }
    status = {
        "schema_version": "stage3_grpo_execution_status_v0",
        "created_at": now_iso(),
        "status": "prepared",
        "execution_bundle": str(bundle_path),
        "runtime_backend": config.rollout.runtime_backend,
        "real_backend_ready": config.rollout.runtime_backend in {"fake", "native_single_focus"},
    }
    write_json(bundle_path, bundle)
    write_json(status_path, status)
    return {
        "execution_dir": str(exec_dir),
        "execution_bundle": str(bundle_path),
        "execution_status": str(status_path),
    }


def precompute_forced_probes(
    config: Stage3GRPOConfig,
    *,
    output_path: str | Path | None,
    limit_prompts: int | None,
) -> dict[str, Any]:
    if not config.probe.enabled:
        return {"status": "skipped_probe_disabled"}
    samples = load_stage3_samples(config.rl_data_path, limit=limit_prompts)
    engine = build_rollout_engine(config)
    rows: list[dict[str, Any]] = []
    for sample in samples:
        off_correct = [
            answer_is_correct(
                engine.forced_off(sample, rollout_id=index).final_answer,
                sample,
            )
            for index in range(config.probe.num_off)
        ]
        on_correct = [
            answer_is_correct(
                engine.forced_on_clean(sample, rollout_id=index).final_answer,
                sample,
            )
            for index in range(config.probe.num_on_clean)
        ]
        mean_off = sum(1.0 for item in off_correct if item) / max(len(off_correct), 1)
        mean_on = sum(1.0 for item in on_correct if item) / max(len(on_correct), 1)
        rows.append(
            {
                "schema_version": STAGE3_GRPO_PROBE_SCHEMA_VERSION,
                "sample_id": sample.sample_id,
                "mean_correct_off": mean_off,
                "mean_correct_on_clean": mean_on,
                "delta_tool": mean_on - mean_off,
                "num_off": config.probe.num_off,
                "num_on_clean": config.probe.num_on_clean,
                "reference_target_available": bool(sample.target_text),
                "created_at": now_iso(),
                "backend": config.rollout.runtime_backend,
            }
        )
    path = Path(output_path or config.probe.cache_path or Path(config.output_dir) / "probe_cache.jsonl")
    ProbeCache().write_rows(path, rows)
    summary = {
        "status": "probe_cache_written",
        "path": str(path),
        "rows": len(rows),
        "mean_delta_tool": (
            sum(float(row["delta_tool"]) for row in rows) / len(rows)
            if rows
            else 0.0
        ),
    }
    write_json(Path(config.output_dir) / "probe_cache_summary.json", summary)
    return summary


def run_rollout_only(config: Stage3GRPOConfig) -> dict[str, Any]:
    trainer = Stage3GRPOTrainer(config)
    rollouts = trainer.rollout_batch(global_step=1, accumulation_index=0)
    rewarded, rewards = trainer.reward_rollouts(rollouts)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rollout_path = out / "rollout_debug.jsonl"
    reward_path = out / "reward_breakdown.jsonl"
    write_jsonl(rollout_path, [item.to_dict() for item in rewarded])
    write_jsonl(reward_path, rewards)
    summary = {
        "status": "rollout_only_completed",
        "rollout_debug": str(rollout_path),
        "reward_breakdown": str(reward_path),
        "rollout_count": len(rewarded),
        "reward_count": len(rewards),
    }
    write_json(out / "rollout_only_summary.json", summary)
    return summary


def launch_stage3_grpo_training(
    config: Stage3GRPOConfig,
    *,
    report_path: str | Path | None,
) -> dict[str, Any]:
    trainer = Stage3GRPOTrainer(config)
    result = trainer.run_training()
    distributed = dict(result.get("distributed") or {})
    if distributed and not distributed.get("is_main", True):
        launch_path = Path(str(result.get("rank_output_dir") or config.output_dir)) / "stage3_grpo_launch_result.json"
    else:
        launch_path = Path(report_path or Path(config.output_dir) / "stage3_grpo_launch_result.json")
    result["launch_report"] = str(launch_path)
    write_json(launch_path, result)
    return result


def launch_stage3_grpo_smoke(
    config: Stage3GRPOConfig,
    *,
    report_path: str | Path | None,
) -> dict[str, Any]:
    return launch_stage3_grpo_training(config, report_path=report_path)


def _preflight_report_path(plan_path: Path, *, requested: str | Path | None) -> Path:
    if requested:
        return Path(requested)
    return plan_path.parent / "stage3_grpo_preflight_report.json"


def _execution_dir(
    plan_path: Path,
    *,
    config: Stage3GRPOConfig,
    requested: str | Path | None,
) -> Path:
    if requested:
        return Path(requested)
    output_dir = Path(config.output_dir)
    if output_dir.exists() or output_dir.parent.exists():
        return output_dir / "stage3_grpo_execution"
    return plan_path.parent / "stage3_grpo_execution"


def _native_stage2_checkpoint_preflight(path: str | Path) -> dict[str, Any]:
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:
        return {
            "status": "failed",
            "errors": [f"stage2_checkpoint_load_failed:{type(exc).__name__}:{exc}"],
        }
    if not isinstance(checkpoint, dict):
        return {"status": "failed", "errors": ["stage2_checkpoint_not_mapping"]}
    keys = set(str(key) for key in checkpoint)
    config = checkpoint.get("config") or {}
    errors: list[str] = []
    for key in ("qwen_lora", "tgvf_module", "config"):
        if key not in checkpoint:
            errors.append(f"stage2_checkpoint_missing_{key}")
    if "qwen_lora" in checkpoint and checkpoint.get("qwen_lora") is None:
        errors.append("stage2_checkpoint_empty_qwen_lora")
    if "tgvf_module" in checkpoint and checkpoint.get("tgvf_module") is None:
        errors.append("stage2_checkpoint_empty_tgvf_module")
    if not isinstance(config, dict):
        errors.append("stage2_checkpoint_config_not_mapping")
        config = {}
    for key in ("tgvf", "training"):
        if key not in config:
            errors.append(f"stage2_checkpoint_config_missing_{key}")
    return {
        "status": "failed" if errors else "passed",
        "errors": errors,
        "keys": sorted(keys),
        "global_step": checkpoint.get("global_step"),
        "protocol": config.get("tgvf_protocol"),
        "model_id": config.get("model_id"),
        "processor_id": config.get("processor_id"),
        "has_tgvf_config": isinstance(config.get("tgvf"), dict),
        "has_training_config": isinstance(config.get("training"), dict),
        "has_lora_config": isinstance(config.get("lora"), dict),
    }


def _judge_cache_preflight(config: Stage3GRPOConfig) -> dict[str, Any]:
    if not config.judge.enabled or config.judge.mode == "disabled":
        return {"status": "disabled", "warnings": []}
    warnings: list[str] = []
    focus = _jsonl_cache_identity(config.judge.focus_cache_path)
    ground = _jsonl_cache_identity(config.judge.grounding_cache_path)
    if config.reward.w_focus > 0 and not focus["exists"]:
        warnings.append("focus judge reward is weighted but focus_cache_path is missing")
    if config.reward.w_ground > 0 and not ground["exists"]:
        warnings.append("grounding judge reward is weighted but grounding_cache_path is missing")
    status = "ready"
    if warnings:
        status = "cache_missing"
    return {
        "status": status,
        "mode": config.judge.mode,
        "judge_model": config.judge.model,
        "focus_cache": focus,
        "grounding_cache": ground,
        "pending_path": None if config.judge.pending_path is None else str(config.judge.pending_path),
        "cache_miss_reward": config.judge.cache_miss_reward,
        "warnings": warnings,
    }


def _jsonl_cache_identity(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"exists": False, "path": None, "rows": 0}
    path = Path(path)
    if not path.exists():
        return {"exists": False, "path": str(path), "rows": 0}
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows += 1
    return {"exists": True, "path": str(path), "rows": rows}


if __name__ == "__main__":
    raise SystemExit(main())
