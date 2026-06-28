"""Clean Stage3 RL-GRPO training launch planner."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

from revisit_vlm_clean.cli.common import exit_not_implemented, print_json
from revisit_vlm_clean.data_generation import file_identity
from revisit_vlm_clean.defaults import DEFAULT_MAX_IMAGE_RESOLUTION, DEFAULT_MODEL_ID, DEFAULT_PROTOCOL
from revisit_vlm_clean.stage3_grpo.data import dataset_identity
from revisit_vlm_clean.stage3_grpo.schemas import (
    JudgeConfig,
    ProbeConfig,
    RewardConfig,
    RolloutConfig,
    STAGE3_GRPO_PLAN_SCHEMA_VERSION,
    Stage3GRPOConfig,
    TrainConfig,
    WandbConfig,
    now_iso,
    write_json,
)
from revisit_vlm_clean.tgvf_protocol import SUPPORTED_PROTOCOLS

DEFAULT_STAGE3_RL_DATA_PATH = (
    "revisit_vlm_clean/data/stage3_rl/"
    "v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean Stage3 RL-GRPO training launch planner.")
    parser.add_argument("--print-defaults", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--rl-data-path", default=DEFAULT_STAGE3_RL_DATA_PATH)
    parser.add_argument("--output-dir")
    parser.add_argument("--policy-checkpoint")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--processor-id", default=None)
    parser.add_argument("--protocol", choices=SUPPORTED_PROTOCOLS, default=DEFAULT_PROTOCOL)
    parser.add_argument("--max-image-resolution", type=int, default=DEFAULT_MAX_IMAGE_RESOLUTION)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")

    parser.add_argument("--runtime-backend", choices=("fake", "native_single_focus"), default="fake")
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=224)
    parser.add_argument("--max-action-tokens", type=int, default=96)
    parser.add_argument("--max-answer-tokens", type=int, default=128)
    parser.add_argument("--max-tool-calls", type=int, default=1)
    parser.add_argument(
        "--dynamic-filter-action",
        choices=("log", "skip", "downweight"),
        default="log",
    )
    parser.add_argument("--low-variance-eps", type=float, default=1e-6)

    parser.add_argument("--probe-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--probe-cache-path", default=None)
    parser.add_argument("--probe-num-off", type=int, default=4)
    parser.add_argument("--probe-num-on-clean", type=int, default=4)
    parser.add_argument("--probe-tau", type=float, default=0.25)
    parser.add_argument(
        "--missing-probe-policy",
        choices=("teacher_hint", "unknown"),
        default="teacher_hint",
    )
    parser.add_argument("--hint-label-weight", type=float, default=0.5)

    parser.add_argument("--w-answer", type=float, default=2.0)
    parser.add_argument("--w-tool", type=float, default=1.0)
    parser.add_argument("--w-focus", type=float, default=1.0)
    parser.add_argument("--w-ground", type=float, default=1.0)
    parser.add_argument("--lambda-call", type=float, default=0.05)
    parser.add_argument("--protocol-penalty", type=float, default=-1.0)
    parser.add_argument("--focus-zero-reward", type=float, default=0.0)
    parser.add_argument("--grounding-zero-reward", type=float, default=-1.0)
    parser.add_argument(
        "--reward-normalization",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--judge-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--judge-mode",
        choices=("offline", "cache_only", "log_only", "disabled"),
        default="cache_only",
    )
    parser.add_argument("--judge-model", default="offline_cache")
    parser.add_argument("--focus-cache-path", default=None)
    parser.add_argument("--grounding-cache-path", default=None)
    parser.add_argument("--judge-pending-path", default=None)
    parser.add_argument("--judge-prompt-version", default="stage3_grpo_judge_v0")
    parser.add_argument("--judge-cache-miss-reward", type=float, default=0.0)

    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--optimizer", choices=("adamw", "manual_sgd"), default="adamw")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--per-device-prompt-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--train-eps", type=float, default=1e-6)
    parser.add_argument("--lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-steps", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260627)

    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-job-type", default="stage3_grpo")
    parser.add_argument("--wandb-tags", default=None, help="Comma-separated W&B tags.")
    parser.add_argument("--wandb-log-artifacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--wandb-log-checkpoint-artifact",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upload Stage3 checkpoint as a W&B artifact. Disabled by default for large checkpoints.",
    )

    parser.add_argument("--dry-run", action="store_true", help="Print the resolved Stage3 plan.")
    parser.add_argument("--write-plan", action="store_true", help="Write Stage3 plan artifacts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.print_defaults:
        print_json(_defaults())
        return 0
    if not (args.dry_run or args.write_plan):
        return exit_not_implemented("use --dry-run or --write-plan to produce a Stage3 GRPO plan")
    config = _config_from_args(args)
    git_commit, dirty_worktree = _git_identity()
    plan = build_stage3_grpo_training_plan(
        config,
        git_commit=git_commit,
        dirty_worktree=dirty_worktree,
    )
    if args.dry_run:
        print_json(plan)
        return 0
    print_json(write_stage3_grpo_training_plan(args.output_dir, plan))
    return 0


def build_stage3_grpo_training_plan(
    config: Stage3GRPOConfig,
    *,
    git_commit: str | None,
    dirty_worktree: bool | None,
) -> dict[str, Any]:
    config.validate()
    data_identity = dataset_identity(config.rl_data_path)
    if not data_identity["file"]["exists"]:
        raise FileNotFoundError(f"rl_data_path does not exist: {config.rl_data_path}")
    checkpoint_identity = file_identity(config.policy_checkpoint).to_dict()
    if not checkpoint_identity["exists"]:
        raise FileNotFoundError(f"policy_checkpoint does not exist: {config.policy_checkpoint}")
    processor_identity = _optional_path_identity(config.processor_id)
    return {
        "schema_version": STAGE3_GRPO_PLAN_SCHEMA_VERSION,
        "created_at": now_iso(),
        "git_commit": git_commit,
        "dirty_worktree": dirty_worktree,
        "config": config.to_dict(),
        "dataset_identity": data_identity,
        "policy_checkpoint_identity": checkpoint_identity,
        "processor_identity": processor_identity,
        "summary": {
            "run_id": config.run_id,
            "runtime_backend": config.rollout.runtime_backend,
            "rl_sample_count": data_identity["rows"],
            "group_size": config.rollout.group_size,
            "rollouts_per_prompt": config.rollout.group_size,
            "world_size": config.train.world_size,
            "global_rollouts_per_step": (
                config.train.world_size
                * config.train.per_device_prompt_batch_size
                * config.train.gradient_accumulation_steps
                * config.rollout.group_size
            ),
            "max_tool_calls": config.rollout.max_tool_calls,
            "reward_weights": {
                "answer": config.reward.w_answer,
                "tool": config.reward.w_tool,
                "focus": config.reward.w_focus,
                "ground": config.reward.w_ground,
                "protocol_gate": "additive",
            },
            "will_launch_training": False,
            "plan_only": True,
            "wandb_enabled": bool(config.wandb.project and config.wandb.mode != "disabled"),
        },
    }


def write_stage3_grpo_training_plan(output_dir: str | Path, plan: dict[str, Any]) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "stage3_grpo_training_plan.json"
    text_path = out / "stage3_grpo_training_plan.txt"
    dataset_path = out / "stage3_grpo_dataset_identity.json"
    write_json(plan_path, plan)
    write_json(dataset_path, plan["dataset_identity"])
    text_path.write_text(_plan_text(plan), encoding="utf-8")
    return {
        "output_dir": str(out),
        "training_plan": str(plan_path),
        "training_plan_txt": str(text_path),
        "dataset_identity": str(dataset_path),
    }


def _optional_path_identity(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    resolved = Path(path)
    if not resolved.exists():
        return {"path": str(resolved), "exists": False}
    if resolved.is_dir():
        return {
            "path": str(resolved),
            "exists": True,
            "type": "directory",
            "child_count": sum(1 for _ in resolved.iterdir()),
        }
    return file_identity(resolved).to_dict()


def _config_from_args(args: argparse.Namespace) -> Stage3GRPOConfig:
    if not args.run_id:
        raise ValueError("--run-id is required")
    if not args.rl_data_path:
        raise ValueError("--rl-data-path is required")
    if not args.output_dir:
        raise ValueError("--output-dir is required")
    if not args.policy_checkpoint:
        raise ValueError("--policy-checkpoint is required")
    return Stage3GRPOConfig(
        run_id=args.run_id,
        rl_data_path=args.rl_data_path,
        output_dir=args.output_dir,
        policy_checkpoint=args.policy_checkpoint,
        model_id=args.model_id,
        processor_id=args.processor_id,
        protocol=args.protocol,
        max_image_resolution=args.max_image_resolution,
        dtype=args.dtype,
        device=args.device,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        rollout=RolloutConfig(
            group_size=args.group_size,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            max_action_tokens=args.max_action_tokens,
            max_answer_tokens=args.max_answer_tokens,
            max_tool_calls=args.max_tool_calls,
            runtime_backend=args.runtime_backend,
            dynamic_filter_action=args.dynamic_filter_action,
            low_variance_eps=args.low_variance_eps,
        ),
        probe=ProbeConfig(
            enabled=args.probe_enabled,
            cache_path=args.probe_cache_path,
            num_off=args.probe_num_off,
            num_on_clean=args.probe_num_on_clean,
            tau=args.probe_tau,
            missing_policy=args.missing_probe_policy,
            hint_label_weight=args.hint_label_weight,
        ),
        reward=RewardConfig(
            w_answer=args.w_answer,
            w_tool=args.w_tool,
            w_focus=args.w_focus,
            w_ground=args.w_ground,
            lambda_call=args.lambda_call,
            protocol_penalty=args.protocol_penalty,
            focus_zero_reward=args.focus_zero_reward,
            grounding_zero_reward=args.grounding_zero_reward,
            reward_normalization=args.reward_normalization,
        ),
        judge=JudgeConfig(
            enabled=args.judge_enabled,
            mode=args.judge_mode,
            model=args.judge_model,
            focus_cache_path=args.focus_cache_path,
            grounding_cache_path=args.grounding_cache_path,
            pending_path=args.judge_pending_path,
            prompt_version=args.judge_prompt_version,
            cache_miss_reward=args.judge_cache_miss_reward,
        ),
        train=TrainConfig(
            optimizer=args.optimizer,
            max_steps=args.max_steps,
            world_size=args.world_size,
            per_device_prompt_batch_size=args.per_device_prompt_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            kl_coef=args.kl_coef,
            clip_range=args.clip_range,
            eps=args.train_eps,
            lora=args.lora,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            max_grad_norm=args.max_grad_norm,
            seed=args.seed,
        ),
        wandb=WandbConfig(
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            name=args.wandb_run_name,
            group=args.wandb_group,
            job_type=args.wandb_job_type,
            tags=_parse_csv(args.wandb_tags),
            log_artifacts=args.wandb_log_artifacts,
            log_checkpoint_artifact=args.wandb_log_checkpoint_artifact,
        ),
    )


def _defaults() -> dict[str, Any]:
    config = Stage3GRPOConfig(
        run_id="stage3_grpo_smoke",
        rl_data_path=DEFAULT_STAGE3_RL_DATA_PATH,
        output_dir="revisit_vlm_clean/outputs/stage3_grpo/stage3_grpo_smoke",
        policy_checkpoint="path/to/stage2_checkpoint",
    )
    return {
        "schema_version": STAGE3_GRPO_PLAN_SCHEMA_VERSION,
        "default_config": config.to_dict(),
        "clean_launcher_actions": ["dry_run", "write_plan"],
        "executor_actions": ["preflight_only", "prepare_execution", "precompute_probes", "rollout_only", "launch_training"],
        "note": "runtime_backend=fake validates lightweight plumbing; native_single_focus runs real Stage2/TGVF rollout/replay/GRPO training updates",
    }


def _plan_text(plan: dict[str, Any]) -> str:
    config = plan["config"]
    summary = plan["summary"]
    weights = summary["reward_weights"]
    lines = [
        f"schema_version: {plan['schema_version']}",
        f"created_at: {plan['created_at']}",
        f"run_id: {config['run_id']}",
        f"rl_data_path: {config['rl_data_path']}",
        f"rl_sample_count: {summary['rl_sample_count']}",
        f"policy_checkpoint: {config['policy_checkpoint']}",
        f"model_id: {config['model_id']}",
        f"processor_id: {config.get('processor_id')}",
        f"protocol: {config['protocol']}",
        f"device: {config['device']}",
        f"device_map: {config.get('device_map')}",
        f"runtime_backend: {summary['runtime_backend']}",
        f"world_size: {config['train']['world_size']}",
        f"optimizer: {config['train'].get('optimizer')}",
        f"group_size: {summary['group_size']}",
        f"global_rollouts_per_step: {summary['global_rollouts_per_step']}",
        f"max_tool_calls: {summary['max_tool_calls']}",
        "reward_weights: "
        f"answer={weights['answer']} tool={weights['tool']} focus={weights['focus']} "
        f"ground={weights['ground']}",
        f"judge_mode: {config['judge']['mode']}",
        f"probe_enabled: {config['probe']['enabled']}",
        f"wandb_project: {config['wandb'].get('project')}",
        f"wandb_mode: {config['wandb'].get('mode')}",
        f"wandb_log_artifacts: {config['wandb'].get('log_artifacts')}",
        f"wandb_log_checkpoint_artifact: {config['wandb'].get('log_checkpoint_artifact')}",
        f"git_commit: {plan.get('git_commit')}",
        f"dirty_worktree: {plan.get('dirty_worktree')}",
    ]
    return "\n".join(lines) + "\n"


def _parse_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _git_identity() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None, None
    return commit or None, bool(status.strip())


if __name__ == "__main__":
    raise SystemExit(main())
