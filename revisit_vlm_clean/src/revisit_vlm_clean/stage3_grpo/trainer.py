"""Small Stage3 GRPO trainer core."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from revisit_vlm_clean.data_generation import file_identity

from .data import BalancedPromptSampler, load_stage3_samples, write_jsonl
from .data import dataset_identity
from .grpo import attach_group_advantages, grpo_loss_from_tensors, rollout_logprob_tensors
from .judge import JudgeBundle
from .probe import ProbeCache
from .reward import score_rollout_reward
from .rollout import build_rollout_engine
from .schemas import RolloutRecord, Stage3GRPOConfig, write_json


class Stage3GRPOTrainer:
    def __init__(self, config: Stage3GRPOConfig) -> None:
        config.validate()
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.samples = load_stage3_samples(config.rl_data_path)
        self.sampler = BalancedPromptSampler(self.samples, seed=config.train.seed)
        self.probe_cache = ProbeCache(config.probe.cache_path)
        self.judge_bundle = JudgeBundle(config.judge, output_dir=self.output_dir)
        self.engine = build_rollout_engine(config)
        self._wandb_logger = None

    def rollout_batch(self, *, prompt_batch_size: int | None = None) -> list[RolloutRecord]:
        prompts = self.sampler.next_batch(prompt_batch_size or self.config.train.per_device_prompt_batch_size)
        rollouts: list[RolloutRecord] = []
        for sample in prompts:
            for rollout_id in range(self.config.rollout.group_size):
                rollouts.append(self.engine.free_rollout(sample, rollout_id=rollout_id))
        return rollouts

    def reward_rollouts(self, rollouts: list[RolloutRecord]) -> tuple[list[RolloutRecord], list[dict[str, Any]]]:
        sample_by_id = {sample.sample_id: sample for sample in self.samples}
        rewarded: list[RolloutRecord] = []
        rewards: list[dict[str, Any]] = []
        for rollout in rollouts:
            sample = sample_by_id[rollout.sample_id]
            breakdown = score_rollout_reward(
                sample=sample,
                rollout=rollout,
                reward_config=self.config.reward,
                probe_cache=self.probe_cache,
                judge_bundle=self.judge_bundle,
                tau=self.config.probe.tau,
                missing_probe_policy=self.config.probe.missing_policy,
                hint_label_weight=self.config.probe.hint_label_weight,
                protocol=self.config.protocol,
                max_tool_calls=self.config.rollout.max_tool_calls,
            )
            payload = rollout.to_dict()
            payload["reward"] = breakdown.to_dict()
            rewarded.append(RolloutRecord(**{**rollout.to_dict(), "reward": breakdown.to_dict()}))
            rewards.append(breakdown.to_dict())
        return rewarded, rewards

    def fake_grpo_update(self, rollouts: list[RolloutRecord], rewards: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        reward_map = {
            (str(row["sample_id"]), int(row["rollout_id"])): float(row["reward_total"])
            for row in rewards
        }
        advantages = attach_group_advantages(
            rollouts,
            reward_map,
            eps=self.config.train.eps,
        )
        bias = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.AdamW([bias], lr=self.config.train.learning_rate)
        tensors = rollout_logprob_tensors(rollouts, advantages, trainable_bias=bias)
        loss, stats = grpo_loss_from_tensors(
            new_logprobs=tensors[0],
            old_logprobs=tensors[1],
            advantages=tensors[2],
            loss_mask=tensors[3],
            ref_logprobs=tensors[4],
            clip_range=self.config.train.clip_range,
            kl_coef=self.config.train.kl_coef,
        )
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_([bias], self.config.train.max_grad_norm))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return {
            **stats,
            "grad_norm": grad_norm,
            "trainable_bias_after_step": float(bias.detach().cpu()),
            "group_count": len({rollout.sample_id for rollout in rollouts}),
            "rollout_count": len(rollouts),
            "mean_reward": sum(reward_map.values()) / max(len(reward_map), 1),
        }

    def run_smoke_step(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rollouts = self.rollout_batch()
        rewarded, rewards = self.reward_rollouts(rollouts)
        write_jsonl(self.output_dir / "rollout_debug.jsonl", [item.to_dict() for item in rewarded])
        write_jsonl(self.output_dir / "reward_breakdown.jsonl", rewards)
        update = self.fake_grpo_update(rewarded, rewards)
        write_json(self.output_dir / "train_metrics.json", update)
        checkpoint_path = self.output_dir / "checkpoint_step_1.pt"
        _save_fake_checkpoint(checkpoint_path, self.config, update)
        result = {
            "status": "stage3_grpo_fake_smoke_completed",
            "output_dir": str(self.output_dir),
            "rollout_debug": str(self.output_dir / "rollout_debug.jsonl"),
            "reward_breakdown": str(self.output_dir / "reward_breakdown.jsonl"),
            "train_metrics": str(self.output_dir / "train_metrics.json"),
            "checkpoint": str(checkpoint_path),
            "metrics": update,
        }
        write_json(self.output_dir / "stage3_grpo_launch_result.json", result)
        self._log_success_to_wandb(result=result, rollouts=rewarded, rewards=rewards)
        return result

    def run_native_readiness_step(self) -> dict[str, Any]:
        if self.config.rollout.runtime_backend != "native_single_focus":
            raise ValueError("native readiness is only for runtime_backend=native_single_focus")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rollouts = self.rollout_batch()
        rewarded, rewards = self.reward_rollouts(rollouts)
        write_jsonl(self.output_dir / "rollout_debug.jsonl", [item.to_dict() for item in rewarded])
        write_jsonl(self.output_dir / "reward_breakdown.jsonl", rewards)
        reward_map = {
            (str(row["sample_id"]), int(row["rollout_id"])): float(row["reward_total"])
            for row in rewards
        }
        advantages = attach_group_advantages(rewarded, reward_map, eps=self.config.train.eps)
        readiness = native_grpo_readiness_report(rewarded, rewards, advantages)
        write_json(self.output_dir / "native_grpo_readiness.json", readiness)
        return readiness

    def run_native_train_step(self) -> dict[str, Any]:
        if self.config.rollout.runtime_backend != "native_single_focus":
            raise ValueError("native train step is only for runtime_backend=native_single_focus")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rollouts = self.rollout_batch()
        rewarded, rewards = self.reward_rollouts(rollouts)
        write_jsonl(self.output_dir / "rollout_debug.jsonl", [item.to_dict() for item in rewarded])
        write_jsonl(self.output_dir / "reward_breakdown.jsonl", rewards)
        update = self.native_grpo_update(rewarded, rewards)
        write_json(self.output_dir / "train_metrics.json", update)
        checkpoint_path = self.output_dir / "checkpoint_step_1.pt"
        _save_native_checkpoint(
            checkpoint_path,
            self.config,
            self.engine,
            update,
            optimizer_state=getattr(self, "_last_native_optimizer_state", None),
        )
        result = {
            "status": "stage3_grpo_native_step_completed",
            "output_dir": str(self.output_dir),
            "rollout_debug": str(self.output_dir / "rollout_debug.jsonl"),
            "reward_breakdown": str(self.output_dir / "reward_breakdown.jsonl"),
            "train_metrics": str(self.output_dir / "train_metrics.json"),
            "checkpoint": str(checkpoint_path),
            "metrics": update,
        }
        write_json(self.output_dir / "stage3_grpo_launch_result.json", result)
        self._log_success_to_wandb(result=result, rollouts=rewarded, rewards=rewards)
        return result

    def native_grpo_update(
        self,
        rollouts: list[RolloutRecord],
        rewards: list[dict[str, Any]],
    ) -> dict[str, Any]:
        import torch

        if not hasattr(self.engine, "replay_rollout_logprobs"):
            raise RuntimeError("native rollout engine does not expose replay_rollout_logprobs")
        native_runtime = getattr(self.engine, "_engine", None)
        if native_runtime is None or getattr(native_runtime, "model", None) is None:
            raise RuntimeError("native runtime is not loaded; run rollouts before native update")
        model = native_runtime.model
        foveal_module = getattr(native_runtime, "foveal_module", None)
        model.train()
        if foveal_module is not None:
            foveal_module.train()
        reward_map = {
            (str(row["sample_id"]), int(row["rollout_id"])): float(row["reward_total"])
            for row in rewards
        }
        advantages = attach_group_advantages(rollouts, reward_map, eps=self.config.train.eps)
        sample_by_id = {sample.sample_id: sample for sample in self.samples}
        new_rows = []
        old_rows = []
        ref_rows = []
        for rollout in rollouts:
            sample = sample_by_id[rollout.sample_id]
            new_logprobs = self.engine.replay_rollout_logprobs(  # type: ignore[attr-defined]
                sample,
                rollout,
                reference=False,
            )
            with torch.no_grad():
                ref_logprobs = self.engine.replay_rollout_logprobs(  # type: ignore[attr-defined]
                    sample,
                    rollout,
                    reference=True,
                ).detach()
            old_logprobs = torch.tensor(
                list(rollout.old_logprobs),
                dtype=torch.float32,
                device=new_logprobs.device,
            )
            if int(new_logprobs.numel()) != int(old_logprobs.numel()):
                raise ValueError(
                    f"replay length mismatch for {rollout.sample_id}/{rollout.rollout_id}: "
                    f"new={int(new_logprobs.numel())} old={int(old_logprobs.numel())}"
                )
            if int(ref_logprobs.numel()) != int(old_logprobs.numel()):
                raise ValueError(
                    f"reference replay length mismatch for {rollout.sample_id}/{rollout.rollout_id}: "
                    f"ref={int(ref_logprobs.numel())} old={int(old_logprobs.numel())}"
                )
            new_rows.append(new_logprobs)
            old_rows.append(old_logprobs)
            ref_rows.append(ref_logprobs)
        new_tensor, old_tensor, mask_tensor = _pad_logprob_rows(new_rows, old_rows)
        ref_tensor, _old_unused, _mask_unused = _pad_logprob_rows(ref_rows, old_rows)
        adv_tensor = torch.tensor(
            [float(advantages[(item.sample_id, item.rollout_id)]) for item in rollouts],
            dtype=torch.float32,
            device=new_tensor.device,
        )
        params = [param for param in model.parameters() if getattr(param, "requires_grad", False)]
        if foveal_module is not None:
            params.extend(
                param
                for param in foveal_module.parameters()
                if getattr(param, "requires_grad", False)
            )
        if not params:
            raise RuntimeError("native GRPO update found no trainable parameters")
        optimizer = torch.optim.AdamW(params, lr=self.config.train.learning_rate)
        loss, stats = grpo_loss_from_tensors(
            new_logprobs=new_tensor,
            old_logprobs=old_tensor,
            advantages=adv_tensor,
            loss_mask=mask_tensor,
            ref_logprobs=ref_tensor,
            clip_range=self.config.train.clip_range,
            kl_coef=self.config.train.kl_coef,
        )
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, self.config.train.max_grad_norm))
        optimizer.step()
        self._last_native_optimizer_state = optimizer.state_dict()
        optimizer.zero_grad(set_to_none=True)
        return {
            **stats,
            "status": "native_grpo_update_completed",
            "grad_norm": grad_norm,
            "group_count": len({rollout.sample_id for rollout in rollouts}),
            "rollout_count": len(rollouts),
            "mean_reward": sum(reward_map.values()) / max(len(reward_map), 1),
            "replayed_tokens": float(mask_tensor.sum().detach().cpu()),
        }

    def _log_success_to_wandb(
        self,
        *,
        result: dict[str, Any],
        rollouts: list[RolloutRecord],
        rewards: list[dict[str, Any]],
    ) -> None:
        logger = _create_stage3_wandb_logger(self.config, self.output_dir)
        if logger is None:
            return
        try:
            metrics = _stage3_wandb_metrics(result=result, rollouts=rollouts, rewards=rewards)
            logger.log(metrics, step=int(result.get("metrics", {}).get("optimizer_step", 1) or 1))
            logger.update_summary(_stage3_wandb_summary(self.config, result, rollouts, rewards))
            if self.config.wandb.log_artifacts:
                logger.log_artifact(
                    name=_safe_artifact_name(f"{self.config.run_id}-stage3-grpo-outputs"),
                    artifact_type="stage3_grpo_outputs",
                    paths=_stage3_output_artifact_paths(
                        self.output_dir,
                        include_checkpoint=False,
                    ),
                    aliases=["latest"],
                )
            checkpoint_path = result.get("checkpoint")
            if self.config.wandb.log_checkpoint_artifact and checkpoint_path:
                logger.log_artifact(
                    name=_safe_artifact_name(f"{self.config.run_id}-stage3-grpo-checkpoint"),
                    artifact_type="model",
                    paths=[str(checkpoint_path)],
                    aliases=["latest", "step_1"],
                )
        finally:
            logger.finish()


def _save_fake_checkpoint(path: Path, config: Stage3GRPOConfig, update: dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "stage3_grpo_checkpoint_v0",
            "run_id": config.run_id,
            "config": config.to_dict(),
            "global_step": 1,
            "optimizer_step": 1,
            "smoke_metrics": update,
            "note": "fake runtime checkpoint validates Stage3 GRPO plumbing only",
        },
        path,
    )


def _pad_logprob_rows(rows: list[Any], old_rows: list[Any]) -> tuple[Any, Any, Any]:
    import torch

    if not rows:
        raise ValueError("cannot pad empty logprob rows")
    max_len = max(int(row.numel()) for row in rows)
    device = rows[0].device
    new_padded = []
    old_padded = []
    mask_padded = []
    for new, old in zip(rows, old_rows, strict=True):
        pad = max_len - int(new.numel())
        new_padded.append(torch.cat([new.float(), torch.zeros(pad, device=device)]))
        old_padded.append(torch.cat([old.to(device).float(), torch.zeros(pad, device=device)]))
        mask_padded.append(
            torch.cat(
                [
                    torch.ones(int(new.numel()), dtype=torch.float32, device=device),
                    torch.zeros(pad, dtype=torch.float32, device=device),
                ]
            )
        )
    return torch.stack(new_padded), torch.stack(old_padded), torch.stack(mask_padded)


def _save_native_checkpoint(
    path: Path,
    config: Stage3GRPOConfig,
    engine: Any,
    update: dict[str, Any],
    *,
    optimizer_state: dict[str, Any] | None,
) -> None:
    import torch
    from peft import get_peft_model_state_dict

    native_runtime = getattr(engine, "_engine", None)
    if native_runtime is None or getattr(native_runtime, "model", None) is None:
        raise RuntimeError("native runtime is not loaded; cannot save checkpoint")
    source_checkpoint_config = (
        dict((getattr(native_runtime, "_checkpoint", None) or {}).get("config") or {})
    )
    if not source_checkpoint_config:
        raise RuntimeError("native runtime source checkpoint did not expose Stage2 config")
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": "stage3_grpo_native_checkpoint_v0",
        "run_id": config.run_id,
        "config": source_checkpoint_config,
        "stage3_config": config.to_dict(),
        "global_step": 1,
        "optimizer_step": 1,
        "qwen_lora": get_peft_model_state_dict(native_runtime.model),
        "tgvf_module": (
            native_runtime.foveal_module.state_dict()
            if getattr(native_runtime, "foveal_module", None) is not None
            else None
        ),
        "stage2_source_checkpoint": config.policy_checkpoint,
        "optimizer": optimizer_state,
        "scheduler": None,
        "smoke_metrics": update,
    }
    torch.save(checkpoint, path)


def native_grpo_readiness_report(
    rollouts: list[RolloutRecord],
    rewards: list[dict[str, Any]],
    advantages: dict[tuple[str, int], float],
) -> dict[str, Any]:
    replay_ready = []
    blockers: list[str] = []
    for rollout in rollouts:
        focus_ids = list((rollout.protocol or {}).get("focus_generated_ids") or [])
        continuation_ids = list((rollout.protocol or {}).get("continuation_generated_ids") or [])
        focus_logprobs = list((rollout.protocol or {}).get("focus_generated_logprobs") or [])
        continuation_logprobs = list(
            (rollout.protocol or {}).get("continuation_generated_logprobs") or []
        )
        has_segmented_ids = bool(focus_ids or continuation_ids)
        has_segmented_old_logprobs = (
            len(focus_ids) == len(focus_logprobs)
            and len(continuation_ids) == len(continuation_logprobs)
            and has_segmented_ids
        )
        has_advantage = (rollout.sample_id, rollout.rollout_id) in advantages
        item_ready = has_segmented_ids and has_segmented_old_logprobs and has_advantage
        replay_ready.append(item_ready)
        if not has_segmented_ids:
            blockers.append("missing_segmented_token_ids")
        if has_segmented_ids and not has_segmented_old_logprobs:
            blockers.append("missing_or_misaligned_segmented_old_logprobs")
        if not has_advantage:
            blockers.append("missing_group_advantage")
    all_segmented = bool(replay_ready) and all(replay_ready)
    return {
        "schema_version": "stage3_grpo_native_readiness_v0",
        "status": "ready" if all_segmented else "blocked",
        "rollout_count": len(rollouts),
        "reward_count": len(rewards),
        "rollouts_with_segmented_replay_inputs": sum(1 for item in replay_ready if item),
        "all_rollouts_have_segmented_replay_inputs": all_segmented,
        "unique_blockers": sorted(set(blockers)),
        "next_required_implementation": [
            "run native launch-training smoke against the actual Stage2 checkpoint",
            "inspect rollout/reward/train_metrics outputs before scaling",
        ],
    }


def _create_stage3_wandb_logger(config: Stage3GRPOConfig, output_dir: Path) -> Any | None:
    wandb_config = config.wandb
    if not wandb_config.project or wandb_config.mode == "disabled":
        return None
    from revisit_vlm.wandb_logging import WandbLogger

    return WandbLogger(
        project=wandb_config.project,
        entity=wandb_config.entity,
        name=wandb_config.name or config.run_id,
        group=wandb_config.group or "stage3_grpo",
        job_type=wandb_config.job_type,
        mode=wandb_config.mode,
        config=_stage3_wandb_config(config, output_dir),
        directory=output_dir / "wandb",
        tags=[
            "stage3-grpo",
            str(config.rollout.runtime_backend),
            str(config.protocol),
            *list(wandb_config.tags),
        ],
    )


def _stage3_wandb_config(config: Stage3GRPOConfig, output_dir: Path) -> dict[str, Any]:
    return {
        "stage": "stage3_grpo",
        "run_id": config.run_id,
        "protocol": config.protocol,
        "model_id": config.model_id,
        "processor_id": config.processor_id,
        "runtime_backend": config.rollout.runtime_backend,
        "output_dir": str(output_dir),
        "stage3_config": config.to_dict(),
        "rl_data": dataset_identity(config.rl_data_path),
        "policy_checkpoint": file_identity(config.policy_checkpoint).to_dict(),
    }


def _stage3_wandb_metrics(
    *,
    result: dict[str, Any],
    rollouts: list[RolloutRecord],
    rewards: list[dict[str, Any]],
) -> dict[str, Any]:
    update = dict(result.get("metrics") or {})
    metrics: dict[str, Any] = {
        "trainer/global_step": 1,
        "trainer/rollout_count": len(rollouts),
        "trainer/reward_count": len(rewards),
        "rollout/tool_trigger_rate": _mean_float(
            [1.0 if item.used_tool else 0.0 for item in rollouts]
        ),
        "rollout/avg_tool_calls": _mean_float([float(item.num_tool_calls) for item in rollouts]),
        "rollout/malformed_rate": _mean_float(
            [1.0 if (item.protocol or {}).get("native_errors") else 0.0 for item in rollouts]
        ),
    }
    for key, value in update.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            metrics[f"train/{key}"] = float(value)
    for key in (
        "reward_total",
        "reward_answer",
        "reward_tool",
        "reward_focus",
        "reward_ground",
        "reward_protocol",
    ):
        metrics[f"reward/{key}_mean"] = _mean_float(
            [float(row[key]) for row in rewards if isinstance(row.get(key), (int, float))]
        )
    metrics["reward/answer_accuracy"] = _mean_float(
        [1.0 if row.get("answer_correct") else 0.0 for row in rewards]
    )
    for label, count in _count_by_key(rewards, "tool_label").items():
        metrics[f"reward/tool_label_count/{label}"] = count
    return {key: value for key, value in metrics.items() if value is not None}


def _stage3_wandb_summary(
    config: Stage3GRPOConfig,
    result: dict[str, Any],
    rollouts: list[RolloutRecord],
    rewards: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "status": result.get("status"),
        "run_id": config.run_id,
        "runtime_backend": config.rollout.runtime_backend,
        "policy_checkpoint": config.policy_checkpoint,
        "rl_data_path": config.rl_data_path,
        "checkpoint": result.get("checkpoint"),
        "rollout_count": len(rollouts),
        "reward_count": len(rewards),
        "metrics": dict(result.get("metrics") or {}),
        "reward_summary": {
            "answer_accuracy": _mean_float(
                [1.0 if row.get("answer_correct") else 0.0 for row in rewards]
            ),
            "mean_total": _mean_float(
                [
                    float(row["reward_total"])
                    for row in rewards
                    if isinstance(row.get("reward_total"), (int, float))
                ]
            ),
            "tool_label_counts": _count_by_key(rewards, "tool_label"),
        },
    }


def _stage3_output_artifact_paths(
    output_dir: Path,
    *,
    include_checkpoint: bool,
) -> list[str | Path]:
    names = [
        "stage3_grpo_training_plan.json",
        "stage3_grpo_training_plan.txt",
        "stage3_grpo_dataset_identity.json",
        "stage3_grpo_preflight_report.json",
        "stage3_grpo_launch_result.json",
        "train_metrics.json",
        "rollout_debug.jsonl",
        "reward_breakdown.jsonl",
        "judge_pending.jsonl",
        "probe_cache.jsonl",
        "probe_cache_summary.json",
    ]
    paths: list[str | Path] = [output_dir / name for name in names]
    if include_checkpoint:
        paths.append(output_dir / "checkpoint_step_1.pt")
    return paths


def _mean_float(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


def _count_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _safe_artifact_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in value)
    return safe.strip("-") or "stage3-grpo-artifact"
