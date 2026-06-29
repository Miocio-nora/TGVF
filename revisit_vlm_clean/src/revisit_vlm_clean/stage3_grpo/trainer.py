"""Small Stage3 GRPO trainer core."""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

from revisit_vlm_clean.data_generation import file_identity

from .data import BalancedPromptSampler, load_stage3_sample_schedule, load_stage3_samples, write_jsonl
from .data import dataset_identity
from .grpo import attach_group_advantages, grpo_loss_from_tensors, rollout_logprob_tensors
from .judge import JudgeBundle
from .probe import ProbeCache
from .reward import score_rollout_reward
from .rollout import build_rollout_engine
from .schemas import RolloutRecord, Stage3GRPOConfig, Stage3Sample, now_iso, write_json


class Stage3GRPOTrainer:
    def __init__(self, config: Stage3GRPOConfig) -> None:
        config.validate()
        self.distributed = _setup_stage3_distributed_context(config)
        self.config = _config_with_distributed_device(config, self.distributed)
        self.output_dir = Path(config.output_dir)
        self.rank_output_dir = (
            self.output_dir
            if self.distributed["is_main"]
            else self.output_dir / f"rank_{self.distributed['rank']}"
        )
        self.samples = load_stage3_samples(self.config.rl_data_path)
        self.sample_by_id = {sample.sample_id: sample for sample in self.samples}
        self.sample_schedule = (
            load_stage3_sample_schedule(self.config.sample_schedule_path)
            if self.config.sample_schedule_path
            else None
        )
        _seed_stage3_runtime(
            int(self.config.train.seed) + int(self.distributed["rank"]) * 100003
        )
        self.sampler = BalancedPromptSampler(
            self.samples,
            seed=int(self.config.train.seed) + int(self.distributed["rank"]) * 100003,
        )
        self.probe_cache = ProbeCache(self.config.probe.cache_path)
        self.judge_bundle = JudgeBundle(self.config.judge, output_dir=self.rank_output_dir)
        self.engine = build_rollout_engine(self.config)
        self.progress_path = self.rank_output_dir / "progress.jsonl"
        self._wandb_logger = None

    def rollout_batch(
        self,
        *,
        prompt_batch_size: int | None = None,
        global_step: int | None = None,
        accumulation_index: int = 0,
    ) -> list[RolloutRecord]:
        prompts, schedule_by_sample_id = self._scheduled_prompts(
            prompt_batch_size=prompt_batch_size,
            global_step=global_step,
            accumulation_index=accumulation_index,
        )
        if global_step is not None:
            self._progress(
                "rollout_batch_selected",
                global_step=global_step,
                accumulation_index=accumulation_index,
                prompt_count=len(prompts),
                sample_ids=[sample.sample_id for sample in prompts],
            )
        rollouts: list[RolloutRecord] = []
        for sample in prompts:
            for rollout_id in range(self.config.rollout.group_size):
                if global_step is not None:
                    self._progress(
                        "free_rollout_start",
                        global_step=global_step,
                        accumulation_index=accumulation_index,
                        sample_id=sample.sample_id,
                        rollout_id=rollout_id,
                    )
                try:
                    rollout = self.engine.free_rollout(sample, rollout_id=rollout_id)
                except BaseException as exc:
                    if global_step is not None:
                        self._progress(
                            "free_rollout_failed",
                            global_step=global_step,
                            accumulation_index=accumulation_index,
                            sample_id=sample.sample_id,
                            rollout_id=rollout_id,
                            **_stage3_exception_payload(exc),
                        )
                    raise
                if global_step is not None:
                    self._progress(
                        "free_rollout_done",
                        global_step=global_step,
                        accumulation_index=accumulation_index,
                        sample_id=sample.sample_id,
                        rollout_id=rollout_id,
                        used_tool=bool(rollout.used_tool),
                        token_count=len(rollout.token_ids or ()),
                        old_logprob_count=len(rollout.old_logprobs or ()),
                    )
                schedule_row = schedule_by_sample_id.get(sample.sample_id)
                if schedule_row:
                    runtime = dict(rollout.runtime or {})
                    runtime["sample_schedule"] = dict(schedule_row)
                    rollouts.append(RolloutRecord(**{**rollout.to_dict(), "runtime": runtime}))
                else:
                    rollouts.append(rollout)
        return rollouts

    def _scheduled_prompts(
        self,
        *,
        prompt_batch_size: int | None,
        global_step: int | None,
        accumulation_index: int,
    ) -> tuple[list[Stage3Sample], dict[str, dict[str, Any]]]:
        expected_batch_size = int(prompt_batch_size or self.config.train.per_device_prompt_batch_size)
        if not self.sample_schedule:
            return self.sampler.next_batch(expected_batch_size), {}
        schedule_step = int(self.config.sample_schedule_start_step) + int(global_step or 1) - 1
        rank = int(self.distributed["rank"])
        rows = [
            row
            for row in self.sample_schedule
            if int(row["global_step"]) == schedule_step
            and int(row["rank"]) == rank
            and int(row.get("accumulation_index", 0)) == int(accumulation_index)
        ]
        rows = sorted(rows, key=lambda row: int(row.get("prompt_index", 0)))
        if len(rows) != expected_batch_size:
            raise ValueError(
                "sample schedule batch size mismatch: "
                f"step={schedule_step} rank={rank} accumulation={accumulation_index} "
                f"expected={expected_batch_size} found={len(rows)}"
            )
        prompts: list[Stage3Sample] = []
        schedule_by_sample_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            sample_id = str(row["sample_id"])
            sample = self.sample_by_id.get(sample_id)
            if sample is None:
                raise ValueError(f"sample schedule references unknown sample_id: {sample_id}")
            prompts.append(sample)
            schedule_by_sample_id[sample_id] = dict(row)
        return prompts, schedule_by_sample_id

    def reward_rollouts(self, rollouts: list[RolloutRecord]) -> tuple[list[RolloutRecord], list[dict[str, Any]]]:
        rewarded: list[RolloutRecord] = []
        rewards: list[dict[str, Any]] = []
        for rollout in rollouts:
            sample = self.sample_by_id[rollout.sample_id]
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

    def fake_grpo_update(
        self,
        rollouts: list[RolloutRecord],
        rewards: list[dict[str, Any]],
        *,
        global_step: int,
    ) -> dict[str, Any]:
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
        if not hasattr(self, "_fake_bias"):
            self._fake_bias = torch.nn.Parameter(torch.zeros(()))
            self._fake_optimizer = torch.optim.AdamW(
                [self._fake_bias],
                lr=self.config.train.learning_rate,
            )
        bias = self._fake_bias
        optimizer = self._fake_optimizer
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
            "global_step": int(global_step),
            "optimizer_step": int(global_step),
            "grad_norm": grad_norm,
            "trainable_bias_after_step": float(bias.detach().cpu()),
            "group_count": len({rollout.sample_id for rollout in rollouts}),
            "rollout_count": len(rollouts),
            "mean_reward": sum(reward_map.values()) / max(len(reward_map), 1),
            "gradient_accumulation_steps": int(self.config.train.gradient_accumulation_steps),
        }

    def run_smoke_step(self) -> dict[str, Any]:
        return self.run_training()

    def run_native_readiness_step(self) -> dict[str, Any]:
        if self.config.rollout.runtime_backend != "native_single_focus":
            raise ValueError("native readiness is only for runtime_backend=native_single_focus")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rollouts = self.rollout_batch(global_step=1, accumulation_index=0)
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
        return self.run_training()

    def run_training(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rank_output_dir.mkdir(parents=True, exist_ok=True)
        rollout_path = self.rank_output_dir / "rollout_debug.jsonl"
        reward_path = self.rank_output_dir / "reward_breakdown.jsonl"
        metrics_path = self.rank_output_dir / "train_metrics.jsonl"
        for path in (rollout_path, reward_path, metrics_path, self.progress_path):
            path.write_text("", encoding="utf-8")
        logger = (
            _create_stage3_wandb_logger(self.config, self.output_dir)
            if self.distributed["is_main"]
            else None
        )
        step_summaries: list[dict[str, Any]] = []
        latest_checkpoint: str | None = None
        latest_metrics: dict[str, Any] = {}
        aggregate = _Stage3Aggregate()
        completed_successfully = False
        try:
            for global_step in range(1, int(self.config.train.max_steps) + 1):
                self._progress("step_start", global_step=global_step)
                rewarded, rewards = self._rollout_and_reward_training_step(global_step)
                self._progress(
                    "rollout_reward_done",
                    global_step=global_step,
                    rollout_count=len(rewarded),
                    reward_count=len(rewards),
                )
                _append_jsonl(
                    rollout_path,
                    [
                        {
                            **item.to_dict(),
                            "global_step": global_step,
                        }
                        for item in rewarded
                    ],
                )
                _append_jsonl(
                    reward_path,
                    [
                        {
                            **row,
                            "global_step": global_step,
                        }
                        for row in rewards
                    ],
                )
                if self.config.rollout.runtime_backend == "native_single_focus":
                    self._progress("update_start", global_step=global_step)
                    update = self.native_grpo_update(
                        rewarded,
                        rewards,
                        global_step=global_step,
                    )
                else:
                    self._progress("update_start", global_step=global_step)
                    update = self.fake_grpo_update(
                        rewarded,
                        rewards,
                        global_step=global_step,
                    )
                self._progress("update_done", global_step=global_step, metrics=dict(update))
                latest_metrics = dict(update)
                _append_jsonl(metrics_path, [latest_metrics])
                write_json(self.output_dir / "train_metrics.json", latest_metrics)
                aggregate.update(rollouts=rewarded, rewards=rewards, metrics=latest_metrics)
                checkpoint_path = None
                if (
                    self.distributed["is_main"]
                    and (
                        global_step % int(self.config.train.save_steps) == 0
                        or global_step == int(self.config.train.max_steps)
                    )
                ):
                    checkpoint_path = self._save_checkpoint(global_step, latest_metrics)
                    latest_checkpoint = str(checkpoint_path)
                    (self.output_dir / "LATEST_CHECKPOINT.txt").write_text(
                        latest_checkpoint + "\n",
                        encoding="utf-8",
                    )
                step_summary = {
                    "global_step": global_step,
                    "rollout_count": len(rewarded),
                    "reward_count": len(rewards),
                    "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
                    "metrics": latest_metrics,
                }
                step_summaries.append(step_summary)
                if logger is not None:
                    logger.log(
                        _stage3_wandb_metrics(
                            result={"metrics": latest_metrics},
                            rollouts=rewarded,
                            rewards=rewards,
                        ),
                        step=global_step,
                    )
                self._progress("step_done", global_step=global_step)
            result = {
                "status": "stage3_grpo_training_completed",
                "output_dir": str(self.output_dir),
                "rank_output_dir": str(self.rank_output_dir),
                "distributed": dict(self.distributed),
                "runtime_backend": self.config.rollout.runtime_backend,
                "global_step": int(self.config.train.max_steps),
                "optimizer_step": int(self.config.train.max_steps),
                "max_steps": int(self.config.train.max_steps),
                "rollout_debug": str(rollout_path),
                "reward_breakdown": str(reward_path),
                "train_metrics": str(self.output_dir / "train_metrics.json"),
                "train_metrics_jsonl": str(metrics_path),
                "checkpoint": latest_checkpoint,
                "step_summaries": step_summaries,
                "metrics": latest_metrics,
                "aggregate": aggregate.summary(),
            }
            if self.distributed["is_main"]:
                write_json(self.output_dir / "stage3_grpo_launch_result.json", result)
            else:
                write_json(self.rank_output_dir / "stage3_grpo_launch_result.json", result)
            if logger is not None:
                logger.update_summary(
                    _stage3_wandb_summary_from_aggregate(self.config, result)
                )
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
                if self.config.wandb.log_checkpoint_artifact and latest_checkpoint:
                    logger.log_artifact(
                        name=_safe_artifact_name(f"{self.config.run_id}-stage3-grpo-checkpoint"),
                        artifact_type="model",
                        paths=[latest_checkpoint],
                        aliases=["latest", f"step_{int(self.config.train.max_steps)}"],
                    )
            completed_successfully = True
            return result
        except BaseException as exc:
            self._progress("run_training_failed", **_stage3_exception_payload(exc))
            raise
        finally:
            if logger is not None:
                logger.finish()
            if completed_successfully:
                _stage3_distributed_barrier(self.distributed)
            _destroy_stage3_distributed_context(self.distributed)

    def _rollout_and_reward_training_step(
        self,
        global_step: int,
    ) -> tuple[list[RolloutRecord], list[dict[str, Any]]]:
        all_rollouts: list[RolloutRecord] = []
        all_rewards: list[dict[str, Any]] = []
        for accumulation_index in range(int(self.config.train.gradient_accumulation_steps)):
            rollouts = self.rollout_batch(
                global_step=global_step,
                accumulation_index=accumulation_index,
            )
            rewarded, rewards = self.reward_rollouts(rollouts)
            rollout_id_map = {
                (item.sample_id, item.rollout_id): (
                    accumulation_index * int(self.config.rollout.group_size) + item.rollout_id
                )
                for item in rewarded
            }
            for rollout in rewarded:
                payload = dict(rollout.runtime or {})
                payload["global_step"] = global_step
                payload["accumulation_index"] = accumulation_index
                mapped_rollout_id = rollout_id_map[(rollout.sample_id, rollout.rollout_id)]
                all_rollouts.append(
                    RolloutRecord(
                        **{
                            **rollout.to_dict(),
                            "rollout_id": mapped_rollout_id,
                            "runtime": payload,
                        }
                    )
                )
            for row in rewards:
                mapped_rollout_id = rollout_id_map[(str(row["sample_id"]), int(row["rollout_id"]))]
                all_rewards.append(
                    {
                        **row,
                        "rollout_id": mapped_rollout_id,
                        "accumulation_index": accumulation_index,
                    }
                )
        return all_rollouts, all_rewards

    def _save_checkpoint(self, global_step: int, update: dict[str, Any]) -> Path:
        checkpoint_path = self.output_dir / f"checkpoint_step_{int(global_step)}.pt"
        if self.config.rollout.runtime_backend == "native_single_focus":
            optimizer_state = (
                getattr(self, "_native_optimizer", None).state_dict()
                if hasattr(self, "_native_optimizer")
                else None
            )
            _save_native_checkpoint(
                checkpoint_path,
                self.config,
                self.engine,
                update,
                global_step=global_step,
                optimizer_state=optimizer_state,
            )
        else:
            _save_fake_checkpoint(
                checkpoint_path,
                self.config,
                update,
                global_step=global_step,
                optimizer_state=(
                    getattr(self, "_fake_optimizer", None).state_dict()
                    if hasattr(self, "_fake_optimizer")
                    else None
                ),
            )
        return checkpoint_path

    def _progress(self, event: str, **payload: Any) -> None:
        _append_jsonl(
            self.progress_path,
            [
                {
                    "created_at": now_iso(),
                    "event": event,
                    "rank": int(self.distributed["rank"]),
                    "local_rank": int(self.distributed["local_rank"]),
                    "world_size": int(self.distributed["world_size"]),
                    **payload,
                }
            ],
        )

    def native_grpo_update(
        self,
        rollouts: list[RolloutRecord],
        rewards: list[dict[str, Any]],
        *,
        global_step: int,
    ) -> dict[str, Any]:
        import torch

        if not hasattr(self.engine, "replay_rollout_logprobs"):
            raise RuntimeError("native rollout engine does not expose replay_rollout_logprobs")
        native_runtime = getattr(self.engine, "_engine", None)
        if native_runtime is None or getattr(native_runtime, "model", None) is None:
            raise RuntimeError("native runtime is not loaded; run rollouts before native update")
        model = native_runtime.model
        foveal_module = getattr(native_runtime, "foveal_module", None)
        trainable_summary = _stage3_configure_native_trainables(model, foveal_module)
        self._progress(
            "trainable_parameters",
            global_step=global_step,
            summary=trainable_summary,
        )
        model.train()
        if foveal_module is not None:
            foveal_module.eval()
        reward_map = {
            (str(row["sample_id"]), int(row["rollout_id"])): float(row["reward_total"])
            for row in rewards
        }
        advantages = attach_group_advantages(rollouts, reward_map, eps=self.config.train.eps)
        sample_by_id = {sample.sample_id: sample for sample in self.samples}
        new_rows = []
        old_rows = []
        ref_rows = []
        recorded_old_logprob_mismatches = 0
        for rollout_index, rollout in enumerate(rollouts):
            sample = sample_by_id[rollout.sample_id]
            self._progress(
                "replay_policy_start",
                global_step=global_step,
                rollout_index=rollout_index,
                sample_id=rollout.sample_id,
                rollout_id=rollout.rollout_id,
            )
            new_logprobs = self.engine.replay_rollout_logprobs(  # type: ignore[attr-defined]
                sample,
                rollout,
                reference=False,
            )
            self._progress(
                "replay_policy_done",
                global_step=global_step,
                rollout_index=rollout_index,
                sample_id=rollout.sample_id,
                rollout_id=rollout.rollout_id,
                token_count=int(new_logprobs.numel()),
            )
            replay_token_count = _stage3_rollout_replay_token_count(rollout)
            if int(new_logprobs.numel()) != replay_token_count:
                raise ValueError(
                    f"replay length mismatch for {rollout.sample_id}/{rollout.rollout_id}: "
                    f"new={int(new_logprobs.numel())} replay_tokens={replay_token_count}"
                )
            recorded_old_count = len(rollout.old_logprobs or ())
            if recorded_old_count not in {0, int(new_logprobs.numel())}:
                recorded_old_logprob_mismatches += 1
            old_logprobs = new_logprobs.detach().float()
            ref_logprobs = old_logprobs.clone()
            self._progress(
                "reference_logprobs_reused_from_replay_policy",
                global_step=global_step,
                rollout_index=rollout_index,
                sample_id=rollout.sample_id,
                rollout_id=rollout.rollout_id,
                token_count=int(ref_logprobs.numel()),
                recorded_old_logprob_count=recorded_old_count,
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
        if not params:
            raise RuntimeError("native GRPO update found no trainable parameters")
        optimizer = None
        if self.config.train.optimizer == "adamw":
            if not hasattr(self, "_native_optimizer"):
                self._native_optimizer = _stage3_native_adamw(
                    params,
                    lr=self.config.train.learning_rate,
                )
            optimizer = self._native_optimizer
        loss, stats = grpo_loss_from_tensors(
            new_logprobs=new_tensor,
            old_logprobs=old_tensor,
            advantages=adv_tensor,
            loss_mask=mask_tensor,
            ref_logprobs=ref_tensor,
            clip_range=self.config.train.clip_range,
            kl_coef=self.config.train.kl_coef,
        )
        self._progress(
            "backward_start",
            global_step=global_step,
            rollout_count=len(rollouts),
            token_count=float(mask_tensor.sum().detach().cpu()),
        )
        loss.backward()
        self._progress("backward_done", global_step=global_step)
        self._progress("gradient_allreduce_start", global_step=global_step)
        _stage3_distributed_average_gradients(params, self.distributed)
        self._progress("gradient_allreduce_done", global_step=global_step)
        self._progress("grad_clip_start", global_step=global_step)
        grad_norm = _stage3_clip_grad_norm(params, float(self.config.train.max_grad_norm))
        self._progress("grad_clip_done", global_step=global_step, grad_norm=grad_norm)
        self._progress(
            "optimizer_step_start",
            global_step=global_step,
            grad_norm=grad_norm,
            optimizer=self.config.train.optimizer,
        )
        if self.config.train.optimizer == "manual_sgd":
            _stage3_manual_sgd_step(params, lr=float(self.config.train.learning_rate))
            self._progress("optimizer_step_torch_done", global_step=global_step)
            _stage3_zero_grad(params)
        elif optimizer is not None:
            optimizer.step()
            self._progress("optimizer_step_torch_done", global_step=global_step)
            optimizer.zero_grad(set_to_none=True)
        else:
            raise RuntimeError(f"unsupported Stage3 optimizer: {self.config.train.optimizer}")
        update = {
            **stats,
            "status": "native_grpo_update_completed",
            "optimizer": self.config.train.optimizer,
            "global_step": int(global_step),
            "optimizer_step": int(global_step),
            "grad_norm": grad_norm,
            "group_count": len({rollout.sample_id for rollout in rollouts}),
            "rollout_count": len(rollouts),
            "mean_reward": sum(reward_map.values()) / max(len(reward_map), 1),
            "replayed_tokens": float(mask_tensor.sum().detach().cpu()),
            "behavior_logprobs_source": "teacher_forced_policy_replay_detached",
            "reference_logprobs_source": "teacher_forced_policy_replay_detached",
            "recorded_old_logprob_mismatches": int(recorded_old_logprob_mismatches),
            "gradient_accumulation_steps": int(self.config.train.gradient_accumulation_steps),
            "rank": int(self.distributed["rank"]),
            "world_size": int(self.distributed["world_size"]),
        }
        update.update(_stage3_distributed_metric_summary(update, self.distributed))
        self._progress("optimizer_step_done", global_step=global_step, metrics=dict(update))
        _stage3_distributed_barrier(self.distributed)
        return update


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _setup_stage3_distributed_context(config: Stage3GRPOConfig) -> dict[str, Any]:
    planned_world_size = int(config.train.world_size)
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", "0"))
    env_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if planned_world_size > 1 and env_world_size != planned_world_size:
        raise ValueError(
            "Stage3 distributed launch requires torchrun WORLD_SIZE to match "
            f"train.world_size ({env_world_size} != {planned_world_size})"
        )
    if planned_world_size == 1 and env_world_size > 1:
        raise ValueError("Stage3 single-process launch cannot run under WORLD_SIZE>1")
    context: dict[str, Any] = {
        "distributed": planned_world_size > 1,
        "rank": env_rank if planned_world_size > 1 else 0,
        "local_rank": env_local_rank if planned_world_size > 1 else 0,
        "world_size": planned_world_size,
        "is_main": (env_rank if planned_world_size > 1 else 0) == 0,
        "device": None,
        "device_map": None,
        "backend": None,
        "process_group_initialized_by_stage3": False,
    }
    if planned_world_size <= 1:
        return context
    import torch
    import torch.distributed as dist

    if torch.cuda.is_available():
        torch.cuda.set_device(env_local_rank)
        context["device"] = f"cuda:{env_local_rank}"
        context["device_map"] = f"cuda:{env_local_rank}"
        context["backend"] = "nccl"
    else:
        context["device"] = "cpu"
        context["device_map"] = None
        context["backend"] = "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=str(context["backend"]))
        context["process_group_initialized_by_stage3"] = True
    return context


def _config_with_distributed_device(
    config: Stage3GRPOConfig,
    context: dict[str, Any],
) -> Stage3GRPOConfig:
    if not context.get("distributed"):
        return config
    return replace(
        config,
        device=str(context["device"]),
        device_map=str(context["device_map"]) if context.get("device_map") else None,
    )


def _stage3_distributed_average_gradients(
    params: list[Any],
    context: dict[str, Any],
) -> None:
    if not context.get("distributed"):
        return
    import torch
    import torch.distributed as dist

    world_size = float(context["world_size"])
    for param in params:
        grad = getattr(param, "grad", None)
        if grad is None:
            grad = torch.zeros_like(param.detach(), memory_format=torch.preserve_format)
            param.grad = grad
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        grad.div_(world_size)


def _stage3_exception_payload(exc: BaseException) -> dict[str, Any]:
    return {
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def _stage3_distributed_metric_summary(
    update: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    if not context.get("distributed"):
        return {}
    import torch
    import torch.distributed as dist

    device = torch.device(str(context["device"] or "cpu"))
    values = torch.tensor(
        [
            float(update.get("loss") or 0.0),
            float(update.get("policy_loss") or 0.0),
            float(update.get("kl") or 0.0),
            float(update.get("clip_fraction") or 0.0),
            float(update.get("grad_norm") or 0.0),
            float(update.get("mean_reward") or 0.0),
            float(update.get("rollout_count") or 0.0),
            float(update.get("group_count") or 0.0),
            float(update.get("replayed_tokens") or 0.0),
            float(update.get("token_count") or 0.0),
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    world_size = float(context["world_size"])
    return {
        "distributed_loss_mean": float(values[0].item() / world_size),
        "distributed_policy_loss_mean": float(values[1].item() / world_size),
        "distributed_kl_mean": float(values[2].item() / world_size),
        "distributed_clip_fraction_mean": float(values[3].item() / world_size),
        "distributed_grad_norm_mean": float(values[4].item() / world_size),
        "distributed_mean_reward": float(values[5].item() / world_size),
        "distributed_rollout_count": int(values[6].item()),
        "distributed_group_count": int(values[7].item()),
        "distributed_replayed_tokens": float(values[8].item()),
        "distributed_token_count": float(values[9].item()),
    }


def _stage3_clip_grad_norm(params: list[Any], max_norm: float) -> float:
    import torch

    if max_norm <= 0:
        return 0.0
    grads = [param.grad for param in params if getattr(param, "grad", None) is not None]
    if not grads:
        return 0.0
    device = grads[0].device
    norms = torch.stack([grad.detach().float().norm(2).to(device) for grad in grads])
    total_norm = torch.linalg.vector_norm(norms, ord=2)
    if max_norm > 0:
        clip_coef = torch.tensor(float(max_norm), device=device) / (total_norm + 1e-6)
        if float(clip_coef.detach().cpu()) < 1.0:
            for grad in grads:
                grad.mul_(clip_coef.to(grad.device, dtype=grad.dtype))
    return float(total_norm.detach().cpu())


def _stage3_native_adamw(params: list[Any], *, lr: float) -> Any:
    import torch

    return torch.optim.AdamW(params, lr=lr, foreach=False, fused=False)


def _stage3_configure_native_trainables(model: Any, foveal_module: Any | None) -> dict[str, Any]:
    policy_summary = _stage3_restrict_policy_trainables(model)
    foveal_summary = _stage3_freeze_module(foveal_module) if foveal_module is not None else {}
    return {
        "policy": policy_summary,
        "foveal_module": foveal_summary,
        "total_trainable_parameters": int(policy_summary.get("trainable_parameters", 0)),
    }


def _stage3_restrict_policy_trainables(model: Any) -> dict[str, Any]:
    markers = ("lora_", "modules_to_save", "trainable_tokens", "token_adapter")
    total_parameters = 0
    trainable_parameters = 0
    trainable_tensors = 0
    sample_names: list[str] = []
    for name, param in model.named_parameters():
        count = int(param.numel())
        total_parameters += count
        trainable = any(marker in name for marker in markers)
        param.requires_grad_(trainable)
        if trainable:
            trainable_parameters += count
            trainable_tensors += 1
            if len(sample_names) < 8:
                sample_names.append(name)
    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_tensors": trainable_tensors,
        "trainable_name_samples": sample_names,
        "trainable_markers": list(markers),
    }


def _stage3_freeze_module(module: Any) -> dict[str, Any]:
    total_parameters = 0
    for param in module.parameters():
        total_parameters += int(param.numel())
        param.requires_grad_(False)
    return {
        "total_parameters": total_parameters,
        "trainable_parameters": 0,
        "trainable_tensors": 0,
        "frozen": True,
    }


def _stage3_manual_sgd_step(params: list[Any], *, lr: float) -> None:
    import torch

    with torch.no_grad():
        for param in params:
            grad = getattr(param, "grad", None)
            if grad is None:
                continue
            param.add_(grad.to(device=param.device, dtype=param.dtype), alpha=-lr)


def _stage3_zero_grad(params: list[Any]) -> None:
    for param in params:
        param.grad = None


def _stage3_distributed_barrier(context: dict[str, Any]) -> None:
    if not context.get("distributed"):
        return
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _destroy_stage3_distributed_context(context: dict[str, Any]) -> None:
    if not context.get("process_group_initialized_by_stage3"):
        return
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _save_fake_checkpoint(
    path: Path,
    config: Stage3GRPOConfig,
    update: dict[str, Any],
    *,
    global_step: int,
    optimizer_state: dict[str, Any] | None,
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "stage3_grpo_checkpoint_v0",
            "run_id": config.run_id,
            "config": config.to_dict(),
            "global_step": int(global_step),
            "optimizer_step": int(global_step),
            "smoke_metrics": update,
            "optimizer": optimizer_state,
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


def _stage3_rollout_replay_token_count(rollout: RolloutRecord) -> int:
    protocol = rollout.protocol or {}
    focus_ids = protocol.get("focus_generated_ids") or []
    continuation_ids = protocol.get("continuation_generated_ids") or []
    segmented_count = len(focus_ids) + len(continuation_ids)
    if segmented_count > 0:
        return int(segmented_count)
    return len(rollout.token_ids or ())


def _save_native_checkpoint(
    path: Path,
    config: Stage3GRPOConfig,
    engine: Any,
    update: dict[str, Any],
    *,
    global_step: int,
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
        "global_step": int(global_step),
        "optimizer_step": int(global_step),
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
    recorded_old_logprobs_aligned = []
    blockers: list[str] = []
    for rollout in rollouts:
        focus_ids = list((rollout.protocol or {}).get("focus_generated_ids") or [])
        continuation_ids = list((rollout.protocol or {}).get("continuation_generated_ids") or [])
        focus_logprobs = list((rollout.protocol or {}).get("focus_generated_logprobs") or [])
        continuation_logprobs = list(
            (rollout.protocol or {}).get("continuation_generated_logprobs") or []
        )
        has_segmented_ids = bool(focus_ids or continuation_ids)
        recorded_old_logprobs_aligned.append(
            has_segmented_ids
            and len(focus_ids) == len(focus_logprobs)
            and len(continuation_ids) == len(continuation_logprobs)
        )
        has_advantage = (rollout.sample_id, rollout.rollout_id) in advantages
        item_ready = has_segmented_ids and has_advantage
        replay_ready.append(item_ready)
        if not has_segmented_ids:
            blockers.append("missing_segmented_token_ids")
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
        "recorded_old_logprobs_are_diagnostic_only": True,
        "rollouts_with_aligned_recorded_old_logprobs": sum(
            1 for item in recorded_old_logprobs_aligned if item
        ),
        "unique_blockers": sorted(set(blockers)),
        "next_required_implementation": [
            "run native launch-training smoke against the actual Stage2 checkpoint",
            "inspect rollout/reward/train_metrics outputs before scaling",
        ],
    }


class _Stage3Aggregate:
    def __init__(self) -> None:
        self.step_count = 0
        self.rollout_count = 0
        self.reward_count = 0
        self.used_tool_count = 0
        self.tool_call_count = 0
        self.malformed_count = 0
        self.answer_correct_count = 0
        self.reward_sums: dict[str, float] = {}
        self.tool_label_counts: dict[str, int] = {}
        self.tool_label_source_counts: dict[str, int] = {}
        self.used_tool_reward_count = 0
        self.focus_judge_hit_count = 0
        self.grounding_judge_hit_count = 0
        self.latest_metrics: dict[str, Any] = {}

    def update(
        self,
        *,
        rollouts: list[RolloutRecord],
        rewards: list[dict[str, Any]],
        metrics: dict[str, Any],
    ) -> None:
        self.step_count += 1
        self.rollout_count += len(rollouts)
        self.reward_count += len(rewards)
        self.used_tool_count += sum(1 for item in rollouts if item.used_tool)
        self.tool_call_count += sum(int(item.num_tool_calls) for item in rollouts)
        self.malformed_count += sum(1 for item in rollouts if (item.protocol or {}).get("native_errors"))
        self.answer_correct_count += sum(1 for row in rewards if row.get("answer_correct"))
        self.latest_metrics = dict(metrics)
        for key in (
            "reward_total",
            "reward_answer",
            "reward_tool",
            "reward_focus",
            "reward_ground",
            "reward_protocol",
        ):
            for row in rewards:
                value = row.get(key)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    self.reward_sums[key] = self.reward_sums.get(key, 0.0) + float(value)
        for label, count in _count_by_key(rewards, "tool_label").items():
            self.tool_label_counts[label] = self.tool_label_counts.get(label, 0) + count
        for source, count in _count_by_key(rewards, "tool_label_source").items():
            self.tool_label_source_counts[source] = self.tool_label_source_counts.get(source, 0) + count
        for row in rewards:
            metadata = dict(row.get("metadata") or {})
            if metadata.get("used_tool"):
                self.used_tool_reward_count += 1
                self.focus_judge_hit_count += 1 if row.get("focus_judge") else 0
                self.grounding_judge_hit_count += 1 if row.get("grounding_judge") else 0

    def summary(self) -> dict[str, Any]:
        reward_means = {
            key: value / max(self.reward_count, 1)
            for key, value in sorted(self.reward_sums.items())
        }
        return {
            "step_count": self.step_count,
            "rollout_count": self.rollout_count,
            "reward_count": self.reward_count,
            "tool_trigger_rate": self.used_tool_count / max(self.rollout_count, 1),
            "avg_tool_calls": self.tool_call_count / max(self.rollout_count, 1),
            "malformed_rate": self.malformed_count / max(self.rollout_count, 1),
            "answer_accuracy": self.answer_correct_count / max(self.reward_count, 1),
            "reward_means": reward_means,
            "tool_label_counts": dict(sorted(self.tool_label_counts.items())),
            "tool_label_source_counts": dict(sorted(self.tool_label_source_counts.items())),
            "focus_judge_hit_rate": self.focus_judge_hit_count / max(self.used_tool_reward_count, 1),
            "grounding_judge_hit_rate": self.grounding_judge_hit_count / max(self.used_tool_reward_count, 1),
            "latest_metrics": self.latest_metrics,
        }


def _create_stage3_wandb_logger(config: Stage3GRPOConfig, output_dir: Path) -> Any | None:
    wandb_config = config.wandb
    if not wandb_config.project or wandb_config.mode == "disabled":
        return None
    wandb_dir = output_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    from revisit_vlm.wandb_logging import WandbLogger

    return WandbLogger(
        project=wandb_config.project,
        entity=wandb_config.entity,
        name=wandb_config.name or config.run_id,
        group=wandb_config.group or "stage3_grpo",
        job_type=wandb_config.job_type,
        mode=wandb_config.mode,
        config=_stage3_wandb_config(config, output_dir),
        tags=[
            "stage3-grpo",
            str(config.rollout.runtime_backend),
            str(config.protocol),
            *list(wandb_config.tags),
        ],
        directory=wandb_dir,
    )


def _stage3_wandb_config(config: Stage3GRPOConfig, output_dir: Path) -> dict[str, Any]:
    git_commit, dirty_worktree = _git_identity()
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
        "git_commit": git_commit,
        "dirty_worktree": dirty_worktree,
    }


def _stage3_wandb_metrics(
    *,
    result: dict[str, Any],
    rollouts: list[RolloutRecord],
    rewards: list[dict[str, Any]],
) -> dict[str, Any]:
    update = dict(result.get("metrics") or {})
    global_step = int(update.get("global_step") or update.get("optimizer_step") or 1)
    metrics: dict[str, Any] = {
        "trainer/global_step": global_step,
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
    for source, count in _count_by_key(rewards, "tool_label_source").items():
        metrics[f"reward/tool_label_source_count/{source}"] = count
    used_tool_rewards = [
        row for row in rewards if dict(row.get("metadata") or {}).get("used_tool")
    ]
    metrics["reward/focus_judge_hit_rate"] = _mean_float(
        [1.0 if row.get("focus_judge") else 0.0 for row in used_tool_rewards]
    )
    metrics["reward/grounding_judge_hit_rate"] = _mean_float(
        [1.0 if row.get("grounding_judge") else 0.0 for row in used_tool_rewards]
    )
    return {key: value for key, value in metrics.items() if value is not None}


def _stage3_wandb_summary_from_aggregate(
    config: Stage3GRPOConfig,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": result.get("status"),
        "run_id": config.run_id,
        "runtime_backend": config.rollout.runtime_backend,
        "policy_checkpoint": config.policy_checkpoint,
        "rl_data_path": config.rl_data_path,
        "checkpoint": result.get("checkpoint"),
        "global_step": result.get("global_step"),
        "max_steps": result.get("max_steps"),
        "metrics": dict(result.get("metrics") or {}),
        "aggregate": dict(result.get("aggregate") or {}),
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
        "train_metrics.jsonl",
        "rollout_debug.jsonl",
        "reward_breakdown.jsonl",
        "LATEST_CHECKPOINT.txt",
        "judge_pending.jsonl",
        "probe_cache.jsonl",
        "probe_cache_summary.json",
    ]
    paths: list[str | Path] = [output_dir / name for name in names]
    if include_checkpoint:
        paths.extend(sorted(output_dir.glob("checkpoint_step_*.pt")))
    return paths


def _mean_float(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


def _seed_stage3_runtime(seed: int) -> None:
    random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass
    try:
        import numpy as np

        np.random.seed(int(seed) % (2**32))
    except Exception:
        pass


def _count_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _safe_artifact_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in value)
    return safe.strip("-") or "stage3-grpo-artifact"


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
