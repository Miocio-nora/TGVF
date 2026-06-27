"""Small Stage3 GRPO trainer core."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .data import BalancedPromptSampler, load_stage3_samples, write_jsonl
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
        return {
            "status": "stage3_grpo_fake_smoke_completed",
            "output_dir": str(self.output_dir),
            "rollout_debug": str(self.output_dir / "rollout_debug.jsonl"),
            "reward_breakdown": str(self.output_dir / "reward_breakdown.jsonl"),
            "train_metrics": str(self.output_dir / "train_metrics.json"),
            "checkpoint": str(checkpoint_path),
            "metrics": update,
        }

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
        return {
            "status": "stage3_grpo_native_step_completed",
            "output_dir": str(self.output_dir),
            "rollout_debug": str(self.output_dir / "rollout_debug.jsonl"),
            "reward_breakdown": str(self.output_dir / "reward_breakdown.jsonl"),
            "train_metrics": str(self.output_dir / "train_metrics.json"),
            "checkpoint": str(checkpoint_path),
            "metrics": update,
        }

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
