"""Minimal self-owned GRPO math for Stage3."""

from __future__ import annotations

import math
from typing import Any

from .schemas import RolloutRecord


def group_advantages(rewards: list[float], *, eps: float = 1e-6) -> list[float]:
    if not rewards:
        return []
    mean = sum(float(item) for item in rewards) / len(rewards)
    var = sum((float(item) - mean) ** 2 for item in rewards) / len(rewards)
    std = math.sqrt(var)
    return [(float(item) - mean) / (std + float(eps)) for item in rewards]


def attach_group_advantages(
    rollouts: list[RolloutRecord],
    rewards: dict[tuple[str, int], float],
    *,
    eps: float,
) -> dict[tuple[str, int], float]:
    by_sample: dict[str, list[RolloutRecord]] = {}
    for rollout in rollouts:
        by_sample.setdefault(rollout.sample_id, []).append(rollout)
    out: dict[tuple[str, int], float] = {}
    for sample_id, group in by_sample.items():
        values = [float(rewards[(item.sample_id, item.rollout_id)]) for item in group]
        advs = group_advantages(values, eps=eps)
        for item, adv in zip(group, advs, strict=True):
            out[(sample_id, item.rollout_id)] = adv
    return out


def grpo_loss_from_tensors(
    *,
    new_logprobs: Any,
    old_logprobs: Any,
    advantages: Any,
    loss_mask: Any,
    clip_range: float,
    ref_logprobs: Any | None = None,
    kl_coef: float = 0.0,
) -> tuple[Any, dict[str, float]]:
    import torch

    new = torch.as_tensor(new_logprobs, dtype=torch.float32)
    old = torch.as_tensor(old_logprobs, dtype=torch.float32, device=new.device)
    adv = torch.as_tensor(advantages, dtype=torch.float32, device=new.device)
    mask = torch.as_tensor(loss_mask, dtype=torch.float32, device=new.device)
    if new.shape != old.shape or new.shape != mask.shape:
        raise ValueError("new_logprobs, old_logprobs, and loss_mask must have the same shape")
    if adv.ndim == 1:
        adv = adv.view(-1, 1).expand_as(new)
    if adv.shape != new.shape:
        raise ValueError("advantages must be [batch] or the same shape as logprobs")
    denom = mask.sum().clamp_min(1.0)
    ratio = torch.exp(new - old)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - float(clip_range), 1.0 + float(clip_range)) * adv
    pg_loss = -(torch.minimum(unclipped, clipped) * mask).sum() / denom
    kl = torch.zeros((), dtype=torch.float32, device=new.device)
    if ref_logprobs is not None and float(kl_coef) > 0:
        ref = torch.as_tensor(ref_logprobs, dtype=torch.float32, device=new.device)
        if ref.shape != new.shape:
            raise ValueError("ref_logprobs must match new_logprobs")
        log_ratio = ref - new
        kl = ((torch.exp(log_ratio) - log_ratio - 1.0) * mask).sum() / denom
    loss = pg_loss + float(kl_coef) * kl
    approx_clip_frac = ((torch.abs(ratio - 1.0) > float(clip_range)).float() * mask).sum() / denom
    return loss, {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(pg_loss.detach().cpu()),
        "kl": float(kl.detach().cpu()),
        "clip_fraction": float(approx_clip_frac.detach().cpu()),
        "token_count": float(denom.detach().cpu()),
    }


def rollout_logprob_tensors(
    rollouts: list[RolloutRecord],
    advantages: dict[tuple[str, int], float],
    *,
    trainable_bias: Any | None = None,
) -> tuple[Any, Any, Any, Any, Any]:
    """Build padded tensors for fake/smoke GRPO updates.

    Real model training will replace this with replayed policy logprobs. This
    helper keeps the optimizer/checkpoint plumbing testable without loading 8B.
    """
    import torch

    max_len = max((len(item.old_logprobs) for item in rollouts), default=1)
    old_rows = []
    new_rows = []
    ref_rows = []
    mask_rows = []
    adv_rows = []
    for rollout in rollouts:
        old = list(rollout.old_logprobs or (-1.0,))
        ref = list(rollout.ref_logprobs or old)
        mask = list(rollout.loss_mask or [1] * len(old))
        pad = max_len - len(old)
        old_rows.append(old + [0.0] * pad)
        ref_rows.append(ref + [0.0] * pad)
        mask_rows.append(mask + [0] * pad)
        adv_rows.append(float(advantages[(rollout.sample_id, rollout.rollout_id)]))
    old_tensor = torch.tensor(old_rows, dtype=torch.float32)
    if trainable_bias is None:
        new_tensor = old_tensor.clone()
    else:
        new_tensor = old_tensor + trainable_bias
    return (
        new_tensor,
        old_tensor,
        torch.tensor(adv_rows, dtype=torch.float32),
        torch.tensor(mask_rows, dtype=torch.float32),
        torch.tensor(ref_rows, dtype=torch.float32),
    )
