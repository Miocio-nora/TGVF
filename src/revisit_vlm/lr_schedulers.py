from __future__ import annotations

import math
from typing import Any

import torch
from torch.optim.lr_scheduler import LambdaLR


def estimate_optimizer_steps(max_steps: int, gradient_accumulation_steps: int) -> int:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    return math.ceil(max_steps / gradient_accumulation_steps)


def resolve_num_training_steps(
    *,
    max_steps: int,
    gradient_accumulation_steps: int,
    num_training_steps: int | None = None,
) -> int:
    if num_training_steps is not None and num_training_steps > 0:
        return int(num_training_steps)
    return estimate_optimizer_steps(max_steps, gradient_accumulation_steps)


def resolve_warmup_steps(
    *,
    num_training_steps: int,
    warmup_ratio: float = 0.03,
    warmup_steps: int = 0,
) -> int:
    if num_training_steps <= 0:
        raise ValueError("num_training_steps must be positive")
    if warmup_steps and warmup_steps > 0:
        actual_warmup_steps = int(warmup_steps)
    else:
        actual_warmup_steps = int(num_training_steps * float(warmup_ratio))
    return max(1, min(actual_warmup_steps, num_training_steps))


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    num_training_steps: int,
    warmup_ratio: float = 0.03,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    if num_training_steps <= 0:
        raise ValueError("num_training_steps must be positive")
    actual_warmup_steps = resolve_warmup_steps(
        num_training_steps=num_training_steps,
        warmup_ratio=warmup_ratio,
        warmup_steps=warmup_steps,
    )
    min_lr_ratio = float(min_lr_ratio)
    if min_lr_ratio < 0.0 or min_lr_ratio > 1.0:
        raise ValueError("min_lr_ratio must be between 0 and 1")

    def lr_lambda(step: int) -> float:
        # LambdaLR calls this with an optimizer-step index. Initial construction
        # applies step 0, so the first optimizer update uses the first warmup LR.
        if step < actual_warmup_steps:
            return float(step + 1) / float(actual_warmup_steps)

        decay_steps = max(1, num_training_steps - actual_warmup_steps)
        progress = float(step - actual_warmup_steps + 1) / float(decay_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    lr_scheduler: str,
    num_training_steps: int,
    warmup_ratio: float = 0.03,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
) -> LambdaLR | None:
    if lr_scheduler == "none":
        return None
    if lr_scheduler == "warmup_cosine":
        return build_warmup_cosine_scheduler(
            optimizer,
            num_training_steps=num_training_steps,
            warmup_ratio=warmup_ratio,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
        )
    raise ValueError(f"Unknown lr_scheduler: {lr_scheduler}")


def scheduler_config_from_args(args: Any) -> dict[str, Any]:
    estimated_optimizer_steps = resolve_num_training_steps(
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_training_steps=args.num_training_steps,
    )
    warmup_steps = (
        resolve_warmup_steps(
            num_training_steps=estimated_optimizer_steps,
            warmup_ratio=args.warmup_ratio,
            warmup_steps=args.warmup_steps,
        )
        if args.lr_scheduler == "warmup_cosine"
        else 0
    )
    return {
        "lr_scheduler": args.lr_scheduler,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": warmup_steps,
        "requested_warmup_steps": args.warmup_steps,
        "min_lr_ratio": args.min_lr_ratio,
        "min_learning_rate": args.learning_rate * args.min_lr_ratio,
        "estimated_optimizer_steps": estimated_optimizer_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "max_steps_semantics": "micro_batch_steps",
    }


def current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])
