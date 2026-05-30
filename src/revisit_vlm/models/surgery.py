from __future__ import annotations

from typing import Any

import torch.nn as nn


def freeze_vision_tower(model: nn.Module) -> None:
    """Freeze Qwen2-VL visual encoder parameters."""

    visual = getattr(model, "visual", None)
    if visual is None:
        raise AttributeError("Expected Qwen2-VL model to expose a 'visual' module")
    for parameter in visual.parameters():
        parameter.requires_grad = False


def apply_research_edits(model: nn.Module, config: dict[str, Any] | None = None) -> nn.Module:
    """Central hook for architecture edits.

    Add experimental module replacement here so the training script can stay
    stable while the model structure changes.
    """

    config = config or {}
    if config.get("freeze_vision_tower", False):
        freeze_vision_tower(model)
    return model
