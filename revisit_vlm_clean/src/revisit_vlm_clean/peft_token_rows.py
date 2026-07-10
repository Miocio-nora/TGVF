"""PEFT protocol-token training helpers shared by train and eval paths."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


PROTOCOL_TOKEN_TRAINING_FULL_MODULES = "full_modules"
PROTOCOL_TOKEN_TRAINING_ROW_ONLY = "row_only"
PROTOCOL_TOKEN_TRAINING_MODES = (
    PROTOCOL_TOKEN_TRAINING_FULL_MODULES,
    PROTOCOL_TOKEN_TRAINING_ROW_ONLY,
)


def protocol_token_peft_kwargs(
    *,
    mode: str,
    token_ids: list[int],
) -> dict[str, Any]:
    """Return mutually exclusive PEFT settings for protocol-token training."""

    normalized_mode = str(mode)
    if normalized_mode not in PROTOCOL_TOKEN_TRAINING_MODES:
        raise ValueError(f"unsupported protocol token training mode: {normalized_mode}")
    ordered_ids = list(dict.fromkeys(int(token_id) for token_id in token_ids))
    if not ordered_ids:
        raise ValueError("protocol token training requires at least one token id")
    if normalized_mode == PROTOCOL_TOKEN_TRAINING_FULL_MODULES:
        return {
            "modules_to_save": ["embed_tokens", "lm_head"],
            "trainable_token_indices": None,
        }
    return {
        "modules_to_save": None,
        "trainable_token_indices": {
            "embed_tokens": ordered_ids,
            "lm_head": ordered_ids,
        },
    }


def trainable_token_indices_from_checkpoint(
    *,
    state: Mapping[str, Any],
    token_ids: list[int],
) -> dict[str, list[int]] | None:
    """Reconstruct the exact TrainableTokens wrappers represented in a state dict."""

    ordered_ids = list(dict.fromkeys(int(token_id) for token_id in token_ids))
    keys = [str(key) for key in state]
    result: dict[str, list[int]] = {}
    if any(
        "embed_tokens" in key and ("token_adapter" in key or "trainable_tokens" in key)
        for key in keys
    ):
        result["embed_tokens"] = ordered_ids
    if any(
        "lm_head" in key and ("token_adapter" in key or "trainable_tokens" in key)
        for key in keys
    ):
        result["lm_head"] = ordered_ids
    return result or None
