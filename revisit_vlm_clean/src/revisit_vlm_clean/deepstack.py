"""Shared DeepStack scope contracts for clean training and evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .schema import DeepStackScope, DeepStackState

DEEPSTACK_SCOPE_CONTRACT_SCHEMA_VERSION = "clean_deepstack_scope_contract_v1"
DEEPSTACK_INJECTION_SOURCE = "native_qwen3_original_image_deepstack_features"
DEEPSTACK_FVT_VISUAL_TOKEN_PATH = "v_merge_level_visual_tokens"


def deepstack_scope_contract(
    state: DeepStackState | Mapping[str, Any],
    *,
    surface: str,
    execution_supported: bool,
    blocking_items: list[str] | tuple[str, ...] | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    """Return the single clean contract for DeepStack scope semantics.

    The contract intentionally does not claim that D has DeepStack-like features.
    Current mainline keeps D as v-merge-level visual tokens and only scopes the
    original-image DeepStack features.
    """

    deepstack = _coerce_deepstack_state(state)
    policy = deepstack_scope_policy(deepstack.original_image_scope)
    blockers = [str(item) for item in (blocking_items or [])]
    return {
        "schema_version": DEEPSTACK_SCOPE_CONTRACT_SCHEMA_VERSION,
        "surface": surface,
        "backend": backend,
        "enabled": bool(deepstack.enabled),
        "requested": deepstack.to_dict(),
        "execution_supported": bool(execution_supported),
        "original_image_scope": str(deepstack.original_image_scope),
        "original_image_deepstack": {
            "required_when_enabled": bool(deepstack.enabled),
            "injection_source": DEEPSTACK_INJECTION_SOURCE,
            **policy,
        },
        "d_deepstack_features": {
            "enabled": bool(deepstack.d_features_enabled),
            "clean_default": False,
            "required_for_current_mainline": False,
        },
        "fvt_visual_token_path": DEEPSTACK_FVT_VISUAL_TOKEN_PATH,
        "blocking_items": blockers,
    }


def deepstack_scope_policy(scope: DeepStackScope | str) -> dict[str, Any]:
    resolved = DeepStackScope(str(scope))
    restore_for_answer = resolved == DeepStackScope.EVIDENCE_ONLY
    block_after_tgvf_append = resolved in {
        DeepStackScope.THROUGH_ANSWER,
        DeepStackScope.EVIDENCE_ONLY,
    }
    return {
        "block_after_tgvf_append": block_after_tgvf_append,
        "block_query_start": "post_tgvf_append" if block_after_tgvf_append else None,
        "block_query_end": "answer_start" if restore_for_answer else None,
        "block_scope": (
            "evidence_only"
            if restore_for_answer
            else "through_answer"
            if block_after_tgvf_append
            else "off"
        ),
        "restore_for_answer": restore_for_answer,
        "must_follow_attention_mask_scope": True,
        "scope_mapping": {
            "through_answer": "block original-image DeepStack from D/evidence through answer",
            "evidence_only": "block original-image DeepStack for D/evidence and restore for answer",
            "off": "do not inject original-image DeepStack in clean TGVF post-D scope",
        },
    }


def _coerce_deepstack_state(state: DeepStackState | Mapping[str, Any]) -> DeepStackState:
    if isinstance(state, DeepStackState):
        return state
    data = dict(state)
    if "original_image_scope" in data:
        data["original_image_scope"] = DeepStackScope(str(data["original_image_scope"]))
    resolved = DeepStackState(**data)
    resolved.validate()
    return resolved
