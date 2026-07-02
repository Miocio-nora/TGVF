"""Shared DeepStack scope contracts and runtime helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .schema import DeepStackScope, DeepStackState

DEEPSTACK_SCOPE_CONTRACT_SCHEMA_VERSION = "clean_deepstack_scope_contract_v1"
DEEPSTACK_RUNTIME_HOOKS_SCHEMA_VERSION = "clean_deepstack_runtime_hooks_v1"
QWEN3_DEEPSTACK_PAYLOAD_SCHEMA_VERSION = "clean_qwen3_deepstack_payload_v1"
DEEPSTACK_INJECTION_SOURCE = "native_qwen3_original_image_deepstack_features"
DEEPSTACK_FVT_VISUAL_TOKEN_PATH = "v_merge_level_visual_tokens"


@dataclass(frozen=True)
class Qwen3OriginalImageDeepStackPayload:
    """Payload passed directly to Qwen3VLTextModel for Qwen3 DeepStack."""

    visual_pos_masks: Any
    deepstack_visual_embeds: list[Any]
    original_image_token_indices: Any
    sequence_length: int
    feature_shapes: list[list[int]] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)
    d_token_indices: Any | None = None
    combined_token_indices: Any | None = None

    def to_debug_dict(self) -> dict[str, Any]:
        d_count = 0
        if self.d_token_indices is not None:
            d_count = len(self.d_token_indices.view(-1))
        combined_count = None
        if self.combined_token_indices is not None:
            combined_count = len(self.combined_token_indices.view(-1))
        return {
            "schema_version": QWEN3_DEEPSTACK_PAYLOAD_SCHEMA_VERSION,
            "sequence_length": self.sequence_length,
            "original_image_token_count": len(self.original_image_token_indices.view(-1)),
            "d_token_count": d_count,
            "combined_token_count": combined_count,
            "visual_pos_masks_shape": list(self.visual_pos_masks.shape),
            "deepstack_feature_count": len(self.deepstack_visual_embeds),
            "deepstack_feature_shapes": self.feature_shapes,
            **self.debug,
        }


def deepstack_scope_contract(
    state: DeepStackState | Mapping[str, Any],
    *,
    surface: str,
    execution_supported: bool,
    blocking_items: list[str] | tuple[str, ...] | None = None,
    backend: str | None = None,
    implemented_hooks: set[str] | frozenset[str] | tuple[str, ...] | list[str] | None = None,
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
        "runtime_hooks": deepstack_runtime_hooks(
            deepstack,
            surface=surface,
            backend=backend,
            implemented_hooks=implemented_hooks,
        ),
        "blocking_items": blockers,
    }


def deepstack_runtime_hooks(
    state: DeepStackState | Mapping[str, Any],
    *,
    surface: str,
    backend: str | None = None,
    implemented_hooks: set[str] | frozenset[str] | tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Return the hook-level implementation contract for DeepStack execution.

    This is intentionally more concrete than the high-level scope contract. It
    names the runtime hooks that must be ported before DeepStack can move from a
    schema-supported blocked state into actual execution support.
    """

    deepstack = _coerce_deepstack_state(state)
    policy = deepstack_scope_policy(deepstack.original_image_scope)
    implemented = {str(item) for item in (implemented_hooks or ())}
    specs = [
        (
            "capture_original_image_deepstack_features",
            bool(deepstack.enabled),
            "capture native Qwen3 original-image DeepStack features during image prefill",
        ),
        (
            "carry_original_image_deepstack_through_post_tgvf_append",
            bool(deepstack.enabled),
            "carry original-image DeepStack features into the post-TGVF append forward",
        ),
        (
            "apply_post_tgvf_deepstack_scope_mask",
            bool(deepstack.enabled and policy["block_after_tgvf_append"]),
            "mask original-image DeepStack features over the requested post-TGVF scope",
        ),
        (
            "restore_deepstack_for_answer_when_scope_requires",
            bool(deepstack.enabled and policy["restore_for_answer"]),
            "restore original-image DeepStack features for answer tokens under evidence_only scope",
        ),
    ]
    hooks = {}
    for name, required, description in specs:
        is_implemented = name in implemented
        status = (
            "ported"
            if required and is_implemented
            else "not_ported"
            if required
            else "not_required"
        )
        hooks[name] = {
            "required": required,
            "implemented": is_implemented,
            "status": status,
            "description": description,
            "blocking_item": None if status != "not_ported" else f"{name}: {description}",
        }
    blocking_items = [
        str(hook["blocking_item"])
        for hook in hooks.values()
        if hook.get("blocking_item")
    ]
    return {
        "schema_version": DEEPSTACK_RUNTIME_HOOKS_SCHEMA_VERSION,
        "surface": surface,
        "backend": backend,
        "enabled": bool(deepstack.enabled),
        "original_image_scope": str(deepstack.original_image_scope),
        "implemented_hooks": sorted(implemented),
        "hooks": hooks,
        "blocking_items": blocking_items,
        "all_required_hooks_implemented": not blocking_items,
    }


def qwen3_deepstack_runtime_hook_names_for_full_sequence_through_answer() -> set[str]:
    """Hooks implemented by the clean full-sequence through-answer path."""

    return {
        "capture_original_image_deepstack_features",
        "carry_original_image_deepstack_through_post_tgvf_append",
        "apply_post_tgvf_deepstack_scope_mask",
    }


def qwen3_deepstack_runtime_hook_names_for_full_sequence_evidence_only() -> set[str]:
    """Hooks implemented by the clean full-sequence evidence-only path."""

    return {
        "capture_original_image_deepstack_features",
        "carry_original_image_deepstack_through_post_tgvf_append",
        "apply_post_tgvf_deepstack_scope_mask",
        "restore_deepstack_for_answer_when_scope_requires",
    }


def qwen3_deepstack_runtime_hook_names_for_full_sequence_no_block() -> set[str]:
    """Hooks implemented by the clean full-sequence no-block path."""

    return {
        "capture_original_image_deepstack_features",
        "carry_original_image_deepstack_through_post_tgvf_append",
    }


def qwen3_deepstack_runtime_hook_names_for_kv_cache_through_answer() -> set[str]:
    """Hooks implemented by the clean cached-prefix through-answer path.

    In KV mode, the original image DeepStack features have already been applied
    by Qwen3's native image prefill and are carried in the prefix KV cache. The
    post-TGVF append path therefore only needs to preserve that cache and apply
    the original-image key block mask to the D/evidence/answer queries.
    """

    return {
        "capture_original_image_deepstack_features",
        "carry_original_image_deepstack_through_post_tgvf_append",
        "apply_post_tgvf_deepstack_scope_mask",
    }


def qwen3_deepstack_runtime_hook_names_for_kv_cache_evidence_only() -> set[str]:
    """Hooks implemented by the clean cached-prefix evidence-only path."""

    return {
        "capture_original_image_deepstack_features",
        "carry_original_image_deepstack_through_post_tgvf_append",
        "apply_post_tgvf_deepstack_scope_mask",
        "restore_deepstack_for_answer_when_scope_requires",
    }


def qwen3_deepstack_runtime_hook_names_for_kv_cache_no_block() -> set[str]:
    """Hooks implemented by the clean cached-prefix no-block path."""

    return {
        "capture_original_image_deepstack_features",
        "carry_original_image_deepstack_through_post_tgvf_append",
    }


def qwen3_deepstack_runtime_hook_names_for_stage2_training() -> set[str]:
    """Hooks implemented by the clean Stage2 full-sequence training path."""

    return {
        "capture_original_image_deepstack_features",
        "carry_original_image_deepstack_through_post_tgvf_append",
        "apply_post_tgvf_deepstack_scope_mask",
        "restore_deepstack_for_answer_when_scope_requires",
    }


def extract_qwen3_deepstack_features(output: Any) -> list[Any]:
    """Extract Qwen3 vision DeepStack feature tensors from a model output."""

    features = getattr(output, "deepstack_features", None)
    if features is None and isinstance(output, Mapping):
        features = output.get("deepstack_features")
    if features is None:
        return []
    return [item for item in features if hasattr(item, "shape")]


def capture_qwen3_original_image_deepstack_features(
    model: Any,
    model_inputs: Mapping[str, Any],
    *,
    detach: bool = True,
) -> list[Any]:
    """Capture native Qwen3 original-image DeepStack features from image inputs."""

    pixel_values = model_inputs.get("pixel_values")
    image_grid_thw = model_inputs.get("image_grid_thw")
    if pixel_values is None or image_grid_thw is None:
        raise ValueError("Qwen3 DeepStack capture requires pixel_values and image_grid_thw")
    if not hasattr(model, "get_image_features"):
        raise ValueError("Qwen3 model does not expose get_image_features for DeepStack capture")
    output = model.get_image_features(
        pixel_values,
        image_grid_thw=image_grid_thw,
        output_hidden_states=True,
        return_dict=True,
    )
    features = extract_qwen3_deepstack_features(output)
    if not features:
        raise ValueError("Qwen3 image feature output did not include deepstack_features")
    if detach:
        features = [item.detach() if hasattr(item, "detach") else item for item in features]
    return features


def build_qwen3_original_image_deepstack_payload(
    *,
    sequence_length: int,
    original_image_token_indices: Any,
    deepstack_features: list[Any] | tuple[Any, ...],
    device: Any,
    dtype: Any,
) -> Qwen3OriginalImageDeepStackPayload:
    """Build the text-model DeepStack payload for original image tokens only."""

    return build_qwen3_deepstack_payload(
        sequence_length=sequence_length,
        original_image_token_indices=original_image_token_indices,
        original_deepstack_features=deepstack_features,
        d_token_indices=None,
        d_deepstack_features=None,
        device=device,
        dtype=dtype,
        visual_pos_mask_policy="original_image_tokens_only",
    )


def build_qwen3_deepstack_payload(
    *,
    sequence_length: int,
    original_image_token_indices: Any | None = None,
    original_deepstack_features: list[Any] | tuple[Any, ...] | None = None,
    d_token_indices: Any | None = None,
    d_deepstack_features: list[Any] | tuple[Any, ...] | None = None,
    device: Any,
    dtype: Any,
    visual_pos_mask_policy: str = "original_and_d_tokens",
) -> Qwen3OriginalImageDeepStackPayload:
    """Build a Qwen3 DeepStack payload for original-image and/or D token positions."""

    import torch

    seq_len = int(sequence_length)
    if seq_len <= 0:
        raise ValueError("sequence_length must be positive")

    sources: list[tuple[str, Any, list[Any] | tuple[Any, ...]]] = []
    original_indices = None
    d_indices = None
    if original_image_token_indices is not None:
        original_indices = original_image_token_indices.to(device=device, dtype=torch.long).view(-1)
        if int(original_indices.numel()) > 0:
            sources.append(("original", original_indices, original_deepstack_features or []))
    if d_token_indices is not None:
        d_indices = d_token_indices.to(device=device, dtype=torch.long).view(-1)
        if int(d_indices.numel()) > 0:
            sources.append(("d", d_indices, d_deepstack_features or []))
    if not sources:
        raise ValueError("at least one DeepStack token index set is required")

    for _name, indices, _features in sources:
        min_index = int(indices.min().detach().cpu().item())
        max_index = int(indices.max().detach().cpu().item())
        if min_index < 0 or max_index >= seq_len:
            raise ValueError("DeepStack token indices are outside the sequence")

    layer_count = None
    for name, indices, features in sources:
        if not features:
            raise ValueError(f"{name} deepstack_features are required for DeepStack payload")
        if layer_count is None:
            layer_count = len(features)
        elif len(features) != layer_count:
            raise ValueError("DeepStack feature layer count mismatch between sources")
        expected_count = int(indices.numel())
        for feature in features:
            if not isinstance(feature, torch.Tensor):
                raise TypeError("deepstack feature must be a torch.Tensor")
            if feature.ndim != 2:
                raise ValueError(
                    f"deepstack feature must have shape [N, D], got {list(feature.shape)}"
                )
            if int(feature.shape[0]) != expected_count:
                raise ValueError(
                    "deepstack feature token count mismatch: "
                    f"feature={int(feature.shape[0])} {name}={expected_count}"
                )

    combined_indices = torch.cat([indices for _name, indices, _features in sources], dim=0)
    sorted_indices, sort_order = torch.sort(combined_indices)
    visual_pos_masks = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
    visual_pos_masks[0, sorted_indices] = True

    prepared = []
    feature_shapes = []
    for layer_index in range(int(layer_count or 0)):
        layer_features = torch.cat(
            [
                features[layer_index].to(device=device, dtype=dtype)
                for _name, _indices, features in sources
            ],
            dim=0,
        )
        layer_features = layer_features.index_select(0, sort_order)
        feature_shapes.append([int(item) for item in layer_features.shape])
        prepared.append(layer_features)

    if original_indices is None:
        original_indices = torch.empty(0, dtype=torch.long, device=device)

    return Qwen3OriginalImageDeepStackPayload(
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=prepared,
        original_image_token_indices=original_indices,
        d_token_indices=d_indices,
        combined_token_indices=sorted_indices,
        sequence_length=seq_len,
        feature_shapes=feature_shapes,
        debug={
            "injection_source": (
                DEEPSTACK_INJECTION_SOURCE
                if d_indices is None or int(d_indices.numel()) == 0
                else "native_qwen3_original_image_plus_tgvf_d_deepstack_features"
            ),
            "d_deepstack_features_enabled": bool(
                d_indices is not None and int(d_indices.numel()) > 0
            ),
            "visual_pos_mask_policy": visual_pos_mask_policy,
        },
    )


def build_original_image_key_block_attention_mask(
    *,
    attention_mask_2d: Any,
    original_image_token_indices: Any,
    block_query_start: int,
    dtype: Any,
    block_query_end: int | None = None,
) -> Any:
    """Create a 4D causal mask that blocks original-image keys over a query span."""

    import torch

    if attention_mask_2d.ndim != 2:
        raise ValueError("attention_mask_2d must have shape [batch, seq_len]")
    batch, seq_len = int(attention_mask_2d.shape[0]), int(attention_mask_2d.shape[-1])
    if batch != 1:
        raise ValueError("clean DeepStack key-block mask currently supports batch size 1")
    device = attention_mask_2d.device
    min_value = torch.finfo(dtype).min
    mask = torch.zeros((batch, 1, seq_len, seq_len), dtype=dtype, device=device)
    future = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)
    mask = mask.masked_fill(future.view(1, 1, seq_len, seq_len), min_value)
    key_padding = attention_mask_2d[:, None, None, :] == 0
    mask = mask.masked_fill(key_padding, min_value)
    original_indices = original_image_token_indices.to(device=device, dtype=torch.long).view(-1)
    if int(original_indices.numel()) > 0:
        query_range = torch.arange(seq_len, device=device)
        query_mask = query_range >= int(block_query_start)
        if block_query_end is not None:
            query_mask = query_mask & (query_range < int(block_query_end))
        query_indices = query_range[query_mask]
        if int(query_indices.numel()) > 0:
            mask[:, :, query_indices[:, None], original_indices[None, :]] = min_value
    return mask


def build_single_query_original_image_key_block_attention_mask(
    *,
    attention_mask_2d: Any,
    original_image_token_indices: Any,
    dtype: Any,
) -> Any:
    """Create a cached-generation 4D mask for one new query token."""

    import torch

    if attention_mask_2d.ndim != 2:
        raise ValueError("attention_mask_2d must have shape [batch, key_len]")
    batch, key_len = int(attention_mask_2d.shape[0]), int(attention_mask_2d.shape[-1])
    if batch != 1:
        raise ValueError("clean DeepStack cached mask currently supports batch size 1")
    device = attention_mask_2d.device
    min_value = torch.finfo(dtype).min
    mask = torch.zeros((batch, 1, 1, key_len), dtype=dtype, device=device)
    key_padding = attention_mask_2d[:, None, None, :] == 0
    mask = mask.masked_fill(key_padding, min_value)
    original_indices = original_image_token_indices.to(device=device, dtype=torch.long).view(-1)
    if int(original_indices.numel()) > 0:
        mask[:, :, :, original_indices] = min_value
    return mask


def build_cached_chunk_original_image_key_block_attention_mask(
    *,
    attention_mask_2d: Any,
    original_image_token_indices: Any,
    query_start: int,
    query_length: int,
    block_query_offset: int = 0,
    dtype: Any,
) -> Any:
    """Create a cached-generation 4D mask for a multi-token append chunk.

    `attention_mask_2d` spans the full key sequence after appending the chunk.
    The returned mask has query length `query_length` and key length equal to
    the full sequence. It applies normal causal masking inside the appended
    chunk and blocks original-image keys for every appended query token.
    """

    import torch

    if attention_mask_2d.ndim != 2:
        raise ValueError("attention_mask_2d must have shape [batch, key_len]")
    batch, key_len = int(attention_mask_2d.shape[0]), int(attention_mask_2d.shape[-1])
    if batch != 1:
        raise ValueError("clean DeepStack cached chunk mask currently supports batch size 1")
    q_start = int(query_start)
    q_len = int(query_length)
    if q_len <= 0:
        raise ValueError("query_length must be positive")
    if q_start < 0 or q_start + q_len > key_len:
        raise ValueError("query span is outside the key sequence")
    device = attention_mask_2d.device
    min_value = torch.finfo(dtype).min
    mask = torch.zeros((batch, 1, q_len, key_len), dtype=dtype, device=device)

    query_positions = torch.arange(q_start, q_start + q_len, device=device)
    key_positions = torch.arange(key_len, device=device)
    future = key_positions.view(1, -1) > query_positions.view(-1, 1)
    mask = mask.masked_fill(future.view(1, 1, q_len, key_len), min_value)

    key_padding = attention_mask_2d[:, None, None, :] == 0
    mask = mask.masked_fill(key_padding, min_value)

    block_offset = int(block_query_offset)
    if block_offset < 0 or block_offset > q_len:
        raise ValueError("block_query_offset must be within the query span")
    original_indices = original_image_token_indices.to(device=device, dtype=torch.long).view(-1)
    if int(original_indices.numel()) > 0 and block_offset < q_len:
        mask[:, :, block_offset:, original_indices] = min_value
    return mask


def deepstack_scope_policy(scope: DeepStackScope | str) -> dict[str, Any]:
    resolved = DeepStackScope(str(scope))
    restore_for_answer = resolved == DeepStackScope.EVIDENCE_ONLY
    block_after_tgvf_append = resolved in {
        DeepStackScope.THROUGH_ANSWER,
        DeepStackScope.EVIDENCE_ONLY,
    }
    if restore_for_answer:
        block_scope = "evidence_only"
    elif block_after_tgvf_append:
        block_scope = "through_answer"
    elif resolved == DeepStackScope.NO_BLOCK:
        block_scope = "no_block"
    else:
        block_scope = "off"
    return {
        "block_after_tgvf_append": block_after_tgvf_append,
        "block_query_start": "post_tgvf_append" if block_after_tgvf_append else None,
        "block_query_end": "answer_start" if restore_for_answer else None,
        "block_scope": block_scope,
        "restore_for_answer": restore_for_answer,
        "must_follow_attention_mask_scope": block_after_tgvf_append,
        "scope_mapping": {
            "no_block": "inject original-image DeepStack and do not block original-image keys after D",
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
