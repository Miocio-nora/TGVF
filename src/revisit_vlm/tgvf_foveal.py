from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class FovealCrossAttentionOutput:
    foveated_visual_tokens: torch.Tensor
    attention_debug: dict[str, torch.Tensor]
    debug_metadata: dict[str, Any]
    conditioned_pre_merge_visual_tokens: torch.Tensor | None = None


@dataclass
class BracketedFVTAppendResult:
    past_key_values: Any
    attention_mask: torch.Tensor | None
    cache_position: torch.Tensor | None
    input_ids: torch.Tensor | None
    last_logits: torch.Tensor | None
    appended_token_ids: torch.Tensor
    appended_inputs_embeds: torch.Tensor
    fvt_token_start: int
    fvt_token_end: int
    model_kwargs: dict[str, Any]
    debug_metadata: dict[str, Any]


@dataclass
class ContinuationResult:
    generated_ids: list[int]
    generated_text: str
    past_key_values: Any
    attention_mask: torch.Tensor | None
    input_ids: torch.Tensor | None
    last_logits: torch.Tensor | None
    stop_reason: str


FVT_APPEND_MODES = ("legacy_text_positions", "qwen_native_pseudo_image")


def validate_fvt_append_mode(value: str) -> str:
    if value not in FVT_APPEND_MODES:
        raise ValueError(f"Invalid FVT append mode {value!r}. Expected one of {list(FVT_APPEND_MODES)}")
    return value


class TokenFovealCrossAttention(nn.Module):
    """Token-level FVT ablation.

    By default this returns one FVT per target token, so the output shape is
    ``[T, d_lm]``. If ``num_output_tokens`` is set, attended token outputs are
    adaptively pooled to that fixed length for append compatibility.
    """

    variant_name = "token_foveal_cross_attention"

    def __init__(
        self,
        *,
        d_lm: int,
        d_v: int,
        attn_dim: int | None = None,
        num_output_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self.d_lm = d_lm
        self.d_v = d_v
        self.attn_dim = attn_dim or d_lm
        self.num_output_tokens = num_output_tokens
        self.q_proj = nn.Linear(d_lm, self.attn_dim)
        self.k_proj = nn.Linear(d_v, self.attn_dim)
        self.v_proj = nn.Linear(d_v, self.attn_dim)
        self.out_proj = nn.Linear(self.attn_dim, d_lm)

    def forward(
        self,
        *,
        target_hidden_states: torch.Tensor,
        pre_merge_visual_tokens: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> FovealCrossAttentionOutput:
        _validate_inputs(target_hidden_states, pre_merge_visual_tokens)
        queries = self.q_proj(target_hidden_states)
        keys = self.k_proj(pre_merge_visual_tokens)
        values = self.v_proj(pre_merge_visual_tokens)
        attended, attention = _cross_attention(queries, keys, values)
        fvt = self.out_proj(attended)
        output_mode = "per_target_token"
        if self.num_output_tokens is not None:
            fvt = _resize_token_sequence(fvt, self.num_output_tokens)
            attention = _resize_token_sequence(attention, self.num_output_tokens)
            output_mode = "adaptive_pooled_fixed_m"
        return FovealCrossAttentionOutput(
            foveated_visual_tokens=fvt,
            attention_debug={"attention_weights": attention},
            debug_metadata=_metadata(
                self.variant_name,
                target_hidden_states,
                pre_merge_visual_tokens,
                fvt,
                attention,
                metadata,
                output_mode=output_mode,
            ),
        )


class PooledFovealCrossAttention(nn.Module):
    """Fixed-M FVTs from mean-pooled target query plus learned output slots."""

    variant_name = "pooled_foveal_cross_attention"

    def __init__(
        self,
        *,
        d_lm: int,
        d_v: int,
        num_fvt_tokens: int,
        attn_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.d_lm = d_lm
        self.d_v = d_v
        self.num_fvt_tokens = num_fvt_tokens
        self.attn_dim = attn_dim or d_lm
        self.learned_slots = nn.Parameter(torch.randn(num_fvt_tokens, self.attn_dim) * 0.02)
        self.query_to_slots = nn.Linear(d_lm, num_fvt_tokens * self.attn_dim)
        self.k_proj = nn.Linear(d_v, self.attn_dim)
        self.v_proj = nn.Linear(d_v, self.attn_dim)
        self.out_proj = nn.Linear(self.attn_dim, d_lm)

    def forward(
        self,
        *,
        target_hidden_states: torch.Tensor,
        pre_merge_visual_tokens: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> FovealCrossAttentionOutput:
        _validate_inputs(target_hidden_states, pre_merge_visual_tokens)
        q_global = target_hidden_states.mean(dim=0)
        conditioned = self.query_to_slots(q_global).view(self.num_fvt_tokens, self.attn_dim)
        queries = self.learned_slots + conditioned
        keys = self.k_proj(pre_merge_visual_tokens)
        values = self.v_proj(pre_merge_visual_tokens)
        attended, attention = _cross_attention(queries, keys, values)
        fvt = self.out_proj(attended)
        return FovealCrossAttentionOutput(
            foveated_visual_tokens=fvt,
            attention_debug={"attention_weights": attention},
            debug_metadata=_metadata(
                self.variant_name,
                target_hidden_states,
                pre_merge_visual_tokens,
                fvt,
                attention,
                metadata,
                num_fvt_tokens=self.num_fvt_tokens,
                pooler="mean",
            ),
        )


class FovealCrossMerger(nn.Module):
    """Pooled query + MxR sub-slots + Qwen-style merger MLP."""

    variant_name = "foveal_cross_merger"

    def __init__(
        self,
        *,
        d_lm: int,
        d_v: int,
        num_fvt_tokens: int,
        spatial_merge_size: int = 2,
    ) -> None:
        super().__init__()
        self.d_lm = d_lm
        self.d_v = d_v
        self.num_fvt_tokens = num_fvt_tokens
        self.spatial_merge_size = spatial_merge_size
        self.num_sub_slots = spatial_merge_size**2
        self.learned_sub_slots = nn.Parameter(
            torch.randn(num_fvt_tokens, self.num_sub_slots, d_v) * 0.02
        )
        self.query_to_slots = nn.Linear(d_lm, num_fvt_tokens * self.num_sub_slots * d_v)
        self.q_proj = nn.Linear(d_v, d_v)
        self.k_proj = nn.Linear(d_v, d_v)
        self.v_proj = nn.Linear(d_v, d_v)
        self.sub_slot_norm = nn.LayerNorm(d_v)
        self.merger = nn.Sequential(
            nn.Linear(self.num_sub_slots * d_v, self.num_sub_slots * d_v),
            nn.GELU(),
            nn.Linear(self.num_sub_slots * d_v, d_lm),
        )

    def forward(
        self,
        *,
        target_hidden_states: torch.Tensor,
        pre_merge_visual_tokens: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> FovealCrossAttentionOutput:
        _validate_inputs(target_hidden_states, pre_merge_visual_tokens)
        q_global = target_hidden_states.mean(dim=0)
        conditioned = self.query_to_slots(q_global).view(
            self.num_fvt_tokens,
            self.num_sub_slots,
            self.d_v,
        )
        sub_slots = self.learned_sub_slots + conditioned
        flat_slots = sub_slots.reshape(self.num_fvt_tokens * self.num_sub_slots, self.d_v)
        queries = self.q_proj(flat_slots)
        keys = self.k_proj(pre_merge_visual_tokens)
        values = self.v_proj(pre_merge_visual_tokens)
        attended, attention = _cross_attention(queries, keys, values)
        attended_groups = attended.view(self.num_fvt_tokens, self.num_sub_slots, self.d_v)
        normed = self.sub_slot_norm(attended_groups)
        fvt = self.merger(normed.reshape(self.num_fvt_tokens, self.num_sub_slots * self.d_v))
        return FovealCrossAttentionOutput(
            foveated_visual_tokens=fvt,
            attention_debug={
                "attention_weights": attention,
                "sub_slot_attention_weights": attention.view(
                    self.num_fvt_tokens,
                    self.num_sub_slots,
                    pre_merge_visual_tokens.shape[0],
                ),
            },
            debug_metadata=_metadata(
                self.variant_name,
                target_hidden_states,
                pre_merge_visual_tokens,
                fvt,
                attention,
                metadata,
                num_fvt_tokens=self.num_fvt_tokens,
                num_sub_slots=self.num_sub_slots,
                spatial_merge_size=self.spatial_merge_size,
                merger_initialization="random",
                pooler="mean",
            ),
        )


class TargetSlotFovealCrossMerger(nn.Module):
    """Target-token structured slot cross-merger.

    Learned MxR sub-slots first cross-attend to the full target token sequence,
    then use those target-conditioned slots as queries into pre-merge visual
    tokens. This avoids the single q_global pooling pathway used by
    ``FovealCrossMerger``.
    """

    variant_name = "target_slot_foveal_cross_merger"

    def __init__(
        self,
        *,
        d_lm: int | None = None,
        d_v: int | None = None,
        llm_dim: int | None = None,
        vision_dim: int | None = None,
        num_fvt_tokens: int | None = None,
        num_foveated_tokens: int | None = None,
        spatial_merge_size: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
        slot_dim: int | None = None,
        merger_hidden_dim: int | None = None,
        use_target_self_attn: bool = False,
        use_slot_self_attn: bool = False,
        use_query_residual: bool = False,
        query_residual_init: float = 0.0,
        use_fvt_calibration: bool = False,
        return_attention: bool = False,
    ) -> None:
        super().__init__()
        self.d_lm = int(d_lm if d_lm is not None else llm_dim)
        self.d_v = int(d_v if d_v is not None else vision_dim)
        self.num_fvt_tokens = int(
            num_fvt_tokens if num_fvt_tokens is not None else num_foveated_tokens
        )
        self.spatial_merge_size = int(spatial_merge_size)
        self.num_sub_slots = self.spatial_merge_size**2
        self.total_sub_slots = self.num_fvt_tokens * self.num_sub_slots
        self.slot_dim = int(slot_dim or self.d_v)
        self.num_heads = _resolve_num_heads(self.slot_dim, num_heads)
        self.use_target_self_attn = use_target_self_attn
        self.use_slot_self_attn = use_slot_self_attn
        self.use_query_residual = use_query_residual
        self.use_fvt_calibration = use_fvt_calibration
        self.return_attention = return_attention

        self.sub_slot_embed = nn.Parameter(torch.randn(self.total_sub_slots, self.slot_dim) * 0.02)
        self.fvt_index_embed = nn.Parameter(torch.randn(self.num_fvt_tokens, self.slot_dim) * 0.02)
        self.sub_index_embed = nn.Parameter(torch.randn(self.num_sub_slots, self.slot_dim) * 0.02)

        target_num_heads = _resolve_num_heads(self.d_lm, num_heads)
        self.target_self_attn = (
            nn.TransformerEncoderLayer(
                d_model=self.d_lm,
                nhead=target_num_heads,
                dim_feedforward=max(self.d_lm * 2, 4),
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            if use_target_self_attn
            else None
        )
        self.target_proj = nn.Linear(self.d_lm, self.slot_dim)
        self.target_cross_attn = nn.MultiheadAttention(
            embed_dim=self.slot_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.target_slot_norm = nn.LayerNorm(self.slot_dim)

        self.slot_self_attn = (
            nn.MultiheadAttention(
                embed_dim=self.slot_dim,
                num_heads=self.num_heads,
                dropout=dropout,
                batch_first=True,
            )
            if use_slot_self_attn
            else None
        )
        self.slot_self_norm = nn.LayerNorm(self.slot_dim) if use_slot_self_attn else None

        self.visual_key_proj = nn.Linear(self.d_v, self.slot_dim)
        self.visual_value_proj = nn.Linear(self.d_v, self.slot_dim)
        self.visual_cross_attn = nn.MultiheadAttention(
            embed_dim=self.slot_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.visual_out_proj = nn.Linear(self.slot_dim, self.d_v)
        self.visual_slot_norm = nn.LayerNorm(self.d_v)

        merger_input_dim = self.num_sub_slots * self.d_v
        merger_hidden = int(merger_hidden_dim or merger_input_dim)
        self.group_merger = nn.Sequential(
            nn.LayerNorm(merger_input_dim),
            nn.Linear(merger_input_dim, merger_hidden),
            nn.GELU(),
            nn.Linear(merger_hidden, self.d_lm),
        )

        if use_query_residual:
            self.query_residual_proj = nn.Linear(self.slot_dim, self.d_lm)
            self.query_residual_gate = nn.Parameter(torch.tensor(float(query_residual_init)))
        else:
            self.query_residual_proj = None
            self.query_residual_gate = None

        if use_fvt_calibration:
            self.fvt_calibration_norm = nn.LayerNorm(self.d_lm)
            self.fvt_scale = nn.Parameter(torch.ones(self.d_lm))
            self.fvt_bias = nn.Parameter(torch.zeros(self.d_lm))
        else:
            self.fvt_calibration_norm = None
            self.fvt_scale = None
            self.fvt_bias = None

    def forward(
        self,
        *,
        target_hidden_states: torch.Tensor,
        pre_merge_visual_tokens: torch.Tensor,
        target_attention_mask: torch.Tensor | None = None,
        visual_attention_mask: torch.Tensor | None = None,
        metadata: dict[str, Any] | None = None,
        return_attention: bool | None = None,
    ) -> FovealCrossAttentionOutput:
        target, visual, target_mask, visual_mask, unbatched = self._normalize_inputs(
            target_hidden_states,
            pre_merge_visual_tokens,
            target_attention_mask,
            visual_attention_mask,
        )
        batch_size = target.shape[0]
        target_key_padding = _mask_to_key_padding(target_mask)
        visual_key_padding = _mask_to_key_padding(visual_mask)

        if self.target_self_attn is not None:
            target = self.target_self_attn(target, src_key_padding_mask=target_key_padding)
        target_tokens = self.target_proj(target)

        sub_slots = self._expanded_sub_slots(batch_size, target.device, target.dtype)
        target_context, target_attn = self.target_cross_attn(
            query=sub_slots,
            key=target_tokens,
            value=target_tokens,
            key_padding_mask=target_key_padding,
            need_weights=True,
            average_attn_weights=False,
        )
        target_conditioned_slots = self.target_slot_norm(sub_slots + target_context)

        if self.slot_self_attn is not None and self.slot_self_norm is not None:
            slot_context, _ = self.slot_self_attn(
                target_conditioned_slots,
                target_conditioned_slots,
                target_conditioned_slots,
                need_weights=False,
            )
            target_conditioned_slots = self.slot_self_norm(target_conditioned_slots + slot_context)

        visual_keys = self.visual_key_proj(visual)
        visual_values = self.visual_value_proj(visual)
        visual_context, visual_attn = self.visual_cross_attn(
            query=target_conditioned_slots,
            key=visual_keys,
            value=visual_values,
            key_padding_mask=visual_key_padding,
            need_weights=True,
            average_attn_weights=False,
        )
        visual_context = self.visual_out_proj(visual_context)
        visual_context = self.visual_slot_norm(visual_context)

        grouped = visual_context.view(
            batch_size,
            self.num_fvt_tokens,
            self.num_sub_slots,
            self.d_v,
        )
        fvt = self.group_merger(grouped.reshape(batch_size, self.num_fvt_tokens, -1))

        if self.use_query_residual and self.query_residual_proj is not None:
            query_groups = target_conditioned_slots.view(
                batch_size,
                self.num_fvt_tokens,
                self.num_sub_slots,
                self.slot_dim,
            ).mean(dim=2)
            fvt = fvt + self.query_residual_gate * self.query_residual_proj(query_groups)

        if self.use_fvt_calibration and self.fvt_calibration_norm is not None:
            fvt = self.fvt_calibration_norm(fvt) * self.fvt_scale + self.fvt_bias

        emit_attention = self.return_attention if return_attention is None else return_attention
        target_attn_debug = target_attn.detach()
        visual_attn_debug = visual_attn.detach()
        attention_debug: dict[str, torch.Tensor] = {}
        if emit_attention:
            attention_debug = {
                "target_attn_weights": target_attn_debug,
                "visual_attn_weights": visual_attn_debug,
                "target_attn_entropy": _attention_entropy(target_attn_debug),
                "visual_attn_entropy": _attention_entropy(visual_attn_debug),
                "target_topk_mass": _attention_topk_mass(target_attn_debug, ks=(1, 5, 10)),
                "visual_topk_mass": _attention_topk_mass(visual_attn_debug, ks=(1, 5, 10)),
            }
        attention_debug["attention_weights"] = visual_attn_debug.mean(dim=1)
        attention_debug["sub_slot_attention_weights"] = visual_attn_debug.mean(dim=1).view(
            batch_size,
            self.num_fvt_tokens,
            self.num_sub_slots,
            visual.shape[1],
        )

        output_fvt = fvt.squeeze(0) if unbatched else fvt
        if unbatched:
            attention_debug = {
                key: value.squeeze(0) if isinstance(value, torch.Tensor) and value.shape[:1] == (1,) else value
                for key, value in attention_debug.items()
            }

        metadata_out = {
            "variant_name": self.variant_name,
            "target_hidden_shape": list(target_hidden_states.shape),
            "pre_merge_visual_shape": list(pre_merge_visual_tokens.shape),
            "num_foveated_tokens": self.num_fvt_tokens,
            "spatial_merge_size": self.spatial_merge_size,
            "sub_slots_per_fvt": self.num_sub_slots,
            "total_sub_slots": self.total_sub_slots,
            "slot_dim": self.slot_dim,
            "num_heads": self.num_heads,
            "output_shape": list(output_fvt.shape),
            "foveated_visual_tokens_shape": list(output_fvt.shape),
            "attention_shape": list(attention_debug["attention_weights"].shape),
            "uses_single_q_global": False,
            "direct_query_residual": bool(self.use_query_residual),
            "use_target_self_attn": bool(self.use_target_self_attn),
            "use_slot_self_attn": bool(self.use_slot_self_attn),
            "use_fvt_calibration": bool(self.use_fvt_calibration),
            "merger_initialization": "random",
            **(metadata or {}),
        }
        return FovealCrossAttentionOutput(
            foveated_visual_tokens=output_fvt,
            attention_debug=attention_debug,
            debug_metadata=metadata_out,
        )

    def _expanded_sub_slots(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        base = self.sub_slot_embed.view(self.num_fvt_tokens, self.num_sub_slots, self.slot_dim)
        base = base + self.fvt_index_embed[:, None, :] + self.sub_index_embed[None, :, :]
        flat = base.reshape(self.total_sub_slots, self.slot_dim)
        return flat.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)

    def _normalize_inputs(
        self,
        target_hidden_states: torch.Tensor,
        pre_merge_visual_tokens: torch.Tensor,
        target_attention_mask: torch.Tensor | None,
        visual_attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, bool]:
        if target_hidden_states.ndim == 2:
            target = target_hidden_states.unsqueeze(0)
            unbatched = True
        elif target_hidden_states.ndim == 3:
            target = target_hidden_states
            unbatched = False
        else:
            raise ValueError("target_hidden_states must have shape [T, d_lm] or [B, T, d_lm]")
        if pre_merge_visual_tokens.ndim == 2:
            visual = pre_merge_visual_tokens.unsqueeze(0)
            visual_unbatched = True
        elif pre_merge_visual_tokens.ndim == 3:
            visual = pre_merge_visual_tokens
            visual_unbatched = False
        else:
            raise ValueError("pre_merge_visual_tokens must have shape [N, d_v] or [B, N, d_v]")
        if unbatched != visual_unbatched:
            raise ValueError("target and visual inputs must both be batched or both be unbatched")
        if target.shape[0] != visual.shape[0]:
            raise ValueError("target and visual batch sizes must match")
        if target.shape[-1] != self.d_lm:
            raise ValueError(f"target hidden dim {target.shape[-1]} does not match d_lm={self.d_lm}")
        if visual.shape[-1] != self.d_v:
            raise ValueError(f"visual hidden dim {visual.shape[-1]} does not match d_v={self.d_v}")
        if target.shape[1] == 0 or visual.shape[1] == 0:
            raise ValueError("target and visual token sequences must be non-empty")
        target_mask = _normalize_attention_mask(target_attention_mask, target.shape[:2], target.device)
        visual_mask = _normalize_attention_mask(visual_attention_mask, visual.shape[:2], visual.device)
        return target, visual, target_mask, visual_mask, unbatched


class TGVFv2VPTGating(nn.Module):
    """VPT-style target conditioning before the frozen Qwen visual merger."""

    variant_name = "tgvf_v2_vpt_gating"

    def __init__(
        self,
        *,
        d_lm: int,
        d_v: int,
        spatial_merge_size: int = 2,
    ) -> None:
        super().__init__()
        self.d_lm = d_lm
        self.d_v = d_v
        self.spatial_merge_size = int(spatial_merge_size)
        self.target_norm = nn.LayerNorm(d_lm)
        self.target_to_condition = nn.Linear(d_lm, d_v)
        self.visual_norm = nn.LayerNorm(d_v)

    def forward(
        self,
        *,
        target_hidden_states: torch.Tensor,
        pre_merge_visual_tokens: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> FovealCrossAttentionOutput:
        _validate_inputs(target_hidden_states, pre_merge_visual_tokens)
        target_hidden_states = target_hidden_states.to(dtype=self.target_norm.weight.dtype)
        pre_merge_visual_tokens = pre_merge_visual_tokens.to(dtype=self.visual_norm.weight.dtype)
        target_condition = self.target_to_condition(
            self.target_norm(target_hidden_states).mean(dim=0)
        )
        visual_base = self.visual_norm(pre_merge_visual_tokens)
        scores = visual_base @ target_condition
        delta = scores.unsqueeze(-1) * target_condition.unsqueeze(0)
        conditioned_visual_tokens = pre_merge_visual_tokens + delta
        visual_salience = F.softmax(delta.float().norm(dim=-1), dim=-1).to(delta.dtype).unsqueeze(0)
        return FovealCrossAttentionOutput(
            foveated_visual_tokens=conditioned_visual_tokens,
            attention_debug={
                "attention_weights": visual_salience,
                "condition_scores": scores.detach(),
            },
            debug_metadata=_metadata(
                self.variant_name,
                target_hidden_states,
                pre_merge_visual_tokens,
                conditioned_visual_tokens,
                visual_salience,
                metadata,
                tgvf_version="v2",
                output_token_count_source="pre_merge_visual_tokens_unmerged",
                conditioned_visual_tokens_shape=list(conditioned_visual_tokens.shape),
                final_fvt_requires_qwen_visual_merger=True,
                conditioning_stage="pre_qwen_visual_merger",
                visual_residual=True,
                target_conditioner="vpt_dot_product_residual",
                spatial_merge_size=self.spatial_merge_size,
            ),
            conditioned_pre_merge_visual_tokens=conditioned_visual_tokens,
        )


class TGVFv2CrossAttention(nn.Module):
    """Let each pre-merge visual token attend to target tokens before Qwen merger."""

    variant_name = "tgvf_v2_cross_attention"

    def __init__(
        self,
        *,
        d_lm: int,
        d_v: int,
        spatial_merge_size: int = 2,
        attn_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.d_lm = d_lm
        self.d_v = d_v
        self.attn_dim = attn_dim or d_v
        self.spatial_merge_size = int(spatial_merge_size)
        self.target_norm = nn.LayerNorm(d_lm)
        self.target_proj = nn.Linear(d_lm, self.attn_dim)
        self.visual_norm = nn.LayerNorm(d_v)
        self.visual_q_proj = nn.Linear(d_v, self.attn_dim)
        self.target_k_proj = nn.Linear(self.attn_dim, self.attn_dim)
        self.target_v_proj = nn.Linear(self.attn_dim, self.attn_dim)
        self.context_to_delta = nn.Linear(self.attn_dim, d_v)
        self.gate_proj = nn.Linear(d_v + self.attn_dim, d_v)

    def forward(
        self,
        *,
        target_hidden_states: torch.Tensor,
        pre_merge_visual_tokens: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> FovealCrossAttentionOutput:
        _validate_inputs(target_hidden_states, pre_merge_visual_tokens)
        target_hidden_states = target_hidden_states.to(dtype=self.target_norm.weight.dtype)
        pre_merge_visual_tokens = pre_merge_visual_tokens.to(dtype=self.visual_norm.weight.dtype)
        target_tokens = self.target_proj(self.target_norm(target_hidden_states))
        visual_tokens = self.visual_norm(pre_merge_visual_tokens)
        queries = self.visual_q_proj(visual_tokens)
        keys = self.target_k_proj(target_tokens)
        values = self.target_v_proj(target_tokens)
        target_context, target_attention = _cross_attention(queries, keys, values)
        delta = self.context_to_delta(target_context)
        gate = torch.sigmoid(self.gate_proj(torch.cat([visual_tokens, target_context], dim=-1)))
        gated_delta = gate * delta
        conditioned_visual_tokens = pre_merge_visual_tokens + gated_delta
        visual_salience = F.softmax(gated_delta.float().norm(dim=-1), dim=-1).to(
            gated_delta.dtype
        ).unsqueeze(0)
        return FovealCrossAttentionOutput(
            foveated_visual_tokens=conditioned_visual_tokens,
            attention_debug={
                "attention_weights": visual_salience,
                "visual_to_target_attention": target_attention.detach(),
            },
            debug_metadata=_metadata(
                self.variant_name,
                target_hidden_states,
                pre_merge_visual_tokens,
                conditioned_visual_tokens,
                visual_salience,
                metadata,
                tgvf_version="v2",
                output_token_count_source="pre_merge_visual_tokens_unmerged",
                conditioned_visual_tokens_shape=list(conditioned_visual_tokens.shape),
                final_fvt_requires_qwen_visual_merger=True,
                conditioning_stage="pre_qwen_visual_merger",
                visual_residual=True,
                target_conditioner="visual_queries_target_tokens",
                spatial_merge_size=self.spatial_merge_size,
                target_attention_shape=list(target_attention.shape),
            ),
            conditioned_pre_merge_visual_tokens=conditioned_visual_tokens,
        )


class TGVFv2Bidirectional(nn.Module):
    """Bidirectional target/vision conditioning before the frozen Qwen merger."""

    variant_name = "tgvf_v2_bidirectional"

    def __init__(
        self,
        *,
        d_lm: int,
        d_v: int,
        spatial_merge_size: int = 2,
        attn_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.d_lm = d_lm
        self.d_v = d_v
        self.attn_dim = attn_dim or d_v
        self.spatial_merge_size = int(spatial_merge_size)
        self.target_norm = nn.LayerNorm(d_lm)
        self.target_proj = nn.Linear(d_lm, self.attn_dim)
        self.visual_norm = nn.LayerNorm(d_v)
        self.visual_proj = nn.Linear(d_v, self.attn_dim)
        self.target_q_proj = nn.Linear(self.attn_dim, self.attn_dim)
        self.visual_k_proj = nn.Linear(self.attn_dim, self.attn_dim)
        self.visual_v_proj = nn.Linear(self.attn_dim, self.attn_dim)
        self.enriched_target_norm = nn.LayerNorm(self.attn_dim)
        self.visual_q_proj = nn.Linear(self.attn_dim, self.attn_dim)
        self.target_k_proj = nn.Linear(self.attn_dim, self.attn_dim)
        self.target_v_proj = nn.Linear(self.attn_dim, self.attn_dim)
        self.context_to_delta = nn.Linear(self.attn_dim, d_v)
        self.gate_proj = nn.Linear(d_v + self.attn_dim, d_v)

    def forward(
        self,
        *,
        target_hidden_states: torch.Tensor,
        pre_merge_visual_tokens: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> FovealCrossAttentionOutput:
        _validate_inputs(target_hidden_states, pre_merge_visual_tokens)
        target_hidden_states = target_hidden_states.to(dtype=self.target_norm.weight.dtype)
        pre_merge_visual_tokens = pre_merge_visual_tokens.to(dtype=self.visual_norm.weight.dtype)
        target_tokens = self.target_proj(self.target_norm(target_hidden_states))
        visual_tokens = self.visual_norm(pre_merge_visual_tokens)
        visual_projected = self.visual_proj(visual_tokens)
        target_context, target_to_visual_attention = _cross_attention(
            self.target_q_proj(target_tokens),
            self.visual_k_proj(visual_projected),
            self.visual_v_proj(visual_projected),
        )
        enriched_target = self.enriched_target_norm(target_tokens + target_context)
        visual_context, visual_to_target_attention = _cross_attention(
            self.visual_q_proj(visual_projected),
            self.target_k_proj(enriched_target),
            self.target_v_proj(enriched_target),
        )
        delta = self.context_to_delta(visual_context)
        gate = torch.sigmoid(self.gate_proj(torch.cat([visual_tokens, visual_context], dim=-1)))
        gated_delta = gate * delta
        conditioned_visual_tokens = pre_merge_visual_tokens + gated_delta
        visual_salience = F.softmax(gated_delta.float().norm(dim=-1), dim=-1).to(
            gated_delta.dtype
        ).unsqueeze(0)
        return FovealCrossAttentionOutput(
            foveated_visual_tokens=conditioned_visual_tokens,
            attention_debug={
                "attention_weights": visual_salience,
                "target_to_visual_attention": target_to_visual_attention.detach(),
                "visual_to_target_attention": visual_to_target_attention.detach(),
            },
            debug_metadata=_metadata(
                self.variant_name,
                target_hidden_states,
                pre_merge_visual_tokens,
                conditioned_visual_tokens,
                visual_salience,
                metadata,
                tgvf_version="v2",
                output_token_count_source="pre_merge_visual_tokens_unmerged",
                conditioned_visual_tokens_shape=list(conditioned_visual_tokens.shape),
                final_fvt_requires_qwen_visual_merger=True,
                conditioning_stage="pre_qwen_visual_merger",
                visual_residual=True,
                target_conditioner="bidirectional_target_visual_attention",
                spatial_merge_size=self.spatial_merge_size,
                target_to_visual_attention_shape=list(target_to_visual_attention.shape),
                visual_to_target_attention_shape=list(visual_to_target_attention.shape),
            ),
            conditioned_pre_merge_visual_tokens=conditioned_visual_tokens,
        )


def finalize_tgvf_output_with_frozen_qwen_merger(
    model: Any,
    output: FovealCrossAttentionOutput,
) -> FovealCrossAttentionOutput:
    """Convert v2 conditioned pre-merge tokens into LLM-side image tokens."""

    if not output.debug_metadata.get("final_fvt_requires_qwen_visual_merger"):
        return output

    conditioned = output.conditioned_pre_merge_visual_tokens
    if conditioned is None:
        conditioned = output.foveated_visual_tokens
    if conditioned.ndim != 2:
        raise ValueError("conditioned_pre_merge_visual_tokens must have shape [N, d_v]")

    spatial_merge_size = _infer_spatial_merge_size(model)
    spatial_merge_unit = spatial_merge_size**2
    if int(conditioned.shape[0]) % spatial_merge_unit != 0:
        raise AssertionError(
            "conditioned pre-merge token count must be divisible by "
            f"spatial_merge_size**2: {int(conditioned.shape[0])} vs {spatial_merge_unit}"
        )

    visual = _visual_module(model)
    if not hasattr(visual, "merger"):
        raise AttributeError("Qwen2-VL visual module does not expose a merger module")
    merger = visual.merger
    for parameter in merger.parameters():
        parameter.requires_grad_(False)
    merger_parameter = next(merger.parameters(), None)
    merger_dtype = merger_parameter.dtype if merger_parameter is not None else conditioned.dtype
    conditioned_for_merger = conditioned.to(dtype=merger_dtype)
    merged = merger(conditioned_for_merger)

    debug_metadata = dict(output.debug_metadata)
    debug_metadata.update(
        {
            "conditioned_pre_merge_visual_shape": list(conditioned_for_merger.shape),
            "foveated_visual_tokens_shape": list(merged.shape),
            "output_token_count_source": "frozen_qwen_visual_merger",
            "final_fvt_stage": "post_qwen_visual_merger",
            "qwen_visual_merger_frozen": not any(
                parameter.requires_grad for parameter in merger.parameters()
            ),
            "qwen_visual_merger_dtype": str(merger_dtype).replace("torch.", ""),
            "spatial_merge_size": int(spatial_merge_size),
            "spatial_merge_unit": int(spatial_merge_unit),
        }
    )
    return FovealCrossAttentionOutput(
        foveated_visual_tokens=merged,
        attention_debug=output.attention_debug,
        debug_metadata=debug_metadata,
        conditioned_pre_merge_visual_tokens=conditioned_for_merger,
    )


class Qwen2VLPreMergeVisualHook:
    """Capture Qwen2-VL visual features immediately before PatchMerger."""

    def __init__(self, model: Any) -> None:
        visual = _visual_module(model)
        if not hasattr(visual, "merger"):
            raise AttributeError("Qwen2-VL visual module does not expose a merger module")
        self.visual = visual
        self.pre_merge_visual_tokens: torch.Tensor | None = None
        self.handle = visual.merger.register_forward_pre_hook(self._capture)

    def _capture(self, _module: nn.Module, inputs: tuple[Any, ...]) -> None:
        if not inputs:
            return
        tokens = inputs[0]
        if isinstance(tokens, torch.Tensor):
            self.pre_merge_visual_tokens = tokens.detach()

    def close(self) -> None:
        self.handle.remove()

    def __enter__(self) -> Qwen2VLPreMergeVisualHook:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


@torch.no_grad()
def append_bracketed_fvt(
    *,
    model: Any,
    tokenizer_or_processor: Any,
    generation_state: Any,
    foveated_visual_tokens: torch.Tensor,
    device: torch.device | str | None = None,
) -> BracketedFVTAppendResult:
    """Append ``<|vision_start|> FVT* <|vision_end|>`` to the existing cache."""

    if foveated_visual_tokens.ndim != 2:
        raise ValueError("foveated_visual_tokens must have shape [M, d_lm]")
    if device is None:
        device = _infer_model_device(model) or foveated_visual_tokens.device

    tokenizer = _tokenizer(tokenizer_or_processor)
    token_ids = bracketed_fvt_token_ids(
        tokenizer_or_processor=tokenizer,
        model=model,
        num_fvt_tokens=foveated_visual_tokens.shape[0],
        device=device,
    )
    input_embeddings = model.get_input_embeddings()
    embeds = input_embeddings(token_ids.unsqueeze(0))
    fvt = foveated_visual_tokens.to(device=embeds.device, dtype=embeds.dtype)
    embeds[:, 1 : 1 + fvt.shape[0], :] = fvt.unsqueeze(0)

    attention_mask = _state_attr(generation_state, "attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
        attention_mask = _extend_attention(attention_mask, token_ids.shape[0])
    past_key_values = _state_attr(generation_state, "past_key_values")
    model_kwargs = dict(_state_attr(generation_state, "model_kwargs") or {})
    position_ids = _chunk_position_ids(
        attention_mask=attention_mask,
        chunk_length=token_ids.shape[0],
        rope_deltas=model_kwargs.get("rope_deltas"),
        device=embeds.device,
    )
    forward_kwargs: dict[str, Any] = {
        "inputs_embeds": embeds,
        "past_key_values": past_key_values,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "use_cache": True,
        "return_dict": True,
    }
    outputs = model(**{key: value for key, value in forward_kwargs.items() if value is not None})

    input_ids = _state_attr(generation_state, "input_ids")
    if input_ids is not None:
        input_ids = torch.cat([input_ids.to(device), token_ids.view(1, -1)], dim=-1)

    return BracketedFVTAppendResult(
        past_key_values=outputs.past_key_values,
        attention_mask=attention_mask,
        cache_position=None,
        input_ids=input_ids,
        last_logits=outputs.logits,
        appended_token_ids=token_ids.detach().cpu(),
        appended_inputs_embeds=embeds.detach().cpu(),
        fvt_token_start=1,
        fvt_token_end=1 + fvt.shape[0],
        model_kwargs=model_kwargs,
        debug_metadata={
            "append_token_count": int(token_ids.shape[0]),
            "num_fvt_tokens": int(fvt.shape[0]),
            "bracketed_append_used": True,
            "second_full_forward_used": False,
            "cache_preserved": past_key_values is not None,
            "appended_token_ids": token_ids.detach().cpu().tolist(),
            "foveated_visual_tokens_shape": list(foveated_visual_tokens.shape),
        },
    )


@torch.no_grad()
def append_text_instruction_to_state(
    *,
    model: Any,
    tokenizer_or_processor: Any,
    generation_state: Any,
    instruction: str,
    device: torch.device | str | None = None,
) -> BracketedFVTAppendResult:
    """Append textual continuation instruction to the existing KV cache."""

    if not instruction.strip():
        raise ValueError("instruction must be non-empty")
    if device is None:
        device = _infer_model_device(model) or torch.device("cpu")

    tokenizer = _tokenizer(tokenizer_or_processor)
    token_ids = torch.tensor(
        tokenizer.encode("\n\n" + instruction.strip(), add_special_tokens=False),
        dtype=torch.long,
        device=device,
    )
    if token_ids.numel() == 0:
        raise ValueError("instruction tokenized to an empty sequence")

    input_embeddings = model.get_input_embeddings()
    embeds = input_embeddings(token_ids.unsqueeze(0))
    attention_mask = _state_attr(generation_state, "attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
        attention_mask = _extend_attention(attention_mask, token_ids.shape[0])
    past_key_values = _state_attr(generation_state, "past_key_values")
    model_kwargs = dict(_state_attr(generation_state, "model_kwargs") or {})
    position_ids = _chunk_position_ids(
        attention_mask=attention_mask,
        chunk_length=token_ids.shape[0],
        rope_deltas=model_kwargs.get("rope_deltas"),
        device=embeds.device,
    )
    outputs = model(
        **{
            key: value
            for key, value in {
                "inputs_embeds": embeds,
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "use_cache": True,
                "return_dict": True,
            }.items()
            if value is not None
        }
    )

    input_ids = _state_attr(generation_state, "input_ids")
    if input_ids is not None:
        input_ids = torch.cat([input_ids.to(device), token_ids.view(1, -1)], dim=-1)

    previous_debug = dict(_state_attr(generation_state, "debug_metadata") or {})
    previous_debug.update(
        {
            "continuation_instruction_appended": True,
            "instruction_token_count": int(token_ids.shape[0]),
            "instruction_text": instruction.strip(),
            "second_full_forward_used": False,
            "cache_preserved_after_instruction": past_key_values is not None,
        }
    )
    return BracketedFVTAppendResult(
        past_key_values=outputs.past_key_values,
        attention_mask=attention_mask,
        cache_position=None,
        input_ids=input_ids,
        last_logits=outputs.logits,
        appended_token_ids=token_ids.detach().cpu(),
        appended_inputs_embeds=embeds.detach().cpu(),
        fvt_token_start=_state_attr(generation_state, "fvt_token_start") or -1,
        fvt_token_end=_state_attr(generation_state, "fvt_token_end") or -1,
        model_kwargs=model_kwargs,
        debug_metadata=previous_debug,
    )



def build_fvt_answer_instruction(
    *,
    target_text: str,
    benchmark_answer_format: str | None = None,
    option_letters: list[str] | tuple[str, ...] | None = None,
    original_question: str | None = None,
    option_texts: list[str] | tuple[str, ...] | None = None,
    repeat_options_in_continuation: bool = False,
) -> str:
    target = target_text.strip() or "the requested visual evidence"
    if repeat_options_in_continuation and option_texts:
        letters = _option_letters(option_texts, option_letters)
        options = "\n".join(
            f"{letter}. {text}" for letter, text in zip(letters, option_texts, strict=False)
        )
        question = (original_question or "the original question").strip()
        letter_text = ", ".join(letters)
        return (
            "The visual tokens above are focused evidence for the target:\n"
            f"{target}\n\n"
            "Answer the original multiple-choice question.\n\n"
            "Question:\n"
            f"{question}\n\n"
            "Options:\n"
            f"{options}\n\n"
            "Use the original image and the focused visual evidence.\n"
            f"Output exactly one letter: {letter_text}.\n"
            "Do not explain."
        )
    if option_letters:
        letters = ", ".join(str(letter) for letter in option_letters)
        return (
            "The visual tokens above are focused evidence for the target:\n"
            f"{target}\n\n"
            "Use this evidence to answer the original multiple-choice question.\n"
            f"Output exactly one letter: {letters}.\n"
            "Do not explain."
        )
    if benchmark_answer_format == "multiple_choice":
        return (
            "The visual tokens above are focused evidence for the target:\n"
            f"{target}\n\n"
            "Use this evidence to answer the original multiple-choice question.\n"
            "Output exactly one option letter.\n"
            "Do not explain."
        )
    return (
        "The visual tokens above are focused evidence for the target:\n"
        f"{target}\n\n"
        "Use this evidence to answer the user's original question.\n"
        "Do not mention the foveation process."
    )


def build_text_answer_instruction(
    *,
    benchmark_answer_format: str | None = None,
    option_letters: list[str] | tuple[str, ...] | None = None,
    original_question: str | None = None,
    option_texts: list[str] | tuple[str, ...] | None = None,
    repeat_options_in_continuation: bool = False,
) -> str:
    if repeat_options_in_continuation and option_texts:
        letters = _option_letters(option_texts, option_letters)
        options = "\n".join(
            f"{letter}. {text}" for letter, text in zip(letters, option_texts, strict=False)
        )
        question = (original_question or "the original question").strip()
        letter_text = ", ".join(letters)
        return (
            "Answer the original multiple-choice question.\n\n"
            "Question:\n"
            f"{question}\n\n"
            "Options:\n"
            f"{options}\n\n"
            "Use the original image.\n"
            f"Output exactly one letter: {letter_text}.\n"
            "Do not explain."
        )
    if option_letters:
        letters = ", ".join(str(letter) for letter in option_letters)
        return (
            "Now answer the original multiple-choice question.\n"
            f"Output exactly one letter: {letters}.\n"
            "Do not explain."
        )
    if benchmark_answer_format == "multiple_choice":
        return (
            "Now answer the original multiple-choice question.\n"
            "Output exactly one option letter.\n"
            "Do not explain."
        )
    return "Now answer the user's original question. Do not mention this continuation step."


def _option_letters(
    option_texts: list[str] | tuple[str, ...],
    option_letters: list[str] | tuple[str, ...] | None,
) -> list[str]:
    if option_letters:
        return [str(letter) for letter in option_letters]
    return [chr(ord("A") + index) for index in range(len(option_texts))]


@torch.no_grad()
def append_fvt_result_and_open_answer_turn(
    *,
    model: Any,
    tokenizer_or_processor: Any,
    generation_state: Any,
    foveated_visual_tokens: torch.Tensor,
    target_text: str,
    benchmark_answer_format: str | None = None,
    option_letters: list[str] | tuple[str, ...] | None = None,
    original_question: str | None = None,
    option_texts: list[str] | tuple[str, ...] | None = None,
    repeat_options_in_continuation: bool = False,
    append_as_new_user_turn: bool = True,
    instruction_text: str | None = None,
    fvt_append_mode: str = "qwen_native_pseudo_image",
    image_grid_thw: torch.Tensor | None = None,
    device: torch.device | str | None = None,
) -> BracketedFVTAppendResult:
    validate_fvt_append_mode(fvt_append_mode)
    if foveated_visual_tokens.ndim != 2:
        raise ValueError("foveated_visual_tokens must have shape [M, d_lm]")
    if device is None:
        device = _infer_model_device(model) or foveated_visual_tokens.device

    tokenizer = _tokenizer(tokenizer_or_processor)
    instruction = (
        instruction_text.strip()
        if instruction_text is not None
        else build_fvt_answer_instruction(
            target_text=target_text,
            benchmark_answer_format=benchmark_answer_format,
            option_letters=option_letters,
            original_question=original_question,
            option_texts=option_texts,
            repeat_options_in_continuation=repeat_options_in_continuation,
        )
    )
    prefix_text = "<|im_end|>\n<|im_start|>user\n" if append_as_new_user_turn else "\n"
    suffix_text = (
        "\n" + instruction.strip() + "\n<|im_end|>\n<|im_start|>assistant\n"
        if append_as_new_user_turn
        else "\n" + instruction.strip()
    )
    prefix_ids = _encode_text(tokenizer, prefix_text, device)
    fvt_ids = bracketed_fvt_token_ids(
        tokenizer_or_processor=tokenizer,
        model=model,
        num_fvt_tokens=foveated_visual_tokens.shape[0],
        device=device,
    )
    suffix_ids = _encode_text(tokenizer, suffix_text, device)
    token_ids = torch.cat([prefix_ids, fvt_ids, suffix_ids], dim=0)

    input_embeddings = model.get_input_embeddings()
    embeds = input_embeddings(token_ids.unsqueeze(0))
    fvt = foveated_visual_tokens.to(device=embeds.device, dtype=embeds.dtype)
    fvt_start = int(prefix_ids.shape[0]) + 1
    fvt_end = fvt_start + int(fvt.shape[0])
    embeds[:, fvt_start:fvt_end, :] = fvt.unsqueeze(0)

    attention_mask = _state_attr(generation_state, "attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
        attention_mask = _extend_attention(attention_mask, token_ids.shape[0])
    past_key_values = _state_attr(generation_state, "past_key_values")
    model_kwargs = dict(_state_attr(generation_state, "model_kwargs") or {})

    image_token_id = int(fvt_ids[1].detach().cpu().item())
    actual_image_pad_token_count = int((token_ids == image_token_id).sum().detach().cpu().item())
    spatial_merge_size = _infer_spatial_merge_size(model)
    fake_image_grid_thw: torch.Tensor | None = None
    source_image_grid_thw: torch.Tensor | None = None
    fvt_grid_thw: torch.Tensor | None = None
    fvt_grid_source = "none"
    expected_llm_image_tokens: int | None = None
    mm_token_type_ids: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None
    position_ids_source = "text_style_chunk"
    visual_tower_called_for_fvt = False

    if fvt_append_mode == "qwen_native_pseudo_image":
        if image_grid_thw is None:
            fvt_grid_thw = make_fake_image_grid(
                num_fvt_tokens=int(fvt.shape[0]),
                spatial_merge_size=spatial_merge_size,
                device=embeds.device,
            )
            fake_image_grid_thw = fvt_grid_thw
            fvt_grid_source = "near_square_fake_grid"
        else:
            fvt_grid_thw = image_grid_thw.to(device=embeds.device, dtype=torch.long)
            source_image_grid_thw = fvt_grid_thw
            fvt_grid_source = "source_image_grid_thw"
        expected_llm_image_tokens = _expected_llm_image_tokens(
            fvt_grid_thw,
            spatial_merge_size=spatial_merge_size,
        )
        if expected_llm_image_tokens != int(fvt.shape[0]):
            raise AssertionError(
                f"expected_llm_image_tokens={expected_llm_image_tokens} != num_fvt_tokens={int(fvt.shape[0])}"
            )
        mm_token_type_ids = _fvt_mm_token_type_ids(
            chunk_length=int(token_ids.shape[0]),
            fvt_token_start=fvt_start,
            fvt_token_end=fvt_end,
            device=embeds.device,
        )
        position_ids = _fvt_chunk_position_ids(
            attention_mask=attention_mask,
            chunk_length=int(token_ids.shape[0]),
            fvt_token_start=fvt_start,
            fvt_token_end=fvt_end,
            fvt_grid_thw=fvt_grid_thw,
            spatial_merge_size=spatial_merge_size,
            rope_deltas=model_kwargs.get("rope_deltas"),
            device=embeds.device,
        )
        position_ids_source = (
            "source_image_grid_mrope"
            if fvt_grid_source == "source_image_grid_thw"
            else "manual_fake_grid_mrope"
        )
    else:
        position_ids = _chunk_position_ids(
            attention_mask=attention_mask,
            chunk_length=token_ids.shape[0],
            rope_deltas=model_kwargs.get("rope_deltas"),
            device=embeds.device,
        )

    mm_token_type_ids_present = mm_token_type_ids is not None
    image_pad_mm_type_is_image = bool(
        mm_token_type_ids is not None
        and torch.all(mm_token_type_ids[:, fvt_start:fvt_end] == 1).detach().cpu().item()
    )
    image_position_ids_are_3d = _position_ids_have_3d_image_span(position_ids, fvt_start, fvt_end)
    text_position_ids_are_1d = _text_positions_are_1d(position_ids, fvt_start, fvt_end)

    if fvt_append_mode == "qwen_native_pseudo_image":
        if actual_image_pad_token_count != int(fvt.shape[0]):
            raise AssertionError(
                f"actual_image_pad_token_count={actual_image_pad_token_count} != num_fvt_tokens={int(fvt.shape[0])}"
            )
        if not image_pad_mm_type_is_image:
            raise AssertionError("FVT image_pad tokens were not marked as image modality")
        if visual_tower_called_for_fvt:
            raise AssertionError("visual tower must not be called for appended FVT")
        if not image_position_ids_are_3d:
            raise AssertionError("FVT image_pad positions were not encoded as 3D image positions")

    before_logits = _state_attr(generation_state, "last_logits")
    before_input_ids = _state_attr(generation_state, "input_ids")
    forward_kwargs: dict[str, Any] = {
        "inputs_embeds": embeds,
        "past_key_values": past_key_values,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "use_cache": True,
        "return_dict": True,
    }
    if fvt_append_mode == "qwen_native_pseudo_image":
        forward_kwargs.update(
            {
                "input_ids": token_ids.view(1, -1),
                "image_grid_thw": fvt_grid_thw,
                "mm_token_type_ids": mm_token_type_ids,
            }
        )
    outputs = model(**{key: value for key, value in forward_kwargs.items() if value is not None})

    updated_model_kwargs = dict(model_kwargs)
    updated_rope_delta = _rope_delta_for_next_token(position_ids, attention_mask)
    if updated_rope_delta is not None:
        updated_model_kwargs["rope_deltas"] = updated_rope_delta

    input_ids = before_input_ids
    if input_ids is not None:
        input_ids = torch.cat([input_ids.to(device), token_ids.view(1, -1)], dim=-1)

    context_text_without_fvt = (
        prefix_text
        + "<|vision_start|>"
        + f"<FVT:{int(fvt.shape[0])} tokens>"
        + "<|vision_end|>"
        + suffix_text
    )
    used_logits_source = "after_append"
    second_full_forward_used = False
    if used_logits_source != "after_append":
        raise AssertionError("FVT append must continue from after-append logits")
    if second_full_forward_used:
        raise AssertionError("FVT append must not perform a second full forward")

    previous_debug = dict(_state_attr(generation_state, "debug_metadata") or {})
    previous_debug.update(
        {
            "append_mode": fvt_append_mode,
            "turn_append_mode": "new_user_turn" if append_as_new_user_turn else "same_turn",
            "append_token_count": int(token_ids.shape[0]),
            "num_fvt_tokens": int(fvt.shape[0]),
            "foveated_visual_tokens_shape": list(foveated_visual_tokens.shape),
            "bracketed_append_used": True,
            "fvt_token_start": fvt_start,
            "fvt_token_end": fvt_end,
            "fvt_position_mode": position_ids_source,
            "fake_image_grid_thw": None if fake_image_grid_thw is None else fake_image_grid_thw.detach().cpu().tolist(),
            "source_image_grid_thw": (
                None
                if source_image_grid_thw is None
                else source_image_grid_thw.detach().cpu().tolist()
            ),
            "fvt_grid_thw": None if fvt_grid_thw is None else fvt_grid_thw.detach().cpu().tolist(),
            "fvt_grid_source": fvt_grid_source,
            "spatial_merge_size": int(spatial_merge_size),
            "fvt_spatial_merge_size": int(spatial_merge_size),
            "expected_llm_image_tokens": expected_llm_image_tokens,
            "actual_image_pad_token_count": actual_image_pad_token_count,
            "mm_token_type_ids_present": bool(mm_token_type_ids_present),
            "image_pad_mm_type_is_image": bool(image_pad_mm_type_is_image),
            "mm_token_type_ids_fvt_count": (
                0 if mm_token_type_ids is None else int(mm_token_type_ids.sum().detach().cpu().item())
            ),
            "position_ids_source": position_ids_source,
            "image_position_ids_are_3d": bool(image_position_ids_are_3d),
            "text_position_ids_are_1d": bool(text_position_ids_are_1d),
            "visual_tower_called_for_fvt": bool(visual_tower_called_for_fvt),
            "updated_rope_delta_after_append": (
                None if updated_rope_delta is None else updated_rope_delta.detach().cpu().tolist()
            ),
            "continuation_instruction_appended": True,
            "repeat_options_in_continuation": bool(repeat_options_in_continuation),
            "instruction_token_count": int(suffix_ids.shape[0]),
            "instruction_text": instruction.strip(),
            "appended_context_text_without_FVT": context_text_without_fvt,
            "final_assistant_prefix_appended": bool(append_as_new_user_turn),
            "appended_context_contains_im_end": "<|im_end|>" in context_text_without_fvt,
            "stop_checked_on_appended_context": False,
            "used_logits_source": used_logits_source,
            "top10_logits_before_append": _topk_logits(tokenizer, before_logits),
            "top10_logits_after_append": _topk_logits(tokenizer, outputs.logits),
            "last_token_before_append": _last_token_debug(tokenizer, before_input_ids),
            "last_token_after_append": _token_debug(tokenizer, int(token_ids[-1].detach().cpu().item())),
            "second_full_forward_used": second_full_forward_used,
            "cache_preserved": past_key_values is not None and outputs.past_key_values is not None,
        }
    )
    return BracketedFVTAppendResult(
        past_key_values=outputs.past_key_values,
        attention_mask=attention_mask,
        cache_position=None,
        input_ids=input_ids,
        last_logits=outputs.logits,
        appended_token_ids=token_ids.detach().cpu(),
        appended_inputs_embeds=embeds.detach().cpu(),
        fvt_token_start=fvt_start,
        fvt_token_end=fvt_end,
        model_kwargs=updated_model_kwargs,
        debug_metadata=previous_debug,
    )


@torch.no_grad()
def prefill_fvt_first_pass_answer_turn(
    *,
    model: Any,
    tokenizer_or_processor: Any,
    foveated_visual_tokens: torch.Tensor,
    answer_prompt: str,
    fvt_append_mode: str = "qwen_native_pseudo_image",
    image_grid_thw: torch.Tensor | None = None,
    device: torch.device | str | None = None,
) -> BracketedFVTAppendResult:
    """Prefill a fresh context containing only a pseudo-image FVT and the answer prompt."""

    validate_fvt_append_mode(fvt_append_mode)
    if foveated_visual_tokens.ndim != 2:
        raise ValueError("foveated_visual_tokens must have shape [M, d_lm]")
    if device is None:
        device = _infer_model_device(model) or foveated_visual_tokens.device

    tokenizer = _tokenizer(tokenizer_or_processor)
    prefix_text = "<|im_start|>user\n"
    suffix_text = f"\n{answer_prompt.strip()}\n<|im_end|>\n<|im_start|>assistant\n"
    prefix_ids = _encode_text(tokenizer, prefix_text, device)
    fvt_ids = bracketed_fvt_token_ids(
        tokenizer_or_processor=tokenizer,
        model=model,
        num_fvt_tokens=foveated_visual_tokens.shape[0],
        device=device,
    )
    suffix_ids = _encode_text(tokenizer, suffix_text, device)
    token_ids = torch.cat([prefix_ids, fvt_ids, suffix_ids], dim=0)

    input_embeddings = model.get_input_embeddings()
    embeds = input_embeddings(token_ids.unsqueeze(0))
    fvt = foveated_visual_tokens.to(device=embeds.device, dtype=embeds.dtype)
    fvt_start = int(prefix_ids.shape[0]) + 1
    fvt_end = fvt_start + int(fvt.shape[0])
    embeds[:, fvt_start:fvt_end, :] = fvt.unsqueeze(0)

    attention_mask = torch.ones((1, int(token_ids.shape[0])), dtype=torch.long, device=device)
    image_token_id = int(fvt_ids[1].detach().cpu().item())
    actual_image_pad_token_count = int((token_ids == image_token_id).sum().detach().cpu().item())
    spatial_merge_size = _infer_spatial_merge_size(model)
    fake_image_grid_thw: torch.Tensor | None = None
    source_image_grid_thw: torch.Tensor | None = None
    fvt_grid_thw: torch.Tensor | None = None
    fvt_grid_source = "none"
    expected_llm_image_tokens: int | None = None
    mm_token_type_ids: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None
    position_ids_source = "text_style_full_context"

    if fvt_append_mode == "qwen_native_pseudo_image":
        if image_grid_thw is None:
            fvt_grid_thw = make_fake_image_grid(
                num_fvt_tokens=int(fvt.shape[0]),
                spatial_merge_size=spatial_merge_size,
                device=embeds.device,
            )
            fake_image_grid_thw = fvt_grid_thw
            fvt_grid_source = "near_square_fake_grid"
        else:
            fvt_grid_thw = image_grid_thw.to(device=embeds.device, dtype=torch.long)
            source_image_grid_thw = fvt_grid_thw
            fvt_grid_source = "source_image_grid_thw"
        expected_llm_image_tokens = _expected_llm_image_tokens(
            fvt_grid_thw,
            spatial_merge_size=spatial_merge_size,
        )
        if expected_llm_image_tokens != int(fvt.shape[0]):
            raise AssertionError(
                f"expected_llm_image_tokens={expected_llm_image_tokens} "
                f"!= num_fvt_tokens={int(fvt.shape[0])}"
            )
        mm_token_type_ids = _fvt_mm_token_type_ids(
            chunk_length=int(token_ids.shape[0]),
            fvt_token_start=fvt_start,
            fvt_token_end=fvt_end,
            device=embeds.device,
        )
        position_ids = _fvt_chunk_position_ids(
            attention_mask=attention_mask,
            chunk_length=int(token_ids.shape[0]),
            fvt_token_start=fvt_start,
            fvt_token_end=fvt_end,
            fvt_grid_thw=fvt_grid_thw,
            spatial_merge_size=spatial_merge_size,
            rope_deltas=None,
            device=embeds.device,
        )
        position_ids_source = (
            "source_image_grid_mrope"
            if fvt_grid_source == "source_image_grid_thw"
            else "manual_fake_grid_mrope"
        )
    else:
        position_ids = None

    mm_token_type_ids_present = mm_token_type_ids is not None
    image_pad_mm_type_is_image = bool(
        mm_token_type_ids is not None
        and torch.all(mm_token_type_ids[:, fvt_start:fvt_end] == 1).detach().cpu().item()
    )
    image_position_ids_are_3d = _position_ids_have_3d_image_span(position_ids, fvt_start, fvt_end)
    text_position_ids_are_1d = _text_positions_are_1d(position_ids, fvt_start, fvt_end)

    if fvt_append_mode == "qwen_native_pseudo_image":
        if actual_image_pad_token_count != int(fvt.shape[0]):
            raise AssertionError(
                f"actual_image_pad_token_count={actual_image_pad_token_count} "
                f"!= num_fvt_tokens={int(fvt.shape[0])}"
            )
        if not image_pad_mm_type_is_image:
            raise AssertionError("FVT image_pad tokens were not marked as image modality")
        if not image_position_ids_are_3d:
            raise AssertionError("FVT image_pad positions were not encoded as 3D image positions")

    forward_kwargs: dict[str, Any] = {
        "inputs_embeds": embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "use_cache": True,
        "return_dict": True,
    }
    if fvt_append_mode == "qwen_native_pseudo_image":
        forward_kwargs.update(
            {
                "input_ids": token_ids.view(1, -1),
                "image_grid_thw": fvt_grid_thw,
                "mm_token_type_ids": mm_token_type_ids,
            }
        )
    outputs = model(**{key: value for key, value in forward_kwargs.items() if value is not None})
    updated_rope_delta = _rope_delta_for_next_token(position_ids, attention_mask)
    model_kwargs = {}
    if updated_rope_delta is not None:
        model_kwargs["rope_deltas"] = updated_rope_delta

    context_text_without_fvt = (
        prefix_text
        + "<|vision_start|>"
        + f"<FVT:{int(fvt.shape[0])} tokens>"
        + "<|vision_end|>"
        + suffix_text
    )
    debug_metadata = {
        "append_mode": fvt_append_mode,
        "turn_append_mode": "fresh_first_user_turn",
        "fresh_context_first_pass": True,
        "capture_cache_cleared_before_answer": True,
        "initial_past_key_values_supplied": False,
        "append_token_count": int(token_ids.shape[0]),
        "num_fvt_tokens": int(fvt.shape[0]),
        "foveated_visual_tokens_shape": list(foveated_visual_tokens.shape),
        "bracketed_append_used": True,
        "fvt_token_start": fvt_start,
        "fvt_token_end": fvt_end,
        "fvt_position_mode": position_ids_source,
        "fake_image_grid_thw": None if fake_image_grid_thw is None else fake_image_grid_thw.detach().cpu().tolist(),
        "source_image_grid_thw": (
            None
            if source_image_grid_thw is None
            else source_image_grid_thw.detach().cpu().tolist()
        ),
        "fvt_grid_thw": None if fvt_grid_thw is None else fvt_grid_thw.detach().cpu().tolist(),
        "fvt_grid_source": fvt_grid_source,
        "spatial_merge_size": int(spatial_merge_size),
        "fvt_spatial_merge_size": int(spatial_merge_size),
        "expected_llm_image_tokens": expected_llm_image_tokens,
        "actual_image_pad_token_count": actual_image_pad_token_count,
        "mm_token_type_ids_present": bool(mm_token_type_ids_present),
        "image_pad_mm_type_is_image": bool(image_pad_mm_type_is_image),
        "mm_token_type_ids_fvt_count": (
            0 if mm_token_type_ids is None else int(mm_token_type_ids.sum().detach().cpu().item())
        ),
        "position_ids_source": position_ids_source,
        "image_position_ids_are_3d": bool(image_position_ids_are_3d),
        "text_position_ids_are_1d": bool(text_position_ids_are_1d),
        "visual_tower_called_for_fvt": False,
        "updated_rope_delta_after_append": (
            None if updated_rope_delta is None else updated_rope_delta.detach().cpu().tolist()
        ),
        "continuation_instruction_appended": False,
        "repeat_options_in_continuation": False,
        "instruction_text": answer_prompt.strip(),
        "appended_context_text_without_FVT": context_text_without_fvt,
        "final_assistant_prefix_appended": True,
        "used_logits_source": "fresh_first_pass_after_fvt_question",
        "top10_logits_after_append": _topk_logits(tokenizer, outputs.logits),
        "last_token_after_append": _token_debug(tokenizer, int(token_ids[-1].detach().cpu().item())),
        "second_full_forward_used": True,
        "cache_preserved": False,
    }
    return BracketedFVTAppendResult(
        past_key_values=outputs.past_key_values,
        attention_mask=attention_mask,
        cache_position=None,
        input_ids=token_ids.view(1, -1),
        last_logits=outputs.logits,
        appended_token_ids=token_ids.detach().cpu(),
        appended_inputs_embeds=embeds.detach().cpu(),
        fvt_token_start=fvt_start,
        fvt_token_end=fvt_end,
        model_kwargs=model_kwargs,
        debug_metadata=debug_metadata,
    )


@torch.no_grad()
def append_answer_turn_and_open(
    *,
    model: Any,
    tokenizer_or_processor: Any,
    generation_state: Any,
    benchmark_answer_format: str | None = None,
    option_letters: list[str] | tuple[str, ...] | None = None,
    original_question: str | None = None,
    option_texts: list[str] | tuple[str, ...] | None = None,
    repeat_options_in_continuation: bool = False,
    append_as_new_user_turn: bool = True,
    instruction_text: str | None = None,
    device: torch.device | str | None = None,
) -> BracketedFVTAppendResult:
    """Append a text-only answer turn incrementally and reopen assistant generation."""

    if device is None:
        device = _infer_model_device(model) or torch.device("cpu")

    tokenizer = _tokenizer(tokenizer_or_processor)
    instruction = (
        instruction_text.strip()
        if instruction_text is not None
        else build_text_answer_instruction(
            benchmark_answer_format=benchmark_answer_format,
            option_letters=option_letters,
            original_question=original_question,
            option_texts=option_texts,
            repeat_options_in_continuation=repeat_options_in_continuation,
        )
    )
    prefix_text = "<|im_end|>\n<|im_start|>user\n" if append_as_new_user_turn else "\n"
    suffix_text = "\n<|im_end|>\n<|im_start|>assistant\n" if append_as_new_user_turn else ""
    context_text = prefix_text + instruction.strip() + suffix_text
    token_ids = _encode_text(tokenizer, context_text, device)
    if token_ids.numel() == 0:
        raise ValueError("answer continuation tokenized to an empty sequence")

    input_embeddings = model.get_input_embeddings()
    embeds = input_embeddings(token_ids.unsqueeze(0))
    attention_mask = _state_attr(generation_state, "attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
        attention_mask = _extend_attention(attention_mask, token_ids.shape[0])
    past_key_values = _state_attr(generation_state, "past_key_values")
    model_kwargs = dict(_state_attr(generation_state, "model_kwargs") or {})
    position_ids = _chunk_position_ids(
        attention_mask=attention_mask,
        chunk_length=token_ids.shape[0],
        rope_deltas=model_kwargs.get("rope_deltas"),
        device=embeds.device,
    )

    before_logits = _state_attr(generation_state, "last_logits")
    before_input_ids = _state_attr(generation_state, "input_ids")
    outputs = model(
        **{
            key: value
            for key, value in {
                "inputs_embeds": embeds,
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "use_cache": True,
                "return_dict": True,
            }.items()
            if value is not None
        }
    )

    input_ids = before_input_ids
    if input_ids is not None:
        input_ids = torch.cat([input_ids.to(device), token_ids.view(1, -1)], dim=-1)

    previous_debug = dict(_state_attr(generation_state, "debug_metadata") or {})
    previous_debug.update(
        {
            "append_mode": "new_user_turn" if append_as_new_user_turn else "same_turn",
            "append_token_count": int(token_ids.shape[0]),
            "num_fvt_tokens": 0,
            "foveated_visual_tokens_shape": None,
            "bracketed_append_used": False,
            "fvt_token_start": -1,
            "fvt_token_end": -1,
            "continuation_instruction_appended": True,
            "repeat_options_in_continuation": bool(repeat_options_in_continuation),
            "instruction_token_count": int(token_ids.shape[0]),
            "instruction_text": instruction.strip(),
            "appended_context_text_without_FVT": context_text,
            "final_assistant_prefix_appended": bool(append_as_new_user_turn),
            "appended_context_contains_im_end": "<|im_end|>" in context_text,
            "stop_checked_on_appended_context": False,
            "used_logits_source": "after_append",
            "top10_logits_before_append": _topk_logits(tokenizer, before_logits),
            "top10_logits_after_append": _topk_logits(tokenizer, outputs.logits),
            "last_token_before_append": _last_token_debug(tokenizer, before_input_ids),
            "last_token_after_append": _token_debug(tokenizer, int(token_ids[-1].detach().cpu().item())),
            "second_full_forward_used": False,
            "cache_preserved": past_key_values is not None and outputs.past_key_values is not None,
        }
    )
    return BracketedFVTAppendResult(
        past_key_values=outputs.past_key_values,
        attention_mask=attention_mask,
        cache_position=None,
        input_ids=input_ids,
        last_logits=outputs.logits,
        appended_token_ids=token_ids.detach().cpu(),
        appended_inputs_embeds=embeds.detach().cpu(),
        fvt_token_start=-1,
        fvt_token_end=-1,
        model_kwargs=model_kwargs,
        debug_metadata=previous_debug,
    )


@torch.no_grad()
def continue_generation_from_state(
    *,
    model: Any,
    tokenizer_or_processor: Any,
    generation_state: BracketedFVTAppendResult,
    max_new_tokens: int = 32,
    eos_token_id: int | None = None,
    suppress_first_token_ids: list[int] | tuple[int, ...] | None = None,
) -> ContinuationResult:
    tokenizer = _tokenizer(tokenizer_or_processor)
    logits = generation_state.last_logits
    past_key_values = generation_state.past_key_values
    attention_mask = generation_state.attention_mask
    input_ids = generation_state.input_ids
    generated_ids: list[int] = []
    stop_reason = "max_new_tokens"
    device = (
        logits.device if logits is not None else (_infer_model_device(model) or torch.device("cpu"))
    )

    suppress_first = [int(token_id) for token_id in (suppress_first_token_ids or [])]
    for _ in range(max_new_tokens):
        step_logits = logits
        if not generated_ids and suppress_first:
            step_logits = logits.clone()
            valid_ids = [token_id for token_id in suppress_first if 0 <= token_id < step_logits.shape[-1]]
            if valid_ids:
                step_logits[:, -1, valid_ids] = -1e4
        next_token = torch.argmax(step_logits[:, -1, :], dim=-1, keepdim=True)
        token_id = int(next_token[0, 0].item())
        generated_ids.append(token_id)
        if input_ids is not None:
            input_ids = torch.cat([input_ids.to(device), next_token.to(device)], dim=-1)
        if attention_mask is not None:
            attention_mask = _extend_attention(attention_mask.to(device), 1)
        position_ids = _chunk_position_ids(
            attention_mask=attention_mask,
            chunk_length=1,
            rope_deltas=generation_state.model_kwargs.get("rope_deltas"),
            device=next_token.device,
        )
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits
        if eos_token_id is not None and token_id == eos_token_id:
            stop_reason = "eos_token"
            break

    return ContinuationResult(
        generated_ids=generated_ids,
        generated_text=_decode(tokenizer, generated_ids),
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        input_ids=input_ids,
        last_logits=logits,
        stop_reason=stop_reason,
    )


def bracketed_fvt_token_ids(
    *,
    tokenizer_or_processor: Any,
    model: Any | None,
    num_fvt_tokens: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    tokenizer = _tokenizer(tokenizer_or_processor)
    vision_start = _resolve_token_id(
        tokenizer,
        model,
        config_names=("vision_start_token_id",),
        token_candidates=("<|vision_start|>",),
    )
    image_token = _resolve_token_id(
        tokenizer,
        model,
        config_names=("image_token_id",),
        token_candidates=("<|image_pad|>", "<|image|>"),
    )
    vision_end = _resolve_token_id(
        tokenizer,
        model,
        config_names=("vision_end_token_id",),
        token_candidates=("<|vision_end|>",),
    )
    return torch.tensor(
        [vision_start, *([image_token] * num_fvt_tokens), vision_end],
        dtype=torch.long,
        device=device,
    )



def _infer_spatial_merge_size(model: Any | None) -> int:
    config = getattr(model, "config", None)
    vision_config = getattr(config, "vision_config", None)
    value = getattr(vision_config, "spatial_merge_size", None)
    if value is not None:
        return int(value)
    visual = None
    if hasattr(model, "visual"):
        visual = model.visual
    elif hasattr(model, "model") and hasattr(model.model, "visual"):
        visual = model.model.visual
    value = getattr(visual, "spatial_merge_size", None)
    return int(value) if value is not None else 2


def make_fake_image_grid(
    *,
    num_fvt_tokens: int,
    spatial_merge_size: int,
    device: torch.device | str | None,
) -> torch.Tensor:
    if num_fvt_tokens <= 0:
        raise ValueError("num_fvt_tokens must be positive")
    merged_h, merged_w = _near_square_factor_pair(num_fvt_tokens)
    return torch.tensor(
        [[1, merged_h * int(spatial_merge_size), merged_w * int(spatial_merge_size)]],
        dtype=torch.long,
        device=device,
    )


def _fvt_grid_thw(
    *,
    num_fvt_tokens: int,
    spatial_merge_size: int,
    device: torch.device | str | None,
) -> torch.Tensor:
    return make_fake_image_grid(
        num_fvt_tokens=num_fvt_tokens,
        spatial_merge_size=spatial_merge_size,
        device=device,
    )


def _expected_llm_image_tokens(image_grid_thw: torch.Tensor, *, spatial_merge_size: int) -> int:
    grid = image_grid_thw.detach().cpu().view(-1, 3)
    if grid.shape[0] != 1:
        raise ValueError("FVT pseudo-image append expects exactly one fake image grid")
    t, h, w = [int(value) for value in grid[0].tolist()]
    s = int(spatial_merge_size)
    if h % s != 0 or w % s != 0:
        raise ValueError(f"fake image grid {(t, h, w)} is not divisible by spatial_merge_size={s}")
    return int(t * (h // s) * (w // s))


def _near_square_factor_pair(value: int) -> tuple[int, int]:
    root = int(value**0.5)
    for height in range(root, 0, -1):
        if value % height == 0:
            return height, value // height
    return 1, value


def _fvt_mm_token_type_ids(
    *,
    chunk_length: int,
    fvt_token_start: int,
    fvt_token_end: int,
    device: torch.device | str | None,
) -> torch.Tensor:
    token_type_ids = torch.zeros((1, chunk_length), dtype=torch.long, device=device)
    token_type_ids[:, fvt_token_start:fvt_token_end] = 1
    return token_type_ids


def _fvt_chunk_position_ids(
    *,
    attention_mask: torch.Tensor | None,
    chunk_length: int,
    fvt_token_start: int,
    fvt_token_end: int,
    fvt_grid_thw: torch.Tensor,
    spatial_merge_size: int,
    rope_deltas: torch.Tensor | None,
    device: torch.device | str,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    base_position = attention_mask.long().cumsum(-1)[:, -chunk_length:] - 1
    batch_size = base_position.shape[0]
    if batch_size != 1:
        raise ValueError("FVT fake-grid append currently supports batch size 1")
    start_position = int(base_position[0, 0].detach().cpu().item())
    current_pos = start_position
    pieces: list[torch.Tensor] = []

    if fvt_token_start > 0:
        pieces.append(_text_position_piece(current_pos, fvt_token_start, device=device))
        current_pos += fvt_token_start

    vision_positions = _vision_position_piece(
        start_position=current_pos,
        grid_thw=fvt_grid_thw[0],
        spatial_merge_size=spatial_merge_size,
        device=device,
    )
    expected_vision_tokens = fvt_token_end - fvt_token_start
    if vision_positions.shape[-1] != expected_vision_tokens:
        raise ValueError(
            f"FVT grid produced {vision_positions.shape[-1]} positions for {expected_vision_tokens} tokens"
        )
    pieces.append(vision_positions)
    current_pos += max(int(fvt_grid_thw[0, 1].item()), int(fvt_grid_thw[0, 2].item())) // int(spatial_merge_size)

    suffix_len = chunk_length - fvt_token_end
    if suffix_len > 0:
        pieces.append(_text_position_piece(current_pos, suffix_len, device=device))

    position_ids = torch.cat(pieces, dim=-1).view(3, 1, chunk_length)
    if rope_deltas is not None:
        delta = rope_deltas.to(device=device)
        delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
        position_ids = position_ids + delta.view(1, batch_size, 1)
    return position_ids.to(device=device)


def _text_position_piece(start_position: int, length: int, *, device: torch.device | str) -> torch.Tensor:
    return torch.arange(length, device=device).view(1, -1).expand(3, -1) + int(start_position)


def _vision_position_piece(
    *,
    start_position: int,
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    device: torch.device | str,
) -> torch.Tensor:
    llm_grid_t = int(grid_thw[0].item())
    llm_grid_h = int(grid_thw[1].item()) // int(spatial_merge_size)
    llm_grid_w = int(grid_thw[2].item()) // int(spatial_merge_size)
    position_temporal = torch.arange(llm_grid_t, device=device)
    position_width = torch.arange(llm_grid_w, device=device) + int(start_position)
    position_height = torch.arange(llm_grid_h, device=device) + int(start_position)
    position_width = position_width.repeat(llm_grid_h * llm_grid_t)
    position_height = position_height.repeat_interleave(llm_grid_w).repeat(llm_grid_t)
    position_temporal = position_temporal.repeat_interleave(llm_grid_h * llm_grid_w) + int(start_position)
    return torch.stack([position_temporal, position_height, position_width], dim=0)


def _position_ids_have_3d_image_span(
    position_ids: torch.Tensor | None,
    fvt_token_start: int,
    fvt_token_end: int,
) -> bool:
    if position_ids is None or position_ids.ndim != 3 or position_ids.shape[0] != 3:
        return False
    if fvt_token_end <= fvt_token_start:
        return False
    image_positions = position_ids[:, 0, fvt_token_start:fvt_token_end]
    if image_positions.numel() == 0:
        return False
    if image_positions.shape[-1] <= 1:
        return True
    all_rows_equal = torch.equal(image_positions[0], image_positions[1]) and torch.equal(
        image_positions[1], image_positions[2]
    )
    return not all_rows_equal


def _text_positions_are_1d(
    position_ids: torch.Tensor | None,
    fvt_token_start: int,
    fvt_token_end: int,
) -> bool:
    if position_ids is None or position_ids.ndim != 3 or position_ids.shape[0] != 3:
        return False
    pieces = []
    if fvt_token_start > 0:
        pieces.append(position_ids[:, 0, :fvt_token_start])
    if fvt_token_end < position_ids.shape[-1]:
        pieces.append(position_ids[:, 0, fvt_token_end:])
    if not pieces:
        return True
    text_positions = torch.cat(pieces, dim=-1)
    return bool(
        torch.equal(text_positions[0], text_positions[1])
        and torch.equal(text_positions[1], text_positions[2])
    )


def _rope_delta_for_next_token(
    position_ids: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if position_ids is None or attention_mask is None or position_ids.numel() == 0:
        return None
    if position_ids.ndim != 3 or position_ids.shape[1] != 1:
        return None
    total_len = int(attention_mask.shape[-1])
    last_position = int(position_ids[:, 0, -1].detach().max().cpu().item())
    return torch.tensor([[last_position + 1 - total_len]], dtype=torch.long, device=position_ids.device)


def _resolve_num_heads(embed_dim: int, requested_heads: int) -> int:
    requested = max(1, int(requested_heads))
    for heads in range(min(requested, embed_dim), 0, -1):
        if embed_dim % heads == 0:
            return heads
    return 1


def _normalize_attention_mask(
    mask: torch.Tensor | None,
    shape: torch.Size | tuple[int, int],
    device: torch.device,
) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.ndim != 2:
        raise ValueError("attention masks must have shape [T] or [B, T]")
    if tuple(mask.shape) != tuple(shape):
        raise ValueError(f"attention mask shape {tuple(mask.shape)} does not match {tuple(shape)}")
    return mask.to(device=device)


def _mask_to_key_padding(mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None:
        return None
    return mask <= 0


def _attention_entropy(attention: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    probs = attention.float().clamp_min(eps)
    return -(probs * probs.log()).sum(dim=-1)


def _attention_topk_mass(attention: torch.Tensor, ks: tuple[int, ...] = (1, 5, 10)) -> torch.Tensor:
    values = []
    last_dim = attention.shape[-1]
    for k in ks:
        topk = min(k, last_dim)
        values.append(attention.float().topk(topk, dim=-1).values.sum(dim=-1).mean())
    return torch.stack(values)

def _cross_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = sqrt(queries.shape[-1])
    scores = queries @ keys.transpose(0, 1) / scale
    attention = F.softmax(scores, dim=-1)
    attended = attention @ values
    return attended, attention


def _resize_token_sequence(sequence: torch.Tensor, output_length: int) -> torch.Tensor:
    if sequence.shape[0] == output_length:
        return sequence
    pooled = F.adaptive_avg_pool1d(sequence.transpose(0, 1).unsqueeze(0), output_length)
    return pooled.squeeze(0).transpose(0, 1)


def _validate_inputs(
    target_hidden_states: torch.Tensor, pre_merge_visual_tokens: torch.Tensor
) -> None:
    if target_hidden_states.ndim != 2:
        raise ValueError("target_hidden_states must have shape [T, d_lm]")
    if pre_merge_visual_tokens.ndim != 2:
        raise ValueError("pre_merge_visual_tokens must have shape [N, d_v]")
    if target_hidden_states.shape[0] == 0:
        raise ValueError("target_hidden_states must contain at least one token")
    if pre_merge_visual_tokens.shape[0] == 0:
        raise ValueError("pre_merge_visual_tokens must contain at least one token")


def _metadata(
    variant_name: str,
    target_hidden_states: torch.Tensor,
    pre_merge_visual_tokens: torch.Tensor,
    fvt: torch.Tensor,
    attention: torch.Tensor,
    metadata: dict[str, Any] | None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "variant_name": variant_name,
        "target_hidden_shape": list(target_hidden_states.shape),
        "pre_merge_visual_shape": list(pre_merge_visual_tokens.shape),
        "foveated_visual_tokens_shape": list(fvt.shape),
        "attention_shape": list(attention.shape),
        **(metadata or {}),
        **extra,
    }


def _visual_module(model: Any) -> Any:
    if hasattr(model, "visual"):
        return model.visual
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        return model.model.visual
    raise AttributeError("Could not find a Qwen2-VL visual module")


def _tokenizer(tokenizer_or_processor: Any) -> Any:
    return getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)


def _resolve_token_id(
    tokenizer: Any,
    model: Any | None,
    *,
    config_names: tuple[str, ...],
    token_candidates: tuple[str, ...],
) -> int:
    config = getattr(model, "config", None)
    for name in config_names:
        token_id = getattr(config, name, None)
        if token_id is not None:
            return int(token_id)
    for token in token_candidates:
        if hasattr(tokenizer, "convert_tokens_to_ids"):
            token_id = tokenizer.convert_tokens_to_ids(token)
            unk_id = getattr(tokenizer, "unk_token_id", None)
            if token_id is not None and token_id != unk_id:
                return int(token_id)
        if hasattr(tokenizer, "encode"):
            ids = tokenizer.encode(token, add_special_tokens=False)
            if len(ids) == 1:
                return int(ids[0])
    raise ValueError(f"Could not resolve token id for candidates: {token_candidates}")



def _encode_text(tokenizer: Any, text: str, device: torch.device | str | None) -> torch.Tensor:
    return torch.tensor(
        tokenizer.encode(text, add_special_tokens=False),
        dtype=torch.long,
        device=device,
    )


def _token_debug(tokenizer: Any, token_id: int) -> dict[str, Any]:
    return {"token_id": int(token_id), "token_text": _decode(tokenizer, [int(token_id)])}


def _last_token_debug(tokenizer: Any, input_ids: torch.Tensor | None) -> dict[str, Any] | None:
    if input_ids is None or input_ids.numel() == 0:
        return None
    return _token_debug(tokenizer, int(input_ids[0, -1].detach().cpu().item()))


def _topk_logits(tokenizer: Any, logits: torch.Tensor | None, k: int = 10) -> list[dict[str, Any]]:
    if logits is None:
        return []
    values, indices = logits[0, -1].detach().float().cpu().topk(k)
    return [
        {
            "token_id": int(token_id),
            "token_text": _decode(tokenizer, [int(token_id)]),
            "logit": float(value),
        }
        for value, token_id in zip(values.tolist(), indices.tolist(), strict=False)
    ]


def _extend_attention(attention_mask: torch.Tensor, chunk_length: int) -> torch.Tensor:
    ones = torch.ones(
        (attention_mask.shape[0], chunk_length),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    return torch.cat([attention_mask, ones], dim=-1)


def _chunk_position_ids(
    *,
    attention_mask: torch.Tensor | None,
    chunk_length: int,
    rope_deltas: torch.Tensor | None,
    device: torch.device | str,
) -> torch.Tensor | None:
    if rope_deltas is None or attention_mask is None:
        return None
    base_position = attention_mask.long().cumsum(-1)[:, -chunk_length:] - 1
    batch_size = base_position.shape[0]
    position_ids = base_position.view(1, batch_size, chunk_length).repeat(3, 1, 1)
    delta = rope_deltas.to(device=device)
    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
    return position_ids.to(device=device) + delta.view(1, batch_size, 1)


def _state_attr(state: Any, name: str) -> Any:
    if isinstance(state, dict):
        return state.get(name)
    return getattr(state, name, None)


def _decode(tokenizer: Any, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _infer_model_device(model: Any) -> torch.device | None:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return None
