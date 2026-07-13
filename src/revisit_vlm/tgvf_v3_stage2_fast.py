from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import torch
from torch.nn import functional as F

from revisit_vlm.qwen3_vl_tgvf import (
    ANSWER_END,
    ANSWER_START,
    EVIDENCE_END,
    EVIDENCE_START,
    EVIDENCE_STATE_END,
    EVIDENCE_STATE_START,
    FOCUS_END,
    FOCUS_START,
    NEED_LOCAL_EVIDENCE,
    PROTOCOL_C_TOOL_OBSERVATION,
    PROTOCOL_C_TOOL_OBSERVATION_QWEN2_NO_THINK,
    PROTOCOL_E_ACTION_EVIDENCE_SPECIAL,
    SUFFICIENT_EVIDENCE,
    TGVF_END,
    TGVF_START,
    TGVFProtocol,
    _bracketed_visual_token_ids,
    _compute_qwen3_position_ids_for_sequence,
    _deepstack_features,
    _encode_text,
    _extract_vision_tensors,
    build_direct_messages,
    build_qwen3_inputs,
    focus_target_char_span,
    normalize_tgvf_protocol,
    protocol_focus_tokens,
    protocol_uses_evidence_tags,
    protocol_uses_tool_observation,
    render_focus_action_text,
    render_focus_readout_answer_text,
    render_no_focus_output_text,
    render_tgvf_prefix_suffix,
)
from revisit_vlm.tgvf_foveal import (
    FovealCrossAttentionOutput,
    finalize_tgvf_output_with_frozen_qwen_merger,
)
from revisit_vlm.tgvf_training import IGNORE_INDEX, same_image_negative_matrix_ce_loss
from revisit_vlm.tgvf_v3_stage1 import (
    _find_focus_target_span_in_forced_ids,
    _full_mm_token_type_ids,
    _image_input,
    _move_tensors_for_stage1,
    _safe_visual_token_manifold_loss,
)
from revisit_vlm.tgvf_v3_stage2 import (
    ORIGINAL_IMAGE_MASK_SCOPE_EVIDENCE_ONLY,
    Stage2LossWeights,
    TGVFv3Stage2Sample,
    TGVFv3Stage2StepOutput,
    merge_protocol_c_boundary_stats,
    original_image_mask_block_query_end,
    protocol_c_boundary_token_accuracy,
    sample_original_image_mask_active,
    _weighted_stage2_tokens,
)


@dataclass
class _BaseItem:
    sample: TGVFv3Stage2Sample
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    model_inputs: dict[str, Any]
    image_grid_thw: torch.Tensor
    image_token_indices: torch.Tensor
    visual_pre_count: int
    visual_merge_count: int


@dataclass
class _FocusPrepared:
    sample: TGVFv3Stage2Sample
    item: _BaseItem
    action_ids: torch.Tensor
    action_weights: torch.Tensor
    target_start: int
    target_end: int
    target_hidden_states: torch.Tensor
    pre_merge_visual_tokens: torch.Tensor
    merged_visual_tokens: torch.Tensor
    foveated_visual_tokens: torch.Tensor
    value_span_matched: bool
    final_input_ids: torch.Tensor
    final_labels: torch.Tensor
    matrix_ce_labels: torch.Tensor | None
    final_weights: torch.Tensor
    final_inputs_embeds: torch.Tensor
    final_attention_mask_2d: torch.Tensor
    final_attention_mask: torch.Tensor
    final_position_ids: torch.Tensor
    final_mm_token_type_ids: torch.Tensor
    fvt_token_indices: torch.Tensor
    d_token_indices: torch.Tensor | None
    d_deepstack_visual_embeds: list[torch.Tensor] | None
    masked_image_key_count: int
    image_key_mask_active: bool
    mask_mode: str
    fvt_shape: list[int]
    target_hidden_shape: list[int]
    deepstack_feature_shapes: list[list[int]] | None = None


@dataclass
class _NoFocusPrepared:
    sample: TGVFv3Stage2Sample
    item: _BaseItem
    final_input_ids: torch.Tensor
    final_labels: torch.Tensor
    final_weights: torch.Tensor
    final_attention_mask: torch.Tensor
    value_span_matched: bool


@dataclass
class Stage2MatrixCEOutput:
    loss: torch.Tensor
    score_matrix: torch.Tensor
    debug: dict[str, Any]


def v3_stage2_batched_training_step(
    *,
    qwen_model: Any,
    qwen_forward_model: Any,
    processor: Any,
    foveal_module: torch.nn.Module,
    samples: list[TGVFv3Stage2Sample],
    loss_weights: Stage2LossWeights,
    device: torch.device | str,
    hidden_state_index: int = -1,
    max_image_resolution: int | None = 512,
    position_mode: str = "native_source_grid",
    mask_original_image_after_tgvf: bool = True,
    mask_original_image_after_tgvf_prob: float = 1.0,
    mask_original_image_after_tgvf_scope: str = ORIGINAL_IMAGE_MASK_SCOPE_EVIDENCE_ONLY,
    protocol: TGVFProtocol = "legacy_v3_tags",
    deepstack_enabled: bool = False,
) -> TGVFv3Stage2StepOutput:
    protocol = normalize_tgvf_protocol(protocol)
    if position_mode != "native_source_grid":
        raise ValueError("fast Stage2 currently supports only native_source_grid")
    if not samples:
        raise ValueError("samples must be non-empty")

    tokenizer = processor.tokenizer
    base_items = [
        _build_base_item(
            qwen_model=qwen_model,
            processor=processor,
            sample=sample,
            device=device,
            max_image_resolution=max_image_resolution,
        )
        for sample in samples
    ]
    _attach_batched_vision_features(qwen_model=qwen_model, base_items=base_items, device=device)

    focus_items = [
        item
        for item in base_items
        if item.sample.need_focus and item.sample.trajectory_type == "single_focus"
    ]
    multi_focus_items = [
        item
        for item in base_items
        if item.sample.need_focus and item.sample.trajectory_type == "multi_focus"
    ]
    no_focus_items = [item for item in base_items if not item.sample.need_focus]

    focus_prepared = _prepare_single_focus_items(
        qwen_model=qwen_model,
        qwen_forward_model=qwen_forward_model,
        processor=processor,
        foveal_module=foveal_module,
        focus_items=focus_items,
        loss_weights=loss_weights,
        device=device,
        hidden_state_index=hidden_state_index,
        max_image_resolution=max_image_resolution,
        mask_original_image_after_tgvf=mask_original_image_after_tgvf,
        mask_original_image_after_tgvf_prob=mask_original_image_after_tgvf_prob,
        mask_original_image_after_tgvf_scope=mask_original_image_after_tgvf_scope,
        protocol=protocol,
        deepstack_enabled=deepstack_enabled,
    )

    for item in multi_focus_items:
        focus_prepared.append(
            _prepare_multi_focus_final(
                qwen_model=qwen_model,
                qwen_forward_model=qwen_forward_model,
                processor=processor,
                foveal_module=foveal_module,
                item=item,
                loss_weights=loss_weights,
                device=device,
                hidden_state_index=hidden_state_index,
                max_image_resolution=max_image_resolution,
                mask_original_image_after_tgvf=mask_original_image_after_tgvf,
                mask_original_image_after_tgvf_prob=mask_original_image_after_tgvf_prob,
                mask_original_image_after_tgvf_scope=mask_original_image_after_tgvf_scope,
                protocol=protocol,
                deepstack_enabled=deepstack_enabled,
            )
        )

    no_focus_prepared = [
        _prepare_no_focus_final(
            processor=processor,
            item=item,
            loss_weights=loss_weights,
            device=device,
            protocol=protocol,
        )
        for item in no_focus_items
    ]

    zero = next(foveal_module.parameters()).new_zeros(())
    loss_focus = zero
    loss_no_focus = zero
    loss_manifold = zero
    focus_loss_tokens = 0.0
    no_focus_loss_tokens = 0.0
    boundary_stat_logs: list[dict[str, float | int]] = []

    if focus_prepared:
        focus_batch = _pad_focus_batch(
            focus_prepared,
            device=device,
            deepstack_enabled=deepstack_enabled,
        )
        outputs = _qwen_manual_forward_with_optional_deepstack(
            qwen_forward_model,
            inputs_embeds=focus_batch["inputs_embeds"],
            attention_mask=focus_batch["attention_mask"],
            position_ids=focus_batch["position_ids"],
            mm_token_type_ids=focus_batch["mm_token_type_ids"],
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
            visual_pos_masks=focus_batch.get("visual_pos_masks"),
            deepstack_visual_embeds=focus_batch.get("deepstack_visual_embeds"),
        )
        loss_focus, focus_loss_tokens = _weighted_lm_loss(
            outputs.logits,
            focus_batch["labels"],
            focus_batch["loss_weights"],
        )
        boundary_stat_logs.append(
            protocol_c_boundary_token_accuracy(
                logits=outputs.logits,
                labels=focus_batch["labels"],
                tokenizer=tokenizer,
                protocol=protocol,
            )
        )
        if loss_weights.visual_token_manifold:
            loss_manifold = torch.stack(
                [
                    _safe_visual_token_manifold_loss(
                        item.foveated_visual_tokens,
                        item.merged_visual_tokens,
                    )
                    for item in focus_prepared
                ]
            ).mean()

    if no_focus_prepared:
        no_focus_batch = _pad_native_batch(
            qwen_model,
            [item.final_input_ids for item in no_focus_prepared],
            [item.final_attention_mask for item in no_focus_prepared],
            [item.item.model_inputs for item in no_focus_prepared],
            device=device,
            pad_token_id=int(getattr(tokenizer, "pad_token_id", 0) or 0),
        )
        outputs = qwen_forward_model(
            **no_focus_batch,
            use_cache=False,
            return_dict=True,
        )
        labels = _pad_2d(
            [item.final_labels for item in no_focus_prepared],
            pad_value=IGNORE_INDEX,
            device=device,
        )
        weights = _pad_2d_float(
            [item.final_weights for item in no_focus_prepared],
            pad_value=0.0,
            device=device,
        )
        loss_no_focus, no_focus_loss_tokens = _weighted_lm_loss(outputs.logits, labels, weights)
        boundary_stat_logs.append(
            protocol_c_boundary_token_accuracy(
                logits=outputs.logits,
                labels=labels,
                tokenizer=tokenizer,
                protocol=protocol,
            )
        )

    total_weight = focus_loss_tokens + no_focus_loss_tokens
    if total_weight > 0:
        loss_total = (
            loss_focus * float(focus_loss_tokens)
            + loss_no_focus * float(no_focus_loss_tokens)
        ) / float(total_weight)
    else:
        loss_total = zero
    if loss_weights.visual_token_manifold:
        loss_total = loss_total + float(loss_weights.visual_token_manifold) * loss_manifold

    value_matches = [
        item.value_span_matched for item in [*focus_prepared, *no_focus_prepared]
        if item.sample.value_span_text
    ]
    examples = []
    for item in [*focus_prepared, *no_focus_prepared][:2]:
        examples.append(
            {
                "need_focus": bool(item.sample.need_focus),
                "question": item.sample.question,
                "target": item.sample.target,
                "answer": item.sample.answer,
                "mask_mode": getattr(item, "mask_mode", "standard_2d_causal"),
                "image_key_mask_active": bool(getattr(item, "image_key_mask_active", False)),
                "masked_image_key_count": getattr(item, "masked_image_key_count", 0),
                "fvt_shape": getattr(item, "fvt_shape", None),
                "target_hidden_shape": getattr(item, "target_hidden_shape", None),
                "deepstack_feature_shapes": getattr(item, "deepstack_feature_shapes", None),
                "value_span_matched": bool(item.value_span_matched),
                "tgvf_protocol": protocol,
            }
        )
    focus_mask_active = sum(1 for item in focus_prepared if item.image_key_mask_active)
    return TGVFv3Stage2StepOutput(
        loss_total=loss_total,
        loss_focus=loss_focus,
        loss_no_focus=loss_no_focus,
        loss_visual_token_manifold=loss_manifold,
        loss_same_image_matrix_ce=zero,
        debug={
            "fast_batched_stage2": True,
            "focus_count": len(focus_prepared),
            "single_focus_count": len(focus_items),
            "multi_focus_count": len(multi_focus_items),
            "no_focus_count": len(no_focus_prepared),
            "focus_sample_mask_active_rate": focus_mask_active / max(len(focus_prepared), 1),
            "no_focus_mask_active_rate": 0.0,
            "mask_original_image_after_tgvf_prob": float(mask_original_image_after_tgvf_prob),
            "mask_original_image_after_tgvf_scope": str(mask_original_image_after_tgvf_scope),
            "deepstack_training_enabled": bool(deepstack_enabled),
            "qwen3_deepstack_features_injected": bool(
                deepstack_enabled and focus_prepared
            ),
            "deepstack_original_image_scope": (
                str(mask_original_image_after_tgvf_scope)
                if deepstack_enabled
                else "off"
            ),
            "value_span_match_rate": (
                sum(1.0 for matched in value_matches if matched) / max(len(value_matches), 1)
                if value_matches
                else None
            ),
            **merge_protocol_c_boundary_stats(boundary_stat_logs),
            "focus_loss_token_weight": focus_loss_tokens,
            "no_focus_loss_token_weight": no_focus_loss_tokens,
            "special_tokens_added": False,
            "tokenizer_resized": False,
            "markers_are_plain_text": protocol == "legacy_v3_tags",
            "tgvf_protocol": protocol,
            "matrix_ce_enabled": False,
            "same_image_negative_enabled": False,
            "debug_examples": examples,
        },
    )


def v3_stage2_same_image_matrix_ce_step(
    *,
    qwen_model: Any,
    qwen_forward_model: Any,
    processor: Any,
    foveal_module: torch.nn.Module,
    samples: list[TGVFv3Stage2Sample],
    loss_weights: Stage2LossWeights,
    device: torch.device | str,
    hidden_state_index: int = -1,
    max_image_resolution: int | None = 512,
    mask_original_image_after_tgvf_scope: str = ORIGINAL_IMAGE_MASK_SCOPE_EVIDENCE_ONLY,
    protocol: TGVFProtocol = "legacy_v3_tags",
    deepstack_enabled: bool = False,
    readout_batch_size: int = 4,
) -> Stage2MatrixCEOutput:
    protocol = normalize_tgvf_protocol(protocol)
    if len(samples) < 2:
        raise ValueError("Stage2 Matrix-CE requires at least two samples")
    if int(readout_batch_size) < 1:
        raise ValueError("Stage2 Matrix-CE readout_batch_size must be >= 1")
    image_keys = {str(sample.image_id or sample.image) for sample in samples}
    if len(image_keys) != 1:
        raise ValueError("Stage2 Matrix-CE samples must share one image")
    if any(not sample.need_focus or sample.trajectory_type != "single_focus" for sample in samples):
        raise ValueError("Stage2 Matrix-CE supports only single-focus samples")

    base_items = [
        _build_base_item(
            qwen_model=qwen_model,
            processor=processor,
            sample=sample,
            device=device,
            max_image_resolution=max_image_resolution,
        )
        for sample in samples
    ]
    _attach_same_image_vision_features(
        qwen_model=qwen_model,
        base_items=base_items,
        device=device,
    )
    prepared = _prepare_single_focus_items(
        qwen_model=qwen_model,
        qwen_forward_model=qwen_forward_model,
        processor=processor,
        foveal_module=foveal_module,
        focus_items=base_items,
        loss_weights=loss_weights,
        device=device,
        hidden_state_index=hidden_state_index,
        max_image_resolution=max_image_resolution,
        mask_original_image_after_tgvf=True,
        mask_original_image_after_tgvf_prob=1.0,
        mask_original_image_after_tgvf_scope=mask_original_image_after_tgvf_scope,
        protocol=protocol,
        deepstack_enabled=deepstack_enabled,
    )
    cross_items = _stage2_matrix_ce_cross_items(prepared)
    scores = []
    for start in range(0, len(cross_items), int(readout_batch_size)):
        chunk = cross_items[start : start + int(readout_batch_size)]
        batch = _pad_focus_batch(chunk, device=device, deepstack_enabled=deepstack_enabled)
        outputs = _qwen_manual_forward_with_optional_deepstack(
            qwen_forward_model,
            inputs_embeds=batch["inputs_embeds"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
            mm_token_type_ids=batch["mm_token_type_ids"],
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
            visual_pos_masks=batch.get("visual_pos_masks"),
            deepstack_visual_embeds=batch.get("deepstack_visual_embeds"),
        )
        scores.extend(_sequence_log_likelihoods(outputs.logits, batch["labels"]).unbind(0))
    group_size = len(prepared)
    score_matrix = torch.stack(scores).view(group_size, group_size)
    loss = same_image_negative_matrix_ce_loss([score_matrix])
    diagonal = score_matrix.diagonal()
    off_diagonal = score_matrix[
        ~torch.eye(group_size, dtype=torch.bool, device=score_matrix.device)
    ]
    top1 = float(
        (score_matrix.argmax(dim=-1) == torch.arange(group_size, device=score_matrix.device))
        .float()
        .mean()
        .detach()
        .cpu()
        .item()
    )
    return Stage2MatrixCEOutput(
        loss=loss,
        score_matrix=score_matrix,
        debug={
            "matrix_ce_enabled": True,
            "same_image_negative_enabled": True,
            "matrix_ce_group_size": group_size,
            "matrix_ce_readout_batch_size": int(readout_batch_size),
            "matrix_ce_score_span": "post_d_readout_before_answer",
            "matrix_ce_candidate_swap": "d_and_d_deepstack_features",
            "matrix_ce_original_image_mask_probability": 1.0,
            "matrix_ce_vision_encode_count": 1,
            "matrix_ce_top1": top1,
            "matrix_ce_positive_score_mean": float(diagonal.detach().float().mean().cpu().item()),
            "matrix_ce_negative_score_mean": float(
                off_diagonal.detach().float().mean().cpu().item()
            ),
            "matrix_ce_positive_negative_margin": float(
                (diagonal.detach().float().mean() - off_diagonal.detach().float().mean())
                .cpu()
                .item()
            ),
        },
    )


def _prepare_single_focus_items(
    *,
    qwen_model: Any,
    qwen_forward_model: Any,
    processor: Any,
    foveal_module: torch.nn.Module,
    focus_items: list[_BaseItem],
    loss_weights: Stage2LossWeights,
    device: torch.device | str,
    hidden_state_index: int,
    max_image_resolution: int | None,
    mask_original_image_after_tgvf: bool,
    mask_original_image_after_tgvf_prob: float,
    mask_original_image_after_tgvf_scope: str,
    protocol: TGVFProtocol,
    deepstack_enabled: bool,
) -> list[_FocusPrepared]:
    if not focus_items:
        return []
    focus_hidden = _batched_focus_first_forward(
        qwen_model=qwen_model,
        qwen_forward_model=qwen_forward_model,
        tokenizer=processor.tokenizer,
        focus_items=focus_items,
        device=device,
        hidden_state_index=hidden_state_index,
        loss_weights=loss_weights,
        protocol=protocol,
        deepstack_enabled=deepstack_enabled,
    )
    prepared = []
    for item, action_ids, action_weights, target_span, target_hidden in focus_hidden:
        pre = item.model_inputs["_v_pre"].to(device)
        merged = item.model_inputs["_v_merge"].to(device)
        output = foveal_module(
            target_hidden_states=target_hidden,
            pre_merge_visual_tokens=pre,
            metadata={
                "target": item.sample.target,
                "stage": "tgvf_v3_stage2_fast",
                "evidence_state": NEED_LOCAL_EVIDENCE,
                "qwen_model": qwen_model,
                "processor": processor,
                "image": _image_input(item.sample.image, max_image_resolution=max_image_resolution),
                "question": item.sample.prompt_question,
                "device": device,
                "deepstack_pre_merge_visual_tokens": _item_deepstack_pre_merge_features(item),
            },
        )
        output = finalize_tgvf_output_with_frozen_qwen_merger(qwen_model, output)
        prepared.append(
            _prepare_focus_final(
                qwen_model=qwen_model,
                processor=processor,
                item=item,
                action_ids=action_ids,
                action_weights=action_weights,
                target_span=target_span,
                target_hidden_states=target_hidden,
                pre_merge_visual_tokens=pre,
                merged_visual_tokens=merged,
                foveated_visual_tokens=output.foveated_visual_tokens,
                d_deepstack_visual_embeds=output.deepstack_visual_embeds,
                loss_weights=loss_weights,
                device=device,
                mask_original_image_after_tgvf=mask_original_image_after_tgvf,
                mask_original_image_after_tgvf_prob=mask_original_image_after_tgvf_prob,
                mask_original_image_after_tgvf_scope=mask_original_image_after_tgvf_scope,
                protocol=protocol,
                deepstack_enabled=deepstack_enabled,
            )
        )
    return prepared


def _stage2_matrix_ce_cross_items(items: list[_FocusPrepared]) -> list[_FocusPrepared]:
    cross_items = []
    for row in items:
        if row.matrix_ce_labels is None:
            raise ValueError("Stage2 Matrix-CE row is missing readout labels")
        for candidate in items:
            embeds = row.final_inputs_embeds.clone()
            embeds = _scatter_visual_embeds(
                embeds,
                token_indices=row.fvt_token_indices,
                visual_embeds=candidate.foveated_visual_tokens,
            )
            cross_items.append(
                replace(
                    row,
                    final_labels=row.matrix_ce_labels,
                    final_inputs_embeds=embeds,
                    d_deepstack_visual_embeds=candidate.d_deepstack_visual_embeds,
                )
            )
    return cross_items


def _sequence_log_likelihoods(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    per_token_nll = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view_as(shift_labels)
    valid = shift_labels != IGNORE_INDEX
    return -(per_token_nll * valid.to(per_token_nll.dtype)).sum(dim=-1)


def _attach_same_image_vision_features(
    *,
    qwen_model: Any,
    base_items: list[_BaseItem],
    device: torch.device | str,
) -> None:
    if not base_items:
        return
    first = base_items[0]
    _attach_batched_vision_features(
        qwen_model=qwen_model,
        base_items=[first],
        device=device,
    )
    feature_names = (
        "_v_pre",
        "_v_merge",
        "_deepstack_visual_embeds",
        "_deepstack_pre_merge_visual_embeds",
    )
    for item in base_items[1:]:
        if item.visual_pre_count != first.visual_pre_count:
            raise RuntimeError("same-image Matrix-CE V_pre token counts differ")
        if item.visual_merge_count != first.visual_merge_count:
            raise RuntimeError("same-image Matrix-CE V_merge token counts differ")
        for name in feature_names:
            item.model_inputs[name] = first.model_inputs[name]


def _build_base_item(
    *,
    qwen_model: Any,
    processor: Any,
    sample: TGVFv3Stage2Sample,
    device: torch.device | str,
    max_image_resolution: int | None,
) -> _BaseItem:
    image = _image_input(sample.image, max_image_resolution=max_image_resolution)
    inputs = _move_tensors_for_stage1(
        build_qwen3_inputs(processor, build_direct_messages(image, sample.prompt_question)),
        device,
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
    image_grid_thw = inputs.get("image_grid_thw")
    if not isinstance(image_grid_thw, torch.Tensor):
        raise RuntimeError("Qwen3 input is missing image_grid_thw")
    image_token_id = getattr(getattr(qwen_model, "config", None), "image_token_id", None)
    if image_token_id is None:
        raise RuntimeError("Qwen3 image_token_id is unavailable")
    image_indices = torch.nonzero(input_ids[0] == int(image_token_id), as_tuple=False).view(-1)
    visual_pre_count = int(torch.prod(image_grid_thw.to(torch.long), dim=-1).sum().detach().cpu().item())
    return _BaseItem(
        sample=sample,
        input_ids=input_ids,
        attention_mask=attention_mask,
        model_inputs=inputs,
        image_grid_thw=image_grid_thw,
        image_token_indices=image_indices,
        visual_pre_count=visual_pre_count,
        visual_merge_count=int(image_indices.numel()),
    )


@torch.no_grad()
def _attach_batched_vision_features(
    *,
    qwen_model: Any,
    base_items: list[_BaseItem],
    device: torch.device | str,
) -> None:
    if not base_items:
        return
    if not hasattr(qwen_model, "get_image_features"):
        raise RuntimeError("Qwen3 model does not expose get_image_features")
    pixel_values = torch.cat([item.model_inputs["pixel_values"].to(device) for item in base_items], dim=0)
    image_grid_thw = torch.cat([item.image_grid_thw.to(device) for item in base_items], dim=0)
    deepstack_pre_merge_features: list[torch.Tensor] = []
    handles = _register_deepstack_premerge_hooks(
        qwen_model,
        deepstack_pre_merge_features,
    )
    try:
        image_output = qwen_model.get_image_features(
            pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    v_pre, _v_merge = _extract_vision_tensors(image_output)
    if v_pre is None:
        raise RuntimeError("Qwen3 get_image_features did not return V_pre tensors")
    deepstack_features = _deepstack_features(image_output)
    pre_splits = _split_visual_tensor(v_pre, [item.visual_pre_count for item in base_items])
    merge_splits = [_merge_pre_tokens_with_frozen_qwen(qwen_model, pre) for pre in pre_splits]
    deepstack_splits_by_layer = [
        _split_visual_tensor(feature, [item.visual_merge_count for item in base_items])
        for feature in deepstack_features
    ]
    deepstack_pre_splits_by_layer = [
        _split_visual_tensor(feature, [item.visual_pre_count for item in base_items])
        for feature in deepstack_pre_merge_features
    ]
    for item, pre, merge in zip(base_items, pre_splits, merge_splits, strict=True):
        if int(merge.shape[0]) != item.visual_merge_count:
            raise RuntimeError(
                f"Merged visual token count mismatch for {item.sample.image}: "
                f"{int(merge.shape[0])} vs {item.visual_merge_count}"
            )
        item.model_inputs["_v_pre"] = pre.detach()
        item.model_inputs["_v_merge"] = merge.detach()
    for item_index, item in enumerate(base_items):
        item.model_inputs["_deepstack_visual_embeds"] = [
            layer_splits[item_index].detach()
            for layer_splits in deepstack_splits_by_layer
        ]
        item.model_inputs["_deepstack_pre_merge_visual_embeds"] = [
            layer_splits[item_index].detach()
            for layer_splits in deepstack_pre_splits_by_layer
        ]


def _register_deepstack_premerge_hooks(
    qwen_model: Any,
    target: list[torch.Tensor],
) -> list[Any]:
    visual = None
    if hasattr(qwen_model, "visual"):
        visual = qwen_model.visual
    elif hasattr(qwen_model, "model") and hasattr(qwen_model.model, "visual"):
        visual = qwen_model.model.visual
    branch_mergers = getattr(visual, "deepstack_merger_list", None)
    if branch_mergers is None:
        return []

    def make_hook() -> Any:
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            if inputs and isinstance(inputs[0], torch.Tensor):
                target.append(inputs[0].detach())

        return hook

    return [merger.register_forward_pre_hook(make_hook()) for merger in branch_mergers]


def _merge_pre_tokens_with_frozen_qwen(model: Any, pre_tokens: torch.Tensor) -> torch.Tensor:
    output = FovealCrossAttentionOutput(
        foveated_visual_tokens=pre_tokens,
        attention_debug={},
        debug_metadata={
            "variant_name": "native_source_visual_merge",
            "final_fvt_requires_qwen_visual_merger": True,
        },
        conditioned_pre_merge_visual_tokens=pre_tokens,
    )
    return finalize_tgvf_output_with_frozen_qwen_merger(model, output).foveated_visual_tokens


def _batched_focus_first_forward(
    *,
    qwen_model: Any,
    qwen_forward_model: Any,
    tokenizer: Any,
    focus_items: list[_BaseItem],
    device: torch.device | str,
    hidden_state_index: int,
    loss_weights: Stage2LossWeights,
    protocol: TGVFProtocol,
    deepstack_enabled: bool,
) -> list[tuple[_BaseItem, torch.Tensor, torch.Tensor, tuple[int, int], torch.Tensor]]:
    action_ids_list = []
    action_weights_list = []
    first_ids = []
    first_attention = []
    first_inputs = []
    spans = []
    for item in focus_items:
        action_text = render_focus_action_text(
            item.sample.target,
            protocol=protocol,
            pre_focus_think=item.sample.pre_focus_think,
            append_im_end=protocol_uses_tool_observation(protocol),
        )
        target_char_span = focus_target_char_span(action_text, protocol=protocol)
        if target_char_span is None:
            raise RuntimeError(f"Could not locate focus target char span for: {item.sample.target!r}")
        state_weight_end = (
            len(f"{EVIDENCE_STATE_START}{NEED_LOCAL_EVIDENCE}{EVIDENCE_STATE_END}")
            if protocol == "legacy_v3_tags"
            else max(0, target_char_span[0] - len(protocol_focus_tokens(protocol)[0]))
        )
        action_ids, action_weights = _weighted_stage2_tokens(
            tokenizer,
            action_text,
            [
                (0, state_weight_end, loss_weights.evidence_state),
                (target_char_span[0], target_char_span[1], loss_weights.focus_target),
            ],
            device=device,
            default_weight=1.0,
        )
        span = _find_focus_target_span_in_forced_ids(
            tokenizer,
            action_ids.view(-1).tolist(),
            protocol=protocol,
        )
        if span is None:
            raise RuntimeError(f"Could not locate focus target span for: {item.sample.target!r}")
        target_start, target_end, _target_text = span
        full_ids = torch.cat([item.input_ids, action_ids], dim=-1)
        full_attention = torch.cat(
            [
                item.attention_mask,
                torch.ones_like(action_ids, dtype=item.attention_mask.dtype),
            ],
            dim=-1,
        )
        embeds = qwen_forward_model.get_input_embeddings()(full_ids).detach().clone()
        embeds = _scatter_visual_embeds(
            embeds,
            token_indices=item.image_token_indices.to(device),
            visual_embeds=item.model_inputs["_v_merge"].to(device),
        )
        action_ids_list.append(action_ids)
        action_weights_list.append(action_weights)
        first_ids.append(full_ids)
        first_attention.append(full_attention)
        first_inputs.append(embeds)
        spans.append((int(item.input_ids.shape[-1]) + target_start, int(item.input_ids.shape[-1]) + target_end))

    input_ids = _pad_2d(first_ids, pad_value=int(getattr(tokenizer, "pad_token_id", 0) or 0), device=device)
    attention_mask = _pad_2d(first_attention, pad_value=0, device=device)
    inputs_embeds = _pad_embeds(first_inputs, max_len=int(input_ids.shape[-1]), device=device)
    position_ids = _batched_position_ids(
        model=qwen_model,
        input_ids_list=first_ids,
        attention_mask_list=first_attention,
        image_grid_thw_list=[item.image_grid_thw for item in focus_items],
        device=device,
        max_len=int(input_ids.shape[-1]),
    )
    mm_token_type_ids = _pad_2d(
        [
            _image_token_type_ids(qwen_model, ids, device=device)
            for ids in first_ids
        ],
        pad_value=0,
        device=device,
    )
    deepstack_kwargs = _batched_deepstack_inputs(
        items=focus_items,
        max_len=int(input_ids.shape[-1]),
        device=device,
        dtype=inputs_embeds.dtype,
    ) if deepstack_enabled else {}
    outputs = _qwen_manual_forward_with_optional_deepstack(
        qwen_forward_model,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        mm_token_type_ids=mm_token_type_ids,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
        **deepstack_kwargs,
    )
    hidden = outputs.hidden_states[hidden_state_index]
    result = []
    for index, item in enumerate(focus_items):
        start, end = spans[index]
        target_hidden = hidden[index, start:end]
        result.append(
            (
                item,
                action_ids_list[index],
                action_weights_list[index],
                (start - int(item.input_ids.shape[-1]), end - int(item.input_ids.shape[-1])),
                target_hidden,
            )
        )
    return result


def _prepare_focus_final(
    *,
    qwen_model: Any,
    processor: Any,
    item: _BaseItem,
    action_ids: torch.Tensor,
    action_weights: torch.Tensor,
    target_span: tuple[int, int],
    target_hidden_states: torch.Tensor,
    pre_merge_visual_tokens: torch.Tensor,
    merged_visual_tokens: torch.Tensor,
    foveated_visual_tokens: torch.Tensor,
    d_deepstack_visual_embeds: list[torch.Tensor] | None,
    loss_weights: Stage2LossWeights,
    device: torch.device | str,
    mask_original_image_after_tgvf: bool,
    mask_original_image_after_tgvf_prob: float,
    mask_original_image_after_tgvf_scope: str,
    protocol: TGVFProtocol,
    deepstack_enabled: bool,
) -> _FocusPrepared:
    tokenizer = processor.tokenizer
    d = foveated_visual_tokens.to(device)
    tgvf_prefix, tgvf_suffix = render_tgvf_prefix_suffix(
        protocol=protocol,
        include_leading_im_end=not protocol_uses_tool_observation(protocol),
    )
    tgvf_ids = _bracketed_visual_token_ids(
        processor,
        qwen_model,
        num_fvt_tokens=int(d.shape[0]),
        prefix=tgvf_prefix,
        suffix=tgvf_suffix,
        device=device,
    ).view(1, -1)
    readout_text = (item.sample.post_focus_think or item.sample.evidence_description).strip()
    ev_answer_text = render_focus_readout_answer_text(
        evidence_description=item.sample.evidence_description,
        answer=item.sample.answer,
        protocol=protocol,
        readout_think=readout_text,
        append_im_end=protocol_uses_tool_observation(protocol),
    )
    if protocol == "legacy_v3_tags":
        evidence_start = ev_answer_text.find(EVIDENCE_START)
        evidence_end = ev_answer_text.find(EVIDENCE_END) + len(EVIDENCE_END)
        answer_start = ev_answer_text.find(ANSWER_START)
    else:
        evidence_start = ev_answer_text.find(readout_text)
        evidence_end = evidence_start + len(readout_text)
        answer_start = ev_answer_text.rfind(item.sample.answer.strip())
    spans = [
        (evidence_start, evidence_end, loss_weights.evidence),
        (answer_start, len(ev_answer_text), loss_weights.answer),
    ]
    value_span_matched = False
    if item.sample.value_span_text:
        value_start = ev_answer_text.find(item.sample.value_span_text)
        if protocol == "legacy_v3_tags":
            evidence_inner_start = ev_answer_text.find(EVIDENCE_START) + len(EVIDENCE_START)
            evidence_inner_end = ev_answer_text.find(EVIDENCE_END, evidence_inner_start)
        else:
            evidence_inner_start = evidence_start
            evidence_inner_end = evidence_end
        if evidence_inner_start <= value_start < evidence_inner_end:
            spans.append((value_start, value_start + len(item.sample.value_span_text), loss_weights.value_span))
            value_span_matched = True
    ev_answer_ids, ev_answer_weights = _weighted_stage2_tokens(
        tokenizer,
        ev_answer_text,
        spans,
        device=device,
        default_weight=1.0,
    )
    input_ids = torch.cat([item.input_ids, action_ids, tgvf_ids, ev_answer_ids], dim=-1)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    weights = torch.zeros(input_ids.shape, dtype=torch.float32, device=device)
    base_len = int(item.input_ids.shape[-1])
    action_end = base_len + int(action_ids.shape[-1])
    ev_start = action_end + int(tgvf_ids.shape[-1])
    labels[:, base_len:action_end] = action_ids
    labels[:, ev_start:] = ev_answer_ids
    weights[:, base_len:action_end] = action_weights
    weights[:, ev_start:] = ev_answer_weights

    embeds = qwen_model.get_input_embeddings()(input_ids).detach().clone()
    embeds = _scatter_visual_embeds(
        embeds,
        token_indices=item.image_token_indices.to(device),
        visual_embeds=item.model_inputs["_v_merge"].to(device),
    )
    image_token_id = int(getattr(qwen_model.config, "image_token_id"))
    all_image_positions = torch.nonzero(input_ids[0] == image_token_id, as_tuple=False).view(-1)
    fvt_positions = all_image_positions[-int(d.shape[0]) :]
    embeds = _scatter_visual_embeds(embeds, token_indices=fvt_positions.to(device), visual_embeds=d)

    attention_mask_2d = torch.ones_like(input_ids)
    fvt_token_start = int(fvt_positions[0].detach().cpu().item())
    fvt_token_end = int(fvt_positions[-1].detach().cpu().item()) + 1
    mm_token_type_ids = _full_mm_token_type_ids(
        model=qwen_model,
        input_ids=input_ids,
        fvt_token_start=fvt_token_start,
        fvt_token_end=fvt_token_end,
        device=device,
    )
    image_grid_thw = torch.cat(
        [
            item.image_grid_thw.to(device=device, dtype=torch.long).view(-1, 3),
            item.image_grid_thw.to(device=device, dtype=torch.long).view(-1, 3),
        ],
        dim=0,
    )
    position_ids = _compute_qwen3_position_ids_for_sequence(
        model=qwen_model,
        input_ids=input_ids,
        attention_mask=attention_mask_2d,
        image_grid_thw=image_grid_thw,
        video_grid_thw=item.model_inputs.get("video_grid_thw"),
        mm_token_type_ids=mm_token_type_ids,
    )
    if position_ids is None:
        raise RuntimeError("Qwen3 position id computation failed for fast focus final")
    answer_query_start = None
    if answer_start >= 0:
        answer_query_start = ev_start + int(
            _encode_text(tokenizer, ev_answer_text[:answer_start], device).view(-1).numel()
        )
    if answer_query_start is None or answer_query_start <= ev_start:
        raise RuntimeError("Stage2 Matrix-CE readout span is empty or missing")
    matrix_ce_labels = torch.full_like(input_ids, IGNORE_INDEX)
    matrix_ce_labels[:, ev_start:answer_query_start] = input_ids[:, ev_start:answer_query_start]
    block_query_end = original_image_mask_block_query_end(
        answer_query_start=answer_query_start,
        scope=mask_original_image_after_tgvf_scope,
    )
    mask_active = sample_original_image_mask_active(
        enabled=mask_original_image_after_tgvf,
        probability=mask_original_image_after_tgvf_prob,
        device=device,
    )
    if mask_active:
        attention_mask = _weak_strict_mask_b1(
            attention_mask_2d=attention_mask_2d,
            original_image_token_indices=item.image_token_indices.to(device),
            block_query_start=action_end,
            block_query_end=block_query_end,
            dtype=embeds.dtype,
        )
        mask_mode = "weak_strict_original_image_keys_4d" if block_query_end is None else "weak_strict_original_image_keys_4d_evidence_only_answer_unmasked"
    else:
        attention_mask = _causal_mask_b1(attention_mask_2d=attention_mask_2d, dtype=embeds.dtype)
        mask_mode = "standard_4d_causal"
    prepared_d_deepstack = _prepare_d_deepstack_for_stage2(
        d_deepstack_visual_embeds,
        token_count=int(d.shape[0]),
        device=device,
        dtype=embeds.dtype,
    )
    return _FocusPrepared(
        sample=item.sample,
        item=item,
        action_ids=action_ids,
        action_weights=action_weights,
        target_start=target_span[0],
        target_end=target_span[1],
        target_hidden_states=target_hidden_states,
        pre_merge_visual_tokens=pre_merge_visual_tokens,
        merged_visual_tokens=merged_visual_tokens,
        foveated_visual_tokens=d,
        value_span_matched=value_span_matched,
        final_input_ids=input_ids,
        final_labels=labels,
        matrix_ce_labels=matrix_ce_labels,
        final_weights=weights,
        final_inputs_embeds=embeds,
        final_attention_mask_2d=attention_mask_2d,
        final_attention_mask=attention_mask,
        final_position_ids=position_ids,
        final_mm_token_type_ids=mm_token_type_ids,
        fvt_token_indices=fvt_positions.to(device),
        d_token_indices=fvt_positions.to(device) if prepared_d_deepstack is not None else None,
        d_deepstack_visual_embeds=prepared_d_deepstack,
        masked_image_key_count=int(item.image_token_indices.numel()) if mask_active else 0,
        image_key_mask_active=bool(mask_active),
        mask_mode=mask_mode,
        fvt_shape=list(d.shape),
        target_hidden_shape=list(target_hidden_states.shape),
        deepstack_feature_shapes=(
            _deepstack_feature_shapes(item) if deepstack_enabled else None
        ),
    )


def _prepare_multi_focus_final(
    *,
    qwen_model: Any,
    qwen_forward_model: Any,
    processor: Any,
    foveal_module: torch.nn.Module,
    item: _BaseItem,
    loss_weights: Stage2LossWeights,
    device: torch.device | str,
    hidden_state_index: int,
    max_image_resolution: int | None,
    mask_original_image_after_tgvf: bool,
    mask_original_image_after_tgvf_prob: float,
    mask_original_image_after_tgvf_scope: str,
    protocol: TGVFProtocol,
    deepstack_enabled: bool,
) -> _FocusPrepared:
    tokenizer = processor.tokenizer
    steps = _multi_focus_steps(item.sample)
    if len(steps) < 2:
        raise RuntimeError("multi_focus sample requires at least two focus steps")
    steps = steps[:2]
    pre = item.model_inputs["_v_pre"].to(device)
    merged = item.model_inputs["_v_merge"].to(device)
    mask_active = sample_original_image_mask_active(
        enabled=mask_original_image_after_tgvf,
        probability=mask_original_image_after_tgvf_prob,
        device=device,
    )

    action_ids_list: list[torch.Tensor] = []
    action_weights_list: list[torch.Tensor] = []
    action_target_spans: list[tuple[int, int]] = []
    readout_ids_list: list[torch.Tensor] = []
    readout_weights_list: list[torch.Tensor] = []
    tgvf_ids_list: list[torch.Tensor] = []
    d_list: list[torch.Tensor] = []
    target_hidden_list: list[torch.Tensor] = []
    d_deepstack_list: list[list[torch.Tensor] | None] = []
    value_span_matched = False

    prefix_ids = item.input_ids
    action1_ids, action1_weights, action1_span = _multi_action_ids_weights(
        tokenizer=tokenizer,
        target=steps[0]["target"],
        pre_focus_think=steps[0].get("pre_think") or item.sample.pre_focus_think,
        include_think=True,
        loss_weights=loss_weights,
        protocol=protocol,
        device=device,
    )
    prefix1_ids = torch.cat([prefix_ids, action1_ids], dim=-1)
    prefix1_embeds = _multi_embeds_with_visual_groups(
        qwen_model=qwen_forward_model,
        input_ids=prefix1_ids,
        original_image_positions=item.image_token_indices.to(device),
        visual_groups=[item.model_inputs["_v_merge"].to(device)],
        device=device,
    )
    prefix1_out = _multi_forward(
        qwen_model=qwen_model,
        qwen_forward_model=qwen_forward_model,
        item=item,
        input_ids=prefix1_ids,
        inputs_embeds=prefix1_embeds,
        device=device,
        image_grid_repeats=1,
        mask_original_image_after_tgvf=False,
        block_query_start=None,
        hidden_state_index=hidden_state_index,
        deepstack_enabled=deepstack_enabled,
    )
    base_len = int(item.input_ids.shape[-1])
    h1 = prefix1_out.hidden_states[hidden_state_index][0, base_len + action1_span[0] : base_len + action1_span[1]]
    out1 = foveal_module(
        target_hidden_states=h1,
        pre_merge_visual_tokens=pre,
        metadata={
            "target": steps[0]["target"],
            "stage": "tgvf_v3_stage2_fast_multi",
            "focus_step": 1,
            "qwen_model": qwen_model,
            "processor": processor,
            "image": _image_input(item.sample.image, max_image_resolution=max_image_resolution),
            "question": item.sample.prompt_question,
            "device": device,
            "deepstack_pre_merge_visual_tokens": _item_deepstack_pre_merge_features(item),
        },
    )
    out1 = finalize_tgvf_output_with_frozen_qwen_merger(qwen_model, out1)
    d1 = out1.foveated_visual_tokens
    d1_deepstack = out1.deepstack_visual_embeds
    d_list.append(d1)
    d_deepstack_list.append(d1_deepstack)
    target_hidden_list.append(h1)
    action_ids_list.append(action1_ids)
    action_weights_list.append(action1_weights)
    action_target_spans.append(action1_span)

    tgvf1_ids = _multi_tgvf_ids(processor, qwen_model, int(d1.shape[0]), protocol=protocol, device=device)
    readout1_ids, readout1_weights, matched1 = _multi_intermediate_readout_ids_weights(
        tokenizer=tokenizer,
        evidence=steps[0].get("post_think") or steps[0]["evidence_description"],
        value_span=steps[0].get("value_span_text"),
        loss_weights=loss_weights,
        protocol=protocol,
        device=device,
    )
    value_span_matched = value_span_matched or matched1
    tgvf_ids_list.append(tgvf1_ids)
    readout_ids_list.append(readout1_ids)
    readout_weights_list.append(readout1_weights)

    action2_ids, action2_weights, action2_span = _multi_action_ids_weights(
        tokenizer=tokenizer,
        target=steps[1]["target"],
        pre_focus_think=None,
        include_think=False,
        loss_weights=loss_weights,
        protocol=protocol,
        device=device,
    )
    prefix2_before_action = torch.cat([item.input_ids, action1_ids, tgvf1_ids, readout1_ids], dim=-1)
    prefix2_ids = torch.cat([prefix2_before_action, action2_ids], dim=-1)
    prefix2_embeds = _multi_embeds_with_visual_groups(
        qwen_model=qwen_forward_model,
        input_ids=prefix2_ids,
        original_image_positions=item.image_token_indices.to(device),
        visual_groups=[item.model_inputs["_v_merge"].to(device), d1.to(device)],
        device=device,
    )
    action1_end = int(item.input_ids.shape[-1]) + int(action1_ids.shape[-1])
    prefix2_out = _multi_forward(
        qwen_model=qwen_model,
        qwen_forward_model=qwen_forward_model,
        item=item,
        input_ids=prefix2_ids,
        inputs_embeds=prefix2_embeds,
        device=device,
        image_grid_repeats=2,
        mask_original_image_after_tgvf=mask_active,
        block_query_start=action1_end,
        hidden_state_index=hidden_state_index,
        deepstack_enabled=deepstack_enabled,
    )
    action2_offset = int(prefix2_before_action.shape[-1])
    h2 = prefix2_out.hidden_states[hidden_state_index][0, action2_offset + action2_span[0] : action2_offset + action2_span[1]]
    out2 = foveal_module(
        target_hidden_states=h2,
        pre_merge_visual_tokens=pre,
        metadata={
            "target": steps[1]["target"],
            "stage": "tgvf_v3_stage2_fast_multi",
            "focus_step": 2,
            "qwen_model": qwen_model,
            "processor": processor,
            "image": _image_input(item.sample.image, max_image_resolution=max_image_resolution),
            "question": item.sample.prompt_question,
            "device": device,
            "deepstack_pre_merge_visual_tokens": _item_deepstack_pre_merge_features(item),
        },
    )
    out2 = finalize_tgvf_output_with_frozen_qwen_merger(qwen_model, out2)
    d2 = out2.foveated_visual_tokens
    d2_deepstack = out2.deepstack_visual_embeds
    d_list.append(d2)
    d_deepstack_list.append(d2_deepstack)
    target_hidden_list.append(h2)
    action_ids_list.append(action2_ids)
    action_weights_list.append(action2_weights)
    action_target_spans.append(action2_span)

    tgvf2_ids = _multi_tgvf_ids(processor, qwen_model, int(d2.shape[0]), protocol=protocol, device=device)
    readout2_ids, readout2_weights, matched2, final_answer_token_start = _multi_final_readout_ids_weights(
        tokenizer=tokenizer,
        evidence=steps[1].get("post_think") or steps[1]["evidence_description"],
        answer=item.sample.answer,
        value_span=steps[1].get("value_span_text") or item.sample.value_span_text,
        loss_weights=loss_weights,
        protocol=protocol,
        device=device,
    )
    value_span_matched = value_span_matched or matched2
    tgvf_ids_list.append(tgvf2_ids)
    readout_ids_list.append(readout2_ids)
    readout_weights_list.append(readout2_weights)

    input_ids = torch.cat(
        [
            item.input_ids,
            action1_ids,
            tgvf1_ids,
            readout1_ids,
            action2_ids,
            tgvf2_ids,
            readout2_ids,
        ],
        dim=-1,
    )
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    weights = torch.zeros(input_ids.shape, dtype=torch.float32, device=device)
    cursor = int(item.input_ids.shape[-1])
    for ids, token_weights, label_active in (
        (action1_ids, action1_weights, True),
        (tgvf1_ids, None, False),
        (readout1_ids, readout1_weights, True),
        (action2_ids, action2_weights, True),
        (tgvf2_ids, None, False),
        (readout2_ids, readout2_weights, True),
    ):
        length = int(ids.shape[-1])
        if label_active:
            labels[:, cursor : cursor + length] = ids
            weights[:, cursor : cursor + length] = token_weights
        cursor += length

    embeds = _multi_embeds_with_visual_groups(
        qwen_model=qwen_model,
        input_ids=input_ids,
        original_image_positions=item.image_token_indices.to(device),
        visual_groups=[item.model_inputs["_v_merge"].to(device), d1.to(device), d2.to(device)],
        device=device,
    )
    image_token_id = int(getattr(qwen_model.config, "image_token_id"))
    all_image_positions = torch.nonzero(input_ids[0] == image_token_id, as_tuple=False).view(-1)
    d_token_indices = all_image_positions[int(item.image_token_indices.numel()) :].to(device)
    d_deepstack_visual_embeds = _combine_d_deepstack_groups_for_stage2(
        d_deepstack_list,
        token_counts=[int(d.shape[0]) for d in d_list],
        device=device,
        dtype=embeds.dtype,
    )
    attention_mask_2d = torch.ones_like(input_ids)
    mm_token_type_ids = _image_token_type_ids(qwen_model, input_ids, device=device)
    image_grid_thw = _repeat_image_grid(item.image_grid_thw, repeats=3, device=device)
    position_ids = _compute_qwen3_position_ids_for_sequence(
        model=qwen_model,
        input_ids=input_ids,
        attention_mask=attention_mask_2d,
        image_grid_thw=image_grid_thw,
        video_grid_thw=item.model_inputs.get("video_grid_thw"),
        mm_token_type_ids=mm_token_type_ids,
    )
    if position_ids is None:
        raise RuntimeError("Qwen3 position id computation failed for multi-focus final")
    final_readout2_start = (
        int(item.input_ids.shape[-1])
        + int(action1_ids.shape[-1])
        + int(tgvf1_ids.shape[-1])
        + int(readout1_ids.shape[-1])
        + int(action2_ids.shape[-1])
        + int(tgvf2_ids.shape[-1])
    )
    answer_query_start = (
        final_readout2_start + int(final_answer_token_start)
        if final_answer_token_start is not None
        else None
    )
    block_query_end = original_image_mask_block_query_end(
        answer_query_start=answer_query_start,
        scope=mask_original_image_after_tgvf_scope,
    )
    if mask_active:
        attention_mask = _weak_strict_mask_b1(
            attention_mask_2d=attention_mask_2d,
            original_image_token_indices=item.image_token_indices.to(device),
            block_query_start=action1_end,
            block_query_end=block_query_end,
            dtype=embeds.dtype,
        )
        mask_mode = "weak_strict_original_image_keys_4d" if block_query_end is None else "weak_strict_original_image_keys_4d_evidence_only_answer_unmasked"
    else:
        attention_mask = _causal_mask_b1(attention_mask_2d=attention_mask_2d, dtype=embeds.dtype)
        mask_mode = "standard_4d_causal"
    target_shapes = [list(h.shape) for h in target_hidden_list]
    return _FocusPrepared(
        sample=item.sample,
        item=item,
        action_ids=torch.cat(action_ids_list, dim=-1),
        action_weights=torch.cat(action_weights_list, dim=-1),
        target_start=action_target_spans[0][0],
        target_end=action_target_spans[0][1],
        target_hidden_states=target_hidden_list[-1],
        pre_merge_visual_tokens=pre,
        merged_visual_tokens=merged,
        foveated_visual_tokens=d_list[-1],
        value_span_matched=value_span_matched,
        final_input_ids=input_ids,
        final_labels=labels,
        matrix_ce_labels=None,
        final_weights=weights,
        final_inputs_embeds=embeds,
        final_attention_mask_2d=attention_mask_2d,
        final_attention_mask=attention_mask,
        final_position_ids=position_ids,
        final_mm_token_type_ids=mm_token_type_ids,
        fvt_token_indices=d_token_indices,
        d_token_indices=d_token_indices if d_deepstack_visual_embeds is not None else None,
        d_deepstack_visual_embeds=d_deepstack_visual_embeds,
        masked_image_key_count=int(item.image_token_indices.numel()) if mask_active else 0,
        image_key_mask_active=bool(mask_active),
        mask_mode=mask_mode,
        fvt_shape=[list(d.shape) for d in d_list],
        target_hidden_shape=target_shapes,
        deepstack_feature_shapes=(
            _deepstack_feature_shapes(item) if deepstack_enabled else None
        ),
    )


def _prepare_no_focus_final(
    *,
    processor: Any,
    item: _BaseItem,
    loss_weights: Stage2LossWeights,
    device: torch.device | str,
    protocol: TGVFProtocol,
) -> _NoFocusPrepared:
    output_text = render_no_focus_output_text(
        item.sample.answer,
        protocol=protocol,
        think_text=item.sample.no_focus_think,
        append_im_end=protocol_uses_tool_observation(protocol),
    )
    answer_start = (
        output_text.find(ANSWER_START)
        if protocol == "legacy_v3_tags"
        else output_text.rfind(item.sample.answer.strip())
    )
    state_end = (
        len(f"{EVIDENCE_STATE_START}{SUFFICIENT_EVIDENCE}{EVIDENCE_STATE_END}")
        if protocol == "legacy_v3_tags"
        else answer_start
    )
    output_ids, output_weights = _weighted_stage2_tokens(
        processor.tokenizer,
        output_text,
        [
            (0, state_end, loss_weights.no_focus_evidence_state),
            (answer_start, len(output_text), loss_weights.no_focus_answer),
        ],
        device=device,
        default_weight=1.0,
    )
    input_ids = torch.cat([item.input_ids, output_ids], dim=-1)
    attention_mask = torch.cat(
        [item.attention_mask, torch.ones_like(output_ids, dtype=item.attention_mask.dtype)],
        dim=-1,
    )
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    weights = torch.zeros(input_ids.shape, dtype=torch.float32, device=device)
    base_len = int(item.input_ids.shape[-1])
    labels[:, base_len:] = output_ids
    weights[:, base_len:] = output_weights
    value_span_matched = bool(item.sample.value_span_text and item.sample.value_span_text in output_text)
    return _NoFocusPrepared(
        sample=item.sample,
        item=item,
        final_input_ids=input_ids,
        final_labels=labels,
        final_weights=weights,
        final_attention_mask=attention_mask,
        value_span_matched=value_span_matched,
    )


def _multi_focus_steps(sample: TGVFv3Stage2Sample) -> list[dict[str, Any]]:
    steps = []
    for raw in sample.focus_steps:
        target = str(raw.get("target") or raw.get("focus_text") or "").strip()
        evidence = str(raw.get("evidence_description") or raw.get("focused_evidence") or "").strip()
        if target and evidence:
            steps.append(
                {
                    "target": target,
                    "evidence_description": evidence,
                    "pre_think": raw.get("pre_think"),
                    "post_think": raw.get("post_think"),
                    "value_span_text": raw.get("value_span_text"),
                }
            )
    return steps


def _multi_action_ids_weights(
    *,
    tokenizer: Any,
    target: str,
    pre_focus_think: str | None = None,
    include_think: bool = True,
    loss_weights: Stage2LossWeights,
    protocol: TGVFProtocol,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    action_text = render_focus_action_text(
        target,
        protocol=protocol,
        pre_focus_think=pre_focus_think,
        include_think=include_think,
        append_im_end=protocol_uses_tool_observation(protocol),
    )
    target_char_span = focus_target_char_span(action_text, protocol=protocol)
    if target_char_span is None:
        raise RuntimeError(f"Could not locate multi-focus target char span for: {target!r}")
    state_weight_end = (
        len(f"{EVIDENCE_STATE_START}{NEED_LOCAL_EVIDENCE}{EVIDENCE_STATE_END}")
        if protocol == "legacy_v3_tags"
        else max(0, target_char_span[0] - len(protocol_focus_tokens(protocol)[0]))
    )
    action_ids, action_weights = _weighted_stage2_tokens(
        tokenizer,
        action_text,
        [
            (0, state_weight_end, loss_weights.evidence_state),
            (target_char_span[0], target_char_span[1], loss_weights.focus_target),
        ],
        device=device,
        default_weight=1.0,
    )
    span = _find_focus_target_span_in_forced_ids(
        tokenizer,
        action_ids.view(-1).tolist(),
        protocol=protocol,
    )
    if span is None:
        raise RuntimeError(f"Could not locate multi-focus target token span for: {target!r}")
    target_start, target_end, _target_text = span
    return action_ids, action_weights, (target_start, target_end)


def _multi_tgvf_ids(
    processor: Any,
    qwen_model: Any,
    num_fvt_tokens: int,
    *,
    protocol: TGVFProtocol,
    device: torch.device | str,
) -> torch.Tensor:
    tgvf_prefix, tgvf_suffix = render_tgvf_prefix_suffix(
        protocol=protocol,
        include_leading_im_end=not protocol_uses_tool_observation(protocol),
    )
    return _bracketed_visual_token_ids(
        processor,
        qwen_model,
        num_fvt_tokens=int(num_fvt_tokens),
        prefix=tgvf_prefix,
        suffix=tgvf_suffix,
        device=device,
    ).view(1, -1)


def _multi_intermediate_readout_ids_weights(
    *,
    tokenizer: Any,
    evidence: str,
    value_span: str | None,
    loss_weights: Stage2LossWeights,
    protocol: TGVFProtocol,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    evidence = evidence.strip()
    if protocol == "legacy_v3_tags":
        text = f"{EVIDENCE_START}{evidence}{EVIDENCE_END}\n"
        evidence_start = text.find(EVIDENCE_START) + len(EVIDENCE_START)
        evidence_end = text.find(EVIDENCE_END, evidence_start)
    elif protocol_uses_evidence_tags(protocol):
        text = f"<|evidence_start|>{evidence}<|evidence_end|>\n"
        evidence_start = text.find(evidence)
        evidence_end = evidence_start + len(evidence)
    else:
        text = f"<think>\n{evidence}\n</think>\n"
        evidence_start = text.find(evidence)
        evidence_end = evidence_start + len(evidence)
    spans: list[tuple[int, int, float]] = [(evidence_start, evidence_end, loss_weights.evidence)]
    matched = False
    if value_span:
        value_start = text.find(str(value_span))
        if evidence_start <= value_start < evidence_end:
            spans.append((value_start, value_start + len(str(value_span)), loss_weights.value_span))
            matched = True
    ids, weights = _weighted_stage2_tokens(
        tokenizer,
        text,
        spans,
        device=device,
        default_weight=1.0,
    )
    return ids, weights, matched


def _multi_final_readout_ids_weights(
    *,
    tokenizer: Any,
    evidence: str,
    answer: str,
    value_span: str | None,
    loss_weights: Stage2LossWeights,
    protocol: TGVFProtocol,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, bool, int | None]:
    text = render_focus_readout_answer_text(
        evidence_description=evidence,
        answer=answer,
        protocol=protocol,
        append_im_end=protocol_uses_tool_observation(protocol),
    )
    if protocol == "legacy_v3_tags":
        evidence_start = text.find(EVIDENCE_START) + len(EVIDENCE_START)
        evidence_end = text.find(EVIDENCE_END, evidence_start)
        answer_start = text.find(ANSWER_START)
    else:
        evidence_start = text.find(evidence.strip())
        evidence_end = evidence_start + len(evidence.strip())
        answer_start = text.rfind(answer.strip())
    spans: list[tuple[int, int, float]] = [
        (evidence_start, evidence_end, loss_weights.evidence),
        (answer_start, len(text), loss_weights.answer),
    ]
    matched = False
    if value_span:
        value_start = text.find(str(value_span))
        if evidence_start <= value_start < evidence_end:
            spans.append((value_start, value_start + len(str(value_span)), loss_weights.value_span))
            matched = True
    ids, weights = _weighted_stage2_tokens(
        tokenizer,
        text,
        spans,
        device=device,
        default_weight=1.0,
    )
    answer_token_start = (
        int(_encode_text(tokenizer, text[:answer_start], device).view(-1).numel())
        if answer_start >= 0
        else None
    )
    return ids, weights, matched, answer_token_start


def _multi_embeds_with_visual_groups(
    *,
    qwen_model: Any,
    input_ids: torch.Tensor,
    original_image_positions: torch.Tensor,
    visual_groups: list[torch.Tensor],
    device: torch.device | str,
) -> torch.Tensor:
    embeds = qwen_model.get_input_embeddings()(input_ids).detach().clone()
    image_token_id = int(getattr(qwen_model.config, "image_token_id"))
    all_image_positions = torch.nonzero(input_ids[0] == image_token_id, as_tuple=False).view(-1).to(device)
    cursor = 0
    for group_index, visual in enumerate(visual_groups):
        visual = visual.to(device=device, dtype=embeds.dtype)
        count = int(visual.shape[0])
        if group_index == 0:
            positions = original_image_positions.to(device=device, dtype=torch.long)
            if int(positions.numel()) != count:
                raise RuntimeError(f"original visual token count mismatch: {int(positions.numel())} vs {count}")
            cursor = int(positions.numel())
        else:
            positions = all_image_positions[cursor : cursor + count]
            if int(positions.numel()) != count:
                raise RuntimeError(f"multi-focus FVT placeholder count mismatch: {int(positions.numel())} vs {count}")
            cursor += count
        embeds = _scatter_visual_embeds(
            embeds,
            token_indices=positions,
            visual_embeds=visual,
        )
    return embeds


def _multi_forward(
    *,
    qwen_model: Any,
    qwen_forward_model: Any,
    item: _BaseItem,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    device: torch.device | str,
    image_grid_repeats: int,
    mask_original_image_after_tgvf: bool,
    block_query_start: int | None,
    hidden_state_index: int,
    deepstack_enabled: bool,
) -> Any:
    del hidden_state_index
    attention_mask_2d = torch.ones_like(input_ids)
    mm_token_type_ids = _image_token_type_ids(qwen_model, input_ids, device=device)
    position_ids = _compute_qwen3_position_ids_for_sequence(
        model=qwen_model,
        input_ids=input_ids,
        attention_mask=attention_mask_2d,
        image_grid_thw=_repeat_image_grid(item.image_grid_thw, repeats=image_grid_repeats, device=device),
        video_grid_thw=item.model_inputs.get("video_grid_thw"),
        mm_token_type_ids=mm_token_type_ids,
    )
    if position_ids is None:
        raise RuntimeError("Qwen3 position id computation failed for multi-focus prefix")
    if mask_original_image_after_tgvf and block_query_start is not None:
        attention_mask = _weak_strict_mask_b1(
            attention_mask_2d=attention_mask_2d,
            original_image_token_indices=item.image_token_indices.to(device),
            block_query_start=block_query_start,
            block_query_end=None,
            dtype=inputs_embeds.dtype,
        )
    else:
        attention_mask = attention_mask_2d
    deepstack_kwargs = _single_deepstack_inputs(
        item=item,
        sequence_length=int(input_ids.shape[-1]),
        device=device,
        dtype=inputs_embeds.dtype,
    ) if deepstack_enabled else {}
    return _qwen_manual_forward_with_optional_deepstack(
        qwen_forward_model,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        mm_token_type_ids=mm_token_type_ids,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
        **deepstack_kwargs,
    )


def _qwen_manual_forward_with_optional_deepstack(
    qwen_forward_model: Any,
    *,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    use_cache: bool,
    output_hidden_states: bool,
    return_dict: bool,
    visual_pos_masks: torch.Tensor | None = None,
    deepstack_visual_embeds: list[torch.Tensor] | None = None,
) -> Any:
    if visual_pos_masks is None or deepstack_visual_embeds is None:
        return qwen_forward_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            mm_token_type_ids=mm_token_type_ids,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    causal_lm = _unwrap_qwen3_causal_lm(qwen_forward_model)
    vl_model = getattr(causal_lm, "model", None)
    language_model = getattr(vl_model, "language_model", None)
    lm_head = getattr(causal_lm, "lm_head", None)
    if language_model is None or lm_head is None:
        raise RuntimeError("Qwen3 DeepStack training requires model.language_model and lm_head")
    outputs = language_model(
        input_ids=None,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=use_cache,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
    )
    return SimpleNamespace(
        logits=lm_head(outputs.last_hidden_state),
        hidden_states=getattr(outputs, "hidden_states", None),
        past_key_values=getattr(outputs, "past_key_values", None),
    )


def _unwrap_qwen3_causal_lm(model: Any) -> Any:
    if hasattr(model, "get_base_model"):
        try:
            return model.get_base_model()
        except Exception:
            pass
    base_model = getattr(model, "base_model", None)
    nested = getattr(base_model, "model", None)
    if nested is not None:
        return nested
    return model


def _single_deepstack_inputs(
    *,
    item: _BaseItem,
    sequence_length: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> dict[str, Any]:
    return _batched_deepstack_inputs(
        items=[item],
        max_len=int(sequence_length),
        device=device,
        dtype=dtype,
    )


def _batched_deepstack_inputs(
    *,
    items: list[_BaseItem],
    max_len: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> dict[str, Any]:
    if not items:
        raise ValueError("DeepStack batch requires at least one item")
    first_features = _item_deepstack_features(items[0])
    if not first_features:
        raise RuntimeError("Qwen3 image feature output did not include DeepStack features")
    visual_pos_masks = []
    per_layer: list[list[torch.Tensor]] = [[] for _ in first_features]
    for item in items:
        features = _item_deepstack_features(item)
        if len(features) != len(first_features):
            raise RuntimeError("DeepStack feature layer count mismatch across batch")
        indices = item.image_token_indices.to(device=device, dtype=torch.long).view(-1)
        mask = torch.zeros((1, int(max_len)), dtype=torch.bool, device=device)
        if int(indices.numel()) == 0:
            raise RuntimeError("DeepStack training requires original image token indices")
        if int(indices.max().detach().cpu().item()) >= int(max_len):
            raise RuntimeError("DeepStack original image token index exceeds padded sequence")
        mask[0, indices] = True
        visual_pos_masks.append(mask)
        for layer_index, feature in enumerate(features):
            if int(feature.shape[0]) != int(indices.numel()):
                raise RuntimeError(
                    "DeepStack feature token count mismatch: "
                    f"feature={int(feature.shape[0])} original={int(indices.numel())}"
                )
            per_layer[layer_index].append(feature.to(device=device, dtype=dtype))
    return {
        "visual_pos_masks": torch.cat(visual_pos_masks, dim=0),
        "deepstack_visual_embeds": [
            torch.cat(layer_features, dim=0)
            for layer_features in per_layer
        ],
    }


def _batched_focus_deepstack_inputs(
    *,
    items: list[_FocusPrepared],
    max_len: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> dict[str, Any]:
    if not items:
        raise ValueError("DeepStack focus batch requires at least one item")
    first_features = _item_deepstack_features(items[0].item)
    if not first_features:
        raise RuntimeError("Qwen3 image feature output did not include DeepStack features")
    visual_pos_masks = []
    per_layer: list[list[torch.Tensor]] = [[] for _ in first_features]
    for prepared in items:
        original_features = _item_deepstack_features(prepared.item)
        if len(original_features) != len(first_features):
            raise RuntimeError("DeepStack feature layer count mismatch across focus batch")
        original_indices = prepared.item.image_token_indices.to(device=device, dtype=torch.long).view(-1)
        d_indices = (
            prepared.d_token_indices.to(device=device, dtype=torch.long).view(-1)
            if prepared.d_token_indices is not None
            else torch.empty(0, dtype=torch.long, device=device)
        )
        indices = torch.cat([original_indices, d_indices], dim=0)
        if int(indices.numel()) == 0:
            raise RuntimeError("DeepStack focus final requires at least one visual token index")
        if int(indices.max().detach().cpu().item()) >= int(max_len):
            raise RuntimeError("DeepStack focus token index exceeds padded sequence")
        sorted_indices, sort_order = torch.sort(indices)
        mask = torch.zeros((1, int(max_len)), dtype=torch.bool, device=device)
        mask[0, sorted_indices] = True
        visual_pos_masks.append(mask)
        for layer_index, original_feature in enumerate(original_features):
            if int(original_feature.shape[0]) != int(original_indices.numel()):
                raise RuntimeError(
                    "DeepStack original feature token count mismatch: "
                    f"feature={int(original_feature.shape[0])} original={int(original_indices.numel())}"
                )
            layer_parts = [original_feature.to(device=device, dtype=dtype)]
            if int(d_indices.numel()) > 0:
                if not prepared.d_deepstack_visual_embeds:
                    raise RuntimeError("D token positions require D DeepStack features")
                d_feature = prepared.d_deepstack_visual_embeds[layer_index]
                if int(d_feature.shape[0]) != int(d_indices.numel()):
                    raise RuntimeError(
                        "D DeepStack feature token count mismatch: "
                        f"feature={int(d_feature.shape[0])} d={int(d_indices.numel())}"
                    )
                layer_parts.append(d_feature.to(device=device, dtype=dtype))
            per_layer[layer_index].append(torch.cat(layer_parts, dim=0).index_select(0, sort_order))
    return {
        "visual_pos_masks": torch.cat(visual_pos_masks, dim=0),
        "deepstack_visual_embeds": [
            torch.cat(layer_features, dim=0)
            for layer_features in per_layer
        ],
    }


def _item_deepstack_features(item: _BaseItem) -> list[torch.Tensor]:
    features = item.model_inputs.get("_deepstack_visual_embeds") or []
    return [feature for feature in features if isinstance(feature, torch.Tensor)]


def _item_deepstack_pre_merge_features(item: _BaseItem) -> list[torch.Tensor]:
    features = item.model_inputs.get("_deepstack_pre_merge_visual_embeds") or []
    return [feature for feature in features if isinstance(feature, torch.Tensor)]


def _deepstack_feature_shapes(item: _BaseItem) -> list[list[int]]:
    return [list(feature.shape) for feature in _item_deepstack_features(item)]


def _prepare_d_deepstack_for_stage2(
    features: list[torch.Tensor] | None,
    *,
    token_count: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> list[torch.Tensor] | None:
    if not features:
        return None
    prepared = []
    for feature in features:
        if not isinstance(feature, torch.Tensor):
            raise TypeError("D DeepStack feature must be a torch.Tensor")
        if feature.ndim != 2:
            raise ValueError(f"D DeepStack feature must have shape [N, D], got {list(feature.shape)}")
        if int(feature.shape[0]) != int(token_count):
            raise ValueError(
                "D DeepStack feature token count mismatch: "
                f"feature={int(feature.shape[0])} d_tokens={int(token_count)}"
            )
        prepared.append(feature.to(device=device, dtype=dtype))
    return prepared


def _combine_d_deepstack_groups_for_stage2(
    groups: list[list[torch.Tensor] | None],
    *,
    token_counts: list[int],
    device: torch.device | str,
    dtype: torch.dtype,
) -> list[torch.Tensor] | None:
    active = [group for group in groups if group]
    if not active:
        return None
    if len(active) != len(groups):
        raise RuntimeError("multi-focus D DeepStack groups cannot mix enabled and disabled branches")
    layer_count = len(active[0])
    combined = []
    for group_index, group in enumerate(groups):
        if group is None or len(group) != layer_count:
            raise RuntimeError("multi-focus D DeepStack layer count mismatch")
        _prepare_d_deepstack_for_stage2(
            group,
            token_count=int(token_counts[group_index]),
            device=device,
            dtype=dtype,
        )
    for layer_index in range(layer_count):
        combined.append(
            torch.cat(
                [group[layer_index].to(device=device, dtype=dtype) for group in groups if group],
                dim=0,
            )
        )
    return combined


def _repeat_image_grid(
    image_grid_thw: torch.Tensor,
    *,
    repeats: int,
    device: torch.device | str,
) -> torch.Tensor:
    return torch.cat(
        [image_grid_thw.to(device=device, dtype=torch.long).view(-1, 3) for _ in range(int(repeats))],
        dim=0,
    )


def _pad_focus_batch(
    items: list[_FocusPrepared],
    *,
    device: torch.device | str,
    deepstack_enabled: bool,
) -> dict[str, Any]:
    max_len = max(int(item.final_input_ids.shape[-1]) for item in items)
    batch: dict[str, Any] = {
        "inputs_embeds": _pad_embeds([item.final_inputs_embeds for item in items], max_len=max_len, device=device),
        "labels": _pad_2d([item.final_labels for item in items], pad_value=IGNORE_INDEX, device=device),
        "loss_weights": _pad_2d_float([item.final_weights for item in items], pad_value=0.0, device=device),
        "attention_mask": _pad_4d_focus([item.final_attention_mask for item in items], max_len=max_len, device=device),
        "position_ids": _pad_position_ids([item.final_position_ids for item in items], max_len=max_len, device=device),
        "mm_token_type_ids": _pad_2d([item.final_mm_token_type_ids for item in items], pad_value=0, device=device),
    }
    if deepstack_enabled:
        batch.update(
            _batched_focus_deepstack_inputs(
                items=items,
                max_len=max_len,
                device=device,
                dtype=batch["inputs_embeds"].dtype,
            )
        )
    return batch


def _pad_native_batch(
    model: Any,
    input_ids_list: list[torch.Tensor],
    attention_list: list[torch.Tensor],
    model_inputs_list: list[dict[str, Any]],
    *,
    device: torch.device | str,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    batch = {
        "input_ids": _pad_2d(input_ids_list, pad_value=pad_token_id, device=device),
        "attention_mask": _pad_2d(attention_list, pad_value=0, device=device),
        "mm_token_type_ids": _pad_2d(
            [_image_token_type_ids(model, input_ids, device=device) for input_ids in input_ids_list],
            pad_value=0,
            device=device,
        ),
    }
    if all("pixel_values" in item for item in model_inputs_list):
        batch["pixel_values"] = torch.cat([item["pixel_values"].to(device) for item in model_inputs_list], dim=0)
    if all("image_grid_thw" in item for item in model_inputs_list):
        batch["image_grid_thw"] = torch.cat([item["image_grid_thw"].to(device) for item in model_inputs_list], dim=0)
    return batch


def _batched_position_ids(
    *,
    model: Any,
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    image_grid_thw_list: list[torch.Tensor],
    device: torch.device | str,
    max_len: int,
) -> torch.Tensor:
    positions = []
    for input_ids, attention_mask, image_grid in zip(input_ids_list, attention_mask_list, image_grid_thw_list, strict=True):
        mm_token_type_ids = _image_token_type_ids(model, input_ids, device=device)
        pos = _compute_qwen3_position_ids_for_sequence(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid.to(device),
            video_grid_thw=None,
            mm_token_type_ids=mm_token_type_ids,
        )
        if pos is None:
            raise RuntimeError("Qwen3 position id computation failed")
        positions.append(pos)
    return _pad_position_ids(positions, max_len=max_len, device=device)


def _image_token_type_ids(model: Any, input_ids: torch.Tensor, *, device: torch.device | str) -> torch.Tensor:
    image_token_id = getattr(getattr(model, "config", None), "image_token_id", None)
    token_type_ids = torch.zeros_like(input_ids, dtype=torch.long, device=device)
    if image_token_id is not None:
        token_type_ids = torch.where(
            input_ids.to(device) == int(image_token_id),
            torch.ones_like(token_type_ids),
            token_type_ids,
        )
    return token_type_ids


def _weak_strict_mask_b1(
    *,
    attention_mask_2d: torch.Tensor,
    original_image_token_indices: torch.Tensor,
    block_query_start: int,
    dtype: torch.dtype,
    block_query_end: int | None = None,
) -> torch.Tensor:
    seq_len = int(attention_mask_2d.shape[-1])
    device = attention_mask_2d.device
    min_value = torch.finfo(dtype).min
    mask = torch.zeros((1, 1, seq_len, seq_len), dtype=dtype, device=device)
    future = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)
    mask = mask.masked_fill(future.view(1, 1, seq_len, seq_len), min_value)
    key_padding = attention_mask_2d[:, None, None, :] == 0
    mask = mask.masked_fill(key_padding, min_value)
    if int(original_image_token_indices.numel()) > 0:
        query_range = torch.arange(seq_len, device=device)
        query_mask = query_range >= int(block_query_start)
        if block_query_end is not None:
            query_mask = query_mask & (query_range < int(block_query_end))
        query_indices = query_range[query_mask]
        if int(query_indices.numel()) > 0:
            mask[:, :, query_indices[:, None], original_image_token_indices.to(device)[None, :]] = min_value
    return mask


def _causal_mask_b1(*, attention_mask_2d: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return _weak_strict_mask_b1(
        attention_mask_2d=attention_mask_2d,
        original_image_token_indices=torch.empty(0, dtype=torch.long, device=attention_mask_2d.device),
        block_query_start=0,
        dtype=dtype,
    )


def _weighted_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_weights = weights[:, 1:].contiguous().to(dtype=shift_logits.dtype)
    flat_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view_as(shift_labels)
    valid = (shift_labels != IGNORE_INDEX) & (shift_weights > 0)
    denom = shift_weights[valid].sum().clamp_min(1.0)
    return (flat_loss * shift_weights).sum() / denom, float(denom.detach().cpu().item())


def _scatter_visual_embeds(
    embeds: torch.Tensor,
    *,
    token_indices: torch.Tensor,
    visual_embeds: torch.Tensor,
) -> torch.Tensor:
    if int(token_indices.numel()) != int(visual_embeds.shape[0]):
        raise ValueError(
            f"visual token count mismatch: positions={int(token_indices.numel())} embeds={int(visual_embeds.shape[0])}"
        )
    embeds[:, token_indices.to(embeds.device), :] = visual_embeds.to(device=embeds.device, dtype=embeds.dtype).unsqueeze(0)
    return embeds


def _split_visual_tensor(tensor: torch.Tensor, counts: list[int]) -> list[torch.Tensor]:
    if tensor.ndim == 3 and tensor.shape[0] == len(counts):
        return [tensor[index, : count] for index, count in enumerate(counts)]
    if int(tensor.shape[0]) != sum(counts):
        raise RuntimeError(f"Cannot split visual tensor shape {list(tensor.shape)} with counts {counts}")
    return list(torch.split(tensor, counts, dim=0))


def _pad_2d(tensors: list[torch.Tensor], *, pad_value: int, device: torch.device | str) -> torch.Tensor:
    max_len = max(int(t.shape[-1]) for t in tensors)
    rows = []
    for tensor in tensors:
        tensor = tensor.to(device)
        pad = max_len - int(tensor.shape[-1])
        if pad:
            tensor = F.pad(tensor, (0, pad), value=pad_value)
        rows.append(tensor)
    return torch.cat(rows, dim=0)


def _pad_2d_float(tensors: list[torch.Tensor], *, pad_value: float, device: torch.device | str) -> torch.Tensor:
    max_len = max(int(t.shape[-1]) for t in tensors)
    rows = []
    for tensor in tensors:
        tensor = tensor.to(device)
        pad = max_len - int(tensor.shape[-1])
        if pad:
            tensor = F.pad(tensor, (0, pad), value=pad_value)
        rows.append(tensor)
    return torch.cat(rows, dim=0)


def _pad_embeds(
    tensors: list[torch.Tensor],
    *,
    max_len: int,
    device: torch.device | str,
) -> torch.Tensor:
    rows = []
    for tensor in tensors:
        tensor = tensor.to(device)
        pad = max_len - int(tensor.shape[1])
        if pad:
            pad_tensor = torch.zeros(
                (1, pad, tensor.shape[-1]),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            tensor = torch.cat([tensor, pad_tensor], dim=1)
        rows.append(tensor)
    return torch.cat(rows, dim=0)


def _pad_position_ids(
    tensors: list[torch.Tensor],
    *,
    max_len: int,
    device: torch.device | str,
) -> torch.Tensor:
    rows = []
    for tensor in tensors:
        tensor = tensor.to(device)
        if tensor.ndim != 3 or tensor.shape[0] != 3:
            raise ValueError(f"Expected position_ids shape [3,1,L], got {list(tensor.shape)}")
        pad = max_len - int(tensor.shape[-1])
        if pad:
            tensor = F.pad(tensor, (0, pad), value=0)
        rows.append(tensor)
    return torch.cat(rows, dim=1)


def _pad_4d_focus(
    tensors: list[torch.Tensor],
    *,
    max_len: int,
    device: torch.device | str,
) -> torch.Tensor:
    rows = []
    for tensor in tensors:
        tensor = tensor.to(device)
        pad = max_len - int(tensor.shape[-1])
        if pad:
            min_value = torch.finfo(tensor.dtype).min
            padded = torch.full((1, 1, max_len, max_len), min_value, dtype=tensor.dtype, device=tensor.device)
            padded[:, :, : tensor.shape[-2], : tensor.shape[-1]] = tensor
            rows.append(padded)
        else:
            rows.append(tensor)
    return torch.cat(rows, dim=0)
