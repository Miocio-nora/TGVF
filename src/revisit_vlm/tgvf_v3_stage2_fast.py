from __future__ import annotations

from dataclasses import dataclass
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
    PROTOCOL_E_ACTION_EVIDENCE_SPECIAL,
    SUFFICIENT_EVIDENCE,
    TGVF_END,
    TGVF_START,
    TGVFProtocol,
    _bracketed_visual_token_ids,
    _compute_qwen3_position_ids_for_sequence,
    _encode_text,
    _extract_vision_tensors,
    build_direct_messages,
    build_qwen3_inputs,
    focus_target_char_span,
    normalize_tgvf_protocol,
    protocol_focus_tokens,
    render_focus_action_text,
    render_focus_readout_answer_text,
    render_no_focus_output_text,
    render_tgvf_prefix_suffix,
)
from revisit_vlm.tgvf_foveal import (
    FovealCrossAttentionOutput,
    finalize_tgvf_output_with_frozen_qwen_merger,
)
from revisit_vlm.tgvf_training import IGNORE_INDEX
from revisit_vlm.tgvf_v3_stage1 import (
    _find_focus_target_span_in_forced_ids,
    _full_mm_token_type_ids,
    _image_input,
    _move_tensors_for_stage1,
    _safe_visual_token_manifold_loss,
)
from revisit_vlm.tgvf_v3_stage2 import (
    Stage2LossWeights,
    TGVFv3Stage2Sample,
    TGVFv3Stage2StepOutput,
    merge_protocol_c_boundary_stats,
    protocol_c_boundary_token_accuracy,
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
    final_weights: torch.Tensor
    final_inputs_embeds: torch.Tensor
    final_attention_mask_2d: torch.Tensor
    final_attention_mask: torch.Tensor
    final_position_ids: torch.Tensor
    final_mm_token_type_ids: torch.Tensor
    masked_image_key_count: int
    fvt_shape: list[int]
    target_hidden_shape: list[int]


@dataclass
class _NoFocusPrepared:
    sample: TGVFv3Stage2Sample
    item: _BaseItem
    final_input_ids: torch.Tensor
    final_labels: torch.Tensor
    final_weights: torch.Tensor
    final_attention_mask: torch.Tensor
    value_span_matched: bool


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
    protocol: TGVFProtocol = "legacy_v3_tags",
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

    focus_prepared: list[_FocusPrepared] = []
    if focus_items:
        focus_hidden = _batched_focus_first_forward(
            qwen_model=qwen_model,
            qwen_forward_model=qwen_forward_model,
            tokenizer=tokenizer,
            focus_items=focus_items,
            device=device,
            hidden_state_index=hidden_state_index,
            loss_weights=loss_weights,
            protocol=protocol,
        )
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
                },
            )
            output = finalize_tgvf_output_with_frozen_qwen_merger(qwen_model, output)
            d = output.foveated_visual_tokens
            focus_prepared.append(
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
                    foveated_visual_tokens=d,
                    loss_weights=loss_weights,
                    device=device,
                    mask_original_image_after_tgvf=mask_original_image_after_tgvf,
                    protocol=protocol,
                )
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
                protocol=protocol,
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
        focus_batch = _pad_focus_batch(focus_prepared, device=device)
        outputs = qwen_forward_model(
            inputs_embeds=focus_batch["inputs_embeds"],
            attention_mask=focus_batch["attention_mask"],
            position_ids=focus_batch["position_ids"],
            mm_token_type_ids=focus_batch["mm_token_type_ids"],
            use_cache=False,
            return_dict=True,
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
                "mask_mode": (
                    "weak_strict_original_image_keys_4d_evidence_only_answer_unmasked"
                    if item.sample.need_focus
                    and mask_original_image_after_tgvf
                    and protocol == PROTOCOL_E_ACTION_EVIDENCE_SPECIAL
                    else "weak_strict_original_image_keys_4d"
                    if item.sample.need_focus and mask_original_image_after_tgvf
                    else "standard_2d_causal"
                ),
                "image_key_mask_active": bool(item.sample.need_focus and mask_original_image_after_tgvf),
                "masked_image_key_count": getattr(item, "masked_image_key_count", 0),
                "fvt_shape": getattr(item, "fvt_shape", None),
                "target_hidden_shape": getattr(item, "target_hidden_shape", None),
                "value_span_matched": bool(item.value_span_matched),
                "tgvf_protocol": protocol,
            }
        )
    return TGVFv3Stage2StepOutput(
        loss_total=loss_total,
        loss_focus=loss_focus,
        loss_no_focus=loss_no_focus,
        loss_visual_token_manifold=loss_manifold,
        debug={
            "fast_batched_stage2": True,
            "focus_count": len(focus_prepared),
            "single_focus_count": len(focus_items),
            "multi_focus_count": len(multi_focus_items),
            "no_focus_count": len(no_focus_prepared),
            "focus_sample_mask_active_rate": 1.0 if focus_prepared and mask_original_image_after_tgvf else 0.0,
            "no_focus_mask_active_rate": 0.0,
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
    image_output = qwen_model.get_image_features(
        pixel_values,
        image_grid_thw=image_grid_thw,
        output_hidden_states=True,
        return_dict=True,
    )
    v_pre, _v_merge = _extract_vision_tensors(image_output)
    if v_pre is None:
        raise RuntimeError("Qwen3 get_image_features did not return V_pre tensors")
    pre_splits = _split_visual_tensor(v_pre, [item.visual_pre_count for item in base_items])
    merge_splits = [_merge_pre_tokens_with_frozen_qwen(qwen_model, pre) for pre in pre_splits]
    for item, pre, merge in zip(base_items, pre_splits, merge_splits, strict=True):
        if int(merge.shape[0]) != item.visual_merge_count:
            raise RuntimeError(
                f"Merged visual token count mismatch for {item.sample.image}: "
                f"{int(merge.shape[0])} vs {item.visual_merge_count}"
            )
        item.model_inputs["_v_pre"] = pre.detach()
        item.model_inputs["_v_merge"] = merge.detach()


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
            append_im_end=protocol == PROTOCOL_C_TOOL_OBSERVATION,
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
    outputs = qwen_forward_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        mm_token_type_ids=mm_token_type_ids,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
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
    loss_weights: Stage2LossWeights,
    device: torch.device | str,
    mask_original_image_after_tgvf: bool,
    protocol: TGVFProtocol,
) -> _FocusPrepared:
    tokenizer = processor.tokenizer
    d = foveated_visual_tokens.to(device)
    tgvf_prefix, tgvf_suffix = render_tgvf_prefix_suffix(
        protocol=protocol,
        include_leading_im_end=protocol != PROTOCOL_C_TOOL_OBSERVATION,
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
    protocol_e_answer_query_start = None
    if protocol == PROTOCOL_E_ACTION_EVIDENCE_SPECIAL and answer_start >= 0:
        protocol_e_answer_query_start = ev_start + int(
            _encode_text(tokenizer, ev_answer_text[:answer_start], device).view(-1).numel()
        )
    if mask_original_image_after_tgvf:
        attention_mask = _weak_strict_mask_b1(
            attention_mask_2d=attention_mask_2d,
            original_image_token_indices=item.image_token_indices.to(device),
            block_query_start=action_end,
            block_query_end=protocol_e_answer_query_start,
            dtype=embeds.dtype,
        )
    else:
        attention_mask = attention_mask_2d
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
        final_weights=weights,
        final_inputs_embeds=embeds,
        final_attention_mask_2d=attention_mask_2d,
        final_attention_mask=attention_mask,
        final_position_ids=position_ids,
        final_mm_token_type_ids=mm_token_type_ids,
        masked_image_key_count=int(item.image_token_indices.numel()) if mask_original_image_after_tgvf else 0,
        fvt_shape=list(d.shape),
        target_hidden_shape=list(target_hidden_states.shape),
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
    protocol: TGVFProtocol,
) -> _FocusPrepared:
    tokenizer = processor.tokenizer
    steps = _multi_focus_steps(item.sample)
    if len(steps) < 2:
        raise RuntimeError("multi_focus sample requires at least two focus steps")
    steps = steps[:2]
    pre = item.model_inputs["_v_pre"].to(device)
    merged = item.model_inputs["_v_merge"].to(device)

    action_ids_list: list[torch.Tensor] = []
    action_weights_list: list[torch.Tensor] = []
    action_target_spans: list[tuple[int, int]] = []
    readout_ids_list: list[torch.Tensor] = []
    readout_weights_list: list[torch.Tensor] = []
    tgvf_ids_list: list[torch.Tensor] = []
    d_list: list[torch.Tensor] = []
    target_hidden_list: list[torch.Tensor] = []
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
        },
    )
    out1 = finalize_tgvf_output_with_frozen_qwen_merger(qwen_model, out1)
    d1 = out1.foveated_visual_tokens
    d_list.append(d1)
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
        mask_original_image_after_tgvf=mask_original_image_after_tgvf,
        block_query_start=action1_end,
        hidden_state_index=hidden_state_index,
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
        },
    )
    out2 = finalize_tgvf_output_with_frozen_qwen_merger(qwen_model, out2)
    d2 = out2.foveated_visual_tokens
    d_list.append(d2)
    target_hidden_list.append(h2)
    action_ids_list.append(action2_ids)
    action_weights_list.append(action2_weights)
    action_target_spans.append(action2_span)

    tgvf2_ids = _multi_tgvf_ids(processor, qwen_model, int(d2.shape[0]), protocol=protocol, device=device)
    readout2_ids, readout2_weights, matched2 = _multi_final_readout_ids_weights(
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
    if mask_original_image_after_tgvf:
        attention_mask = _weak_strict_mask_b1(
            attention_mask_2d=attention_mask_2d,
            original_image_token_indices=item.image_token_indices.to(device),
            block_query_start=action1_end,
            block_query_end=None,
            dtype=embeds.dtype,
        )
    else:
        attention_mask = attention_mask_2d
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
        final_weights=weights,
        final_inputs_embeds=embeds,
        final_attention_mask_2d=attention_mask_2d,
        final_attention_mask=attention_mask,
        final_position_ids=position_ids,
        final_mm_token_type_ids=mm_token_type_ids,
        masked_image_key_count=int(item.image_token_indices.numel()) if mask_original_image_after_tgvf else 0,
        fvt_shape=[list(d.shape) for d in d_list],
        target_hidden_shape=target_shapes,
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
        append_im_end=protocol == PROTOCOL_C_TOOL_OBSERVATION,
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
        include_leading_im_end=protocol != PROTOCOL_C_TOOL_OBSERVATION,
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
    elif protocol == PROTOCOL_E_ACTION_EVIDENCE_SPECIAL:
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
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    text = render_focus_readout_answer_text(
        evidence_description=evidence,
        answer=answer,
        protocol=protocol,
        append_im_end=protocol == PROTOCOL_C_TOOL_OBSERVATION,
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
    return ids, weights, matched


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
    return qwen_forward_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        mm_token_type_ids=mm_token_type_ids,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )


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


def _pad_focus_batch(items: list[_FocusPrepared], *, device: torch.device | str) -> dict[str, torch.Tensor]:
    max_len = max(int(item.final_input_ids.shape[-1]) for item in items)
    return {
        "inputs_embeds": _pad_embeds([item.final_inputs_embeds for item in items], max_len=max_len, device=device),
        "labels": _pad_2d([item.final_labels for item in items], pad_value=IGNORE_INDEX, device=device),
        "loss_weights": _pad_2d_float([item.final_weights for item in items], pad_value=0.0, device=device),
        "attention_mask": _pad_4d_focus([item.final_attention_mask for item in items], max_len=max_len, device=device),
        "position_ids": _pad_position_ids([item.final_position_ids for item in items], max_len=max_len, device=device),
        "mm_token_type_ids": _pad_2d([item.final_mm_token_type_ids for item in items], pad_value=0, device=device),
    }


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
