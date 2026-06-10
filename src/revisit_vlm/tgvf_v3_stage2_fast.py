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
    SUFFICIENT_EVIDENCE,
    TGVF_END,
    TGVF_START,
    _bracketed_visual_token_ids,
    _compute_qwen3_position_ids_for_sequence,
    _encode_text,
    _extract_vision_tensors,
    build_direct_messages,
    build_qwen3_inputs,
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
) -> TGVFv3Stage2StepOutput:
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

    focus_items = [item for item in base_items if item.sample.need_focus]
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
                )
            )

    no_focus_prepared = [
        _prepare_no_focus_final(
            processor=processor,
            item=item,
            loss_weights=loss_weights,
            device=device,
        )
        for item in no_focus_items
    ]

    zero = next(foveal_module.parameters()).new_zeros(())
    loss_focus = zero
    loss_no_focus = zero
    loss_manifold = zero
    focus_loss_tokens = 0.0
    no_focus_loss_tokens = 0.0

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
                "mask_mode": "weak_strict_original_image_keys_4d"
                if item.sample.need_focus and mask_original_image_after_tgvf
                else "standard_2d_causal",
                "image_key_mask_active": bool(item.sample.need_focus and mask_original_image_after_tgvf),
                "masked_image_key_count": getattr(item, "masked_image_key_count", 0),
                "fvt_shape": getattr(item, "fvt_shape", None),
                "target_hidden_shape": getattr(item, "target_hidden_shape", None),
                "value_span_matched": bool(item.value_span_matched),
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
            "no_focus_count": len(no_focus_prepared),
            "focus_sample_mask_active_rate": 1.0 if focus_prepared and mask_original_image_after_tgvf else 0.0,
            "no_focus_mask_active_rate": 0.0,
            "value_span_match_rate": (
                sum(1.0 for matched in value_matches if matched) / max(len(value_matches), 1)
                if value_matches
                else None
            ),
            "focus_loss_token_weight": focus_loss_tokens,
            "no_focus_loss_token_weight": no_focus_loss_tokens,
            "special_tokens_added": False,
            "tokenizer_resized": False,
            "markers_are_plain_text": True,
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
) -> list[tuple[_BaseItem, torch.Tensor, torch.Tensor, tuple[int, int], torch.Tensor]]:
    action_ids_list = []
    action_weights_list = []
    first_ids = []
    first_attention = []
    first_inputs = []
    spans = []
    for item in focus_items:
        action_text = (
            f"{EVIDENCE_STATE_START}{NEED_LOCAL_EVIDENCE}{EVIDENCE_STATE_END}\n"
            f"{FOCUS_START}{item.sample.target}{FOCUS_END}"
        )
        focus_start = action_text.find(FOCUS_START)
        action_ids, action_weights = _weighted_stage2_tokens(
            tokenizer,
            action_text,
            [
                (0, len(f"{EVIDENCE_STATE_START}{NEED_LOCAL_EVIDENCE}{EVIDENCE_STATE_END}"), 0.2),
                (focus_start, len(action_text), 1.5),
            ],
            device=device,
            default_weight=1.0,
        )
        span = _find_focus_target_span_in_forced_ids(tokenizer, action_ids.view(-1).tolist())
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
) -> _FocusPrepared:
    tokenizer = processor.tokenizer
    d = foveated_visual_tokens.to(device)
    tgvf_ids = _bracketed_visual_token_ids(
        processor,
        qwen_model,
        num_fvt_tokens=int(d.shape[0]),
        prefix=f"\n{TGVF_START}\n",
        suffix=f"\n{TGVF_END}\n",
        device=device,
    ).view(1, -1)
    ev_answer_text = (
        f"{EVIDENCE_START}{item.sample.evidence_description}{EVIDENCE_END}\n"
        f"{ANSWER_START}{item.sample.answer}{ANSWER_END}"
    )
    spans = [
        (ev_answer_text.find(EVIDENCE_START), ev_answer_text.find(EVIDENCE_END) + len(EVIDENCE_END), loss_weights.evidence),
        (ev_answer_text.find(ANSWER_START), len(ev_answer_text), loss_weights.answer),
    ]
    value_span_matched = False
    if item.sample.value_span_text:
        value_start = ev_answer_text.find(item.sample.value_span_text)
        evidence_inner_start = ev_answer_text.find(EVIDENCE_START) + len(EVIDENCE_START)
        evidence_inner_end = ev_answer_text.find(EVIDENCE_END, evidence_inner_start)
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
    if mask_original_image_after_tgvf:
        attention_mask = _weak_strict_mask_b1(
            attention_mask_2d=attention_mask_2d,
            original_image_token_indices=item.image_token_indices.to(device),
            block_query_start=action_end,
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


def _prepare_no_focus_final(
    *,
    processor: Any,
    item: _BaseItem,
    loss_weights: Stage2LossWeights,
    device: torch.device | str,
) -> _NoFocusPrepared:
    output_text = (
        f"{EVIDENCE_STATE_START}{SUFFICIENT_EVIDENCE}{EVIDENCE_STATE_END}\n"
        f"{ANSWER_START}{item.sample.answer}{ANSWER_END}"
    )
    state_end = len(f"{EVIDENCE_STATE_START}{SUFFICIENT_EVIDENCE}{EVIDENCE_STATE_END}")
    answer_start = output_text.find(ANSWER_START)
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
        query_indices = torch.arange(seq_len, device=device)
        query_indices = query_indices[query_indices >= int(block_query_start)]
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
