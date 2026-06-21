from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

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
    TGVF_END,
    TGVF_START,
    TGVFProtocol,
    Qwen3FocusCapture,
    _bracketed_visual_token_ids,
    _compute_qwen3_position_ids_for_sequence,
    _decode,
    _encode_text,
    build_focus_force_messages,
    build_qwen3_inputs,
    capture_focus_single_pass_qwen3,
    extract_qwen3_source_visual_geometry,
    focus_target_char_span,
    llm_hidden_dim,
    normalize_tgvf_protocol,
    protocol_focus_tokens,
    render_focus_action_text,
    render_stage1_readout_text,
    render_tgvf_prefix_suffix,
    tap_qwen3_vision_features,
)
from revisit_vlm.tgvf_foveal import finalize_tgvf_output_with_frozen_qwen_merger
from revisit_vlm.tgvf_training import (
    IGNORE_INDEX,
    LossWeights,
    TGVFTrainStepOutput,
    attention_diagnostics,
    fvt_norm_diagnostics,
    same_image_negative_groups,
    same_image_negative_loss,
    same_image_negative_matrix_ce_loss,
    same_image_negative_pairs,
    summarize_diagnostics,
    visual_token_manifold_loss,
)


PositionMode = Literal["native_source_grid", "inherit_source_visual_positions"]


@dataclass
class TGVFv3Stage1Sample:
    image: str
    question: str
    target: str
    evidence_description: str
    image_id: str | None = None
    choices: Any | None = None
    answer: str | None = None
    short_answer: str | None = None
    answer_format: str | None = None
    value_span_text: str | None = None
    evidence_type: str | None = None
    target_style: str | None = None
    target_cues: list[str] = field(default_factory=list)
    source_dataset: str | None = None
    source_profile: str | None = None
    schema_version: str | None = None
    teacher_prompt_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_question(self) -> str:
        return format_question_with_choices(self.question, self.choices)


@dataclass
class TGVFv3Stage1Features:
    capture: Qwen3FocusCapture
    target_hidden_states: torch.Tensor
    pre_merge_visual_tokens: torch.Tensor
    merged_visual_tokens: torch.Tensor
    image_grid_thw: torch.Tensor | None
    vision_tap: Any


class TGVFv3Stage1Dataset(Dataset[TGVFv3Stage1Sample]):
    required_fields = ("image", "question", "target", "evidence_description")

    def __init__(
        self,
        jsonl_path: str | Path,
        *,
        focus_only: bool = True,
        min_confidence: float | None = None,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.focus_only = focus_only
        self.min_confidence = min_confidence
        self.samples: list[TGVFv3Stage1Sample] = []
        self.skipped_rows: list[dict[str, Any]] = []
        for line_no, record in self._records():
            parsed = self._parse_record(record, line_no)
            if parsed is not None:
                self.samples.append(parsed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TGVFv3Stage1Sample:
        return self.samples[index]

    def _records(self) -> list[tuple[int, dict[str, Any]]]:
        records = []
        with self.jsonl_path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if line:
                    records.append((line_no, json.loads(line)))
        return records

    def _parse_record(
        self,
        record: dict[str, Any],
        line_no: int,
    ) -> TGVFv3Stage1Sample | None:
        need_focus = bool(record.get("need_focus", True))
        trajectory_type = record.get("trajectory_type", "single_focus")
        evidence_state = record.get("evidence_state", NEED_LOCAL_EVIDENCE)
        if self.focus_only and (
            not need_focus
            or trajectory_type != "single_focus"
            or evidence_state != NEED_LOCAL_EVIDENCE
        ):
            self.skipped_rows.append(
                {"line_no": line_no, "reason": "not_stage1_focus_sample"}
            )
            return None
        confidence = record.get("confidence")
        if self.min_confidence is not None and confidence is not None:
            if float(confidence) < float(self.min_confidence):
                self.skipped_rows.append({"line_no": line_no, "reason": "low_confidence"})
                return None
        missing = [field_name for field_name in self.required_fields if not record.get(field_name)]
        if missing:
            raise ValueError(f"{self.jsonl_path}:{line_no} missing required fields: {missing}")

        image = Path(record["image"])
        if not image.is_absolute():
            image = (self.jsonl_path.parent / image).resolve()

        metadata = dict(record.get("metadata") or {})
        for key in (
            "need_focus",
            "trajectory_type",
            "evidence_state",
            "confidence",
            "visual_difficulty",
            "visibility",
            "target_leakage_risk",
            "evidence_specificity",
        ):
            if key in record:
                metadata[key] = record[key]
        return TGVFv3Stage1Sample(
            image=str(image),
            question=str(record["question"]),
            target=str(record["target"]),
            evidence_description=str(record["evidence_description"]),
            image_id=record.get("image_id"),
            choices=record.get("choices"),
            answer=record.get("answer"),
            short_answer=record.get("short_answer"),
            answer_format=record.get("answer_format"),
            value_span_text=record.get("value_span_text"),
            evidence_type=record.get("evidence_type"),
            target_style=record.get("target_style", "unknown"),
            target_cues=list(record.get("target_cues") or []),
            source_dataset=record.get("source_dataset"),
            source_profile=record.get("source_profile"),
            schema_version=record.get("schema_version"),
            teacher_prompt_version=record.get("teacher_prompt_version"),
            metadata=metadata,
        )


def format_question_with_choices(question: str, choices: Any | None) -> str:
    if not choices:
        return question
    if isinstance(choices, dict):
        choice_lines = [f"{key}. {value}" for key, value in choices.items()]
    elif isinstance(choices, list):
        choice_lines = []
        for index, choice in enumerate(choices):
            if isinstance(choice, dict):
                label = str(choice.get("label") or chr(ord("A") + index)).strip()
                text = str(choice.get("text") or choice.get("value") or "").strip()
                choice_lines.append(f"{label}. {text}" if text else label)
            else:
                choice_lines.append(str(choice))
    else:
        choice_lines = [str(choices)]
    return question.rstrip() + "\nChoices:\n" + "\n".join(choice_lines)


def tgvf_v3_stage1_collate(samples: list[TGVFv3Stage1Sample]) -> list[TGVFv3Stage1Sample]:
    return samples


def freeze_qwen_backbone(model: nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False


@torch.no_grad()
def infer_qwen3_stage1_dims(
    *,
    model: Any,
    processor: Any,
    sample: TGVFv3Stage1Sample,
    device: torch.device | str,
    max_image_resolution: int | None = 512,
) -> dict[str, int]:
    tap, v_pre, _v_merge = tap_qwen3_vision_features(
        model,
        processor,
        image=_image_input(sample.image, max_image_resolution=max_image_resolution),
        question=sample.prompt_question,
        device=device,
    )
    if v_pre is None:
        raise RuntimeError(f"Could not infer V_pre shape: {tap.errors}")
    return {
        "d_lm": int(llm_hidden_dim(model)),
        "d_v": int(v_pre.shape[-1]),
        "spatial_merge_size": int(tap.spatial_merge_size or tap.merge_size or 2),
    }


@torch.no_grad()
def collect_v3_stage1_features(
    *,
    model: Any,
    processor: Any,
    sample: TGVFv3Stage1Sample,
    device: torch.device | str,
    hidden_state_index: int = -1,
    max_image_resolution: int | None = 512,
    capture_mode: Literal["teacher_forced", "decode_loop"] = "teacher_forced",
    vision_cache: dict[str, tuple[Any, torch.Tensor, torch.Tensor]] | None = None,
    protocol: TGVFProtocol = "legacy_v3_tags",
    focus_action_im_end: bool = False,
) -> TGVFv3Stage1Features:
    image_input = _image_input(sample.image, max_image_resolution=max_image_resolution)
    if capture_mode == "teacher_forced":
        capture = capture_v3_stage1_focus_teacher_forced(
            model=model,
            processor=processor,
            image=image_input,
            question=sample.prompt_question,
            target=sample.target,
            device=device,
            hidden_state_index=hidden_state_index,
            protocol=protocol,
            append_im_end=focus_action_im_end,
        )
    elif capture_mode == "decode_loop":
        capture = capture_focus_single_pass_qwen3(
            model,
            processor,
            image=image_input,
            question=sample.prompt_question,
            scripted_target_text=sample.target,
            max_new_tokens=max(32, len(sample.target.split()) + 24),
            device=device,
            hidden_state_index=hidden_state_index,
            eos_token_id=getattr(processor.tokenizer, "eos_token_id", None),
            protocol=protocol,
        )
    else:
        raise ValueError(f"Unsupported capture_mode: {capture_mode}")
    if not capture.capture_found:
        raise RuntimeError(f"Forced v3 focus span was not captured: {capture.generated_text!r}")
    cache_key = f"{sample.image}|{max_image_resolution}"
    if vision_cache is not None and cache_key in vision_cache:
        tap, v_pre, v_merge = vision_cache[cache_key]
    else:
        tap, v_pre, v_merge = tap_qwen3_vision_features(
            model,
            processor,
            image=image_input,
            question=sample.prompt_question,
            device=device,
        )
        if vision_cache is not None and v_pre is not None and v_merge is not None:
            vision_cache[cache_key] = (tap, v_pre, v_merge)
    if v_pre is None:
        raise RuntimeError(f"Qwen3 V_pre tap failed: {tap.errors}")
    if v_merge is None:
        raise RuntimeError(f"Qwen3 V_merge tap failed: {tap.errors}")
    return TGVFv3Stage1Features(
        capture=capture,
        target_hidden_states=capture.target_hidden_states.detach().cpu(),
        pre_merge_visual_tokens=v_pre.detach().cpu(),
        merged_visual_tokens=v_merge.detach().cpu(),
        image_grid_thw=(
            None if capture.image_grid_thw is None else capture.image_grid_thw.detach().cpu()
        ),
        vision_tap=tap,
    )


@torch.no_grad()
def capture_v3_stage1_focus_teacher_forced(
    *,
    model: Any,
    processor: Any,
    image: Any,
    question: str,
    target: str,
    device: torch.device | str | None = None,
    hidden_state_index: int = -1,
    protocol: TGVFProtocol = "legacy_v3_tags",
    append_im_end: bool = False,
) -> Qwen3FocusCapture:
    """Teacher-forced Stage1 H_q extraction.

    This is deliberately separate from the deployment single-pass decode loop.
    Stage1 already knows the teacher target, so a single frozen Qwen forward is
    enough to obtain contextual target hidden states.
    """
    protocol = normalize_tgvf_protocol(protocol)
    tokenizer = processor.tokenizer
    messages = build_focus_force_messages(image, question, protocol=protocol)
    inputs = build_qwen3_inputs(processor, messages)
    if device is None:
        device = _infer_model_device(model)
    model_inputs = _move_tensors_for_stage1(dict(inputs), device)
    base_input_ids = model_inputs["input_ids"]
    base_attention = model_inputs.get("attention_mask")
    source_visual_geometry = extract_qwen3_source_visual_geometry(model, model_inputs)
    forced_text = render_focus_action_text(
        target,
        protocol=protocol,
        pad_target_spaces=True,
        append_im_end=append_im_end,
    )
    forced_ids = _encode_text(tokenizer, forced_text, base_input_ids.device).view(1, -1)
    full_input_ids = torch.cat([base_input_ids, forced_ids], dim=-1)
    if base_attention is None:
        full_attention = torch.ones_like(full_input_ids)
    else:
        full_attention = torch.cat(
            [base_attention, torch.ones_like(forced_ids, dtype=base_attention.dtype)],
            dim=-1,
        )
    forward_inputs = dict(model_inputs)
    forward_inputs["input_ids"] = full_input_ids
    forward_inputs["attention_mask"] = full_attention
    if isinstance(forward_inputs.get("mm_token_type_ids"), torch.Tensor):
        mm_token_type_ids = forward_inputs["mm_token_type_ids"]
        forward_inputs["mm_token_type_ids"] = torch.cat(
            [
                mm_token_type_ids,
                torch.zeros_like(forced_ids, dtype=mm_token_type_ids.dtype),
            ],
            dim=-1,
        )
    outputs = model(
        **forward_inputs,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    generated_hidden = outputs.hidden_states[hidden_state_index][0, -int(forced_ids.shape[-1]) :]
    span = _find_focus_target_span_in_forced_ids(
        tokenizer,
        forced_ids.view(-1).tolist(),
        protocol=protocol,
    )
    if span is None:
        target_start = target_end = 0
        target_text = ""
        target_ids: list[int] = []
        target_hidden = generated_hidden[:0].detach()
        capture_found = False
        malformed = True
        errors = ["teacher_forced_focus_span_not_found"]
    else:
        target_start, target_end, target_text = span
        target_ids = forced_ids.view(-1).tolist()[target_start:target_end]
        target_hidden = generated_hidden[target_start:target_end].detach()
        capture_found = True
        malformed = False
        errors = []
    return Qwen3FocusCapture(
        target_text=target_text,
        target_token_ids=target_ids,
        target_hidden_states=target_hidden,
        generated_ids=forced_ids.view(-1).tolist(),
        generated_text=_decode(tokenizer, forced_ids.view(-1).tolist()),
        generated_hidden_states=generated_hidden.detach(),
        past_key_values=None,
        attention_mask=full_attention.detach(),
        cache_position=None,
        input_ids=full_input_ids.detach(),
        last_logits=None,
        model_kwargs={"focus_target_source": "teacher_forced_stage1"},
        image_grid_thw=model_inputs.get("image_grid_thw"),
        video_grid_thw=model_inputs.get("video_grid_thw"),
        source_visual_geometry=source_visual_geometry,
        target_token_start=target_start,
        target_token_end=target_end,
        stop_reason="teacher_forced_focus_end_marker",
        capture_found=capture_found,
        second_full_forward_used=False,
        malformed=malformed,
        errors=errors,
    )


def prepare_v3_stage1_readout_inputs(
    *,
    model: Any,
    tokenizer_or_processor: Any,
    capture: Qwen3FocusCapture,
    evidence_description: str,
    foveated_visual_tokens: torch.Tensor,
    device: torch.device | str | None = None,
    mask_original_image_after_tgvf: bool = True,
    position_mode: PositionMode = "native_source_grid",
    protocol: TGVFProtocol = "legacy_v3_tags",
    focus_action_im_end: bool = False,
) -> dict[str, Any]:
    if not capture.capture_found:
        raise ValueError("capture must contain a valid focus span")
    if capture.input_ids is None or capture.attention_mask is None:
        raise ValueError("capture input_ids and attention_mask are required for v3 Stage1 readout")
    source_geometry = capture.source_visual_geometry
    if source_geometry is None:
        raise ValueError("capture is missing source visual geometry")
    source_count = int(source_geometry.source_visual_token_count)
    if source_count <= 0:
        raise ValueError("source visual token count must be positive")
    if int(foveated_visual_tokens.shape[0]) != source_count:
        raise ValueError(
            f"D token count {int(foveated_visual_tokens.shape[0])} must equal source visual token count {source_count}"
        )
    if position_mode not in {"native_source_grid", "inherit_source_visual_positions"}:
        raise ValueError(f"Unsupported position_mode: {position_mode}")

    tokenizer = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)
    protocol = normalize_tgvf_protocol(protocol)
    if device is None:
        device = _infer_model_device(model) or foveated_visual_tokens.device
    base_input_ids = capture.input_ids.to(device)
    base_len = int(base_input_ids.shape[-1])
    d = foveated_visual_tokens.to(device)
    tgvf_prefix, tgvf_suffix = render_tgvf_prefix_suffix(
        protocol=protocol,
        include_leading_im_end=not (
            protocol == "protocol_c_tool_observation" and focus_action_im_end
        ),
    )
    readout_prefix, readout_text = render_stage1_readout_text(
        evidence_description=evidence_description,
        protocol=protocol,
    )
    prefix = tgvf_prefix
    suffix = f"{tgvf_suffix}{readout_prefix}"
    tgvf_ids = _bracketed_visual_token_ids(
        tokenizer_or_processor,
        model,
        num_fvt_tokens=int(d.shape[0]),
        prefix=prefix,
        suffix=suffix,
        device=device,
    ).view(1, -1)
    evidence_ids = _encode_text(tokenizer, readout_text, device).view(1, -1)
    input_ids = torch.cat([base_input_ids, tgvf_ids, evidence_ids], dim=-1)

    prefix_ids = _encode_text(tokenizer, prefix, device)
    local_fvt_start = int(prefix_ids.shape[0]) + 1
    local_fvt_end = local_fvt_start + int(d.shape[0])
    fvt_token_start = base_len + local_fvt_start
    fvt_token_end = base_len + local_fvt_end
    evidence_start = base_len + int(tgvf_ids.shape[-1])

    base_embeds = model.get_input_embeddings()(input_ids)
    embeds = torch.cat(
        [
            base_embeds[:, :fvt_token_start],
            d.to(dtype=base_embeds.dtype).unsqueeze(0),
            base_embeds[:, fvt_token_end:],
        ],
        dim=1,
    )

    labels = torch.full_like(input_ids, IGNORE_INDEX)
    labels[:, evidence_start:] = input_ids[:, evidence_start:]
    attention_mask_2d = torch.ones_like(input_ids)
    mm_token_type_ids = _full_mm_token_type_ids(
        model=model,
        input_ids=input_ids,
        fvt_token_start=fvt_token_start,
        fvt_token_end=fvt_token_end,
        device=device,
    )
    image_grid_thw = _stage1_image_grid_thw(
        capture=capture,
        source_image_grid_thw=source_geometry.image_grid_thw,
        device=device,
    )
    if position_mode == "native_source_grid":
        position_ids = _compute_qwen3_position_ids_for_sequence(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask_2d,
            image_grid_thw=image_grid_thw,
            video_grid_thw=capture.video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )
        position_ids_source = "qwen3_native_source_grid_full_trajectory"
    else:
        position_ids = _compute_qwen3_position_ids_for_sequence(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask_2d,
            image_grid_thw=image_grid_thw,
            video_grid_thw=capture.video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )
        if position_ids is not None and source_geometry.source_visual_position_ids is not None:
            source_positions = source_geometry.source_visual_position_ids.to(device=device)
            position_ids[:, 0, fvt_token_start:fvt_token_end] = source_positions
        position_ids_source = "reuse_original_visual_positions"
    if position_ids is None:
        raise RuntimeError("Qwen3 position id computation is unavailable")

    original_image_indices = source_geometry.source_visual_token_indices
    if original_image_indices is None:
        raise RuntimeError("source visual token indices are unavailable")
    original_image_indices = original_image_indices.to(device=device, dtype=torch.long)
    if mask_original_image_after_tgvf:
        attention_mask = build_weak_strict_attention_mask(
            attention_mask_2d=attention_mask_2d,
            original_image_token_indices=original_image_indices,
            block_query_start=base_len,
            dtype=embeds.dtype,
        )
        mask_mode = "weak_strict_original_image_keys_4d"
    else:
        attention_mask = attention_mask_2d
        mask_mode = "standard_2d_causal"

    mask_summary = summarize_weak_strict_mask(
        attention_mask=attention_mask,
        original_image_token_indices=original_image_indices,
        block_query_start=base_len,
    )
    return {
        "input_ids": input_ids,
        "inputs_embeds": embeds,
        "labels": labels,
        "attention_mask": attention_mask,
        "attention_mask_2d": attention_mask_2d,
        "position_ids": position_ids,
        "image_grid_thw": image_grid_thw,
        "mm_token_type_ids": mm_token_type_ids,
        "fvt_token_start": fvt_token_start,
        "fvt_token_end": fvt_token_end,
        "evidence_start": evidence_start,
        "answer_token_count": int(evidence_ids.shape[-1]),
        "mask_mode": mask_mode,
        "position_mode": position_mode,
        "position_ids_source": position_ids_source,
        "original_image_token_count": int(original_image_indices.numel()),
        "original_image_token_span_detection": "source_image_token_id_scan",
        "block_query_start": base_len,
        "blocked_original_image_keys_for_post_tgvf": mask_summary[
            "blocked_original_image_keys_for_post_tgvf"
        ],
        "pre_tgvf_queries_keep_original_image_keys": mask_summary[
            "pre_tgvf_queries_keep_original_image_keys"
        ],
        "mask_summary": mask_summary,
        "tgvf_protocol": protocol,
    }


def build_weak_strict_attention_mask(
    *,
    attention_mask_2d: torch.Tensor,
    original_image_token_indices: torch.Tensor,
    block_query_start: int,
    dtype: torch.dtype,
    block_query_end: int | None = None,
) -> torch.Tensor:
    if attention_mask_2d.ndim != 2 or attention_mask_2d.shape[0] != 1:
        raise ValueError("weak-strict mask currently supports batch size 1")
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
        query_indices = torch.nonzero(query_mask, as_tuple=False).view(-1)
        if int(query_indices.numel()) > 0:
            mask[:, :, query_indices[:, None], original_image_token_indices[None, :]] = min_value
    return mask


def summarize_weak_strict_mask(
    *,
    attention_mask: torch.Tensor,
    original_image_token_indices: torch.Tensor,
    block_query_start: int,
) -> dict[str, Any]:
    if attention_mask.ndim != 4:
        return {
            "mask_is_4d": False,
            "blocked_original_image_keys_for_post_tgvf": False,
            "pre_tgvf_queries_keep_original_image_keys": False,
        }
    if int(original_image_token_indices.numel()) == 0:
        return {
            "mask_is_4d": True,
            "blocked_original_image_keys_for_post_tgvf": False,
            "pre_tgvf_queries_keep_original_image_keys": False,
        }
    first_image_key = int(original_image_token_indices[0].detach().cpu().item())
    post_query = min(int(block_query_start), int(attention_mask.shape[-1]) - 1)
    pre_query = max(post_query - 1, first_image_key)
    blocked_post = bool(attention_mask[0, 0, post_query, first_image_key].detach().cpu().item() < 0)
    kept_pre = bool(attention_mask[0, 0, pre_query, first_image_key].detach().cpu().item() == 0)
    return {
        "mask_is_4d": True,
        "blocked_original_image_keys_for_post_tgvf": blocked_post,
        "pre_tgvf_queries_keep_original_image_keys": kept_pre,
        "original_image_key_count": int(original_image_token_indices.numel()),
        "first_original_image_key": first_image_key,
        "block_query_start": int(block_query_start),
    }


def compute_v3_stage1_lm_loss(
    *,
    model: Any,
    readout_inputs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    losses, log_likelihoods = compute_v3_stage1_lm_losses_batched(
        model=model,
        readout_inputs_list=[readout_inputs],
    )
    return losses[0], log_likelihoods[0]


def compute_v3_stage1_lm_losses_batched(
    *,
    model: Any,
    readout_inputs_list: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not readout_inputs_list:
        raise ValueError("readout_inputs_list must not be empty")
    batched = batch_v3_stage1_readout_inputs(readout_inputs_list)
    outputs = model(
        inputs_embeds=batched["inputs_embeds"],
        attention_mask=batched["attention_mask"],
        position_ids=batched["position_ids"],
        image_grid_thw=batched["image_grid_thw"],
        mm_token_type_ids=batched["mm_token_type_ids"],
        return_dict=True,
    )
    logits = outputs.logits
    labels = batched["labels"]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    per_token_nll = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view(shift_labels.shape)
    valid = shift_labels != IGNORE_INDEX
    nll_sum = (per_token_nll * valid.to(per_token_nll.dtype)).sum(dim=-1)
    token_count = valid.sum(dim=-1).clamp_min(1)
    return nll_sum / token_count, -nll_sum


def batch_v3_stage1_readout_inputs(readout_inputs_list: list[dict[str, Any]]) -> dict[str, Any]:
    max_len = max(int(item["inputs_embeds"].shape[1]) for item in readout_inputs_list)
    embeds_list = []
    labels_list = []
    input_ids_list = []
    token_type_list = []
    attention_2d_list = []
    position_list = []
    attention_4d_list = []
    image_grids = []
    dtype = readout_inputs_list[0]["inputs_embeds"].dtype
    device = readout_inputs_list[0]["inputs_embeds"].device
    min_value = torch.finfo(dtype).min
    for item in readout_inputs_list:
        seq_len = int(item["inputs_embeds"].shape[1])
        pad_len = max_len - seq_len
        embeds = item["inputs_embeds"]
        if pad_len:
            embeds = torch.cat(
                [embeds, embeds.new_zeros((1, pad_len, embeds.shape[-1]))],
                dim=1,
            )
        embeds_list.append(embeds)

        labels = item["labels"]
        if pad_len:
            labels = torch.cat(
                [labels, torch.full((1, pad_len), IGNORE_INDEX, dtype=labels.dtype, device=labels.device)],
                dim=1,
            )
        labels_list.append(labels)

        input_ids = item["input_ids"]
        if pad_len:
            input_ids = torch.cat(
                [input_ids, torch.zeros((1, pad_len), dtype=input_ids.dtype, device=input_ids.device)],
                dim=1,
            )
        input_ids_list.append(input_ids)

        token_type_ids = item["mm_token_type_ids"]
        if pad_len:
            token_type_ids = torch.cat(
                [
                    token_type_ids,
                    torch.zeros((1, pad_len), dtype=token_type_ids.dtype, device=token_type_ids.device),
                ],
                dim=1,
            )
        token_type_list.append(token_type_ids)

        attention_2d = item["attention_mask_2d"]
        if pad_len:
            attention_2d = torch.cat(
                [
                    attention_2d,
                    torch.zeros((1, pad_len), dtype=attention_2d.dtype, device=attention_2d.device),
                ],
                dim=1,
            )
        attention_2d_list.append(attention_2d)

        position_ids = item["position_ids"]
        if pad_len:
            position_ids = torch.cat(
                [
                    position_ids,
                    torch.zeros(
                        (position_ids.shape[0], 1, pad_len),
                        dtype=position_ids.dtype,
                        device=position_ids.device,
                    ),
                ],
                dim=-1,
            )
        position_list.append(position_ids)

        attention = item["attention_mask"]
        if attention.ndim == 2:
            attention = build_weak_strict_attention_mask(
                attention_mask_2d=item["attention_mask_2d"],
                original_image_token_indices=torch.empty(0, dtype=torch.long, device=device),
                block_query_start=max_len + 1,
                dtype=dtype,
            )
        padded_attention = torch.full(
            (1, 1, max_len, max_len),
            min_value,
            dtype=attention.dtype,
            device=attention.device,
        )
        padded_attention[:, :, :seq_len, :seq_len] = attention
        attention_4d_list.append(padded_attention)
        image_grids.append(item["image_grid_thw"])

    return {
        "input_ids": torch.cat(input_ids_list, dim=0),
        "inputs_embeds": torch.cat(embeds_list, dim=0),
        "labels": torch.cat(labels_list, dim=0),
        "attention_mask": torch.cat(attention_4d_list, dim=0),
        "attention_mask_2d": torch.cat(attention_2d_list, dim=0),
        "position_ids": torch.cat(position_list, dim=1),
        "image_grid_thw": torch.cat(image_grids, dim=0),
        "mm_token_type_ids": torch.cat(token_type_list, dim=0),
    }


def compute_v3_stage1_lm_losses_chunked(
    *,
    model: Any,
    readout_inputs_list: list[dict[str, Any]],
    chunk_size: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if chunk_size <= 0:
        chunk_size = len(readout_inputs_list)
    losses: list[torch.Tensor] = []
    log_likelihoods: list[torch.Tensor] = []
    for start in range(0, len(readout_inputs_list), chunk_size):
        chunk = readout_inputs_list[start : start + chunk_size]
        chunk_losses, chunk_ll = compute_v3_stage1_lm_losses_batched(
            model=model,
            readout_inputs_list=chunk,
        )
        losses.extend(chunk_losses.unbind(0))
        log_likelihoods.extend(chunk_ll.unbind(0))
    return losses, log_likelihoods


def v3_stage1_training_step(
    *,
    qwen_model: Any,
    processor: Any,
    foveal_module: nn.Module,
    reencode_qwen_model: Any | None = None,
    samples: list[TGVFv3Stage1Sample],
    loss_weights: LossWeights,
    device: torch.device | str,
    hidden_state_index: int = -1,
    same_image_negative_margin: float = 1.0,
    same_image_negative_mode: str = "matrix_ce",
    mask_original_image_after_tgvf: bool = True,
    position_mode: PositionMode = "native_source_grid",
    max_image_resolution: int | None = 512,
    capture_mode: Literal["teacher_forced", "decode_loop"] = "teacher_forced",
    readout_batch_size: int = 4,
    protocol: TGVFProtocol = "legacy_v3_tags",
    focus_action_im_end: bool = False,
) -> TGVFTrainStepOutput:
    protocol = normalize_tgvf_protocol(protocol)
    vision_cache: dict[str, tuple[Any, torch.Tensor, torch.Tensor]] = {}
    features = [
        collect_v3_stage1_features(
            model=qwen_model,
            processor=processor,
            sample=sample,
            device=device,
            hidden_state_index=hidden_state_index,
            max_image_resolution=max_image_resolution,
            capture_mode=capture_mode,
            vision_cache=vision_cache,
            protocol=protocol,
            focus_action_im_end=focus_action_im_end,
        )
        for sample in samples
    ]
    fvt_outputs: list[torch.Tensor] = []
    loss_man_values: list[torch.Tensor] = []
    attention_debug_values: list[dict[str, Any]] = []
    norm_debug_values: list[dict[str, Any]] = []
    readout_debug_values: list[dict[str, Any]] = []
    positive_readouts: list[dict[str, Any]] = []

    for sample, feature in zip(samples, features, strict=True):
        output = foveal_module(
            target_hidden_states=feature.target_hidden_states.to(device),
            pre_merge_visual_tokens=feature.pre_merge_visual_tokens.to(device),
            metadata={
                "target": sample.target,
                "stage": "tgvf_v3_stage1",
                "evidence_state": NEED_LOCAL_EVIDENCE,
                "qwen_model": reencode_qwen_model if reencode_qwen_model is not None else qwen_model,
                "processor": processor,
                "image": _image_input(sample.image, max_image_resolution=max_image_resolution),
                "question": sample.prompt_question,
                "device": device,
                "original_qwen_model_used_for_readout": reencode_qwen_model is None,
                "separate_reencode_qwen_model": reencode_qwen_model is not None,
            },
        )
        output = finalize_tgvf_output_with_frozen_qwen_merger(qwen_model, output)
        d = output.foveated_visual_tokens
        fvt_outputs.append(d)
        readout_inputs = prepare_v3_stage1_readout_inputs(
            model=qwen_model,
            tokenizer_or_processor=processor,
            capture=feature.capture,
            evidence_description=sample.evidence_description,
            foveated_visual_tokens=d,
            device=device,
                            mask_original_image_after_tgvf=mask_original_image_after_tgvf,
                            position_mode=position_mode,
                            protocol=protocol,
                            focus_action_im_end=focus_action_im_end,
                        )
        positive_readouts.append(readout_inputs)
        loss_man_values.append(
            _safe_visual_token_manifold_loss(d, feature.merged_visual_tokens.to(device))
        )
        attention_debug_values.append(attention_diagnostics(output.attention_debug))
        norm_debug_values.append(
            fvt_norm_diagnostics(
                foveated_visual_tokens=d,
                merged_visual_tokens=feature.merged_visual_tokens.to(device),
            )
        )
        readout_debug_values.append(_readout_debug(readout_inputs))

    loss_gen_values, positive_ll = compute_v3_stage1_lm_losses_chunked(
        model=qwen_model,
        readout_inputs_list=positive_readouts,
        chunk_size=readout_batch_size,
    )
    loss_gen = torch.stack(loss_gen_values).mean()
    loss_man = torch.stack(loss_man_values).mean()
    zero = loss_gen.new_zeros(())
    loss_same = zero
    if loss_weights.same_image_negative:
        if same_image_negative_mode == "cyclic_margin":
            pairs = same_image_negative_pairs(samples)
            negative_ll = []
            positive_selected = []
            for pos_index, neg_index in pairs:
                readout_inputs = prepare_v3_stage1_readout_inputs(
                    model=qwen_model,
                    tokenizer_or_processor=processor,
                    capture=features[pos_index].capture,
                    evidence_description=samples[pos_index].evidence_description,
                    foveated_visual_tokens=fvt_outputs[neg_index],
                    device=device,
                    mask_original_image_after_tgvf=mask_original_image_after_tgvf,
                    position_mode=position_mode,
                    protocol=protocol,
                    focus_action_im_end=focus_action_im_end,
                )
                _, log_likelihood = compute_v3_stage1_lm_loss(
                    model=qwen_model,
                    readout_inputs=readout_inputs,
                )
                negative_ll.append(log_likelihood)
                positive_selected.append(positive_ll[pos_index])
            if negative_ll:
                loss_same = same_image_negative_loss(
                    positive_log_likelihoods=torch.stack(positive_selected),
                    negative_log_likelihoods=torch.stack(negative_ll),
                    margin=same_image_negative_margin,
                )
        elif same_image_negative_mode == "matrix_ce":
            score_matrices = []
            for indices in same_image_negative_groups(samples):
                pending_readouts: list[dict[str, Any]] = []
                pending_slots: list[tuple[int, int]] = []
                matrix_cells: list[list[torch.Tensor | None]] = [
                    [None for _ in indices] for _ in indices
                ]
                rows = []
                for row_index, pos_index in enumerate(indices):
                    row = []
                    for col_index, fvt_index in enumerate(indices):
                        if pos_index == fvt_index:
                            matrix_cells[row_index][col_index] = positive_ll[pos_index]
                            continue
                        readout_inputs = prepare_v3_stage1_readout_inputs(
                            model=qwen_model,
                            tokenizer_or_processor=processor,
                            capture=features[pos_index].capture,
                            evidence_description=samples[pos_index].evidence_description,
                            foveated_visual_tokens=fvt_outputs[fvt_index],
                            device=device,
	                            mask_original_image_after_tgvf=mask_original_image_after_tgvf,
	                            position_mode=position_mode,
	                            protocol=protocol,
	                            focus_action_im_end=focus_action_im_end,
	                        )
                        pending_readouts.append(readout_inputs)
                        pending_slots.append((row_index, col_index))
                if pending_readouts:
                    _, pending_ll = compute_v3_stage1_lm_losses_chunked(
                        model=qwen_model,
                        readout_inputs_list=pending_readouts,
                        chunk_size=readout_batch_size,
                    )
                    for (row_index, col_index), log_likelihood in zip(
                        pending_slots, pending_ll, strict=True
                    ):
                        matrix_cells[row_index][col_index] = log_likelihood
                for row_cells in matrix_cells:
                    row = [cell for cell in row_cells if cell is not None]
                    if len(row) != len(indices):
                        raise RuntimeError("incomplete same-image matrix CE score row")
                    rows.append(torch.stack(row))
                score_matrices.append(torch.stack(rows))
            if score_matrices:
                loss_same = same_image_negative_matrix_ce_loss(score_matrices)
        else:
            raise ValueError(f"Unsupported same_image_negative_mode: {same_image_negative_mode}")

    loss_contrastive = zero
    loss_total = (
        loss_weights.gen * loss_gen
        + loss_weights.visual_token_manifold * loss_man
        + loss_weights.same_image_negative * loss_same
        + loss_weights.contrastive_alignment * loss_contrastive
    )
    first_feature = features[0]
    first_d = fvt_outputs[0]
    finite_values = [
        torch.isfinite(first_feature.target_hidden_states).float().mean(),
        torch.isfinite(first_feature.pre_merge_visual_tokens).float().mean(),
        torch.isfinite(first_d.detach()).float().mean(),
    ]
    return TGVFTrainStepOutput(
        loss_total=loss_total,
        loss_gen=loss_gen,
        loss_visual_token_manifold=loss_man,
        loss_same_image_negative=loss_same,
        loss_contrastive_alignment=loss_contrastive,
        debug={
            "stage": "tgvf_v3_stage1",
            "tgvf_protocol": protocol,
            "target_hidden_shape": list(first_feature.target_hidden_states.shape),
            "pre_merge_visual_shape": list(first_feature.pre_merge_visual_tokens.shape),
            "merged_visual_shape": list(first_feature.merged_visual_tokens.shape),
            "foveated_visual_tokens_shape": list(first_d.shape),
            "loss_weights": asdict(loss_weights),
            "same_image_negative_mode": same_image_negative_mode,
            "readout_append_mode": "qwen3_visual_special_tokens_embedding_replace",
            "attention_mask_mode": readout_debug_values[0]["mask_mode"],
            "original_image_token_span_detection": readout_debug_values[0][
                "original_image_token_span_detection"
            ],
            "image_keys_blocked_for_tgvf_evidence_answer": readout_debug_values[0][
                "blocked_original_image_keys_for_post_tgvf"
            ],
            "pre_tgvf_queries_keep_original_image_keys": readout_debug_values[0][
                "pre_tgvf_queries_keep_original_image_keys"
            ],
            "position_mode": position_mode,
            "position_ids_source": readout_debug_values[0]["position_ids_source"],
            "source_visual_token_count": readout_debug_values[0]["original_image_token_count"],
            "answer_token_count": readout_debug_values[0]["answer_token_count"],
            "attention_diagnostics": summarize_diagnostics(attention_debug_values),
            "norm_diagnostics": _compact_diagnostics(summarize_diagnostics(norm_debug_values)),
            "finite_rate": float(torch.stack([value.cpu() for value in finite_values]).mean()),
            "visual_token_manifold_active": bool(
                first_d.shape[-1] == first_feature.merged_visual_tokens.shape[-1]
            ),
            "qwen_frozen": not any(parameter.requires_grad for parameter in qwen_model.parameters()),
            "separate_reencode_qwen_model": reencode_qwen_model is not None,
            "reencode_qwen_trainable": False if reencode_qwen_model is None else any(parameter.requires_grad for parameter in reencode_qwen_model.parameters()),
            "second_full_forward_used": False,
            "encoder_reencode": bool(getattr(_unwrap_module(foveal_module), "variant_name", "") == "tgvf_encoder_bidir_8_16_24"),
            "encoder_adapter_layers": first_d.detach().new_tensor([]).detach().cpu().tolist() if not hasattr(_unwrap_module(foveal_module), "requested_adapter_layers") else list(getattr(_unwrap_module(foveal_module), "requested_adapter_layers")),
            "encoder_adapter_actual_indices": [] if not hasattr(_unwrap_module(foveal_module), "actual_adapter_indices") else list(getattr(_unwrap_module(foveal_module), "actual_adapter_indices")),
            "encoder_adapter_gate_values": first_d.detach().new_tensor([]).detach().cpu().tolist() if not hasattr(_unwrap_module(foveal_module), "last_reencode_debug") else getattr(_unwrap_module(foveal_module), "last_reencode_debug", {}).get("encoder_adapter_gate_values"),
            "vision_tower_rerun": bool(getattr(_unwrap_module(foveal_module), "last_reencode_debug", {}).get("vision_tower_rerun", False)),
            "backward_performed": False,
            "debug_examples": [
                {
                    "question": samples[0].prompt_question,
                    "target": samples[0].target,
                    "evidence_description": samples[0].evidence_description,
                    "target_token_count": len(first_feature.capture.target_token_ids),
                    "h_q_shape": list(first_feature.target_hidden_states.shape),
                    "v_pre_shape": list(first_feature.pre_merge_visual_tokens.shape),
                    "d_shape": list(first_d.shape),
                    "mask_span_summary": readout_debug_values[0]["mask_summary"],
                }
            ],
        },
    )


def _unwrap_module(module: nn.Module) -> nn.Module:
    return getattr(module, "module", module)

def _readout_debug(readout_inputs: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "mask_mode",
        "position_mode",
        "position_ids_source",
        "original_image_token_count",
        "original_image_token_span_detection",
        "blocked_original_image_keys_for_post_tgvf",
        "pre_tgvf_queries_keep_original_image_keys",
        "answer_token_count",
        "mask_summary",
        "tgvf_protocol",
    )
    return {key: readout_inputs.get(key) for key in keys}


def _compact_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in diagnostics.items()
        if not (isinstance(key, str) and key.endswith("_values"))
    }


def _safe_visual_token_manifold_loss(
    foveated_visual_tokens: torch.Tensor,
    merged_visual_tokens: torch.Tensor,
) -> torch.Tensor:
    if foveated_visual_tokens.shape[-1] != merged_visual_tokens.shape[-1]:
        return foveated_visual_tokens.sum() * 0.0
    return visual_token_manifold_loss(foveated_visual_tokens, merged_visual_tokens)


def _full_mm_token_type_ids(
    *,
    model: Any,
    input_ids: torch.Tensor,
    fvt_token_start: int,
    fvt_token_end: int,
    device: torch.device | str,
) -> torch.Tensor:
    image_token_id = getattr(getattr(model, "config", None), "image_token_id", None)
    token_type_ids = torch.zeros_like(input_ids, dtype=torch.long, device=device)
    if image_token_id is not None:
        token_type_ids = torch.where(
            input_ids.to(device) == int(image_token_id),
            torch.ones_like(token_type_ids),
            token_type_ids,
        )
    token_type_ids[:, fvt_token_start:fvt_token_end] = 1
    return token_type_ids


def _stage1_image_grid_thw(
    *,
    capture: Qwen3FocusCapture,
    source_image_grid_thw: torch.Tensor | None,
    device: torch.device | str,
) -> torch.Tensor:
    if capture.image_grid_thw is None:
        raise ValueError("capture.image_grid_thw is required")
    if source_image_grid_thw is None:
        raise ValueError("source image_grid_thw is required")
    original = capture.image_grid_thw.to(device=device, dtype=torch.long).view(-1, 3)
    source = source_image_grid_thw.to(device=device, dtype=torch.long).view(-1, 3)
    return torch.cat([original, source], dim=0)


def _infer_model_device(model: Any) -> torch.device | None:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return None


def _image_input(image: str, *, max_image_resolution: int | None) -> Any:
    if max_image_resolution is None or max_image_resolution <= 0:
        return image
    return {
        "type": "image",
        "image": image,
        "max_pixels": int(max_image_resolution) * int(max_image_resolution),
    }


def _move_tensors_for_stage1(
    value: dict[str, Any],
    device: torch.device | str | None,
) -> dict[str, Any]:
    if device is None:
        return value
    moved = {}
    for key, item in value.items():
        moved[key] = item.to(device) if isinstance(item, torch.Tensor) else item
    return moved


def _find_focus_target_span_in_forced_ids(
    tokenizer: Any,
    token_ids: list[int],
    *,
    protocol: str | None = None,
) -> tuple[int, int, str] | None:
    if normalize_tgvf_protocol(protocol) == "protocol_d_qwen_tool":
        return _find_focus_target_span_from_decoded_text(
            tokenizer,
            token_ids,
            protocol=protocol,
        )
    focus_start, focus_end = protocol_focus_tokens(protocol)
    focus_start_ids = _encode_text(tokenizer, focus_start, "cpu").view(-1).tolist()
    focus_end_ids = _encode_text(tokenizer, focus_end, "cpu").view(-1).tolist()
    start_marker = _find_subsequence(token_ids, focus_start_ids)
    if start_marker < 0:
        return _find_focus_target_span_from_decoded_text(
            tokenizer,
            token_ids,
            protocol=protocol,
        )
    target_start = start_marker + len(focus_start_ids)
    end_marker = _find_subsequence(token_ids[target_start:], focus_end_ids)
    if end_marker < 0:
        return _find_focus_target_span_from_decoded_text(
            tokenizer,
            token_ids,
            protocol=protocol,
        )
    target_end = target_start + end_marker
    stripped_start, stripped_end = _strip_space_like_token_edges(
        tokenizer,
        token_ids,
        target_start,
        target_end,
    )
    target_text = _decode(tokenizer, token_ids[stripped_start:stripped_end]).strip()
    return stripped_start, stripped_end, target_text


def _find_focus_target_span_from_decoded_text(
    tokenizer: Any,
    token_ids: list[int],
    *,
    protocol: str | None = None,
) -> tuple[int, int, str] | None:
    text = _decode(tokenizer, token_ids)
    if normalize_tgvf_protocol(protocol) == "protocol_d_qwen_tool":
        span = focus_target_char_span(text, protocol=protocol)
        if span is None:
            return None
        char_start, char_end = span
        target_text = text[char_start:char_end].strip()
        if not target_text:
            return None
        offsets = _decoded_token_offsets_for_stage1(tokenizer, token_ids)
        token_indices = [
            index
            for index, (tok_start, tok_end) in enumerate(offsets)
            if tok_start < char_end and tok_end > char_start
        ]
        if not token_indices:
            return None
        return token_indices[0], token_indices[-1] + 1, target_text
    focus_start, focus_end = protocol_focus_tokens(protocol)
    start_index = text.find(focus_start)
    if start_index < 0:
        return None
    inner_start = start_index + len(focus_start)
    end_index = text.find(focus_end, inner_start)
    if end_index < 0:
        return None
    raw_focus = text[inner_start:end_index]
    target_text = raw_focus.strip()
    if not target_text:
        return None
    leading = len(raw_focus) - len(raw_focus.lstrip())
    trailing = len(raw_focus.rstrip())
    char_start = inner_start + leading
    char_end = inner_start + trailing
    offsets = _decoded_token_offsets_for_stage1(tokenizer, token_ids)
    token_indices = [
        index
        for index, (tok_start, tok_end) in enumerate(offsets)
        if tok_start < char_end and tok_end > char_start
    ]
    if not token_indices:
        return None
    return token_indices[0], token_indices[-1] + 1, target_text


def _find_subsequence(values: list[int], pattern: list[int]) -> int:
    if not pattern or len(pattern) > len(values):
        return -1
    limit = len(values) - len(pattern) + 1
    for index in range(limit):
        if values[index : index + len(pattern)] == pattern:
            return index
    return -1


def _strip_space_like_token_edges(
    tokenizer: Any,
    token_ids: list[int],
    start: int,
    end: int,
) -> tuple[int, int]:
    while start < end and _decode(tokenizer, [token_ids[start]]).strip() == "":
        start += 1
    while end > start and _decode(tokenizer, [token_ids[end - 1]]).strip() == "":
        end -= 1
    return start, end


def _decoded_token_offsets_for_stage1(
    tokenizer: Any,
    token_ids: list[int],
) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token_id in token_ids:
        token_text = _decode(tokenizer, [token_id])
        end = cursor + len(token_text)
        offsets.append((cursor, end))
        cursor = end
    return offsets
