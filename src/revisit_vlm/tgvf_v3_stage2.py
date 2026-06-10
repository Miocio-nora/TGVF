from __future__ import annotations

import json
from collections import Counter
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
    SUFFICIENT_EVIDENCE,
    TGVF_END,
    TGVF_START,
    Qwen3FocusCapture,
    _bracketed_visual_token_ids,
    _compute_qwen3_position_ids_for_sequence,
    _decode,
    _encode_text,
    build_direct_messages,
    build_qwen3_inputs,
    extract_qwen3_source_visual_geometry,
    tap_qwen3_vision_features,
)
from revisit_vlm.tgvf_foveal import finalize_tgvf_output_with_frozen_qwen_merger
from revisit_vlm.tgvf_training import IGNORE_INDEX
from revisit_vlm.tgvf_v3_stage1 import (
    PositionMode,
    TGVFv3Stage1Features,
    _find_focus_target_span_in_forced_ids,
    _full_mm_token_type_ids,
    _image_input,
    _infer_model_device,
    _move_tensors_for_stage1,
    _safe_visual_token_manifold_loss,
    _stage1_image_grid_thw,
    build_weak_strict_attention_mask,
    format_question_with_choices,
)


TrajectoryType = Literal["single_focus", "direct_answer"]


@dataclass
class TGVFv3Stage2Sample:
    image: str
    question: str
    answer: str
    need_focus: bool
    evidence_state: str
    trajectory_type: TrajectoryType
    target: str = ""
    evidence_description: str = ""
    image_id: str | None = None
    choices: Any | None = None
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
class Stage2LossWeights:
    evidence_state: float = 0.2
    focus_target: float = 1.5
    evidence: float = 1.0
    value_span: float = 3.0
    answer: float = 1.0
    no_focus_evidence_state: float = 0.2
    no_focus_answer: float = 1.0
    visual_token_manifold: float = 0.0


@dataclass
class TGVFv3Stage2StepOutput:
    loss_total: torch.Tensor
    loss_focus: torch.Tensor
    loss_no_focus: torch.Tensor
    loss_visual_token_manifold: torch.Tensor
    debug: dict[str, Any]


class TGVFv3Stage2Dataset(Dataset[TGVFv3Stage2Sample]):
    def __init__(
        self,
        jsonl_path: str | Path,
        *,
        min_confidence: float | None = None,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.min_confidence = min_confidence
        self.samples: list[TGVFv3Stage2Sample] = []
        self.skipped_rows: list[dict[str, Any]] = []
        for line_no, record in self._records():
            parsed = self._parse_record(record, line_no)
            if parsed is not None:
                self.samples.append(parsed)
        self.stats = self._stats()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TGVFv3Stage2Sample:
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
    ) -> TGVFv3Stage2Sample | None:
        confidence = record.get("confidence")
        if self.min_confidence is not None and confidence is not None:
            if float(confidence) < float(self.min_confidence):
                self.skipped_rows.append({"line_no": line_no, "reason": "low_confidence"})
                return None

        need_focus = bool(record.get("need_focus", True))
        evidence_state = str(
            record.get(
                "evidence_state",
                NEED_LOCAL_EVIDENCE if need_focus else SUFFICIENT_EVIDENCE,
            )
        )
        trajectory_type = str(
            record.get("trajectory_type", "single_focus" if need_focus else "direct_answer")
        )
        if need_focus:
            required = ("image", "question", "target", "evidence_description")
            expected_state = NEED_LOCAL_EVIDENCE
            expected_trajectory = "single_focus"
        else:
            required = ("image", "question")
            expected_state = SUFFICIENT_EVIDENCE
            expected_trajectory = "direct_answer"
        missing = [field_name for field_name in required if not record.get(field_name)]
        if missing:
            raise ValueError(f"{self.jsonl_path}:{line_no} missing required fields: {missing}")
        if evidence_state != expected_state or trajectory_type != expected_trajectory:
            self.skipped_rows.append(
                {
                    "line_no": line_no,
                    "reason": "invalid_stage2_trajectory_fields",
                    "need_focus": need_focus,
                    "evidence_state": evidence_state,
                    "trajectory_type": trajectory_type,
                }
            )
            return None

        answer = record.get("answer") or record.get("short_answer")
        if not answer:
            raise ValueError(f"{self.jsonl_path}:{line_no} missing answer/short_answer")
        image = Path(record["image"])
        if not image.is_absolute():
            image = (self.jsonl_path.parent / image).resolve()

        metadata = dict(record.get("metadata") or {})
        for key in (
            "confidence",
            "visual_difficulty",
            "visibility",
            "target_leakage_risk",
            "evidence_specificity",
        ):
            if key in record:
                metadata[key] = record[key]
        return TGVFv3Stage2Sample(
            image=str(image),
            question=str(record["question"]),
            answer=str(answer),
            need_focus=need_focus,
            evidence_state=evidence_state,
            trajectory_type=trajectory_type,  # type: ignore[arg-type]
            target=str(record.get("target") or ""),
            evidence_description=str(record.get("evidence_description") or ""),
            image_id=record.get("image_id"),
            choices=record.get("choices"),
            short_answer=record.get("short_answer"),
            answer_format=record.get("answer_format"),
            value_span_text=record.get("value_span_text"),
            evidence_type=record.get("evidence_type"),
            target_style=record.get("target_style", "unknown" if need_focus else "none"),
            target_cues=list(record.get("target_cues") or []),
            source_dataset=record.get("source_dataset"),
            source_profile=record.get("source_profile"),
            schema_version=record.get("schema_version"),
            teacher_prompt_version=record.get("teacher_prompt_version"),
            metadata=metadata,
        )

    def _stats(self) -> dict[str, Any]:
        counts = Counter("focus" if sample.need_focus else "no_focus" for sample in self.samples)
        total = len(self.samples)
        return {
            "total": total,
            "focus": counts["focus"],
            "no_focus": counts["no_focus"],
            "focus_ratio": counts["focus"] / total if total else 0.0,
            "no_focus_ratio": counts["no_focus"] / total if total else 0.0,
        }


def tgvf_v3_stage2_collate(samples: list[TGVFv3Stage2Sample]) -> list[TGVFv3Stage2Sample]:
    return samples


@torch.no_grad()
def collect_v3_stage2_focus_features(
    *,
    model: Any,
    processor: Any,
    sample: TGVFv3Stage2Sample,
    device: torch.device | str,
    hidden_state_index: int = -1,
    max_image_resolution: int | None = 512,
    vision_cache: dict[str, tuple[Any, torch.Tensor, torch.Tensor]] | None = None,
) -> TGVFv3Stage1Features:
    image_input = _image_input(sample.image, max_image_resolution=max_image_resolution)
    capture = capture_v3_stage2_focus_teacher_forced(
        model=model,
        processor=processor,
        image=image_input,
        question=sample.prompt_question,
        target=sample.target,
        device=device,
        hidden_state_index=hidden_state_index,
    )
    if not capture.capture_found:
        raise RuntimeError(f"Stage2 focus span was not captured: {capture.generated_text!r}")
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
        image_grid_thw=None if capture.image_grid_thw is None else capture.image_grid_thw.detach().cpu(),
        vision_tap=tap,
    )


@torch.no_grad()
def capture_v3_stage2_focus_teacher_forced(
    *,
    model: Any,
    processor: Any,
    image: Any,
    question: str,
    target: str,
    device: torch.device | str | None = None,
    hidden_state_index: int = -1,
) -> Qwen3FocusCapture:
    tokenizer = processor.tokenizer
    messages = build_direct_messages(image, question)
    inputs = build_qwen3_inputs(processor, messages)
    if device is None:
        device = _infer_model_device(model)
    model_inputs = _move_tensors_for_stage1(dict(inputs), device)
    base_input_ids = model_inputs["input_ids"]
    base_attention = model_inputs.get("attention_mask")
    source_visual_geometry = extract_qwen3_source_visual_geometry(model, model_inputs)
    forced_text = (
        f"{EVIDENCE_STATE_START}{NEED_LOCAL_EVIDENCE}{EVIDENCE_STATE_END}\n"
        f"{FOCUS_START}{target.strip()}{FOCUS_END}"
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
            [mm_token_type_ids, torch.zeros_like(forced_ids, dtype=mm_token_type_ids.dtype)],
            dim=-1,
        )
    outputs = model(
        **forward_inputs,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    generated_hidden = outputs.hidden_states[hidden_state_index][0, -int(forced_ids.shape[-1]) :]
    span = _find_focus_target_span_in_forced_ids(tokenizer, forced_ids.view(-1).tolist())
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
        model_kwargs={"focus_target_source": "teacher_forced_stage2"},
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


def prepare_v3_stage2_focus_inputs(
    *,
    model: Any,
    processor: Any,
    sample: TGVFv3Stage2Sample,
    capture: Qwen3FocusCapture,
    foveated_visual_tokens: torch.Tensor,
    loss_weights: Stage2LossWeights,
    device: torch.device | str,
    position_mode: PositionMode = "native_source_grid",
    mask_original_image_after_tgvf: bool = True,
    max_image_resolution: int | None = 512,
) -> dict[str, Any]:
    del position_mode
    if not capture.capture_found:
        raise ValueError("capture must contain a valid focus span")
    if capture.input_ids is None or capture.attention_mask is None:
        raise ValueError("capture input_ids and attention_mask are required")
    source_geometry = capture.source_visual_geometry
    if source_geometry is None:
        raise ValueError("capture is missing source visual geometry")
    source_count = int(source_geometry.source_visual_token_count)
    d = foveated_visual_tokens.to(device)
    if int(d.shape[0]) != source_count:
        raise ValueError(f"D token count {int(d.shape[0])} must equal source count {source_count}")

    tokenizer = processor.tokenizer
    base_ids = _base_direct_input_ids(
        processor=processor,
        image=_image_input(sample.image, max_image_resolution=max_image_resolution),
        question=sample.prompt_question,
        device=device,
    )
    base_len = int(base_ids.shape[-1])
    action_text = (
        f"{EVIDENCE_STATE_START}{NEED_LOCAL_EVIDENCE}{EVIDENCE_STATE_END}\n"
        f"{FOCUS_START}{sample.target}{FOCUS_END}"
    )
    action_ids, action_weights = _weighted_stage2_tokens(
        tokenizer,
        action_text,
        [
            (0, len(f"{EVIDENCE_STATE_START}{NEED_LOCAL_EVIDENCE}{EVIDENCE_STATE_END}"), loss_weights.evidence_state),
            (action_text.find(FOCUS_START), len(action_text), loss_weights.focus_target),
        ],
        device=device,
        default_weight=1.0,
    )
    tgvf_ids = _bracketed_visual_token_ids(
        processor,
        model,
        num_fvt_tokens=int(d.shape[0]),
        prefix=f"\n{TGVF_START}\n",
        suffix=f"\n{TGVF_END}\n",
        device=device,
    ).view(1, -1)
    ev_answer_text = (
        f"{EVIDENCE_START}{sample.evidence_description}{EVIDENCE_END}\n"
        f"{ANSWER_START}{sample.answer}{ANSWER_END}"
    )
    evidence_start = ev_answer_text.find(EVIDENCE_START)
    evidence_end = ev_answer_text.find(EVIDENCE_END) + len(EVIDENCE_END)
    answer_start = ev_answer_text.find(ANSWER_START)
    answer_end = len(ev_answer_text)
    spans = [
        (evidence_start, evidence_end, loss_weights.evidence),
        (answer_start, answer_end, loss_weights.answer),
    ]
    value_span_matched = False
    if sample.value_span_text:
        value_start = ev_answer_text.find(sample.value_span_text)
        evidence_inner_start = ev_answer_text.find(EVIDENCE_START) + len(EVIDENCE_START)
        evidence_inner_end = ev_answer_text.find(EVIDENCE_END, evidence_inner_start)
        if evidence_inner_start <= value_start < evidence_inner_end:
            spans.append(
                (
                    value_start,
                    value_start + len(sample.value_span_text),
                    loss_weights.value_span,
                )
            )
            value_span_matched = True
    ev_answer_ids, ev_answer_weights = _weighted_stage2_tokens(
        tokenizer,
        ev_answer_text,
        spans,
        device=device,
        default_weight=1.0,
    )
    input_ids = torch.cat([base_ids, action_ids, tgvf_ids, ev_answer_ids], dim=-1)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    weights = torch.zeros(input_ids.shape, dtype=torch.float32, device=device)
    action_start = base_len
    action_end = action_start + int(action_ids.shape[-1])
    ev_answer_start = action_end + int(tgvf_ids.shape[-1])
    ev_answer_end = ev_answer_start + int(ev_answer_ids.shape[-1])
    labels[:, action_start:action_end] = action_ids
    labels[:, ev_answer_start:ev_answer_end] = ev_answer_ids
    weights[:, action_start:action_end] = action_weights
    weights[:, ev_answer_start:ev_answer_end] = ev_answer_weights

    base_embeds = model.get_input_embeddings()(input_ids).detach()
    embeds = base_embeds.clone()
    image_token_id = getattr(getattr(model, "config", None), "image_token_id", None)
    if image_token_id is None:
        raise ValueError("Qwen3 image_token_id is unavailable")
    fvt_positions = torch.nonzero(input_ids[0] == int(image_token_id), as_tuple=False).view(-1)
    source_positions = source_geometry.source_visual_token_indices
    if source_positions is None:
        raise ValueError("source visual token indices are unavailable")
    original_count = int(source_positions.numel())
    fvt_positions = fvt_positions[-int(d.shape[0]) :]
    if int(fvt_positions.numel()) != int(d.shape[0]):
        raise ValueError("could not locate all TGVF image placeholder positions")
    embeds[:, fvt_positions, :] = d.to(dtype=embeds.dtype).unsqueeze(0)

    attention_mask_2d = torch.ones_like(input_ids)
    fvt_token_start = int(fvt_positions[0].detach().cpu().item())
    fvt_token_end = int(fvt_positions[-1].detach().cpu().item()) + 1
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
    position_ids = _compute_qwen3_position_ids_for_sequence(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask_2d,
        image_grid_thw=image_grid_thw,
        video_grid_thw=capture.video_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
    )
    if position_ids is None:
        raise RuntimeError("Qwen3 position id computation is unavailable")
    original_image_indices = source_positions.to(device=device, dtype=torch.long)
    block_query_start = action_end
    if mask_original_image_after_tgvf:
        attention_mask = build_weak_strict_attention_mask(
            attention_mask_2d=attention_mask_2d,
            original_image_token_indices=original_image_indices,
            block_query_start=block_query_start,
            dtype=embeds.dtype,
        )
        mask_mode = "weak_strict_original_image_keys_4d"
    else:
        attention_mask = attention_mask_2d
        mask_mode = "standard_2d_causal"
    return {
        "input_ids": input_ids,
        "inputs_embeds": embeds,
        "labels": labels,
        "loss_weights": weights,
        "attention_mask": attention_mask,
        "attention_mask_2d": attention_mask_2d,
        "position_ids": position_ids,
        "image_grid_thw": image_grid_thw,
        "mm_token_type_ids": mm_token_type_ids,
        "mask_mode": mask_mode,
        "image_key_mask_active": bool(mask_original_image_after_tgvf),
        "masked_image_key_count": original_count if mask_original_image_after_tgvf else 0,
        "fvt_shape": list(d.shape),
        "target_hidden_shape": list(capture.target_hidden_states.shape),
        "value_span_matched": value_span_matched,
        "need_focus": True,
    }


def prepare_v3_stage2_no_focus_inputs(
    *,
    model: Any,
    processor: Any,
    sample: TGVFv3Stage2Sample,
    loss_weights: Stage2LossWeights,
    device: torch.device | str,
    max_image_resolution: int | None = 512,
) -> dict[str, Any]:
    image_input = _image_input(sample.image, max_image_resolution=max_image_resolution)
    messages = build_direct_messages(image_input, sample.prompt_question)
    inputs = _move_tensors_for_stage1(build_qwen3_inputs(processor, messages), device)
    base_ids = inputs["input_ids"]
    base_len = int(base_ids.shape[-1])
    output_text = (
        f"{EVIDENCE_STATE_START}{SUFFICIENT_EVIDENCE}{EVIDENCE_STATE_END}\n"
        f"{ANSWER_START}{sample.answer}{ANSWER_END}"
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
    input_ids = torch.cat([base_ids, output_ids], dim=-1)
    attention_mask_2d = torch.cat(
        [
            inputs.get("attention_mask", torch.ones_like(base_ids)),
            torch.ones_like(output_ids),
        ],
        dim=-1,
    )
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    weights = torch.zeros(input_ids.shape, dtype=torch.float32, device=device)
    labels[:, base_len:] = output_ids
    weights[:, base_len:] = output_weights
    mm_token_type_ids = inputs.get("mm_token_type_ids")
    if isinstance(mm_token_type_ids, torch.Tensor):
        mm_token_type_ids = torch.cat(
            [
                mm_token_type_ids,
                torch.zeros_like(output_ids, dtype=mm_token_type_ids.dtype),
            ],
            dim=-1,
        )
    else:
        mm_token_type_ids = _full_mm_token_type_ids(
            model=model,
            input_ids=input_ids,
            fvt_token_start=input_ids.shape[-1],
            fvt_token_end=input_ids.shape[-1],
            device=device,
        )
    embeds = model.get_input_embeddings()(input_ids).detach()
    position_ids = _compute_qwen3_position_ids_for_sequence(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask_2d,
        image_grid_thw=inputs.get("image_grid_thw"),
        video_grid_thw=inputs.get("video_grid_thw"),
        mm_token_type_ids=mm_token_type_ids,
    )
    if position_ids is None:
        raise RuntimeError("Qwen3 position id computation is unavailable")
    return {
        "input_ids": input_ids,
        "inputs_embeds": embeds,
        "labels": labels,
        "loss_weights": weights,
        "attention_mask": attention_mask_2d,
        "attention_mask_2d": attention_mask_2d,
        "position_ids": position_ids,
        "image_grid_thw": inputs.get("image_grid_thw"),
        "mm_token_type_ids": mm_token_type_ids,
        "mask_mode": "standard_2d_causal",
        "image_key_mask_active": False,
        "masked_image_key_count": 0,
        "fvt_shape": None,
        "target_hidden_shape": None,
        "value_span_matched": bool(sample.value_span_text and sample.value_span_text in output_text),
        "need_focus": False,
    }


def compute_weighted_lm_loss(
    *,
    model: Any,
    trajectory_inputs: dict[str, Any],
) -> torch.Tensor:
    outputs = model(
        inputs_embeds=trajectory_inputs["inputs_embeds"],
        attention_mask=trajectory_inputs["attention_mask"],
        position_ids=trajectory_inputs["position_ids"],
        image_grid_thw=trajectory_inputs["image_grid_thw"],
        mm_token_type_ids=trajectory_inputs["mm_token_type_ids"],
        return_dict=True,
    )
    logits = outputs.logits
    labels = trajectory_inputs["labels"]
    weights = trajectory_inputs["loss_weights"].to(dtype=torch.float32)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_weights = weights[:, 1:].contiguous()
    per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view(shift_labels.shape)
    valid = (shift_labels != IGNORE_INDEX).to(per_token.dtype)
    weighted = per_token * valid * shift_weights
    denom = (valid * shift_weights).sum().clamp_min(1.0)
    return weighted.sum() / denom


def v3_stage2_training_step(
    *,
    qwen_model: Any,
    qwen_forward_model: Any,
    processor: Any,
    foveal_module: nn.Module,
    samples: list[TGVFv3Stage2Sample],
    loss_weights: Stage2LossWeights,
    device: torch.device | str,
    hidden_state_index: int = -1,
    max_image_resolution: int | None = 512,
    position_mode: PositionMode = "native_source_grid",
    mask_original_image_after_tgvf: bool = True,
) -> TGVFv3Stage2StepOutput:
    focus_losses: list[torch.Tensor] = []
    no_focus_losses: list[torch.Tensor] = []
    manifold_losses: list[torch.Tensor] = []
    debug_items: list[dict[str, Any]] = []
    vision_cache: dict[str, tuple[Any, torch.Tensor, torch.Tensor]] = {}
    value_span_attempted = 0
    value_span_matched = 0

    for sample in samples:
        if sample.need_focus:
            feature = collect_v3_stage2_focus_features(
                model=qwen_model,
                processor=processor,
                sample=sample,
                device=device,
                hidden_state_index=hidden_state_index,
                max_image_resolution=max_image_resolution,
                vision_cache=vision_cache,
            )
            output = foveal_module(
                target_hidden_states=feature.target_hidden_states.to(device),
                pre_merge_visual_tokens=feature.pre_merge_visual_tokens.to(device),
                metadata={
                    "target": sample.target,
                    "stage": "tgvf_v3_stage2",
                    "evidence_state": NEED_LOCAL_EVIDENCE,
                },
            )
            output = finalize_tgvf_output_with_frozen_qwen_merger(qwen_model, output)
            d = output.foveated_visual_tokens
            trajectory_inputs = prepare_v3_stage2_focus_inputs(
                model=qwen_model,
                processor=processor,
                sample=sample,
                capture=feature.capture,
                foveated_visual_tokens=d,
                loss_weights=loss_weights,
                device=device,
                position_mode=position_mode,
                mask_original_image_after_tgvf=mask_original_image_after_tgvf,
                max_image_resolution=max_image_resolution,
            )
            loss = compute_weighted_lm_loss(
                model=qwen_forward_model,
                trajectory_inputs=trajectory_inputs,
            )
            focus_losses.append(loss)
            if loss_weights.visual_token_manifold:
                manifold_losses.append(
                    _safe_visual_token_manifold_loss(d, feature.merged_visual_tokens.to(device))
                )
            if sample.value_span_text:
                value_span_attempted += 1
                value_span_matched += int(bool(trajectory_inputs["value_span_matched"]))
        else:
            trajectory_inputs = prepare_v3_stage2_no_focus_inputs(
                model=qwen_model,
                processor=processor,
                sample=sample,
                loss_weights=loss_weights,
                device=device,
                max_image_resolution=max_image_resolution,
            )
            loss = compute_weighted_lm_loss(
                model=qwen_forward_model,
                trajectory_inputs=trajectory_inputs,
            )
            no_focus_losses.append(loss)
            if sample.value_span_text:
                value_span_attempted += 1
                value_span_matched += int(bool(trajectory_inputs["value_span_matched"]))
        if len(debug_items) < 2:
            debug_items.append(
                {
                    "need_focus": sample.need_focus,
                    "question": sample.prompt_question,
                    "target": sample.target,
                    "answer": sample.answer,
                    "mask_mode": trajectory_inputs["mask_mode"],
                    "image_key_mask_active": trajectory_inputs["image_key_mask_active"],
                    "masked_image_key_count": trajectory_inputs["masked_image_key_count"],
                    "fvt_shape": trajectory_inputs["fvt_shape"],
                    "target_hidden_shape": trajectory_inputs["target_hidden_shape"],
                    "value_span_matched": trajectory_inputs["value_span_matched"],
                }
            )

    zero = (
        focus_losses[0].new_zeros(())
        if focus_losses
        else no_focus_losses[0].new_zeros(())
        if no_focus_losses
        else torch.zeros((), device=device)
    )
    loss_focus = torch.stack(focus_losses).mean() if focus_losses else zero
    loss_no_focus = torch.stack(no_focus_losses).mean() if no_focus_losses else zero
    loss_man = torch.stack(manifold_losses).mean() if manifold_losses else zero
    active = int(bool(focus_losses)) + int(bool(no_focus_losses))
    loss_total = (loss_focus + loss_no_focus) / max(active, 1)
    loss_total = loss_total + float(loss_weights.visual_token_manifold) * loss_man
    focus_count = len(focus_losses)
    no_focus_count = len(no_focus_losses)
    return TGVFv3Stage2StepOutput(
        loss_total=loss_total,
        loss_focus=loss_focus,
        loss_no_focus=loss_no_focus,
        loss_visual_token_manifold=loss_man,
        debug={
            "focus_count": focus_count,
            "no_focus_count": no_focus_count,
            "focus_ratio_batch": focus_count / max(len(samples), 1),
            "no_focus_ratio_batch": no_focus_count / max(len(samples), 1),
            "matrix_ce_enabled": False,
            "same_image_negative_enabled": False,
            "contrastive_alignment_enabled": False,
            "special_tokens_added": False,
            "tokenizer_resized": False,
            "markers_are_plain_text": True,
            "mask_mode": "weak_strict_original_image_keys_4d",
            "focus_sample_mask_active_rate": 1.0 if focus_count else 0.0,
            "no_focus_mask_active_rate": 0.0,
            "value_span_match_rate": (
                value_span_matched / value_span_attempted if value_span_attempted else None
            ),
            "debug_examples": debug_items,
        },
    )


def _base_direct_input_ids(
    *,
    processor: Any,
    image: Any,
    question: str,
    device: torch.device | str,
) -> torch.Tensor:
    messages = build_direct_messages(image, question)
    inputs = _move_tensors_for_stage1(build_qwen3_inputs(processor, messages), device)
    return inputs["input_ids"]


def _weighted_stage2_tokens(
    tokenizer: Any,
    text: str,
    spans: list[tuple[int, int, float]],
    *,
    device: torch.device | str,
    default_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = _encode_text(tokenizer, text, device).view(1, -1)
    offsets = _decoded_token_offsets(tokenizer, ids.view(-1).detach().cpu().tolist())
    weights = torch.full((1, int(ids.shape[-1])), float(default_weight), device=device)
    for char_start, char_end, weight in spans:
        if char_start < 0 or char_end <= char_start:
            continue
        for index, (tok_start, tok_end) in enumerate(offsets):
            if tok_start < char_end and tok_end > char_start:
                weights[0, index] = float(weight)
    return ids, weights


def _decoded_token_offsets(tokenizer: Any, token_ids: list[int]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token_id in token_ids:
        token_text = _decode(tokenizer, [token_id])
        end = cursor + len(token_text)
        offsets.append((cursor, end))
        cursor = end
    return offsets


def dataset_stage2_stats(dataset: TGVFv3Stage2Dataset) -> dict[str, Any]:
    return dict(dataset.stats, skipped_rows=len(dataset.skipped_rows), path=str(dataset.jsonl_path))
