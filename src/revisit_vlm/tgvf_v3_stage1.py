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
    Qwen3FocusCapture,
    _bracketed_visual_token_ids,
    _compute_qwen3_position_ids_for_sequence,
    _encode_text,
    capture_focus_single_pass_qwen3,
    llm_hidden_dim,
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
        choice_lines = [str(choice) for choice in choices]
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
) -> TGVFv3Stage1Features:
    image_input = _image_input(sample.image, max_image_resolution=max_image_resolution)
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
    )
    if not capture.capture_found:
        raise RuntimeError(f"Forced v3 focus span was not captured: {capture.generated_text!r}")
    tap, v_pre, v_merge = tap_qwen3_vision_features(
        model,
        processor,
        image=image_input,
        question=sample.prompt_question,
        device=device,
    )
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
    if device is None:
        device = _infer_model_device(model) or foveated_visual_tokens.device
    base_input_ids = capture.input_ids.to(device)
    base_len = int(base_input_ids.shape[-1])
    d = foveated_visual_tokens.to(device)
    prefix = f"\n{TGVF_START}\n"
    suffix = f"\n{TGVF_END}\n{EVIDENCE_START}"
    tgvf_ids = _bracketed_visual_token_ids(
        tokenizer_or_processor,
        model,
        num_fvt_tokens=int(d.shape[0]),
        prefix=prefix,
        suffix=suffix,
        device=device,
    ).view(1, -1)
    evidence_ids = _encode_text(tokenizer, f"{evidence_description}{EVIDENCE_END}", device).view(1, -1)
    input_ids = torch.cat([base_input_ids, tgvf_ids, evidence_ids], dim=-1)

    prefix_ids = _encode_text(tokenizer, prefix, device)
    local_fvt_start = int(prefix_ids.shape[0]) + 1
    local_fvt_end = local_fvt_start + int(d.shape[0])
    fvt_token_start = base_len + local_fvt_start
    fvt_token_end = base_len + local_fvt_end
    evidence_start = base_len + int(tgvf_ids.shape[-1])

    base_embeds = model.get_input_embeddings()(input_ids).detach()
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
    }


def build_weak_strict_attention_mask(
    *,
    attention_mask_2d: torch.Tensor,
    original_image_token_indices: torch.Tensor,
    block_query_start: int,
    dtype: torch.dtype,
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
        query_indices = torch.nonzero(
            torch.arange(seq_len, device=device) >= int(block_query_start),
            as_tuple=False,
        ).view(-1)
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
    outputs = model(
        inputs_embeds=readout_inputs["inputs_embeds"],
        attention_mask=readout_inputs["attention_mask"],
        position_ids=readout_inputs["position_ids"],
        image_grid_thw=readout_inputs["image_grid_thw"],
        mm_token_type_ids=readout_inputs["mm_token_type_ids"],
        labels=readout_inputs["labels"],
        return_dict=True,
    )
    logits = outputs.logits
    labels = readout_inputs["labels"]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    token_count = (shift_labels != IGNORE_INDEX).sum().clamp_min(1)
    nll_sum = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    return nll_sum / token_count, -nll_sum


def v3_stage1_training_step(
    *,
    qwen_model: Any,
    processor: Any,
    foveal_module: nn.Module,
    samples: list[TGVFv3Stage1Sample],
    loss_weights: LossWeights,
    device: torch.device | str,
    hidden_state_index: int = -1,
    same_image_negative_margin: float = 1.0,
    same_image_negative_mode: str = "matrix_ce",
    mask_original_image_after_tgvf: bool = True,
    position_mode: PositionMode = "native_source_grid",
    max_image_resolution: int | None = 512,
) -> TGVFTrainStepOutput:
    features = [
        collect_v3_stage1_features(
            model=qwen_model,
            processor=processor,
            sample=sample,
            device=device,
            hidden_state_index=hidden_state_index,
            max_image_resolution=max_image_resolution,
        )
        for sample in samples
    ]
    fvt_outputs: list[torch.Tensor] = []
    loss_gen_values: list[torch.Tensor] = []
    positive_ll: list[torch.Tensor] = []
    loss_man_values: list[torch.Tensor] = []
    attention_debug_values: list[dict[str, Any]] = []
    norm_debug_values: list[dict[str, Any]] = []
    readout_debug_values: list[dict[str, Any]] = []

    for sample, feature in zip(samples, features, strict=True):
        output = foveal_module(
            target_hidden_states=feature.target_hidden_states.to(device),
            pre_merge_visual_tokens=feature.pre_merge_visual_tokens.to(device),
            metadata={
                "target": sample.target,
                "stage": "tgvf_v3_stage1",
                "evidence_state": NEED_LOCAL_EVIDENCE,
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
        )
        gen_loss, log_likelihood = compute_v3_stage1_lm_loss(
            model=qwen_model,
            readout_inputs=readout_inputs,
        )
        loss_gen_values.append(gen_loss)
        positive_ll.append(log_likelihood)
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
                rows = []
                for pos_index in indices:
                    row = []
                    for fvt_index in indices:
                        if pos_index == fvt_index:
                            row.append(positive_ll[pos_index])
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
                        )
                        _, log_likelihood = compute_v3_stage1_lm_loss(
                            model=qwen_model,
                            readout_inputs=readout_inputs,
                        )
                        row.append(log_likelihood)
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
            "norm_diagnostics": summarize_diagnostics(norm_debug_values),
            "finite_rate": float(torch.stack([value.cpu() for value in finite_values]).mean()),
            "visual_token_manifold_active": bool(
                first_d.shape[-1] == first_feature.merged_visual_tokens.shape[-1]
            ),
            "qwen_frozen": not any(parameter.requires_grad for parameter in qwen_model.parameters()),
            "second_full_forward_used": False,
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
    )
    return {key: readout_inputs.get(key) for key in keys}


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
