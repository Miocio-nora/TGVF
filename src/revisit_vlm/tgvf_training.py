from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from revisit_vlm.tgvf import FOVEATE_END, FOVEATE_START
from revisit_vlm.tgvf_capture import capture_tgvf_single_pass
from revisit_vlm.tgvf_foveal import (
    BracketedFVTAppendResult,
    FovealCrossMerger,
    PooledFovealCrossAttention,
    Qwen2VLPreMergeVisualHook,
    TargetSlotFovealCrossMerger,
    TGVFv2Bidirectional,
    TGVFv2CrossAttention,
    TGVFv2VPTGating,
    TokenFovealCrossAttention,
    _expected_llm_image_tokens,
    _fvt_chunk_position_ids,
    _fvt_mm_token_type_ids,
    _infer_spatial_merge_size,
    _position_ids_have_3d_image_span,
    _rope_delta_for_next_token,
    _text_positions_are_1d,
    bracketed_fvt_token_ids,
    continue_generation_from_state,
    finalize_tgvf_output_with_frozen_qwen_merger,
    make_fake_image_grid,
)

IGNORE_INDEX = -100
TGVF_V2_VARIANTS = (
    "tgvf_v2_vpt_gating",
    "tgvf_v2_cross_attention",
    "tgvf_v2_bidirectional",
)
TGVF_VARIANTS = (
    "token_direct",
    "pooled",
    "foveal_cross_merger",
    "target_slot_foveal_cross_merger",
    *TGVF_V2_VARIANTS,
)
TGVF_DYNAMIC_NUM_FVT_VARIANTS = ("token_direct", *TGVF_V2_VARIANTS)


@dataclass
class TeacherGuideSample:
    image: str
    question: str
    target: str
    evidence_description: str
    image_id: str | None = None
    short_answer: str | None = None
    evidence_type: str | None = None
    confidence: float | None = None
    source_dataset: str | None = None
    teacher_model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LossWeights:
    gen: float = 1.0
    same_image_negative: float = 0.0
    contrastive_alignment: float = 0.0
    visual_token_manifold: float = 0.01


@dataclass
class TGVFModuleConfig:
    variant: str = "foveal_cross_merger"
    num_foveated_tokens: int | None = 16
    spatial_merge_size: int = 2
    attn_dim: int | None = None


@dataclass
class TGVFTrainingConfig:
    model_name_or_path: str = "Qwen/Qwen2-VL-2B-Instruct"
    tgvf: TGVFModuleConfig = field(default_factory=TGVFModuleConfig)
    loss_weights: LossWeights = field(default_factory=LossWeights)
    learning_rate: float = 1e-4
    lr_scheduler: str = "none"
    warmup_ratio: float = 0.03
    warmup_steps: int = 0
    min_lr_ratio: float = 0.1
    num_training_steps: int | None = None
    estimated_optimizer_steps: int | None = None
    max_steps_semantics: str = "micro_batch_steps"
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_steps: int = 1000
    save_every: int = 100
    capture_layer: int = -1
    qwen_freeze: bool = True
    contrastive_temperature: float = 0.07
    same_image_negative_margin: float = 1.0
    same_image_negative_mode: str = "cyclic_margin"
    readout_prompt_target_dropout: float = 0.3
    readout_append_mode: str = "qwen_native_pseudo_image"


@dataclass
class TGVFTrainingFeatures:
    capture: Any
    target_hidden_states: torch.Tensor
    pre_merge_visual_tokens: torch.Tensor
    merged_visual_tokens: torch.Tensor
    image_grid_thw: torch.Tensor | None
    forward_input_lengths: list[int]


@dataclass
class TGVFTrainStepOutput:
    loss_total: torch.Tensor
    loss_gen: torch.Tensor
    loss_visual_token_manifold: torch.Tensor
    loss_same_image_negative: torch.Tensor
    loss_contrastive_alignment: torch.Tensor
    debug: dict[str, Any]


class TeacherGuideDataset(Dataset[TeacherGuideSample]):
    required_fields = ("image", "question", "target", "evidence_description")

    def __init__(
        self,
        jsonl_path: str | Path,
        *,
        warn_on_leakage: bool = True,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.warn_on_leakage = warn_on_leakage
        self.samples = [self._parse_record(record, line_no) for line_no, record in self._records()]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TeacherGuideSample:
        return self.samples[index]

    def _records(self) -> list[tuple[int, dict[str, Any]]]:
        records = []
        with self.jsonl_path.open() as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                records.append((line_no, json.loads(line)))
        return records

    def _parse_record(self, record: dict[str, Any], line_no: int) -> TeacherGuideSample:
        missing = [field for field in self.required_fields if not record.get(field)]
        if missing:
            raise ValueError(f"{self.jsonl_path}:{line_no} missing required fields: {missing}")

        image = Path(record["image"])
        if not image.is_absolute():
            image = (self.jsonl_path.parent / image).resolve()

        sample = TeacherGuideSample(
            image=str(image),
            question=record["question"],
            target=record["target"],
            evidence_description=record["evidence_description"],
            image_id=record.get("image_id"),
            short_answer=record.get("short_answer"),
            evidence_type=record.get("evidence_type"),
            confidence=record.get("confidence"),
            source_dataset=record.get("source_dataset"),
            teacher_model=record.get("teacher_model"),
            metadata=record.get("metadata") or {},
        )
        if self.warn_on_leakage:
            warn_if_target_leaks_answer(sample, source=f"{self.jsonl_path}:{line_no}")
        return sample


def generate_teacher_guided_dataset(*_args: Any, **_kwargs: Any) -> None:
    raise NotImplementedError(
        "Teacher-guide dataset generation is intentionally out of scope for v0. "
        "Prepare a JSONL file with image/question/target/evidence_description fields."
    )


def warn_if_target_leaks_answer(sample: TeacherGuideSample, *, source: str | None = None) -> None:
    if not sample.short_answer:
        return
    target_terms = set(_terms(sample.target))
    answer_terms = set(_terms(sample.short_answer))
    if not target_terms or not answer_terms:
        return
    overlap = target_terms & answer_terms
    if overlap:
        prefix = f"{source}: " if source else ""
        warnings.warn(
            f"{prefix}target may leak short_answer terms: {sorted(overlap)}",
            stacklevel=2,
        )


def freeze_qwen2vl(model: nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False


def build_tgvf_module(
    *,
    variant: str,
    d_lm: int,
    d_v: int,
    num_foveated_tokens: int | None,
    spatial_merge_size: int = 2,
    attn_dim: int | None = None,
) -> nn.Module:
    if variant not in TGVF_DYNAMIC_NUM_FVT_VARIANTS and num_foveated_tokens is None:
        supported = ", ".join(TGVF_DYNAMIC_NUM_FVT_VARIANTS)
        raise ValueError(f"num_foveated_tokens=None is supported only for: {supported}")
    if variant == "token_direct":
        return TokenFovealCrossAttention(
            d_lm=d_lm,
            d_v=d_v,
            attn_dim=attn_dim,
            num_output_tokens=num_foveated_tokens,
        )
    if variant == "tgvf_v2_vpt_gating":
        return TGVFv2VPTGating(
            d_lm=d_lm,
            d_v=d_v,
            spatial_merge_size=spatial_merge_size,
        )
    if variant == "tgvf_v2_cross_attention":
        return TGVFv2CrossAttention(
            d_lm=d_lm,
            d_v=d_v,
            spatial_merge_size=spatial_merge_size,
            attn_dim=attn_dim,
        )
    if variant == "tgvf_v2_bidirectional":
        return TGVFv2Bidirectional(
            d_lm=d_lm,
            d_v=d_v,
            spatial_merge_size=spatial_merge_size,
            attn_dim=attn_dim,
        )
    if variant == "pooled":
        return PooledFovealCrossAttention(
            d_lm=d_lm,
            d_v=d_v,
            num_fvt_tokens=int(num_foveated_tokens),
            attn_dim=attn_dim,
        )
    if variant == "foveal_cross_merger":
        return FovealCrossMerger(
            d_lm=d_lm,
            d_v=d_v,
            num_fvt_tokens=int(num_foveated_tokens),
            spatial_merge_size=spatial_merge_size,
        )
    if variant in {"target_slot", "target_slot_cross_merger", "target_slot_foveal_cross_merger"}:
        return TargetSlotFovealCrossMerger(
            d_lm=d_lm,
            d_v=d_v,
            num_fvt_tokens=int(num_foveated_tokens),
            spatial_merge_size=spatial_merge_size,
            num_heads=8,
            slot_dim=attn_dim,
        )
    raise ValueError(f"Unsupported TGVF variant: {variant}")


def infer_tgvf_dims(model: Any) -> tuple[int, int, int]:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    d_lm = getattr(text_config, "hidden_size", None) or config.hidden_size
    d_v = None
    spatial_merge_size = 2
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        visual = model.model.visual
        spatial_merge_size = getattr(visual, "spatial_merge_size", 2)
        merger = getattr(visual, "merger", None)
        ln_q = getattr(merger, "ln_q", None)
        normalized_shape = getattr(ln_q, "normalized_shape", None)
        if normalized_shape:
            d_v = normalized_shape[0]
        if d_v is None:
            patch_proj = getattr(getattr(visual, "patch_embed", None), "proj", None)
            d_v = getattr(patch_proj, "out_channels", None)
    if d_v is None:
        raise ValueError("Could not infer pre-merge visual hidden size from Qwen2-VL visual merger")
    return int(d_lm), int(d_v), int(spatial_merge_size)


@torch.no_grad()
def collect_training_features(
    *,
    model: Any,
    processor: Any,
    sample: TeacherGuideSample,
    device: torch.device | str,
    hidden_state_index: int = -1,
) -> TGVFTrainingFeatures:
    forced_text = f"{FOVEATE_START}{sample.target}{FOVEATE_END}"
    forced_ids = processor.tokenizer.encode(forced_text, add_special_tokens=False)
    wrapped_model = ForcedFoveationWrapper(model, forced_ids)
    merged_hook = Qwen2VLMergedVisualHook(model)
    try:
        with Qwen2VLPreMergeVisualHook(model) as pre_hook:
            capture = capture_tgvf_single_pass(
                wrapped_model,
                processor,
                image=sample.image,
                question=sample.question,
                max_new_tokens=len(forced_ids) + 8,
                device=device,
                hidden_state_index=hidden_state_index,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
    finally:
        merged_hook.close()

    if not capture.capture_found:
        raise RuntimeError(f"Forced foveation request was not captured: {capture.generated_text!r}")
    if pre_hook.pre_merge_visual_tokens is None:
        raise RuntimeError("Pre-merge visual tokens were not captured")
    if merged_hook.merged_visual_tokens is None:
        raise RuntimeError("Merged visual tokens were not captured")

    image_grid_thw = (
        None if capture.image_grid_thw is None else capture.image_grid_thw.detach().cpu()
    )
    capture_summary = SimpleNamespace(generated_text=capture.generated_text)
    return TGVFTrainingFeatures(
        capture=capture_summary,
        target_hidden_states=capture.target_hidden_states.detach().cpu(),
        pre_merge_visual_tokens=pre_hook.pre_merge_visual_tokens.detach().cpu(),
        merged_visual_tokens=merged_hook.merged_visual_tokens.detach().cpu(),
        image_grid_thw=image_grid_thw,
        forward_input_lengths=wrapped_model.forward_input_lengths,
    )


class ForcedFoveationWrapper:
    def __init__(self, model: Any, forced_ids: list[int]) -> None:
        self.model = model
        self.forced_ids = forced_ids
        self.forward_calls = 0
        self.forward_input_lengths: list[int] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        outputs = self.model(*args, **kwargs)
        self._record_input_length(args, kwargs)
        step = self.forward_calls
        self.forward_calls += 1
        if step < len(self.forced_ids):
            logits = torch.full_like(outputs.logits, -1e4)
            logits[:, -1, self.forced_ids[step]] = 1e4
            outputs.logits = logits
        return outputs

    def _record_input_length(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is not None:
            self.forward_input_lengths.append(int(input_ids.shape[-1]))

    def prepare_inputs_for_generation(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def parameters(self) -> Any:
        return self.model.parameters()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)


class ForcedFoveationBracketWrapper:
    """Force only the foveation markers while leaving the target text model-generated."""

    def __init__(
        self,
        model: Any,
        *,
        start_ids: list[int],
        end_ids: list[int],
        max_target_tokens: int = 24,
        suppress_ids: list[int] | None = None,
    ) -> None:
        if not start_ids or not end_ids:
            raise ValueError("start_ids and end_ids must be non-empty")
        if max_target_tokens < 1:
            raise ValueError("max_target_tokens must be >= 1")
        self.model = model
        self.start_ids = list(start_ids)
        self.end_ids = list(end_ids)
        self.max_target_tokens = int(max_target_tokens)
        self.suppress_ids = list(suppress_ids or [])
        self.forward_calls = 0
        self.forward_input_lengths: list[int] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        outputs = self.model(*args, **kwargs)
        self._record_input_length(args, kwargs)
        step = self.forward_calls
        self.forward_calls += 1
        forced_id = self._forced_id_for_step(step)
        if forced_id is not None:
            logits = torch.full_like(outputs.logits, -1e4)
            logits[:, -1, forced_id] = 1e4
            outputs.logits = logits
        elif self._is_target_step(step) and self.suppress_ids:
            logits = outputs.logits.clone()
            valid_suppress_ids = [idx for idx in self.suppress_ids if 0 <= idx < logits.shape[-1]]
            if valid_suppress_ids:
                logits[:, -1, valid_suppress_ids] = -1e4
                outputs.logits = logits
        return outputs

    def _forced_id_for_step(self, step: int) -> int | None:
        if step < len(self.start_ids):
            return self.start_ids[step]
        end_start = len(self.start_ids) + self.max_target_tokens
        if end_start <= step < end_start + len(self.end_ids):
            return self.end_ids[step - end_start]
        return None

    def _is_target_step(self, step: int) -> bool:
        return len(self.start_ids) <= step < len(self.start_ids) + self.max_target_tokens

    def _record_input_length(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is not None:
            self.forward_input_lengths.append(int(input_ids.shape[-1]))

    def prepare_inputs_for_generation(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def parameters(self) -> Any:
        return self.model.parameters()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)


class Qwen2VLMergedVisualHook:
    def __init__(self, model: Any) -> None:
        visual = _visual_module(model)
        if not hasattr(visual, "merger"):
            raise AttributeError("Qwen2-VL visual module does not expose a merger module")
        self.merged_visual_tokens: torch.Tensor | None = None
        self.handle = visual.merger.register_forward_hook(self._capture)

    def _capture(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        if isinstance(output, torch.Tensor):
            self.merged_visual_tokens = output.detach()

    def close(self) -> None:
        self.handle.remove()

    def __enter__(self) -> Qwen2VLMergedVisualHook:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def prepare_readout_inputs(
    *,
    model: Any,
    tokenizer_or_processor: Any,
    target: str | None,
    evidence_description: str,
    foveated_visual_tokens: torch.Tensor,
    image_grid_thw: torch.Tensor | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    prefix = prepare_readout_prefix_inputs(
        model=model,
        tokenizer_or_processor=tokenizer_or_processor,
        target=target,
        foveated_visual_tokens=foveated_visual_tokens,
        image_grid_thw=image_grid_thw,
        device=device,
    )
    tokenizer = _tokenizer(tokenizer_or_processor)
    answer_ids = _encode(tokenizer, evidence_description)
    answer_input_ids = torch.tensor(
        [answer_ids],
        dtype=torch.long,
        device=prefix["input_ids"].device,
    )
    answer_embeds = model.get_input_embeddings()(answer_input_ids)
    input_ids = torch.cat([prefix["input_ids"], answer_input_ids], dim=-1)
    embeds = torch.cat([prefix["inputs_embeds"], answer_embeds], dim=1)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    answer_start = int(prefix["prefix_token_count"])
    labels[:, answer_start:] = input_ids[:, answer_start:]
    attention_mask = torch.ones_like(input_ids)
    position_ids = _fvt_chunk_position_ids(
        attention_mask=attention_mask,
        chunk_length=int(input_ids.shape[-1]),
        fvt_token_start=int(prefix["fvt_token_start"]),
        fvt_token_end=int(prefix["fvt_token_end"]),
        fvt_grid_thw=prefix["image_grid_thw"],
        spatial_merge_size=int(prefix["spatial_merge_size"]),
        rope_deltas=None,
        device=embeds.device,
    )
    image_position_ids_are_3d = _position_ids_have_3d_image_span(
        position_ids,
        int(prefix["fvt_token_start"]),
        int(prefix["fvt_token_end"]),
    )
    text_position_ids_are_1d = _text_positions_are_1d(
        position_ids,
        int(prefix["fvt_token_start"]),
        int(prefix["fvt_token_end"]),
    )
    if not image_position_ids_are_3d:
        raise AssertionError("training FVT readout image positions must use 3D image-style mRoPE")
    return {
        "input_ids": input_ids,
        "inputs_embeds": embeds,
        "labels": labels,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "image_grid_thw": prefix["image_grid_thw"],
        "mm_token_type_ids": torch.cat(
            [
                prefix["mm_token_type_ids"],
                torch.zeros_like(answer_input_ids),
            ],
            dim=-1,
        ),
        "answer_token_count": len(answer_ids),
        "fvt_token_start": prefix["fvt_token_start"],
        "fvt_token_end": prefix["fvt_token_end"],
        "answer_start": answer_start,
        "readout_append_mode": prefix["readout_append_mode"],
        "fake_image_grid_thw": prefix["fake_image_grid_thw"],
        "source_image_grid_thw": prefix["source_image_grid_thw"],
        "fvt_grid_thw": prefix["fvt_grid_thw"],
        "fvt_grid_source": prefix["fvt_grid_source"],
        "spatial_merge_size": prefix["spatial_merge_size"],
        "expected_llm_image_tokens": prefix["expected_llm_image_tokens"],
        "actual_image_pad_token_count": prefix["actual_image_pad_token_count"],
        "mm_token_type_ids_present": True,
        "image_pad_mm_type_is_image": prefix["image_pad_mm_type_is_image"],
        "position_ids_source": prefix["position_ids_source"],
        "image_position_ids_are_3d": bool(image_position_ids_are_3d),
        "text_position_ids_are_1d": bool(text_position_ids_are_1d),
        "visual_tower_called_for_fvt": False,
    }


def prepare_readout_prefix_inputs(
    *,
    model: Any,
    tokenizer_or_processor: Any,
    target: str | None,
    foveated_visual_tokens: torch.Tensor,
    image_grid_thw: torch.Tensor | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    tokenizer = _tokenizer(tokenizer_or_processor)
    if device is None:
        device = _infer_model_device(model) or foveated_visual_tokens.device
    before, after = readout_prompt_parts(target)
    before_ids = _encode(tokenizer, before)
    bracket_ids = bracketed_fvt_token_ids(
        tokenizer_or_processor=tokenizer,
        model=model,
        num_fvt_tokens=foveated_visual_tokens.shape[0],
        device=device,
    ).tolist()
    after_ids = _encode(tokenizer, after)
    input_ids = torch.tensor(
        [before_ids + bracket_ids + after_ids],
        dtype=torch.long,
        device=device,
    )
    embeds = model.get_input_embeddings()(input_ids)
    fvt_start = len(before_ids) + 1
    fvt_end = fvt_start + foveated_visual_tokens.shape[0]
    embeds[:, fvt_start:fvt_end, :] = foveated_visual_tokens.to(
        device=embeds.device,
        dtype=embeds.dtype,
    ).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    spatial_merge_size = _infer_spatial_merge_size(model)
    source_image_grid_thw = None
    fake_image_grid_thw = None
    if image_grid_thw is None:
        fvt_grid_thw = make_fake_image_grid(
            num_fvt_tokens=int(foveated_visual_tokens.shape[0]),
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
    num_fvt_tokens = int(foveated_visual_tokens.shape[0])
    if expected_llm_image_tokens != num_fvt_tokens:
        raise AssertionError(
            f"expected_llm_image_tokens={expected_llm_image_tokens} "
            f"!= num_fvt_tokens={num_fvt_tokens}"
        )
    image_token_id = bracket_ids[1]
    actual_image_pad_token_count = int((input_ids == image_token_id).sum().detach().cpu().item())
    mm_token_type_ids = _fvt_mm_token_type_ids(
        chunk_length=int(input_ids.shape[-1]),
        fvt_token_start=fvt_start,
        fvt_token_end=fvt_end,
        device=embeds.device,
    )
    image_pad_mm_type_is_image = bool(
        torch.all(mm_token_type_ids[:, fvt_start:fvt_end] == 1).detach().cpu().item()
    )
    position_ids = _fvt_chunk_position_ids(
        attention_mask=attention_mask,
        chunk_length=int(input_ids.shape[-1]),
        fvt_token_start=fvt_start,
        fvt_token_end=fvt_end,
        fvt_grid_thw=fvt_grid_thw,
        spatial_merge_size=spatial_merge_size,
        rope_deltas=None,
        device=embeds.device,
    )
    image_position_ids_are_3d = _position_ids_have_3d_image_span(position_ids, fvt_start, fvt_end)
    text_position_ids_are_1d = _text_positions_are_1d(position_ids, fvt_start, fvt_end)
    if actual_image_pad_token_count != num_fvt_tokens:
        raise AssertionError(
            f"actual_image_pad_token_count={actual_image_pad_token_count} "
            f"!= num_fvt_tokens={num_fvt_tokens}"
        )
    if not image_pad_mm_type_is_image:
        raise AssertionError("training FVT image_pad tokens must be marked as image modality")
    if not image_position_ids_are_3d:
        raise AssertionError("training FVT readout image positions must use 3D image-style mRoPE")
    return {
        "input_ids": input_ids,
        "inputs_embeds": embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "image_grid_thw": fvt_grid_thw,
        "mm_token_type_ids": mm_token_type_ids,
        "fvt_token_start": fvt_start,
        "fvt_token_end": fvt_end,
        "prefix_token_count": input_ids.shape[-1],
        "readout_append_mode": "qwen_native_pseudo_image",
        "fake_image_grid_thw": (
            None if fake_image_grid_thw is None else fake_image_grid_thw.detach().cpu().tolist()
        ),
        "source_image_grid_thw": (
            None if source_image_grid_thw is None else source_image_grid_thw.detach().cpu().tolist()
        ),
        "fvt_grid_thw": fvt_grid_thw.detach().cpu().tolist(),
        "fvt_grid_source": fvt_grid_source,
        "position_ids_source": (
            "source_image_grid_mrope"
            if fvt_grid_source == "source_image_grid_thw"
            else "manual_fake_grid_mrope"
        ),
        "spatial_merge_size": int(spatial_merge_size),
        "expected_llm_image_tokens": expected_llm_image_tokens,
        "actual_image_pad_token_count": actual_image_pad_token_count,
        "image_pad_mm_type_is_image": bool(image_pad_mm_type_is_image),
        "image_position_ids_are_3d": bool(image_position_ids_are_3d),
        "text_position_ids_are_1d": bool(text_position_ids_are_1d),
        "visual_tower_called_for_fvt": False,
    }


def readout_prompt_parts(target: str | None) -> tuple[str, str]:
    before = "<|im_start|>user\n"
    if target is None:
        after = (
            "\nThe visual tokens above are focused evidence.\n\n"
            "Describe only what is visible in this focused evidence.\n"
            "<|im_end|>\n<|im_start|>assistant\n"
        )
    else:
        after = (
            "\nThe visual tokens above are focused evidence for the target:\n"
            f"{target}\n\n"
            "Describe only what is visible in this focused evidence.\n"
            "<|im_end|>\n<|im_start|>assistant\n"
        )
    return before, after


def compute_readout_lm_loss(
    *,
    model: Any,
    readout_inputs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(
        input_ids=readout_inputs.get("input_ids"),
        inputs_embeds=readout_inputs["inputs_embeds"],
        attention_mask=readout_inputs["attention_mask"],
        position_ids=readout_inputs.get("position_ids"),
        image_grid_thw=readout_inputs.get("image_grid_thw"),
        mm_token_type_ids=readout_inputs.get("mm_token_type_ids"),
        labels=readout_inputs["labels"],
        return_dict=True,
    )
    if getattr(outputs, "loss", None) is not None:
        loss = outputs.loss
    else:
        logits = outputs.logits[:, :-1, :].contiguous()
        labels = readout_inputs["labels"][:, 1:].contiguous()
        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
            ignore_index=IGNORE_INDEX,
        )
    token_count = (readout_inputs["labels"] != IGNORE_INDEX).sum()
    log_likelihood = -loss * token_count.to(device=loss.device, dtype=loss.dtype)
    return loss, log_likelihood


def _accumulate_fvt_grad_from_scalar(
    grad_buffers: list[torch.Tensor],
    index: int,
    scalar: torch.Tensor,
    fvt_output: torch.Tensor,
) -> None:
    grad = torch.autograd.grad(scalar, fvt_output, allow_unused=True)[0]
    if grad is not None:
        grad_buffers[index].add_(grad.detach())


@torch.no_grad()
def generate_readout_from_fvt(
    *,
    model: Any,
    tokenizer_or_processor: Any,
    target: str | None,
    foveated_visual_tokens: torch.Tensor,
    image_grid_thw: torch.Tensor | None = None,
    max_new_tokens: int = 64,
    eos_token_id: int | None = None,
    device: torch.device | str | None = None,
) -> str:
    """Greedy fresh-context readout for training/debug only.

    This does not participate in inference continuation from the original cache.
    """

    tokenizer = _tokenizer(tokenizer_or_processor)
    prefix = prepare_readout_prefix_inputs(
        model=model,
        tokenizer_or_processor=tokenizer,
        target=target,
        foveated_visual_tokens=foveated_visual_tokens,
        image_grid_thw=image_grid_thw,
        device=device,
    )
    attention_mask = prefix["attention_mask"]
    outputs = model(
        input_ids=prefix["input_ids"],
        inputs_embeds=prefix["inputs_embeds"],
        attention_mask=attention_mask,
        position_ids=prefix.get("position_ids"),
        image_grid_thw=prefix.get("image_grid_thw"),
        mm_token_type_ids=prefix.get("mm_token_type_ids"),
        use_cache=True,
        return_dict=True,
    )
    rope_delta = _rope_delta_for_next_token(prefix.get("position_ids"), attention_mask)
    model_kwargs = {}
    if rope_delta is not None:
        model_kwargs["rope_deltas"] = rope_delta
    state = BracketedFVTAppendResult(
        past_key_values=getattr(outputs, "past_key_values", None),
        attention_mask=attention_mask,
        cache_position=None,
        input_ids=prefix["input_ids"],
        last_logits=outputs.logits,
        appended_token_ids=prefix["input_ids"].detach().cpu().view(-1),
        appended_inputs_embeds=prefix["inputs_embeds"].detach().cpu(),
        fvt_token_start=int(prefix["fvt_token_start"]),
        fvt_token_end=int(prefix["fvt_token_end"]),
        model_kwargs=model_kwargs,
        debug_metadata={
            "readout_append_mode": prefix.get("readout_append_mode"),
            "position_ids_source": prefix.get("position_ids_source"),
            "fvt_grid_source": prefix.get("fvt_grid_source"),
        },
    )
    continuation = continue_generation_from_state(
        model=model,
        tokenizer_or_processor=tokenizer,
        generation_state=state,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
    )
    return continuation.generated_text


def visual_token_manifold_loss(
    foveated_visual_tokens: torch.Tensor,
    merged_visual_tokens: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    if foveated_visual_tokens.numel() == 0 or merged_visual_tokens.numel() == 0:
        return foveated_visual_tokens.new_zeros(())
    d = foveated_visual_tokens.float()
    v = merged_visual_tokens.detach().float()
    mean_loss = F.mse_loss(d.mean(dim=0), v.mean(dim=0))
    std_d = d.std(dim=0, unbiased=False).clamp_min(eps)
    std_v = v.std(dim=0, unbiased=False).clamp_min(eps)
    return mean_loss + F.mse_loss(std_d, std_v)


def contrastive_alignment_loss(
    foveated_visual_tokens: list[torch.Tensor] | torch.Tensor,
    text_embeddings: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    if isinstance(foveated_visual_tokens, torch.Tensor):
        pooled_d = foveated_visual_tokens.mean(dim=1)
    else:
        if len(foveated_visual_tokens) < 2:
            device = text_embeddings.device if text_embeddings.numel() else torch.device("cpu")
            return torch.zeros((), device=device)
        pooled_d = torch.stack([tokens.mean(dim=0) for tokens in foveated_visual_tokens], dim=0)
    if pooled_d.shape[0] < 2:
        return pooled_d.new_zeros(())
    z_d = F.normalize(pooled_d.float(), dim=-1)
    z_text = F.normalize(text_embeddings.float(), dim=-1)
    logits = z_d @ z_text.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


def attention_diagnostics(
    attention_debug: dict[str, torch.Tensor],
    *,
    topk: int = 5,
    eps: float = 1e-12,
) -> dict[str, float | list[float]]:
    attention = _fvt_slot_attention(attention_debug)
    if attention is None or attention.numel() == 0:
        return {}
    weights = attention.detach().float()
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(eps)
    entropy = -(weights.clamp_min(eps) * weights.clamp_min(eps).log()).sum(dim=-1)
    sorted_weights = weights.sort(dim=-1, descending=True).values
    k = min(topk, sorted_weights.shape[-1])
    top1 = sorted_weights[:, 0]
    topk_mass = sorted_weights[:, :k].sum(dim=-1)
    topk_indices = weights.topk(k=k, dim=-1).indices
    coverage = topk_indices.unique().numel() / max(weights.shape[-1], 1)
    return {
        "attn_entropy_values": _tensor_values(entropy),
        "top1_attn_mass_values": _tensor_values(top1),
        f"top{k}_attn_mass_values": _tensor_values(topk_mass),
        f"top{k}_visual_token_coverage": float(coverage),
    }


def fvt_norm_diagnostics(
    *,
    foveated_visual_tokens: torch.Tensor,
    merged_visual_tokens: torch.Tensor,
    eps: float = 1e-12,
) -> dict[str, float | list[float]]:
    if foveated_visual_tokens.numel() == 0 or merged_visual_tokens.numel() == 0:
        return {}
    d_norm = foveated_visual_tokens.detach().float().norm(dim=-1)
    v_norm = merged_visual_tokens.detach().float().norm(dim=-1)
    v_mean = v_norm.mean().clamp_min(eps)
    ratio = d_norm / v_mean
    return {
        "d_norm_values": _tensor_values(d_norm),
        "v_merge_norm_values": _tensor_values(v_norm),
        "norm_ratio_values": _tensor_values(ratio),
    }


def summarize_diagnostics(
    diagnostics: list[dict[str, float | list[float]]],
) -> dict[str, float | list[float]]:
    merged: dict[str, list[float]] = {}
    scalars: dict[str, list[float]] = {}
    for item in diagnostics:
        for key, value in item.items():
            if isinstance(value, list):
                merged.setdefault(key, []).extend(float(entry) for entry in value)
            elif isinstance(value, (int, float)):
                scalars.setdefault(key, []).append(float(value))
    summary: dict[str, float | list[float]] = {}
    for key, values in merged.items():
        if not values:
            continue
        tensor = torch.tensor(values, dtype=torch.float32)
        base = key.removesuffix("_values")
        summary[key] = values
        summary[f"{base}_mean"] = float(tensor.mean().item())
        summary[f"{base}_min"] = float(tensor.min().item())
        summary[f"{base}_max"] = float(tensor.max().item())
    for key, values in scalars.items():
        if values:
            summary[key] = float(torch.tensor(values, dtype=torch.float32).mean().item())
    return summary


def same_image_negative_loss(
    *,
    positive_log_likelihoods: torch.Tensor,
    negative_log_likelihoods: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    if positive_log_likelihoods.numel() == 0 or negative_log_likelihoods.numel() == 0:
        device = (
            positive_log_likelihoods.device
            if positive_log_likelihoods.numel()
            else negative_log_likelihoods.device
        )
        return torch.zeros((), device=device)
    return F.relu(margin - positive_log_likelihoods + negative_log_likelihoods).mean()


def same_image_negative_matrix_ce_loss(score_matrices: list[torch.Tensor]) -> torch.Tensor:
    if not score_matrices:
        return torch.zeros(())
    total_loss = score_matrices[0].new_zeros(())
    total_rows = 0
    for scores in score_matrices:
        if scores.numel() == 0:
            continue
        labels = torch.arange(scores.shape[0], device=scores.device)
        total_loss = total_loss + F.cross_entropy(scores, labels, reduction="sum")
        total_rows += scores.shape[0]
    if total_rows == 0:
        return score_matrices[0].new_zeros(())
    return total_loss / total_rows


def same_image_negative_matrix_ce_score_gradients(
    score_matrices: list[torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Return matrix-CE value and d(loss)/d(scores) without keeping score graphs."""

    if not score_matrices:
        return torch.zeros(()), []
    total_rows = sum(int(scores.shape[0]) for scores in score_matrices if scores.numel())
    if total_rows == 0:
        return score_matrices[0].new_zeros(()), [
            torch.zeros_like(scores) for scores in score_matrices
        ]

    total_loss = score_matrices[0].new_zeros(())
    gradients: list[torch.Tensor] = []
    for scores in score_matrices:
        if scores.numel() == 0:
            gradients.append(torch.zeros_like(scores))
            continue
        labels = torch.arange(scores.shape[0], device=scores.device)
        total_loss = total_loss + F.cross_entropy(scores, labels, reduction="sum")
        probs = torch.softmax(scores.detach().float(), dim=-1).to(dtype=scores.dtype)
        grads = probs
        grads[torch.arange(scores.shape[0], device=scores.device), labels] -= 1
        gradients.append(grads / total_rows)
    return total_loss / total_rows, gradients


def same_image_negative_groups(samples: list[TeacherGuideSample]) -> list[list[int]]:
    by_image: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        key = sample.image_id or sample.image
        if key:
            by_image.setdefault(key, []).append(index)
    return [indices for indices in by_image.values() if len(indices) >= 2]


def same_image_negative_pairs(samples: list[TeacherGuideSample]) -> list[tuple[int, int]]:
    pairs = []
    for indices in same_image_negative_groups(samples):
        for offset, index in enumerate(indices):
            pairs.append((index, indices[(offset + 1) % len(indices)]))
    return pairs


@torch.no_grad()
def encode_evidence_text_with_qwen(
    *,
    model: Any,
    tokenizer_or_processor: Any,
    texts: list[str],
    device: torch.device | str,
) -> torch.Tensor:
    tokenizer = _tokenizer(tokenizer_or_processor)
    encoded = tokenizer(texts, padding=True, return_tensors="pt")
    encoded = {
        key: value.to(device) if hasattr(value, "to") else value for key, value in encoded.items()
    }
    outputs = model(**encoded, output_hidden_states=True, return_dict=True)
    hidden = outputs.hidden_states[-1]
    mask = encoded.get("attention_mask", torch.ones(hidden.shape[:2], device=hidden.device))
    pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
    return pooled.detach()


def _image_grid_thw_for_fvt_output(
    fvt_output: Any,
    feature: TGVFTrainingFeatures,
) -> torch.Tensor | None:
    if fvt_output.debug_metadata.get("tgvf_version") != "v2":
        return None
    if feature.image_grid_thw is None:
        raise RuntimeError("TGVF v2 requires source image_grid_thw for image-position readout")
    return feature.image_grid_thw


def training_step(
    *,
    qwen_model: Any,
    processor: Any,
    foveal_module: nn.Module,
    samples: list[TeacherGuideSample],
    loss_weights: LossWeights,
    device: torch.device | str,
    hidden_state_index: int = -1,
    contrastive_temperature: float = 0.07,
    same_image_negative_margin: float = 1.0,
    same_image_negative_mode: str = "cyclic_margin",
    readout_prompt_target_dropout: float = 0.3,
    backward_loss_scale: float | None = None,
) -> TGVFTrainStepOutput:
    if not 0.0 <= readout_prompt_target_dropout <= 1.0:
        raise ValueError("readout_prompt_target_dropout must be in [0, 1]")
    if backward_loss_scale is not None and backward_loss_scale <= 0:
        raise ValueError("backward_loss_scale must be positive when provided")
    streaming_backward = backward_loss_scale is not None
    if streaming_backward and same_image_negative_mode not in {"matrix_ce", "cyclic_margin"}:
        raise ValueError(
            "streaming backward is currently supported only for matrix_ce and cyclic_margin"
        )

    features = [
        collect_training_features(
            model=qwen_model,
            processor=processor,
            sample=sample,
            device=device,
            hidden_state_index=hidden_state_index,
        )
        for sample in samples
    ]
    fvt_outputs = []
    fvt_image_grid_thws: list[torch.Tensor | None] = []
    fvt_grad_buffers: list[torch.Tensor] = []
    loss_gen_values = []
    loss_man_values = []
    positive_ll = []
    attention_debug_values = []
    norm_debug_values = []
    readout_targets: list[str | None] = []

    for sample_index, (sample, feature) in enumerate(zip(samples, features, strict=True)):
        drop_target = (
            readout_prompt_target_dropout > 0.0
            and float(torch.rand((), device=device).item()) < readout_prompt_target_dropout
        )
        readout_targets.append(None if drop_target else sample.target)
        output = foveal_module(
            target_hidden_states=feature.target_hidden_states.to(device),
            pre_merge_visual_tokens=feature.pre_merge_visual_tokens.to(device),
            metadata={"target": sample.target},
        )
        output = finalize_tgvf_output_with_frozen_qwen_merger(qwen_model, output)
        d = output.foveated_visual_tokens
        fvt_outputs.append(d)
        if streaming_backward:
            fvt_grad_buffers.append(torch.zeros_like(d))
        fvt_image_grid_thw = _image_grid_thw_for_fvt_output(output, feature)
        fvt_image_grid_thws.append(fvt_image_grid_thw)
        attention_debug_values.append(attention_diagnostics(output.attention_debug))
        norm_debug_values.append(
            fvt_norm_diagnostics(
                foveated_visual_tokens=d,
                merged_visual_tokens=feature.merged_visual_tokens.to(device),
            )
        )
        readout_inputs = prepare_readout_inputs(
            model=qwen_model,
            tokenizer_or_processor=processor,
            target=readout_targets[sample_index],
            evidence_description=sample.evidence_description,
            foveated_visual_tokens=d,
            image_grid_thw=fvt_image_grid_thw,
            device=device,
        )
        gen_loss, log_likelihood = compute_readout_lm_loss(
            model=qwen_model,
            readout_inputs=readout_inputs,
        )
        if streaming_backward:
            loss_gen_values.append(gen_loss.detach())
            positive_ll.append(log_likelihood.detach())
            if loss_weights.gen:
                _accumulate_fvt_grad_from_scalar(
                    fvt_grad_buffers,
                    sample_index,
                    gen_loss
                    * (float(loss_weights.gen) * float(backward_loss_scale) / max(len(samples), 1)),
                    d,
                )
        else:
            loss_gen_values.append(gen_loss)
            positive_ll.append(log_likelihood)
        loss_man_values.append(
            visual_token_manifold_loss(d, feature.merged_visual_tokens.to(device))
        )

    loss_gen = torch.stack(loss_gen_values).mean()
    loss_man = torch.stack(loss_man_values).mean()
    zero = loss_gen.new_zeros(())
    loss_same = zero
    if loss_weights.same_image_negative:
        if same_image_negative_mode == "cyclic_margin" and streaming_backward:
            pairs = same_image_negative_pairs(samples)
            margin_values = []
            with torch.no_grad():
                for pos_index, neg_index in pairs:
                    readout_inputs = prepare_readout_inputs(
                        model=qwen_model,
                        tokenizer_or_processor=processor,
                        target=readout_targets[pos_index],
                        evidence_description=samples[pos_index].evidence_description,
                        foveated_visual_tokens=fvt_outputs[neg_index].detach(),
                        image_grid_thw=fvt_image_grid_thws[neg_index],
                        device=device,
                    )
                    _, log_likelihood = compute_readout_lm_loss(
                        model=qwen_model,
                        readout_inputs=readout_inputs,
                    )
                    margin_values.append(
                        same_image_negative_margin
                        - positive_ll[pos_index]
                        + log_likelihood.detach()
                    )
            if margin_values:
                margins = torch.stack(margin_values)
                loss_same = F.relu(margins).mean()
                coeff = (
                    float(loss_weights.same_image_negative)
                    * float(backward_loss_scale)
                    / float(len(pairs))
                )
                if coeff:
                    for (pos_index, neg_index), margin_value in zip(
                        pairs, margin_values, strict=True
                    ):
                        if float(margin_value.detach().cpu()) <= 0.0:
                            continue
                        readout_inputs = prepare_readout_inputs(
                            model=qwen_model,
                            tokenizer_or_processor=processor,
                            target=readout_targets[pos_index],
                            evidence_description=samples[pos_index].evidence_description,
                            foveated_visual_tokens=fvt_outputs[pos_index],
                            image_grid_thw=fvt_image_grid_thws[pos_index],
                            device=device,
                        )
                        _, positive_log_likelihood = compute_readout_lm_loss(
                            model=qwen_model,
                            readout_inputs=readout_inputs,
                        )
                        _accumulate_fvt_grad_from_scalar(
                            fvt_grad_buffers,
                            pos_index,
                            positive_log_likelihood * (-coeff),
                            fvt_outputs[pos_index],
                        )
                        readout_inputs = prepare_readout_inputs(
                            model=qwen_model,
                            tokenizer_or_processor=processor,
                            target=readout_targets[pos_index],
                            evidence_description=samples[pos_index].evidence_description,
                            foveated_visual_tokens=fvt_outputs[neg_index],
                            image_grid_thw=fvt_image_grid_thws[neg_index],
                            device=device,
                        )
                        _, negative_log_likelihood = compute_readout_lm_loss(
                            model=qwen_model,
                            readout_inputs=readout_inputs,
                        )
                        _accumulate_fvt_grad_from_scalar(
                            fvt_grad_buffers,
                            neg_index,
                            negative_log_likelihood * coeff,
                            fvt_outputs[neg_index],
                        )
        elif same_image_negative_mode == "cyclic_margin":
            pairs = same_image_negative_pairs(samples)
            negative_ll = []
            positive_selected = []
            for pos_index, neg_index in pairs:
                readout_inputs = prepare_readout_inputs(
                    model=qwen_model,
                    tokenizer_or_processor=processor,
                    target=readout_targets[pos_index],
                    evidence_description=samples[pos_index].evidence_description,
                    foveated_visual_tokens=fvt_outputs[neg_index],
                    image_grid_thw=fvt_image_grid_thws[neg_index],
                    device=device,
                )
                _, log_likelihood = compute_readout_lm_loss(
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
        elif same_image_negative_mode == "matrix_ce" and streaming_backward:
            score_matrices = []
            score_group_indices = []
            with torch.no_grad():
                for indices in same_image_negative_groups(samples):
                    rows = []
                    for pos_index in indices:
                        row = []
                        for fvt_index in indices:
                            if pos_index == fvt_index:
                                row.append(positive_ll[pos_index])
                                continue
                            readout_inputs = prepare_readout_inputs(
                                model=qwen_model,
                                tokenizer_or_processor=processor,
                                target=readout_targets[pos_index],
                                evidence_description=samples[pos_index].evidence_description,
                                foveated_visual_tokens=fvt_outputs[fvt_index].detach(),
                                image_grid_thw=fvt_image_grid_thws[fvt_index],
                                device=device,
                            )
                            _, log_likelihood = compute_readout_lm_loss(
                                model=qwen_model,
                                readout_inputs=readout_inputs,
                            )
                            row.append(log_likelihood.detach())
                        rows.append(torch.stack(row))
                    score_matrices.append(torch.stack(rows))
                    score_group_indices.append(indices)
            if score_matrices:
                loss_same, score_grad_matrices = same_image_negative_matrix_ce_score_gradients(
                    score_matrices
                )
                coeff_scale = float(loss_weights.same_image_negative) * float(backward_loss_scale)
                if coeff_scale:
                    for indices, score_grads in zip(
                        score_group_indices, score_grad_matrices, strict=True
                    ):
                        for row_offset, pos_index in enumerate(indices):
                            for col_offset, fvt_index in enumerate(indices):
                                coeff = score_grads[row_offset, col_offset].detach() * coeff_scale
                                if float(coeff.detach().cpu()) == 0.0:
                                    continue
                                readout_inputs = prepare_readout_inputs(
                                    model=qwen_model,
                                    tokenizer_or_processor=processor,
                                    target=readout_targets[pos_index],
                                    evidence_description=samples[pos_index].evidence_description,
                                    foveated_visual_tokens=fvt_outputs[fvt_index],
                                    image_grid_thw=fvt_image_grid_thws[fvt_index],
                                    device=device,
                                )
                                _, log_likelihood = compute_readout_lm_loss(
                                    model=qwen_model,
                                    readout_inputs=readout_inputs,
                                )
                                _accumulate_fvt_grad_from_scalar(
                                    fvt_grad_buffers,
                                    fvt_index,
                                    log_likelihood.to(dtype=coeff.dtype) * coeff,
                                    fvt_outputs[fvt_index],
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
                        readout_inputs = prepare_readout_inputs(
                            model=qwen_model,
                            tokenizer_or_processor=processor,
                            target=readout_targets[pos_index],
                            evidence_description=samples[pos_index].evidence_description,
                            foveated_visual_tokens=fvt_outputs[fvt_index],
                            image_grid_thw=fvt_image_grid_thws[fvt_index],
                            device=device,
                        )
                        _, log_likelihood = compute_readout_lm_loss(
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
    if loss_weights.contrastive_alignment:
        text_embeddings = encode_evidence_text_with_qwen(
            model=qwen_model,
            tokenizer_or_processor=processor,
            texts=[sample.evidence_description for sample in samples],
            device=device,
        )
        loss_contrastive = contrastive_alignment_loss(
            fvt_outputs,
            text_embeddings,
            temperature=contrastive_temperature,
        )

    loss_total = (
        loss_weights.gen * loss_gen
        + loss_weights.visual_token_manifold * loss_man
        + loss_weights.same_image_negative * loss_same
        + loss_weights.contrastive_alignment * loss_contrastive
    )
    if streaming_backward:
        scalar_backward_losses = []
        if loss_weights.visual_token_manifold:
            scalar_backward_losses.append(
                loss_man * (float(loss_weights.visual_token_manifold) * float(backward_loss_scale))
            )
        if loss_weights.contrastive_alignment:
            scalar_backward_losses.append(
                loss_contrastive
                * (float(loss_weights.contrastive_alignment) * float(backward_loss_scale))
            )
        torch.autograd.backward(
            [*fvt_outputs, *scalar_backward_losses],
            grad_tensors=[*fvt_grad_buffers, *([None] * len(scalar_backward_losses))],
        )
        loss_total = loss_total.detach()
        loss_gen = loss_gen.detach()
        loss_man = loss_man.detach()
        loss_same = loss_same.detach()
        loss_contrastive = loss_contrastive.detach()
    first_feature = features[0]
    first_d = fvt_outputs[0]
    return TGVFTrainStepOutput(
        loss_total=loss_total,
        loss_gen=loss_gen,
        loss_visual_token_manifold=loss_man,
        loss_same_image_negative=loss_same,
        loss_contrastive_alignment=loss_contrastive,
        debug={
            "target_hidden_shape": list(first_feature.target_hidden_states.shape),
            "pre_merge_visual_shape": list(first_feature.pre_merge_visual_tokens.shape),
            "merged_visual_shape": list(first_feature.merged_visual_tokens.shape),
            "foveated_visual_tokens_shape": list(first_d.shape),
            "loss_weights": asdict(loss_weights),
            "same_image_negative_mode": same_image_negative_mode,
            "readout_prompt_target_dropout": readout_prompt_target_dropout,
            "readout_append_mode": "qwen_native_pseudo_image",
            "readout_fake_image_grid_thw": readout_inputs.get("fake_image_grid_thw"),
            "readout_source_image_grid_thw": readout_inputs.get("source_image_grid_thw"),
            "readout_fvt_grid_thw": readout_inputs.get("fvt_grid_thw"),
            "readout_fvt_grid_source": readout_inputs.get("fvt_grid_source"),
            "readout_spatial_merge_size": readout_inputs.get("spatial_merge_size"),
            "readout_expected_llm_image_tokens": readout_inputs.get("expected_llm_image_tokens"),
            "readout_actual_image_pad_token_count": readout_inputs.get(
                "actual_image_pad_token_count"
            ),
            "readout_mm_token_type_ids_present": readout_inputs.get("mm_token_type_ids_present"),
            "readout_image_pad_mm_type_is_image": readout_inputs.get("image_pad_mm_type_is_image"),
            "readout_position_ids_source": readout_inputs.get("position_ids_source"),
            "readout_image_position_ids_are_3d": readout_inputs.get("image_position_ids_are_3d"),
            "readout_text_position_ids_are_1d": readout_inputs.get("text_position_ids_are_1d"),
            "readout_visual_tower_called_for_fvt": readout_inputs.get(
                "visual_tower_called_for_fvt"
            ),
            "readout_prompt_target_dropout_observed": (
                sum(target is None for target in readout_targets) / max(len(readout_targets), 1)
            ),
            "attention_diagnostics": summarize_diagnostics(attention_debug_values),
            "norm_diagnostics": summarize_diagnostics(norm_debug_values),
            "qwen_frozen": not any(
                parameter.requires_grad for parameter in qwen_model.parameters()
            ),
            "second_full_forward_used": False,
            "streaming_matrix_ce_backward": bool(
                streaming_backward and same_image_negative_mode == "matrix_ce"
            ),
            "streaming_cyclic_margin_backward": bool(
                streaming_backward and same_image_negative_mode == "cyclic_margin"
            ),
            "backward_performed": bool(streaming_backward),
        },
    )


def save_tgvf_checkpoint(
    *,
    path: str | Path,
    foveal_module: nn.Module,
    config: TGVFTrainingConfig | dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    global_step: int = 0,
    optimizer_step: int = 0,
) -> None:
    checkpoint: dict[str, Any] = {
        "tgvf_module": foveal_module.state_dict(),
        "config": asdict(config) if hasattr(config, "__dataclass_fields__") else config,
        "global_step": global_step,
        "optimizer_step": optimizer_step,
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    checkpoint["scheduler"] = scheduler.state_dict() if scheduler is not None else None
    torch.save(checkpoint, path)


def load_tgvf_module_checkpoint(
    module: nn.Module, path: str | Path, *, strict: bool = True
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    module.load_state_dict(checkpoint["tgvf_module"], strict=strict)
    return checkpoint


def teacher_guide_collate(samples: list[TeacherGuideSample]) -> list[TeacherGuideSample]:
    return samples


def _encode(tokenizer: Any, text: str) -> list[int]:
    return [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)]


def _terms(text: str) -> list[str]:
    normalized = text.replace("/", " ").replace("-", " ")
    return [term.lower() for term in normalized.split() if len(term) > 2]


def _tokenizer(tokenizer_or_processor: Any) -> Any:
    return getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)


def _visual_module(model: Any) -> Any:
    if hasattr(model, "visual"):
        return model.visual
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        return model.model.visual
    raise AttributeError("Could not find a Qwen2-VL visual module")


def _fvt_slot_attention(attention_debug: dict[str, torch.Tensor]) -> torch.Tensor | None:
    sub_slot_attention = attention_debug.get("sub_slot_attention_weights")
    if isinstance(sub_slot_attention, torch.Tensor):
        if sub_slot_attention.ndim == 3:
            return sub_slot_attention.mean(dim=1)
        if sub_slot_attention.ndim == 2:
            return sub_slot_attention
    attention = attention_debug.get("attention_weights")
    if isinstance(attention, torch.Tensor) and attention.ndim == 2:
        return attention
    return None


def _tensor_values(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().flatten().tolist()]


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
