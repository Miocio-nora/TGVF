from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


EVIDENCE_STATE_START = "<EVIDENCE_STATE>"
EVIDENCE_STATE_END = "</EVIDENCE_STATE>"
FOCUS_START = "<FOCUS>"
FOCUS_END = "</FOCUS>"
TGVF_START = "<TGVF>"
TGVF_END = "</TGVF>"
EVIDENCE_START = "<EVIDENCE>"
EVIDENCE_END = "</EVIDENCE>"
ANSWER_START = "<ANSWER>"
ANSWER_END = "</ANSWER>"

NEED_LOCAL_EVIDENCE = "need_local_visual_evidence"
SUFFICIENT_EVIDENCE = "sufficient_visual_evidence"


GENERIC_TARGETS = {
    "the image",
    "the scene",
    "the object",
    "the answer",
    "something",
    "relevant visual evidence",
    "the relevant visual evidence",
    "visual target description",
    "specific local visual target",
}


@dataclass
class LoadedQwen3VL:
    model: Qwen3VLForConditionalGeneration
    processor: Any
    model_id: str
    processor_id: str
    dtype: str
    device_map: str | dict[str, Any] | None
    attn_implementation: str | None


@dataclass
class V3ActionParse:
    raw_text: str
    evidence_state: str | None = None
    focus_target: str = ""
    answer: str = ""
    has_focus_open: bool = False
    has_focus_close: bool = False
    focus_valid: bool = False
    answer_valid: bool = False
    malformed: bool = False
    malformed_reasons: list[str] = field(default_factory=list)


@dataclass
class Qwen3SourceVisualGeometry:
    image_grid_thw: torch.Tensor | None
    video_grid_thw: torch.Tensor | None
    source_visual_position_ids: torch.Tensor | None
    source_visual_token_indices: torch.Tensor | None
    source_visual_token_count: int
    image_token_id: int | None
    position_ids_shape: list[int] | None
    mm_token_type_ids_present: bool
    extraction_mode: str
    errors: list[str] = field(default_factory=list)


@dataclass
class Qwen3FocusCapture:
    target_text: str
    target_token_ids: list[int]
    target_hidden_states: torch.Tensor
    generated_ids: list[int]
    generated_text: str
    generated_hidden_states: torch.Tensor
    past_key_values: Any
    attention_mask: torch.Tensor | None
    cache_position: torch.Tensor | None
    input_ids: torch.Tensor | None
    last_logits: torch.Tensor | None
    model_kwargs: dict[str, Any]
    image_grid_thw: torch.Tensor | None = None
    video_grid_thw: torch.Tensor | None = None
    source_visual_geometry: Qwen3SourceVisualGeometry | None = None
    target_token_start: int | None = None
    target_token_end: int | None = None
    stop_reason: str = "max_new_tokens"
    capture_found: bool = False
    second_full_forward_used: bool = False
    malformed: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass
class Qwen3VisionTap:
    image_grid_thw: list[list[int]] | None
    video_grid_thw: list[list[int]] | None
    v_pre_shape: list[int] | None
    v_merge_shape: list[int] | None
    deepstack_feature_count: int
    deepstack_feature_shapes: list[list[int]]
    vision_dim: int | None
    llm_hidden_dim: int | None
    image_token_id: int | None
    video_token_id: int | None
    vision_start_token_id: int | None
    vision_end_token_id: int | None
    spatial_merge_size: int | None
    merge_size: int | None
    patch_size: int | None
    temporal_patch_size: int | None
    output_type: str
    errors: list[str] = field(default_factory=list)


@dataclass
class Qwen3AppendResult:
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
class Qwen3Continuation:
    generated_ids: list[int]
    generated_text: str
    past_key_values: Any
    attention_mask: torch.Tensor | None
    input_ids: torch.Tensor | None
    last_logits: torch.Tensor | None
    stop_reason: str


def load_qwen3_vl(
    model_id: str,
    *,
    processor_id: str | None = None,
    dtype: str = "bfloat16",
    device_map: str | dict[str, Any] | None = "auto",
    attn_implementation: str | None = "sdpa",
    trust_remote_code: bool = True,
) -> LoadedQwen3VL:
    processor_name = processor_id or model_id
    model_kwargs: dict[str, Any] = {
        "torch_dtype": _resolve_dtype(dtype),
        "trust_remote_code": trust_remote_code,
    }
    if device_map:
        model_kwargs["device_map"] = device_map
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained(processor_name, trust_remote_code=trust_remote_code)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return LoadedQwen3VL(
        model=model,
        processor=processor,
        model_id=model_id,
        processor_id=processor_name,
        dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )


def build_direct_messages(image: Any, question: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                *_vision_content_items(image),
                {"type": "text", "text": question},
            ],
        }
    ]


def build_focus_force_prompt(question: str) -> str:
    return (
        "We are building a Target-Guided Visual Foveation system.\n"
        "The system first asks the language model what local visual evidence is needed, "
        "then a separate visual module will inspect that target.\n\n"
        "Your task is NOT to answer the question yet.\n"
        "Your task is to choose one specific local visual focus target in the image.\n"
        "The focus target must be a neutral visual pointer: local, visually locatable, "
        "and not leaking the answer value.\n\n"
        "If you use a <think>...</think> section, that is allowed.\n"
        "After the thinking section, your final visible action must still be exactly "
        "the format below.\n\n"
        f"Question: {question}\n\n"
        "Output exactly the following format and stop:\n\n"
        f"{EVIDENCE_STATE_START}{NEED_LOCAL_EVIDENCE}{EVIDENCE_STATE_END}\n"
        f"{FOCUS_START}your specific local visual target here{FOCUS_END}\n\n"
        "Rules for the focus target:\n"
        "- Put only the target text inside <FOCUS> and </FOCUS>.\n"
        "- Do not include the answer inside the focus target.\n"
        "- Do not use generic targets like the image, the scene, the object, or the answer.\n"
        "- Use a short noun phrase such as the small text below the barcode, "
        "the number inside the blue circle, or the label above the tallest bar.\n"
        "- Do not explain.\n"
        "- Do not output <ANSWER>."
    )


def build_focus_force_messages(image: Any, question: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                *_vision_content_items(image),
                {"type": "text", "text": build_focus_force_prompt(question)},
            ],
        }
    ]


def build_free_router_prompt(question: str) -> str:
    return (
        "We are building a Target-Guided Visual Foveation system.\n"
        "Decide whether the question can be answered directly from the current image view, "
        "or whether a local visual focus target is needed before answering.\n\n"
        f"Question: {question}\n\n"
        "Either answer directly:\n"
        f"{EVIDENCE_STATE_START}{SUFFICIENT_EVIDENCE}{EVIDENCE_STATE_END}\n"
        f"{ANSWER_START}...{ANSWER_END}\n\n"
        "or request focused visual evidence:\n"
        f"{EVIDENCE_STATE_START}{NEED_LOCAL_EVIDENCE}{EVIDENCE_STATE_END}\n"
        f"{FOCUS_START}your specific local visual target here{FOCUS_END}\n\n"
        "If you use a <think>...</think> section, put the compact protocol after it.\n"
        "If focusing, put only a neutral, local, visually locatable target inside "
        "<FOCUS> and </FOCUS>.\n"
        "Do not leak the answer in the focus target.\n"
        "Do not use long chain-of-thought. Use only the compact protocol."
    )


def build_free_router_messages(image: Any, question: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                *_vision_content_items(image),
                {"type": "text", "text": build_free_router_prompt(question)},
            ],
        }
    ]


def build_qwen3_inputs(processor: Any, messages: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if isinstance(inputs, dict) and "input_ids" in inputs:
            return dict(inputs)
    except TypeError:
        pass

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    return dict(
        processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    )


@torch.no_grad()
def generate_direct_qwen3(
    model: Any,
    processor: Any,
    *,
    image: Any,
    question: str,
    max_new_tokens: int = 64,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    messages = build_direct_messages(image, question)
    inputs = build_qwen3_inputs(processor, messages)
    tokenizer = processor.tokenizer
    model_inputs = _move_tensors(inputs, device or _infer_model_device(model))
    prompt_len = int(model_inputs["input_ids"].shape[-1])
    start = time.perf_counter()
    generated = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    wall = time.perf_counter() - start
    new_ids = generated[0, prompt_len:].detach().cpu().tolist()
    text = _decode(tokenizer, new_ids)
    parsed = parse_v3_action(text)
    return {
        "raw_output": text,
        "generated_ids": new_ids,
        "parsed_answer": parsed.answer if parsed.answer_valid else text.strip(),
        "wall_time_sec": wall,
        "output_tokens": len(new_ids),
    }


@torch.no_grad()
def capture_focus_single_pass_qwen3(
    model: Any,
    processor: Any,
    *,
    image: Any,
    question: str,
    messages: list[dict[str, Any]] | None = None,
    max_new_tokens: int = 96,
    device: torch.device | str | None = None,
    hidden_state_index: int = -1,
    eos_token_id: int | None = None,
    force_action_prefix: bool = False,
    scripted_target_text: str | None = None,
) -> Qwen3FocusCapture:
    messages = messages or build_focus_force_messages(image, question)
    inputs = build_qwen3_inputs(processor, messages)
    return capture_focus_single_pass_from_inputs_qwen3(
        model,
        processor.tokenizer,
        inputs,
        max_new_tokens=max_new_tokens,
        device=device,
        hidden_state_index=hidden_state_index,
        eos_token_id=eos_token_id,
        forced_prefix_text=(
            f"{EVIDENCE_STATE_START}{NEED_LOCAL_EVIDENCE}{EVIDENCE_STATE_END}\n"
            f"{FOCUS_START} {scripted_target_text.strip()} {FOCUS_END}"
            if scripted_target_text
            else
            f"{EVIDENCE_STATE_START}{NEED_LOCAL_EVIDENCE}{EVIDENCE_STATE_END}\n{FOCUS_START} "
            if force_action_prefix
            else None
        ),
    )


@torch.no_grad()
def capture_focus_single_pass_from_inputs_qwen3(
    model: Any,
    tokenizer: Any,
    inputs: dict[str, Any],
    *,
    max_new_tokens: int = 96,
    device: torch.device | str | None = None,
    hidden_state_index: int = -1,
    eos_token_id: int | None = None,
    forced_prefix_text: str | None = None,
) -> Qwen3FocusCapture:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    focus_start_ids = _marker_ids(tokenizer, FOCUS_START)
    focus_end_ids = _marker_ids(tokenizer, FOCUS_END)
    if not focus_start_ids or not focus_end_ids:
        raise ValueError("FOCUS markers must tokenize to non-empty id sequences")

    if device is None:
        device = _infer_model_device(model)
    model_inputs = _move_tensors(dict(inputs), device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs.get("attention_mask")
    full_input_ids = input_ids
    source_visual_geometry = extract_qwen3_source_visual_geometry(model, model_inputs)

    outputs = model(
        **model_inputs,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    past_key_values = outputs.past_key_values
    logits = outputs.logits
    if getattr(outputs, "rope_deltas", None) is not None:
        model_inputs["rope_deltas"] = outputs.rope_deltas

    generated_ids: list[int] = []
    generated_hidden_states: list[torch.Tensor] = []
    cache_position = model_inputs.get("cache_position")

    if forced_prefix_text:
        forced_ids = _encode_text(tokenizer, forced_prefix_text, input_ids.device)
        for forced_token in forced_ids.tolist():
            next_token = torch.tensor([[int(forced_token)]], dtype=torch.long, device=input_ids.device)
            full_input_ids = torch.cat([full_input_ids.to(next_token.device), next_token], dim=-1)
            attention_mask = _append_attention(attention_mask, next_token)
            step_inputs = _prepare_decode_step(
                model,
                full_input_ids=full_input_ids,
                next_token=next_token,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                base_inputs=model_inputs,
                cache_position=cache_position,
                generated_token_count=len(generated_ids) + 1,
            )
            outputs = model(
                **step_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[hidden_state_index][0, -1].detach()
            generated_ids.append(int(forced_token))
            generated_hidden_states.append(hidden)
            past_key_values = outputs.past_key_values
            logits = outputs.logits
            cache_position = step_inputs.get("cache_position")
        completed = _find_completed_focus(generated_ids, focus_start_ids, focus_end_ids, tokenizer)
        if completed is not None:
            start, end, target_text = completed
            target_ids = generated_ids[start:end]
            target_hidden = _stack_hidden_states(generated_hidden_states[start:end])
            return Qwen3FocusCapture(
                target_text=target_text,
                target_token_ids=target_ids,
                target_hidden_states=target_hidden,
                generated_ids=list(generated_ids),
                generated_text=_decode(tokenizer, generated_ids),
                generated_hidden_states=_stack_hidden_states(generated_hidden_states),
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                cache_position=cache_position,
                input_ids=full_input_ids,
                last_logits=logits,
                model_kwargs={
                    **_resume_model_kwargs(model_inputs),
                    "focus_target_source": "scripted_for_smoke",
                },
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                source_visual_geometry=source_visual_geometry,
                target_token_start=start,
                target_token_end=end,
                stop_reason="forced_focus_end_marker",
                capture_found=True,
                second_full_forward_used=False,
                malformed=False,
                errors=[],
            )

    for _ in range(max_new_tokens):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        token_id = int(next_token[0, 0].detach().cpu().item())
        full_input_ids = torch.cat([full_input_ids.to(next_token.device), next_token], dim=-1)
        attention_mask = _append_attention(attention_mask, next_token)

        step_inputs = _prepare_decode_step(
            model,
            full_input_ids=full_input_ids,
            next_token=next_token,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            base_inputs=model_inputs,
            cache_position=cache_position,
            generated_token_count=len(generated_ids) + 1,
        )
        outputs = model(
            **step_inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[hidden_state_index][0, -1].detach()
        generated_ids.append(token_id)
        generated_hidden_states.append(hidden)
        past_key_values = outputs.past_key_values
        logits = outputs.logits
        cache_position = step_inputs.get("cache_position")

        completed = _find_completed_focus(generated_ids, focus_start_ids, focus_end_ids, tokenizer)
        if completed is not None:
            start, end, target_text = completed
            target_ids = generated_ids[start:end]
            target_hidden = _stack_hidden_states(generated_hidden_states[start:end])
            return Qwen3FocusCapture(
                target_text=target_text,
                target_token_ids=target_ids,
                target_hidden_states=target_hidden,
                generated_ids=list(generated_ids),
                generated_text=_decode(tokenizer, generated_ids),
                generated_hidden_states=_stack_hidden_states(generated_hidden_states),
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                cache_position=cache_position,
                input_ids=full_input_ids,
                last_logits=logits,
                model_kwargs=_resume_model_kwargs(model_inputs),
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                source_visual_geometry=source_visual_geometry,
                target_token_start=start,
                target_token_end=end,
                stop_reason="focus_end_marker",
                capture_found=True,
                second_full_forward_used=False,
                malformed=is_generic_target(target_text),
                errors=["generic_target"] if is_generic_target(target_text) else [],
            )

        if eos_token_id is not None and token_id == eos_token_id:
            break

    generated_text = _decode(tokenizer, generated_ids)
    parsed = parse_v3_action(generated_text)
    errors = list(parsed.malformed_reasons)
    if parsed.has_focus_open and not parsed.has_focus_close:
        errors.append("missing_closing_focus")
    return Qwen3FocusCapture(
        target_text="",
        target_token_ids=[],
        target_hidden_states=_empty_hidden_like(generated_hidden_states),
        generated_ids=list(generated_ids),
        generated_text=generated_text,
        generated_hidden_states=_stack_hidden_states(generated_hidden_states),
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        cache_position=cache_position,
        input_ids=full_input_ids,
        last_logits=logits,
        model_kwargs=_resume_model_kwargs(model_inputs),
        image_grid_thw=model_inputs.get("image_grid_thw"),
        video_grid_thw=model_inputs.get("video_grid_thw"),
        source_visual_geometry=source_visual_geometry,
        stop_reason="eos_token" if generated_ids and generated_ids[-1] == eos_token_id else "max_new_tokens",
        capture_found=False,
        second_full_forward_used=False,
        malformed=True,
        errors=errors or ["no_complete_focus_span"],
    )


@torch.no_grad()
def tap_qwen3_vision_features(
    model: Any,
    processor: Any,
    *,
    image: Any,
    question: str = "Describe the image.",
    device: torch.device | str | None = None,
) -> tuple[Qwen3VisionTap, torch.Tensor | None, torch.Tensor | None]:
    messages = build_direct_messages(image, question)
    inputs = build_qwen3_inputs(processor, messages)
    model_inputs = _move_tensors(dict(inputs), device or _infer_model_device(model))
    config = getattr(model, "config", None)
    vision_config = getattr(config, "vision_config", None)
    llm_hidden_dim_value = (
        getattr(config, "hidden_size", None)
        or getattr(getattr(config, "text_config", None), "hidden_size", None)
    )
    llm_hidden_dim = int(llm_hidden_dim_value) if llm_hidden_dim_value is not None else None
    if llm_hidden_dim is None:
        try:
            llm_hidden_dim = int(model.get_input_embeddings().weight.shape[-1])
        except Exception:
            llm_hidden_dim = None
    errors: list[str] = []
    image_output = None
    v_pre = None
    v_merge = None
    if model_inputs.get("pixel_values") is None:
        errors.append("missing_pixel_values")
    elif not hasattr(model, "get_image_features"):
        errors.append("missing_get_image_features")
    else:
        try:
            image_output = model.get_image_features(
                model_inputs["pixel_values"],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                output_hidden_states=True,
                return_dict=True,
            )
            v_pre, v_merge = _extract_vision_tensors(image_output)
        except Exception as exc:
            errors.append(f"get_image_features_failed:{type(exc).__name__}:{exc}")

    deepstack = _deepstack_features(image_output)
    tap = Qwen3VisionTap(
        image_grid_thw=_tensor_to_nested_ints(model_inputs.get("image_grid_thw")),
        video_grid_thw=_tensor_to_nested_ints(model_inputs.get("video_grid_thw")),
        v_pre_shape=list(v_pre.shape) if isinstance(v_pre, torch.Tensor) else None,
        v_merge_shape=list(v_merge.shape) if isinstance(v_merge, torch.Tensor) else None,
        deepstack_feature_count=len(deepstack),
        deepstack_feature_shapes=[list(t.shape) for t in deepstack],
        vision_dim=int(v_pre.shape[-1]) if isinstance(v_pre, torch.Tensor) and v_pre.ndim > 0 else None,
        llm_hidden_dim=llm_hidden_dim,
        image_token_id=_config_int(config, "image_token_id"),
        video_token_id=_config_int(config, "video_token_id"),
        vision_start_token_id=_config_int(config, "vision_start_token_id"),
        vision_end_token_id=_config_int(config, "vision_end_token_id"),
        spatial_merge_size=_config_int(vision_config, "spatial_merge_size"),
        merge_size=_config_int(vision_config, "merge_size"),
        patch_size=_config_int(vision_config, "patch_size"),
        temporal_patch_size=_config_int(vision_config, "temporal_patch_size"),
        output_type=type(image_output).__name__ if image_output is not None else "none",
        errors=errors,
    )
    return tap, v_pre.detach().cpu() if isinstance(v_pre, torch.Tensor) else None, (
        v_merge.detach().cpu() if isinstance(v_merge, torch.Tensor) else None
    )


@torch.no_grad()
def extract_qwen3_source_visual_geometry(
    model: Any,
    inputs: dict[str, Any],
) -> Qwen3SourceVisualGeometry:
    errors: list[str] = []
    input_ids = inputs.get("input_ids")
    image_grid_thw = inputs.get("image_grid_thw")
    video_grid_thw = inputs.get("video_grid_thw")
    mm_token_type_ids = inputs.get("mm_token_type_ids")
    config = getattr(model, "config", None)
    image_token_id = _config_int(config, "image_token_id")

    if not isinstance(input_ids, torch.Tensor):
        errors.append("missing_input_ids")
        return Qwen3SourceVisualGeometry(
            image_grid_thw=_detach_cpu_tensor(image_grid_thw),
            video_grid_thw=_detach_cpu_tensor(video_grid_thw),
            source_visual_position_ids=None,
            source_visual_token_indices=None,
            source_visual_token_count=0,
            image_token_id=image_token_id,
            position_ids_shape=None,
            mm_token_type_ids_present=isinstance(mm_token_type_ids, torch.Tensor),
            extraction_mode="unavailable",
            errors=errors,
        )
    if image_token_id is None:
        errors.append("missing_image_token_id")
        return Qwen3SourceVisualGeometry(
            image_grid_thw=_detach_cpu_tensor(image_grid_thw),
            video_grid_thw=_detach_cpu_tensor(video_grid_thw),
            source_visual_position_ids=None,
            source_visual_token_indices=None,
            source_visual_token_count=0,
            image_token_id=None,
            position_ids_shape=None,
            mm_token_type_ids_present=isinstance(mm_token_type_ids, torch.Tensor),
            extraction_mode="unavailable",
            errors=errors,
        )

    image_mask = input_ids == int(image_token_id)
    source_token_count = int(image_mask.sum().detach().cpu().item())
    if image_mask.shape[0] != 1:
        errors.append("batch_size_not_one")
        return Qwen3SourceVisualGeometry(
            image_grid_thw=_detach_cpu_tensor(image_grid_thw),
            video_grid_thw=_detach_cpu_tensor(video_grid_thw),
            source_visual_position_ids=None,
            source_visual_token_indices=None,
            source_visual_token_count=source_token_count,
            image_token_id=image_token_id,
            position_ids_shape=None,
            mm_token_type_ids_present=isinstance(mm_token_type_ids, torch.Tensor),
            extraction_mode="unavailable",
            errors=errors,
        )

    token_indices = torch.nonzero(image_mask[0], as_tuple=False).view(-1)
    if int(token_indices.numel()) == 0:
        errors.append("no_source_image_tokens")
        return Qwen3SourceVisualGeometry(
            image_grid_thw=_detach_cpu_tensor(image_grid_thw),
            video_grid_thw=_detach_cpu_tensor(video_grid_thw),
            source_visual_position_ids=None,
            source_visual_token_indices=token_indices.detach().cpu(),
            source_visual_token_count=0,
            image_token_id=image_token_id,
            position_ids_shape=None,
            mm_token_type_ids_present=isinstance(mm_token_type_ids, torch.Tensor),
            extraction_mode="unavailable",
            errors=errors,
        )

    try:
        position_ids = _compute_qwen3_position_ids_for_sequence(
            model=model,
            input_ids=input_ids,
            attention_mask=inputs.get("attention_mask"),
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )
        if position_ids is None:
            errors.append("missing_compute_3d_position_ids")
            source_positions = None
            position_shape = None
            extraction_mode = "image_token_indices_only"
        else:
            source_positions = position_ids[:, 0, token_indices.to(position_ids.device)].detach().cpu()
            position_shape = list(position_ids.shape)
            extraction_mode = "qwen3_compute_3d_position_ids"
    except Exception as exc:
        errors.append(f"compute_source_position_ids_failed:{type(exc).__name__}:{exc}")
        source_positions = None
        position_shape = None
        extraction_mode = "image_token_indices_only"

    return Qwen3SourceVisualGeometry(
        image_grid_thw=_detach_cpu_tensor(image_grid_thw),
        video_grid_thw=_detach_cpu_tensor(video_grid_thw),
        source_visual_position_ids=source_positions,
        source_visual_token_indices=token_indices.detach().cpu(),
        source_visual_token_count=int(token_indices.numel()),
        image_token_id=image_token_id,
        position_ids_shape=position_shape,
        mm_token_type_ids_present=isinstance(mm_token_type_ids, torch.Tensor),
        extraction_mode=extraction_mode,
        errors=errors,
    )


def make_smoke_d(
    *,
    source: Literal["zero", "random", "random_calibrated"],
    num_fvt_tokens: int,
    hidden_dim: int,
    reference: torch.Tensor | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if num_fvt_tokens <= 0:
        raise ValueError("num_fvt_tokens must be positive")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    dtype = dtype or torch.float32
    if source == "zero":
        return torch.zeros((num_fvt_tokens, hidden_dim), device=device, dtype=dtype)
    if source == "random":
        return torch.randn((num_fvt_tokens, hidden_dim), device=device, dtype=dtype) * 0.02
    if source == "random_calibrated":
        if reference is not None and reference.numel() > 1 and reference.shape[-1] == hidden_dim:
            ref = reference.float()
            mean = ref.mean(dim=tuple(range(ref.ndim - 1)), keepdim=False)
            std = ref.std(dim=tuple(range(ref.ndim - 1)), keepdim=False).clamp_min(1e-5)
            noise = torch.randn((num_fvt_tokens, hidden_dim), device=device, dtype=torch.float32)
            return (noise * std.to(device) + mean.to(device)).to(dtype=dtype)
        return torch.randn((num_fvt_tokens, hidden_dim), device=device, dtype=dtype) * 0.02
    raise ValueError(f"Unsupported D source: {source}")


@torch.no_grad()
def append_tgvf_visual_tokens_qwen3(
    model: Any,
    tokenizer_or_processor: Any,
    capture: Qwen3FocusCapture,
    foveated_visual_tokens: torch.Tensor,
    *,
    continuation_instruction: str = "",
    force_answer_tag: bool = True,
    position_mode: Literal["native_source_grid", "inherit_source_visual_positions"] = "native_source_grid",
) -> Qwen3AppendResult:
    tokenizer = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)
    if not capture.capture_found:
        raise ValueError("capture must contain a valid focus span before FVT append")
    d = foveated_visual_tokens
    if d.ndim != 2:
        raise ValueError("foveated_visual_tokens must have shape [M, hidden_dim]")
    device = _infer_model_device(model) or d.device
    embed = model.get_input_embeddings()
    hidden_dim = int(embed.weight.shape[-1])
    if d.shape[-1] != hidden_dim:
        raise ValueError(
            f"FVT dim {int(d.shape[-1])} does not match Qwen3 LLM hidden dim {hidden_dim}"
        )
    source_geometry = capture.source_visual_geometry
    if source_geometry is None:
        raise ValueError("capture is missing source visual geometry")
    source_token_count = int(source_geometry.source_visual_token_count)
    if source_token_count <= 0:
        raise ValueError("source visual geometry has no image visual tokens")
    if int(d.shape[0]) != source_token_count:
        raise ValueError(
            f"FVT token count {int(d.shape[0])} must equal source visual token count {source_token_count}"
        )
    prefix = f"\n{TGVF_START}\n"
    suffix = f"\n{TGVF_END}\n"
    if continuation_instruction:
        suffix += continuation_instruction
        if not suffix.endswith("\n"):
            suffix += "\n"
    if force_answer_tag:
        suffix += ANSWER_START

    token_ids = _bracketed_visual_token_ids(
        tokenizer_or_processor,
        model,
        num_fvt_tokens=int(d.shape[0]),
        prefix=prefix,
        suffix=suffix,
        device=device,
    )
    prefix_ids = _encode_text(tokenizer, prefix, device)
    fvt_token_start = int(prefix_ids.shape[0]) + 1
    fvt_token_end = fvt_token_start + int(d.shape[0])
    embeds = embed(token_ids.unsqueeze(0).to(device)).detach().clone()
    embeds[0, fvt_token_start:fvt_token_end] = d.to(device=device, dtype=embeds.dtype)

    attention_mask = capture.attention_mask
    if attention_mask is not None:
        attention_mask = _extend_attention(attention_mask.to(device), int(token_ids.shape[0]))
    input_ids = capture.input_ids
    if input_ids is not None:
        input_ids = torch.cat([input_ids.to(device), token_ids.view(1, -1).to(device)], dim=-1)
    mm_token_type_ids = _fvt_mm_token_type_ids(
        chunk_length=int(token_ids.shape[0]),
        fvt_token_start=fvt_token_start,
        fvt_token_end=fvt_token_end,
        device=device,
    )
    if position_mode == "native_source_grid":
        position_ids = _chunk_position_ids_native_source_grid(
            model=model,
            capture=capture,
            token_ids=token_ids,
            attention_mask=attention_mask,
            chunk_mm_token_type_ids=mm_token_type_ids,
            fvt_token_start=fvt_token_start,
            fvt_token_end=fvt_token_end,
            source_geometry=source_geometry,
            device=device,
        )
    elif position_mode == "inherit_source_visual_positions":
        if source_geometry.source_visual_position_ids is None:
            raise ValueError("source visual position ids are unavailable for inherit_source_visual_positions")
        position_ids = _chunk_position_ids_inherit_source_visual_positions(
            attention_mask=attention_mask,
            chunk_length=int(token_ids.shape[0]),
            visual_token_start=fvt_token_start,
            visual_token_end=fvt_token_end,
            source_visual_position_ids=source_geometry.source_visual_position_ids,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported TGVF position mode: {position_mode}")

    outputs = model(
        inputs_embeds=embeds,
        past_key_values=capture.past_key_values,
        attention_mask=attention_mask,
        position_ids=position_ids,
        mm_token_type_ids=mm_token_type_ids,
        use_cache=True,
        return_dict=True,
    )
    source_positions = source_geometry.source_visual_position_ids
    metadata = {
        "fvt_append_path": "qwen3_visual_special_tokens_embedding_replace",
        "uses_deepstack_for_fvt": False,
        "fvt_shape": list(d.shape),
        "source_image_grid_thw": _tensor_to_nested_ints(source_geometry.image_grid_thw),
        "fvt_token_start": fvt_token_start,
        "fvt_token_end": fvt_token_end,
        "num_fvt_tokens": int(d.shape[0]),
        "source_visual_token_count": source_token_count,
        "d_token_count_source": "source_image_visual_tokens",
        "fvt_position_mode": position_mode,
        "position_ids_shape": list(position_ids.shape) if position_ids is not None else None,
        "mm_token_type_ids_shape": list(mm_token_type_ids.shape),
        "position_ids_have_3d_image_span": _position_ids_have_3d_image_span(
            position_ids,
            fvt_token_start,
            fvt_token_end,
        ),
        "text_positions_are_1d": _text_positions_are_1d(
            position_ids,
            fvt_token_start,
            fvt_token_end,
        ),
        "source_visual_position_shape": list(source_positions.shape)
        if isinstance(source_positions, torch.Tensor)
        else None,
        "visual_position_ids_equal_source": _visual_position_ids_equal_source(
            position_ids,
            fvt_token_start,
            fvt_token_end,
            source_positions,
        ),
        "native_qwen3_position_compute_used": position_mode == "native_source_grid",
        "appended_token_count": int(token_ids.shape[0]),
        "continuation_instruction": continuation_instruction,
        "force_answer_tag": bool(force_answer_tag),
        "second_full_forward_used": False,
        "past_key_values_preserved": capture.past_key_values is not None
        and outputs.past_key_values is not None,
        "deepstack_caution": (
            "TGVF append uses Qwen3 visual special tokens and real 3D positions, "
            "but does not provide native Qwen3 DeepStack visual features."
        ),
    }
    return Qwen3AppendResult(
        past_key_values=outputs.past_key_values,
        attention_mask=attention_mask,
        cache_position=None,
        input_ids=input_ids,
        last_logits=outputs.logits,
        appended_token_ids=token_ids.detach().cpu(),
        appended_inputs_embeds=embeds.detach().cpu(),
        fvt_token_start=fvt_token_start,
        fvt_token_end=fvt_token_end,
        model_kwargs=dict(capture.model_kwargs),
        debug_metadata=metadata,
    )


@torch.no_grad()
def continue_generation_qwen3(
    model: Any,
    tokenizer_or_processor: Any,
    state: Qwen3AppendResult | Qwen3FocusCapture,
    *,
    max_new_tokens: int = 64,
    eos_token_id: int | None = None,
) -> Qwen3Continuation:
    tokenizer = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)
    logits = state.last_logits
    past_key_values = state.past_key_values
    attention_mask = state.attention_mask
    input_ids = state.input_ids
    generated_ids: list[int] = []
    stop_reason = "max_new_tokens"
    device = logits.device if logits is not None else (_infer_model_device(model) or torch.device("cpu"))
    for _ in range(max_new_tokens):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        token_id = int(next_token[0, 0].detach().cpu().item())
        generated_ids.append(token_id)
        if input_ids is not None:
            input_ids = torch.cat([input_ids.to(device), next_token.to(device)], dim=-1)
        if attention_mask is not None:
            attention_mask = _extend_attention(attention_mask.to(device), 1)
        position_ids = _chunk_position_ids_1d(
            attention_mask=attention_mask,
            chunk_length=1,
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
    return Qwen3Continuation(
        generated_ids=generated_ids,
        generated_text=_decode(tokenizer, generated_ids),
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        input_ids=input_ids,
        last_logits=logits,
        stop_reason=stop_reason,
    )


def parse_v3_action(text: str) -> V3ActionParse:
    evidence_state = _extract_tag(text, EVIDENCE_STATE_START, EVIDENCE_STATE_END)
    focus = _extract_tag(text, FOCUS_START, FOCUS_END)
    answer = _extract_tag(text, ANSWER_START, ANSWER_END)
    has_focus_open = FOCUS_START in text
    has_focus_close = FOCUS_END in text
    reasons: list[str] = []
    if has_focus_open and not has_focus_close:
        reasons.append("missing_closing_focus")
    if has_focus_close and not has_focus_open:
        reasons.append("missing_opening_focus")
    if evidence_state is not None and evidence_state not in {NEED_LOCAL_EVIDENCE, SUFFICIENT_EVIDENCE}:
        reasons.append("invalid_evidence_state")
    if focus is not None and not focus.strip():
        reasons.append("empty_focus")
    if focus is not None and is_generic_target(focus):
        reasons.append("generic_target")
    focus_valid = focus is not None and not any(
        reason in reasons for reason in ("missing_closing_focus", "empty_focus", "generic_target")
    )
    answer_valid = answer is not None and bool(answer.strip())
    return V3ActionParse(
        raw_text=text,
        evidence_state=evidence_state,
        focus_target=focus.strip() if focus else "",
        answer=answer.strip() if answer else "",
        has_focus_open=has_focus_open,
        has_focus_close=has_focus_close,
        focus_valid=focus_valid,
        answer_valid=answer_valid,
        malformed=bool(reasons) or (has_focus_open and not focus_valid),
        malformed_reasons=reasons,
    )


def is_generic_target(target: str) -> bool:
    normalized = re.sub(r"\s+", " ", target.strip().lower())
    if normalized in GENERIC_TARGETS:
        return True
    if len(normalized.split()) <= 2 and any(word in normalized.split() for word in {"object", "thing", "part"}):
        return True
    return False


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summarize_smoke(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "direct_completion_rate": _mean(bool(row.get("final_raw_output") or row.get("raw_output")) for row in rows),
        "focus_valid_rate": _mean(bool(row.get("focus_valid")) for row in rows),
        "malformed_rate": _mean(bool(row.get("malformed")) for row in rows),
        "append_success_rate": _mean(bool(row.get("append_success")) for row in rows),
        "im_end_only_rate": _mean(
            _is_im_end_only(str(row.get("final_raw_output") or row.get("raw_output") or ""))
            for row in rows
        ),
        "answer_parse_rate": _mean(bool(row.get("parsed_answer")) for row in rows),
        "second_full_forward_used_any": any(bool(row.get("second_full_forward_used")) for row in rows),
        "avg_wall_time_sec": sum(float(row.get("wall_time_sec") or 0.0) for row in rows) / n,
        "avg_target_tokens": sum(int(row.get("target_token_count") or 0) for row in rows) / n,
        "avg_fvt_tokens": sum(int(row.get("avg_fvt_tokens") or row.get("num_fvt_tokens") or 0) for row in rows) / n,
    }


def capture_to_row(
    *,
    sample_id: str,
    image: str,
    question: str,
    mode: str,
    model_id: str,
    capture: Qwen3FocusCapture | None = None,
    vision_tap: Qwen3VisionTap | None = None,
    append_result: Qwen3AppendResult | None = None,
    continuation: Qwen3Continuation | None = None,
    direct: dict[str, Any] | None = None,
    errors: list[str] | None = None,
    wall_time_sec: float = 0.0,
    peak_memory_gb: float | None = None,
) -> dict[str, Any]:
    final_raw = continuation.generated_text if continuation is not None else None
    parsed_final = parse_v3_action(final_raw or "")
    parsed_capture = parse_v3_action(capture.generated_text if capture is not None else "")
    row = {
        "id": sample_id,
        "image": image,
        "question": question,
        "mode": mode,
        "model_id": model_id,
        "focus_raw_output": capture.generated_text if capture is not None else None,
        "focus_target_text": capture.target_text if capture is not None else "",
        "focus_valid": bool(capture.capture_found) if capture is not None else False,
        "malformed": bool(capture.malformed or parsed_capture.malformed) if capture is not None else False,
        "target_token_count": len(capture.target_token_ids) if capture is not None else 0,
        "target_hidden_shape": list(capture.target_hidden_states.shape) if capture is not None else None,
        "source_visual_token_count": (
            capture.source_visual_geometry.source_visual_token_count
            if capture is not None and capture.source_visual_geometry is not None
            else 0
        ),
        "source_visual_position_shape": (
            list(capture.source_visual_geometry.source_visual_position_ids.shape)
            if capture is not None
            and capture.source_visual_geometry is not None
            and isinstance(capture.source_visual_geometry.source_visual_position_ids, torch.Tensor)
            else None
        ),
        "source_visual_geometry_errors": (
            list(capture.source_visual_geometry.errors)
            if capture is not None and capture.source_visual_geometry is not None
            else []
        ),
        "past_key_values_preserved": (
            capture.past_key_values is not None if capture is not None else None
        ),
        "second_full_forward_used": (
            capture.second_full_forward_used if capture is not None else False
        ),
        "vision_tap": asdict(vision_tap) if vision_tap is not None else None,
        "fvt_shape": append_result.debug_metadata.get("fvt_shape") if append_result is not None else None,
        "fvt_append_path": (
            append_result.debug_metadata.get("fvt_append_path") if append_result is not None else None
        ),
        "fvt_position_mode": (
            append_result.debug_metadata.get("fvt_position_mode") if append_result is not None else None
        ),
        "native_qwen3_position_compute_used": (
            append_result.debug_metadata.get("native_qwen3_position_compute_used")
            if append_result is not None
            else None
        ),
        "visual_position_ids_equal_source": (
            append_result.debug_metadata.get("visual_position_ids_equal_source")
            if append_result is not None
            else None
        ),
        "uses_deepstack_for_fvt": (
            append_result.debug_metadata.get("uses_deepstack_for_fvt") if append_result is not None else None
        ),
        "continuation_instruction": (
            append_result.debug_metadata.get("continuation_instruction") if append_result is not None else None
        ),
        "append_success": append_result is not None,
        "final_raw_output": final_raw,
        "parsed_answer": (
            parsed_final.answer if parsed_final.answer_valid else (direct or {}).get("parsed_answer")
        ),
        "raw_output": (direct or {}).get("raw_output"),
        "errors": list(errors or []) + (capture.errors if capture is not None else []),
        "wall_time_sec": wall_time_sec,
        "peak_memory_gb": peak_memory_gb,
        "num_fvt_tokens": (
            append_result.debug_metadata.get("num_fvt_tokens") if append_result is not None else 0
        ),
    }
    return row


def peak_memory_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated() / (1024**3))


def llm_hidden_dim(model: Any) -> int:
    config_dim = int(getattr(getattr(model, "config", None), "hidden_size", 0) or 0)
    if config_dim > 0:
        return config_dim
    return int(model.get_input_embeddings().weight.shape[-1])


def _vision_content_items(media: Any) -> list[dict[str, Any]]:
    if isinstance(media, dict):
        if media.get("type") in {"image", "video"}:
            return [media]
        if "image" in media:
            return [{"type": "image", **media}]
    if isinstance(media, (list, tuple)):
        return [
            item
            if isinstance(item, dict) and item.get("type") in {"image", "video"}
            else {"type": "image", "image": item}
            for item in media
        ]
    return [{"type": "image", "image": media}]


def _prepare_decode_step(
    model: Any,
    *,
    full_input_ids: torch.Tensor,
    next_token: torch.Tensor,
    past_key_values: Any,
    attention_mask: torch.Tensor | None,
    base_inputs: dict[str, Any],
    cache_position: torch.Tensor | None,
    generated_token_count: int,
) -> dict[str, Any]:
    kwargs = {
        "past_key_values": past_key_values,
        "attention_mask": attention_mask,
        "use_cache": True,
        "pixel_values": base_inputs.get("pixel_values"),
        "pixel_values_videos": base_inputs.get("pixel_values_videos"),
        "image_grid_thw": base_inputs.get("image_grid_thw"),
        "video_grid_thw": base_inputs.get("video_grid_thw"),
        "is_first_iteration": False,
    }
    for key in (
        "rope_deltas",
        "mm_token_type_ids",
        "position_ids",
        "second_per_grid_ts",
        "video_second_per_grid",
    ):
        if key in base_inputs:
            kwargs[key] = base_inputs[key]
    if cache_position is not None:
        kwargs["cache_position"] = cache_position
    position_ids = _decode_position_ids(
        attention_mask=attention_mask,
        next_token=next_token,
        past_key_values=past_key_values,
        rope_deltas=base_inputs.get("rope_deltas"),
        generated_token_count=generated_token_count,
    )
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    if hasattr(model, "prepare_inputs_for_generation"):
        return model.prepare_inputs_for_generation(full_input_ids, **kwargs)
    return {
        "input_ids": next_token,
        "past_key_values": past_key_values,
        "attention_mask": attention_mask,
        "use_cache": True,
    }


def _decode_position_ids(
    *,
    attention_mask: torch.Tensor | None,
    next_token: torch.Tensor,
    past_key_values: Any,
    rope_deltas: torch.Tensor | None,
    generated_token_count: int,
) -> torch.Tensor | None:
    if rope_deltas is None:
        return None
    batch_size = next_token.shape[0]
    if attention_mask is not None:
        base_position = attention_mask.long().cumsum(-1)[:, -1:] - 1
    else:
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        base_position = torch.full(
            (batch_size, 1),
            past_length + generated_token_count - 1,
            dtype=torch.long,
            device=next_token.device,
        )
    base_position = base_position.view(1, batch_size, 1).repeat(3, 1, 1)
    delta = rope_deltas.to(device=next_token.device)
    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
    return base_position + delta.view(1, batch_size, 1)


def _find_completed_focus(
    generated_ids: list[int],
    start_marker_ids: list[int],
    end_marker_ids: list[int],
    tokenizer: Any,
) -> tuple[int, int, str] | None:
    start_index = _find_subsequence(generated_ids, start_marker_ids)
    if start_index is not None:
        target_start = start_index + len(start_marker_ids)
        relative_end = _find_subsequence(generated_ids[target_start:], end_marker_ids)
        if relative_end is not None:
            target_end = target_start + relative_end
            start, end = _trim_whitespace_edges(tokenizer, generated_ids, target_start, target_end)
            return start, end, _decode(tokenizer, generated_ids[start:end]).strip()

    text = _decode(tokenizer, generated_ids)
    focus = _extract_tag(text, FOCUS_START, FOCUS_END)
    if focus is None:
        return None
    offsets = _decoded_token_offsets(tokenizer, generated_ids)
    inner_start = text.find(FOCUS_START) + len(FOCUS_START)
    raw_focus = text[inner_start : text.find(FOCUS_END, inner_start)]
    leading = len(raw_focus) - len(raw_focus.lstrip())
    trailing = len(raw_focus.rstrip())
    char_start = inner_start + leading
    char_end = inner_start + trailing
    token_indices = [
        index
        for index, (tok_start, tok_end) in enumerate(offsets)
        if tok_start < char_end and tok_end > char_start
    ]
    if not token_indices:
        return None
    return token_indices[0], token_indices[-1] + 1, focus.strip()


def _extract_tag(text: str, start: str, end: str) -> str | None:
    start_index = text.find(start)
    if start_index < 0:
        return None
    inner_start = start_index + len(start)
    end_index = text.find(end, inner_start)
    if end_index < 0:
        return None
    return text[inner_start:end_index].strip()


def _find_subsequence(values: list[int], pattern: list[int]) -> int | None:
    if not pattern or len(pattern) > len(values):
        return None
    for index in range(len(values) - len(pattern) + 1):
        if values[index : index + len(pattern)] == pattern:
            return index
    return None


def _trim_whitespace_edges(
    tokenizer: Any,
    generated_ids: list[int],
    target_start: int,
    target_end: int,
) -> tuple[int, int]:
    start = target_start
    end = target_end
    while start < end and _decode(tokenizer, [generated_ids[start]]).isspace():
        start += 1
    while end > start and _decode(tokenizer, [generated_ids[end - 1]]).isspace():
        end -= 1
    return start, end


def _decoded_token_offsets(tokenizer: Any, token_ids: list[int]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token_id in token_ids:
        token_text = _decode(tokenizer, [token_id])
        end = cursor + len(token_text)
        offsets.append((cursor, end))
        cursor = end
    return offsets


def _extract_vision_tensors(output: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if output is None:
        return None, None
    hidden_states = getattr(output, "hidden_states", None)
    last_hidden = getattr(output, "last_hidden_state", None)
    pooler = getattr(output, "pooler_output", None)
    if last_hidden is None and isinstance(output, (tuple, list)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            last_hidden = first
    v_pre = None
    if hidden_states:
        candidates = [item for item in hidden_states if isinstance(item, torch.Tensor)]
        if candidates:
            v_pre = candidates[0]
    if v_pre is None:
        v_pre = last_hidden if isinstance(last_hidden, torch.Tensor) else None
    v_merge = pooler if isinstance(pooler, torch.Tensor) else None
    if v_merge is None:
        v_merge = last_hidden if isinstance(last_hidden, torch.Tensor) else None
    return v_pre, v_merge


def _deepstack_features(output: Any) -> list[torch.Tensor]:
    features = getattr(output, "deepstack_features", None)
    if features is None and isinstance(output, dict):
        features = output.get("deepstack_features")
    if features is None:
        return []
    return [item for item in features if isinstance(item, torch.Tensor)]


def _bracketed_visual_token_ids(
    tokenizer_or_processor: Any,
    model: Any,
    *,
    num_fvt_tokens: int,
    prefix: str,
    suffix: str,
    device: torch.device | str | None,
) -> torch.Tensor:
    tokenizer = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)
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
    prefix_ids = _encode_text(tokenizer, prefix, device)
    suffix_ids = _encode_text(tokenizer, suffix, device)
    middle = torch.tensor(
        [vision_start, *([image_token] * num_fvt_tokens), vision_end],
        dtype=torch.long,
        device=device,
    )
    return torch.cat([prefix_ids, middle, suffix_ids], dim=0)


def _marker_ids(tokenizer: Any, marker: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(marker, add_special_tokens=False)
    else:
        ids = tokenizer(marker, add_special_tokens=False)["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _encode_text(tokenizer: Any, text: str, device: torch.device | str | None) -> torch.Tensor:
    return torch.tensor(
        tokenizer.encode(text, add_special_tokens=False),
        dtype=torch.long,
        device=device,
    )


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


def _append_attention(attention_mask: torch.Tensor | None, next_token: torch.Tensor) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    next_attention = torch.ones(
        (attention_mask.shape[0], 1),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    return torch.cat([attention_mask, next_attention], dim=-1)


def _extend_attention(attention_mask: torch.Tensor, chunk_length: int) -> torch.Tensor:
    ones = torch.ones(
        (attention_mask.shape[0], chunk_length),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    return torch.cat([attention_mask, ones], dim=-1)


def _chunk_position_ids_1d(
    *,
    attention_mask: torch.Tensor | None,
    chunk_length: int,
    device: torch.device | str,
) -> torch.Tensor | None:
    if attention_mask is None:
        raise ValueError("attention_mask is required for inherit_source_visual_positions")
    base = attention_mask.long().cumsum(-1)[:, -chunk_length:] - 1
    batch_size = base.shape[0]
    return base.view(1, batch_size, chunk_length).repeat(3, 1, 1).to(device=device)


def _chunk_position_ids_inherit_source_visual_positions(
    *,
    attention_mask: torch.Tensor | None,
    chunk_length: int,
    visual_token_start: int,
    visual_token_end: int,
    source_visual_position_ids: torch.Tensor,
    device: torch.device | str,
) -> torch.Tensor | None:
    if attention_mask is None:
        raise ValueError("attention_mask is required for native_source_grid position ids")
    if visual_token_start < 0 or visual_token_end <= visual_token_start:
        raise ValueError("invalid visual token span")
    if visual_token_end > chunk_length:
        raise ValueError("visual token span exceeds chunk length")
    visual_len = visual_token_end - visual_token_start
    source_positions = source_visual_position_ids.to(device=device)
    if source_positions.shape != (3, visual_len):
        raise ValueError(
            f"source visual positions shape {tuple(source_positions.shape)} != (3, {visual_len})"
        )
    base = attention_mask.long().cumsum(-1)[:, -chunk_length:] - 1
    if base.shape[0] != 1:
        raise ValueError("Qwen3 TGVF append currently supports batch size 1")
    position_ids = base.view(1, 1, chunk_length).repeat(3, 1, 1).to(device=device)
    position_ids[:, 0, visual_token_start:visual_token_end] = source_positions
    return position_ids


def _chunk_position_ids_native_source_grid(
    *,
    model: Any,
    capture: Qwen3FocusCapture,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    chunk_mm_token_type_ids: torch.Tensor,
    fvt_token_start: int,
    fvt_token_end: int,
    source_geometry: Qwen3SourceVisualGeometry,
    device: torch.device | str,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if capture.input_ids is None:
        raise ValueError("capture.input_ids is required for native_source_grid position ids")
    if not isinstance(source_geometry.image_grid_thw, torch.Tensor):
        raise ValueError("source image_grid_thw is required for native_source_grid position ids")
    if source_geometry.image_grid_thw.detach().cpu().view(-1, 3).shape[0] != 1:
        raise ValueError("native_source_grid append currently supports exactly one source image grid")
    full_input_ids = torch.cat(
        [capture.input_ids.to(device), token_ids.view(1, -1).to(device)],
        dim=-1,
    )
    full_mm_token_type_ids = _full_mm_token_type_ids_for_append(
        model=model,
        capture_input_ids=capture.input_ids.to(device),
        chunk_mm_token_type_ids=chunk_mm_token_type_ids.to(device),
        device=device,
    )
    image_grid_thw = _append_source_image_grid(
        capture.image_grid_thw,
        source_geometry.image_grid_thw,
        device=device,
    )
    position_ids = _compute_qwen3_position_ids_for_sequence(
        model=model,
        input_ids=full_input_ids,
        attention_mask=attention_mask,
        image_grid_thw=image_grid_thw,
        video_grid_thw=capture.video_grid_thw,
        mm_token_type_ids=full_mm_token_type_ids,
    )
    if position_ids is None:
        raise ValueError("Qwen3 model does not expose compute_3d_position_ids")
    chunk_len = int(token_ids.shape[0])
    visual_len = fvt_token_end - fvt_token_start
    if visual_len != int(source_geometry.source_visual_token_count):
        raise ValueError("native_source_grid visual span length does not match source token count")
    return position_ids[:, :, -chunk_len:].to(device=device)


def _compute_qwen3_position_ids_for_sequence(
    *,
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
    mm_token_type_ids: torch.Tensor | None,
) -> torch.Tensor | None:
    model_core = getattr(model, "model", None)
    compute_fn = getattr(model_core, "compute_3d_position_ids", None)
    if compute_fn is None:
        compute_fn = getattr(model, "compute_3d_position_ids", None)
    if compute_fn is None or not hasattr(model, "get_input_embeddings"):
        return None
    inputs_embeds = model.get_input_embeddings()(input_ids)
    kwargs = {
        "input_ids": input_ids,
        "inputs_embeds": inputs_embeds,
        "image_grid_thw": image_grid_thw,
        "video_grid_thw": video_grid_thw,
        "attention_mask": attention_mask,
        "past_key_values": None,
        "mm_token_type_ids": mm_token_type_ids,
    }
    try:
        result = compute_fn(**kwargs)
    except TypeError:
        kwargs.pop("mm_token_type_ids", None)
        result = compute_fn(**kwargs)
    if isinstance(result, (tuple, list)):
        result = result[0] if result else None
    return result if isinstance(result, torch.Tensor) else None


def _full_mm_token_type_ids_for_append(
    *,
    model: Any,
    capture_input_ids: torch.Tensor,
    chunk_mm_token_type_ids: torch.Tensor,
    device: torch.device | str,
) -> torch.Tensor:
    config = getattr(model, "config", None)
    image_token_id = _config_int(config, "image_token_id")
    prefix = torch.zeros_like(capture_input_ids, dtype=torch.long, device=device)
    if image_token_id is not None:
        prefix = torch.where(capture_input_ids == int(image_token_id), torch.ones_like(prefix), prefix)
    return torch.cat([prefix, chunk_mm_token_type_ids.to(device)], dim=-1)


def _append_source_image_grid(
    original_image_grid_thw: torch.Tensor | None,
    source_image_grid_thw: torch.Tensor,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    source = source_image_grid_thw.to(device=device, dtype=torch.long).view(-1, 3)
    if original_image_grid_thw is None:
        return source
    original = original_image_grid_thw.to(device=device, dtype=torch.long).view(-1, 3)
    return torch.cat([original, source], dim=0)


def _position_ids_have_3d_image_span(
    position_ids: torch.Tensor | None,
    visual_token_start: int,
    visual_token_end: int,
) -> bool:
    if position_ids is None or position_ids.ndim != 3 or position_ids.shape[0] != 3:
        return False
    visual_positions = position_ids[:, 0, visual_token_start:visual_token_end]
    if visual_positions.numel() == 0:
        return False
    if visual_positions.shape[-1] <= 1:
        return True
    all_rows_equal = torch.equal(visual_positions[0], visual_positions[1]) and torch.equal(
        visual_positions[1], visual_positions[2]
    )
    return not all_rows_equal


def _text_positions_are_1d(
    position_ids: torch.Tensor | None,
    visual_token_start: int,
    visual_token_end: int,
) -> bool:
    if position_ids is None or position_ids.ndim != 3 or position_ids.shape[0] != 3:
        return False
    pieces = []
    if visual_token_start > 0:
        pieces.append(position_ids[:, 0, :visual_token_start])
    if visual_token_end < position_ids.shape[-1]:
        pieces.append(position_ids[:, 0, visual_token_end:])
    if not pieces:
        return True
    text_positions = torch.cat(pieces, dim=-1)
    return bool(
        torch.equal(text_positions[0], text_positions[1])
        and torch.equal(text_positions[1], text_positions[2])
    )


def _visual_position_ids_equal_source(
    position_ids: torch.Tensor | None,
    visual_token_start: int,
    visual_token_end: int,
    source_visual_position_ids: torch.Tensor | None,
) -> bool:
    if (
        position_ids is None
        or source_visual_position_ids is None
        or position_ids.ndim != 3
        or position_ids.shape[0] != 3
    ):
        return False
    visual_positions = position_ids[:, 0, visual_token_start:visual_token_end].detach().cpu()
    return bool(torch.equal(visual_positions, source_visual_position_ids.detach().cpu()))


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


def _stack_hidden_states(hidden_states: list[torch.Tensor]) -> torch.Tensor:
    if not hidden_states:
        return torch.empty((0, 0))
    return torch.stack([hidden.detach().cpu() for hidden in hidden_states], dim=0)


def _empty_hidden_like(hidden_states: list[torch.Tensor]) -> torch.Tensor:
    stacked = _stack_hidden_states(hidden_states)
    hidden_dim = stacked.shape[-1] if stacked.ndim == 2 else 0
    return stacked.new_empty((0, hidden_dim))


def _resume_model_kwargs(base_inputs: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "image_grid_thw",
        "video_grid_thw",
        "rope_deltas",
        "mm_token_type_ids",
        "position_ids",
        "second_per_grid_ts",
        "video_second_per_grid",
    )
    return {key: base_inputs[key] for key in keys if key in base_inputs}


def _move_tensors(inputs: dict[str, Any], device: torch.device | str | None) -> dict[str, Any]:
    if device is None:
        return inputs
    return {
        key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()
    }


def _tensor_to_nested_ints(value: Any) -> list[list[int]] | None:
    if not isinstance(value, torch.Tensor):
        return None
    return [[int(item) for item in row] for row in value.detach().cpu().view(-1, value.shape[-1]).tolist()]


def _detach_cpu_tensor(value: Any) -> torch.Tensor | None:
    if not isinstance(value, torch.Tensor):
        return None
    return value.detach().cpu()


def _config_int(config: Any, name: str) -> int | None:
    value = getattr(config, name, None)
    return int(value) if value is not None else None


def _resolve_dtype(dtype: str) -> torch.dtype | str:
    if dtype == "auto":
        return "auto"
    if dtype in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype in {"float16", "fp16"}:
        return torch.float16
    if dtype in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _infer_model_device(model: Any) -> torch.device | None:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return None


def _decode(tokenizer: Any, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _mean(values: Any) -> float:
    vals = [1.0 if value else 0.0 for value in values]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _is_im_end_only(text: str) -> bool:
    stripped = text.strip()
    return stripped in {"", "<|im_end|>", "</s>"}
