from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from qwen_vl_utils import process_vision_info

from revisit_vlm.tgvf import FOVEATE_END, FOVEATE_START, parse_foveation_spans


@dataclass(frozen=True)
class _CompletedRequest:
    target_token_start: int
    target_token_end: int
    raw_target_text: str | None = None
    target_text: str | None = None
    boundary_note: str | None = None


@dataclass
class TGVFCaptureResult:
    target_text: str
    raw_target_text: str
    target_token_ids: list[int]
    target_hidden_states: torch.Tensor
    target_token_start: int | None
    target_token_end: int | None
    generated_ids: list[int]
    generated_text: str
    generated_hidden_states: torch.Tensor
    past_key_values: Any
    attention_mask: torch.Tensor | None
    cache_position: torch.Tensor | None
    input_ids: torch.Tensor | None
    last_logits: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    video_grid_thw: torch.Tensor | None
    model_kwargs: dict[str, Any]
    stop_reason: str
    capture_found: bool
    whitespace_trim_note: str | None = None


GenerationEventType = Literal["foveate", "final_answer", "eos_token", "max_new_tokens"]


@dataclass
class TGVFGenerationEvent:
    event_type: GenerationEventType
    target_text: str
    raw_target_text: str
    target_token_ids: list[int]
    target_hidden_states: torch.Tensor
    target_token_start: int | None
    target_token_end: int | None
    generated_ids: list[int]
    generated_text: str
    generated_hidden_states: torch.Tensor
    past_key_values: Any
    attention_mask: torch.Tensor | None
    cache_position: torch.Tensor | None
    input_ids: torch.Tensor | None
    last_logits: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    video_grid_thw: torch.Tensor | None
    model_kwargs: dict[str, Any]
    stop_reason: str
    capture_found: bool = False
    final_answer: str | None = None
    debug_metadata: dict[str, Any] | None = None
    whitespace_trim_note: str | None = None


_FINAL_ANSWER_RE = re.compile(
    r"final answer\s*(?:is|:)?\s*[\(\[]?([A-Z])[\)\].,:\s]?",
    re.IGNORECASE,
)


def trim_capture_target_at_delimiters(
    result: TGVFCaptureResult,
    tokenizer: Any,
    delimiters: tuple[str, ...] = ("|", "\n", "<|endoftext|>", "<|im_end|>"),
) -> TGVFCaptureResult:
    """Trim generated force targets at common answer/turn separators in-place."""

    if not result.capture_found or not result.target_text or not result.target_token_ids:
        return result
    raw_text = result.target_text
    cut = _first_delimiter_index(raw_text, delimiters)
    if cut is None:
        return result
    trimmed_text = raw_text[:cut].strip()
    if not trimmed_text:
        return result
    keep_tokens = _target_prefix_token_count(result.target_token_ids, tokenizer, delimiters)
    if keep_tokens <= 0:
        return result
    if keep_tokens < len(result.target_token_ids):
        result.target_token_ids = result.target_token_ids[:keep_tokens]
        result.target_hidden_states = result.target_hidden_states[:keep_tokens]
        if result.target_token_start is not None:
            result.target_token_end = result.target_token_start + keep_tokens
    result.target_text = trimmed_text
    note = f"Trimmed generated target at delimiter from {raw_text!r} to {trimmed_text!r}."
    result.whitespace_trim_note = (
        note if result.whitespace_trim_note is None else f"{result.whitespace_trim_note} {note}"
    )
    return result


def _first_delimiter_index(text: str, delimiters: tuple[str, ...]) -> int | None:
    indices = [text.find(delimiter) for delimiter in delimiters if delimiter and text.find(delimiter) >= 0]
    return min(indices) if indices else None


def _target_prefix_token_count(token_ids: list[int], tokenizer: Any, delimiters: tuple[str, ...]) -> int:
    count = 0
    for token_id in token_ids:
        token_text = _decode(tokenizer, [token_id])
        cut = _first_delimiter_index(token_text, delimiters)
        if cut is None:
            count += 1
            continue
        if cut > 0:
            count += 1
        break
    return count

def format_tgvf_debug(result: TGVFCaptureResult) -> str:
    return (
        "generated_text:\n"
        f"{result.generated_text}\n\n"
        "target_text:\n"
        f"{result.target_text}\n\n"
        "target token count:\n"
        f"{len(result.target_token_ids)}\n\n"
        "target_hidden_states shape:\n"
        f"{tuple(result.target_hidden_states.shape)}\n\n"
        "stop_reason:\n"
        f"{result.stop_reason}\n\n"
        "past_key_values:\n"
        f"{'present' if result.past_key_values is not None else 'missing'}"
    )


def print_tgvf_debug(result: TGVFCaptureResult) -> None:
    print(format_tgvf_debug(result))


def build_tgvf_prompt(question: str) -> str:
    return (
        f"{question}\n\n"
        "If fine-grained visual evidence is needed, output exactly one foveation request:\n"
        f"{FOVEATE_START}visual target description{FOVEATE_END}\n"
        f"After {FOVEATE_END}, stop.\n"
        "Do not answer the question yet.\n"
        "Do not emit intent, mode, scope, JSON, tool metadata, explanations, "
        "or natural-language commentary.\n"
        "Only emit the foveation request."
    )


def build_qwen2vl_tgvf_inputs(
    processor: Any,
    *,
    image: Any,
    question: str,
    messages: list[dict[str, Any]] | None = None,
    image_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare the multimodal Qwen2-VL prompt through the project processor path."""

    if messages is None:
        messages = [
            {
                "role": "user",
                "content": [
                    *_vision_content_items(image, image_kwargs),
                    {"type": "text", "text": build_tgvf_prompt(question)},
                ],
            }
        ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


def _vision_content_items(media: Any, image_kwargs: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if _looks_like_video(media):
        frames = _sample_video_frames(media, int((image_kwargs or {}).get("nframes") or 8))
        if frames:
            frame_kwargs = {
                key: value
                for key, value in (image_kwargs or {}).items()
                if key != "nframes" and value is not None
            }
            return [_image_content(frame, frame_kwargs) for frame in frames]
    return [_image_content(media, image_kwargs)]


def _image_content(image: Any, image_kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
    content = {"type": "image", "image": image}
    if image_kwargs:
        content.update(
            {key: value for key, value in image_kwargs.items() if key != "nframes" and value is not None}
        )
    return content


def _looks_like_video(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    lowered = value.lower().split("?", 1)[0]
    return lowered.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))


def _sample_video_frames(video_path: str, nframes: int) -> list[Any]:
    if nframes <= 0 or not Path(video_path).exists():
        return []
    try:
        import av
    except Exception:
        return []

    frames: list[Any] = []
    try:
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            total = int(stream.frames or 0)
            if total > 0:
                count = min(nframes, total)
                if count == 1:
                    targets = {max(0, total // 2)}
                else:
                    targets = {round(index * (total - 1) / (count - 1)) for index in range(count)}
                max_target = max(targets)
                for frame_index, frame in enumerate(container.decode(stream)):
                    if frame_index in targets:
                        frames.append(frame.to_image().convert("RGB"))
                    if frame_index >= max_target:
                        break
            else:
                for frame in container.decode(stream):
                    frames.append(frame.to_image().convert("RGB"))
                    if len(frames) >= nframes:
                        break
    except Exception:
        return []
    return frames


@torch.no_grad()
def capture_tgvf_single_pass(
    model: Any,
    processor: Any,
    *,
    image: Any,
    question: str,
    messages: list[dict[str, Any]] | None = None,
    max_new_tokens: int = 128,
    device: torch.device | str | None = None,
    hidden_state_index: int = -1,
    eos_token_id: int | None = None,
    image_kwargs: dict[str, Any] | None = None,
) -> TGVFCaptureResult:
    """Run Qwen2-VL and capture a TGVF request without replaying the full context."""

    inputs = build_qwen2vl_tgvf_inputs(
        processor,
        image=image,
        question=question,
        messages=messages,
        image_kwargs=image_kwargs,
    )
    tokenizer = processor.tokenizer
    return capture_tgvf_single_pass_from_inputs(
        model,
        tokenizer,
        inputs,
        max_new_tokens=max_new_tokens,
        device=device,
        hidden_state_index=hidden_state_index,
        eos_token_id=eos_token_id,
    )


@torch.no_grad()
def capture_tgvf_single_pass_from_inputs(
    model: Any,
    tokenizer: Any,
    inputs: dict[str, Any],
    *,
    max_new_tokens: int = 128,
    device: torch.device | str | None = None,
    hidden_state_index: int = -1,
    eos_token_id: int | None = None,
    image_kwargs: dict[str, Any] | None = None,
) -> TGVFCaptureResult:
    """Greedy single-pass TGVF capture from already processed model inputs.

    The full multimodal prompt is forwarded once for prefill. Every generated token is
    then fed back as a one-token decode step, and its hidden state is stored from that
    same step.
    """

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    start_marker_ids = _marker_ids(tokenizer, FOVEATE_START)
    end_marker_ids = _marker_ids(tokenizer, FOVEATE_END)
    if not start_marker_ids or not end_marker_ids:
        raise ValueError("Foveation markers must tokenize to non-empty id sequences")

    if device is None:
        device = _infer_model_device(model)
    model_inputs = _move_tensors(dict(inputs), device)

    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs.get("attention_mask")
    full_input_ids = input_ids

    outputs = model(
        **model_inputs,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    past_key_values = outputs.past_key_values
    logits = outputs.logits
    rope_deltas = getattr(outputs, "rope_deltas", None)
    if rope_deltas is not None:
        model_inputs["rope_deltas"] = rope_deltas

    generated_ids: list[int] = []
    generated_hidden_states: list[torch.Tensor] = []
    cache_position = model_inputs.get("cache_position")

    for _ in range(max_new_tokens):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        token_id = int(next_token[0, 0].item())

        full_input_ids = torch.cat([full_input_ids, next_token], dim=-1)
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

        token_hidden = outputs.hidden_states[hidden_state_index][0, -1].detach()
        generated_ids.append(token_id)
        generated_hidden_states.append(token_hidden)

        past_key_values = outputs.past_key_values
        logits = outputs.logits
        cache_position = step_inputs.get("cache_position")
        capture = _find_completed_request(
            generated_ids,
            start_marker_ids,
            end_marker_ids,
            tokenizer=tokenizer,
        )
        if capture is not None:
            return _build_capture_result(
                tokenizer=tokenizer,
                generated_ids=generated_ids,
                generated_hidden_states=generated_hidden_states,
                capture=capture,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                cache_position=cache_position,
                input_ids=full_input_ids,
                last_logits=logits,
                base_inputs=model_inputs,
                stop_reason="foveation_end_marker",
                capture_found=True,
            )

        if eos_token_id is not None and token_id == eos_token_id:
            break

    stop_reason = (
        "eos_token" if generated_ids and generated_ids[-1] == eos_token_id else "max_new_tokens"
    )
    return _build_empty_result(
        tokenizer=tokenizer,
        generated_ids=generated_ids,
        generated_hidden_states=generated_hidden_states,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        cache_position=cache_position,
        input_ids=full_input_ids,
        last_logits=logits,
        base_inputs=model_inputs,
        stop_reason=stop_reason,
    )


@torch.no_grad()
def generate_until_event_from_inputs(
    model: Any,
    tokenizer: Any,
    inputs: dict[str, Any],
    *,
    max_new_tokens: int = 128,
    device: torch.device | str | None = None,
    hidden_state_index: int = -1,
    eos_token_id: int | None = None,
    suppress_first_token_ids: list[int] | tuple[int, ...] | None = None,
) -> TGVFGenerationEvent:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if device is None:
        device = _infer_model_device(model)
    model_inputs = _move_tensors(dict(inputs), device)
    outputs = model(
        **model_inputs,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    rope_deltas = getattr(outputs, "rope_deltas", None)
    if rope_deltas is not None:
        model_inputs["rope_deltas"] = rope_deltas
    return _generate_until_event_loop(
        model=model,
        tokenizer=tokenizer,
        logits=outputs.logits,
        past_key_values=outputs.past_key_values,
        attention_mask=model_inputs.get("attention_mask"),
        input_ids=model_inputs["input_ids"],
        cache_position=model_inputs.get("cache_position"),
        base_inputs=model_inputs,
        max_new_tokens=max_new_tokens,
        hidden_state_index=hidden_state_index,
        eos_token_id=eos_token_id,
        suppress_first_token_ids=suppress_first_token_ids,
        debug_metadata={"event_source": "prefill_inputs"},
    )


@torch.no_grad()
def generate_until_event_from_state(
    model: Any,
    tokenizer: Any,
    generation_state: Any,
    *,
    max_new_tokens: int = 128,
    hidden_state_index: int = -1,
    eos_token_id: int | None = None,
    suppress_first_token_ids: list[int] | tuple[int, ...] | None = None,
) -> TGVFGenerationEvent:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    model_kwargs = dict(_state_attr(generation_state, "model_kwargs") or {})
    if _state_attr(generation_state, "image_grid_thw") is not None:
        model_kwargs["image_grid_thw"] = _state_attr(generation_state, "image_grid_thw")
    if _state_attr(generation_state, "video_grid_thw") is not None:
        model_kwargs["video_grid_thw"] = _state_attr(generation_state, "video_grid_thw")
    return _generate_until_event_loop(
        model=model,
        tokenizer=tokenizer,
        logits=_state_attr(generation_state, "last_logits"),
        past_key_values=_state_attr(generation_state, "past_key_values"),
        attention_mask=_state_attr(generation_state, "attention_mask"),
        input_ids=_state_attr(generation_state, "input_ids"),
        cache_position=_state_attr(generation_state, "cache_position"),
        base_inputs=model_kwargs,
        max_new_tokens=max_new_tokens,
        hidden_state_index=hidden_state_index,
        eos_token_id=eos_token_id,
        suppress_first_token_ids=suppress_first_token_ids,
        debug_metadata=dict(_state_attr(generation_state, "debug_metadata") or {}),
    )


def _generate_until_event_loop(
    *,
    model: Any,
    tokenizer: Any,
    logits: torch.Tensor,
    past_key_values: Any,
    attention_mask: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    cache_position: torch.Tensor | None,
    base_inputs: dict[str, Any],
    max_new_tokens: int,
    hidden_state_index: int,
    eos_token_id: int | None,
    suppress_first_token_ids: list[int] | tuple[int, ...] | None,
    debug_metadata: dict[str, Any] | None = None,
) -> TGVFGenerationEvent:
    start_marker_ids = _marker_ids(tokenizer, FOVEATE_START)
    end_marker_ids = _marker_ids(tokenizer, FOVEATE_END)
    if not start_marker_ids or not end_marker_ids:
        raise ValueError("Foveation markers must tokenize to non-empty id sequences")
    if logits is None:
        raise ValueError("generation_state.last_logits is required")

    full_input_ids = input_ids
    generated_ids: list[int] = []
    generated_hidden_states: list[torch.Tensor] = []
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
        full_input_ids = (
            next_token
            if full_input_ids is None
            else torch.cat([full_input_ids.to(next_token.device), next_token], dim=-1)
        )
        attention_mask = _append_attention(attention_mask, next_token)
        step_inputs = _prepare_decode_step(
            model,
            full_input_ids=full_input_ids,
            next_token=next_token,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            base_inputs=base_inputs,
            cache_position=cache_position,
            generated_token_count=len(generated_ids) + 1,
        )
        outputs = model(
            **step_inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        token_hidden = outputs.hidden_states[hidden_state_index][0, -1].detach()
        generated_ids.append(token_id)
        generated_hidden_states.append(token_hidden)
        past_key_values = outputs.past_key_values
        logits = outputs.logits
        cache_position = step_inputs.get("cache_position")

        capture = _find_completed_request(
            generated_ids,
            start_marker_ids,
            end_marker_ids,
            tokenizer=tokenizer,
        )
        if capture is not None:
            capture_result = _build_capture_result(
                tokenizer=tokenizer,
                generated_ids=generated_ids,
                generated_hidden_states=generated_hidden_states,
                capture=capture,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                cache_position=cache_position,
                input_ids=full_input_ids,
                last_logits=logits,
                base_inputs=base_inputs,
                stop_reason="foveation_end_marker",
                capture_found=True,
            )
            return _event_from_capture_result(
                capture_result,
                event_type="foveate",
                debug_metadata=debug_metadata,
            )

        final_answer = _find_final_answer(_decode(tokenizer, generated_ids))
        if final_answer is not None:
            return _build_non_foveation_event(
                tokenizer=tokenizer,
                event_type="final_answer",
                generated_ids=generated_ids,
                generated_hidden_states=generated_hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                cache_position=cache_position,
                input_ids=full_input_ids,
                last_logits=logits,
                base_inputs=base_inputs,
                stop_reason="final_answer",
                final_answer=final_answer,
                debug_metadata=debug_metadata,
            )

        if eos_token_id is not None and token_id == eos_token_id:
            return _build_non_foveation_event(
                tokenizer=tokenizer,
                event_type="eos_token",
                generated_ids=generated_ids,
                generated_hidden_states=generated_hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                cache_position=cache_position,
                input_ids=full_input_ids,
                last_logits=logits,
                base_inputs=base_inputs,
                stop_reason="eos_token",
                final_answer=None,
                debug_metadata=debug_metadata,
            )

    return _build_non_foveation_event(
        tokenizer=tokenizer,
        event_type="max_new_tokens",
        generated_ids=generated_ids,
        generated_hidden_states=generated_hidden_states,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        cache_position=cache_position,
        input_ids=full_input_ids,
        last_logits=logits,
        base_inputs=base_inputs,
        stop_reason="max_new_tokens",
        final_answer=_find_final_answer(_decode(tokenizer, generated_ids)),
        debug_metadata=debug_metadata,
    )


def _event_from_capture_result(
    result: TGVFCaptureResult,
    *,
    event_type: GenerationEventType,
    debug_metadata: dict[str, Any] | None,
) -> TGVFGenerationEvent:
    return TGVFGenerationEvent(
        event_type=event_type,
        target_text=result.target_text,
        raw_target_text=result.raw_target_text,
        target_token_ids=result.target_token_ids,
        target_hidden_states=result.target_hidden_states,
        target_token_start=result.target_token_start,
        target_token_end=result.target_token_end,
        generated_ids=result.generated_ids,
        generated_text=result.generated_text,
        generated_hidden_states=result.generated_hidden_states,
        past_key_values=result.past_key_values,
        attention_mask=result.attention_mask,
        cache_position=result.cache_position,
        input_ids=result.input_ids,
        last_logits=result.last_logits,
        image_grid_thw=result.image_grid_thw,
        video_grid_thw=result.video_grid_thw,
        model_kwargs=result.model_kwargs,
        stop_reason=result.stop_reason,
        capture_found=result.capture_found,
        final_answer=None,
        debug_metadata=dict(debug_metadata or {}),
        whitespace_trim_note=result.whitespace_trim_note,
    )


def _build_non_foveation_event(
    *,
    tokenizer: Any,
    event_type: GenerationEventType,
    generated_ids: list[int],
    generated_hidden_states: list[torch.Tensor],
    past_key_values: Any,
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    last_logits: torch.Tensor | None,
    base_inputs: dict[str, Any],
    stop_reason: str,
    final_answer: str | None,
    debug_metadata: dict[str, Any] | None,
) -> TGVFGenerationEvent:
    all_hidden_states = _stack_hidden_states(generated_hidden_states)
    hidden_size = all_hidden_states.shape[-1] if all_hidden_states.ndim == 2 else 0
    empty_hidden_states = all_hidden_states.new_empty((0, hidden_size))
    return TGVFGenerationEvent(
        event_type=event_type,
        target_text="",
        raw_target_text="",
        target_token_ids=[],
        target_hidden_states=empty_hidden_states,
        target_token_start=None,
        target_token_end=None,
        generated_ids=list(generated_ids),
        generated_text=_decode(tokenizer, generated_ids),
        generated_hidden_states=all_hidden_states,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        cache_position=cache_position,
        input_ids=input_ids,
        last_logits=last_logits,
        image_grid_thw=base_inputs.get("image_grid_thw"),
        video_grid_thw=base_inputs.get("video_grid_thw"),
        model_kwargs=_resume_model_kwargs(base_inputs),
        stop_reason=stop_reason,
        capture_found=False,
        final_answer=final_answer,
        debug_metadata=dict(debug_metadata or {}),
    )


def _find_final_answer(text: str) -> str | None:
    matches = list(_FINAL_ANSWER_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).upper()


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
    generation_kwargs = {
        "past_key_values": past_key_values,
        "attention_mask": attention_mask,
        "use_cache": True,
        "pixel_values": base_inputs.get("pixel_values"),
        "pixel_values_videos": base_inputs.get("pixel_values_videos"),
        "image_grid_thw": base_inputs.get("image_grid_thw"),
        "video_grid_thw": base_inputs.get("video_grid_thw"),
        "is_first_iteration": False,
        "next_sequence_length": 1,
    }
    for key in ("rope_deltas", "mm_token_type_ids", "position_ids"):
        if key in base_inputs:
            generation_kwargs[key] = base_inputs[key]
    if cache_position is not None:
        generation_kwargs["cache_position"] = cache_position

    position_ids = _decode_position_ids(
        attention_mask=attention_mask,
        next_token=next_token,
        past_key_values=past_key_values,
        rope_deltas=base_inputs.get("rope_deltas"),
        generated_token_count=generated_token_count,
    )
    if position_ids is not None:
        generation_kwargs["position_ids"] = position_ids

    if hasattr(model, "prepare_inputs_for_generation"):
        return model.prepare_inputs_for_generation(
            full_input_ids,
            **generation_kwargs,
        )

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
    return base_position.to(next_token.device) + delta.view(1, batch_size, 1)


def _find_completed_request(
    generated_ids: list[int],
    start_marker_ids: list[int],
    end_marker_ids: list[int],
    *,
    tokenizer: Any | None = None,
) -> _CompletedRequest | None:
    start_index = _find_subsequence(generated_ids, start_marker_ids)
    if start_index is not None:
        target_start = start_index + len(start_marker_ids)
        relative_end = _find_subsequence(generated_ids[target_start:], end_marker_ids)
        if relative_end is not None:
            target_end = target_start + relative_end
            return _CompletedRequest(target_start, target_end)

    if tokenizer is None:
        return None
    return _find_completed_request_from_decoded_text(generated_ids, tokenizer)


def _find_completed_request_from_decoded_text(
    generated_ids: list[int],
    tokenizer: Any,
) -> _CompletedRequest | None:
    generated_text = _decode(tokenizer, generated_ids)
    if FOVEATE_START not in generated_text or FOVEATE_END not in generated_text:
        return None
    spans = parse_foveation_spans(generated_text)
    if not spans:
        return None

    span = spans[0]
    offsets = _decoded_token_offsets(tokenizer, generated_ids)
    token_indices = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_start < span.end_char and token_end > span.start_char
    ]
    if not token_indices:
        return None

    return _CompletedRequest(
        target_token_start=token_indices[0],
        target_token_end=token_indices[-1] + 1,
        raw_target_text=generated_text[span.start_char : span.end_char],
        target_text=span.target_text,
        boundary_note=(
            "Recovered target span from decoded text because tokenizer merged a marker "
            "boundary with a neighboring token; edge hidden states may include marker text."
        ),
    )


def _build_capture_result(
    *,
    tokenizer: Any,
    generated_ids: list[int],
    generated_hidden_states: list[torch.Tensor],
    capture: _CompletedRequest,
    past_key_values: Any,
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    last_logits: torch.Tensor | None,
    base_inputs: dict[str, Any],
    stop_reason: str,
    capture_found: bool,
) -> TGVFCaptureResult:
    target_start = capture.target_token_start
    target_end = capture.target_token_end
    trim_start, trim_end, whitespace_trim_note = _trim_whitespace_token_edges(
        tokenizer,
        generated_ids,
        target_start,
        target_end,
    )
    if capture.boundary_note is not None:
        whitespace_trim_note = (
            capture.boundary_note
            if whitespace_trim_note is None
            else f"{whitespace_trim_note} {capture.boundary_note}"
        )
    target_token_ids = generated_ids[trim_start:trim_end]
    target_hidden_states = _stack_hidden_states(generated_hidden_states[trim_start:trim_end])
    raw_target_ids = generated_ids[target_start:target_end]
    raw_target_text = capture.raw_target_text if capture.raw_target_text is not None else _decode(tokenizer, raw_target_ids)
    target_text = capture.target_text if capture.target_text is not None else raw_target_text.strip()

    return TGVFCaptureResult(
        target_text=target_text,
        raw_target_text=raw_target_text,
        target_token_ids=target_token_ids,
        target_hidden_states=target_hidden_states,
        target_token_start=trim_start,
        target_token_end=trim_end,
        generated_ids=list(generated_ids),
        generated_text=_decode(tokenizer, generated_ids),
        generated_hidden_states=_stack_hidden_states(generated_hidden_states),
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        cache_position=cache_position,
        input_ids=input_ids,
        last_logits=last_logits,
        image_grid_thw=base_inputs.get("image_grid_thw"),
        video_grid_thw=base_inputs.get("video_grid_thw"),
        model_kwargs=_resume_model_kwargs(base_inputs),
        stop_reason=stop_reason,
        capture_found=capture_found,
        whitespace_trim_note=whitespace_trim_note,
    )


def _build_empty_result(
    *,
    tokenizer: Any,
    generated_ids: list[int],
    generated_hidden_states: list[torch.Tensor],
    past_key_values: Any,
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    last_logits: torch.Tensor | None,
    base_inputs: dict[str, Any],
    stop_reason: str,
) -> TGVFCaptureResult:
    all_hidden_states = _stack_hidden_states(generated_hidden_states)
    hidden_size = all_hidden_states.shape[-1] if all_hidden_states.ndim == 2 else 0
    empty_hidden_states = all_hidden_states.new_empty((0, hidden_size))

    return TGVFCaptureResult(
        target_text="",
        raw_target_text="",
        target_token_ids=[],
        target_hidden_states=empty_hidden_states,
        target_token_start=None,
        target_token_end=None,
        generated_ids=list(generated_ids),
        generated_text=_decode(tokenizer, generated_ids),
        generated_hidden_states=all_hidden_states,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        cache_position=cache_position,
        input_ids=input_ids,
        last_logits=last_logits,
        image_grid_thw=base_inputs.get("image_grid_thw"),
        video_grid_thw=base_inputs.get("video_grid_thw"),
        model_kwargs=_resume_model_kwargs(base_inputs),
        stop_reason=stop_reason,
        capture_found=False,
    )


def _trim_whitespace_token_edges(
    tokenizer: Any,
    generated_ids: list[int],
    target_start: int,
    target_end: int,
) -> tuple[int, int, str | None]:
    trim_start = target_start
    trim_end = target_end

    while trim_start < trim_end and _decode(tokenizer, [generated_ids[trim_start]]).isspace():
        trim_start += 1
    while trim_end > trim_start and _decode(tokenizer, [generated_ids[trim_end - 1]]).isspace():
        trim_end -= 1

    if trim_start != target_start or trim_end != target_end:
        return trim_start, trim_end, "Trimmed standalone whitespace tokens at target edges."

    raw_text = _decode(tokenizer, generated_ids[target_start:target_end])
    if raw_text != raw_text.strip():
        return (
            trim_start,
            trim_end,
            "Target text was stripped for debug output, but edge whitespace may be merged "
            "inside non-whitespace tokenizer tokens.",
        )

    return trim_start, trim_end, None


def _resume_model_kwargs(base_inputs: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "image_grid_thw",
        "video_grid_thw",
        "rope_deltas",
        "mm_token_type_ids",
        "position_ids",
    )
    return {key: base_inputs[key] for key in keys if key in base_inputs}


def _marker_ids(tokenizer: Any, marker: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(marker, add_special_tokens=False)
    else:
        ids = tokenizer(marker, add_special_tokens=False)["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _decode(tokenizer: Any, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _find_subsequence(values: list[int], pattern: list[int]) -> int | None:
    if not pattern or len(pattern) > len(values):
        return None
    for index in range(len(values) - len(pattern) + 1):
        if values[index : index + len(pattern)] == pattern:
            return index
    return None



def _decoded_token_offsets(tokenizer: Any, token_ids: list[int]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token_id in token_ids:
        token_text = _decode(tokenizer, [token_id])
        token_end = cursor + len(token_text)
        offsets.append((cursor, token_end))
        cursor = token_end
    return offsets

def _stack_hidden_states(hidden_states: list[torch.Tensor]) -> torch.Tensor:
    if not hidden_states:
        return torch.empty((0, 0))
    return torch.stack([hidden_state.detach().cpu() for hidden_state in hidden_states], dim=0)


def _append_attention(
    attention_mask: torch.Tensor | None,
    next_token: torch.Tensor,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    next_attention = torch.ones(
        (attention_mask.shape[0], 1),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    return torch.cat([attention_mask, next_attention], dim=-1)


def _move_tensors(inputs: dict[str, Any], device: torch.device | str | None) -> dict[str, Any]:
    if device is None:
        return inputs
    return {
        key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()
    }


def _state_attr(state: Any, name: str) -> Any:
    if isinstance(state, dict):
        return state.get(name)
    return getattr(state, name, None)


def _infer_model_device(model: Any) -> torch.device | None:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return None
