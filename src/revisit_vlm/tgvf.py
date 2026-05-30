from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

FOVEATE_START = "<|foveate|>"
FOVEATE_END = "<|/foveate|>"


@dataclass(frozen=True)
class FoveationSpan:
    """Character span for a stripped TGVF target."""

    target_text: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class FoveationTokenSpan:
    """Token span for a stripped TGVF target."""

    target_text: str
    start_char: int
    end_char: int
    token_start: int
    token_end: int


def parse_foveation_spans(text: str) -> list[FoveationSpan]:
    """Extract stripped target spans from ``<|foveate|> xxx <|/foveate|>`` markers."""

    spans: list[FoveationSpan] = []
    search_from = 0

    while True:
        start_marker = text.find(FOVEATE_START, search_from)
        if start_marker == -1:
            break

        inner_start = start_marker + len(FOVEATE_START)
        end_marker = text.find(FOVEATE_END, inner_start)
        next_start = text.find(FOVEATE_START, inner_start)

        if end_marker == -1:
            break
        if next_start != -1 and next_start < end_marker:
            search_from = next_start
            continue

        raw_target = text[inner_start:end_marker]
        leading_ws = len(raw_target) - len(raw_target.lstrip())
        trailing_ws = len(raw_target.rstrip())
        target_start = inner_start + leading_ws
        target_end = inner_start + trailing_ws
        target_text = text[target_start:target_end]

        if target_text:
            spans.append(
                FoveationSpan(
                    target_text=target_text,
                    start_char=target_start,
                    end_char=target_end,
                )
            )

        search_from = end_marker + len(FOVEATE_END)

    return spans


def find_foveation_token_spans(text: str, tokenizer: Any) -> list[FoveationTokenSpan]:
    """Map TGVF target character spans to token spans using tokenizer offsets."""

    spans = parse_foveation_spans(text)
    if not spans:
        return []

    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = _first_row_offsets(encoded["offset_mapping"])

    token_spans: list[FoveationTokenSpan] = []
    for span in spans:
        token_indices = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_start < span.end_char and token_end > span.start_char
        ]
        if token_indices:
            token_spans.append(
                FoveationTokenSpan(
                    target_text=span.target_text,
                    start_char=span.start_char,
                    end_char=span.end_char,
                    token_start=token_indices[0],
                    token_end=token_indices[-1] + 1,
                )
            )

    return token_spans


@torch.no_grad()
def extract_foveation_queries(
    text: str,
    tokenizer: Any,
    model: Any,
    *,
    hidden_state_index: int = -1,
    device: torch.device | str | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> list[torch.Tensor]:
    """Return hidden states for each stripped TGVF target span.

    Each returned tensor has shape ``[target_token_count, hidden_size]``.
    """

    spans = parse_foveation_spans(text)
    if not spans:
        return []

    encode_kwargs = dict(tokenizer_kwargs or {})
    encode_kwargs.update(
        {
            "return_offsets_mapping": True,
            "return_tensors": "pt",
        }
    )
    encoded = tokenizer(text, **encode_kwargs)
    offsets = _first_row_offsets(encoded["offset_mapping"])

    token_spans = [_to_token_span(span, offsets) for span in spans]
    token_spans = [span for span in token_spans if span is not None]
    if not token_spans:
        return []

    model_inputs = {
        key: value
        for key, value in encoded.items()
        if key != "offset_mapping"
    }
    if device is None:
        device = _infer_model_device(model)
    if device is not None:
        model_inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in model_inputs.items()
        }

    forward_kwargs = dict(model_kwargs or {})
    forward_kwargs["output_hidden_states"] = True
    outputs = model(**model_inputs, **forward_kwargs)
    hidden_states = outputs.hidden_states[hidden_state_index]

    return [
        hidden_states[0, span.token_start : span.token_end].detach().cpu()
        for span in token_spans
    ]


def _to_token_span(
    span: FoveationSpan,
    offsets: list[tuple[int, int]],
) -> FoveationTokenSpan | None:
    token_indices = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_start < span.end_char and token_end > span.start_char
    ]
    if not token_indices:
        return None

    return FoveationTokenSpan(
        target_text=span.target_text,
        start_char=span.start_char,
        end_char=span.end_char,
        token_start=token_indices[0],
        token_end=token_indices[-1] + 1,
    )


def _first_row_offsets(offset_mapping: Any) -> list[tuple[int, int]]:
    if isinstance(offset_mapping, torch.Tensor):
        offset_mapping = offset_mapping[0].tolist()
    elif offset_mapping and isinstance(offset_mapping[0], list):
        offset_mapping = offset_mapping[0]

    return [(int(start), int(end)) for start, end in offset_mapping]


def _infer_model_device(model: Any) -> torch.device | None:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return None
