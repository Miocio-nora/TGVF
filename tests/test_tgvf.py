from __future__ import annotations

import re
from types import SimpleNamespace

import torch

from revisit_vlm.tgvf import (
    FoveationTokenSpan,
    extract_foveation_queries,
    find_foveation_token_spans,
    parse_foveation_spans,
)


class FakeTokenizer:
    def __call__(
        self,
        text: str,
        *,
        return_offsets_mapping: bool = False,
        return_tensors: str | None = None,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"

        offsets = []
        input_ids = []
        for token_id, match in enumerate(re.finditer(r"\S+", text), start=1):
            offsets.append(match.span())
            input_ids.append(token_id)

        encoded = {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
        }
        if return_offsets_mapping:
            encoded["offset_mapping"] = torch.tensor([offsets], dtype=torch.long)
        return encoded


class FakeModel:
    def __call__(
        self,
        input_ids: torch.Tensor,
        *,
        output_hidden_states: bool = False,
        **_: object,
    ) -> SimpleNamespace:
        assert output_hidden_states
        batch_size, seq_len = input_ids.shape
        hidden_size = 4
        positions = torch.arange(seq_len, dtype=torch.float32).view(1, seq_len, 1)
        features = torch.arange(hidden_size, dtype=torch.float32).view(1, 1, hidden_size)
        hidden = positions + features
        return SimpleNamespace(hidden_states=(hidden.expand(batch_size, -1, -1),))

    def parameters(self):
        return iter(())


def test_single_foveation_span() -> None:
    spans = parse_foveation_spans("Look at <|foveate|> something blue <|/foveate|> now.")

    assert len(spans) == 1
    assert spans[0].target_text == "something blue"


def test_multiple_foveation_spans() -> None:
    spans = parse_foveation_spans(
        "<|foveate|> red sign <|/foveate|> and <|foveate|> blue car <|/foveate|>"
    )

    assert [span.target_text for span in spans] == ["red sign", "blue car"]


def test_no_span_returns_empty_list() -> None:
    assert parse_foveation_spans("There is no target here.") == []
    assert extract_foveation_queries("There is no target here.", FakeTokenizer(), FakeModel()) == []


def test_start_marker_without_end_marker_is_ignored() -> None:
    assert parse_foveation_spans("Broken <|foveate|> target") == []


def test_leading_and_trailing_whitespace_inside_markers_is_stripped() -> None:
    spans = parse_foveation_spans("<|foveate|>   small text\t\n <|/foveate|>")

    assert len(spans) == 1
    assert spans[0].target_text == "small text"


def test_extracted_hidden_state_shape_matches_token_span_length() -> None:
    text = "Look at <|foveate|> something blue <|/foveate|> now."
    tokenizer = FakeTokenizer()

    token_spans = find_foveation_token_spans(text, tokenizer)
    queries = extract_foveation_queries(text, tokenizer, FakeModel())

    assert token_spans == [
        FoveationTokenSpan(
            target_text="something blue",
            start_char=text.index("something"),
            end_char=text.index("blue") + len("blue"),
            token_start=3,
            token_end=5,
        )
    ]
    assert len(queries) == 1
    assert queries[0].shape == (token_spans[0].token_end - token_spans[0].token_start, 4)
