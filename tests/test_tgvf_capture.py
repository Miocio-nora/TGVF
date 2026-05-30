from __future__ import annotations

from types import SimpleNamespace

import torch

from revisit_vlm.tgvf import FOVEATE_END, FOVEATE_START
from revisit_vlm.tgvf_capture import capture_tgvf_single_pass_from_inputs

START_IDS = [11, 12, 13]
END_IDS = [14, 12, 13]
TARGET_IDS = [21, 22, 23]
SPACE_ID = 30
EOS_ID = 99


class FakeTokenizer:
    pieces = {
        11: "<|",
        12: "foveate",
        13: "|>",
        14: "<|/",
        21: "the",
        22: " small",
        23: " date",
        24: ".<|/",
        30: " ",
        40: "hello",
        41: " world",
        99: "<eos>",
    }

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        if text == FOVEATE_START:
            return START_IDS
        if text == FOVEATE_END:
            return END_IDS
        raise AssertionError(f"Unexpected marker text: {text}")

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert not skip_special_tokens
        assert not clean_up_tokenization_spaces
        return "".join(self.pieces[token_id] for token_id in token_ids)


class FakeSinglePassModel:
    def __init__(self, generated_sequence: list[int], *, hidden_size: int = 4) -> None:
        self.generated_sequence = generated_sequence
        self.hidden_size = hidden_size
        self.decode_calls = 0
        self.forward_input_lengths: list[int] = []

    def __call__(
        self,
        input_ids: torch.Tensor,
        *,
        past_key_values: object | None = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
        **_: object,
    ) -> SimpleNamespace:
        assert use_cache
        assert output_hidden_states
        assert return_dict
        self.forward_input_lengths.append(input_ids.shape[-1])

        if past_key_values is None:
            next_id = self.generated_sequence[0]
            hidden = torch.zeros((1, input_ids.shape[-1], self.hidden_size))
        else:
            assert input_ids.shape[-1] == 1
            token_id = int(input_ids[0, 0].item())
            hidden = torch.full((1, 1, self.hidden_size), float(token_id))
            self.decode_calls += 1
            next_index = self.decode_calls
            next_id = (
                self.generated_sequence[next_index]
                if next_index < len(self.generated_sequence)
                else EOS_ID
            )

        logits = torch.full((1, 1, 128), -1000.0)
        logits[0, -1, next_id] = 1000.0
        return SimpleNamespace(
            logits=logits,
            hidden_states=(hidden,),
            past_key_values={"decode_calls": self.decode_calls},
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        *,
        past_key_values: object,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = True,
        **_: object,
    ) -> dict[str, object]:
        return {
            "input_ids": input_ids[:, -1:],
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    def parameters(self):
        return iter(())


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.ones((1, 5), dtype=torch.long),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }


def test_detects_single_foveation_request() -> None:
    result = capture_tgvf_single_pass_from_inputs(
        FakeSinglePassModel(START_IDS + TARGET_IDS + END_IDS),
        FakeTokenizer(),
        _inputs(),
    )

    assert result.capture_found
    assert result.generated_text == "<|foveate|>the small date<|/foveate|>"
    assert result.target_text == "the small date"
    assert result.stop_reason == "foveation_end_marker"


def test_detects_markers_that_tokenize_into_multiple_ids() -> None:
    result = capture_tgvf_single_pass_from_inputs(
        FakeSinglePassModel(START_IDS + TARGET_IDS + END_IDS),
        FakeTokenizer(),
        _inputs(),
    )

    assert result.generated_ids[: len(START_IDS)] == START_IDS
    assert result.generated_ids[-len(END_IDS) :] == END_IDS


def test_excludes_marker_tokens_from_target_hidden_states() -> None:
    result = capture_tgvf_single_pass_from_inputs(
        FakeSinglePassModel(START_IDS + TARGET_IDS + END_IDS),
        FakeTokenizer(),
        _inputs(),
    )

    assert result.target_token_ids == TARGET_IDS
    assert result.target_token_start == len(START_IDS)
    assert result.target_token_end == len(START_IDS) + len(TARGET_IDS)
    assert result.target_hidden_states[:, 0].tolist() == [
        float(token_id) for token_id in TARGET_IDS
    ]


def test_stops_exactly_after_end_marker() -> None:
    extra_id_after_end = 40
    result = capture_tgvf_single_pass_from_inputs(
        FakeSinglePassModel(START_IDS + TARGET_IDS + END_IDS + [extra_id_after_end]),
        FakeTokenizer(),
        _inputs(),
    )

    assert result.generated_ids == START_IDS + TARGET_IDS + END_IDS
    assert extra_id_after_end not in result.generated_ids


def test_returns_empty_capture_if_no_complete_request_appears() -> None:
    result = capture_tgvf_single_pass_from_inputs(
        FakeSinglePassModel([40, 41, EOS_ID]),
        FakeTokenizer(),
        _inputs(),
        eos_token_id=EOS_ID,
    )

    assert not result.capture_found
    assert result.target_text == ""
    assert result.target_token_ids == []
    assert result.target_hidden_states.shape == (0, 4)


def test_ignores_incomplete_start_marker_without_end_marker() -> None:
    result = capture_tgvf_single_pass_from_inputs(
        FakeSinglePassModel(START_IDS + TARGET_IDS + [EOS_ID]),
        FakeTokenizer(),
        _inputs(),
        eos_token_id=EOS_ID,
    )

    assert not result.capture_found
    assert result.target_token_start is None
    assert result.target_token_end is None


def test_keeps_one_hidden_state_per_generated_token() -> None:
    generated = START_IDS + TARGET_IDS + END_IDS
    result = capture_tgvf_single_pass_from_inputs(
        FakeSinglePassModel(generated),
        FakeTokenizer(),
        _inputs(),
    )

    assert result.generated_ids == generated
    assert result.generated_hidden_states.shape[0] == len(generated)


def test_target_hidden_state_count_matches_target_token_count() -> None:
    result = capture_tgvf_single_pass_from_inputs(
        FakeSinglePassModel(START_IDS + [SPACE_ID] + TARGET_IDS + [SPACE_ID] + END_IDS),
        FakeTokenizer(),
        _inputs(),
    )

    assert result.target_text == "the small date"
    assert result.raw_target_text == " the small date "
    assert result.target_token_ids == TARGET_IDS
    assert result.target_hidden_states.shape[0] == len(result.target_token_ids)



def test_detects_end_marker_when_tokenizer_merges_marker_boundary() -> None:
    merged_period_end_start = 24
    generated = START_IDS + TARGET_IDS + [merged_period_end_start, 12, 13]
    result = capture_tgvf_single_pass_from_inputs(
        FakeSinglePassModel(generated),
        FakeTokenizer(),
        _inputs(),
    )

    assert result.capture_found
    assert result.generated_text == "<|foveate|>the small date.<|/foveate|>"
    assert result.target_text == "the small date."
    assert result.raw_target_text == "the small date."
    assert result.target_token_ids == TARGET_IDS + [merged_period_end_start]
    assert result.target_hidden_states.shape[0] == len(result.target_token_ids)
    assert result.whitespace_trim_note is not None
    assert "Recovered target span from decoded text" in result.whitespace_trim_note

def test_does_not_run_second_full_forward_over_prompt_plus_generated_span() -> None:
    prompt_len = _inputs()["input_ids"].shape[-1]
    model = FakeSinglePassModel(START_IDS + TARGET_IDS + END_IDS)

    capture_tgvf_single_pass_from_inputs(model, FakeTokenizer(), _inputs())

    assert model.forward_input_lengths[0] == prompt_len
    assert model.forward_input_lengths[1:] == [1] * (
        len(START_IDS) + len(TARGET_IDS) + len(END_IDS)
    )
