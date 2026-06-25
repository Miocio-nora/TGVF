from __future__ import annotations

from types import SimpleNamespace

import torch

from revisit_vlm import qwen3_vl_tgvf as qwen3_tgvf
from revisit_vlm.qwen3_vl_tgvf import (
    FOCUS_END,
    FOCUS_START,
    PROTOCOL_C_FOCUS_END,
    PROTOCOL_C_FOCUS_START,
    PROTOCOL_C_SPECIAL_TOKENS,
    PROTOCOL_C_THINKING_SPECIAL,
    PROTOCOL_C_TOOL_OBSERVATION,
    _chunk_position_ids_inherit_source_visual_positions,
    _text_positions_are_1d,
    _visual_position_ids_equal_source,
    capture_focus_single_pass_from_inputs_qwen3,
    continue_generation_qwen3,
    ensure_protocol_c_special_tokens,
    is_generic_target,
    parse_v3_action,
    render_focus_action_text,
    render_focus_readout_answer_text,
    render_no_focus_output_text,
)


FOCUS_START_IDS = [101, 102, 103]
FOCUS_END_IDS = [104, 102, 103]
PROTOCOL_C_FOCUS_START_IDS = [501]
PROTOCOL_C_FOCUS_END_IDS = [502]
TARGET_IDS = [201, 202, 203, 204]
SPACE_ID = 210
EOS_ID = 999


class FakeQwen3Tokenizer:
    pieces = {
        101: "<",
        102: "FOCUS",
        103: ">",
        104: "</",
        201: "the",
        202: " small",
        203: " date-like",
        204: " text",
        210: " ",
        301: "<EVIDENCE_STATE>",
        302: "need_local_visual_evidence",
        303: "</EVIDENCE_STATE>",
        401: "hello",
        402: " world",
        501: "<|focus_start|>",
        502: "<|focus_end|>",
        999: "<|im_end|>",
    }

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        if text == FOCUS_START:
            return FOCUS_START_IDS
        if text == FOCUS_END:
            return FOCUS_END_IDS
        if text == PROTOCOL_C_FOCUS_START:
            return PROTOCOL_C_FOCUS_START_IDS
        if text == PROTOCOL_C_FOCUS_END:
            return PROTOCOL_C_FOCUS_END_IDS
        if text == "<|im_end|>":
            return [EOS_ID]
        raise AssertionError(f"Unexpected encode text: {text}")

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert not skip_special_tokens
        assert not clean_up_tokenization_spaces
        return "".join(self.pieces[int(token_id)] for token_id in token_ids)


class FakeQwen3Model:
    def __init__(self, generated_sequence: list[int], *, hidden_size: int = 6) -> None:
        self.generated_sequence = list(generated_sequence)
        self.hidden_size = hidden_size
        self.decode_calls = 0
        self.prepare_calls = 0
        self.forward_input_lengths: list[int] = []

    def __call__(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        past_key_values: object | None = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
        **_: object,
    ) -> SimpleNamespace:
        assert use_cache
        assert return_dict
        assert input_ids is not None
        self.forward_input_lengths.append(int(input_ids.shape[-1]))
        if past_key_values is None:
            next_id = self.generated_sequence[0]
            hidden = torch.zeros((1, input_ids.shape[-1], self.hidden_size))
        else:
            assert input_ids.shape[-1] == 1
            token_id = int(input_ids[0, 0].item())
            hidden = torch.full((1, 1, self.hidden_size), float(token_id))
            self.decode_calls += 1
            next_id = (
                self.generated_sequence[self.decode_calls]
                if self.decode_calls < len(self.generated_sequence)
                else EOS_ID
            )
        logits = torch.full((1, 1, 1200), -1000.0)
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
        self.prepare_calls += 1
        return {
            "input_ids": input_ids[:, -1:],
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    def parameters(self):
        return iter(())


class FakeResizableTokenizer:
    def __init__(self) -> None:
        self.vocab = {"<|vision_start|>": 0, "<|vision_end|>": 1, "<think>": 2, "</think>": 3}
        self.additional_special_tokens: list[str] = []
        self.unk_token_id = -1

    def __len__(self) -> int:
        return len(self.vocab)

    def add_special_tokens(self, payload: dict[str, list[str]]) -> int:
        added = 0
        self.additional_special_tokens = list(payload.get("additional_special_tokens", []))
        for token in self.additional_special_tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
        return added

    def add_tokens(self, tokens: list[str], special_tokens: bool = False) -> int:
        assert not special_tokens
        added = 0
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
        return added

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [self.vocab[text]] if text in self.vocab else []


class FakeResizableModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 4, hidden_size: int = 5) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, hidden_size)

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.emb

    def resize_token_embeddings(self, size: int) -> None:
        old = self.emb
        self.emb = torch.nn.Embedding(size, old.embedding_dim)
        with torch.no_grad():
            self.emb.weight[: old.num_embeddings].copy_(old.weight)


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }


def test_v3_parser_accepts_valid_focus_action() -> None:
    parsed = parse_v3_action(
        "<EVIDENCE_STATE>need_local_visual_evidence</EVIDENCE_STATE>\n"
        "<FOCUS>the date-like small text below the barcode</FOCUS>"
    )

    assert parsed.evidence_state == "need_local_visual_evidence"
    assert parsed.focus_valid
    assert parsed.focus_target == "the date-like small text below the barcode"
    assert not parsed.malformed


def test_v3_parser_rejects_malformed_focus_without_close() -> None:
    parsed = parse_v3_action("<FOCUS>the small text below the barcode")

    assert not parsed.focus_valid
    assert parsed.malformed
    assert "missing_closing_focus" in parsed.malformed_reasons


def test_protocol_c_parser_extracts_focus_and_raw_answer_after_think() -> None:
    parsed = parse_v3_action(
        "<think>\nI need to inspect the small mark.\n</think>\n"
        "<|focus_start|>the small mark near the bottom<|focus_end|>\n"
        "<|tgvf_start|>\n<|tgvf_end|>\n"
        "<think>\nThe focused evidence matches option C.\n</think>\n"
        "C",
        protocol=PROTOCOL_C_THINKING_SPECIAL,
    )

    assert parsed.focus_valid
    assert parsed.focus_target == "the small mark near the bottom"
    assert parsed.answer == "C"
    assert parsed.answer_valid
    assert not parsed.malformed


def test_protocol_c_rendering_omits_legacy_tags() -> None:
    focus = render_focus_action_text("the small text below the barcode", protocol=PROTOCOL_C_THINKING_SPECIAL)
    readout = render_focus_readout_answer_text(
        evidence_description="The focused text reads EXP 08/2026.",
        answer="EXP 08/2026",
        protocol=PROTOCOL_C_THINKING_SPECIAL,
    )
    no_focus = render_no_focus_output_text("dog", protocol=PROTOCOL_C_THINKING_SPECIAL)

    rendered = "\n".join([focus, readout, no_focus])
    assert "<|focus_start|>the small text below the barcode<|focus_end|>" in rendered
    assert "<EVIDENCE_STATE>" not in rendered
    assert "<EVIDENCE>" not in rendered
    assert "<ANSWER>" not in rendered


def test_protocol_c_special_tokens_are_added_and_resized() -> None:
    tokenizer = FakeResizableTokenizer()
    model = FakeResizableModel(vocab_size=len(tokenizer))

    info = ensure_protocol_c_special_tokens(tokenizer, model)

    assert info["normal_tokens_added"]
    assert not info["special_tokens_added"]
    assert info["protocol_c_token_registration"] == "normal_added_tokens"
    assert info["tokenizer_resized"]
    assert set(info["protocol_c_special_token_ids"]) == set(PROTOCOL_C_SPECIAL_TOKENS)
    assert model.get_input_embeddings().num_embeddings == len(tokenizer)


def test_generic_target_filter_keeps_specific_visual_cue_object() -> None:
    assert is_generic_target("the object")
    assert not is_generic_target("the small green object near the left edge")


def test_qwen3_capture_excludes_focus_marker_tokens() -> None:
    generated = FOCUS_START_IDS + TARGET_IDS + FOCUS_END_IDS
    result = capture_focus_single_pass_from_inputs_qwen3(
        FakeQwen3Model(generated),
        FakeQwen3Tokenizer(),
        _inputs(),
    )

    assert result.capture_found
    assert result.generated_text == "<FOCUS>the small date-like text</FOCUS>"
    assert result.target_text == "the small date-like text"
    assert result.target_token_ids == TARGET_IDS
    assert result.target_token_start == len(FOCUS_START_IDS)
    assert result.target_token_end == len(FOCUS_START_IDS) + len(TARGET_IDS)
    assert result.target_hidden_states[:, 0].tolist() == [float(token_id) for token_id in TARGET_IDS]
    assert not result.second_full_forward_used


def test_protocol_c_capture_excludes_special_focus_marker_tokens() -> None:
    generated = PROTOCOL_C_FOCUS_START_IDS + TARGET_IDS + PROTOCOL_C_FOCUS_END_IDS
    result = capture_focus_single_pass_from_inputs_qwen3(
        FakeQwen3Model(generated),
        FakeQwen3Tokenizer(),
        _inputs(),
        protocol=PROTOCOL_C_THINKING_SPECIAL,
    )

    assert result.capture_found
    assert result.target_text == "the small date-like text"
    assert result.target_token_ids == TARGET_IDS
    assert result.target_hidden_states[:, 0].tolist() == [float(token_id) for token_id in TARGET_IDS]
    assert not result.second_full_forward_used


def test_toolobs_capture_waits_for_im_end_after_focus_end() -> None:
    generated_without_im_end = PROTOCOL_C_FOCUS_START_IDS + TARGET_IDS + PROTOCOL_C_FOCUS_END_IDS
    incomplete = capture_focus_single_pass_from_inputs_qwen3(
        FakeQwen3Model(generated_without_im_end),
        FakeQwen3Tokenizer(),
        _inputs(),
        max_new_tokens=len(generated_without_im_end),
        protocol=PROTOCOL_C_TOOL_OBSERVATION,
    )
    assert not incomplete.capture_found
    assert "no_complete_focus_span" in incomplete.errors

    generated_with_im_end = generated_without_im_end + [EOS_ID]
    complete = capture_focus_single_pass_from_inputs_qwen3(
        FakeQwen3Model(generated_with_im_end),
        FakeQwen3Tokenizer(),
        _inputs(),
        max_new_tokens=len(generated_with_im_end),
        protocol=PROTOCOL_C_TOOL_OBSERVATION,
    )
    assert complete.capture_found
    assert complete.generated_ids == generated_with_im_end
    assert complete.target_text == "the small date-like text"
    assert complete.target_token_ids == TARGET_IDS


def test_qwen3_capture_trims_standalone_target_whitespace() -> None:
    generated = FOCUS_START_IDS + [SPACE_ID] + TARGET_IDS + [SPACE_ID] + FOCUS_END_IDS
    result = capture_focus_single_pass_from_inputs_qwen3(
        FakeQwen3Model(generated),
        FakeQwen3Tokenizer(),
        _inputs(),
    )

    assert result.target_text == "the small date-like text"
    assert result.target_token_ids == TARGET_IDS
    assert result.target_hidden_states.shape[0] == len(TARGET_IDS)


def test_qwen3_capture_no_second_full_forward_over_prompt_plus_generated() -> None:
    prompt_len = _inputs()["input_ids"].shape[-1]
    generated = FOCUS_START_IDS + TARGET_IDS + FOCUS_END_IDS
    model = FakeQwen3Model(generated)

    capture_focus_single_pass_from_inputs_qwen3(model, FakeQwen3Tokenizer(), _inputs())

    assert model.forward_input_lengths[0] == prompt_len
    assert model.forward_input_lengths[1:] == [1] * len(generated)


def test_qwen3_capture_defaults_to_generate_when_available(monkeypatch) -> None:
    calls = []

    def fake_generate_capture(*_args: object, **_kwargs: object) -> SimpleNamespace:
        calls.append(True)
        return SimpleNamespace(capture_found=True, generated_text="generate branch")

    monkeypatch.delenv("TGVF_QWEN3_CAPTURE_GENERATE", raising=False)
    monkeypatch.setattr(qwen3_tgvf, "_capture_focus_generate_from_inputs_qwen3", fake_generate_capture)
    model = SimpleNamespace(generate=lambda **_kwargs: None)

    result = qwen3_tgvf.capture_focus_single_pass_from_inputs_qwen3(
        model,
        FakeQwen3Tokenizer(),
        _inputs(),
    )

    assert calls
    assert result.generated_text == "generate branch"


def test_qwen3_capture_can_force_action_prefix_without_forcing_target() -> None:
    generated = FOCUS_START_IDS + TARGET_IDS + FOCUS_END_IDS
    result = capture_focus_single_pass_from_inputs_qwen3(
        FakeQwen3Model(generated),
        FakeQwen3Tokenizer(),
        _inputs(),
        forced_prefix_text=FOCUS_START,
    )

    assert result.capture_found
    assert result.target_text == "the small date-like text"
    assert result.target_token_ids == TARGET_IDS
    assert result.generated_ids[: len(FOCUS_START_IDS)] == FOCUS_START_IDS
    assert not result.second_full_forward_used


def test_continue_generation_stops_on_repetitive_tail(monkeypatch) -> None:
    monkeypatch.setenv("TGVF_BLOCK_FOCUS_IN_CONTINUATION", "0")
    model = FakeQwen3Model([402] * 20)
    logits = torch.full((1, 1, 1200), -1000.0)
    logits[0, -1, 402] = 1000.0
    state = SimpleNamespace(
        last_logits=logits,
        past_key_values={"decode_calls": 0},
        attention_mask=torch.ones((1, 4), dtype=torch.long),
        input_ids=torch.tensor([[1, 2, 3, 4]]),
    )

    result = continue_generation_qwen3(
        model,
        FakeQwen3Tokenizer(),
        state,
        max_new_tokens=20,
        eos_token_id=EOS_ID,
    )

    assert result.stop_reason == "repetition"
    assert len(result.generated_ids) < 20
    assert model.prepare_calls == len(result.generated_ids)


def test_inherited_source_visual_positions_replace_only_visual_span() -> None:
    attention_mask = torch.ones((1, 12), dtype=torch.long)
    source_positions = torch.tensor(
        [
            [2, 2],
            [5, 5],
            [7, 8],
        ],
        dtype=torch.long,
    )

    position_ids = _chunk_position_ids_inherit_source_visual_positions(
        attention_mask=attention_mask,
        chunk_length=6,
        visual_token_start=2,
        visual_token_end=4,
        source_visual_position_ids=source_positions,
        device="cpu",
    )

    assert position_ids is not None
    assert torch.equal(position_ids[:, 0, 2:4], source_positions)
    assert _text_positions_are_1d(position_ids, 2, 4)
    assert _visual_position_ids_equal_source(position_ids, 2, 4, source_positions)
