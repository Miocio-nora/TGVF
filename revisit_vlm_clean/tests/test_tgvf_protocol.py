import pytest
from revisit_vlm_clean.tgvf_protocol import parse_tgvf_action


def test_parse_protocol_c_focus_action() -> None:
    parsed = parse_tgvf_action(
        "<think>\nI need visual focus before answering.\n</think>\n"
        "<|focus_start|>the glove<|focus_end|><|im_end|>",
        protocol="protocol_c_tool_observation",
    )

    assert parsed.focus_target == "the glove"
    assert parsed.focus_valid
    assert not parsed.malformed


def test_parse_protocol_c_direct_answer() -> None:
    parsed = parse_tgvf_action(
        "<think>\nThe answer is directly visible.\n</think>\nA<|im_end|>",
        protocol="protocol_c_tool_observation",
    )

    assert parsed.answer == "A"
    assert parsed.answer_valid
    assert not parsed.focus_valid


def test_parse_protocol_c_rejects_generic_focus() -> None:
    parsed = parse_tgvf_action(
        "<|focus_start|>the image<|focus_end|>",
        protocol="protocol_c_tool_observation_qwen2_no_think",
    )

    assert parsed.malformed
    assert "generic_target" in parsed.malformed_reasons


def test_parse_protocol_rejects_unknown_clean_protocol() -> None:
    with pytest.raises(ValueError):
        parse_tgvf_action("<|focus_start|>x<|focus_end|>", protocol="legacy_v3_tags")
