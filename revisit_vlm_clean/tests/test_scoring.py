import pytest

from revisit_vlm_clean.schema import ScoringBackend
from revisit_vlm_clean.scoring import extract_choice_strict, parse_and_score


def test_extract_choice_strict_answer_tag() -> None:
    assert extract_choice_strict("<ANSWER>(B)</ANSWER>", ["red", "blue", "green"]) == "B"


def test_extract_choice_strict_option_text() -> None:
    assert extract_choice_strict("The correct one is blue.", ["red", "blue", "green"]) == "B"


def test_parse_and_score_choice() -> None:
    result = parse_and_score("final answer is C", choices=["a", "b", "c"], gold_answer="C")
    assert result.parsed_answer == "C"
    assert result.score == 1.0
    assert result.answer_parse_success


def test_parse_and_score_open_answer_tag() -> None:
    result = parse_and_score("<ANSWER>42</ANSWER>", gold_answer="42")
    assert result.parsed_answer == "42"
    assert result.score == 1.0


def test_official_backend_not_ported_yet() -> None:
    with pytest.raises(NotImplementedError):
        parse_and_score("A", choices=["x"], gold_answer="A", scoring_backend=ScoringBackend.OFFICIAL)
