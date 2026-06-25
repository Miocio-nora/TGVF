import importlib.util
import sys
from pathlib import Path

import pytest

from revisit_vlm_clean.schema import ScoringBackend
from revisit_vlm_clean.scoring import (
    extract_choice_official_compatible,
    extract_choice_strict,
    parse_and_score,
    score_output_rows,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _legacy_official_tools():
    path = REPO_ROOT / "src" / "tgvf_eval" / "official_tools.py"
    spec = importlib.util.spec_from_file_location("legacy_official_tools_for_clean_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def test_parse_and_score_open_html_gold_uses_same_cleaning() -> None:
    html = '<table><tr><td rowspan="2"> 项目</td><td> 期初余额</td></tr></table>'
    result = parse_and_score(html, gold_answer=html)

    assert result.score == 1.0


def test_official_compatible_choice_parser_matches_legacy_helper() -> None:
    legacy = _legacy_official_tools()
    choices = ["red", "blue", "green"]
    examples = [
        "final answer is C",
        "(B) blue",
        "blue",
        "<|im_end|>\nA",
    ]
    for text in examples:
        assert extract_choice_official_compatible(text, choices) == legacy._extract_choice(
            text,
            num_choices=len(choices),
        )


def test_official_compatible_choice_scoring_matches_legacy_blink() -> None:
    legacy = _legacy_official_tools()
    choices = ["red", "blue"]
    text = "(B) blue"

    result = parse_and_score(
        text,
        choices=choices,
        gold_answer="B",
        benchmark="blink",
        scoring_backend=ScoringBackend.OFFICIAL,
    )

    legacy_pred = legacy._extract_choice(text, num_choices=len(choices))
    assert result.parsed_answer == legacy_pred
    assert result.score == legacy._score_choice(legacy_pred, "B", choices)
    assert result.scorer_name == "official_blink_exact_match"
    assert result.official_tool_used is True
    assert result.official_compatible is False


def test_auto_uses_official_compatible_choice_for_hrbench() -> None:
    result = parse_and_score(
        "(A) right",
        choices=["right", "left"],
        gold_answer="A",
        benchmark="hr_bench_4k",
        scoring_backend=ScoringBackend.AUTO,
    )

    assert result.score == 1.0
    assert result.scorer_name == "official_compatible_hrbench4k_mc"
    assert result.official_tool_used is True
    assert result.official_compatible is True


def test_project_and_official_choice_parser_differ_on_option_text_only() -> None:
    choices = ["red", "blue"]
    project = parse_and_score(
        "The correct one is blue.",
        choices=choices,
        gold_answer="B",
        benchmark="blink",
        scoring_backend=ScoringBackend.PROJECT,
    )
    official = parse_and_score(
        "The correct one is blue.",
        choices=choices,
        gold_answer="B",
        benchmark="blink",
        scoring_backend=ScoringBackend.OFFICIAL,
    )

    assert project.parsed_answer == "B"
    assert project.score == 1.0
    assert official.parsed_answer == ""
    assert official.score == 0.0


def test_official_backend_not_ported_for_unsupported_benchmark() -> None:
    with pytest.raises(NotImplementedError):
        parse_and_score(
            "A",
            choices=["x"],
            gold_answer="A",
            benchmark="vstar_bench",
            scoring_backend=ScoringBackend.OFFICIAL,
        )


def test_score_output_rows_batch_scores_and_skips_error_rows() -> None:
    rows = [
        {
            "benchmark": "blink",
            "raw_output": "(B) two",
            "choices": ["one", "two"],
            "gold_answer": "B",
            "error": None,
        },
        {
            "benchmark": "blink",
            "raw_output": "(A) one",
            "choices": ["one", "two"],
            "gold_answer": "A",
            "error": "RuntimeError: failed",
        },
    ]

    score_output_rows(rows, scoring_backend=ScoringBackend.AUTO)

    assert rows[0]["parsed_answer"] == "B"
    assert rows[0]["score"] == 1.0
    assert rows[0]["scorer_name"] == "official_blink_exact_match"
    assert rows[0]["official_tool_used"] is True
    assert rows[1]["parsed_answer"] == ""
    assert rows[1]["score"] is None
    assert rows[1]["answer_parse_success"] is False
    assert rows[1]["scorer_name"] == ""
