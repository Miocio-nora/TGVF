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


def _write_fake_ocrbench_official(root: Path) -> Path:
    eval_path = root / "ocrbench_v2" / "official_code" / "OCRBench_v2" / "eval_scripts" / "eval.py"
    eval_path.parent.mkdir(parents=True)
    eval_path.write_text(
        """
import json


def process_predictions(input_path, output_path):
    with open(input_path, encoding="utf-8") as handle:
        rows = json.load(handle)
    for row in rows:
        answers = row.get("answers") or []
        if not isinstance(answers, list):
            answers = [answers]
        row["score"] = 1.0 if str(row.get("predict") or "") in {str(item) for item in answers} else 0.0
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle)
""".strip()
        + "\n"
    )
    return eval_path


def _write_fake_mmmu_pro_official(root: Path) -> Path:
    eval_path = root / "mmmu_pro" / "official_code" / "mmmu-pro" / "evaluate.py"
    eval_path.parent.mkdir(parents=True)
    eval_path.write_text(
        """
def get_multi_choice_info(options):
    letters = []
    index2ans = {}
    for index, option in enumerate(options):
        letter = chr(ord("A") + index)
        letters.append(letter)
        index2ans[letter] = option
    return index2ans, letters


def parse_multi_choice_response(response, all_choices, index2ans):
    for choice in all_choices:
        if f"({choice})" in response or response.strip() == choice:
            return choice
    for choice, answer in index2ans.items():
        if str(answer).lower() in str(response).lower():
            return choice
    return all_choices[0]


def eval_multi_choice(gold_i, pred_i):
    return gold_i == pred_i
""".strip()
        + "\n"
    )
    return eval_path


def _write_fake_mathvista_official(root: Path) -> Path:
    eval_path = root / "mathvista" / "official_code" / "evaluation" / "calculate_score.py"
    eval_path.parent.mkdir(parents=True)
    eval_path.write_text("# fake MathVista official scorer path for clean tests\n")
    return eval_path


def _write_fake_mathverse_official(root: Path) -> Path:
    eval_path = root / "mathverse" / "official_code" / "evaluation" / "score_answer_s2.py"
    eval_path.parent.mkdir(parents=True)
    eval_path.write_text("# fake MathVerse official scorer path for clean tests\n")
    return eval_path


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


def test_score_output_rows_ocrbench_v2_official_batch_matches_legacy(tmp_path) -> None:
    eval_path = _write_fake_ocrbench_official(tmp_path)
    rows = [
        {
            "sample_id": "ocrbench_v2/sample/0",
            "benchmark": "ocrbench_v2",
            "question": "Read the word.",
            "raw_output": "<THINK>ignore</THINK><ANSWER>blue</ANSWER>",
            "choices": [],
            "gold_answer": "blue",
            "metadata": {"type": "text recognition en", "answers": ["blue"]},
            "error": None,
        }
    ]

    score_output_rows(rows, scoring_backend=ScoringBackend.OFFICIAL, benchmark_root=tmp_path)

    legacy = _legacy_official_tools()
    legacy_info = legacy.OfficialToolInfo(
        official_tool_used=False,
        official_tool_path=str(tmp_path / "ocrbench_v2" / "official_code"),
        scorer_name="fallback",
        prompt_source="project",
    )
    legacy_scorer = legacy.OCRBenchV2OfficialScorer(legacy_info)
    legacy_rows = [
        {
            "sample_id": rows[0]["sample_id"],
            "question": rows[0]["question"],
            "raw_output": rows[0]["raw_output"],
            "parsed_answer": legacy_scorer.parse_prediction(rows[0]["raw_output"], object()),
            "gold_answer": rows[0]["gold_answer"],
            "metadata": rows[0]["metadata"],
        }
    ]
    legacy_scorer.score_predictions(legacy_rows)

    assert rows[0]["parsed_answer"] == "blue"
    assert rows[0]["score"] == legacy_rows[0]["score"] == 1.0
    assert rows[0]["scorer_name"] == "official_ocrbench_v2"
    assert rows[0]["official_tool_used"] is True
    assert rows[0]["official_tool_path"] == str(eval_path)


def test_score_output_rows_ocrbench_v2_official_requires_local_tool(tmp_path) -> None:
    rows = [
        {
            "benchmark": "ocrbench_v2",
            "raw_output": "blue",
            "gold_answer": "blue",
            "metadata": {"type": "text recognition en", "answers": ["blue"]},
            "error": None,
        }
    ]

    with pytest.raises(NotImplementedError):
        score_output_rows(rows, scoring_backend=ScoringBackend.OFFICIAL, benchmark_root=tmp_path)


def test_score_output_rows_mmmu_pro_official_batch_matches_legacy(tmp_path) -> None:
    eval_path = _write_fake_mmmu_pro_official(tmp_path)
    rows = [
        {
            "sample_id": "mmmu_pro/sample/0",
            "benchmark": "mmmu_pro",
            "raw_output": "After checking the diagram, the answer is (C).",
            "choices": ["red", "blue", "green"],
            "gold_answer": "C",
            "metadata": {"subject": "Art"},
            "error": None,
        }
    ]

    score_output_rows(rows, scoring_backend=ScoringBackend.OFFICIAL, benchmark_root=tmp_path)

    legacy = _legacy_official_tools()
    legacy_info = legacy.OfficialToolInfo(
        official_tool_used=False,
        official_tool_path=str(tmp_path / "mmmu_pro" / "official_code"),
        scorer_name="fallback",
        prompt_source="project",
    )
    legacy_scorer = legacy.MMMUProOfficialScorer(legacy_info)
    sample = type("Sample", (), {"choices": rows[0]["choices"]})()
    legacy_rows = [
        {
            "sample_id": rows[0]["sample_id"],
            "raw_output": rows[0]["raw_output"],
            "parsed_answer": legacy_scorer.parse_prediction(rows[0]["raw_output"], sample),
            "choices": rows[0]["choices"],
            "gold_answer": rows[0]["gold_answer"],
            "metadata": rows[0]["metadata"],
        }
    ]
    legacy_scorer.score_predictions(legacy_rows)

    assert rows[0]["parsed_answer"] == "C"
    assert rows[0]["score"] == legacy_rows[0]["score"] == 1.0
    assert rows[0]["scorer_name"] == "official_mmmu_pro"
    assert rows[0]["official_tool_used"] is True
    assert rows[0]["official_tool_path"] == str(eval_path)


def test_score_output_rows_mmmu_pro_official_requires_local_tool(tmp_path) -> None:
    rows = [
        {
            "benchmark": "mmmu_pro",
            "raw_output": "A",
            "choices": ["red", "blue"],
            "gold_answer": "A",
            "metadata": {"subject": "Art"},
            "error": None,
        }
    ]

    with pytest.raises(NotImplementedError):
        score_output_rows(rows, scoring_backend=ScoringBackend.OFFICIAL, benchmark_root=tmp_path)


def test_score_output_rows_mathvista_official_disabled_llm_matches_legacy(tmp_path) -> None:
    eval_path = _write_fake_mathvista_official(tmp_path)
    rows = [
        {
            "sample_id": "mathvista/sample/0",
            "benchmark": "mathvista",
            "raw_output": "We compute the value. Final answer is 1.23",
            "choices": [],
            "gold_answer": "1.2",
            "metadata": {"question_type": "free_form", "answer_type": "float", "precision": 1},
            "error": None,
        }
    ]

    score_output_rows(rows, scoring_backend=ScoringBackend.OFFICIAL, benchmark_root=tmp_path)

    legacy = _legacy_official_tools()
    legacy_info = legacy.OfficialToolInfo(
        official_tool_used=False,
        official_tool_path=str(tmp_path / "mathvista" / "official_code"),
        scorer_name="fallback",
        prompt_source="project",
    )
    legacy_scorer = legacy.MathVistaOfficialScorer(legacy_info, official_llm_mode="disabled")
    legacy_rows = [
        {
            "sample_id": rows[0]["sample_id"],
            "raw_output": rows[0]["raw_output"],
            "choices": rows[0]["choices"],
            "gold_answer": rows[0]["gold_answer"],
            "metadata": rows[0]["metadata"],
        }
    ]
    legacy_scorer.score_predictions(legacy_rows)

    assert rows[0]["parsed_answer"] == legacy_rows[0]["parsed_answer"] == "1.23"
    assert rows[0]["prediction"] == legacy_rows[0]["prediction"] == "1.2"
    assert rows[0]["score"] == legacy_rows[0]["score"] == 1.0
    assert rows[0]["scorer_name"] == "official_mathvista"
    assert rows[0]["official_tool_path"] == str(eval_path)
    assert rows[0]["llm_judge_used"] is False


def test_score_output_rows_mathvista_official_requires_local_tool(tmp_path) -> None:
    rows = [
        {
            "benchmark": "mathvista",
            "raw_output": "1",
            "gold_answer": "1",
            "metadata": {"question_type": "free_form", "answer_type": "integer"},
            "error": None,
        }
    ]

    with pytest.raises(NotImplementedError):
        score_output_rows(rows, scoring_backend=ScoringBackend.OFFICIAL, benchmark_root=tmp_path)


def test_score_output_rows_mathverse_official_disabled_llm_matches_legacy(tmp_path) -> None:
    eval_path = _write_fake_mathverse_official(tmp_path)
    rows = [
        {
            "sample_id": "mathverse/sample/0",
            "benchmark": "mathverse",
            "raw_output": "The final answer is (D).",
            "choices": ["40°", "60°", "120°", "140°"],
            "gold_answer": "D",
            "metadata": {"problem_version": "Text Dominant", "question_type": "multi-choice"},
            "error": None,
        }
    ]

    score_output_rows(rows, scoring_backend=ScoringBackend.OFFICIAL, benchmark_root=tmp_path)

    legacy = _legacy_official_tools()
    legacy_info = legacy.OfficialToolInfo(
        official_tool_used=False,
        official_tool_path=str(tmp_path / "mathverse" / "official_code"),
        scorer_name="fallback",
        prompt_source="project",
    )
    legacy_scorer = legacy.MathVerseOfficialScorer(legacy_info, official_llm_mode="disabled")
    legacy_rows = [
        {
            "sample_id": rows[0]["sample_id"],
            "raw_output": rows[0]["raw_output"],
            "choices": rows[0]["choices"],
            "gold_answer": rows[0]["gold_answer"],
            "metadata": rows[0]["metadata"],
        }
    ]
    legacy_scorer.score_predictions(legacy_rows)

    assert rows[0]["parsed_answer"] == legacy_rows[0]["parsed_answer"] == "D"
    assert rows[0]["score"] == legacy_rows[0]["score"] == 1.0
    assert rows[0]["scorer_name"] == "official_mathverse"
    assert rows[0]["official_tool_path"] == str(eval_path)
    assert rows[0]["llm_judge_used"] is False


def test_score_output_rows_mathverse_official_requires_local_tool(tmp_path) -> None:
    rows = [
        {
            "benchmark": "mathverse",
            "raw_output": "A",
            "choices": ["x", "y"],
            "gold_answer": "A",
            "metadata": {"question_type": "multi-choice"},
            "error": None,
        }
    ]

    with pytest.raises(NotImplementedError):
        score_output_rows(rows, scoring_backend=ScoringBackend.OFFICIAL, benchmark_root=tmp_path)
