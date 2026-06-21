from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from convert_stage2_choices_to_open_answer import convert_file, convert_choice_record


def test_convert_choice_record_uses_short_answer_and_removes_choices() -> None:
    row = convert_choice_record(
        {
            "question": "What color is it?",
            "choices": [{"label": "A", "text": "red"}, {"label": "B", "text": "orange"}],
            "answer": "B. orange",
            "short_answer": "orange",
            "answer_format": "multiple_choice",
            "value_span_text": "orange",
            "focus_steps": [{"target": "the colored part", "value_span_text": "orange"}],
        },
        line_no=1,
    )

    assert row["question"] == "What color is it?"
    assert row["choices"] == []
    assert row["answer"] == "orange"
    assert row["short_answer"] == "orange"
    assert row["answer_format"] == "short_text"
    assert row["value_span_text"] == "orange"
    assert row["focus_steps"][0]["value_span_text"] == "orange"
    assert row["metadata"]["choice_to_open_answer"]["original_answer"] == "B. orange"


def test_convert_choice_record_recovers_answer_from_letter_and_choices() -> None:
    row = convert_choice_record(
        {
            "question": "Which option is higher?",
            "choices": ["low", "high"],
            "answer": "B. high",
            "answer_format": "multiple_choice",
        },
        line_no=1,
    )

    assert row["answer"] == "high"
    assert row["short_answer"] == "high"


def test_convert_file_preserves_count_and_zeroes_choice_ratio(tmp_path: Path) -> None:
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    records = [
        {
            "question": "Question?\nA. red\nB. blue",
            "choices": [{"label": "A", "text": "red"}, {"label": "B", "text": "blue"}],
            "answer": "B. blue",
            "short_answer": "blue",
            "answer_format": "multiple_choice",
        },
        {
            "question": "What is visible?",
            "answer": "a bird",
            "short_answer": "a bird",
            "answer_format": "short_text",
        },
    ]
    input_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    report = convert_file(input_path, output_path)
    converted = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert report["input"] == 2
    assert report["converted_choice"] == 1
    assert report["kept_non_choice"] == 1
    assert report["output_choice_ratio"] == 0.0
    assert converted[0]["question"] == "Question?"
    assert converted[0]["answer"] == "blue"
    assert converted[1] == records[1]
