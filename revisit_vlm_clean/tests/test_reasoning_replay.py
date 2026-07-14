from __future__ import annotations

import json
from pathlib import Path

from revisit_vlm_clean.reasoning_replay import (
    SourceRecord,
    build_replay_split,
    parse_original_reasoning,
    select_direct_records,
    validate_generation,
)


def test_parse_original_reasoning_accepts_qwen_thinking_prefill_output() -> None:
    parsed = parse_original_reasoning(
        "I can identify the prominent animal directly.\n</think>\nThe answer is a bird."
    )

    assert parsed.valid is True
    assert parsed.reasoning == "I can identify the prominent animal directly."
    assert parsed.answer_candidate == "a bird"


def test_parse_original_reasoning_accepts_explicit_think_and_boxed_answer() -> None:
    parsed = parse_original_reasoning("<think>Compute 2 + 2.</think>\n\\boxed{4}")

    assert parsed.valid is True
    assert parsed.reasoning == "Compute 2 + 2."
    assert parsed.answer_candidate == "4"


def test_validate_generation_rejects_wrong_answer_and_budget_hits() -> None:
    row = {"answer": "4", "short_answer": "4", "value_span_text": "4"}

    mismatch = validate_generation(
        row,
        raw_output="The chart shows five units.\n</think>\n5",
        output_tokens=20,
        generation_max_tokens=2048,
        max_replay_target_tokens=1024,
    )
    budget_hit = validate_generation(
        row,
        raw_output="still reasoning",
        output_tokens=2048,
        generation_max_tokens=2048,
        max_replay_target_tokens=1024,
    )

    assert mismatch["accepted"] is False
    assert mismatch["reason"] == "answer_mismatch"
    assert budget_hit["accepted"] is False
    assert budget_hit["reason"] == "generation_budget_hit"


def test_validate_generation_extracts_conclusion_from_verbose_final_answer() -> None:
    validation = validate_generation(
        {"answer": "41 percentage points"},
        raw_output=(
            "Subtract 10 from 51.\n</think>\n"
            "First identify 51% and 10%. Thus, the answer is "
            "**41 percentage points**."
        ),
        output_tokens=80,
        generation_max_tokens=2048,
        max_replay_target_tokens=1024,
    )

    assert validation["accepted"] is True
    assert validation["answer_candidate"] == "41 percentage points"


def test_validate_generation_does_not_match_number_only_mentioned_in_explanation() -> None:
    validation = validate_generation(
        {"answer": "4"},
        raw_output=(
            "I see values 3.0, 4.0, and 5.0, but count six lines.\n</think>\n"
            "The legend includes 4.0 as one value. **Answer:** 6"
        ),
        output_tokens=80,
        generation_max_tokens=2048,
        max_replay_target_tokens=1024,
    )

    assert validation["accepted"] is False
    assert validation["reason"] == "answer_mismatch"


def test_length_limit_applies_to_serialized_replay_target() -> None:
    accepted = validate_generation(
        {"answer": "4"},
        raw_output="Compute the value.\n</think>\nThe answer is 4.",
        output_tokens=1500,
        replay_target_tokens=80,
        generation_max_tokens=2048,
        max_replay_target_tokens=1024,
    )
    rejected = validate_generation(
        {"answer": "4"},
        raw_output="Compute the value.\n</think>\nThe answer is 4.",
        output_tokens=1500,
        replay_target_tokens=1100,
        generation_max_tokens=2048,
        max_replay_target_tokens=1024,
    )

    assert accepted["accepted"] is True
    assert rejected["accepted"] is False
    assert rejected["reason"] == "replay_target_length_limit_exceeded"


def test_sequence_limit_combines_prompt_and_replay_target() -> None:
    accepted = validate_generation(
        {"answer": "4"},
        raw_output="Compute the value.\n</think>\nThe answer is 4.",
        output_tokens=1500,
        replay_target_tokens=1281,
        prompt_tokens=274,
        generation_max_tokens=2048,
        max_sequence_tokens=2048,
    )
    rejected = validate_generation(
        {"answer": "4"},
        raw_output="Compute the value.\n</think>\nThe answer is 4.",
        output_tokens=1500,
        replay_target_tokens=1800,
        prompt_tokens=274,
        generation_max_tokens=2048,
        max_sequence_tokens=2048,
    )

    assert accepted["accepted"] is True
    assert rejected["accepted"] is False
    assert rejected["reason"] == "replay_sequence_length_limit_exceeded"


def test_explicit_answer_equivalence_accepts_units_sentences_and_number_words() -> None:
    cases = [
        ("538", "Belgium has **538 more cohabitations** than the region."),
        ("Blue", "The bars in the chart are blue."),
        ("2", "There are two people visible in the image."),
    ]
    for gold, final_text in cases:
        validation = validate_generation(
            {"answer": gold, "question": "What is the answer?"},
            raw_output=f"Inspect the image.\n</think>\n{final_text}",
            output_tokens=80,
            generation_max_tokens=2048,
            max_replay_target_tokens=1024,
        )
        assert validation["accepted"] is True
        assert validation["match_type"] == "explicit_answer_equivalent"


def test_explicit_answer_equivalence_handles_approximation_and_latex_bold() -> None:
    approximate = validate_generation(
        {"answer": "72.16%", "question": "Approximately what percentage?"},
        raw_output="Add the values.\n</think>\nThus, the answer is **72%**.",
        output_tokens=80,
        generation_max_tokens=2048,
        max_replay_target_tokens=1024,
    )
    latex = validate_generation(
        {"answer": "0.91 billion U.S. dollars", "question": "What is the difference?"},
        raw_output=(
            "Subtract the values.\n</think>\n"
            "The difference is \\boldsymbol{0.91} billion U.S. dollars."
        ),
        output_tokens=80,
        generation_max_tokens=2048,
        max_replay_target_tokens=1024,
    )

    assert approximate["accepted"] is True
    assert latex["accepted"] is True


def test_color_compound_is_not_relaxed_to_single_color_gold() -> None:
    validation = validate_generation(
        {"answer": "orange", "question": "What color is the wall?"},
        raw_output=(
            "The tone is mixed.\n</think>\n"
            "The wall is a warm **orange-brown** shade."
        ),
        output_tokens=80,
        generation_max_tokens=2048,
        max_replay_target_tokens=1024,
    )

    assert validation["accepted"] is False
    assert validation["reason"] == "answer_mismatch"


def test_validate_generation_rejects_tool_protocol_leakage() -> None:
    validation = validate_generation(
        {"answer": "yes"},
        raw_output="Need a crop. <|focus_start|>sign<|focus_end|>\n</think>\nyes",
        output_tokens=32,
        generation_max_tokens=2048,
        max_replay_target_tokens=1024,
    )

    assert validation["accepted"] is False
    assert validation["reason"] == "tool_protocol_leakage"


def test_pilot_selection_covers_math_and_whole_image_first() -> None:
    records = []
    for index, question_type in enumerate(
        ["color_attribute", "counting", "whole_image_obvious", "math_reasoning"]
    ):
        row = {
            "v4_uid": f"row-{index}",
            "need_focus": False,
            "question_type": question_type,
            "source_dataset": "source-a",
        }
        records.append(
            SourceRecord(
                split="train",
                source_index=index,
                row=row,
                raw_line=json.dumps(row) + "\n",
                row_sha256=str(index),
            )
        )

    selected = select_direct_records(records, max_records=2, seed=7)

    assert [record.row["question_type"] for record in selected] == [
        "math_reasoning",
        "whole_image_obvious",
    ]


def test_build_replay_split_preserves_focus_line_and_replaces_only_accepted_direct(
    tmp_path: Path,
) -> None:
    focus = {
        "v4_uid": "focus-1",
        "need_focus": True,
        "question": "What is written?",
        "answer": "open",
        "no_focus_think": None,
    }
    accepted_direct = {
        "v4_uid": "direct-1",
        "need_focus": False,
        "question": "What animal is shown?",
        "answer": "a bird",
        "no_focus_think": "short teacher template",
    }
    rejected_direct = {
        "v4_uid": "direct-2",
        "need_focus": False,
        "question": "How many?",
        "answer": "2",
        "no_focus_think": "short teacher template",
    }
    focus_line = json.dumps(focus, separators=(",", ":")) + "\n"
    source_path = tmp_path / "source.jsonl"
    source_path.write_text(
        focus_line
        + json.dumps(accepted_direct)
        + "\n"
        + json.dumps(rejected_direct)
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "output.jsonl"

    stats = build_replay_split(
        source_path=source_path,
        output_path=output_path,
        split="train",
        generation_rows=[
            {
                "split": "train",
                "v4_uid": "direct-1",
                "source_row_sha256": "row-hash",
                "output_tokens": 31,
                "validation": {
                    "accepted": True,
                    "reasoning": "The bird is the main visible subject.",
                    "answer_candidate": "a bird",
                    "matched_alias": "a bird",
                },
            },
            {
                "split": "train",
                "v4_uid": "direct-2",
                "source_row_sha256": "row-hash-2",
                "output_tokens": 20,
                "validation": {"accepted": False, "reason": "answer_mismatch"},
            },
        ],
        run_metadata={"run_id": "replay-test"},
    )

    output_lines = output_path.read_text(encoding="utf-8").splitlines(keepends=True)
    replayed = json.loads(output_lines[1])
    assert output_lines[0] == focus_line
    assert len(output_lines) == 2
    assert replayed["no_focus_think"] == "The bird is the main visible subject."
    assert replayed["answer"] == "a bird"
    assert replayed["reasoning_replay"]["run_id"] == "replay-test"
    assert stats["focus_payload_identical"] is True
    assert stats["written_direct_rows"] == 1
    assert stats["dropped_direct_rows"] == 1
