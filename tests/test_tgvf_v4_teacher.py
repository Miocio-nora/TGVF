from __future__ import annotations

import json

from tgvf_data.generate_teacher import (
    GenerationConfig,
    get_teacher_prompt,
    get_teacher_schema,
    validate_image_level_output,
)
from tgvf_data.tgvf_teacher_schema_v4 import (
    SCHEMA_VERSION_V4,
    TEACHER_VERSION_V4,
    render_tgvf_v4_sft_item,
    validate_focus_text_no_answer_leakage,
    validate_item_distribution_batch,
    validate_multiple_choice,
    validate_trace_shape,
    validate_v4_item,
)
from tgvf_data.tgvf_v4_teacher import parse_v4_teacher_output_text


def test_v4_prompt_and_schema_registry() -> None:
    assert "visual re-focus trace annotator" in get_teacher_prompt(TEACHER_VERSION_V4)
    schema = get_teacher_schema(SCHEMA_VERSION_V4)
    assert schema["properties"]["schema_version"]["enum"] == [SCHEMA_VERSION_V4]
    assert schema["properties"]["teacher_version"]["enum"] == [TEACHER_VERSION_V4]
    assert schema["additionalProperties"] is False


def test_v4_valid_single_refocus_validates_flattens_and_renders() -> None:
    image_output = _v4_output([_single_refocus_item()])
    accepted, rejected = validate_image_level_output(
        image_output,
        _image_record(),
        config=GenerationConfig(prompt_version=TEACHER_VERSION_V4, schema_version=SCHEMA_VERSION_V4),
        teacher_run_id="teacher_v4_test",
        model="gpt-test",
    )

    assert rejected == []
    assert len(accepted) == 1
    row = accepted[0]
    assert row["schema_version"] == SCHEMA_VERSION_V4
    assert row["teacher_version"] == TEACHER_VERSION_V4
    assert row["item_type"] == "single_refocus"
    assert row["num_focus_steps"] == 1
    assert row["focus_texts"] == ["close-up surface texture, folds, and sheen of the glove"]
    assert "<|focus_start|>close-up surface texture" in row["sft_text"]
    assert "Input: <|tgvf_start|>[D visual embeddings 1]<|tgvf_end|>" in row["sft_text"]
    assert "focused_evidence" not in row["sft_text"]


def test_v4_no_refocus_has_no_focus_tokens_or_tgvf_placeholder() -> None:
    item = _no_refocus_continue_item()
    assert validate_v4_item(item) == []
    rendered = render_tgvf_v4_sft_item(item)
    assert "<|focus_start|>" not in rendered
    assert "<|tgvf_start|>" not in rendered
    assert "<think>" in rendered


def test_v4_multi_refocus_exactly_two_focus_steps() -> None:
    item = _multi_refocus_item()
    assert validate_trace_shape(item) == []
    rendered = render_tgvf_v4_sft_item(item)
    assert rendered.count("<|focus_start|>") == 2
    assert "[D visual embeddings 2]" in rendered


def test_v4_rejects_focus_text_answer_leakage_and_task_verbs() -> None:
    leaking = _single_refocus_item()
    leaking["trace"][1]["focus_text"] = "determine the rubber material of the glove"
    reasons = validate_v4_item(leaking)
    assert "answer_value_in_focus_text" in reasons
    assert "focus_text_contains_task_verb" in reasons
    assert validate_focus_text_no_answer_leakage(leaking)


def test_v4_multiple_choice_formatting_checks_label_and_text() -> None:
    item = _single_refocus_item()
    assert validate_multiple_choice(item) == []
    item["trace"][-1]["text"] = "B"
    assert "answer_step_missing_correct_choice_text" in validate_multiple_choice(item)


def test_v4_distribution_report_tracks_main_categories_not_only_cues() -> None:
    report = validate_item_distribution_batch(
        [_single_refocus_item(), _multi_refocus_item(), _no_refocus_continue_item()]
    )
    assert report["question_type_distribution"]["texture_material"] == 1
    assert report["focus_category_distribution"]["chart_table_region"] == 1
    assert "surface" in report["focus_descriptor_cues_distribution"]


def test_v4_parser_strips_trivial_json_fence() -> None:
    payload = _v4_output([_single_refocus_item()])
    parsed = parse_v4_teacher_output_text("```json\n" + json.dumps(payload) + "\n```")
    assert parsed["schema_version"] == SCHEMA_VERSION_V4


def _image_record() -> dict:
    return {
        "stable_image_uid": "image:1",
        "source_dataset": "unit",
        "source_profile": "natural_image",
        "image_path": "/tmp/image.jpg",
    }


def _v4_output(items: list[dict]) -> dict:
    return {
        "schema_version": SCHEMA_VERSION_V4,
        "teacher_version": TEACHER_VERSION_V4,
        "image_id": "image:1",
        "image_summary": "A person is wearing a glove near a work surface.",
        "items": items,
    }


def _single_refocus_item() -> dict:
    return {
        "item_id": "single_0",
        "item_type": "single_refocus",
        "answer_format": "multiple_choice",
        "question_type": "texture_material",
        "focus_category": "surface_detail",
        "question": "What material does the glove appear to be made of?",
        "choices": [
            {"label": "A", "text": "metal"},
            {"label": "B", "text": "rubber"},
            {"label": "C", "text": "wood"},
            {"label": "D", "text": "paper"},
        ],
        "correct_choice": "B",
        "answer_text": "rubber",
        "trace": [
            {
                "type": "think",
                "text": "The glove is visible, but the question depends on fine surface appearance rather than object identity. A detailed surface view would be useful.",
            },
            {
                "type": "focus",
                "focus_text": "close-up surface texture, folds, and sheen of the glove",
                "focused_evidence": "The focused evidence shows a smooth, slightly shiny glove surface consistent with rubber.",
                "metadata": {
                    "evidence_type": "texture_material",
                    "focus_descriptor_cues": ["surface", "texture", "object_part", "material_cue"],
                    "target_leakage_risk": "low",
                    "confidence": 0.9,
                },
            },
            {
                "type": "think",
                "text": "The focused surface cues show a smooth and slightly shiny material, which matches rubber better than the other choices.",
            },
            {"type": "answer", "text": "B. rubber"},
        ],
        "quality": {"confidence": 0.9, "num_focus_steps": 1, "notes": None},
    }


def _no_refocus_continue_item() -> dict:
    return {
        "item_id": "direct_0",
        "item_type": "no_refocus_continue",
        "answer_format": "open",
        "question_type": "math_reasoning",
        "focus_category": "none",
        "question": "What is the value of the expression shown in the image?",
        "choices": [],
        "correct_choice": None,
        "answer_text": "36",
        "trace": [
            {
                "type": "think",
                "text": "The expression is already clearly readable, so no focused visual observation is needed. I should apply the order of operations before answering.",
            },
            {"type": "answer", "text": "36"},
        ],
        "quality": {"confidence": 0.9, "num_focus_steps": 0, "notes": None},
    }


def _multi_refocus_item() -> dict:
    return {
        "item_id": "multi_0",
        "item_type": "multi_refocus",
        "answer_format": "multiple_choice",
        "question_type": "chart_table",
        "focus_category": "chart_table_region",
        "question": "What value does the category represented by the striped legend entry reach?",
        "choices": [
            {"label": "A", "text": "20"},
            {"label": "B", "text": "25"},
            {"label": "C", "text": "30"},
            {"label": "D", "text": "35"},
        ],
        "correct_choice": "C",
        "answer_text": "30",
        "trace": [
            {"type": "think", "text": "The chart question depends on matching the legend style to the correct bar. I first need the legend area clearly."},
            {
                "type": "focus",
                "focus_text": "legend entries and pattern swatches beside the chart",
                "focused_evidence": "The striped legend entry corresponds to the target category.",
                "metadata": {
                    "evidence_type": "chart_value",
                    "focus_descriptor_cues": ["chart_anchor", "pattern", "nearby_anchor"],
                    "target_leakage_risk": "low",
                    "confidence": 0.9,
                },
            },
            {"type": "think", "text": "The legend identifies which visual mark to use, but the value still requires the matching bar and nearby axis labels."},
            {
                "type": "focus",
                "focus_text": "the matching patterned bar and nearby y-axis tick labels",
                "focused_evidence": "The matching patterned bar reaches the y-axis value of 30.",
                "metadata": {
                    "evidence_type": "chart_value",
                    "focus_descriptor_cues": ["chart_anchor", "pattern", "number_like"],
                    "target_leakage_risk": "low",
                    "confidence": 0.9,
                },
            },
            {"type": "think", "text": "The second focused view gives the chart value for the matched bar."},
            {"type": "answer", "text": "C. 30"},
        ],
        "quality": {"confidence": 0.9, "num_focus_steps": 2, "notes": None},
    }
