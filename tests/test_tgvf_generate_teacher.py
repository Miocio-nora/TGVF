from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tgvf_data.generate_teacher import (
    DEFAULT_TEACHER_PROMPT_VERSION,
    DEFAULT_TEACHER_SCHEMA_VERSION,
    PROMPT_VERSION_V0,
    PROMPT_VERSION_VISUAL_CUE_V1,
    PROMPT_VERSION_V3,
    SCHEMA_VERSION_V0,
    SCHEMA_VERSION_VISUAL_CUE_V1,
    SCHEMA_VERSION_V3,
    GenerationConfig,
    OpenAIConfig,
    TeacherResponse,
    call_teacher_with_retries,
    get_teacher_prompt,
    get_teacher_schema,
    make_ledger_entry,
    prepare_batch,
    quality_report,
    read_successful_ledger_uids,
    request_custom_id,
    resume_sync,
    teacher_output_schema,
    usage_report,
    validate_image_level_output,
    validate_item,
    validate_item_with_warnings,
)
from tgvf_data.prepare import read_jsonl, write_jsonl


class FlakyClient:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls = 0

    def create_response(self, payload: dict, *, timeout_seconds: int) -> TeacherResponse:
        del payload, timeout_seconds
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("temporary timeout")
        return TeacherResponse(
            response_id="resp_1",
            output_text=self.response_text,
            raw_response={"id": "resp_1", "output_text": self.response_text},
            usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        )


def test_teacher_schema_is_strict_and_enum_constrained() -> None:
    schema = teacher_output_schema()

    assert schema["additionalProperties"] is False
    assert schema["properties"]["focus_items"]["items"]["additionalProperties"] is False
    assert schema["properties"]["direct_items"]["items"]["additionalProperties"] is False
    evidence_enum = schema["properties"]["focus_items"]["items"]["properties"]["evidence_type"]["enum"]
    assert "ocr_text" in evidence_enum
    assert schema["properties"]["schema_version"]["enum"] == [SCHEMA_VERSION_V3]


def test_prompt_registry_and_defaults() -> None:
    assert DEFAULT_TEACHER_PROMPT_VERSION == PROMPT_VERSION_V3
    assert DEFAULT_TEACHER_SCHEMA_VERSION == SCHEMA_VERSION_V3
    assert "visual evidence annotator" in get_teacher_prompt(PROMPT_VERSION_V0)
    assert "visual-cue target" in get_teacher_prompt(PROMPT_VERSION_VISUAL_CUE_V1)
    assert "Target-Guided Visual Foveation" in get_teacher_prompt(PROMPT_VERSION_V3)


def test_v1_schema_adds_visual_cue_fields_and_v0_omits_them() -> None:
    v1_item = get_teacher_schema(SCHEMA_VERSION_VISUAL_CUE_V1)["properties"]["items"]["items"]
    v0_item = get_teacher_schema(SCHEMA_VERSION_V0)["properties"]["items"]["items"]

    assert "target_style" in v1_item["required"]
    assert "target_cues" in v1_item["required"]
    assert v1_item["properties"]["target_style"]["enum"] == [
        "semantic",
        "visual_cue",
        "mixed",
    ]
    assert "target_style" not in v0_item["required"]
    assert "target_style" not in v0_item["properties"]


def test_v1_visual_cue_item_validates_and_flattens() -> None:
    accepted, rejected = validate_image_level_output(
        _image_output([_visual_cue_item()]),
        _image_record(),
        config=GenerationConfig(
            prompt_version=PROMPT_VERSION_VISUAL_CUE_V1,
            schema_version=SCHEMA_VERSION_VISUAL_CUE_V1,
        ),
        teacher_run_id="teacher_run_test",
        model="gpt-test",
        raw_response_id="resp_1",
    )

    assert rejected == []
    assert accepted[0]["target_style"] == "visual_cue"
    assert accepted[0]["target_cues"] == ["location", "color", "size"]
    assert accepted[0]["teacher_prompt_version"] == PROMPT_VERSION_VISUAL_CUE_V1
    assert accepted[0]["teacher_schema_version"] == SCHEMA_VERSION_VISUAL_CUE_V1


def test_legacy_v0_item_normalizes_visual_cue_fields() -> None:
    accepted, rejected = validate_image_level_output(
        _image_output([_good_item()]),
        _image_record(),
        config=GenerationConfig(
            prompt_version=PROMPT_VERSION_V0,
            schema_version=SCHEMA_VERSION_V0,
        ),
        teacher_run_id="teacher_run_test",
        model="gpt-test",
    )

    assert rejected == []
    assert accepted[0]["target_style"] == "unknown"
    assert accepted[0]["target_cues"] == []


def test_visual_cue_generic_filtering_and_strict_mode() -> None:
    valid = _visual_cue_item()
    assert validate_item(valid, schema_version=SCHEMA_VERSION_VISUAL_CUE_V1) == []

    generic = {
        **valid,
        "target": "something green",
        "target_cues": ["color"],
    }
    reasons, warnings = validate_item_with_warnings(
        generic,
        schema_version=SCHEMA_VERSION_VISUAL_CUE_V1,
    )
    assert "visually_generic_target" in warnings
    assert "visually_generic_target" not in reasons

    strict_reasons = validate_item(
        generic,
        schema_version=SCHEMA_VERSION_VISUAL_CUE_V1,
        strict_visual_cue_schema=True,
    )
    assert "visually_generic_target" in strict_reasons


def test_visual_cue_leakage_allows_locator_cues_not_answers() -> None:
    leaking = {
        **_visual_cue_item(),
        "question": "What color is the logo?",
        "target": "the red logo in the upper-right corner",
        "short_answer": "red",
        "target_style": "mixed",
        "target_cues": ["location", "color", "symbol_like"],
    }
    assert "short_answer_in_target" in validate_item(
        leaking,
        schema_version=SCHEMA_VERSION_VISUAL_CUE_V1,
    )

    locator = {
        **_visual_cue_item(),
        "question": "What text is on the red label near the bottom of the package?",
        "target": "the red label near the bottom of the package",
        "evidence_description": "The red label near the bottom contains the word ORGANIC.",
        "short_answer": "ORGANIC",
        "evidence_type": "ocr_text",
        "answer_type": "text_string",
        "target_style": "mixed",
        "target_cues": ["location", "color", "text_like", "region_type"],
    }
    assert validate_item(locator, schema_version=SCHEMA_VERSION_VISUAL_CUE_V1) == []


def test_quality_report_counts_visual_cue_fields() -> None:
    accepted, rejected = validate_image_level_output(
        _image_output(
            [
                _visual_cue_item(),
                {
                    **_visual_cue_item(),
                    "item_id": "1",
                    "target": "the dark rectangular patch near the bottom edge",
                    "evidence_description": "The dark rectangular patch shows faint pale text inside it.",
                    "short_answer": "pale text",
                    "target_style": "mixed",
                    "target_cues": ["location", "shape", "region_type"],
                },
            ]
        ),
        _image_record(),
        config=GenerationConfig(
            prompt_version=PROMPT_VERSION_VISUAL_CUE_V1,
            schema_version=SCHEMA_VERSION_VISUAL_CUE_V1,
        ),
        teacher_run_id="teacher_run_test",
        model="gpt-test",
    )

    report = quality_report(
        accepted,
        rejected,
        [
            make_ledger_entry(
                teacher_run_id="teacher_run_test",
                image_record=_image_record(),
                status="succeeded",
                accepted_item_count=len(accepted),
                request_custom_id="request_1",
            )
        ],
    )

    assert report["target_style_distribution"]["visual_cue"] == 1
    assert report["target_style_distribution"]["mixed"] == 1
    assert report["visual_cue_item_count"] == 1
    assert report["mixed_item_count"] == 1
    assert report["target_cues_distribution"]["location"] == 2


def test_validate_good_image_output_and_flatten() -> None:
    accepted, rejected = validate_image_level_output(
        _image_output([_good_item()]),
        _image_record(),
        config=GenerationConfig(
            prompt_version=PROMPT_VERSION_V0,
            schema_version=SCHEMA_VERSION_V0,
        ),
        teacher_run_id="teacher_run_test",
        model="gpt-test",
        raw_response_id="resp_1",
    )

    assert rejected == []
    assert len(accepted) == 1
    record = accepted[0]
    assert record["image"] == "/datasets/textvqa/img.jpg"
    assert record["question"] == "What text is printed below the barcode?"
    assert record["target"] == "the small text printed below the barcode"
    assert record["evidence_description"].endswith("EXP 08/2026.")
    assert record["item_content_hash"]


def test_default_v3_normalizes_legacy_old_json() -> None:
    accepted, rejected = validate_image_level_output(
        _image_output([_good_item()]),
        _image_record(),
        config=GenerationConfig(),
        teacher_run_id="teacher_run_test",
        model="gpt-test",
    )

    assert rejected == []
    assert accepted[0]["schema_version"] == SCHEMA_VERSION_V3
    assert accepted[0]["need_focus"] is True
    assert accepted[0]["evidence_state"] == "need_local_visual_evidence"
    assert accepted[0]["trajectory_type"] == "single_focus"
    assert accepted[0]["target_style"] == "unknown"
    assert accepted[0]["target_cues"] == []
    assert accepted[0]["answer"] == "EXP 08/2026"
    assert accepted[0]["answer_format"] == "short_text"
    assert accepted[0]["value_span_text"] == "EXP 08/2026"


def test_v3_focus_and_direct_items_validate_flatten_and_report() -> None:
    accepted, rejected = validate_image_level_output(
        _v3_image_output([_v3_focus_item()], [_direct_item()]),
        _image_record(),
        config=GenerationConfig(),
        teacher_run_id="teacher_run_test",
        model="gpt-test",
    )

    assert rejected == []
    assert len(accepted) == 2
    focus = accepted[0]
    direct = accepted[1]
    assert focus["need_focus"] is True
    assert focus["evidence_state"] == "need_local_visual_evidence"
    assert focus["target_style"] == "visual_cue"
    assert focus["answer"] == "EXP 08/2026"
    assert direct["need_focus"] is False
    assert direct["target"] == ""
    assert direct["target_style"] == "none"
    assert direct["answer"] == "dog"

    report = quality_report(
        accepted,
        rejected,
        [
            make_ledger_entry(
                teacher_run_id="teacher_run_test",
                image_record=_image_record(),
                status="succeeded",
                accepted_item_count=len(accepted),
                request_custom_id="request_1",
                num_focus_items_raw=1,
                num_direct_items_raw=1,
            )
        ],
    )
    assert report["focus_item_count"] == 1
    assert report["direct_item_count"] == 1
    assert report["answer_format_distribution"]["short_text"] == 2
    assert report["image_group_size_histogram"] == {"2": 1}


def test_v3_filtering_rejects_bad_visual_cue_and_direct_local_question() -> None:
    generic_focus = {
        **_v3_focus_item(),
        "target": "the object",
        "target_cues": ["color"],
    }
    reasons = validate_item(generic_focus, schema_version=SCHEMA_VERSION_V3)
    assert "generic_target" in reasons
    assert "visually_generic_target" in reasons
    assert "visual_cue_target_missing_cues" in reasons

    good_visual_cue = _v3_focus_item()
    assert validate_item(good_visual_cue, schema_version=SCHEMA_VERSION_V3) == []

    direct_reasons = validate_item(
        {**_direct_item(), "question": "What tiny text is printed below the barcode?"},
        schema_version=SCHEMA_VERSION_V3,
    )
    assert "direct_question_requires_focus" in direct_reasons


def test_validation_rejects_answer_leakage_and_generic_evidence() -> None:
    leakage_item = {
        **_good_item(),
        "target": "the EXP 08/2026 text printed below the barcode",
    }
    generic_item = {
        **_good_item(),
        "evidence_description": "There is some text.",
    }

    assert "short_answer_in_target" in validate_item(leakage_item)
    assert "generic_evidence_description" in validate_item(generic_item)


def test_ledger_skip_logic_and_request_custom_id(tmp_path: Path) -> None:
    ledger = tmp_path / "teacher_generation_ledger.jsonl"
    write_jsonl(
        [
            make_ledger_entry(
                teacher_run_id="teacher_run_old",
                image_record=_image_record(),
                status="succeeded",
            )
        ],
        ledger,
    )

    assert read_successful_ledger_uids(ledger) == {"textvqa:123"}
    assert request_custom_id("teacher_run_000001", "textvqa:123") == (
        "teacher_run_000001__textvqa_123"
    )


def test_usage_aggregation() -> None:
    report = usage_report(
        [
            {
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                "accepted_item_count": 3,
                "request_custom_id": "a",
            },
            {
                "usage": {"input_tokens": 2, "output_tokens": 8, "total_tokens": 10},
                "accepted_item_count": 2,
                "request_custom_id": "b",
            },
        ]
    )

    assert report["input_tokens"] == 12
    assert report["output_tokens"] == 13
    assert report["total_tokens"] == 25
    assert report["tokens_per_accepted_item"] == 5.0


def test_retryable_timeout_then_success() -> None:
    client = FlakyClient(json.dumps(_image_output([_good_item()])))

    response = call_teacher_with_retries(
        client=client,
        payload={"model": "gpt-test"},
        config=OpenAIConfig(max_retries=2),
        sleep_fn=lambda _seconds: None,
    )

    assert client.calls == 2
    assert response.response_id == "resp_1"


def test_prepare_batch_writes_responses_request_jsonl(tmp_path: Path) -> None:
    image_path = _write_image(tmp_path / "datasets" / "textvqa" / "img.jpg")
    selection = tmp_path / "selection.jsonl"
    write_jsonl([{**_image_record(), "image_path": str(image_path)}], selection)

    report = prepare_batch(
        selection=selection,
        project_root=tmp_path / "project",
        run_id="teacher_run_test",
        openai_config=OpenAIConfig(model="gpt-test", image_detail="high"),
        limit_images=1,
    )
    request_file = Path(report["request_files"][0])
    request = json.loads(request_file.read_text().splitlines()[0])

    assert request["custom_id"] == "teacher_run_test__textvqa_123"
    assert request["method"] == "POST"
    assert request["url"] == "/v1/responses"
    assert request["body"]["text"]["format"]["strict"] is True
    assert request["body"]["input"][1]["content"][1]["image_url"].startswith("data:image/")


def test_resume_sync_uses_ledger_and_writes_outputs(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    image_path = _write_image(tmp_path / "datasets" / "textvqa" / "img.jpg")
    selection = tmp_path / "selection.jsonl"
    image_record = {**_image_record(), "image_path": str(image_path)}
    write_jsonl([image_record], selection)
    client = FlakyClient(json.dumps(_image_output([_good_item()])))

    report = resume_sync(
        selection=selection,
        project_root=project_root,
        run_id="teacher_run_test",
        openai_config=OpenAIConfig(model="gpt-test", max_retries=2),
        generation_config=GenerationConfig(
            target_accepted_samples=10,
            prompt_version=PROMPT_VERSION_V0,
            schema_version=SCHEMA_VERSION_V0,
        ),
        client=client,
    )

    run_dir = project_root / "data" / "tgvf_teacher" / "generated" / "runs" / "teacher_run_test"
    accepted = read_jsonl(run_dir / "final" / "tgvf_teacher_items.accepted.jsonl")
    ledger = read_jsonl(
        project_root
        / "data"
        / "tgvf_teacher"
        / "generated"
        / "teacher_generation_ledger.jsonl"
    )
    validated = read_jsonl(run_dir / "parsed" / "image_level_items.validated.jsonl")

    assert report["accepted_items"] == 1
    assert len(accepted) == 1
    assert ledger[-1]["status"] == "succeeded"
    assert validated[-1]["accepted_item_count"] == 1


def _image_record() -> dict:
    return {
        "stable_image_uid": "textvqa:123",
        "source_dataset": "textvqa",
        "source_profile": "scene_text",
        "image_path": "/datasets/textvqa/img.jpg",
        "source_image_id": "123",
    }


def _image_output(items: list[dict]) -> dict:
    return {
        "image_id": "textvqa:123",
        "source_dataset": "textvqa",
        "source_profile": "scene_text",
        "global_notes": "contains visible local text",
        "items": items,
    }


def _good_item() -> dict:
    return {
        "item_id": "0",
        "question": "What text is printed below the barcode?",
        "target": "the small text printed below the barcode",
        "evidence_description": "The small text below the barcode reads EXP 08/2026.",
        "short_answer": "EXP 08/2026",
        "evidence_type": "ocr_text",
        "locality": "small_region",
        "answer_type": "text_string",
        "visual_difficulty": "medium",
        "visibility": "small",
        "target_leakage_risk": "low",
        "evidence_specificity": "specific",
        "confidence": 0.93,
    }


def _visual_cue_item() -> dict:
    return {
        "item_id": "0",
        "question": "What item is visible in the small green region near the left edge?",
        "target": "the small green object near the left edge",
        "evidence_description": "A small green toy block sits near the left edge.",
        "short_answer": "toy block",
        "evidence_type": "attribute",
        "locality": "small_region",
        "answer_type": "category",
        "visual_difficulty": "clear",
        "visibility": "clear",
        "target_leakage_risk": "low",
        "evidence_specificity": "specific",
        "confidence": 0.94,
        "target_style": "visual_cue",
        "target_cues": ["location", "color", "size"],
    }


def _v3_image_output(focus_items: list[dict], direct_items: list[dict]) -> dict:
    return {
        "schema_version": SCHEMA_VERSION_V3,
        "teacher_prompt_version": PROMPT_VERSION_V3,
        "image_id": "textvqa:123",
        "source_dataset": "textvqa",
        "source_profile": "scene_text",
        "global_notes": "contains visible local text and an obvious animal",
        "focus_items": focus_items,
        "direct_items": direct_items,
    }


def _v3_focus_item() -> dict:
    return {
        **_visual_cue_item(),
        "need_focus": True,
        "evidence_state": "need_local_visual_evidence",
        "trajectory_type": "single_focus",
        "question": "What text is printed below the barcode?",
        "target": "the date-like small text printed below the barcode",
        "evidence_description": "The small text below the barcode reads EXP 08/2026.",
        "short_answer": "EXP 08/2026",
        "answer": "EXP 08/2026",
        "answer_format": "short_text",
        "value_span_text": "EXP 08/2026",
        "evidence_type": "ocr_text",
        "answer_type": "text_string",
        "visual_difficulty": "medium",
        "visibility": "small",
        "target_style": "visual_cue",
        "target_cues": ["text_like", "location", "nearby_anchor", "region_type"],
    }


def _direct_item() -> dict:
    return {
        "item_id": "direct_0",
        "need_focus": False,
        "evidence_state": "sufficient_visual_evidence",
        "trajectory_type": "direct_answer",
        "question": "What animal is in the center of the image?",
        "target": "",
        "target_style": "none",
        "target_cues": [],
        "evidence_description": "A dog is clearly visible in the center of the image.",
        "short_answer": "dog",
        "answer": "dog",
        "answer_format": "short_text",
        "value_span_text": "dog",
        "evidence_type": "attribute",
        "locality": "single_object",
        "answer_type": "category",
        "visual_difficulty": "clear",
        "visibility": "clear",
        "target_leakage_risk": "none",
        "evidence_specificity": "specific",
        "confidence": 0.95,
    }


def _write_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (160, 160), color=(40, 80, 120)).save(path)
    return path
