from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION_V4 = "tgvf_teacher_schema_v4"
TEACHER_VERSION_V4 = "tgvf_v4_teacher"

ITEM_TYPES = [
    "single_refocus",
    "multi_refocus",
    "no_refocus_continue",
    "no_refocus_answer",
]
ANSWER_FORMATS_V4 = ["open", "multiple_choice"]
QUESTION_TYPES = [
    "color_attribute",
    "texture_material",
    "pattern",
    "shape_boundary",
    "object_part",
    "spatial_relation",
    "counting",
    "state_action",
    "ocr_text",
    "number_reading",
    "chart_table",
    "diagram_visual_fact",
    "math_reasoning",
    "whole_image_obvious",
    "other",
]
FOCUS_CATEGORIES = [
    "color_surface",
    "surface_detail",
    "material_cue",
    "pattern_detail",
    "shape_boundary",
    "object_part_detail",
    "relation_layout",
    "group_counting",
    "contact_state",
    "text_region",
    "number_region",
    "chart_table_region",
    "diagram_region",
    "none",
]
TRACE_STEP_TYPES = ["think", "focus", "answer"]
EVIDENCE_TYPES_V4 = [
    "ocr_text",
    "number",
    "document_field",
    "chart_value",
    "table_cell",
    "logo_symbol",
    "object_part",
    "color_attribute",
    "texture_material",
    "pattern",
    "shape_boundary",
    "spatial_relation",
    "counting",
    "state_action",
    "diagram_detail",
    "math_visual_fact",
    "other",
]
FOCUS_DESCRIPTOR_CUES = [
    "location",
    "nearby_anchor",
    "surface",
    "texture",
    "pattern",
    "color",
    "shape",
    "size",
    "text_like",
    "number_like",
    "chart_anchor",
    "table_anchor",
    "object_part",
    "material_cue",
    "relation",
    "contact_area",
]
LEAKAGE_RISKS = ["none", "low", "medium", "high"]
REFocus_ITEM_TYPES = {"single_refocus", "multi_refocus"}
NO_REFOCUS_ITEM_TYPES = {"no_refocus_continue", "no_refocus_answer"}
TEXT_LIKE_QUESTION_TYPES = {"ocr_text", "number_reading", "chart_table"}
TEXT_LIKE_FOCUS_CATEGORIES = {"text_region", "number_region", "chart_table_region"}
VISUAL_DETAIL_QUESTION_TYPES = {
    "color_attribute",
    "texture_material",
    "pattern",
    "shape_boundary",
    "object_part",
    "spatial_relation",
    "counting",
    "state_action",
}
BAD_FOCUS_TEXT_EXACT = {
    "the glove",
    "the object",
    "the upper-left area",
    "something",
    "texture_material",
    "tight_crop",
    "the answer",
}
BAD_FOCUS_TASK_VERBS = {
    "determine",
    "judge",
    "answer",
    "verify",
    "decide",
    "infer",
    "whether",
    "inspect",
    "compare",
    "count",
    "read",
}
SPECIAL_TOKEN_RE = re.compile(
    r"<\|endoftext\|>|<\|im_end\|>|<think>|</think>|<\|focus_start\|>|<\|focus_end\|>",
    re.IGNORECASE,
)
SENSITIVE_RE = re.compile(
    r"(?:\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\b(?:\d[ -]*?){12,19}\b)"
)


@dataclass(frozen=True)
class Choice:
    label: str
    text: str


@dataclass(frozen=True)
class FocusMetadata:
    evidence_type: str
    focus_descriptor_cues: tuple[str, ...]
    target_leakage_risk: str
    confidence: float


@dataclass(frozen=True)
class Quality:
    confidence: float
    num_focus_steps: int
    notes: str | None = None


@dataclass(frozen=True)
class ThinkStep:
    text: str
    type: str = "think"


@dataclass(frozen=True)
class FocusStep:
    focus_text: str
    focused_evidence: str
    metadata: FocusMetadata
    type: str = "focus"


@dataclass(frozen=True)
class AnswerStep:
    text: str
    type: str = "answer"


@dataclass(frozen=True)
class Item:
    item_id: str | None
    item_type: str
    answer_format: str
    question_type: str
    focus_category: str
    question: str
    choices: tuple[Choice, ...]
    correct_choice: str | None
    answer_text: str
    trace: tuple[ThinkStep | FocusStep | AnswerStep, ...]
    quality: Quality


@dataclass(frozen=True)
class TeacherOutput:
    schema_version: str
    teacher_version: str
    image_id: str | None
    image_summary: str
    items: tuple[Item, ...]


class V4ValidationError(ValueError):
    pass


def teacher_output_schema_v4() -> dict[str, Any]:
    choice_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["label", "text"],
        "properties": {"label": {"type": "string"}, "text": {"type": "string"}},
    }
    focus_metadata_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["evidence_type", "focus_descriptor_cues", "target_leakage_risk", "confidence"],
        "properties": {
            "evidence_type": {"type": "string", "enum": EVIDENCE_TYPES_V4},
            "focus_descriptor_cues": {
                "type": "array",
                "items": {"type": "string", "enum": FOCUS_DESCRIPTOR_CUES},
            },
            "target_leakage_risk": {"type": "string", "enum": LEAKAGE_RISKS},
            "confidence": {"type": "number"},
        },
    }
    # OpenAI structured-output JSON schema supports a restricted subset; avoid oneOf/anyOf
    # and enforce step-specific requirements in validate_trace_steps instead.
    trace_step_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["type", "text", "focus_text", "focused_evidence", "metadata"],
        "properties": {
            "type": {"type": "string", "enum": TRACE_STEP_TYPES},
            "text": {"type": ["string", "null"]},
            "focus_text": {"type": ["string", "null"]},
            "focused_evidence": {"type": ["string", "null"]},
            "metadata": {
                "type": ["object", "null"],
                "additionalProperties": False,
                "required": [
                    "evidence_type",
                    "focus_descriptor_cues",
                    "target_leakage_risk",
                    "confidence",
                ],
                "properties": focus_metadata_schema["properties"],
            },
        },
    }
    quality_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["confidence", "num_focus_steps", "notes"],
        "properties": {
            "confidence": {"type": "number"},
            "num_focus_steps": {"type": "integer"},
            "notes": {"type": ["string", "null"]},
        },
    }
    item_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "item_id",
            "item_type",
            "answer_format",
            "question_type",
            "focus_category",
            "question",
            "choices",
            "correct_choice",
            "answer_text",
            "trace",
            "quality",
        ],
        "properties": {
            "item_id": {"type": ["string", "null"]},
            "item_type": {"type": "string", "enum": ITEM_TYPES},
            "answer_format": {"type": "string", "enum": ANSWER_FORMATS_V4},
            "question_type": {"type": "string", "enum": QUESTION_TYPES},
            "focus_category": {"type": "string", "enum": FOCUS_CATEGORIES},
            "question": {"type": "string"},
            "choices": {"type": "array", "items": choice_schema},
            "correct_choice": {"type": ["string", "null"]},
            "answer_text": {"type": "string"},
            "trace": {"type": "array", "items": trace_step_schema},
            "quality": quality_schema,
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "teacher_version", "image_id", "image_summary", "items"],
        "properties": {
            "schema_version": {"type": "string", "enum": [SCHEMA_VERSION_V4]},
            "teacher_version": {"type": "string", "enum": [TEACHER_VERSION_V4]},
            "image_id": {"type": ["string", "null"]},
            "image_summary": {"type": "string"},
            "items": {"type": "array", "items": item_schema},
        },
    }


def parse_teacher_output_v4(data: dict[str, Any]) -> TeacherOutput:
    reasons = validate_teacher_output_shape_v4(data)
    if reasons:
        raise V4ValidationError(",".join(reasons))
    return TeacherOutput(
        schema_version=data["schema_version"],
        teacher_version=data["teacher_version"],
        image_id=data.get("image_id"),
        image_summary=str(data.get("image_summary") or ""),
        items=tuple(_parse_item(item) for item in data.get("items") or []),
    )


def validate_teacher_output_shape_v4(data: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    allowed = {"schema_version", "teacher_version", "image_id", "image_summary", "items"}
    reasons.extend(_extra_or_missing_keys(data, allowed, allowed, "top"))
    if data.get("schema_version") != SCHEMA_VERSION_V4:
        reasons.append("invalid_schema_version")
    if data.get("teacher_version") != TEACHER_VERSION_V4:
        reasons.append("invalid_teacher_version")
    if not str(data.get("image_summary") or "").strip():
        reasons.append("missing_image_summary")
    if not isinstance(data.get("items"), list):
        reasons.append("items_not_list")
    return sorted(set(reasons))


def validate_v4_item(item: dict[str, Any], *, confidence_threshold: float = 0.75) -> list[str]:
    reasons: list[str] = []
    allowed = {
        "item_id",
        "item_type",
        "answer_format",
        "question_type",
        "focus_category",
        "question",
        "choices",
        "correct_choice",
        "answer_text",
        "trace",
        "quality",
    }
    required = allowed - {"item_id"}
    reasons.extend(_extra_or_missing_keys(item, allowed, required, "item"))
    item_type = str(item.get("item_type") or "")
    answer_format = str(item.get("answer_format") or "")
    question_type = str(item.get("question_type") or "")
    focus_category = str(item.get("focus_category") or "")
    if item_type not in ITEM_TYPES:
        reasons.append("invalid_item_type")
    if answer_format not in ANSWER_FORMATS_V4:
        reasons.append("invalid_answer_format")
    if question_type not in QUESTION_TYPES:
        reasons.append("invalid_question_type")
    if focus_category not in FOCUS_CATEGORIES:
        reasons.append("invalid_focus_category")
    if item_type in NO_REFOCUS_ITEM_TYPES and focus_category != "none":
        reasons.append("no_refocus_focus_category_not_none")
    if item_type in REFocus_ITEM_TYPES and focus_category == "none":
        reasons.append("refocus_focus_category_none")
    if not str(item.get("question") or "").strip():
        reasons.append("missing_question")
    if not str(item.get("answer_text") or "").strip():
        reasons.append("missing_answer_text")
    quality = item.get("quality")
    if not isinstance(quality, dict):
        reasons.append("quality_not_object")
    else:
        reasons.extend(_extra_or_missing_keys(quality, {"confidence", "num_focus_steps", "notes"}, {"confidence", "num_focus_steps"}, "quality"))
        if _float(quality.get("confidence")) < confidence_threshold:
            reasons.append("low_confidence")
    reasons.extend(validate_trace_shape(item))
    reasons.extend(validate_multiple_choice(item))
    reasons.extend(validate_focus_text_no_answer_leakage(item))
    reasons.extend(validate_focus_text_style(item))
    reasons.extend(validate_trace_steps(item, confidence_threshold=confidence_threshold))
    if contains_sensitive_identifier(json.dumps(item, ensure_ascii=False)):
        reasons.append("sensitive_personal_identifier")
    return sorted(set(reasons))


def validate_trace_shape(item: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    trace = item.get("trace")
    if not isinstance(trace, list) or not trace:
        return ["trace_not_list"]
    step_types = [str(step.get("type")) if isinstance(step, dict) else "invalid" for step in trace]
    focus_count = step_types.count("focus")
    item_type = str(item.get("item_type") or "")
    quality = item.get("quality") if isinstance(item.get("quality"), dict) else {}
    try:
        declared_focus_steps = int(quality.get("num_focus_steps"))
    except (TypeError, ValueError):
        declared_focus_steps = -1
    if declared_focus_steps != focus_count:
        reasons.append("quality_num_focus_steps_mismatch")
    if item_type == "single_refocus":
        if focus_count != 1:
            reasons.append("single_refocus_focus_count_not_one")
        if len(trace) < 4 or step_types[0] != "think" or "focus" not in step_types or step_types[-1] != "answer":
            reasons.append("invalid_single_refocus_trace_shape")
        else:
            focus_index = step_types.index("focus")
            if focus_index == 0 or focus_index >= len(trace) - 2 or step_types[focus_index + 1] != "think":
                reasons.append("invalid_single_refocus_trace_shape")
    elif item_type == "multi_refocus":
        if step_types != ["think", "focus", "think", "focus", "think", "answer"]:
            reasons.append("invalid_multi_refocus_trace_shape")
        if focus_count != 2:
            reasons.append("multi_refocus_focus_count_not_two")
        if len(trace) >= 4 and isinstance(trace[1], dict) and isinstance(trace[3], dict):
            if normalize_text(trace[1].get("focus_text", "")) == normalize_text(trace[3].get("focus_text", "")):
                reasons.append("multi_refocus_repeated_focus_text")
    elif item_type in NO_REFOCUS_ITEM_TYPES:
        if focus_count != 0:
            reasons.append("no_refocus_has_focus_steps")
        if len(trace) < 2 or step_types[0] != "think" or step_types[-1] != "answer":
            reasons.append("invalid_no_refocus_trace_shape")
    return reasons


def validate_multiple_choice(item: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    choices = item.get("choices")
    correct = item.get("correct_choice")
    answer_format = item.get("answer_format")
    if not isinstance(choices, list):
        return ["choices_not_list"]
    labels = [str(choice.get("label") or "") for choice in choices if isinstance(choice, dict)]
    if answer_format == "multiple_choice":
        if not (2 <= len(choices) <= 4):
            reasons.append("multiple_choice_count_out_of_range")
        if len(set(labels)) != len(labels) or not all(label for label in labels):
            reasons.append("invalid_choice_labels")
        if correct not in set(labels):
            reasons.append("correct_choice_not_in_choices")
        correct_text = _choice_text(item, str(correct or ""))
        answer_step = _answer_step_text(item)
        if correct and str(correct) not in answer_step:
            reasons.append("answer_step_missing_correct_choice_label")
        if correct_text and normalize_text(correct_text) not in normalize_text(answer_step):
            reasons.append("answer_step_missing_correct_choice_text")
    else:
        if correct is not None:
            reasons.append("open_answer_has_correct_choice")
    for choice in choices:
        if not isinstance(choice, dict):
            reasons.append("choice_not_object")
            continue
        reasons.extend(_extra_or_missing_keys(choice, {"label", "text"}, {"label", "text"}, "choice"))
        if not str(choice.get("text") or "").strip():
            reasons.append("empty_choice_text")
    return reasons


def validate_focus_text_no_answer_leakage(item: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    answer_values = _answer_values(item)
    focus_steps = _focus_steps(item)
    for focus_index, step in enumerate(focus_steps):
        focus_text = str(step.get("focus_text") or "")
        normalized_focus = normalize_text(focus_text)
        for answer in answer_values:
            if answer and _contains_value(normalized_focus, normalize_text(answer)):
                reasons.append("answer_value_in_focus_text")
                break
        if SPECIAL_TOKEN_RE.search(focus_text) or "|" in focus_text:
            reasons.append("focus_text_contains_special_token_or_pipe")
        metadata = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
        if metadata.get("target_leakage_risk") in {"medium", "high"}:
            reasons.append("target_leakage_risk_too_high")
        if not str(step.get("focused_evidence") or "").strip():
            reasons.append("missing_focused_evidence")
        if focus_index == 0:
            pre_text = _pre_first_focus_text(item)
            for answer in answer_values:
                if answer and _contains_value(normalize_text(pre_text), normalize_text(answer)):
                    reasons.append("answer_value_in_pre_focus_think")
                    break
    return reasons


def validate_focus_text_style(item: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for step in _focus_steps(item):
        focus_text = str(step.get("focus_text") or "").strip()
        normalized = normalize_text(focus_text)
        words = _words(focus_text)
        if not focus_text:
            reasons.append("missing_focus_text")
            continue
        if len(words) < 6 or len(words) > 24:
            reasons.append("focus_text_length_out_of_range")
        if normalized in BAD_FOCUS_TEXT_EXACT:
            reasons.append("generic_focus_text")
        if len(words) <= 2:
            reasons.append("bare_object_focus_text")
        if focus_text.endswith("?"):
            reasons.append("focus_text_looks_like_question")
        if any(verb in normalized.split() for verb in BAD_FOCUS_TASK_VERBS):
            reasons.append("focus_text_contains_task_verb")
        metadata = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
        cues = metadata.get("focus_descriptor_cues") if isinstance(metadata, dict) else []
        if isinstance(cues, list) and not cues:
            reasons.append("missing_focus_descriptor_cues")
    return reasons


def validate_trace_steps(item: dict[str, Any], *, confidence_threshold: float) -> list[str]:
    reasons: list[str] = []
    trace = item.get("trace") if isinstance(item.get("trace"), list) else []
    for index, step in enumerate(trace):
        if not isinstance(step, dict):
            reasons.append("trace_step_not_object")
            continue
        step_type = step.get("type")
        if step_type not in TRACE_STEP_TYPES:
            reasons.append("invalid_trace_step_type")
            continue
        if step_type in {"think", "answer"}:
            reasons.extend(
                _extra_or_missing_keys(
                    step,
                    {"type", "text", "focus_text", "focused_evidence", "metadata"},
                    {"type", "text"},
                    f"trace_{index}",
                )
            )
            if not str(step.get("text") or "").strip():
                reasons.append(f"empty_{step_type}_text")
        elif step_type == "focus":
            reasons.extend(
                _extra_or_missing_keys(
                    step,
                    {"type", "text", "focus_text", "focused_evidence", "metadata"},
                    {"type", "focus_text", "focused_evidence", "metadata"},
                    f"trace_{index}",
                )
            )
            metadata = step.get("metadata")
            if not isinstance(metadata, dict):
                reasons.append("focus_metadata_not_object")
                continue
            reasons.extend(
                _extra_or_missing_keys(
                    metadata,
                    {"evidence_type", "focus_descriptor_cues", "target_leakage_risk", "confidence"},
                    {"evidence_type", "focus_descriptor_cues", "target_leakage_risk", "confidence"},
                    f"trace_{index}_metadata",
                )
            )
            if metadata.get("evidence_type") not in EVIDENCE_TYPES_V4:
                reasons.append("invalid_evidence_type")
            if metadata.get("target_leakage_risk") not in LEAKAGE_RISKS:
                reasons.append("invalid_target_leakage_risk")
            if _float(metadata.get("confidence")) < confidence_threshold:
                reasons.append("low_focus_confidence")
            cues = metadata.get("focus_descriptor_cues")
            if not isinstance(cues, list):
                reasons.append("focus_descriptor_cues_not_list")
            elif any(cue not in FOCUS_DESCRIPTOR_CUES for cue in cues):
                reasons.append("invalid_focus_descriptor_cue")
    return reasons


def validate_item_distribution_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    total = max(len(items), 1)
    item_type_counts = Counter(str(item.get("item_type") or "unknown") for item in items)
    answer_format_counts = Counter(str(item.get("answer_format") or "unknown") for item in items)
    question_counts = Counter(str(item.get("question_type") or "unknown") for item in items)
    focus_counts = Counter(str(item.get("focus_category") or "unknown") for item in items)
    cue_counts = Counter(cue for item in items for cue in _all_focus_cues(item))
    warnings: list[str] = []
    no_refocus_ratio = (item_type_counts["no_refocus_continue"] + item_type_counts["no_refocus_answer"]) / total
    multi_ratio = item_type_counts["multi_refocus"] / total
    mc_ratio = answer_format_counts["multiple_choice"] / total
    ocr_like_ratio = (question_counts["ocr_text"] + question_counts["number_reading"]) / total
    if no_refocus_ratio > 0.25:
        warnings.append("no_refocus_ratio_above_soft_max")
    if items and not (0.03 <= multi_ratio <= 0.12):
        warnings.append("multi_refocus_ratio_outside_target")
    if items and not (0.35 <= mc_ratio <= 0.65):
        warnings.append("multiple_choice_ratio_outside_target")
    if ocr_like_ratio > 0.15:
        warnings.append("ocr_number_ratio_above_cap")
    visual_detail_count = sum(question_counts[key] for key in VISUAL_DETAIL_QUESTION_TYPES)
    if items and visual_detail_count / total < 0.55:
        warnings.append("visual_detail_categories_underrepresented")
    return {
        "total": len(items),
        "item_type_distribution": dict(item_type_counts),
        "answer_format_distribution": dict(answer_format_counts),
        "question_type_distribution": dict(question_counts),
        "focus_category_distribution": dict(focus_counts),
        "focus_descriptor_cues_distribution": dict(cue_counts),
        "warnings": warnings,
    }


def rebalance_v4_items(
    items: list[dict[str, Any]],
    *,
    max_no_refocus_ratio: float = 0.25,
    max_ocr_number_ratio: float = 0.15,
    target_multiple_choice_range: tuple[float, float] = (0.40, 0.60),
) -> list[dict[str, Any]]:
    del target_multiple_choice_range  # Distribution is reported but not forced by this conservative hook.
    ranked = sorted(items, key=_rebalance_priority, reverse=True)
    selected: list[dict[str, Any]] = []
    for item in ranked:
        candidate = [*selected, item]
        total = len(candidate)
        no_refocus = sum(1 for row in candidate if row.get("item_type") in NO_REFOCUS_ITEM_TYPES)
        ocr_like = sum(1 for row in candidate if row.get("question_type") in {"ocr_text", "number_reading"})
        if total and no_refocus / total > max_no_refocus_ratio:
            continue
        if total and ocr_like / total > max_ocr_number_ratio:
            continue
        selected.append(item)
    return sorted(selected, key=lambda item: items.index(item))


def render_tgvf_v4_sft_item(
    item: dict[str, Any],
    *,
    include_choices: bool = True,
    d_placeholder: bool = True,
    include_image_placeholder: bool = True,
) -> str:
    lines: list[str] = []
    if include_image_placeholder:
        lines.append("<image>")
    lines.append(f"Question: {str(item.get('question') or '').strip()}")
    choices = item.get("choices") if isinstance(item.get("choices"), list) else []
    if include_choices and choices:
        lines.append("Choices:")
        for choice in choices:
            if isinstance(choice, dict):
                lines.append(f"{choice.get('label')}. {choice.get('text')}")
    focus_index = 0
    for step in item.get("trace") or []:
        if not isinstance(step, dict):
            continue
        step_type = step.get("type")
        if step_type == "think":
            lines.append(f"<think>{str(step.get('text') or '').strip()}</think>")
        elif step_type == "focus":
            focus_index += 1
            lines.append(f"<|focus_start|>{str(step.get('focus_text') or '').strip()}<|focus_end|>")
            if d_placeholder:
                lines.append(
                    f"Input: <|tgvf_start|>[D visual embeddings {focus_index}]<|tgvf_end|>"
                )
        elif step_type == "answer":
            lines.append(str(step.get("text") or "").strip())
    return "\n".join(line for line in lines if line != "")


def render_tgvf_v4_sft_batch(items: list[dict[str, Any]], **kwargs: Any) -> list[str]:
    return [render_tgvf_v4_sft_item(item, **kwargs) for item in items]


def validate_v4_image_level_output(
    image_output: dict[str, Any],
    image_record: dict[str, Any],
    *,
    teacher_run_id: str,
    model: str,
    prompt_version: str = TEACHER_VERSION_V4,
    schema_version: str = SCHEMA_VERSION_V4,
    raw_response_id: str | None = None,
    confidence_threshold: float = 0.75,
    existing_item_hashes: set[str] | None = None,
    allow_duplicate_items: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    existing_item_hashes = existing_item_hashes or set()
    accepted_candidates: list[tuple[int, dict[str, Any]]] = []
    rejected: list[dict[str, Any]] = []
    top_reasons = validate_teacher_output_shape_v4(image_output)
    if top_reasons:
        return [], [
            _rejected_v4_record(
                image_record=image_record,
                item={},
                rejection_reasons=top_reasons,
                teacher_run_id=teacher_run_id,
                model=model,
                prompt_version=prompt_version,
                schema_version=schema_version,
                raw_response_id=raw_response_id,
                item_index=-1,
            )
        ]
    for index, item in enumerate(image_output.get("items") or []):
        if not isinstance(item, dict):
            rejected.append(
                _rejected_v4_record(
                    image_record=image_record,
                    item={"raw_item": item},
                    rejection_reasons=["item_not_object"],
                    teacher_run_id=teacher_run_id,
                    model=model,
                    prompt_version=prompt_version,
                    schema_version=schema_version,
                    raw_response_id=raw_response_id,
                    item_index=index,
                )
            )
            continue
        reasons = validate_v4_item(item, confidence_threshold=confidence_threshold)
        if reasons:
            rejected.append(
                _rejected_v4_record(
                    image_record=image_record,
                    item=item,
                    rejection_reasons=reasons,
                    teacher_run_id=teacher_run_id,
                    model=model,
                    prompt_version=prompt_version,
                    schema_version=schema_version,
                    raw_response_id=raw_response_id,
                    item_index=index,
                )
            )
        else:
            accepted_candidates.append((index, item))
    accepted: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    seen_focus_signatures: set[str] = set()
    for out_index, (original_index, item) in enumerate(accepted_candidates):
        content_hash = item_content_hash_v4(image_record["stable_image_uid"], item)
        reasons = []
        question_key = normalize_text(item.get("question", ""))
        focus_signature = "|".join(normalize_text(step.get("focus_text", "")) for step in _focus_steps(item))
        if question_key in seen_questions:
            reasons.append("duplicate_question")
        if focus_signature and focus_signature in seen_focus_signatures:
            reasons.append("duplicate_focus_signature")
        if content_hash in existing_item_hashes and not allow_duplicate_items:
            reasons.append("duplicate_existing_item_content_hash")
        if reasons:
            rejected.append(
                _rejected_v4_record(
                    image_record=image_record,
                    item=item,
                    rejection_reasons=reasons,
                    teacher_run_id=teacher_run_id,
                    model=model,
                    prompt_version=prompt_version,
                    schema_version=schema_version,
                    raw_response_id=raw_response_id,
                    item_index=original_index,
                )
            )
            continue
        seen_questions.add(question_key)
        if focus_signature:
            seen_focus_signatures.add(focus_signature)
        accepted.append(
            flatten_v4_item(
                image_record=image_record,
                item=item,
                teacher_run_id=teacher_run_id,
                model=model,
                prompt_version=prompt_version,
                schema_version=schema_version,
                raw_response_id=raw_response_id,
                item_index=out_index,
                image_summary=str(image_output.get("image_summary") or ""),
                content_hash=content_hash,
            )
        )
    return accepted, rejected


def flatten_v4_item(
    *,
    image_record: dict[str, Any],
    item: dict[str, Any],
    teacher_run_id: str,
    model: str,
    prompt_version: str,
    schema_version: str,
    raw_response_id: str | None,
    item_index: int,
    image_summary: str,
    content_hash: str | None = None,
) -> dict[str, Any]:
    stable_uid = image_record["stable_image_uid"]
    focus_steps = _focus_steps(item)
    focus_texts = [str(step.get("focus_text") or "").strip() for step in focus_steps]
    focused_evidence = [str(step.get("focused_evidence") or "").strip() for step in focus_steps]
    evidence_types = [str((step.get("metadata") or {}).get("evidence_type") or "") for step in focus_steps]
    focus_cues = [cue for step in focus_steps for cue in ((step.get("metadata") or {}).get("focus_descriptor_cues") or [])]
    confidence = _float((item.get("quality") or {}).get("confidence"))
    return {
        "uid": f"{teacher_run_id}:{stable_uid}:{item_index}",
        "teacher_run_id": teacher_run_id,
        "stable_image_uid": stable_uid,
        "image": image_record["image_path"],
        "image_id": stable_uid,
        "source_dataset": image_record.get("source_dataset"),
        "source_profile": image_record.get("source_profile"),
        "schema_version": schema_version,
        "teacher_version": TEACHER_VERSION_V4,
        "teacher_prompt_version": prompt_version,
        "teacher_schema_version": schema_version,
        "image_summary": image_summary,
        "item_id": item.get("item_id"),
        "item_type": item.get("item_type"),
        "answer_format": item.get("answer_format"),
        "question_type": item.get("question_type"),
        "focus_category": item.get("focus_category"),
        "question": str(item.get("question") or "").strip(),
        "choices": item.get("choices") or [],
        "correct_choice": item.get("correct_choice"),
        "answer_text": str(item.get("answer_text") or "").strip(),
        "answer": _answer_step_text(item) or str(item.get("answer_text") or "").strip(),
        "trace": item.get("trace") or [],
        "sft_text": render_tgvf_v4_sft_item(item),
        "num_focus_steps": len(focus_steps),
        "need_focus": bool(focus_steps),
        "trajectory_type": item.get("item_type"),
        "focus_texts": focus_texts,
        "focused_evidence": focused_evidence,
        "evidence_types": evidence_types,
        "focus_descriptor_cues": focus_cues,
        "target_leakage_risk": _max_leakage_risk(focus_steps),
        "confidence": confidence,
        "teacher_model": model,
        "raw_response_id": raw_response_id,
        "item_content_hash": content_hash or item_content_hash_v4(stable_uid, item),
        "created_at": now_iso(),
    }


def item_content_hash_v4(stable_image_uid: str, item: dict[str, Any]) -> str:
    payload = {
        "stable_image_uid": stable_image_uid,
        "question": item.get("question"),
        "answer_text": item.get("answer_text"),
        "item_type": item.get("item_type"),
        "focus_texts": [step.get("focus_text") for step in _focus_steps(item)],
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _rejected_v4_record(
    *,
    image_record: dict[str, Any],
    item: dict[str, Any],
    rejection_reasons: list[str],
    teacher_run_id: str,
    model: str,
    prompt_version: str,
    schema_version: str,
    raw_response_id: str | None,
    item_index: int,
) -> dict[str, Any]:
    return {
        "teacher_run_id": teacher_run_id,
        "stable_image_uid": image_record["stable_image_uid"],
        "image": image_record["image_path"],
        "image_id": image_record["stable_image_uid"],
        "source_dataset": image_record.get("source_dataset"),
        "source_profile": image_record.get("source_profile"),
        "schema_version": schema_version,
        "teacher_version": TEACHER_VERSION_V4,
        "teacher_prompt_version": prompt_version,
        "teacher_schema_version": schema_version,
        "item_index": item_index,
        "item": item,
        "item_type": item.get("item_type") if isinstance(item, dict) else None,
        "question_type": item.get("question_type") if isinstance(item, dict) else None,
        "focus_category": item.get("focus_category") if isinstance(item, dict) else None,
        "rejection_reasons": sorted(set(rejection_reasons)),
        "teacher_model": model,
        "raw_response_id": raw_response_id,
        "created_at": now_iso(),
    }


def _parse_item(item: dict[str, Any]) -> Item:
    return Item(
        item_id=item.get("item_id"),
        item_type=item["item_type"],
        answer_format=item["answer_format"],
        question_type=item["question_type"],
        focus_category=item["focus_category"],
        question=item["question"],
        choices=tuple(Choice(label=choice["label"], text=choice["text"]) for choice in item.get("choices") or []),
        correct_choice=item.get("correct_choice"),
        answer_text=item["answer_text"],
        trace=tuple(_parse_step(step) for step in item.get("trace") or []),
        quality=Quality(
            confidence=float(item["quality"]["confidence"]),
            num_focus_steps=int(item["quality"]["num_focus_steps"]),
            notes=item["quality"].get("notes"),
        ),
    )


def _parse_step(step: dict[str, Any]) -> ThinkStep | FocusStep | AnswerStep:
    if step["type"] == "think":
        return ThinkStep(text=step["text"])
    if step["type"] == "answer":
        return AnswerStep(text=step["text"])
    metadata = step["metadata"]
    return FocusStep(
        focus_text=step["focus_text"],
        focused_evidence=step["focused_evidence"],
        metadata=FocusMetadata(
            evidence_type=metadata["evidence_type"],
            focus_descriptor_cues=tuple(metadata.get("focus_descriptor_cues") or []),
            target_leakage_risk=metadata["target_leakage_risk"],
            confidence=float(metadata["confidence"]),
        ),
    )


def _extra_or_missing_keys(
    data: dict[str, Any],
    allowed: set[str],
    required: set[str],
    prefix: str,
) -> list[str]:
    reasons = []
    extra = set(data) - allowed
    missing = required - set(data)
    if extra:
        reasons.append(f"{prefix}_extra_fields:{','.join(sorted(extra))}")
    for key in sorted(missing):
        reasons.append(f"{prefix}_missing_{key}")
    return reasons


def _focus_steps(item: dict[str, Any]) -> list[dict[str, Any]]:
    return [step for step in item.get("trace") or [] if isinstance(step, dict) and step.get("type") == "focus"]


def _all_focus_cues(item: dict[str, Any]) -> list[str]:
    return [cue for step in _focus_steps(item) for cue in ((step.get("metadata") or {}).get("focus_descriptor_cues") or [])]


def _answer_step_text(item: dict[str, Any]) -> str:
    for step in reversed(item.get("trace") or []):
        if isinstance(step, dict) and step.get("type") == "answer":
            return str(step.get("text") or "").strip()
    return ""


def _choice_text(item: dict[str, Any], label: str) -> str:
    for choice in item.get("choices") or []:
        if isinstance(choice, dict) and str(choice.get("label")) == label:
            return str(choice.get("text") or "")
    return ""


def _answer_values(item: dict[str, Any]) -> list[str]:
    values = [str(item.get("answer_text") or "").strip()]
    correct = item.get("correct_choice")
    if correct:
        values.append(_choice_text(item, str(correct)))
    if item.get("answer_format") == "multiple_choice":
        values.extend(str(choice.get("text") or "") for choice in item.get("choices") or [] if isinstance(choice, dict))
    return sorted({value for value in values if value}, key=len, reverse=True)


def _pre_first_focus_text(item: dict[str, Any]) -> str:
    parts = []
    for step in item.get("trace") or []:
        if not isinstance(step, dict):
            continue
        if step.get("type") == "focus":
            break
        if step.get("type") == "think":
            parts.append(str(step.get("text") or ""))
    return " ".join(parts)


def _contains_value(haystack: str, needle: str) -> bool:
    if not needle or len(needle) <= 1:
        return False
    if len(needle) <= 3:
        return bool(re.search(rf"\b{re.escape(needle)}\b", haystack))
    return needle in haystack


def _max_leakage_risk(focus_steps: list[dict[str, Any]]) -> str:
    order = {"none": 0, "low": 1, "medium": 2, "high": 3}
    risks = [str((step.get("metadata") or {}).get("target_leakage_risk") or "none") for step in focus_steps]
    return max(risks or ["none"], key=lambda risk: order.get(risk, -1))


def _rebalance_priority(item: dict[str, Any]) -> tuple[int, float]:
    q_type = str(item.get("question_type") or "")
    item_type = str(item.get("item_type") or "")
    score = 0
    if q_type in VISUAL_DETAIL_QUESTION_TYPES:
        score += 4
    if item.get("answer_format") == "multiple_choice":
        score += 2
    if item_type == "single_refocus":
        score += 2
    if item_type == "multi_refocus":
        score += 1
    if q_type in {"ocr_text", "number_reading"}:
        score -= 3
    if item_type in NO_REFOCUS_ITEM_TYPES:
        score -= 1
    return score, _float((item.get("quality") or {}).get("confidence"))


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text)


def normalize_text(text: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def contains_sensitive_identifier(text: str) -> bool:
    return bool(SENSITIVE_RE.search(text))


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
