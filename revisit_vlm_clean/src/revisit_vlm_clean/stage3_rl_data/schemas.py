"""Schemas and deterministic helpers for Stage3 RL data generation."""

from __future__ import annotations

import hashlib
import json
import re
import string
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from revisit_vlm_clean.schema import _to_jsonable

PLAN_SCHEMA_VERSION = "stage3_rl_data_plan_v0"
IMAGE_BUNDLE_SCHEMA_VERSION = "stage3_rl_image_bundle_v0"
QA_PROMPT_SCHEMA_VERSION = "stage3_rl_qa_prompt_v0"
GENERATOR_VERSION = "stage3_rl_source_pool_v0"
FILTER_VERSION = "stage3_rl_filters_v0"
BALANCE_VERSION = "stage3_rl_balance_v0"

SUPPORTED_EVAL_METRICS = {
    "mcq",
    "exact_match",
    "normalized_exact_match",
    "numeric",
    "date",
    "token_f1",
}

SUPPORTED_ANSWER_TYPES = {
    "multiple_choice",
    "ocr_text",
    "number",
    "date",
    "short_text",
    "count",
    "boolean_state",
    "other",
}

SUPPORTED_EVIDENCE_TYPES = {
    "ocr_text",
    "document_field",
    "chart_value",
    "table_cell",
    "logo_symbol",
    "object_part",
    "color_attribute",
    "attribute",
    "texture_material",
    "pattern",
    "shape_boundary",
    "spatial_relation",
    "counting",
    "state_action",
    "diagram_visual_fact",
    "math_reasoning",
    "unknown",
}

TARGET_FOCUS_TYPES = {
    "attribute",
    "relative_position",
    "ocr",
    "document_field",
    "chart_value",
    "table_cell",
    "object_part",
    "texture_material",
    "counting",
    "other",
}

BAD_TARGET_EXACT = {
    "the answer",
    "the object",
    "the region",
    "the image",
    "the whole image",
    "the entire image",
    "something",
    "the relevant area",
    "the visible area",
}

BAD_TARGET_TASK_VERBS = {
    "answer",
    "determine",
    "decide",
    "judge",
    "infer",
    "verify",
    "compare",
    "read",
    "count",
    "whether",
}

SENSITIVE_RE = re.compile(
    r"(?:\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b|"
    r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b|"
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|"
    r"\b(?:\d[ -]*?){12,19}\b)"
)

SPECIAL_TOKEN_RE = re.compile(
    r"<\|endoftext\|>|<\|im_end\|>|<think>|</think>|<\|focus_start\|>|<\|focus_end\|>",
    re.IGNORECASE,
)

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


@dataclass(frozen=True)
class TargetSpec:
    target_text: str
    focus_type: str
    entities: tuple[str, ...] = ()
    attribute_type: str | None = None
    relation_type: str | None = None
    source: str = "rule"
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class SourceQA:
    qa_id: str
    question: str
    gold_answer: str | list[str]
    choices: tuple[str, ...] = ()
    original_choices: tuple[str, ...] = ()
    answer_aliases: tuple[str, ...] = ()
    answer_format: str = "short_text"
    answer_type: str = "short_text"
    eval_metric: str = "normalized_exact_match"
    evidence_type: str = "unknown"
    difficulty: str = "unknown"
    question_family: str | None = None
    reference_target: str | None = None
    target_spec: TargetSpec | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class ImageBundle:
    bundle_id: str
    stable_image_uid: str
    image_id: str
    image_path: str
    image_sha256: str | None
    image_phash: str | None
    width: int
    height: int
    source_dataset: str
    source_split: str
    source_profile: str
    source_record_id: str
    metadata: dict[str, Any]
    qa_items: tuple[SourceQA, ...]
    schema_version: str = IMAGE_BUNDLE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(*parts: Any, length: int = 24) -> str:
    payload = json.dumps([str(part) for part in parts], ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def stable_image_uid(source_dataset: str, source_image_id: str | int | None, image_path: str = "") -> str:
    source_image = str(source_image_id or "").strip()
    if not source_image:
        source_image = stable_hash("image_path", Path(image_path).as_posix(), length=16)
    return f"{source_dataset}:{source_image}"


def stable_bundle_id(source_dataset: str, image_uid: str) -> str:
    return stable_hash("stage3rl", "bundle", source_dataset, image_uid)


def stable_qa_id(source_dataset: str, source_record_id: str, question: str, gold_answer: Any) -> str:
    return stable_hash("stage3rl", "qa", source_dataset, source_record_id, question, gold_answer)


def stable_sample_id(bundle_id: str, qa_id: str) -> str:
    return stable_hash("stage3rl", "sample", bundle_id, qa_id)


def normalize_text(value: Any) -> str:
    text = str(value or "").lower().strip()
    text = SPECIAL_TOKEN_RE.sub(" ", text)
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def normalize_answer(value: Any) -> str:
    text = normalize_text(value)
    number = parse_number(text)
    if number is not None:
        if float(number).is_integer():
            return str(int(number))
        return str(number)
    return text


def parse_number(value: Any) -> float | None:
    text = str(value or "").strip().lower().replace(",", "")
    if text in _NUMBER_WORDS:
        return float(_NUMBER_WORDS[text])
    match = re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", text)
    if not match:
        return None
    try:
        return float(text.rstrip("%"))
    except ValueError:
        return None


def looks_like_date(value: Any) -> bool:
    text = str(value or "").strip()
    if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text):
        return True
    if re.search(
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b",
        text,
        re.IGNORECASE,
    ):
        return bool(re.search(r"\d", text))
    return False


def infer_answer_type(
    answer: Any,
    *,
    question: str = "",
    choices: list[str] | tuple[str, ...] | None = None,
    evidence_type: str = "unknown",
) -> str:
    if choices:
        return "multiple_choice"
    normalized = normalize_answer(answer)
    q_norm = normalize_text(question)
    if normalized in {"yes", "no", "true", "false"}:
        return "boolean_state"
    if looks_like_date(answer):
        return "date"
    if parse_number(answer) is not None:
        if "how many" in q_norm or "number of" in q_norm:
            return "count"
        return "number"
    if evidence_type == "ocr_text":
        return "ocr_text"
    if len(normalized.split()) <= 5:
        return "short_text"
    return "other"


def infer_eval_metric(answer_type: str, choices: list[str] | tuple[str, ...] | None = None) -> str:
    if choices or answer_type == "multiple_choice":
        return "mcq"
    if answer_type in {"number", "count"}:
        return "numeric"
    if answer_type == "date":
        return "date"
    if answer_type in {"short_text", "ocr_text", "boolean_state"}:
        return "normalized_exact_match"
    return "token_f1"


def infer_difficulty(
    *,
    question: str,
    answer_type: str,
    evidence_type: str,
    source_profile: str,
    choices: list[str] | tuple[str, ...] | None = None,
) -> str:
    q_words = len(str(question or "").split())
    if choices:
        return "easy"
    if source_profile in {"document", "chart", "table"}:
        return "medium" if q_words < 14 else "hard"
    if evidence_type in {"spatial_relation", "counting", "object_part", "chart_value", "table_cell"}:
        return "medium" if q_words < 12 else "hard"
    if answer_type in {"number", "date", "ocr_text"}:
        return "medium"
    return "easy" if q_words <= 8 else "medium"


def infer_evidence_type(source_dataset: str, source_profile: str, question: str) -> str:
    q = normalize_text(question)
    if source_dataset == "textvqa" or source_profile == "scene_text":
        if any(word in q for word in ("logo", "brand")):
            return "logo_symbol"
        if any(word in q for word in ("number", "date", "time", "price")):
            return "ocr_text"
        return "ocr_text"
    if source_dataset == "docvqa" or source_profile == "document":
        if "table" in q or "cell" in q:
            return "table_cell"
        return "document_field"
    if source_dataset == "chartqa" or source_profile == "chart":
        if "table" in q:
            return "table_cell"
        return "chart_value"
    if "how many" in q or "number of" in q:
        return "counting"
    if any(word in q for word in ("where", "left", "right", "above", "below", "behind", "front")):
        return "spatial_relation"
    if any(word in q for word in ("part", "handle", "wheel", "leg", "arm", "head")):
        return "object_part"
    if any(word in q for word in ("texture", "material", "wood", "metal", "plastic")):
        return "texture_material"
    if any(word in q for word in ("doing", "holding", "wearing", "standing", "sitting")):
        return "state_action"
    return "attribute"


def build_rule_target_spec(
    *,
    question: str,
    answer: Any,
    answer_aliases: list[str] | tuple[str, ...] | None,
    choices: list[str] | tuple[str, ...] | None,
    source_dataset: str,
    source_profile: str,
    evidence_type: str,
) -> TargetSpec | None:
    anchor = _question_anchor(question)
    forbidden = _forbidden_values(answer, answer_aliases, choices)
    anchor = _remove_forbidden_tokens(anchor, forbidden)
    if not anchor:
        anchor = _fallback_anchor(evidence_type, source_profile)

    if source_profile == "document" or evidence_type == "document_field":
        text = f"the document text field for {anchor} on the page"
    elif source_profile == "chart" or evidence_type in {"chart_value", "table_cell"}:
        text = f"the chart mark label or axis area for {anchor}"
    elif evidence_type in {"ocr_text", "logo_symbol"}:
        text = f"the text or logo region for {anchor} in the image"
    elif evidence_type == "counting":
        text = f"the visible group of {anchor} in the image"
    elif evidence_type == "spatial_relation":
        text = f"the objects and nearby layout for {anchor} in the image"
    elif evidence_type == "object_part":
        text = f"the object part area for {anchor} in the image"
    elif evidence_type == "texture_material":
        text = f"the surface detail for {anchor} in the image"
    else:
        text = f"the {anchor} area and nearby visual details in the image"

    text = _trim_words(" ".join(text.split()), max_words=24)
    focus_type = _focus_type_for_evidence(evidence_type)
    spec = TargetSpec(
        target_text=text,
        focus_type=focus_type,
        entities=tuple(_entity_words(anchor)),
        source="rule",
        confidence=0.55,
    )
    if not validate_target_spec(
        spec,
        answer=answer,
        answer_aliases=answer_aliases or (),
        choices=choices or (),
    ):
        return spec
    return None


def validate_target_spec(
    spec: TargetSpec | dict[str, Any] | None,
    *,
    answer: Any,
    answer_aliases: list[str] | tuple[str, ...] = (),
    choices: list[str] | tuple[str, ...] = (),
) -> list[str]:
    reasons: list[str] = []
    if spec is None:
        return ["missing_target_spec"]
    if isinstance(spec, dict):
        target_text = str(spec.get("target_text") or "")
        focus_type = str(spec.get("focus_type") or "other")
    else:
        target_text = spec.target_text
        focus_type = spec.focus_type
    words = target_text.split()
    normalized = normalize_text(target_text)
    if not target_text.strip():
        reasons.append("missing_target_text")
    if len(words) < 6 or len(words) > 24:
        reasons.append("target_length_out_of_range")
    if focus_type not in TARGET_FOCUS_TYPES:
        reasons.append("unsupported_focus_type")
    if SPECIAL_TOKEN_RE.search(target_text) or "|" in target_text:
        reasons.append("target_contains_special_token")
    if target_is_generic(target_text):
        reasons.append("generic_target")
    if contains_answer_or_choice(target_text, answer, answer_aliases, choices):
        reasons.append("target_leakage")
    if contains_sensitive_identifier(target_text):
        reasons.append("sensitive_personal_info")
    if target_contains_task_intent(normalized):
        reasons.append("target_contains_task_verb")
    return sorted(set(reasons))


def target_contains_task_intent(normalized_target_text: str) -> bool:
    """Reject imperative/instruction-like targets without banning noun phrases.

    A target such as "the line showing page count" is a normal visual locator.
    A target such as "count the people" or "area to read the label" asks the
    target planner to solve the task, which should be rejected.
    """
    normalized = normalize_text(normalized_target_text)
    if not normalized:
        return False
    task_verbs = sorted(BAD_TARGET_TASK_VERBS - {"whether"})
    task_verb_alt = "|".join(re.escape(verb) for verb in task_verbs)
    gerunds = {
        "answering",
        "determining",
        "deciding",
        "judging",
        "inferring",
        "verifying",
        "comparing",
        "reading",
        "counting",
    }
    gerund_alt = "|".join(sorted(gerunds))
    return bool(
        re.search(rf"^(?:{task_verb_alt})\b", normalized)
        or re.search(rf"\b(?:to|for)\s+(?:{task_verb_alt})\b", normalized)
        or re.search(rf"\b(?:for|while)\s+(?:{gerund_alt})\b", normalized)
        or re.search(r"\bwhether\s+(?:the|it|there|any)\b", normalized)
    )


def target_is_generic(target_text: str) -> bool:
    normalized = normalize_text(target_text)
    if normalized in BAD_TARGET_EXACT:
        return True
    if normalized in {"object", "region", "area", "image"}:
        return True
    content_words = [word for word in normalized.split() if len(word) > 2]
    if len(content_words) < 4:
        return True
    if "whole image" in normalized or "entire image" in normalized:
        return True
    return False


def contains_answer_or_choice(
    text: str,
    answer: Any,
    answer_aliases: list[str] | tuple[str, ...] = (),
    choices: list[str] | tuple[str, ...] = (),
) -> bool:
    normalized = normalize_text(text)
    for value in _forbidden_values(answer, answer_aliases, choices):
        value_norm = normalize_text(value)
        if _forbidden_value_matches(normalized, value_norm):
            return True
    return False


def contains_sensitive_identifier(text: str) -> bool:
    return bool(SENSITIVE_RE.search(str(text or "")))


def qa_prompt_from_bundle_item(
    *,
    bundle: dict[str, Any],
    qa_item: dict[str, Any],
    data_plan_id: str,
    created_at: str,
) -> dict[str, Any]:
    sample_id = stable_sample_id(str(bundle["bundle_id"]), str(qa_item["qa_id"]))
    qa_metadata = qa_item.get("metadata") or {}
    provenance = str(qa_metadata.get("stage3_provenance") or qa_metadata.get("provenance") or "source_qa")
    tool_need_hint = qa_metadata.get("tool_need_hint")
    return {
        "schema_version": QA_PROMPT_SCHEMA_VERSION,
        "sample_id": sample_id,
        "bundle_id": bundle["bundle_id"],
        "stable_image_uid": bundle["stable_image_uid"],
        "qa_id": qa_item["qa_id"],
        "image_path": bundle["image_path"],
        "image_sha256": bundle.get("image_sha256"),
        "source_dataset": bundle["source_dataset"],
        "source_split": bundle["source_split"],
        "source_profile": bundle["source_profile"],
        "question": qa_item["question"],
        "choices": qa_item.get("choices") or [],
        "original_choices": qa_item.get("original_choices") or [],
        "gold_answer": qa_item.get("gold_answer"),
        "answer_aliases": qa_item.get("answer_aliases") or [],
        "answer_format": qa_item.get("answer_format") or "short_text",
        "answer_type": qa_item.get("answer_type", "other"),
        "eval_metric": qa_item.get("eval_metric", "normalized_exact_match"),
        "evidence_type": qa_item.get("evidence_type", "unknown"),
        "difficulty": qa_item.get("difficulty", "unknown"),
        "provenance": provenance,
        "tool_need_hint": tool_need_hint,
        "question_family": qa_item.get("question_family"),
        "reference_target": qa_item.get("reference_target"),
        "target_spec": qa_item.get("target_spec"),
        "forced_probe": {
            "forced_on_reference_available": bool((qa_item.get("target_spec") or {}).get("target_text")),
            "forced_on_self_target_allowed": True,
            "tool_decision_label": None,
        },
        "rl_metadata": {
            "is_rl_train_eligible": True,
            "is_validation_reserved": False,
            "notes": [],
        },
        "generation_metadata": {
            "data_plan_id": data_plan_id,
            "created_at": created_at,
            "generator_version": GENERATOR_VERSION,
            "filter_version": FILTER_VERSION,
            "balance_version": BALANCE_VERSION,
        },
        "source_metadata": {
            "source_record_id": bundle.get("source_record_id"),
            "qa_metadata": qa_metadata,
            "bundle_metadata": bundle.get("metadata") or {},
        },
    }


def _forbidden_values(
    answer: Any,
    answer_aliases: list[str] | tuple[str, ...] | None,
    choices: list[str] | tuple[str, ...] | None,
) -> list[str]:
    values: list[str] = []
    if isinstance(answer, list):
        values.extend(str(item) for item in answer)
    else:
        values.append(str(answer or ""))
    values.extend(str(item) for item in (answer_aliases or ()))
    values.extend(str(item) for item in (choices or ()))
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        norm = normalize_text(value)
        if norm and norm not in seen:
            seen.add(norm)
            deduped.append(value)
    return deduped


def _forbidden_value_matches(text_norm: str, value_norm: str) -> bool:
    if not value_norm:
        return False
    if value_norm in {"yes", "no", "true", "false"}:
        return False
    if len(value_norm) <= 1:
        return False
    value_number = parse_number(value_norm)
    if value_number is not None:
        return bool(re.search(rf"(?<!\d){re.escape(str(int(value_number)))}(?!\d)", text_norm))
    if len(value_norm) < 3 and not value_norm.isdigit():
        return False
    return f" {value_norm} " in f" {text_norm} "


def _remove_forbidden_tokens(anchor: str, forbidden: list[str]) -> str:
    text = anchor
    for value in forbidden:
        norm = normalize_text(value)
        if not norm or norm in {"yes", "no", "true", "false"}:
            continue
        if len(norm) < 3:
            continue
        text = re.sub(rf"\b{re.escape(norm)}\b", " ", normalize_text(text), flags=re.IGNORECASE)
    return " ".join(text.split())


def _question_anchor(question: str) -> str:
    q = normalize_text(question)
    replacements = [
        "what is",
        "what are",
        "what does",
        "what do",
        "which",
        "where is",
        "where are",
        "how many",
        "how much",
        "is there",
        "are there",
        "is",
        "are",
        "does",
        "do",
        "shown",
        "displayed",
        "mentioned",
        "value",
    ]
    for phrase in replacements:
        q = re.sub(rf"\b{re.escape(phrase)}\b", " ", q)
    q = re.sub(
        r"\b(what|which|where|who|whom|whose|why|how|in|on|at|of|for|to|with|from|by|this|that|there|image|picture)\b",
        " ",
        q,
    )
    q = " ".join(q.split())
    words = [word for word in q.split() if len(word) > 1]
    return " ".join(words[:8])


def _fallback_anchor(evidence_type: str, source_profile: str) -> str:
    if evidence_type in {"ocr_text", "logo_symbol"}:
        return "the relevant visible text"
    if source_profile == "document":
        return "the relevant document field"
    if source_profile == "chart":
        return "the referenced chart element"
    if evidence_type == "counting":
        return "the relevant object group"
    return "the referenced visual object"


def _focus_type_for_evidence(evidence_type: str) -> str:
    return {
        "ocr_text": "ocr",
        "logo_symbol": "ocr",
        "document_field": "document_field",
        "chart_value": "chart_value",
        "table_cell": "table_cell",
        "object_part": "object_part",
        "texture_material": "texture_material",
        "spatial_relation": "relative_position",
        "counting": "counting",
        "attribute": "attribute",
    }.get(evidence_type, "other")


def _entity_words(anchor: str) -> list[str]:
    stop = {"visible", "relevant", "nearby", "image", "picture", "area", "region"}
    return [word for word in normalize_text(anchor).split() if len(word) > 2 and word not in stop][:6]


def _trim_words(text: str, *, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])
