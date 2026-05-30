from __future__ import annotations

import re
from dataclasses import dataclass, field

from revisit_vlm.tgvf import FOVEATE_END, FOVEATE_START
from tgvf_eval.prompts import strip_answer_choices

TARGET_EXAMPLES = {
    "the material of the glove",
    "the color of the dustpan",
    "the color of the man's helmet",
    "the telephone and the hand lamp",
    "the dog and the river",
    "the man with green shorts and the blue boat",
}

_FORBIDDEN_ATOMIC_TARGETS = {"left", "right", "yes", "no"}
_SPECIAL_TOKENS = ("|", "<|", "<|endoftext|>", "<|im_end|>", FOVEATE_START, FOVEATE_END)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "below",
    "camera",
    "closer",
    "color",
    "colors",
    "compared",
    "distance",
    "for",
    "have",
    "in",
    "is",
    "kind",
    "left",
    "material",
    "near",
    "number",
    "of",
    "on",
    "or",
    "right",
    "side",
    "the",
    "to",
    "type",
    "what",
    "which",
    "with",
}


@dataclass(frozen=True)
class TargetSpec:
    target_text: str
    focus_type: str
    question_category: str
    question_without_choices: str
    attribute_type: str | None = None
    relation_type: str | None = None
    entities: list[str] = field(default_factory=list)
    parse_rule: str = ""


@dataclass(frozen=True)
class TargetValidation:
    valid: bool
    invalid_reason: str | None
    target_word_count: int
    choice_value_hit: bool
    single_word_target: bool
    copied_example_detected: bool

    def to_debug(self) -> dict[str, object]:
        return {
            "invalid_reason": self.invalid_reason,
            "target_word_count": self.target_word_count,
            "choice_value_hit": self.choice_value_hit,
            "single_word_target": self.single_word_target,
            "copied_example_detected": self.copied_example_detected,
        }


def build_vstar_target_spec(
    question: str,
    *,
    category: str | None = None,
    choices: list[str] | None = None,
) -> TargetSpec | None:
    question_without_choices = strip_answer_choices(question)
    normalized_question = _normalize_question_punctuation(question_without_choices)
    inferred_category = category or _infer_vstar_category(normalized_question)
    if inferred_category == "direct_attributes":
        return _build_direct_attribute_spec(
            normalized_question,
            question_without_choices=question_without_choices,
            choices=choices or [],
        )
    if inferred_category == "relative_position":
        return _build_relative_position_spec(
            normalized_question,
            question_without_choices=question_without_choices,
            choices=choices or [],
        )
    return None


def validate_target_text(
    target_text: str,
    *,
    spec: TargetSpec | None,
    choices: list[str] | None = None,
) -> TargetValidation:
    target = (target_text or "").strip()
    normalized = _normalize_text(target)
    word_count = len(_words(target))
    single_word = bool(target and word_count == 1)
    copied_example = normalized in {_normalize_text(example) for example in TARGET_EXAMPLES}
    choice_hit = _contains_choice_value(target, choices or [])

    invalid_reason = None
    if not target:
        invalid_reason = "empty_target"
    elif any(token in target for token in _SPECIAL_TOKENS):
        invalid_reason = "contains_pipe_or_special_token"
    elif normalized in _FORBIDDEN_ATOMIC_TARGETS:
        invalid_reason = "forbidden_atomic_target"
    elif spec and spec.focus_type == "attribute" and single_word:
        invalid_reason = "single_word_attribute_target"
    elif choice_hit:
        invalid_reason = "contains_answer_option_value"
    elif spec and _has_low_entity_overlap(target, spec):
        invalid_reason = "low_overlap_with_question_entities"

    return TargetValidation(
        valid=invalid_reason is None,
        invalid_reason=invalid_reason,
        target_word_count=word_count,
        choice_value_hit=choice_hit,
        single_word_target=single_word,
        copied_example_detected=copied_example,
    )


def target_spec_debug(spec: TargetSpec | None) -> dict[str, object]:
    if spec is None:
        return {
            "rule_target_text": None,
            "focus_type": None,
            "attribute_type": None,
            "relation_type": None,
            "entities": [],
            "target_parse_rule": None,
            "question_without_choices": None,
        }
    return {
        "rule_target_text": spec.target_text,
        "focus_type": spec.focus_type,
        "attribute_type": spec.attribute_type,
        "relation_type": spec.relation_type,
        "entities": spec.entities,
        "target_parse_rule": spec.parse_rule,
        "question_without_choices": spec.question_without_choices,
    }


def _build_direct_attribute_spec(
    question: str,
    *,
    question_without_choices: str,
    choices: list[str],
) -> TargetSpec | None:
    match = re.match(r"^What is the (?P<attribute>.+?) of (?P<object>.+?)\?$", question, re.IGNORECASE)
    if match:
        attribute = _clean_phrase(match.group("attribute"))
        obj = _strip_choice_values_from_phrase(_clean_phrase(match.group("object")), choices)
        return _attribute_spec(
            target_text=f"the {attribute} of {obj}",
            question_without_choices=question_without_choices,
            attribute_type=attribute,
            entities=[obj],
            parse_rule="what_is_attribute_of_object",
        )

    match = re.match(
        r"^What kind of (?P<kind>.+?) is (?P<prep>in|on|inside|within|at) (?P<object>.+?)\?$",
        question,
        re.IGNORECASE,
    )
    if match:
        kind = _clean_phrase(match.group("kind"))
        prep = match.group("prep").lower()
        obj = _strip_choice_values_from_phrase(_clean_phrase(match.group("object")), choices)
        return _attribute_spec(
            target_text=f"the kind of {kind} {prep} {obj}",
            question_without_choices=question_without_choices,
            attribute_type=f"kind of {kind}",
            entities=[obj],
            parse_rule="what_kind_of_entity_in_region",
        )

    match = re.match(r"^What is the (?P<attribute>.+?) (?P<prep>in|on|at) (?P<object>.+?)\??$", question, re.IGNORECASE)
    if match:
        attribute = _clean_phrase(match.group("attribute"))
        prep = match.group("prep").lower()
        obj = _strip_choice_values_from_phrase(_clean_phrase(match.group("object")), choices)
        return _attribute_spec(
            target_text=f"the {attribute} {prep} {obj}",
            question_without_choices=question_without_choices,
            attribute_type=attribute,
            entities=[obj],
            parse_rule="what_is_attribute_on_region",
        )

    match = re.match(r"^Is the (?P<attribute>color|material|pose|breed) of (?P<body>.+?)\?$", question, re.IGNORECASE)
    if match:
        attribute = match.group("attribute").lower()
        obj = _strip_choice_values_from_phrase(_object_before_choice_suffix(match.group("body"), choices), choices)
        return _attribute_spec(
            target_text=f"the {attribute} of {obj}",
            question_without_choices=question_without_choices,
            attribute_type=attribute,
            entities=[obj],
            parse_rule="is_attribute_of_object_candidates",
        )

    match = re.match(r"^Is (?P<body>.+?)\?$", question, re.IGNORECASE)
    if match:
        obj = _strip_choice_values_from_phrase(_object_before_choice_suffix(match.group("body"), choices), choices)
        attribute = _infer_attribute_from_choices(choices) or "visual attribute"
        return _attribute_spec(
            target_text=f"the {attribute} of {obj}",
            question_without_choices=question_without_choices,
            attribute_type=attribute,
            entities=[obj],
            parse_rule="is_object_candidate_attribute",
        )

    match = re.match(r"^Does (?P<object>.+?) have (?P<first>.+?) or (?P<second>.+?)\?$", question, re.IGNORECASE)
    if match:
        obj = _strip_choice_values_from_phrase(_clean_phrase(match.group("object")), choices)
        first = _clean_phrase(match.group("first"))
        second = _clean_phrase(match.group("second"))
        if "hair" in f"{first} {second}".lower():
            attribute = "hair type"
            target = f"the hair type of {obj}"
        elif "color" in f"{first} {second}".lower() or "colors" in f"{first} {second}".lower():
            attribute = "number of colors"
            target = f"the number of colors on {obj}"
        else:
            attribute = "visual attribute"
            target = f"the visual attribute of {obj}"
        return _attribute_spec(
            target_text=target,
            question_without_choices=question_without_choices,
            attribute_type=attribute,
            entities=[obj],
            parse_rule="does_object_have_candidate_attribute",
        )

    match = re.match(r"^How many (?P<entity>.+?) are in (?P<region>.+?)\?$", question, re.IGNORECASE)
    if match:
        entity = _clean_phrase(match.group("entity"))
        region = _strip_choice_values_from_phrase(_clean_phrase(match.group("region")), choices)
        return _attribute_spec(
            target_text=f"the number of {entity} in {region}",
            question_without_choices=question_without_choices,
            attribute_type="count",
            entities=[region, entity],
            parse_rule="how_many_entities_in_region",
        )

    return None


def _build_relative_position_spec(
    question: str,
    *,
    question_without_choices: str,
    choices: list[str],
) -> TargetSpec | None:
    del choices
    match = re.match(r"^Is (?P<a>.+?) on the left or right side of (?P<b>.+?)\?$", question, re.IGNORECASE)
    if match:
        a = _clean_phrase(match.group("a"))
        b = _clean_phrase(match.group("b"))
        return _relation_spec(
            target_text=f"{a} and {b}",
            question_without_choices=question_without_choices,
            relation_type="left_or_right",
            entities=[a, b],
            parse_rule="left_or_right_relation",
        )

    match = re.match(r"^Is (?P<a>.+?) above or below (?P<b>.+?)\?$", question, re.IGNORECASE)
    if match:
        a = _clean_phrase(match.group("a"))
        b = _clean_phrase(match.group("b"))
        return _relation_spec(
            target_text=f"{a} and {b}",
            question_without_choices=question_without_choices,
            relation_type="above_or_below",
            entities=[a, b],
            parse_rule="above_or_below_relation",
        )

    match = re.match(
        r"^Which one is closer to the camera, (?P<a>.+?) or (?P<b>.+?)\?$",
        question,
        re.IGNORECASE,
    )
    if match:
        a = _clean_phrase(match.group("a"))
        b = _clean_phrase(match.group("b"))
        return _relation_spec(
            target_text="the relative camera distance between the two compared objects",
            question_without_choices=question_without_choices,
            relation_type="closer_to_camera",
            entities=[a, b],
            parse_rule="closer_to_camera_relation",
        )

    return None


def _attribute_spec(
    *,
    target_text: str,
    question_without_choices: str,
    attribute_type: str,
    entities: list[str],
    parse_rule: str,
) -> TargetSpec:
    return TargetSpec(
        target_text=_clean_phrase(target_text),
        focus_type="attribute",
        question_category="direct_attributes",
        question_without_choices=question_without_choices,
        attribute_type=_clean_phrase(attribute_type),
        relation_type=None,
        entities=[_clean_phrase(entity) for entity in entities if _clean_phrase(entity)],
        parse_rule=parse_rule,
    )


def _relation_spec(
    *,
    target_text: str,
    question_without_choices: str,
    relation_type: str,
    entities: list[str],
    parse_rule: str,
) -> TargetSpec:
    return TargetSpec(
        target_text=_clean_phrase(target_text),
        focus_type="relative_position",
        question_category="relative_position",
        question_without_choices=question_without_choices,
        attribute_type=None,
        relation_type=relation_type,
        entities=[_clean_phrase(entity) for entity in entities if _clean_phrase(entity)],
        parse_rule=parse_rule,
    )


def _infer_vstar_category(question: str) -> str | None:
    lowered = question.lower()
    if "left or right" in lowered or "above or below" in lowered or "closer to the camera" in lowered:
        return "relative_position"
    return "direct_attributes"


def _infer_attribute_from_choices(choices: list[str]) -> str | None:
    normalized_choices = {_normalize_text(choice) for choice in choices}
    color_terms = {
        "black",
        "blue",
        "brown",
        "green",
        "grey",
        "orange",
        "pink",
        "purple",
        "red",
        "silver",
        "white",
        "yellow",
    }
    if normalized_choices and all(
        all(part in color_terms or part == "and" for part in choice.split()) for choice in normalized_choices
    ):
        return "color"
    return None


def _contains_choice_value(target: str, choices: list[str]) -> bool:
    normalized_target = _normalize_text(target)
    if not normalized_target:
        return False
    for choice in choices:
        normalized_choice = _normalize_text(choice)
        if not normalized_choice:
            continue
        if re.search(rf"\b{re.escape(normalized_choice)}\b", normalized_target):
            return True
    return False


def _has_low_entity_overlap(target: str, spec: TargetSpec) -> bool:
    if spec.relation_type == "closer_to_camera":
        return False
    target_terms = set(_content_terms(target))
    entity_term_groups = [set(_content_terms(entity)) for entity in spec.entities]
    entity_term_groups = [terms for terms in entity_term_groups if terms]
    if not entity_term_groups:
        return bool(spec.entities)
    if spec.focus_type == "relative_position":
        return not all(target_terms & terms for terms in entity_term_groups)
    return not any(target_terms & terms for terms in entity_term_groups)


def _normalize_question_punctuation(question: str) -> str:
    question = " ".join(question.strip().split())
    if question.endswith("."):
        question = f"{question[:-1]}?"
    return question


def _clean_phrase(text: str) -> str:
    text = " ".join(str(text).strip().split())
    text = text.strip(" .")
    return text


def _object_before_choice_suffix(text: str, choices: list[str]) -> str:
    body = _clean_phrase(text).rstrip("?")
    lowered = body.lower()
    positions = []
    for choice in choices:
        choice_text = _clean_phrase(choice).lower()
        if not choice_text:
            continue
        match = re.search(rf"\b{re.escape(choice_text)}\b", lowered)
        if match:
            positions.append(match.start())
    if positions:
        body = body[: min(positions)]
    body = re.sub(r"\b(or|and)\s*$", "", body, flags=re.IGNORECASE)
    return _clean_phrase(body)


def _strip_choice_values_from_phrase(text: str, choices: list[str]) -> str:
    cleaned = text
    for choice in choices:
        normalized_choice = _normalize_text(choice)
        if not normalized_choice:
            continue
        choice_words = normalized_choice.split()
        if len(choice_words) > 1:
            continue
        word = re.escape(choice_words[0])
        cleaned = re.sub(rf"\b{word}-[A-Za-z0-9']+\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf"\b{word}\b", "", cleaned, flags=re.IGNORECASE)
    return _clean_phrase(cleaned)


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text)


def _content_terms(text: str) -> list[str]:
    terms = []
    for word in _words(text.lower()):
        if word in _STOPWORDS:
            continue
        if len(word) <= 1:
            continue
        terms.append(word)
    return terms


def _normalize_text(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split())
