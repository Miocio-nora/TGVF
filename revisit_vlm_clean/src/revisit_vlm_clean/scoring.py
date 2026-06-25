"""Clean parser/scorer helpers ported from the stable V3 external path."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass

from .defaults import DEFAULT_CHOICE_PARSER_IDENTITY, DEFAULT_PARSER_IDENTITY
from .schema import ScoringBackend


@dataclass(frozen=True)
class ParseScoreResult:
    parsed_answer: str
    score: float | None
    answer_parse_success: bool
    model_output_parser: str = DEFAULT_PARSER_IDENTITY
    choice_parser: str = DEFAULT_CHOICE_PARSER_IDENTITY
    scorer_name: str = "project_fallback_exact_match"
    official_tool_used: bool = False
    official_compatible: bool = False


def parse_and_score(
    text: str,
    *,
    choices: list[str] | None = None,
    gold_answer: str | None = None,
    scoring_backend: ScoringBackend | str = ScoringBackend.AUTO,
) -> ParseScoreResult:
    backend = ScoringBackend(str(scoring_backend))
    if backend == ScoringBackend.OFFICIAL:
        raise NotImplementedError("official scorer execution is not ported in the clean skeleton")

    choices = choices or []
    if choices:
        parsed = extract_choice_strict(text, choices)
        scorer_name = "project_choice_exact_match"
    else:
        parsed = extract_answer_text(text) or clean_answer(text)
        scorer_name = "project_open_exact_match"

    score = None
    if gold_answer not in (None, ""):
        if choices:
            score = score_choice(parsed, str(gold_answer), choices)
        else:
            score = 1.0 if normalize_open_answer(parsed) == normalize_open_answer(str(gold_answer)) else 0.0

    return ParseScoreResult(
        parsed_answer=parsed,
        score=score,
        answer_parse_success=bool(parsed),
        scorer_name=scorer_name,
    )


def extract_answer_text(text: str) -> str:
    cleaned = str(text or "")
    matches = list(re.finditer(r"<ANSWER>\s*(.*?)(?:</ANSWER>|$)", cleaned, flags=re.IGNORECASE | re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return ""


def extract_choice_strict(text: str, choices: list[str]) -> str:
    cleaned = str(text or "")
    cleaned = re.sub(r"<\|[^>]+\|>", " ", cleaned)
    valid_letters = "".join(chr(ord("A") + index) for index in range(len(choices)))
    valid = set(valid_letters)
    for pattern in (
        r"<ANSWER>\s*\(?\s*([A-Z])\s*\)?",
        r"(?i)(?:final\s+answer|answer|option|choice)\s*(?:is|:|=)?\s*\(?\s*([A-Z])\s*\)?",
        r"(?m)^\s*\(?([A-Z])\)?\s*$",
    ):
        match = re.search(pattern, cleaned.strip())
        if match:
            letter = match.group(1).upper()
            if letter in valid:
                return letter

    normalized_output = normalize_choice_text(cleaned)
    option_hits: list[tuple[int, str]] = []
    for index, choice in enumerate(choices):
        normalized_choice = normalize_choice_text(choice)
        if not normalized_choice:
            continue
        found = normalized_output.find(normalized_choice)
        if found >= 0:
            option_hits.append((found, chr(ord("A") + index)))
    if option_hits:
        option_hits.sort()
        return option_hits[0][1]
    return ""


def score_choice(prediction: str, gold: str, choices: list[str]) -> float:
    pred_letter = choice_letter(prediction)
    gold_letter = choice_letter(gold)
    if pred_letter and gold_letter:
        return 1.0 if pred_letter == gold_letter else 0.0
    pred = str(prediction or "").strip()
    gold_text = str(gold or "").strip()
    if pred_letter:
        pred_index = ord(pred_letter) - ord("A")
        if 0 <= pred_index < len(choices):
            pred = choices[pred_index]
    if gold_letter:
        gold_index = ord(gold_letter) - ord("A")
        if 0 <= gold_index < len(choices):
            gold_text = choices[gold_index]
    return 1.0 if normalize_open_answer(pred) == normalize_open_answer(gold_text) else 0.0


def choice_letter(text: str) -> str:
    compact = str(text or "").strip().upper().strip("()[]{}.: ")
    return compact if len(compact) == 1 and "A" <= compact <= "Z" else ""


def normalize_choice_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_open_answer(text: str) -> str:
    text = str(text or "").lower().strip()
    text = text.translate(str.maketrans({char: " " for char in string.punctuation}))
    return " ".join(text.split())


def clean_answer(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()
