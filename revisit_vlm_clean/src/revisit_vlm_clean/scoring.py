"""Clean parser/scorer helpers ported from the stable V3 external path."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass

from .defaults import DEFAULT_CHOICE_PARSER_IDENTITY, DEFAULT_PARSER_IDENTITY
from .schema import ScoringBackend


OFFICIAL_COMPATIBLE_CHOICE_SCORERS = {
    "blink": ("official_blink_exact_match", False),
    "hr_bench_4k": ("official_compatible_hrbench4k_mc", True),
}


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
    benchmark: str | None = None,
    scoring_backend: ScoringBackend | str = ScoringBackend.AUTO,
) -> ParseScoreResult:
    backend = ScoringBackend(str(scoring_backend))
    choices = choices or []
    official_choice = _official_choice_scorer(benchmark)
    if official_choice is not None and backend in {ScoringBackend.AUTO, ScoringBackend.OFFICIAL}:
        return _parse_and_score_official_choice(
            text,
            choices=choices,
            gold_answer=gold_answer,
            scorer_name=official_choice[0],
            official_compatible=official_choice[1],
        )
    if backend == ScoringBackend.OFFICIAL:
        raise NotImplementedError(
            f"official scorer execution is not ported for benchmark={benchmark!r} in the clean runner"
        )

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
            gold_text = clean_answer(str(gold_answer))
            score = 1.0 if normalize_open_answer(parsed) == normalize_open_answer(gold_text) else 0.0

    return ParseScoreResult(
        parsed_answer=parsed,
        score=score,
        answer_parse_success=bool(parsed),
        scorer_name=scorer_name,
    )


def score_output_rows(
    rows: list[dict],
    *,
    scoring_backend: ScoringBackend | str = ScoringBackend.AUTO,
) -> None:
    for row in rows:
        if row.get("error"):
            row.setdefault("parsed_answer", "")
            row.setdefault("score", None)
            row.setdefault("answer_parse_success", False)
            row.setdefault("scorer_name", "")
            row.setdefault("official_tool_used", False)
            row.setdefault("official_compatible", False)
            continue
        parsed = parse_and_score(
            str(row.get("raw_output") or ""),
            choices=list(row.get("choices") or []),
            gold_answer=row.get("gold_answer"),
            benchmark=row.get("benchmark"),
            scoring_backend=scoring_backend,
        )
        row["parsed_answer"] = parsed.parsed_answer
        row["score"] = parsed.score
        row["answer_parse_success"] = parsed.answer_parse_success
        row["scorer_name"] = parsed.scorer_name
        row["official_tool_used"] = parsed.official_tool_used
        row["official_compatible"] = parsed.official_compatible


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


def extract_choice_official_compatible(text: str, choices: list[str]) -> str:
    cleaned = str(text or "")
    cleaned = re.sub(r"<\|[^>]+\|>", " ", cleaned)
    valid = {chr(ord("A") + index) for index in range(max(len(choices), 1))}
    for pattern in (
        r"<ANSWER>\s*\(?\s*([A-Z])\s*\)?",
        r"(?i)(?:final\s+answer|answer|option|choice)\s*(?:is|:|=)?\s*\(?\s*([A-Z])\s*\)?",
        r"\(([A-Z])\)",
        r"(?m)^\s*([A-Z])\s*$",
    ):
        match = re.search(pattern, cleaned.strip())
        if match:
            letter = match.group(1).upper()
            if letter in valid:
                return letter
    compact = choice_letter(cleaned)
    return compact if compact in valid else ""


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


def _parse_and_score_official_choice(
    text: str,
    *,
    choices: list[str],
    gold_answer: str | None,
    scorer_name: str,
    official_compatible: bool,
) -> ParseScoreResult:
    parsed = extract_choice_official_compatible(text, choices)
    score = None
    if gold_answer not in (None, ""):
        score = score_choice(parsed, str(gold_answer), choices)
    return ParseScoreResult(
        parsed_answer=choice_letter(parsed) or parsed,
        score=score,
        answer_parse_success=bool(parsed),
        scorer_name=scorer_name,
        official_tool_used=True,
        official_compatible=official_compatible,
    )


def _official_choice_scorer(benchmark: str | None) -> tuple[str, bool] | None:
    if not benchmark:
        return None
    return OFFICIAL_COMPATIBLE_CHOICE_SCORERS.get(str(benchmark))


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
