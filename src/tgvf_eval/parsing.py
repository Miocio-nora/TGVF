from __future__ import annotations

import re
import string

from revisit_vlm.tgvf import FOVEATE_END, FOVEATE_START, parse_foveation_spans


_ANSWER_MARKER_RE = re.compile(
    r"(?:answer|option|final answer|the answer is)\s*(?:is|:)?\s*[\(\[]?([A-Z])[\)\].,:\s]",
    re.IGNORECASE,
)
_FINAL_ANSWER_RE = re.compile(
    r"final answer\s*(?:is|:)?\s*[\(\[]?([A-Z])[\)\].,:\s]?",
    re.IGNORECASE,
)


def parse_multiple_choice(raw_output: str, valid_options: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ") -> str | None:
    if not raw_output:
        return None
    valid = set(valid_options.upper())
    normalized = raw_output.strip()
    final_matches = list(_FINAL_ANSWER_RE.finditer(normalized + " "))
    for match in reversed(final_matches):
        letter = match.group(1).upper()
        if letter in valid:
            return letter
    for match in _ANSWER_MARKER_RE.finditer(normalized + " "):
        letter = match.group(1).upper()
        if letter in valid:
            return letter
    for token in re.findall(r"\b([A-Z])\b", normalized.upper()):
        if token in valid:
            return token
    compact = normalized.strip().upper()
    if len(compact) == 1 and compact in valid:
        return compact
    return None


def normalize_open_answer(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans({char: " " for char in string.punctuation}))
    return " ".join(text.split())


def parse_prediction(raw_output: str, *, choices: list[str] | None = None) -> str:
    if choices:
        valid = "".join(chr(ord("A") + index) for index in range(len(choices)))
        parsed = parse_multiple_choice(raw_output, valid_options=valid)
        if parsed:
            return parsed
        normalized_output = normalize_open_answer(raw_output)
        for index, choice in enumerate(choices):
            normalized_choice = normalize_open_answer(choice)
            if normalized_output == normalized_choice or normalized_choice in normalized_output:
                return chr(ord("A") + index)
        return ""
    return raw_output.strip()


def extract_foveation_target(raw_output: str) -> str:
    spans = parse_foveation_spans(raw_output)
    return spans[0].target_text if spans else ""


def strip_foveation_span(raw_output: str) -> str:
    if FOVEATE_START not in raw_output:
        return raw_output
    end = raw_output.find(FOVEATE_END)
    if end == -1:
        return raw_output
    return raw_output[end + len(FOVEATE_END) :].strip()
