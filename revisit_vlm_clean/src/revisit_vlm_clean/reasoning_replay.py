"""Data contracts for replaying original-Qwen reasoning on direct Stage2 rows."""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable


REPLAY_SCHEMA_VERSION = "tgvf_direct_original_reasoning_replay_v1"
REJECTED_TOOL_MARKERS = (
    "<|focus_start|>",
    "<|focus_end|>",
    "<|tgvf_start|>",
    "<|tgvf_end|>",
    "<tool_call>",
    "</tool_call>",
    "<|im_start|>tool",
)
NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}
COLOR_WORDS = {
    "black",
    "blue",
    "brown",
    "gray",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
}


@dataclass(frozen=True)
class SourceRecord:
    split: str
    source_index: int
    row: dict[str, Any]
    raw_line: str
    row_sha256: str

    @property
    def uid(self) -> str:
        return str(self.row.get("v4_uid") or f"{self.split}:{self.source_index}")

    @property
    def is_direct(self) -> bool:
        return self.row.get("need_focus") is False


@dataclass(frozen=True)
class ParsedReasoning:
    reasoning: str
    final_text: str
    answer_candidate: str
    answer_candidates: tuple[str, ...]
    valid: bool
    error: str | None = None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_records(path: str | Path, *, split: str) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for source_index, raw_line in enumerate(handle):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if row.get("need_focus") not in {True, False}:
                raise ValueError(
                    f"row {source_index} in {path} has invalid need_focus={row.get('need_focus')!r}"
                )
            records.append(
                SourceRecord(
                    split=split,
                    source_index=source_index,
                    row=row,
                    raw_line=raw_line,
                    row_sha256=hashlib.sha256(raw_line.encode("utf-8")).hexdigest(),
                )
            )
    return records


def select_direct_records(
    records: Iterable[SourceRecord],
    *,
    max_records: int | None,
    seed: int,
) -> list[SourceRecord]:
    direct = [record for record in records if record.is_direct]
    if max_records is None or max_records >= len(direct):
        return direct
    if max_records < 1:
        raise ValueError("max_records must be >= 1 when provided")

    groups: dict[str, dict[str, list[SourceRecord]]] = {}
    for record in direct:
        question_type = str(record.row.get("question_type") or "unknown")
        source = str(record.row.get("source_dataset") or "unknown")
        groups.setdefault(question_type, {}).setdefault(source, []).append(record)

    flattened: dict[str, list[SourceRecord]] = {}
    for question_type, source_groups in groups.items():
        for group in source_groups.values():
            group.sort(key=lambda record: _selection_hash(record.uid, seed))
        flattened[question_type] = _round_robin_sources(
            source_groups,
            seed=seed,
            question_type=question_type,
        )

    selected: list[SourceRecord] = []
    keys = sorted(flattened, key=_question_type_selection_key)
    while len(selected) < max_records:
        added = False
        for key in keys:
            group = flattened[key]
            if group:
                selected.append(group.pop(0))
                added = True
                if len(selected) == max_records:
                    break
        if not added:
            break
    return selected


def parse_original_reasoning(raw_output: str) -> ParsedReasoning:
    raw = str(raw_output or "").strip()
    if not raw:
        return ParsedReasoning("", "", "", (), False, "empty_output")
    lowered = raw.lower()
    if any(marker in lowered for marker in REJECTED_TOOL_MARKERS):
        return ParsedReasoning("", "", "", (), False, "tool_protocol_leakage")
    if raw.count("</think>") != 1:
        return ParsedReasoning("", "", "", (), False, "missing_or_repeated_think_end")

    reasoning_part, final_part = raw.split("</think>", 1)
    if "<think>" in reasoning_part:
        if reasoning_part.count("<think>") != 1:
            return ParsedReasoning("", "", "", (), False, "repeated_think_start")
        prefix, reasoning_part = reasoning_part.split("<think>", 1)
        if prefix.strip():
            return ParsedReasoning("", "", "", (), False, "text_before_think_start")
    if "<think>" in final_part:
        return ParsedReasoning("", "", "", (), False, "think_start_after_think_end")

    reasoning = reasoning_part.strip()
    final_text = _strip_special_tokens(final_part).strip()
    if not reasoning:
        return ParsedReasoning("", final_text, "", (), False, "empty_reasoning")
    if not final_text:
        return ParsedReasoning(reasoning, "", "", (), False, "empty_final_answer")
    answer_candidates = tuple(extract_answer_candidates(final_text))
    if not answer_candidates:
        return ParsedReasoning(
            reasoning, final_text, "", (), False, "empty_answer_candidate"
        )
    return ParsedReasoning(
        reasoning,
        final_text,
        answer_candidates[0],
        answer_candidates,
        True,
    )


def extract_answer_candidate(final_text: str) -> str:
    candidates = extract_answer_candidates(final_text)
    return candidates[0] if candidates else ""


def extract_answer_candidates(final_text: str) -> list[str]:
    text = _strip_special_tokens(final_text).strip()
    candidates: list[str] = []
    answer_tag_matches = re.findall(
        r"<ANSWER>\s*(.*?)(?:</ANSWER>|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if answer_tag_matches:
        _append_candidate(candidates, answer_tag_matches[-1])

    boxed_matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed_matches:
        _append_candidate(candidates, boxed_matches[-1])

    latex_bold_matches = re.findall(
        r"\\(?:boldsymbol|mathbf|textbf)\{([^{}]+)\}", text
    )
    for match in reversed(latex_bold_matches):
        _append_candidate(candidates, match)

    plain = re.sub(r"[*_`]", "", text)
    answer_matches = list(
        re.finditer(
            r"(?is)(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)\s*"
            r"([^\n]+)",
            plain,
        )
    )
    for match in reversed(answer_matches):
        _append_candidate(candidates, match.group(1))

    leading_yes_no = re.match(r"(?is)^\s*(yes|no)\b", plain)
    if leading_yes_no:
        _append_candidate(candidates, leading_yes_no.group(1))

    numeric_conclusions = list(
        re.finditer(
            r"(?is)\b(?:is|are|equals?|totals?)\s+"
            r"(?:\\\(\s*)?(?:\\(?:boldsymbol|mathbf)\{)?\$?"
            r"([+-]?\d[\d,]*(?:\.\d+)?)",
            plain,
        )
    )
    for match in reversed(numeric_conclusions):
        _append_candidate(candidates, match.group(1))

    bold_matches = re.findall(r"\*\*([^*\n]+)\*\*", text)
    for match in reversed(bold_matches):
        if normalize_answer(match) not in {"answer", "final answer"}:
            _append_candidate(candidates, match)

    image_matches = list(
        re.finditer(r"(?is)\bthis\s+image\s+is\s+(.+?)(?:\.|\n|$)", plain)
    )
    for match in reversed(image_matches):
        _append_candidate(candidates, match.group(1))

    _append_candidate(candidates, text)
    return candidates


def answer_aliases(row: dict[str, Any]) -> list[str]:
    aliases: list[str] = []
    for key in ("answer", "short_answer", "value_span_text"):
        value = str(row.get(key) or "").strip()
        if value and value not in aliases:
            aliases.append(value)
    return aliases


def answers_match(candidate: str, aliases: Iterable[str]) -> tuple[bool, str | None]:
    normalized_candidate = normalize_answer(candidate)
    if not normalized_candidate:
        return False, None
    for alias in aliases:
        normalized_alias = normalize_answer(alias)
        if normalized_candidate == normalized_alias:
            return True, str(alias)
        if _strip_leading_article(normalized_candidate) == _strip_leading_article(
            normalized_alias
        ):
            return True, str(alias)
        if _equal_decimal(normalized_candidate, normalized_alias):
            return True, str(alias)
        if _equal_number_with_optional_percentage_unit(
            normalized_candidate, normalized_alias
        ):
            return True, str(alias)
    return False, None


def normalize_answer(text: str) -> str:
    value = str(text or "").lower().strip()
    value = re.sub(r"<[^>]+>", " ", value)
    value = value.replace(",", "")
    value = re.sub(r"[^\w.%+\-/]+", " ", value)
    return " ".join(value.split()).strip(" .")


def validate_generation(
    source_row: dict[str, Any],
    *,
    raw_output: str,
    output_tokens: int,
    generation_max_tokens: int,
    max_replay_target_tokens: int | None = None,
    max_sequence_tokens: int | None = None,
    prompt_tokens: int | None = None,
    replay_target_tokens: int | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    if error:
        return _validation_payload(False, "generation_error", error=error)
    if int(output_tokens) >= int(generation_max_tokens):
        return _validation_payload(False, "generation_budget_hit")

    parsed = parse_original_reasoning(raw_output)
    if not parsed.valid:
        return _validation_payload(
            False,
            str(parsed.error),
            reasoning=parsed.reasoning,
            final_text=parsed.final_text,
            answer_candidate=parsed.answer_candidate,
        )
    matched_candidate = ""
    matched_alias = None
    match_type = None
    for candidate in parsed.answer_candidates:
        matched, alias = answers_match(candidate, answer_aliases(source_row))
        if matched:
            matched_candidate = candidate
            matched_alias = alias
            match_type = "normalized_exact"
            break
    if not matched_candidate:
        for alias in answer_aliases(source_row):
            if _explicit_answer_matches(
                parsed.answer_candidate,
                alias,
                question=str(source_row.get("question") or ""),
            ):
                matched_candidate = parsed.answer_candidate
                matched_alias = alias
                match_type = "explicit_answer_equivalent"
                break
    if not matched_candidate:
        return _validation_payload(
            False,
            "answer_mismatch",
            reasoning=parsed.reasoning,
            final_text=parsed.final_text,
            answer_candidate=parsed.answer_candidate,
        )
    accepted_length = (
        int(replay_target_tokens)
        if replay_target_tokens is not None
        else int(output_tokens)
    )
    if (
        max_replay_target_tokens is not None
        and accepted_length > int(max_replay_target_tokens)
    ):
        return _validation_payload(
            False,
            "replay_target_length_limit_exceeded",
            reasoning=parsed.reasoning,
            final_text=parsed.final_text,
            answer_candidate=matched_candidate,
            matched_alias=matched_alias,
            match_type=match_type,
        )
    if (
        max_sequence_tokens is not None
        and prompt_tokens is not None
        and replay_target_tokens is not None
        and int(prompt_tokens) + int(replay_target_tokens) > int(max_sequence_tokens)
    ):
        return _validation_payload(
            False,
            "replay_sequence_length_limit_exceeded",
            reasoning=parsed.reasoning,
            final_text=parsed.final_text,
            answer_candidate=matched_candidate,
            matched_alias=matched_alias,
            match_type=match_type,
        )
    return _validation_payload(
        True,
        "accepted",
        reasoning=parsed.reasoning,
        final_text=parsed.final_text,
        answer_candidate=matched_candidate,
        matched_alias=matched_alias,
        match_type=match_type,
    )


def build_replay_split(
    *,
    source_path: str | Path,
    output_path: str | Path,
    split: str,
    generation_rows: Iterable[dict[str, Any]],
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    accepted: dict[str, dict[str, Any]] = {}
    for generation_row in generation_rows:
        if generation_row.get("split") != split:
            continue
        validation = dict(generation_row.get("validation") or {})
        if not validation.get("accepted"):
            continue
        uid = str(generation_row.get("v4_uid") or "")
        if not uid:
            raise ValueError("accepted generation row is missing v4_uid")
        if uid in accepted:
            raise ValueError(f"duplicate accepted generation row: {uid}")
        accepted[uid] = generation_row

    source_focus_hash = hashlib.sha256()
    output_focus_hash = hashlib.sha256()
    source_rows = 0
    source_direct = 0
    source_focus = 0
    written_rows = 0
    written_direct = 0
    written_focus = 0
    used_accepted: set[str] = set()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(source_path).open(encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as output:
        for source_index, raw_line in enumerate(source):
            if not raw_line.strip():
                continue
            source_rows += 1
            row = json.loads(raw_line)
            if row.get("need_focus") is True:
                source_focus += 1
                payload = raw_line if raw_line.endswith("\n") else raw_line + "\n"
                source_focus_hash.update(payload.encode("utf-8"))
                output_focus_hash.update(payload.encode("utf-8"))
                output.write(payload)
                written_rows += 1
                written_focus += 1
                continue
            if row.get("need_focus") is not False:
                raise ValueError(f"invalid need_focus at {source_path}:{source_index + 1}")
            source_direct += 1
            uid = str(row.get("v4_uid") or f"{split}:{source_index}")
            generation_row = accepted.get(uid)
            if generation_row is None:
                continue
            validation = dict(generation_row["validation"])
            row["no_focus_think"] = str(validation["reasoning"])
            row["reasoning_replay"] = {
                "schema_version": REPLAY_SCHEMA_VERSION,
                **run_metadata,
                "source_split": split,
                "source_index": source_index,
                "source_row_sha256": generation_row.get("source_row_sha256"),
                "output_tokens": generation_row.get("output_tokens"),
                "reasoning_tokens": generation_row.get("reasoning_tokens"),
                "replay_target_tokens": generation_row.get("replay_target_tokens"),
                "generation_prompt_tokens": generation_row.get(
                    "generation_prompt_tokens"
                ),
                "replay_sequence_tokens": generation_row.get(
                    "replay_sequence_tokens"
                ),
                "answer_candidate": validation.get("answer_candidate"),
                "matched_alias": validation.get("matched_alias"),
                "answer_match_type": validation.get("match_type"),
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            written_rows += 1
            written_direct += 1
            used_accepted.add(uid)

    unused = sorted(set(accepted) - used_accepted)
    if unused:
        raise ValueError(f"accepted rows were not found in source split {split}: {unused[:3]}")
    source_hash = source_focus_hash.hexdigest()
    output_hash = output_focus_hash.hexdigest()
    if source_hash != output_hash or written_focus != source_focus:
        raise AssertionError("focus rows changed while publishing replay split")
    return {
        "split": split,
        "source_rows": source_rows,
        "source_direct_rows": source_direct,
        "source_focus_rows": source_focus,
        "written_rows": written_rows,
        "written_direct_rows": written_direct,
        "written_focus_rows": written_focus,
        "dropped_direct_rows": source_direct - written_direct,
        "focus_payload_sha256": source_hash,
        "focus_payload_identical": True,
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
    }


def token_statistics(values: Iterable[int]) -> dict[str, float | int | None]:
    items = sorted(int(value) for value in values)
    if not items:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(items),
        "mean": statistics.fmean(items),
        "median": statistics.median(items),
        "p90": _nearest_rank(items, 0.90),
        "p95": _nearest_rank(items, 0.95),
        "p99": _nearest_rank(items, 0.99),
        "max": items[-1],
    }


def _validation_payload(
    accepted: bool,
    reason: str,
    *,
    reasoning: str = "",
    final_text: str = "",
    answer_candidate: str = "",
    matched_alias: str | None = None,
    match_type: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "accepted": bool(accepted),
        "reason": reason,
        "reasoning": reasoning,
        "final_text": final_text,
        "answer_candidate": answer_candidate,
        "matched_alias": matched_alias,
        "match_type": match_type,
        "error": error,
    }


def _selection_hash(uid: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{uid}".encode("utf-8")).hexdigest()


def _round_robin_sources(
    source_groups: dict[str, list[SourceRecord]],
    *,
    seed: int,
    question_type: str,
) -> list[SourceRecord]:
    ordered: list[SourceRecord] = []
    source_keys = sorted(
        source_groups,
        key=lambda source: hashlib.sha256(
            f"{seed}:{question_type}:{source}".encode("utf-8")
        ).hexdigest(),
    )
    while True:
        added = False
        for source in source_keys:
            group = source_groups[source]
            if group:
                ordered.append(group.pop(0))
                added = True
        if not added:
            return ordered


def _question_type_selection_key(question_type: str) -> tuple[int, str]:
    priority = {"math_reasoning": 0, "whole_image_obvious": 1}
    return (priority.get(question_type, 2), question_type)


def _strip_special_tokens(text: str) -> str:
    return re.sub(r"<\|[^>]+\|>", " ", str(text or ""))


def _equal_decimal(left: str, right: str) -> bool:
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", left):
        return False
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", right):
        return False
    try:
        return Decimal(left) == Decimal(right)
    except InvalidOperation:
        return False


def _equal_number_with_optional_percentage_unit(left: str, right: str) -> bool:
    pattern = re.compile(
        r"^([+-]?\d+(?:\.\d+)?)\s*(%|percent|percentage\s+points?)?$"
    )
    left_match = pattern.fullmatch(left)
    right_match = pattern.fullmatch(right)
    if left_match is None or right_match is None:
        return False
    try:
        return Decimal(left_match.group(1)) == Decimal(right_match.group(1))
    except InvalidOperation:
        return False


def _strip_leading_article(text: str) -> str:
    return re.sub(r"^(?:a|an|the)\s+", "", text)


def _append_candidate(candidates: list[str], value: str) -> None:
    candidate = str(value or "").strip().strip("`*_ ").rstrip(".").strip()
    if candidate and candidate not in candidates:
        candidates.append(candidate)


def _explicit_answer_matches(candidate: str, alias: str, *, question: str) -> bool:
    candidate_normalized = normalize_answer(candidate)
    alias_normalized = normalize_answer(alias)
    if not candidate_normalized or not alias_normalized:
        return False

    candidate_numbers = _numbers_in_answer(candidate_normalized)
    alias_numbers = _numbers_in_answer(alias_normalized)
    if len(candidate_numbers) == len(alias_numbers) == 1:
        if candidate_numbers[0] == alias_numbers[0]:
            return True
        approximation_requested = any(
            marker in f"{question} {alias}".lower()
            for marker in ("about", "approximately", "approximate", "roughly")
        )
        if approximation_requested and _approximately_equal(
            candidate_numbers[0], alias_numbers[0]
        ):
            return True

    alias_tokens = _answer_tokens(alias_normalized)
    candidate_tokens = _answer_tokens(candidate_normalized)
    if not alias_tokens or not candidate_tokens:
        return False
    if alias_tokens in (["yes"], ["no"]):
        return candidate_tokens[0] == alias_tokens[0]
    if len(alias_tokens) == 1 and alias_tokens[0] in COLOR_WORDS:
        candidate_colors = COLOR_WORDS.intersection(candidate_tokens)
        if candidate_colors != {alias_tokens[0]}:
            return False
    if _contains_token_sequence(candidate_tokens, alias_tokens):
        alias_text = " ".join(alias_tokens)
        negated = re.search(
            rf"\b(?:not|no)\s+(?:\w+\s+){{0,2}}{re.escape(alias_text)}\b",
            candidate_normalized,
        )
        return negated is None
    return False


def _numbers_in_answer(text: str) -> list[Decimal]:
    expanded = str(text)
    for word, number in NUMBER_WORDS.items():
        expanded = re.sub(rf"\b{word}\b", number, expanded)
    values: list[Decimal] = []
    for match in re.findall(r"[+-]?\d+(?:\.\d+)?", expanded):
        try:
            values.append(Decimal(match))
        except InvalidOperation:
            continue
    return values


def _approximately_equal(left: Decimal, right: Decimal) -> bool:
    difference = abs(left - right)
    scale = max(abs(left), abs(right), Decimal("1"))
    return difference <= Decimal("0.05") * scale


def _answer_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _contains_token_sequence(tokens: list[str], expected: list[str]) -> bool:
    width = len(expected)
    return any(
        tokens[index : index + width] == expected
        for index in range(len(tokens) - width + 1)
    )


def _nearest_rank(values: list[int], quantile: float) -> int:
    index = max(0, min(len(values) - 1, int(quantile * len(values) + 0.999999) - 1))
    return values[index]
