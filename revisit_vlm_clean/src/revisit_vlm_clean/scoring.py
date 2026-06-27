"""Clean parser/scorer helpers ported from the stable V3 external path."""

from __future__ import annotations

import importlib.util
import json
import random
import re
import string
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

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
            "official scorer execution is not ported for "
            f"benchmark={benchmark!r} in the clean runner"
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
            score = (
                1.0 if normalize_open_answer(parsed) == normalize_open_answer(gold_text) else 0.0
            )

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
    benchmark_root: str | Path | None = None,
) -> None:
    backend = ScoringBackend(str(scoring_backend))
    pending_rows: list[dict] = []
    for row in rows:
        if row.get("error"):
            row.setdefault("parsed_answer", "")
            row.setdefault("score", None)
            row.setdefault("answer_parse_success", False)
            row.setdefault("scorer_name", "")
            row.setdefault("official_tool_used", False)
            row.setdefault("official_compatible", False)
            row.setdefault("official_tool_path", None)
            continue
        pending_rows.append(row)

    consumed_row_ids: set[int] = set()
    if backend != ScoringBackend.PROJECT:
        mathvista_rows = [row for row in pending_rows if row.get("benchmark") == "mathvista"]
        if mathvista_rows:
            official_path = _mathvista_eval_path(benchmark_root)
            if official_path is None:
                if backend == ScoringBackend.OFFICIAL:
                    raise NotImplementedError(
                        "official MathVista scoring requires benchmark_root/mathvista/official_code"
                    )
            else:
                _score_mathvista_rows(mathvista_rows, official_eval_path=official_path)
                consumed_row_ids.update(id(row) for row in mathvista_rows)

        mathverse_rows = [row for row in pending_rows if row.get("benchmark") == "mathverse"]
        if mathverse_rows:
            official_path = _mathverse_eval_path(benchmark_root)
            if official_path is None:
                if backend == ScoringBackend.OFFICIAL:
                    raise NotImplementedError(
                        "official MathVerse scoring requires benchmark_root/mathverse/official_code"
                    )
            else:
                _score_mathverse_rows(mathverse_rows, official_eval_path=official_path)
                consumed_row_ids.update(id(row) for row in mathverse_rows)

        mmmu_rows = [row for row in pending_rows if row.get("benchmark") == "mmmu_pro"]
        if mmmu_rows:
            official_path = _mmmu_pro_eval_path(benchmark_root)
            if official_path is None:
                if backend == ScoringBackend.OFFICIAL:
                    raise NotImplementedError(
                        "official MMMU-Pro scoring requires benchmark_root/mmmu_pro/official_code"
                    )
            else:
                _score_mmmu_pro_rows(mmmu_rows, official_eval_path=official_path)
                consumed_row_ids.update(id(row) for row in mmmu_rows)

        ocrbench_rows = [row for row in pending_rows if row.get("benchmark") == "ocrbench_v2"]
        if ocrbench_rows:
            official_path = _ocrbench_v2_eval_path(benchmark_root)
            if official_path is None:
                if backend == ScoringBackend.OFFICIAL:
                    raise NotImplementedError(
                        "official OCRBench-v2 scoring requires "
                        "benchmark_root/ocrbench_v2/official_code"
                    )
            else:
                _score_ocrbench_v2_rows(ocrbench_rows, official_eval_path=official_path)
                consumed_row_ids.update(id(row) for row in ocrbench_rows)

    for row in pending_rows:
        if id(row) in consumed_row_ids:
            continue
        parsed = parse_and_score(
            str(row.get("raw_output") or ""),
            choices=list(row.get("choices") or []),
            gold_answer=row.get("gold_answer"),
            benchmark=row.get("benchmark"),
            scoring_backend=backend,
        )
        row["parsed_answer"] = parsed.parsed_answer
        row["score"] = parsed.score
        row["answer_parse_success"] = parsed.answer_parse_success
        row["scorer_name"] = parsed.scorer_name
        row["official_tool_used"] = parsed.official_tool_used
        row["official_compatible"] = parsed.official_compatible
        row["official_tool_path"] = None


def extract_answer_text(text: str) -> str:
    cleaned = str(text or "")
    matches = list(
        re.finditer(r"<ANSWER>\s*(.*?)(?:</ANSWER>|$)", cleaned, flags=re.IGNORECASE | re.DOTALL)
    )
    if matches:
        return matches[-1].group(1).strip()
    return ""


def extract_final_answer(text: Any) -> str:
    cleaned = str(text or "").strip()
    answer = extract_answer_text(cleaned)
    if answer:
        return answer
    answer_matches = list(
        re.finditer(r"(?i)(?:final\s+answer|answer)\s*(?:is|:|=)\s*(.+)$", cleaned)
    )
    if answer_matches:
        return answer_matches[-1].group(1).strip().strip(". ")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", cleaned.replace(",", ""))
    if numbers:
        return numbers[-1]
    return clean_answer(cleaned)


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


def _ocrbench_v2_eval_path(benchmark_root: str | Path | None) -> Path | None:
    if benchmark_root is None:
        return None
    path = (
        Path(benchmark_root)
        / "ocrbench_v2"
        / "official_code"
        / "OCRBench_v2"
        / "eval_scripts"
        / "eval.py"
    )
    return path if path.exists() else None


def _mmmu_pro_eval_path(benchmark_root: str | Path | None) -> Path | None:
    if benchmark_root is None:
        return None
    path = Path(benchmark_root) / "mmmu_pro" / "official_code" / "mmmu-pro" / "evaluate.py"
    return path if path.exists() else None


def _mathvista_eval_path(benchmark_root: str | Path | None) -> Path | None:
    if benchmark_root is None:
        return None
    path = (
        Path(benchmark_root) / "mathvista" / "official_code" / "evaluation" / "calculate_score.py"
    )
    return path if path.exists() else None


def _mathverse_eval_path(benchmark_root: str | Path | None) -> Path | None:
    if benchmark_root is None:
        return None
    path = (
        Path(benchmark_root) / "mathverse" / "official_code" / "evaluation" / "score_answer_s2.py"
    )
    return path if path.exists() else None


def _score_mathvista_rows(rows: list[dict], *, official_eval_path: Path) -> None:
    for row in rows:
        row["scorer_name"] = "official_mathvista"
        row["official_tool_used"] = True
        row["official_tool_path"] = str(official_eval_path)
        row["official_compatible"] = False
        row["llm_judge_used"] = False
        metadata = dict(row.get("metadata") or {})
        choices = list(row.get("choices") or metadata.get("choices") or [])
        gold = row.get("gold_answer")
        if gold in (None, ""):
            row["parsed_answer"] = ""
            row["answer_parse_success"] = False
            row["prediction"] = None
            row["score"] = None
            continue
        extraction = row.get("parsed_answer") or extract_final_answer(row.get("raw_output") or "")
        prediction = _mathvista_normalize(
            extraction,
            choices,
            str(metadata.get("question_type") or ("multi_choice" if choices else "free_form")),
            str(metadata.get("answer_type") or "text"),
            metadata.get("precision", 0),
        )
        answer = _mathvista_gold(str(gold), choices)
        row["parsed_answer"] = str(extraction or "")
        row["answer_parse_success"] = bool(row["parsed_answer"])
        row["prediction"] = prediction
        row["score"] = (
            1.0 if prediction is not None and _safe_equal(str(prediction), str(answer)) else 0.0
        )


def _score_mathverse_rows(rows: list[dict], *, official_eval_path: Path) -> None:
    for row in rows:
        row["scorer_name"] = "official_mathverse"
        row["official_tool_used"] = True
        row["official_tool_path"] = str(official_eval_path)
        row["official_compatible"] = False
        row["llm_judge_used"] = False
        gold = row.get("gold_answer")
        if gold in (None, ""):
            row["parsed_answer"] = ""
            row["answer_parse_success"] = False
            row["score"] = None
            continue
        choices = list(row.get("choices") or (row.get("metadata") or {}).get("choices") or [])
        pred = (
            row.get("parsed_answer")
            or extract_choice_official_compatible(
                str(row.get("raw_output") or ""),
                choices,
            )
            or extract_final_answer(row.get("raw_output") or "")
        )
        row["parsed_answer"] = str(pred or "")
        row["answer_parse_success"] = bool(row["parsed_answer"])
        row["score"] = score_choice(row["parsed_answer"], str(gold), choices)


def _score_mmmu_pro_rows(rows: list[dict], *, official_eval_path: Path) -> None:
    module = _load_module_from_path(official_eval_path)
    random_state = random.getstate()
    try:
        for index, row in enumerate(rows):
            row["scorer_name"] = "official_mmmu_pro"
            row["official_tool_used"] = True
            row["official_tool_path"] = str(official_eval_path)
            row["official_compatible"] = False
            choices = list(row.get("choices") or (row.get("metadata") or {}).get("choices") or [])
            gold = row.get("gold_answer")
            if gold in (None, ""):
                row["parsed_answer"] = ""
                row["answer_parse_success"] = False
                row["score"] = None
                continue
            pred = row.get("parsed_answer")
            if (pred is None or pred == "") and choices and row.get("raw_output") is not None:
                index2ans, all_choices = module.get_multi_choice_info(choices)
                random.seed(_stable_seed(row, index))
                pred = module.parse_multi_choice_response(
                    str(row.get("raw_output") or ""),
                    all_choices,
                    index2ans,
                )
            row["parsed_answer"] = str(pred or "")
            row["answer_parse_success"] = bool(row["parsed_answer"])
            row["score"] = 1.0 if module.eval_multi_choice(str(gold), row["parsed_answer"]) else 0.0
    finally:
        random.setstate(random_state)


def _score_ocrbench_v2_rows(rows: list[dict], *, official_eval_path: Path) -> None:
    module = _load_module_from_path(
        official_eval_path,
        extra_sys_paths=[official_eval_path.parent],
        stub_modules=["ipdb"],
    )
    items: list[dict[str, Any]] = []
    scored_rows: list[dict] = []
    for row in rows:
        row["scorer_name"] = "official_ocrbench_v2"
        row["official_tool_used"] = True
        row["official_tool_path"] = str(official_eval_path)
        row["official_compatible"] = False
        metadata = dict(row.get("metadata") or {})
        answers = _as_list(metadata.get("answers"))
        if not answers and row.get("gold_answer") not in (None, ""):
            answers = [row["gold_answer"]]
        task_type = metadata.get("type") or metadata.get("task")
        prediction = row.get("parsed_answer") or extract_final_answer(row.get("raw_output") or "")
        row["parsed_answer"] = str(prediction or "")
        row["answer_parse_success"] = bool(row["parsed_answer"])
        if not task_type or not answers:
            row["score"] = None
            continue
        item = dict(metadata)
        item.update(
            type=task_type,
            question=row.get("question") or metadata.get("question") or "",
            answers=answers,
            predict=row["parsed_answer"],
        )
        if task_type == "text counting en" and "eval" not in item:
            item["eval"] = _ocrbench_text_counting_eval_method(answers)
        items.append(item)
        scored_rows.append(row)

    if not items:
        return

    with tempfile.TemporaryDirectory(prefix="tgvf_clean_ocrbench_official_") as tmp:
        input_path = Path(tmp) / "predictions.json"
        output_path = Path(tmp) / "scores.json"
        input_path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        module.process_predictions(str(input_path), str(output_path))
        scored_items = json.loads(output_path.read_text(encoding="utf-8"))

    for row, item in zip(scored_rows, scored_items, strict=True):
        row["score"] = float(item.get("score") or 0.0)


def _ocrbench_text_counting_eval_method(answers: list[Any]) -> str:
    for answer in answers:
        try:
            int(str(answer).strip())
        except Exception:
            return "exact match"
    return "regression"


def _as_list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _stable_seed(row: dict, index: int) -> int:
    key = str(row.get("sample_id") or row.get("id") or index)
    return sum((offset + 1) * ord(char) for offset, char in enumerate(key)) % (2**32)


def _mathvista_normalize(
    extraction: Any,
    choices: list[str],
    question_type: str,
    answer_type: str,
    precision: Any,
) -> str | None:
    extraction_text = clean_answer(extraction)
    if question_type in {"multi_choice", "multi-choice"} or choices:
        letter = choice_letter(extraction_text)
        if letter and choices:
            index = ord(letter) - ord("A")
            if 0 <= index < len(choices):
                return choices[index]
        if extraction_text in choices:
            return extraction_text
        if choices:
            return min(choices, key=lambda choice: _edit_distance(extraction_text, choice))
        return extraction_text or None
    if answer_type == "integer":
        numbers = re.findall(r"-?\d+(?:\.\d+)?", extraction_text.replace(",", ""))
        if not numbers:
            return None
        try:
            return str(int(float(numbers[-1])))
        except Exception:
            return None
    if answer_type == "float":
        numbers = re.findall(r"-?\d+(?:\.\d+)?", extraction_text.replace(",", ""))
        if not numbers:
            return None
        try:
            return str(round(float(numbers[-1]), int(float(precision or 0))))
        except Exception:
            return None
    if answer_type == "list":
        return extraction_text
    return extraction_text or None


def _mathvista_gold(gold: str, choices: list[str]) -> str:
    letter = choice_letter(gold)
    if letter and choices:
        index = ord(letter) - ord("A")
        if 0 <= index < len(choices):
            return choices[index]
    return gold


def _safe_equal(prediction: str, answer: str) -> bool:
    return prediction == answer


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


@contextmanager
def _temporary_sys_path(paths: list[Path] | None):
    if not paths:
        yield
        return
    additions = [str(path) for path in paths]
    old = list(sys.path)
    sys.path[:0] = additions
    try:
        yield
    finally:
        sys.path[:] = old


def _load_module_from_path(
    path: Path,
    *,
    extra_sys_paths: list[Path] | None = None,
    stub_modules: list[str] | None = None,
) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"revisit_vlm_clean_official_{path.stem}_{abs(hash(str(path)))}",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load official scorer module from {path}")
    module = importlib.util.module_from_spec(spec)
    previous_stubs: dict[str, ModuleType | None] = {}
    for name in stub_modules or []:
        previous_stubs[name] = sys.modules.get(name)
        if name not in sys.modules:
            sys.modules[name] = ModuleType(name)
    try:
        with _temporary_sys_path(extra_sys_paths):
            spec.loader.exec_module(module)
    finally:
        for name, previous in previous_stubs.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module
