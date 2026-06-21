#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT_ROOT = Path("data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k/splits")
DEFAULT_OUTPUT_ROOT = Path("data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_open_answer/splits")
DEFAULT_FILES = (
    "tgvf_v4_teacher_stage2_protocol_c.train.jsonl",
    "tgvf_v4_teacher_stage2_protocol_c.test.jsonl",
)


def main() -> None:
    args = parse_args()
    reports: dict[str, Any] = {}
    for rel in args.files:
        input_path = Path(args.input_root) / rel
        output_path = Path(args.output_root) / rel
        report = convert_file(input_path, output_path)
        reports[rel] = report
        print(json.dumps({"file": rel, **report}, ensure_ascii=False), flush=True)
    report_path = Path(args.report) if args.report else Path(args.output_root) / "choice_to_open_answer_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path)}, ensure_ascii=False), flush=True)


def convert_file(input_path: Path, output_path: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            counts["input"] += 1
            record = json.loads(line)
            before_format = str(record.get("answer_format") or "unknown")
            counts[f"input_answer_format_{before_format}"] += 1
            if is_choice_record(record):
                row = convert_choice_record(record, line_no=line_no)
                counts["converted_choice"] += 1
            else:
                row = dict(record)
                counts["kept_non_choice"] += 1
            after_format = str(row.get("answer_format") or "unknown")
            counts[f"output_answer_format_{after_format}"] += 1
            if is_choice_record(row):
                counts["output_choice_like"] += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["output"] += 1
    counts["input_choice_ratio"] = counts["converted_choice"] / counts["input"] if counts["input"] else 0.0
    counts["output_choice_ratio"] = counts["output_choice_like"] / counts["output"] if counts["output"] else 0.0
    return dict(counts)


def convert_choice_record(record: dict[str, Any], *, line_no: int) -> dict[str, Any]:
    short_answer = open_answer_text(record)
    if not short_answer:
        raise ValueError(f"{line_no}: choice record has no recoverable open answer")

    row = dict(record)
    original_answer = row.get("answer")
    original_choices = row.get("choices") or []
    original_answer_format = row.get("answer_format")
    original_value_span = row.get("value_span_text")

    row["question"] = strip_answer_choices(str(row.get("question") or ""))
    row["choices"] = []
    row["answer"] = short_answer
    row["short_answer"] = short_answer
    row["answer_format"] = "short_text"
    row["value_span_text"] = short_answer
    row["metadata"] = {
        **dict(row.get("metadata") or {}),
        "choice_to_open_answer": {
            "original_answer": original_answer,
            "original_answer_format": original_answer_format,
            "original_choices": original_choices,
            "original_value_span_text": original_value_span,
        },
    }
    for step in row.get("focus_steps") or []:
        if isinstance(step, dict):
            step["value_span_text"] = short_answer
    return row


def is_choice_record(record: dict[str, Any]) -> bool:
    return bool(record.get("choices")) or record.get("answer_format") == "multiple_choice"


def open_answer_text(record: dict[str, Any]) -> str:
    short = str(record.get("short_answer") or "").strip()
    if short:
        return strip_choice_letter_prefix(short)
    value_span = str(record.get("value_span_text") or "").strip()
    if value_span:
        return strip_choice_letter_prefix(value_span)
    answer = str(record.get("answer") or "").strip()
    choices = choice_texts(record.get("choices") or [])
    letter = answer_choice_letter(answer)
    if letter and choices:
        index = ord(letter) - ord("A")
        if 0 <= index < len(choices):
            return choices[index].strip()
    return strip_choice_letter_prefix(answer)


def answer_choice_letter(answer: str) -> str:
    match = re.match(r"^\s*([A-Z])(?:[.)]|:)\s+", answer.strip(), flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""


def strip_choice_letter_prefix(text: str) -> str:
    return re.sub(r"^\s*[A-Z](?:[.)]|:)\s+", "", text.strip(), count=1, flags=re.IGNORECASE).strip()


def choice_texts(choices: Any) -> list[str]:
    out: list[str] = []
    for choice in choices or []:
        if isinstance(choice, dict):
            text = choice.get("text") or choice.get("answer") or choice.get("label") or ""
        else:
            text = str(choice)
        out.append(str(text).strip())
    return out


def strip_answer_choices(question: str) -> str:
    lines = []
    for line in question.splitlines():
        stripped = line.strip()
        if len(stripped) > 3 and stripped[0] == "(" and stripped[2] == ")" and stripped[1].isalpha():
            continue
        if len(stripped) > 2 and stripped[0].isalpha() and stripped[1] == ".":
            continue
        lowered = stripped.lower()
        if lowered.startswith("answer with") or lowered.startswith("answer only with"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Stage2 multiple-choice rows into open-answer rows.")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--files", nargs="+", default=list(DEFAULT_FILES))
    parser.add_argument("--report", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
