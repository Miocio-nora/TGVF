#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


TEXT_KEYS = {
    "question",
    "answer",
    "short_answer",
    "value_span_text",
    "target",
    "evidence_description",
    "pre_focus_think",
    "post_focus_think",
    "no_focus_think",
}

BAD_SUBSTRINGS = (
    "'}],",
    '"}],',
    "'}]}",
    '"}]}',
    "'choices':",
    '"choices":',
    "'question':",
    '"question":',
    "'answer':",
    '"answer":',
    "metadata:",
)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    reports = {}
    for rel in args.files:
        report = clean_file(input_root / rel, output_root / rel)
        reports[rel] = report
        print(json.dumps({"file": rel, **report}, ensure_ascii=False), flush=True)
    report_path = output_root / "clean_report.json"
    report_path.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"report": str(report_path)}, ensure_ascii=False), flush=True)


def clean_file(input_path: Path, output_path: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            counts["input"] += 1
            row = json.loads(line)
            reason = bad_reason(row)
            if reason:
                counts["dropped"] += 1
                counts[f"dropped_{reason}"] += 1
                continue
            counts["kept"] += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    return dict(counts)


def bad_reason(value: Any, *, path: str = "") -> str:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            reason = bad_reason(child, path=child_path)
            if reason:
                return reason
        return ""
    if isinstance(value, list):
        for index, child in enumerate(value):
            reason = bad_reason(child, path=f"{path}[{index}]")
            if reason:
                return reason
        return ""
    if not isinstance(value, str):
        return ""
    if path.split(".")[-1] not in TEXT_KEYS and not path.endswith(".text") and not path.endswith(".focus_text"):
        return ""
    text = value.strip()
    if not text:
        return ""
    for bad in BAD_SUBSTRINGS:
        if bad in text:
            return f"bad_text_{path or 'text'}"
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter polluted Protocol-C TGVF split rows.")
    parser.add_argument(
        "--input-root",
        default="data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k/splits",
    )
    parser.add_argument(
        "--output-root",
        default="data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=[
            "tgvf_v4_teacher_stage1_protocol_c_focus.train.jsonl",
            "tgvf_v4_teacher_stage1_protocol_c_focus.test.jsonl",
            "tgvf_v4_teacher_stage2_protocol_c.train.jsonl",
            "tgvf_v4_teacher_stage2_protocol_c.test.jsonl",
        ],
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
