#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    report = convert(Path(args.input), Path(args.output))
    report_path = Path(args.report) if args.report else Path(args.output).with_suffix(Path(args.output).suffix + ".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


def convert(input_path: Path, output_path: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            counts["raw_total"] += 1
            item_type = str(record.get("item_type") or record.get("trajectory_type") or "")
            counts[f"raw_{item_type}"] += 1
            row = convert_record(record, line_no=line_no)
            if row is None:
                counts[f"skipped_{item_type or 'unknown'}"] += 1
                continue
            counts[f"written_{row['trajectory_type']}"] += 1
            counts["written_total"] += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    focus = counts["written_single_focus"] + counts["written_multi_focus"]
    no_focus = counts["written_direct_answer"]
    total = counts["written_total"]
    counts["written_focus"] = focus
    counts["written_no_focus"] = no_focus
    counts["focus_ratio"] = focus / total if total else 0.0
    counts["no_focus_ratio"] = no_focus / total if total else 0.0
    return dict(counts)


def convert_record(record: dict[str, Any], *, line_no: int) -> dict[str, Any] | None:
    item_type = str(record.get("item_type") or record.get("trajectory_type") or "")
    common = common_fields(record)
    if not common.get("image") or not common.get("question") or not common.get("answer"):
        return None
    if item_type == "single_refocus":
        focus = first_focus(record)
        if not focus:
            return None
        trace_parts = trace_parts_for_single(record)
        return common | {
            "need_focus": True,
            "evidence_state": "need_local_visual_evidence",
            "trajectory_type": "single_focus",
            "target": focus["target"],
            "evidence_description": focus["evidence_description"],
            "pre_focus_think": trace_parts.get("pre_focus_think"),
            "post_focus_think": trace_parts.get("post_focus_think"),
            "target_style": "visual_descriptor",
            "target_cues": focus.get("target_cues") or record.get("focus_descriptor_cues") or [],
            "target_leakage_risk": focus.get("target_leakage_risk") or record.get("target_leakage_risk") or "low",
            "evidence_specificity": "specific",
        }
    if item_type == "multi_refocus":
        steps = focus_steps(record)
        if len(steps) != 2:
            raise ValueError(f"{line_no}: V4 cold-start multi_refocus must have exactly two focus steps")
        trace_parts = trace_parts_for_multi(record)
        for index, step in enumerate(steps):
            if index < len(trace_parts):
                step.update(trace_parts[index])
        return common | {
            "need_focus": True,
            "evidence_state": "need_local_visual_evidence",
            "trajectory_type": "multi_focus",
            "target": steps[0]["target"],
            "evidence_description": steps[-1]["evidence_description"],
            "target_style": "visual_descriptor",
            "target_cues": sorted({cue for step in steps for cue in step.get("target_cues", [])}),
            "target_leakage_risk": max((step.get("target_leakage_risk") or "low" for step in steps), default="low"),
            "evidence_specificity": "specific",
            "focus_steps": steps,
        }
    if item_type in {"no_refocus_continue", "no_refocus_answer"}:
        think = first_think(record)
        return common | {
            "need_focus": False,
            "evidence_state": "sufficient_visual_evidence",
            "trajectory_type": "direct_answer",
            "target": "",
            "evidence_description": think,
            "no_focus_think": think,
            "target_style": "none",
            "target_cues": [],
            "target_leakage_risk": "none",
            "evidence_specificity": "specific",
        }
    return None


def common_fields(record: dict[str, Any]) -> dict[str, Any]:
    answer = answer_text(record)
    return {
        "schema_version": "tgvf_teacher_schema_v4_stage2_compat",
        "teacher_prompt_version": record.get("teacher_prompt_version") or record.get("teacher_version") or "tgvf_v4_teacher",
        "image": record.get("image"),
        "image_id": record.get("image_id") or record.get("stable_image_uid"),
        "source_dataset": record.get("source_dataset"),
        "source_profile": record.get("source_profile"),
        "question": record.get("question"),
        "choices": record.get("choices") or [],
        "answer": answer,
        "short_answer": record.get("answer_text") or answer,
        "answer_format": "multiple_choice" if record.get("answer_format") == "multiple_choice" else "short_text",
        "value_span_text": record.get("answer_text") or answer,
        "evidence_type": first_value(record.get("evidence_types")) or record.get("evidence_type") or "other",
        "confidence": record.get("confidence"),
        "v4_item_type": record.get("item_type"),
        "v4_uid": record.get("uid"),
        "question_type": record.get("question_type"),
        "focus_category": record.get("focus_category"),
    }


def first_focus(record: dict[str, Any]) -> dict[str, Any] | None:
    steps = focus_steps(record)
    return steps[0] if steps else None


def focus_steps(record: dict[str, Any]) -> list[dict[str, Any]]:
    steps = []
    for step in record.get("trace") or []:
        if step.get("type") != "focus":
            continue
        metadata = step.get("metadata") or {}
        target = str(step.get("focus_text") or "").strip()
        evidence = str(step.get("focused_evidence") or "").strip()
        if not target or not evidence:
            continue
        steps.append(
            {
                "target": target,
                "evidence_description": evidence,
                "target_cues": metadata.get("focus_descriptor_cues") or [],
                "target_leakage_risk": metadata.get("target_leakage_risk") or "low",
                "evidence_type": metadata.get("evidence_type") or first_value(record.get("evidence_types")) or "other",
                "value_span_text": record.get("answer_text"),
            }
        )
    return steps


def trace_parts_for_single(record: dict[str, Any]) -> dict[str, str | None]:
    trace = record.get("trace") or []
    focus_indices = [index for index, step in enumerate(trace) if step.get("type") == "focus"]
    if not focus_indices:
        return {"pre_focus_think": None, "post_focus_think": None}
    focus_index = focus_indices[0]
    return {
        "pre_focus_think": nearest_think_before(trace, focus_index),
        "post_focus_think": nearest_think_after(trace, focus_index),
    }


def trace_parts_for_multi(record: dict[str, Any]) -> list[dict[str, str | None]]:
    trace = record.get("trace") or []
    focus_indices = [index for index, step in enumerate(trace) if step.get("type") == "focus"]
    parts: list[dict[str, str | None]] = []
    for focus_index in focus_indices:
        parts.append(
            {
                "pre_think": nearest_think_before(trace, focus_index),
                "post_think": nearest_think_after(trace, focus_index),
            }
        )
    return parts


def nearest_think_before(trace: list[dict[str, Any]], index: int) -> str | None:
    for cursor in range(index - 1, -1, -1):
        step = trace[cursor]
        if step.get("type") == "focus":
            break
        if step.get("type") == "think" and step.get("text"):
            return str(step["text"]).strip()
    return None


def nearest_think_after(trace: list[dict[str, Any]], index: int) -> str | None:
    for cursor in range(index + 1, len(trace)):
        step = trace[cursor]
        if step.get("type") == "focus":
            break
        if step.get("type") == "think" and step.get("text"):
            return str(step["text"]).strip()
    return None


def answer_text(record: dict[str, Any]) -> str:
    answer = str(record.get("answer") or record.get("answer_text") or "").strip()
    if answer:
        return answer
    for step in reversed(record.get("trace") or []):
        if step.get("type") == "answer" and step.get("text"):
            return str(step["text"]).strip()
    return ""


def first_think(record: dict[str, Any]) -> str:
    for step in record.get("trace") or []:
        if step.get("type") == "think" and step.get("text"):
            return str(step["text"]).strip()
    return ""


def first_value(value: Any) -> Any:
    return value[0] if isinstance(value, list) and value else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert V4 teacher rows to Stage2 Protocol C-compatible JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
