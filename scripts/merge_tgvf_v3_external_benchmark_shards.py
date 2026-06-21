#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded TGVF external benchmark eval outputs.")
    parser.add_argument("--mode-dir", required=True)
    parser.add_argument("--row-file", required=True)
    parser.add_argument("--summary-name", default="merged_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode_dir = Path(args.mode_dir)
    rows: list[dict[str, Any]] = []
    shard_summaries: list[dict[str, Any]] = []
    for shard in sorted(mode_dir.glob("shard_*")):
        row_path = shard / args.row_file
        if row_path.exists():
            with row_path.open() as handle:
                for line in handle:
                    if line.strip():
                        rows.append(json.loads(line))
        for name in ("benchmark_base_direct_summary.json", "benchmark_force_summary.json"):
            path = shard / name
            if path.exists():
                shard_summaries.append(json.loads(path.read_text()))
                break
    rows.sort(key=lambda row: (str(row.get("id") or row.get("sample_id") or ""), str(row.get("method") or "")))
    write_jsonl(mode_dir / "merged_rows.jsonl", rows)
    summary = summarize(rows)
    summary["num_shards_found"] = len(list(mode_dir.glob("shard_*")))
    summary["num_shard_summaries"] = len(shard_summaries)
    summary["shard_summaries"] = shard_summaries
    write_json(mode_dir / args.summary_name, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, dict[str, Any]] = {}
    for method in sorted({str(row.get("method") or "") for row in rows}):
        rs = [row for row in rows if str(row.get("method") or "") == method]
        by_method[method] = {
            "n": len(rs),
            "accuracy": mean(row.get("score") for row in rs),
            "answer_parse_rate": mean(row.get("answer_parse_success") for row in rs),
            "focus_valid_rate": mean(row.get("focus_valid") for row in rs if row.get("focus_valid") is not None),
            "append_success_rate": mean(row.get("append_success") for row in rs if row.get("append_success") is not None),
            "trigger_rate": mean(row.get("trigger_focus_decision") for row in rs if row.get("trigger_focus_decision") is not None),
            "continuation_not_im_end_rate": mean(row.get("continuation_not_im_end") for row in rs if row.get("continuation_not_im_end") is not None),
            "pred_counts": count(row.get("pred_letter") or row.get("parsed_answer") or "" for row in rs),
            "gold_counts": count(row.get("label") or row.get("gold_answer") or "" for row in rs),
            "category_accuracy": group_accuracy(rs, "category"),
        }
    free_policy = [row for row in rows if row.get("method") in {"free_direct_or_miss", "free_correct_D"}]
    return {
        "n_total_rows": len(rows),
        "by_method": by_method,
        "free_policy_correct_D": {
            "n": len(free_policy),
            "accuracy": mean(row.get("score") for row in free_policy),
            "answer_parse_rate": mean(row.get("answer_parse_success") for row in free_policy),
            "trigger_rate": mean(row.get("trigger_focus_decision") for row in free_policy),
        } if free_policy else None,
        "second_full_forward_used_any": any(bool(row.get("second_full_forward_used")) for row in rows),
    }


def group_accuracy(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        value = row.get(key) or (row.get("metadata") or {}).get(key)
        if value is None or row.get("score") is None:
            continue
        groups.setdefault(str(value), []).append(float(row.get("score") or 0.0))
    return {name: {"n": len(vals), "accuracy": mean(vals)} for name, vals in sorted(groups.items())}


def mean(values: Any) -> float | None:
    vals = [float(value) for value in values if value is not None]
    return None if not vals else sum(vals) / len(vals)


def count(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return dict(sorted(out.items()))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


if __name__ == "__main__":
    main()
