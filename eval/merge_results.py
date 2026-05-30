from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from eval.common import save_summary, write_json, write_jsonl
from eval.metrics import grouped_means, mean, median, pct_positive


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge sharded TGVF eval outputs.")
    parser.add_argument("--task", choices=["readout", "query", "distribution", "end2end"], required=True)
    parser.add_argument("--input-root", required=True, help="Directory containing shard_*_of_* subdirectories.")
    parser.add_argument("--output-dir", required=True, help="Task root where merged outputs are written.")
    parser.add_argument("--max-groups", type=int, default=None, help="Optional query-only cap after merging shards.")
    parser.add_argument("--require-groups", type=int, default=0, help="Optional query-only minimum after merging shards.")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.task == "readout":
        report = merge_readout(input_root, output_dir)
    elif args.task == "query":
        report = merge_query(
            input_root,
            output_dir,
            max_groups=args.max_groups,
            require_groups=args.require_groups,
        )
    elif args.task == "distribution":
        report = merge_distribution(input_root, output_dir)
    elif args.task == "end2end":
        report = merge_end2end(input_root, output_dir)
    else:  # pragma: no cover
        raise ValueError(args.task)
    print(json.dumps(report, indent=2))


def merge_readout(input_root: Path, output_dir: Path) -> dict[str, Any]:
    rows = _read_all_jsonl(input_root, "per_sample_results.jsonl")
    metric_keys = [
        "delta_correct_vs_target_only",
        "delta_correct_vs_random",
        "delta_correct_vs_wrong_same",
        "delta_correct_vs_wrong_diff",
    ]
    report = {
        "num_shards": len(_shard_dirs(input_root)),
        "num_samples_evaluated": len(rows),
        "metrics": {
            "mean_nll_correct_D": mean(row["nlls"]["correct_D_plus_target"] for row in rows),
            "median_nll_correct_D": median(row["nlls"]["correct_D_plus_target"] for row in rows),
            "mean_nll_target_only": mean(row["nlls"]["target_only_no_D"] for row in rows),
            "mean_nll_random_D": mean(row["nlls"]["random_D_plus_target"] for row in rows),
            "mean_delta_correct_vs_target_only": mean(row["delta_correct_vs_target_only"] for row in rows),
            "mean_delta_correct_vs_random": mean(row["delta_correct_vs_random"] for row in rows),
            "mean_delta_correct_vs_wrong_same": mean(row["delta_correct_vs_wrong_same"] for row in rows),
            "mean_delta_correct_vs_wrong_diff": mean(row["delta_correct_vs_wrong_diff"] for row in rows),
            "pct_correct_D_beats_target_only": pct_positive(row["delta_correct_vs_target_only"] for row in rows),
            "pct_correct_D_beats_random": pct_positive(row["delta_correct_vs_random"] for row in rows),
            "pct_correct_D_beats_wrong_same": pct_positive(row["delta_correct_vs_wrong_same"] for row in rows),
            "pct_correct_D_beats_wrong_diff": pct_positive(row["delta_correct_vs_wrong_diff"] for row in rows),
        },
        "metrics_by_evidence_type": grouped_means(rows, group_key="evidence_type", metric_keys=metric_keys),
        "metrics_by_source_profile": grouped_means(rows, group_key="source_profile", metric_keys=metric_keys),
        "metrics_by_answer_type": grouped_means(rows, group_key="answer_type", metric_keys=metric_keys),
        "metrics_by_visual_difficulty": grouped_means(rows, group_key="visual_difficulty", metric_keys=metric_keys),
    }
    write_json(output_dir / "readout_eval_report.json", report)
    write_jsonl(output_dir / "per_sample_results.jsonl", rows)
    save_summary(
        output_dir / "readout_eval_summary.txt",
        "Merged TGVF Readout Evaluation",
        [
            f"samples_evaluated: {len(rows)}",
            f"mean_nll_correct_D: {report['metrics']['mean_nll_correct_D']}",
            f"mean_nll_target_only: {report['metrics']['mean_nll_target_only']}",
            f"mean_delta_correct_vs_target_only: {report['metrics']['mean_delta_correct_vs_target_only']}",
            f"mean_delta_correct_vs_wrong_same: {report['metrics']['mean_delta_correct_vs_wrong_same']}",
        ],
    )
    return report


def merge_query(
    input_root: Path,
    output_dir: Path,
    *,
    max_groups: int | None = None,
    require_groups: int = 0,
) -> dict[str, Any]:
    group_rows = _read_all_jsonl(input_root, "per_group_results.jsonl")
    item_rows = _read_all_jsonl(input_root, "per_item_results.jsonl")
    if require_groups and len(group_rows) < require_groups:
        raise RuntimeError(f"query groups evaluated {len(group_rows)} < required {require_groups}")
    if max_groups is not None:
        group_rows = group_rows[:max_groups]
        selected_group_ids = {row["stable_image_uid"] for row in group_rows}
        item_rows = [row for row in item_rows if row["stable_image_uid"] in selected_group_ids]
    matrices_dir = output_dir / "score_matrices"
    matrices_dir.mkdir(exist_ok=True)
    for shard in _shard_dirs(input_root):
        source = shard / "score_matrices"
        if not source.exists():
            continue
        for matrix in source.glob("*.json"):
            target = matrices_dir / f"{shard.name}__{matrix.name}"
            shutil.copyfile(matrix, target)
    metric_keys = ["top1", "top2", "diagonal_gap"]
    report = {
        "num_shards": len(_shard_dirs(input_root)),
        "num_groups_evaluated": len(group_rows),
        "num_items_evaluated": len(item_rows),
        "score_type": "nll_lower_is_better",
        "metrics": {
            "retrieval_top1": mean(row["top1"] for row in item_rows),
            "retrieval_top2": mean(row["top2"] for row in item_rows),
            "mrr": mean(1.0 / row["diagonal_rank"] for row in item_rows),
            "mean_diagonal_gap": mean(row["diagonal_gap"] for row in item_rows),
            "median_diagonal_gap": median(row["diagonal_gap"] for row in item_rows),
        },
        "metrics_by_evidence_type": grouped_means(item_rows, group_key="evidence_type", metric_keys=metric_keys),
        "metrics_by_source_profile": grouped_means(item_rows, group_key="source_profile", metric_keys=metric_keys),
    }
    write_json(output_dir / "query_sensitivity_report.json", report)
    write_jsonl(output_dir / "per_group_results.jsonl", group_rows)
    write_jsonl(output_dir / "per_item_results.jsonl", item_rows)
    save_summary(
        output_dir / "query_sensitivity_summary.txt",
        "Merged TGVF Query Sensitivity Evaluation",
        [
            f"groups_evaluated: {len(group_rows)}",
            f"items_evaluated: {len(item_rows)}",
            f"retrieval_top1: {report['metrics']['retrieval_top1']}",
            f"mrr: {report['metrics']['mrr']}",
            f"mean_diagonal_gap: {report['metrics']['mean_diagonal_gap']}",
            "score_type: nll_lower_is_better",
        ],
    )
    return report


def merge_distribution(input_root: Path, output_dir: Path) -> dict[str, Any]:
    rows = _read_all_jsonl(input_root, "per_sample_distribution.jsonl")
    finite_rate = mean(1.0 if row["finite"] else 0.0 for row in rows)
    collapse_rate = mean(row["collapse_near_identical_tokens"] for row in rows)
    report = {
        "num_shards": len(_shard_dirs(input_root)),
        "num_samples_evaluated": len(rows),
        "metrics": {
            "avg_manifold_loss": mean(row["manifold_loss"] for row in rows),
            "median_manifold_loss": median(row["manifold_loss"] for row in rows),
            "avg_mean_mse": mean(row["mean_mse"] for row in rows),
            "avg_std_mse": mean(row["std_mse"] for row in rows),
            "avg_norm_D": mean(row["D_stats"]["mean_token_norm"] for row in rows),
            "avg_norm_V_merge": mean(row["V_merge_stats"]["mean_token_norm"] for row in rows),
            "norm_ratio_D_to_Vmerge": mean(row["norm_ratio_D_to_Vmerge"] for row in rows),
            "finite_rate": finite_rate,
            "collapse_near_identical_rate": collapse_rate,
            "collapse_warning": bool((collapse_rate or 0.0) > 0.1 or (finite_rate or 0.0) < 1.0),
        },
        "metrics_by_evidence_type": grouped_means(rows, group_key="evidence_type", metric_keys=["manifold_loss", "norm_ratio_D_to_Vmerge"]),
        "metrics_by_source_profile": grouped_means(rows, group_key="source_profile", metric_keys=["manifold_loss", "norm_ratio_D_to_Vmerge"]),
    }
    write_json(output_dir / "fvt_distribution_report.json", report)
    write_jsonl(output_dir / "per_sample_distribution.jsonl", rows)
    save_summary(
        output_dir / "fvt_distribution_summary.txt",
        "Merged TGVF FVT Distribution Evaluation",
        [
            f"samples_evaluated: {len(rows)}",
            f"avg_manifold_loss: {report['metrics']['avg_manifold_loss']}",
            f"norm_ratio_D_to_Vmerge: {report['metrics']['norm_ratio_D_to_Vmerge']}",
            f"finite_rate: {report['metrics']['finite_rate']}",
            f"collapse_warning: {report['metrics']['collapse_warning']}",
        ],
    )
    return report


def merge_end2end(input_root: Path, output_dir: Path) -> dict[str, Any]:
    rows = _read_all_jsonl(input_root, "per_sample_generations.jsonl")
    condition_names = sorted({name for row in rows for name in row["outputs"].keys()})
    report_metrics = {}
    for condition in condition_names:
        condition_rows = [row["outputs"][condition] for row in rows if condition in row["outputs"]]
        report_metrics[condition] = {
            "exact_match": mean(item["scores"]["exact_match"] for item in condition_rows),
            "substring_match": mean(item["scores"]["substring_match"] for item in condition_rows),
            "token_f1": mean(item["scores"]["token_f1"] for item in condition_rows),
            "char_f1": mean(item["scores"]["char_f1"] for item in condition_rows),
            "completion_rate": mean(1.0 if item.get("answer") else 0.0 for item in condition_rows),
        }
    correct = report_metrics.get("correct_TGVF_D", {})
    no_d = report_metrics.get("foveation_no_D", {})
    wrong = report_metrics.get("wrong_D", {})
    report = {
        "num_shards": len(_shard_dirs(input_root)),
        "num_samples_evaluated": len(rows),
        "metrics": report_metrics,
        "highlight": {
            "correct_D_token_f1": correct.get("token_f1"),
            "no_D_token_f1": no_d.get("token_f1"),
            "wrong_D_token_f1": wrong.get("token_f1"),
            "correct_D_improvement_over_no_D": _delta(correct.get("token_f1"), no_d.get("token_f1")),
            "correct_D_improvement_over_wrong_D": _delta(correct.get("token_f1"), wrong.get("token_f1")),
            "second_full_forward_used": False,
        },
    }
    write_json(output_dir / "end2end_eval_report.json", report)
    write_jsonl(output_dir / "per_sample_generations.jsonl", rows)
    save_summary(
        output_dir / "end2end_eval_summary.txt",
        "Merged TGVF End-to-End Evaluation",
        [
            f"samples_evaluated: {len(rows)}",
            f"correct_D_token_f1: {report['highlight']['correct_D_token_f1']}",
            f"correct_D_improvement_over_no_D: {report['highlight']['correct_D_improvement_over_no_D']}",
            f"correct_D_improvement_over_wrong_D: {report['highlight']['correct_D_improvement_over_wrong_D']}",
            "second_full_forward_used: false",
        ],
    )
    return report


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _shard_dirs(input_root: Path) -> list[Path]:
    return sorted(path for path in input_root.glob("shard_*_of_*") if path.is_dir())


def _read_all_jsonl(input_root: Path, filename: str) -> list[dict[str, Any]]:
    rows = []
    for shard in _shard_dirs(input_root):
        path = shard / filename
        if not path.exists():
            continue
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
