from __future__ import annotations

import argparse

import torch

from eval.metrics import grouped_means, mean, median
from eval.v3_common import (
    add_v3_common_eval_args,
    can_score_fvt_for_item,
    compute_or_load_v3_eval_item,
    compute_v3_readout_nll,
    group_indices_by_v3_image,
    load_qwen3_and_tgvf,
    load_v3_eval_samples,
    make_output_dir,
    progress_iter,
    sample_metadata_row,
    save_summary,
    set_seed,
    summarize_config,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TGVF-v3 same-image query sensitivity.")
    add_v3_common_eval_args(parser)
    parser.add_argument("--min-targets-per-image", type=int, default=3)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--require-groups", type=int, default=0)
    parser.add_argument("--save-score-matrices", action="store_true")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = make_output_dir(
        args.output_dir,
        prefix="v3_query_sensitivity",
        overwrite=args.overwrite_output_dir,
    )
    matrices_dir = output_dir / "score_matrices"
    matrices_dir.mkdir(exist_ok=True)
    samples = load_v3_eval_samples(
        args.eval_jsonl,
        max_samples=args.max_samples,
        min_confidence=args.min_confidence,
        source_profile_filter=args.source_profile_filter,
        evidence_type_filter=args.evidence_type_filter,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        shard_key=args.shard_key,
    )
    if not samples:
        raise RuntimeError("No v3 focus samples found for query sensitivity evaluation.")
    model, processor, foveal_module, device, model_info = load_qwen3_and_tgvf(args, example_sample=samples[0])
    write_json(output_dir / "config.json", summarize_config(args, model_info))

    items = [
        compute_or_load_v3_eval_item(
            model=model,
            processor=processor,
            foveal_module=foveal_module,
            sample=sample,
            device=device,
            capture_layer=args.capture_layer,
            max_image_resolution=args.max_image_resolution,
            position_mode=args.fvt_position_mode,
            mask_original_image_after_tgvf=args.mask_original_image_after_tgvf,
            cache_dir=args.fvt_cache_dir,
            checkpoint_path=args.tgvf_checkpoint,
            variant=args.variant,
            use_cache=args.use_fvt_cache,
        )
        for sample in progress_iter(
            samples,
            desc="v3 query: build FVT",
            total=len(samples),
            enabled=args.progress,
            progress_file=args.progress_file,
        )
    ]
    groups = {
        group_id: indices
        for group_id, indices in group_indices_by_v3_image(items).items()
        if len(indices) >= args.min_targets_per_image
    }
    num_eligible_groups = len(groups)
    if args.require_groups and num_eligible_groups < args.require_groups:
        raise RuntimeError(
            f"Need at least {args.require_groups} image groups with "
            f">= {args.min_targets_per_image} targets, found {num_eligible_groups}"
        )
    if args.max_groups is not None:
        groups = dict(list(groups.items())[: args.max_groups])

    group_rows = []
    per_item_rows = []
    with torch.no_grad():
        for group_id, indices in progress_iter(
            groups.items(),
            desc="v3 query: score groups",
            total=len(groups),
            enabled=args.progress,
            progress_file=args.progress_file,
        ):
            group_items = [items[index] for index in indices]
            n = len(group_items)
            nll_matrix = []
            for row_item in group_items:
                row = []
                for col_item in group_items:
                    if not can_score_fvt_for_item(row_item, col_item.foveated_visual_tokens):
                        row.append(float("inf"))
                        continue
                    nll = compute_v3_readout_nll(
                        model=model,
                        tokenizer_or_processor=processor,
                        capture=row_item.capture,
                        evidence_description=row_item.sample.evidence_description,
                        foveated_visual_tokens=col_item.foveated_visual_tokens,
                        device=device,
                        mask_original_image_after_tgvf=args.mask_original_image_after_tgvf,
                        position_mode=args.fvt_position_mode,
                    )
                    row.append(float(nll["avg_nll"]))
                nll_matrix.append(row)

            diagonal_ranks = []
            diagonal_gaps = []
            top1 = []
            top2 = []
            for i, row in enumerate(nll_matrix):
                ranked = sorted(range(n), key=lambda j: row[j])
                rank = ranked.index(i) + 1
                diagonal_ranks.append(rank)
                top1.append(float(rank == 1))
                top2.append(float(rank <= 2))
                wrong = [row[j] for j in range(n) if j != i and row[j] != float("inf")]
                gap = min(wrong) - row[i] if wrong and row[i] != float("inf") else None
                diagonal_gaps.append(gap)
                per_item_rows.append(
                    {
                        **sample_metadata_row(group_items[i].sample),
                        "diagonal_rank": rank,
                        "diagonal_gap": gap,
                        "top1": float(rank == 1),
                        "top2": float(rank <= 2),
                        "group_size": n,
                        "shape_compatible_group": all(
                            can_score_fvt_for_item(group_items[i], other.foveated_visual_tokens)
                            for other in group_items
                        ),
                    }
                )

            group_row = {
                "stable_image_uid": group_id,
                "group_size": n,
                "targets": [item.sample.target for item in group_items],
                "evidence_descriptions": [item.sample.evidence_description for item in group_items],
                "evidence_types": [item.sample.evidence_type for item in group_items],
                "score_type": "nll_lower_is_better",
                "nll_matrix": nll_matrix,
                "diagonal_ranks": diagonal_ranks,
                "diagonal_gaps": diagonal_gaps,
                "top1_accuracy": mean(top1),
                "top2_accuracy": mean(top2),
                "mrr": mean(1.0 / rank for rank in diagonal_ranks),
                "mean_diagonal_gap": mean(diagonal_gaps),
                "median_diagonal_gap": median(diagonal_gaps),
            }
            group_rows.append(group_row)
            if args.save_score_matrices:
                safe_name = group_id.replace("/", "_").replace(":", "_")
                write_json(matrices_dir / f"{safe_name}.json", group_row)

    metric_keys = ["top1", "top2", "diagonal_gap"]
    report = {
        "config": summarize_config(args, model_info),
        "num_samples_attempted": len(samples),
        "num_eligible_groups": num_eligible_groups,
        "num_groups_evaluated": len(group_rows),
        "num_items_evaluated": len(per_item_rows),
        "score_type": "nll_lower_is_better",
        "metrics": {
            "retrieval_top1": mean(row["top1"] for row in per_item_rows),
            "retrieval_top2": mean(row["top2"] for row in per_item_rows),
            "mrr": mean(1.0 / row["diagonal_rank"] for row in per_item_rows),
            "mean_diagonal_gap": mean(row["diagonal_gap"] for row in per_item_rows),
            "median_diagonal_gap": median(row["diagonal_gap"] for row in per_item_rows),
            "second_full_forward_used_any": False,
        },
        "metrics_by_evidence_type": grouped_means(per_item_rows, group_key="evidence_type", metric_keys=metric_keys),
        "metrics_by_source_profile": grouped_means(per_item_rows, group_key="source_profile", metric_keys=metric_keys),
        "position_mode": args.fvt_position_mode,
        "mask_original_image_after_tgvf": args.mask_original_image_after_tgvf,
    }
    write_json(output_dir / "query_sensitivity_report.json", report)
    write_jsonl(output_dir / "per_group_results.jsonl", group_rows)
    write_jsonl(output_dir / "per_item_results.jsonl", per_item_rows)
    save_summary(
        output_dir / "query_sensitivity_summary.txt",
        "TGVF-v3 Query Sensitivity Evaluation",
        [
            f"groups_evaluated: {len(group_rows)}",
            f"items_evaluated: {len(per_item_rows)}",
            f"retrieval_top1: {report['metrics']['retrieval_top1']}",
            f"mrr: {report['metrics']['mrr']}",
            f"mean_diagonal_gap: {report['metrics']['mean_diagonal_gap']}",
            f"position_mode: {args.fvt_position_mode}",
            "score_type: nll_lower_is_better",
            "second_full_forward_used_any: False",
        ],
    )


if __name__ == "__main__":
    main()
