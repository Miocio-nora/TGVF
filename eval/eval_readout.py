from __future__ import annotations

import argparse

import torch

from eval.common import (
    add_common_eval_args,
    compute_or_load_eval_item,
    compute_readout_nll,
    different_image_index,
    group_indices_by_image,
    load_eval_samples,
    load_qwen_and_tgvf,
    make_output_dir,
    progress_iter,
    random_d_like,
    same_image_wrong_index,
    save_summary,
    set_seed,
    summarize_config,
    write_json,
    write_jsonl,
)
from eval.metrics import grouped_means, mean, median, pct_positive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fresh-context TGVF FVT readout.")
    add_common_eval_args(parser)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = make_output_dir(args.output_dir, prefix="readout", overwrite=args.overwrite_output_dir)
    samples = load_eval_samples(
        args.eval_jsonl,
        max_samples=args.max_samples,
        source_profile_filter=args.source_profile_filter,
        evidence_type_filter=args.evidence_type_filter,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        shard_key=args.shard_key,
    )
    model, processor, foveal_module, device, model_info = load_qwen_and_tgvf(args)
    write_json(output_dir / "config.json", summarize_config(args, model_info))

    items = [
        compute_or_load_eval_item(
            model=model,
            processor=processor,
            foveal_module=foveal_module,
            sample=sample,
            device=device,
            capture_layer=args.capture_layer,
            cache_dir=args.fvt_cache_dir,
            checkpoint_path=args.tgvf_checkpoint,
            variant=args.variant,
            use_cache=args.use_fvt_cache,
        )
        for sample in progress_iter(
            samples,
            desc="readout: build FVT",
            total=len(samples),
            enabled=args.progress,
            progress_file=args.progress_file,
        )
    ]
    groups = group_indices_by_image(items)
    rows = []
    with torch.no_grad():
        for index, item in progress_iter(
            enumerate(items),
            desc="readout: score conditions",
            total=len(items),
            enabled=args.progress,
            progress_file=args.progress_file,
        ):
            sample = item.sample
            d = item.foveated_visual_tokens
            same_index = same_image_wrong_index(groups, item, index)
            diff_index = different_image_index(items, index)
            random_d = random_d_like(d, reference=item.merged_visual_tokens, seed=args.seed + index)

            nll_correct = compute_readout_nll(
                model=model,
                tokenizer_or_processor=processor,
                target_text=sample.target,
                evidence_description=sample.evidence_description,
                foveated_visual_tokens=d,
                device=device,
                image_grid_thw=item.image_grid_thw,
            )
            nll_target_only = compute_readout_nll(
                model=model,
                tokenizer_or_processor=processor,
                target_text=sample.target,
                evidence_description=sample.evidence_description,
                foveated_visual_tokens=None,
                device=device,
            )
            nll_random = compute_readout_nll(
                model=model,
                tokenizer_or_processor=processor,
                target_text=sample.target,
                evidence_description=sample.evidence_description,
                foveated_visual_tokens=random_d,
                device=device,
                image_grid_thw=item.image_grid_thw,
            )
            nll_wrong_same = None
            if same_index is not None:
                nll_wrong_same = compute_readout_nll(
                    model=model,
                    tokenizer_or_processor=processor,
                    target_text=sample.target,
                    evidence_description=sample.evidence_description,
                    foveated_visual_tokens=items[same_index].foveated_visual_tokens,
                    device=device,
                    image_grid_thw=items[same_index].image_grid_thw,
                )
            nll_wrong_diff = None
            if diff_index is not None:
                nll_wrong_diff = compute_readout_nll(
                    model=model,
                    tokenizer_or_processor=processor,
                    target_text=sample.target,
                    evidence_description=sample.evidence_description,
                    foveated_visual_tokens=items[diff_index].foveated_visual_tokens,
                    device=device,
                    image_grid_thw=items[diff_index].image_grid_thw,
                )

            nlls = {
                "correct_D_plus_target": nll_correct["avg_nll"],
                "target_only_no_D": nll_target_only["avg_nll"],
                "random_D_plus_target": nll_random["avg_nll"],
                "wrong_D_same_image_plus_target": None if nll_wrong_same is None else nll_wrong_same["avg_nll"],
                "wrong_D_different_image_plus_target": None if nll_wrong_diff is None else nll_wrong_diff["avg_nll"],
            }
            row = {
                "uid": sample.uid,
                "stable_image_uid": sample.group_id,
                "target": sample.target,
                "evidence_description": sample.evidence_description,
                "evidence_type": sample.evidence_type,
                "source_profile": sample.source_profile,
                "answer_type": sample.answer_type,
                "visual_difficulty": sample.visual_difficulty,
                "nlls": nlls,
                "delta_correct_vs_target_only": nlls["target_only_no_D"] - nlls["correct_D_plus_target"],
                "delta_correct_vs_random": nlls["random_D_plus_target"] - nlls["correct_D_plus_target"],
                "delta_correct_vs_wrong_same": None if nlls["wrong_D_same_image_plus_target"] is None else nlls["wrong_D_same_image_plus_target"] - nlls["correct_D_plus_target"],
                "delta_correct_vs_wrong_diff": None if nlls["wrong_D_different_image_plus_target"] is None else nlls["wrong_D_different_image_plus_target"] - nlls["correct_D_plus_target"],
                "condition_availability": {
                    "wrong_same_image": same_index is not None,
                    "wrong_different_image": diff_index is not None,
                },
                "shapes": item.shapes,
            }
            rows.append(row)

    metric_keys = [
        "delta_correct_vs_target_only",
        "delta_correct_vs_random",
        "delta_correct_vs_wrong_same",
        "delta_correct_vs_wrong_diff",
    ]
    report = {
        "config": summarize_config(args, model_info),
        "num_samples_attempted": len(samples),
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
        "TGVF Readout Evaluation",
        [
            f"samples_evaluated: {len(rows)}",
            f"mean_nll_correct_D: {report['metrics']['mean_nll_correct_D']}",
            f"mean_nll_target_only: {report['metrics']['mean_nll_target_only']}",
            f"mean_delta_correct_vs_target_only: {report['metrics']['mean_delta_correct_vs_target_only']}",
            f"mean_delta_correct_vs_wrong_same: {report['metrics']['mean_delta_correct_vs_wrong_same']}",
            f"pct_correct_D_beats_target_only: {report['metrics']['pct_correct_D_beats_target_only']}",
            f"pct_correct_D_beats_wrong_same: {report['metrics']['pct_correct_D_beats_wrong_same']}",
        ],
    )


if __name__ == "__main__":
    main()
