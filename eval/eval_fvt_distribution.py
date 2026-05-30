from __future__ import annotations

import argparse

import torch

from eval.common import (
    add_common_eval_args,
    compute_or_load_eval_item,
    load_eval_samples,
    load_qwen_and_tgvf,
    make_output_dir,
    progress_iter,
    save_summary,
    set_seed,
    summarize_config,
    write_json,
    write_jsonl,
)
from eval.metrics import grouped_means, mean, median, tensor_distribution_stats
from revisit_vlm.tgvf_training import visual_token_manifold_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated FVT distribution.")
    add_common_eval_args(parser)
    parser.add_argument("--save-histograms", action="store_true")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = make_output_dir(
        args.output_dir,
        prefix="fvt_distribution",
        overwrite=args.overwrite_output_dir,
    )
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

    rows = []
    d_norms = []
    v_norms = []
    with torch.no_grad():
        for sample in progress_iter(
            samples,
            desc="distribution: build and score FVT",
            total=len(samples),
            enabled=args.progress,
            progress_file=args.progress_file,
        ):
            item = compute_or_load_eval_item(
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
            d = item.foveated_visual_tokens.float()
            v = item.merged_visual_tokens.float()
            man_loss = visual_token_manifold_loss(d, v)
            d_stats = tensor_distribution_stats(d)
            v_stats = tensor_distribution_stats(v)
            norm_ratio = None
            if v_stats["mean_token_norm"] and v_stats["mean_token_norm"] != 0:
                norm_ratio = d_stats["mean_token_norm"] / v_stats["mean_token_norm"]
            pooled_cosine = torch.nn.functional.cosine_similarity(
                d.mean(dim=0).view(1, -1),
                v.mean(dim=0).view(1, -1),
            ).item()
            token_cos = torch.nn.functional.cosine_similarity(
                d.float(),
                d.float().mean(dim=0, keepdim=True),
                dim=-1,
            )
            near_identical = float(token_cos.std(unbiased=False).item() < 1e-4)
            row = {
                "uid": sample.uid,
                "stable_image_uid": sample.group_id,
                "evidence_type": sample.evidence_type,
                "source_profile": sample.source_profile,
                "answer_type": sample.answer_type,
                "visual_difficulty": sample.visual_difficulty,
                "D_stats": d_stats,
                "V_merge_stats": v_stats,
                "manifold_loss": float(man_loss.detach().cpu()),
                "mean_mse": float(torch.nn.functional.mse_loss(d.mean(dim=0), v.mean(dim=0)).item()),
                "std_mse": float(torch.nn.functional.mse_loss(d.std(dim=0, unbiased=False), v.std(dim=0, unbiased=False)).item()),
                "norm_ratio_D_to_Vmerge": norm_ratio,
                "pooled_D_to_pooled_Vmerge_cosine": pooled_cosine,
                "finite": bool(torch.isfinite(d).all().item() and torch.isfinite(v).all().item()),
                "collapse_near_identical_tokens": near_identical,
                "shapes": item.shapes,
            }
            rows.append(row)
            d_norms.extend(d.norm(dim=-1).cpu().tolist())
            v_norms.extend(v.norm(dim=-1).cpu().tolist())

    finite_rate = mean(1.0 if row["finite"] else 0.0 for row in rows)
    collapse_rate = mean(row["collapse_near_identical_tokens"] for row in rows)
    report = {
        "config": summarize_config(args, model_info),
        "num_samples_attempted": len(samples),
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
        "metrics_by_evidence_type": grouped_means(
            rows,
            group_key="evidence_type",
            metric_keys=["manifold_loss", "norm_ratio_D_to_Vmerge"],
        ),
        "metrics_by_source_profile": grouped_means(
            rows,
            group_key="source_profile",
            metric_keys=["manifold_loss", "norm_ratio_D_to_Vmerge"],
        ),
    }
    write_json(output_dir / "fvt_distribution_report.json", report)
    write_jsonl(output_dir / "per_sample_distribution.jsonl", rows)

    if args.save_histograms:
        try:
            import matplotlib.pyplot as plt

            hist_dir = output_dir / "histograms"
            hist_dir.mkdir(exist_ok=True)
            for name, values in {"D_token_norms": d_norms, "V_merge_token_norms": v_norms}.items():
                plt.figure()
                plt.hist(values, bins=50)
                plt.title(name)
                plt.savefig(hist_dir / f"{name}.png")
                plt.close()
        except Exception as exc:  # pragma: no cover - optional plotting path
            write_json(output_dir / "histogram_error.json", {"error": repr(exc)})

    save_summary(
        output_dir / "fvt_distribution_summary.txt",
        "TGVF FVT Distribution Evaluation",
        [
            f"samples_evaluated: {len(rows)}",
            f"avg_manifold_loss: {report['metrics']['avg_manifold_loss']}",
            f"norm_ratio_D_to_Vmerge: {report['metrics']['norm_ratio_D_to_Vmerge']}",
            f"finite_rate: {report['metrics']['finite_rate']}",
            f"collapse_warning: {report['metrics']['collapse_warning']}",
        ],
    )


if __name__ == "__main__":
    main()
