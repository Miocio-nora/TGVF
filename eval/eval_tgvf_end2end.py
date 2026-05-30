from __future__ import annotations

import argparse

import torch

from eval.common import (
    add_common_eval_args,
    compute_or_load_eval_item,
    different_image_index,
    generate_direct_qwen_answer,
    group_indices_by_image,
    load_eval_samples,
    load_qwen_and_tgvf,
    make_output_dir,
    progress_iter,
    random_d_like,
    run_forced_target_end2end,
    same_image_wrong_index,
    save_summary,
    set_seed,
    summarize_config,
    write_json,
    write_jsonl,
)
from eval.metrics import char_f1, exact_match, mean, substring_match, token_f1
from revisit_vlm.tgvf_inference import run_tgvf_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate end-to-end TGVF inference.")
    add_common_eval_args(parser)
    parser.add_argument("--mode", choices=["forced", "free"], default="forced")
    parser.add_argument("--answer-max-new-tokens", type=int, default=64)
    parser.add_argument("--capture-max-new-tokens", type=int, default=128)
    parser.add_argument("--include-direct-qwen", action="store_true", default=True)
    parser.add_argument("--skip-direct-qwen", action="store_false", dest="include_direct_qwen")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    return parser.parse_args()


def _score(answer: str, sample) -> dict[str, float | None]:
    reference = sample.short_answer or sample.evidence_description
    if not reference:
        return {"exact_match": None, "substring_match": None, "token_f1": None, "char_f1": None}
    return {
        "exact_match": exact_match(answer, reference),
        "substring_match": substring_match(answer, reference),
        "token_f1": token_f1(answer, reference),
        "char_f1": char_f1(answer, reference),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = make_output_dir(args.output_dir, prefix="end2end", overwrite=args.overwrite_output_dir)
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

    prepared_items = [
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
            desc="end2end: build FVT",
            total=len(samples),
            enabled=args.progress,
            progress_file=args.progress_file,
        )
    ]
    groups = group_indices_by_image(prepared_items)
    rows = []
    with torch.no_grad():
        for index, item in progress_iter(
            enumerate(prepared_items),
            desc="end2end: run conditions",
            total=len(prepared_items),
            enabled=args.progress,
            progress_file=args.progress_file,
        ):
            sample = item.sample
            outputs = {}
            if args.include_direct_qwen:
                direct = generate_direct_qwen_answer(
                    model=model,
                    processor=processor,
                    image=sample.image,
                    question=sample.question,
                    device=device,
                    max_new_tokens=args.answer_max_new_tokens,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )
                outputs["direct_qwen"] = {"answer": direct, "scores": _score(direct, sample)}

            if args.mode == "free":
                result = run_tgvf_inference(
                    model=model,
                    processor=processor,
                    foveal_module=foveal_module,
                    image=sample.image,
                    question=sample.question,
                    device=device,
                    capture_max_new_tokens=args.capture_max_new_tokens,
                    answer_max_new_tokens=args.answer_max_new_tokens,
                    hidden_state_index=args.capture_layer,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )
                outputs["correct_TGVF_D"] = {
                    "answer": result.answer,
                    "scores": _score(result.answer, sample),
                    "generated_target": result.target_text,
                    "foveation_request": result.foveation_request,
                    "stop_reason": result.stop_reason,
                    "D_shape": None if result.fvt_output is None else list(result.fvt_output.foveated_visual_tokens.shape),
                    "second_full_forward_used": False,
                }
            else:
                same_index = same_image_wrong_index(groups, item, index)
                diff_index = different_image_index(prepared_items, index)
                random_d = random_d_like(
                    item.foveated_visual_tokens,
                    reference=item.merged_visual_tokens,
                    seed=args.seed + index,
                )
                condition_specs = {
                    "foveation_no_D": {
                        "condition": "no_D",
                        "replacement": None,
                        "replacement_image_grid_thw": None,
                    },
                    "random_D": {
                        "condition": "random_D",
                        "replacement": random_d,
                        "replacement_image_grid_thw": item.image_grid_thw,
                    },
                    "wrong_D": {
                        "condition": "wrong_D",
                        "replacement": (
                            prepared_items[same_index].foveated_visual_tokens
                            if same_index is not None
                            else (
                                prepared_items[diff_index].foveated_visual_tokens
                                if diff_index is not None
                                else random_d
                            )
                        ),
                        "replacement_image_grid_thw": (
                            prepared_items[same_index].image_grid_thw
                            if same_index is not None
                            else (
                                prepared_items[diff_index].image_grid_thw
                                if diff_index is not None
                                else item.image_grid_thw
                            )
                        ),
                    },
                    "correct_TGVF_D": {
                        "condition": "correct",
                        "replacement": None,
                        "replacement_image_grid_thw": None,
                    },
                }
                for name, spec in condition_specs.items():
                    result = run_forced_target_end2end(
                        model=model,
                        processor=processor,
                        foveal_module=foveal_module,
                        sample=sample,
                        device=device,
                        capture_layer=args.capture_layer,
                        answer_max_new_tokens=args.answer_max_new_tokens,
                        condition=spec["condition"],
                        replacement_d=spec["replacement"],
                        replacement_image_grid_thw=spec["replacement_image_grid_thw"],
                    )
                    outputs[name] = {**result, "scores": _score(result.get("answer", ""), sample)}

            row = {
                "uid": sample.uid,
                "stable_image_uid": sample.group_id,
                "image": sample.image,
                "question": sample.question,
                "dataset_target": sample.target,
                "evidence_description": sample.evidence_description,
                "short_answer": sample.short_answer,
                "evidence_type": sample.evidence_type,
                "answer_type": sample.answer_type,
                "source_profile": sample.source_profile,
                "outputs": outputs,
                "second_full_forward_used": False,
            }
            rows.append(row)

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
        "config": summarize_config(args, model_info),
        "num_samples_attempted": len(samples),
        "num_samples_evaluated": len(rows),
        "mode": args.mode,
        "metrics": report_metrics,
        "highlight": {
            "correct_D_token_f1": correct.get("token_f1"),
            "no_D_token_f1": no_d.get("token_f1"),
            "wrong_D_token_f1": wrong.get("token_f1"),
            "correct_D_improvement_over_no_D": (
                None
                if correct.get("token_f1") is None or no_d.get("token_f1") is None
                else correct["token_f1"] - no_d["token_f1"]
            ),
            "correct_D_improvement_over_wrong_D": (
                None
                if correct.get("token_f1") is None or wrong.get("token_f1") is None
                else correct["token_f1"] - wrong["token_f1"]
            ),
            "second_full_forward_used": False,
        },
    }
    write_json(output_dir / "end2end_eval_report.json", report)
    write_jsonl(output_dir / "per_sample_generations.jsonl", rows)
    save_summary(
        output_dir / "end2end_eval_summary.txt",
        "TGVF End-to-End Evaluation",
        [
            f"mode: {args.mode}",
            f"samples_evaluated: {len(rows)}",
            f"correct_D_token_f1: {report['highlight']['correct_D_token_f1']}",
            f"correct_D_improvement_over_no_D: {report['highlight']['correct_D_improvement_over_no_D']}",
            f"correct_D_improvement_over_wrong_D: {report['highlight']['correct_D_improvement_over_wrong_D']}",
            "second_full_forward_used: false",
        ],
    )


if __name__ == "__main__":
    main()
