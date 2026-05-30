from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

from tgvf_eval.adapters import BenchmarkRegistry
from tgvf_eval.config import (
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEED,
    DEFAULT_TOOLS_ROOT,
    method_config_from_name,
    tier_video_nframes,
)
from tgvf_eval.model_runner import DryRunModelRunner, QwenTGVFModelRunner
from tgvf_eval.progress import configure_quiet_external_progress, iter_progress
from tgvf_eval.results import ResultWriter, make_run_paths, summarize_rows, write_config_yaml


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one TGVF benchmark evaluation.")
    parser.add_argument("--benchmark", required=True, choices=BenchmarkRegistry.names())
    parser.add_argument("--tier", choices=["light", "medium", "full"], default="light")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--method", default="direct_qwen")
    parser.add_argument("--trigger-mode", choices=["direct", "force", "free"], default=None)
    parser.add_argument("--tgvf-mode", choices=["none", "prompt_only", "module"], default=None)
    parser.add_argument("--cot", type=_bool, default=False)
    parser.add_argument(
        "--reasoning-mode",
        choices=["final_only", "internal_cot", "visible_tool_cot"],
        default="final_only",
    )
    parser.add_argument("--max-foveations", type=int, default=1)
    parser.add_argument("--image-budget", choices=["low", "mid", "high"], default="mid")
    parser.add_argument("--video-nframes", type=int, default=None)
    parser.add_argument(
        "--video-foveation-mode",
        choices=["off", "per_frame_reencode", "nextframe_encode_no_reencode"],
        default="off",
    )
    parser.add_argument("--nextframe-reuse-ttl", type=int, default=1)
    parser.add_argument("--model-path", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--processor-path", default=None)
    parser.add_argument("--tgvf-checkpoint", default=None)
    parser.add_argument("--tgvf-variant", default="target_slot_foveal_cross_merger")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--answer-max-new-tokens", type=int, default=128)
    parser.add_argument("--capture-max-new-tokens", type=int, default=128)
    parser.add_argument("--num-foveated-tokens", type=_optional_positive_int, default=16)
    parser.add_argument("--spatial-merge-size", default="auto")
    parser.add_argument("--benchmark-root", default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--tools-root", default=DEFAULT_TOOLS_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--no-progress", action="store_false", dest="progress")
    parser.add_argument("--repeat-options-in-continuation", type=_bool, default=False)
    parser.add_argument("--force-target-mode", choices=["generated", "fixed"], default="generated")
    parser.add_argument("--fixed-force-target", default=None)
    parser.add_argument(
        "--force-target-source",
        choices=["rule", "llm", "llm_with_rule_fallback"],
        default=None,
        help="Source for force-mode target text. Defaults to rule for VSTAR force runs, otherwise llm.",
    )
    parser.add_argument("--dump-target-quality-report", type=_bool, default=True)
    parser.add_argument("--dump-answer-distribution", type=_bool, default=True)
    parser.add_argument("--force-ablation-mode", default=None)
    parser.add_argument("--suppress-im-end-first-token-after-fvt", action="store_true")
    parser.add_argument(
        "--fvt-append-mode",
        choices=["legacy_text_positions", "qwen_native_pseudo_image"],
        default="qwen_native_pseudo_image",
    )
    parser.add_argument(
        "--fresh-answer-context",
        choices=["direct", "focused_target"],
        default="direct",
        help=(
            "Answer prompt for conditioned_D_fresh. focused_target prepends "
            "'The visual tokens above are focused evidence for the target: ...'."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_benchmark(args)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    configure_quiet_external_progress()
    video_nframes = args.video_nframes
    if video_nframes is None and args.benchmark in {"ovo_bench"}:
        video_nframes = tier_video_nframes(args.tier)
    method_config = method_config_from_name(
        args.method,
        trigger_mode=args.trigger_mode,
        tgvf_mode=args.tgvf_mode,
        cot_enabled=args.cot,
        reasoning_mode=args.reasoning_mode,
        max_foveations=args.max_foveations,
        image_budget=args.image_budget,
        video_nframes=video_nframes,
        video_foveation_mode=args.video_foveation_mode,
        nextframe_reuse_ttl=args.nextframe_reuse_ttl,
        fvt_append_mode=args.fvt_append_mode,
    )
    if method_config.video_foveation_mode != "off" and not args.dry_run:
        raise NotImplementedError(
            "Real video foveation modes are not wired into QwenTGVFModelRunner yet. "
            "Use --dry-run for state-machine/progress checks, or --video-foveation-mode off "
            "for current real model benchmark runs."
        )
    force_target_source = args.force_target_source or (
        "rule"
        if args.benchmark == "vstar_bench" and method_config.trigger_mode == "force"
        else "llm"
    )
    run_id = args.run_id or datetime.now(timezone.utc).strftime("eval_%Y%m%d_%H%M%S")
    paths = make_run_paths(args.output_root, run_id=run_id)
    config_payload = {
        "run_id": run_id,
        "benchmark_root": args.benchmark_root,
        "tools_root": args.tools_root,
        "output_root": args.output_root,
        "model": {
            "qwen_model_path": args.model_path,
            "processor_path": args.processor_path,
            "dtype": args.dtype,
            "device": args.device,
            "temperature": 0.0,
            "max_new_tokens": args.answer_max_new_tokens,
        },
        "tgvf": {
            "mode": method_config.tgvf_mode,
            "checkpoint": args.tgvf_checkpoint,
            "variant": args.tgvf_variant,
            "trigger_mode": method_config.trigger_mode,
            "max_foveations": method_config.max_foveations,
            "num_foveated_tokens": args.num_foveated_tokens,
            "spatial_merge_size": args.spatial_merge_size,
            "prompt_version": "benchmark_v0",
            "continuation_instruction": method_config.continuation_instruction,
            "second_full_forward_allowed": method_config.second_full_forward_allowed,
            "repeat_options_in_continuation": args.repeat_options_in_continuation,
            "force_target_mode": args.force_target_mode,
            "fixed_force_target": args.fixed_force_target,
            "force_target_source": force_target_source,
            "suppress_im_end_first_token_after_fvt": args.suppress_im_end_first_token_after_fvt,
            "fvt_append_mode": args.fvt_append_mode,
            "fresh_answer_context": args.fresh_answer_context,
        },
        "cot": {
            "enabled": method_config.cot_enabled,
            "reasoning_mode": method_config.reasoning_mode,
        },
        "image": {"budget": method_config.image_budget},
        "video": {
            "nframes": method_config.video_nframes,
            "foveation_mode": method_config.video_foveation_mode,
            "nextframe_reuse_ttl": method_config.nextframe_reuse_ttl,
        },
        "eval": {
            "benchmark": args.benchmark,
            "tier": args.tier,
            "seed": args.seed,
            "official_tools": True,
            "resume": not args.no_resume,
        },
    }
    write_config_yaml(paths.root / "config.yaml", config_payload)

    adapter = BenchmarkRegistry.get(
        args.benchmark,
        benchmark_root=args.benchmark_root,
        tools_root=args.tools_root,
    )
    samples = adapter.sample(tier=args.tier, limit=args.limit, seed=args.seed)
    writer = ResultWriter(paths, benchmark=args.benchmark, method=args.method)
    completed = set() if args.no_resume or args.overwrite else writer.completed_ids()
    runner = (
        DryRunModelRunner()
        if args.dry_run
        else QwenTGVFModelRunner(
            model_path=args.model_path,
            processor_path=args.processor_path,
            tgvf_checkpoint=args.tgvf_checkpoint,
            tgvf_variant=args.tgvf_variant,
            dtype=args.dtype,
            device=args.device,
            answer_max_new_tokens=args.answer_max_new_tokens,
            capture_max_new_tokens=args.capture_max_new_tokens,
            num_foveated_tokens=args.num_foveated_tokens,
            spatial_merge_size=args.spatial_merge_size,
            repeat_options_in_continuation=args.repeat_options_in_continuation,
            force_target_mode=args.force_target_mode,
            fixed_force_target=args.fixed_force_target,
            force_target_source=force_target_source,
            suppress_im_end_first_token_after_fvt=args.suppress_im_end_first_token_after_fvt,
            fvt_append_mode=args.fvt_append_mode,
            fresh_answer_context=args.fresh_answer_context,
        )
    )

    pending_samples = [sample for sample in samples if sample.sample_id not in completed]
    if pending_samples:
        runner.prepare(method_config)
    desc = f"{args.benchmark} {args.method}"
    for sample in iter_progress(
        pending_samples,
        desc=desc,
        total=len(pending_samples),
        enabled=getattr(args, "progress", True),
    ):
        result = runner.run(sample, method_config)
        parsed = adapter.parse_prediction(result.raw_output, sample) if not result.error else ""
        score = None
        if sample.gold_answer is not None and not result.error:
            score = adapter.score_predictions(
                [
                    {
                        "parsed_answer": parsed,
                        "gold_answer": sample.gold_answer,
                        "choices": sample.choices,
                        "metadata": sample.metadata,
                    }
                ]
            ).get("score")
        row = {
            "benchmark": args.benchmark,
            "sample_id": sample.sample_id,
            "method": method_config.method,
            "trigger_mode": method_config.trigger_mode,
            "tgvf_mode": method_config.tgvf_mode,
            "cot_enabled": method_config.cot_enabled,
            "cot_prompt_source": method_config.cot_prompt_source,
            "reasoning_mode": method_config.reasoning_mode,
            "image_budget": method_config.image_budget,
            "video_nframes": method_config.video_nframes,
            "video_foveation_mode": method_config.video_foveation_mode,
            "nextframe_reuse_ttl": method_config.nextframe_reuse_ttl,
            "question": sample.question,
            "media": sample.media,
            "choices": sample.choices,
            "raw_output": result.raw_output,
            "parsed_answer": parsed,
            "gold_answer": sample.gold_answer,
            "score": score,
            "metadata": sample.metadata,
            "triggered": result.triggered,
            "foveation_target": result.foveation_target,
            "num_foveations": result.num_foveations,
            "second_full_forward_used": result.second_full_forward_used,
            "wall_time_sec": result.wall_time_sec,
            "visual_token_count": result.visual_token_count,
            "output_tokens": result.output_tokens,
            "debug_metadata": result.debug or {},
            "official_tool_used": adapter.tool_info.official_tool_used,
            "official_tool_path": adapter.tool_info.official_tool_path,
            "scorer_name": adapter.tool_info.scorer_name,
            "prompt_source": adapter.tool_info.prompt_source,
            "error": result.error,
        }
        writer.append_row(row)
        if result.error and args.fail_fast:
            raise RuntimeError(result.error)

    rows = writer.read_rows()
    scorer_payload = adapter.score_predictions(rows)
    summary = summarize_rows(
        rows,
        run_id=run_id,
        benchmark=args.benchmark,
        tier=args.tier,
        method=args.method,
        cot_enabled=method_config.cot_enabled,
        scorer_payload=scorer_payload,
    )
    writer.write_summary(summary)
    (paths.scores / f"{args.benchmark}__{args.method}.scores.json").write_text(
        __import__("json").dumps(scorer_payload, indent=2, ensure_ascii=False) + "\n"
    )
    return summary


def _bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _optional_positive_int(value: str) -> int | None:
    if value.lower() in {"none", "null"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer or 'none'")
    return parsed


if __name__ == "__main__":
    main()
