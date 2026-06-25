"""Clean benchmark runner CLI skeleton."""

from __future__ import annotations

import argparse

from revisit_vlm_clean.cli.common import exit_not_implemented, print_json
from revisit_vlm_clean.manifest import build_manifest
from revisit_vlm_clean.outputs import write_empty_benchmark_output
from revisit_vlm_clean.populations import get_population, get_subset
from revisit_vlm_clean.schema import (
    DeepStackScope,
    DeepStackState,
    EvalFamily,
    EvalMode,
    ForwardMode,
    RunConfig,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean TGVF external benchmark runner skeleton.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Thinking")
    parser.add_argument("--processor-id", default=None)
    parser.add_argument("--mode", choices=[item.value for item in EvalMode], required=True)
    parser.add_argument("--post-tgvf-forward-mode", choices=[item.value for item in ForwardMode], required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--population-id")
    group.add_argument("--subset-id")
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--manifest-hash", default=None)
    parser.add_argument("--benchmark-root", default="/home/dredvpn009/Flash_Storage/datasets/benchmarks")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-image-resolution", type=int, default=512)
    parser.add_argument("--max-action-tokens", type=int, default=64)
    parser.add_argument("--max-answer-tokens", type=int, default=128)
    parser.add_argument("--softforce-prompt-text", default="")
    parser.add_argument("--deepstack-enabled", action="store_true")
    parser.add_argument(
        "--deepstack-original-image-scope",
        choices=[item.value for item in DeepStackScope],
        default=DeepStackScope.OFF.value,
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved run_config.json and exit.")
    parser.add_argument(
        "--write-empty-output",
        action="store_true",
        help="Write run_config/rows/summary/sample_manifest schema files without model inference.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.population_id:
        get_population(args.population_id)
    if args.subset_id:
        get_subset(args.subset_id)
    config = RunConfig(
        run_id=args.run_id,
        checkpoint_path=args.checkpoint_path,
        model_id=args.model_id,
        processor_id=args.processor_id,
        eval_family=EvalFamily.PROJECT_NATIVE_EXTERNAL,
        mode=EvalMode(args.mode),
        population_id=args.population_id,
        subset_id=args.subset_id,
        manifest_path=args.manifest_path,
        manifest_hash=args.manifest_hash,
        max_image_resolution=args.max_image_resolution,
        max_action_tokens=args.max_action_tokens,
        max_answer_tokens=args.max_answer_tokens,
        post_tgvf_forward_mode=ForwardMode(args.post_tgvf_forward_mode),
        softforce_prompt_text=args.softforce_prompt_text,
        deepstack=DeepStackState(
            enabled=bool(args.deepstack_enabled),
            original_image_scope=DeepStackScope(args.deepstack_original_image_scope),
        ),
    )
    config.validate()
    if args.dry_run:
        print_json(config)
        return 0
    if args.write_empty_output:
        if not args.output_dir:
            raise ValueError("--write-empty-output requires --output-dir")
        manifest = None
        if args.subset_id and not args.manifest_path:
            manifest = build_manifest(subset_id=args.subset_id, benchmark_root=args.benchmark_root)
        print_json(write_empty_benchmark_output(args.output_dir, config=config, manifest=manifest))
        return 0
    return exit_not_implemented("benchmark execution is phase 4; phase 1 only validates run identity")


if __name__ == "__main__":
    raise SystemExit(main())
