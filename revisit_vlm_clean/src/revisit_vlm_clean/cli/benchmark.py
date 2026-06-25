"""Clean benchmark runner CLI skeleton."""

from __future__ import annotations

import argparse

from revisit_vlm_clean.benchmark_data import (
    load_manifest_payload,
    materialize_samples_from_manifest_payload,
)
from revisit_vlm_clean.cli.common import exit_not_implemented, print_json
from revisit_vlm_clean.manifest import build_manifest, manifest_payload
from revisit_vlm_clean.outputs import (
    write_empty_benchmark_output,
    write_materialized_sample_output,
    write_rendered_input_output,
)
from revisit_vlm_clean.populations import get_population, get_subset
from revisit_vlm_clean.rendering import render_benchmark_inputs
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
    parser.add_argument(
        "--materialize-samples",
        action="store_true",
        help="Write materialized benchmark sample rows from a clean manifest without model inference.",
    )
    parser.add_argument(
        "--render-inputs",
        action="store_true",
        help="Write rendered prompt/media/control rows from a clean manifest without model inference.",
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
    if args.materialize_samples:
        if not args.output_dir:
            raise ValueError("--materialize-samples requires --output-dir")
        resolved_manifest = _resolve_manifest_payload(args)
        samples = materialize_samples_from_manifest_payload(
            resolved_manifest,
            benchmark_root=args.benchmark_root,
            metadata_only=True,
        )
        print_json(
            write_materialized_sample_output(
                args.output_dir,
                config=config,
                manifest=resolved_manifest,
                samples=samples,
            )
        )
        return 0
    if args.render_inputs:
        if not args.output_dir:
            raise ValueError("--render-inputs requires --output-dir")
        resolved_manifest = _resolve_manifest_payload(args)
        samples = materialize_samples_from_manifest_payload(
            resolved_manifest,
            benchmark_root=args.benchmark_root,
            metadata_only=True,
        )
        rendered_inputs = render_benchmark_inputs(samples, config)
        print_json(
            write_rendered_input_output(
                args.output_dir,
                config=config,
                manifest=resolved_manifest,
                rendered_inputs=rendered_inputs,
            )
        )
        return 0
    return exit_not_implemented("benchmark execution is phase 4; phase 1 only validates run identity")


def _resolve_manifest_payload(args: argparse.Namespace) -> dict:
    if args.manifest_path:
        resolved_manifest = load_manifest_payload(args.manifest_path)
    elif args.subset_id:
        resolved_manifest = manifest_payload(
            build_manifest(subset_id=args.subset_id, benchmark_root=args.benchmark_root)
        )
    else:
        raise ValueError("manifest-backed smoke output with --population-id requires --manifest-path")
    if args.manifest_hash and resolved_manifest.get("manifest_hash") != args.manifest_hash:
        raise ValueError(
            f"manifest hash mismatch: expected {args.manifest_hash}, "
            f"got {resolved_manifest.get('manifest_hash')}"
        )
    return resolved_manifest


if __name__ == "__main__":
    raise SystemExit(main())
