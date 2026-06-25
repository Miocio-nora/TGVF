"""Clean benchmark runner CLI skeleton."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import replace

from revisit_vlm_clean.benchmark_data import (
    load_manifest_payload,
    materialize_samples_from_manifest_payload,
)
from revisit_vlm_clean.cli.common import exit_not_implemented, print_json
from revisit_vlm_clean.defaults import DEFAULT_BENCHMARK_ROOT
from revisit_vlm_clean.manifest import build_manifest, manifest_payload
from revisit_vlm_clean.outputs import (
    write_empty_benchmark_output,
    write_executed_benchmark_output,
    write_materialized_sample_output,
    write_rendered_input_output,
)
from revisit_vlm_clean.populations import get_population, get_subset
from revisit_vlm_clean.rendering import render_benchmark_inputs
from revisit_vlm_clean.runner import (
    RUNNER_BACKENDS,
    STAGE2_LEGACY_BACKEND,
    STAGE2_NATIVE_BACKEND,
    BackendConfig,
    resolve_backend_name,
    run_benchmark_rows,
)
from revisit_vlm_clean.schema import (
    DeepStackScope,
    DeepStackState,
    EvalFamily,
    EvalMode,
    ForwardMode,
    ParserScorerIdentity,
    RunConfig,
    ScoringBackend,
)
from revisit_vlm_clean.stage2_runtime import (
    Stage2RuntimeConfig,
    checkpoint_identity,
    eval_jsonl_identity,
)
from revisit_vlm_clean.tgvf_protocol import SUPPORTED_PROTOCOLS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean TGVF external benchmark runner skeleton.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Thinking")
    parser.add_argument("--processor-id", default=None)
    parser.add_argument("--mode", choices=[item.value for item in EvalMode], required=True)
    parser.add_argument(
        "--post-tgvf-forward-mode",
        choices=[item.value for item in ForwardMode],
        required=True,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--population-id")
    group.add_argument("--subset-id")
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--manifest-hash", default=None)
    parser.add_argument("--benchmark-root", default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-image-resolution", type=int, default=512)
    parser.add_argument("--max-action-tokens", type=int, default=64)
    parser.add_argument("--max-answer-tokens", type=int, default=128)
    parser.add_argument(
        "--tgvf-protocol",
        choices=SUPPORTED_PROTOCOLS,
        default="protocol_c_tool_observation",
    )
    parser.add_argument(
        "--scoring-backend",
        choices=[item.value for item in ScoringBackend],
        default=ScoringBackend.AUTO.value,
    )
    parser.add_argument("--softforce-prompt-text", default="")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute a clean runner backend and write scored rows.",
    )
    parser.add_argument(
        "--runner-backend",
        choices=RUNNER_BACKENDS,
        default="dry_run",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stage2-checkpoint", default="")
    parser.add_argument("--stage2-eval-jsonl", default="")
    parser.add_argument("--stage2-d-condition", default="correct_D")
    parser.add_argument("--force-prefix-mode", default="target_hint")
    parser.add_argument(
        "--validate-stage2-runtime",
        action="store_true",
        help="Validate and print Stage2 checkpoint/eval_jsonl identity without running inference.",
    )
    parser.add_argument("--deepstack-enabled", action="store_true")
    parser.add_argument(
        "--deepstack-original-image-scope",
        choices=[item.value for item in DeepStackScope],
        default=DeepStackScope.OFF.value,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved run_config.json and exit.",
    )
    parser.add_argument(
        "--write-empty-output",
        action="store_true",
        help="Write run_config/rows/summary/sample_manifest schema files without model inference.",
    )
    parser.add_argument(
        "--materialize-samples",
        action="store_true",
        help=(
            "Write materialized benchmark sample rows from a clean manifest "
            "without model inference."
        ),
    )
    parser.add_argument(
        "--render-inputs",
        action="store_true",
        help=(
            "Write rendered prompt/media/control rows from a clean manifest "
            "without model inference."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.population_id:
        get_population(args.population_id)
    if args.subset_id:
        get_subset(args.subset_id)
    git_commit, dirty_worktree = _git_identity()
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
        benchmark_root=args.benchmark_root,
        max_image_resolution=args.max_image_resolution,
        max_action_tokens=args.max_action_tokens,
        max_answer_tokens=args.max_answer_tokens,
        tgvf_protocol=args.tgvf_protocol,
        post_tgvf_forward_mode=ForwardMode(args.post_tgvf_forward_mode),
        softforce_prompt_text=args.softforce_prompt_text,
        deepstack=DeepStackState(
            enabled=bool(args.deepstack_enabled),
            original_image_scope=DeepStackScope(args.deepstack_original_image_scope),
        ),
        parser_scorer=ParserScorerIdentity(
            scoring_backend=ScoringBackend(args.scoring_backend),
            fallback_allowed=ScoringBackend(args.scoring_backend) == ScoringBackend.AUTO,
        ),
        git_commit=git_commit,
        dirty_worktree=dirty_worktree,
    )
    config.validate()
    if args.dry_run:
        print_json(config)
        return 0
    if args.validate_stage2_runtime:
        stage2_config = _stage2_runtime_config(args, config)
        stage2_config.validate()
        print_json(
            {
                "stage2_runtime": stage2_config.to_dict(),
                "checkpoint": checkpoint_identity(stage2_config.stage2_checkpoint),
                "eval_jsonl": eval_jsonl_identity(stage2_config.eval_jsonl),
            }
        )
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
    if args.execute:
        if not args.output_dir:
            raise ValueError("--execute requires --output-dir")
        resolved_manifest = _resolve_manifest_payload(args)
        runtime_config = replace(
            config,
            manifest_hash=config.manifest_hash or resolved_manifest.get("manifest_hash"),
        )
        resolved_backend = resolve_backend_name(args.runner_backend)
        samples = materialize_samples_from_manifest_payload(
            resolved_manifest,
            benchmark_root=args.benchmark_root,
            metadata_only=args.runner_backend == "dry_run",
        )
        rendered_inputs = render_benchmark_inputs(samples, runtime_config)
        backend_config = BackendConfig(
            backend=args.runner_backend,
            dtype=args.dtype,
            device=args.device,
            device_map=None if args.device_map in {"", "none", "None", "null"} else args.device_map,
            attn_implementation=(
                None
                if args.attn_implementation in {"", "none", "None", "null"}
                else args.attn_implementation
            ),
            trust_remote_code=bool(args.trust_remote_code),
            stage2=(
                _stage2_runtime_config(args, runtime_config)
                if resolved_backend in {STAGE2_NATIVE_BACKEND, STAGE2_LEGACY_BACKEND}
                else None
            ),
        )
        rows, summary = run_benchmark_rows(
            samples,
            rendered_inputs,
            config=runtime_config,
            backend_config=backend_config,
        )
        print_json(
            write_executed_benchmark_output(
                args.output_dir,
                config=runtime_config,
                manifest=resolved_manifest,
                rows=rows,
                summary=summary,
                backend_config=backend_config,
            )
        )
        return 0
    return exit_not_implemented(
        "benchmark execution is phase 4; phase 1 only validates run identity"
    )


def _resolve_manifest_payload(args: argparse.Namespace) -> dict:
    if args.manifest_path:
        resolved_manifest = load_manifest_payload(args.manifest_path)
    elif args.subset_id:
        resolved_manifest = manifest_payload(
            build_manifest(subset_id=args.subset_id, benchmark_root=args.benchmark_root)
        )
    else:
        raise ValueError(
            "manifest-backed smoke output with --population-id requires --manifest-path"
        )
    _validate_manifest_identity(args, resolved_manifest)
    if args.manifest_hash and resolved_manifest.get("manifest_hash") != args.manifest_hash:
        raise ValueError(
            f"manifest hash mismatch: expected {args.manifest_hash}, "
            f"got {resolved_manifest.get('manifest_hash')}"
        )
    return resolved_manifest


def _validate_manifest_identity(args: argparse.Namespace, resolved_manifest: dict) -> None:
    if args.subset_id:
        manifest_id = resolved_manifest.get("manifest_id")
        if manifest_id != args.subset_id:
            raise ValueError(
                f"manifest id mismatch: --subset-id {args.subset_id} "
                f"but manifest_id is {manifest_id}"
            )
        return

    if not args.population_id:
        return

    populations = set(resolved_manifest.get("source_population_ids") or [])
    if not populations:
        populations = {
            sample.get("population_id")
            for sample in resolved_manifest.get("samples", [])
            if sample.get("population_id")
        }
    if populations != {args.population_id}:
        raise ValueError(
            f"manifest population mismatch: --population-id {args.population_id} "
            f"but manifest populations are {sorted(populations)}"
        )


def _stage2_runtime_config(args: argparse.Namespace, config: RunConfig) -> Stage2RuntimeConfig:
    if not args.stage2_checkpoint:
        raise ValueError("--stage2-checkpoint is required for Stage2 runtime validation")
    if not args.stage2_eval_jsonl:
        raise ValueError("--stage2-eval-jsonl is required for Stage2 runtime validation")
    return Stage2RuntimeConfig(
        stage2_checkpoint=args.stage2_checkpoint,
        eval_jsonl=args.stage2_eval_jsonl,
        protocol=config.tgvf_protocol,
        d_condition=args.stage2_d_condition,
        force_prefix_mode=args.force_prefix_mode,
        append_forward_mode=config.post_tgvf_forward_mode,
        max_action_tokens=config.max_action_tokens,
        max_answer_tokens=config.max_answer_tokens,
    )


def _git_identity() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None, None
    return commit or None, bool(status.strip())


if __name__ == "__main__":
    raise SystemExit(main())
