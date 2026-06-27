"""CLI for clean Stage1-style TGVF internal diagnostics."""

from __future__ import annotations

import argparse
import subprocess

from revisit_vlm_clean.cli.common import exit_not_implemented, print_json
from revisit_vlm_clean.defaults import DEFAULT_MAX_IMAGE_RESOLUTION, DEFAULT_MODEL_ID, DEFAULT_PROTOCOL
from revisit_vlm_clean.stage_diagnostics import (
    DIAGNOSTIC_TASKS,
    StageDiagnosticConfig,
    build_stage_diagnostic_plan,
    parse_diagnostic_tasks,
    run_stage_diagnostics,
    write_stage_diagnostic_plan,
)
from revisit_vlm_clean.tgvf_protocol import SUPPORTED_PROTOCOLS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean Stage1-style TGVF readout/query/distribution diagnostics."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--eval-jsonl",
        required=True,
        help="Stage1/FVT focus eval JSONL used by readout/query/distribution diagnostics.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--processor-id", default=None)
    parser.add_argument("--protocol", choices=SUPPORTED_PROTOCOLS, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--focus-action-im-end",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--variant", default="tgvf_v2_bidirectional")
    parser.add_argument("--num-foveated-tokens", default="none")
    parser.add_argument("--encoder-adapter-type", default="bidirectional")
    parser.add_argument("--max-image-resolution", type=int, default=DEFAULT_MAX_IMAGE_RESOLUTION)
    parser.add_argument(
        "--fvt-position-mode",
        choices=("native_source_grid",),
        default="native_source_grid",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument(
        "--tasks",
        default="all",
        help=f"'all' or comma-separated subset of {','.join(DIAGNOSTIC_TASKS)}.",
    )
    parser.add_argument("--readout-max-samples", type=int, default=200)
    parser.add_argument("--distribution-max-samples", type=int, default=200)
    parser.add_argument("--query-max-groups", type=int, default=50)
    parser.add_argument("--query-require-groups", type=int, default=0)
    parser.add_argument("--query-min-targets-per-image", type=int, default=3)
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--use-fvt-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--stage2-load-lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Load qwen_lora from Stage2 checkpoints for diagnostics. "
            "Use --no-stage2-load-lora for legacy-comparable TGVF-module readout."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--wandb-log-eval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-plan", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not (args.dry_run or args.write_plan or args.execute):
        return exit_not_implemented("use --dry-run, --write-plan, or --execute")
    config = _config_from_args(args)
    plan = build_stage_diagnostic_plan(config)
    if args.dry_run:
        print_json(plan)
        return 0
    artifacts = write_stage_diagnostic_plan(args.output_dir, plan)
    if args.write_plan and not args.execute:
        print_json(artifacts)
        return 0
    result = run_stage_diagnostics(plan)
    print_json({"artifacts": artifacts, "result": result})
    return 0


def _config_from_args(args: argparse.Namespace) -> StageDiagnosticConfig:
    git_commit, dirty_worktree = _git_identity()
    return StageDiagnosticConfig(
        run_id=args.run_id,
        stage=args.stage,
        checkpoint=args.checkpoint,
        eval_jsonl=args.eval_jsonl,
        output_dir=args.output_dir,
        model_id=args.model_id,
        processor_id=args.processor_id,
        protocol=args.protocol,
        focus_action_im_end=args.focus_action_im_end,
        variant=args.variant,
        num_foveated_tokens=args.num_foveated_tokens,
        encoder_adapter_type=args.encoder_adapter_type,
        max_image_resolution=args.max_image_resolution,
        fvt_position_mode=args.fvt_position_mode,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        device=args.device,
        device_map=args.device_map,
        tasks=parse_diagnostic_tasks(args.tasks),
        readout_max_samples=args.readout_max_samples,
        distribution_max_samples=args.distribution_max_samples,
        query_max_groups=args.query_max_groups,
        query_require_groups=args.query_require_groups,
        query_min_targets_per_image=args.query_min_targets_per_image,
        eval_workers=args.eval_workers,
        use_fvt_cache=args.use_fvt_cache,
        stage2_load_lora=args.stage2_load_lora,
        seed=args.seed,
        wandb_log_eval=args.wandb_log_eval,
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
        wandb_run_name=args.wandb_run_name,
        wandb_group=args.wandb_group,
        wandb_tags=args.wandb_tags,
        git_commit=git_commit,
        dirty_worktree=dirty_worktree,
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
