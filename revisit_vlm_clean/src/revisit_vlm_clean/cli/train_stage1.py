"""Clean Stage1 training launch planner."""

from __future__ import annotations

import argparse
import subprocess

from revisit_vlm_clean.cli.common import exit_not_implemented, print_json
from revisit_vlm_clean.defaults import (
    DEFAULT_MAX_IMAGE_RESOLUTION,
    DEFAULT_MODEL_ID,
    DEFAULT_PROTOCOL,
    DEFAULT_STAGE1_GLOBAL_BATCH,
    DEFAULT_STAGE1_MAX_STEPS,
)
from revisit_vlm_clean.tgvf_protocol import SUPPORTED_PROTOCOLS
from revisit_vlm_clean.training_plan import (
    Stage1LaunchConfig,
    build_stage1_launch_plan,
    resolve_batch_identity,
    write_training_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean TGVF Stage1 training launch planner.")
    parser.add_argument("--print-defaults", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--train-file")
    parser.add_argument("--output-dir")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--processor-id", default=None)
    parser.add_argument("--protocol", choices=SUPPORTED_PROTOCOLS, default=DEFAULT_PROTOCOL)
    parser.add_argument("--max-image-resolution", type=int, default=DEFAULT_MAX_IMAGE_RESOLUTION)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_STAGE1_MAX_STEPS)
    parser.add_argument("--save-every", type=int, default=DEFAULT_STAGE1_MAX_STEPS)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--variant", default="tgvf_v2_bidirectional")
    parser.add_argument("--token-row-mode", choices=("row_only",), default="row_only")
    parser.add_argument("--capture-mode", choices=("teacher_forced",), default="teacher_forced")
    parser.add_argument(
        "--fvt-position-mode",
        choices=("native_source_grid",),
        default="native_source_grid",
    )
    parser.add_argument(
        "--focus-action-im-end",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--mask-original-image-after-tgvf",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--lr-scheduler",
        choices=("constant", "linear", "cosine"),
        default="cosine",
    )
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--loss-gen", type=float, default=1.0)
    parser.add_argument("--loss-visual-token-manifold", type=float, default=0.1)
    parser.add_argument("--loss-same-image-negative", type=float, default=1.0)
    parser.add_argument("--same-image-negative-margin", type=float, default=1.0)
    parser.add_argument(
        "--same-image-negative-mode",
        choices=("matrix_ce", "cyclic_margin"),
        default="matrix_ce",
    )
    parser.add_argument("--readout-batch-size", type=int, default=4)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--global-batch", type=int, default=DEFAULT_STAGE1_GLOBAL_BATCH)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved launch plan.")
    parser.add_argument("--write-plan", action="store_true", help="Write launch plan artifacts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.print_defaults:
        print_json(_defaults())
        return 0
    if not (args.dry_run or args.write_plan):
        return exit_not_implemented("use --dry-run or --write-plan to produce a Stage1 launch plan")
    config = _config_from_args(args)
    git_commit, dirty_worktree = _git_identity()
    plan = build_stage1_launch_plan(
        config,
        git_commit=git_commit,
        dirty_worktree=dirty_worktree,
    )
    if args.dry_run:
        print_json(plan)
        return 0
    print_json(write_training_plan(args.output_dir, plan))
    return 0


def _defaults() -> dict[str, object]:
    batch = resolve_batch_identity(
        global_batch_size=DEFAULT_STAGE1_GLOBAL_BATCH,
        world_size=1,
        micro_batch_size=1,
        gradient_accumulation_steps=None,
    )
    return {
        "model_id": DEFAULT_MODEL_ID,
        "protocol": DEFAULT_PROTOCOL,
        "variant": "tgvf_v2_bidirectional",
        "token_row_mode": "row_only",
        "capture_mode": "teacher_forced",
        "fvt_position_mode": "native_source_grid",
        "max_image_resolution": DEFAULT_MAX_IMAGE_RESOLUTION,
        "batch": batch.to_dict(),
        "max_steps": DEFAULT_STAGE1_MAX_STEPS,
        "focus_action_im_end": True,
        "same_image_negative": "matrix_ce",
        "same_image_negative_margin": 1.0,
        "readout_batch_size": 4,
        "visual_token_manifold_loss": 0.1,
        "lr_scheduler": "cosine",
        "warmup_steps": 100,
        "min_lr_ratio": 0.1,
        "max_grad_norm": 1.0,
        "clean_launcher_actions": ["dry_run", "write_plan"],
    }


def _config_from_args(args: argparse.Namespace) -> Stage1LaunchConfig:
    if not args.run_id:
        raise ValueError("--run-id is required")
    if not args.train_file:
        raise ValueError("--train-file is required")
    if not args.output_dir:
        raise ValueError("--output-dir is required")
    batch = resolve_batch_identity(
        global_batch_size=args.global_batch,
        world_size=args.world_size,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    return Stage1LaunchConfig(
        run_id=args.run_id,
        train_file=args.train_file,
        output_dir=args.output_dir,
        model_id=args.model_id,
        processor_id=args.processor_id,
        protocol=args.protocol,
        max_image_resolution=args.max_image_resolution,
        max_steps=args.max_steps,
        save_every=args.save_every,
        seed=args.seed,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        variant=args.variant,
        token_row_mode=args.token_row_mode,
        capture_mode=args.capture_mode,
        fvt_position_mode=args.fvt_position_mode,
        focus_action_im_end=args.focus_action_im_end,
        mask_original_image_after_tgvf=args.mask_original_image_after_tgvf,
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        max_grad_norm=args.max_grad_norm,
        loss_gen=args.loss_gen,
        loss_visual_token_manifold=args.loss_visual_token_manifold,
        loss_same_image_negative=args.loss_same_image_negative,
        same_image_negative_margin=args.same_image_negative_margin,
        same_image_negative_mode=args.same_image_negative_mode,
        readout_batch_size=args.readout_batch_size,
        min_confidence=args.min_confidence,
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
        batch=batch,
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
