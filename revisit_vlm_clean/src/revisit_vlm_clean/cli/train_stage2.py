"""Clean Stage2 training launch planner."""

from __future__ import annotations

import argparse
import subprocess

from revisit_vlm_clean.cli.common import exit_not_implemented, print_json
from revisit_vlm_clean.defaults import (
    DEFAULT_MAX_IMAGE_RESOLUTION,
    DEFAULT_MODEL_ID,
    DEFAULT_PROTOCOL,
    DEFAULT_STAGE2_GLOBAL_BATCH,
    DEFAULT_STAGE2_MAX_STEPS,
)
from revisit_vlm_clean.schema import DeepStackScope, DeepStackState
from revisit_vlm_clean.tgvf_protocol import SUPPORTED_PROTOCOLS
from revisit_vlm_clean.training_plan import (
    DEFAULT_STAGE2_LORA_TARGET_MODULES,
    DEFAULT_STAGE2_SPAN_WEIGHTS,
    OriginalImageMaskScope,
    Stage2LaunchConfig,
    build_stage2_launch_plan,
    resolve_batch_identity,
    write_training_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean TGVF Stage2 training launch planner.")
    parser.add_argument("--print-defaults", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--train-file")
    parser.add_argument("--val-file", default=None)
    parser.add_argument("--stage1-checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--processor-id", default=None)
    parser.add_argument("--protocol", choices=SUPPORTED_PROTOCOLS, default=DEFAULT_PROTOCOL)
    parser.add_argument("--max-image-resolution", type=int, default=DEFAULT_MAX_IMAGE_RESOLUTION)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_STAGE2_MAX_STEPS)
    parser.add_argument("--save-every", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--variant", default="tgvf_v2_bidirectional")
    parser.add_argument(
        "--use-stage1-tgvf-config",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fast-batched-stage2",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fvt-position-mode",
        choices=("native_source_grid",),
        default="native_source_grid",
    )
    parser.add_argument("--target-focus-ratio", type=float, default=0.8)
    parser.add_argument(
        "--mask-original-image-after-tgvf",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--mask-original-image-after-tgvf-prob", type=float, default=1.0)
    parser.add_argument(
        "--mask-original-image-after-tgvf-scope",
        choices=[item.value for item in OriginalImageMaskScope],
        default=OriginalImageMaskScope.THROUGH_ANSWER.value,
    )
    parser.add_argument("--deepstack-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--deepstack-original-image-scope",
        choices=[DeepStackScope.THROUGH_ANSWER.value, DeepStackScope.EVIDENCE_ONLY.value],
        default=None,
        help="Defaults to the mask scope when --deepstack-enabled is set.",
    )
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=256)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-bias", choices=("none", "all", "lora_only"), default="none")
    parser.add_argument(
        "--lora-target-modules",
        default=",".join(DEFAULT_STAGE2_LORA_TARGET_MODULES),
    )
    parser.add_argument("--lr-lora", type=float, default=2e-5)
    parser.add_argument("--lr-tgvf", type=float, default=5e-6)
    parser.add_argument("--lr-calibration", type=float, default=1e-5)
    parser.add_argument(
        "--lr-scheduler",
        choices=("constant", "linear", "cosine"),
        default="cosine",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--loss-visual-token-manifold", type=float, default=0.0)
    parser.add_argument(
        "--loss-evidence-state",
        type=float,
        default=DEFAULT_STAGE2_SPAN_WEIGHTS["evidence_state"],
    )
    parser.add_argument(
        "--loss-focus-target",
        type=float,
        default=DEFAULT_STAGE2_SPAN_WEIGHTS["focus_target"],
    )
    parser.add_argument(
        "--loss-evidence",
        type=float,
        default=DEFAULT_STAGE2_SPAN_WEIGHTS["evidence"],
    )
    parser.add_argument(
        "--loss-value-span",
        type=float,
        default=DEFAULT_STAGE2_SPAN_WEIGHTS["value_span"],
    )
    parser.add_argument("--loss-answer", type=float, default=DEFAULT_STAGE2_SPAN_WEIGHTS["answer"])
    parser.add_argument(
        "--loss-no-focus-evidence-state",
        type=float,
        default=DEFAULT_STAGE2_SPAN_WEIGHTS["no_focus_evidence_state"],
    )
    parser.add_argument(
        "--loss-no-focus-answer",
        type=float,
        default=DEFAULT_STAGE2_SPAN_WEIGHTS["no_focus_answer"],
    )
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--global-batch", type=int, default=DEFAULT_STAGE2_GLOBAL_BATCH)
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
        return exit_not_implemented("use --dry-run or --write-plan to produce a Stage2 launch plan")
    config = _config_from_args(args)
    git_commit, dirty_worktree = _git_identity()
    plan = build_stage2_launch_plan(
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
        global_batch_size=DEFAULT_STAGE2_GLOBAL_BATCH,
        world_size=1,
        micro_batch_size=1,
        gradient_accumulation_steps=None,
    )
    return {
        "model_id": DEFAULT_MODEL_ID,
        "protocol": DEFAULT_PROTOCOL,
        "stage2_path": "fast_batched",
        "target_focus_ratio": 0.8,
        "max_image_resolution": DEFAULT_MAX_IMAGE_RESOLUTION,
        "batch": batch.to_dict(),
        "max_steps": DEFAULT_STAGE2_MAX_STEPS,
        "mask_original_image_after_tgvf": True,
        "mask_original_image_after_tgvf_prob": 1.0,
        "mask_original_image_after_tgvf_scope": "through_answer",
        "deepstack_enabled": False,
        "deepstack_supported": True,
        "lora": {
            "rank": 64,
            "alpha": 256,
            "dropout": 0.05,
            "bias": "none",
            "target_modules": list(DEFAULT_STAGE2_LORA_TARGET_MODULES),
        },
        "weighted_span_loss": dict(DEFAULT_STAGE2_SPAN_WEIGHTS),
        "lr_scheduler": "cosine",
        "warmup_steps": 100,
        "warmup_ratio": 0.03,
        "min_lr_ratio": 0.1,
        "adam_betas": [0.9, 0.95],
        "adam_eps": 1e-8,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "clean_launcher_actions": ["dry_run", "write_plan"],
    }


def _config_from_args(args: argparse.Namespace) -> Stage2LaunchConfig:
    if not args.run_id:
        raise ValueError("--run-id is required")
    if not args.train_file:
        raise ValueError("--train-file is required")
    if not args.stage1_checkpoint:
        raise ValueError("--stage1-checkpoint is required")
    if not args.output_dir:
        raise ValueError("--output-dir is required")
    mask_scope = OriginalImageMaskScope(args.mask_original_image_after_tgvf_scope)
    deepstack_scope = (
        DeepStackScope(args.deepstack_original_image_scope)
        if args.deepstack_original_image_scope
        else DeepStackScope(str(mask_scope))
    )
    deepstack = (
        DeepStackState(enabled=True, original_image_scope=deepstack_scope)
        if args.deepstack_enabled
        else DeepStackState()
    )
    batch = resolve_batch_identity(
        global_batch_size=args.global_batch,
        world_size=args.world_size,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    weighted_span_loss = {
        "evidence_state": args.loss_evidence_state,
        "focus_target": args.loss_focus_target,
        "evidence": args.loss_evidence,
        "value_span": args.loss_value_span,
        "answer": args.loss_answer,
        "no_focus_evidence_state": args.loss_no_focus_evidence_state,
        "no_focus_answer": args.loss_no_focus_answer,
    }
    return Stage2LaunchConfig(
        run_id=args.run_id,
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        stage1_checkpoint=args.stage1_checkpoint,
        model_id=args.model_id,
        processor_id=args.processor_id,
        protocol=args.protocol,
        max_image_resolution=args.max_image_resolution,
        max_seq_len=args.max_seq_len,
        max_steps=args.max_steps,
        save_every=args.save_every,
        eval_every=args.eval_every,
        seed=args.seed,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        variant=args.variant,
        use_stage1_tgvf_config=args.use_stage1_tgvf_config,
        fast_batched_stage2=args.fast_batched_stage2,
        fvt_position_mode=args.fvt_position_mode,
        target_focus_ratio=args.target_focus_ratio,
        mask_original_image_after_tgvf=args.mask_original_image_after_tgvf,
        mask_original_image_after_tgvf_prob=args.mask_original_image_after_tgvf_prob,
        mask_original_image_after_tgvf_scope=mask_scope,
        deepstack=deepstack,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_bias=args.lora_bias,
        lora_target_modules=_parse_lora_target_modules(args.lora_target_modules),
        lr_lora=args.lr_lora,
        lr_tgvf=args.lr_tgvf,
        lr_calibration=args.lr_calibration,
        lr_scheduler=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        loss_visual_token_manifold=args.loss_visual_token_manifold,
        weighted_span_loss=weighted_span_loss,
        min_confidence=args.min_confidence,
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
        batch=batch,
    )


def _parse_lora_target_modules(text: str) -> tuple[str, ...]:
    modules = tuple(item.strip() for item in str(text).split(",") if item.strip())
    if not modules:
        raise ValueError("--lora-target-modules must contain at least one module name")
    return modules


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
