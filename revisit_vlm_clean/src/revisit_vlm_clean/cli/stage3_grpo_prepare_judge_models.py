"""Prepare local Stage3 GRPO judge model checkpoints."""

from __future__ import annotations

import argparse

from revisit_vlm_clean.cli.common import print_json
from revisit_vlm_clean.stage3_grpo.judge_runner import DEFAULT_MODEL_ROOT, MODEL_PRESETS
from revisit_vlm_clean.stage3_grpo.model_prepare import (
    JUDGE_MODEL_TARGET_SETS,
    JudgeModelPrepareConfig,
    run_judge_model_prepare,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare local Stage3 GRPO judge models.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-root", default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--target", action="append", default=[], help="Preset name, HF id, or local path.")
    parser.add_argument(
        "--target-set",
        action="append",
        default=[],
        choices=sorted(JUDGE_MODEL_TARGET_SETS),
        help="Named model set to prepare.",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--remote-metadata", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--write-plan", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--execute-download", action="store_true")
    parser.add_argument("--allow-large-download", action="store_true")
    parser.add_argument("--print-presets", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.print_presets:
        print_json({"presets": MODEL_PRESETS, "target_sets": JUDGE_MODEL_TARGET_SETS})
        return 0
    if not args.output_dir:
        raise SystemExit("--output-dir is required unless --print-presets is used")
    if not (args.write_plan or args.preflight_only or args.execute_download):
        raise SystemExit("use --write-plan, --preflight-only, or --execute-download")
    target_sets = tuple(
        args.target_set or (() if args.target else ("stage3_qwen3_recommended",))
    )
    config = JudgeModelPrepareConfig(
        output_dir=args.output_dir,
        model_root=args.model_root,
        targets=tuple(args.target),
        target_sets=target_sets,
        revision=args.revision,
        max_workers=args.max_workers,
        execute_download=args.execute_download,
        allow_large_download=args.allow_large_download,
        local_files_only=args.local_files_only,
        remote_metadata=args.remote_metadata,
    )
    result = run_judge_model_prepare(
        config,
        preflight_only=args.preflight_only,
        write_plan_only=args.write_plan,
    )
    print_json(result)
    return 1 if result.get("status") == "missing_models" else 0


if __name__ == "__main__":
    raise SystemExit(main())
