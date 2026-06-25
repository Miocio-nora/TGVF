"""Clean ValKit/VLMEvalKit preflight CLI."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from revisit_vlm_clean.cli.common import exit_not_implemented, print_json
from revisit_vlm_clean.defaults import DEFAULT_MAX_IMAGE_RESOLUTION, DEFAULT_MODEL_ID
from revisit_vlm_clean.schema import EvalMode, ForwardMode
from revisit_vlm_clean.tgvf_protocol import SUPPORTED_PROTOCOLS
from revisit_vlm_clean.valkit import (
    ValKitRunConfig,
    build_valkit_plan,
    build_valkit_preflight_report,
    execute_valkit_plan,
    write_valkit_execution_bundle,
    write_valkit_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean ValKit/VLMEvalKit preflight runner.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--benchmark", action="append", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--processor-id", default=None)
    parser.add_argument("--mode", choices=[item.value for item in EvalMode], default="original")
    parser.add_argument("--max-image-resolution", type=int, default=DEFAULT_MAX_IMAGE_RESOLUTION)
    parser.add_argument(
        "--tgvf-protocol", choices=SUPPORTED_PROTOCOLS, default="protocol_c_tool_observation"
    )
    parser.add_argument(
        "--post-tgvf-forward-mode",
        choices=[item.value for item in ForwardMode],
        default=ForwardMode.KV_CACHE.value,
    )
    parser.add_argument("--stage2-checkpoint", default=None)
    parser.add_argument("--valkit-root", default=None)
    parser.add_argument(
        "--valkit-model-name",
        default=None,
        help="Model key registered in VLMEvalKit config. Required for --execute.",
    )
    parser.add_argument(
        "--valkit-run-mode",
        choices=("all", "infer", "eval"),
        default="all",
        help="VLMEvalKit run.py --mode value.",
    )
    parser.add_argument("--reuse", action="store_true", help="Pass --reuse to VLMEvalKit.")
    parser.add_argument("--verbose", action="store_true", help="Pass --verbose to VLMEvalKit.")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-plan", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--prepare-execution", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execution-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    git_commit, dirty_worktree = _git_identity()
    config = ValKitRunConfig(
        run_id=args.run_id,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        benchmarks=tuple(args.benchmark),
        model_id=args.model_id,
        processor_id=args.processor_id,
        mode=EvalMode(args.mode),
        max_image_resolution=args.max_image_resolution,
        tgvf_protocol=args.tgvf_protocol,
        post_tgvf_forward_mode=ForwardMode(args.post_tgvf_forward_mode),
        stage2_checkpoint=args.stage2_checkpoint,
        valkit_root=args.valkit_root,
        valkit_model_name=args.valkit_model_name,
        valkit_run_mode=args.valkit_run_mode,
        reuse=args.reuse,
        verbose=args.verbose,
        work_dir=args.work_dir,
        git_commit=git_commit,
        dirty_worktree=dirty_worktree,
    )
    plan = build_valkit_plan(config)
    if args.dry_run:
        print_json(plan)
        return 0
    if args.prepare_execution:
        plan_artifacts = write_valkit_plan(args.output_dir, plan)
        execution_dir = args.execution_dir or str(Path(args.output_dir) / "clean_valkit_execution")
        execution_artifacts = write_valkit_execution_bundle(execution_dir, plan)
        print_json({**plan_artifacts, **execution_artifacts})
        return 0
    if args.execute:
        plan_artifacts = write_valkit_plan(args.output_dir, plan)
        execution_dir = args.execution_dir or str(Path(args.output_dir) / "clean_valkit_execution")
        execution_artifacts = execute_valkit_plan(execution_dir, plan)
        print_json({**plan_artifacts, **execution_artifacts})
        return int(execution_artifacts["returncode"])
    if args.write_plan or args.preflight_only:
        print_json(write_valkit_plan(args.output_dir, plan))
        return 0
    print_json(build_valkit_preflight_report(plan))
    return exit_not_implemented(
        "clean ValKit execution requires explicit --execute; use --preflight-only, "
        "--write-plan, --prepare-execution, or --execute"
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
