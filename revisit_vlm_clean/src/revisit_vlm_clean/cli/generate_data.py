"""Clean data-generation planning CLI."""

from __future__ import annotations

import argparse
import subprocess

from revisit_vlm_clean.cli.common import exit_not_implemented, print_json
from revisit_vlm_clean.data_generation import (
    DataGenerationConfig,
    DataGenerationStage,
    DataGenerationTransform,
    build_data_generation_plan,
    execute_data_generation,
    write_data_generation_plan,
)
from revisit_vlm_clean.defaults import DEFAULT_PROTOCOL
from revisit_vlm_clean.tgvf_protocol import SUPPORTED_PROTOCOLS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean TGVF data-generation planner.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=[item.value for item in DataGenerationStage], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-root", default=".")
    parser.add_argument("--input-files", nargs="*", default=[])
    parser.add_argument("--protocol", choices=SUPPORTED_PROTOCOLS, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--transform",
        choices=[item.value for item in DataGenerationTransform],
        default=DataGenerationTransform.NONE.value,
    )
    parser.add_argument("--source-manifest-path", default=None)
    parser.add_argument("--source-manifest-hash", default=None)
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--split-policy", default="preserve_input")
    parser.add_argument("--field-weight", action="append", default=[], help="Field loss weight as name=value.")
    parser.add_argument("--mask-policy", action="append", default=[], help="Mask policy entry as name=value.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved identity plan without writing files.")
    parser.add_argument("--write-plan", action="store_true", help="Write identity-only plan files.")
    parser.add_argument("--execute", action="store_true", help="Execute a ported deterministic data transform.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    git_commit, dirty_worktree = _git_identity()
    config = DataGenerationConfig(
        run_id=args.run_id,
        stage=DataGenerationStage(args.stage),
        output_dir=args.output_dir,
        input_root=args.input_root,
        input_files=tuple(args.input_files),
        protocol=args.protocol,
        transform=DataGenerationTransform(args.transform),
        source_manifest_path=args.source_manifest_path,
        source_manifest_hash=args.source_manifest_hash,
        source_run_id=args.source_run_id,
        split_policy=args.split_policy,
        field_weights=_parse_float_mapping(args.field_weight),
        mask_policy=_parse_string_mapping(args.mask_policy),
        git_commit=git_commit,
        dirty_worktree=dirty_worktree,
    )
    config.validate()
    if args.dry_run:
        print_json(build_data_generation_plan(config))
        return 0
    if args.write_plan:
        print_json(write_data_generation_plan(args.output_dir, config=config))
        return 0
    if args.execute:
        print_json(execute_data_generation(config))
        return 0
    return exit_not_implemented("use --dry-run, --write-plan, or --execute with a ported transform")


def _parse_float_mapping(values: list[str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for value in values:
        key, raw = _split_key_value(value)
        parsed[key] = float(raw)
    return parsed


def _parse_string_mapping(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        key, raw = _split_key_value(value)
        parsed[key] = raw
    return parsed


def _split_key_value(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"expected key=value, got {value!r}")
    key, raw = value.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"empty key in {value!r}")
    return key, raw.strip()


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
