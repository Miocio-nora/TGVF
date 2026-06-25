"""Manifest CLI skeleton."""

from __future__ import annotations

import argparse
from pathlib import Path

from revisit_vlm_clean.cli.common import print_json
from revisit_vlm_clean.manifest import (
    build_manifest,
    describe_subset,
    manifest_payload,
    write_manifest,
)
from revisit_vlm_clean.populations import iter_core_populations, iter_subsets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean TGVF benchmark manifest tool.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--list",
        action="store_true",
        help="List known clean populations and subsets.",
    )
    group.add_argument("--describe-subset", help="Describe a known subset id.")
    group.add_argument("--build", help="Build a manifest for a known subset id.")
    parser.add_argument(
        "--benchmark-root",
        default="/home/dredvpn009/Flash_Storage/datasets/benchmarks",
    )
    parser.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        print_json(
            {
                "populations": list(iter_core_populations()),
                "subsets": list(iter_subsets()),
            }
        )
        return 0
    if args.describe_subset:
        print_json(describe_subset(args.describe_subset))
        return 0
    manifest = build_manifest(subset_id=args.build, benchmark_root=args.benchmark_root)
    if args.output:
        write_manifest(args.output, manifest)
        print_json(
            {
                "output": str(Path(args.output)),
                "manifest_hash": manifest.stable_hash(),
                "n": len(manifest.samples),
            }
        )
    else:
        print_json(manifest_payload(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
