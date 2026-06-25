"""CLI for deterministic clean benchmark shard merging."""

from __future__ import annotations

import argparse

from revisit_vlm_clean.benchmark_merge import merge_benchmark_shards
from revisit_vlm_clean.cli.common import print_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge clean benchmark shard outputs.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--expected-num-shards", type=int, default=None)
    parser.add_argument("--expected-source-manifest-hash", default=None)
    parser.add_argument("shard_dirs", nargs="+")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print_json(
        merge_benchmark_shards(
            args.shard_dirs,
            output_dir=args.output_dir,
            run_id=args.run_id,
            expected_num_shards=args.expected_num_shards,
            expected_source_manifest_hash=args.expected_source_manifest_hash,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
