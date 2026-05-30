from __future__ import annotations

import argparse
import json

from tgvf_eval.run import parse_args as parse_run_args
from tgvf_eval.run import run_benchmark


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile small TGVF benchmark slices.")
    parser.add_argument("--benchmarks", required=True)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--limit", type=int, default=50)
    known, rest = parser.parse_known_args(argv)
    base = parse_run_args(["--benchmark", known.benchmarks.split(",")[0], *rest])
    base.benchmarks = known.benchmarks
    base.methods = known.methods
    base.limit = known.limit
    return base


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summaries = []
    for benchmark in args.benchmarks.split(","):
        for method in args.methods.split(","):
            args.benchmark = benchmark
            args.method = method
            args.trigger_mode = None
            args.tgvf_mode = None
            summary = run_benchmark(args)
            samples = max(1, int(summary["num_samples"]))
            summary["estimated_full_runtime_sec_per_1000"] = (summary.get("avg_wall_time_sec") or 0.0) * 1000
            summary["profile_limit"] = samples
            summaries.append(summary)
    print(json.dumps({"profiles": summaries}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
