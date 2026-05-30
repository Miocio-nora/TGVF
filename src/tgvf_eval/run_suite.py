from __future__ import annotations

import argparse
from copy import copy
from typing import Any

from tgvf_eval.run import parse_args as parse_run_args
from tgvf_eval.run import run_benchmark


SUITES: dict[str, dict[str, Any]] = {
    "core_light_image": {
        "tier": "light",
        "benchmarks": ["vstar_bench", "hr_bench_4k", "ocrbench_v2", "blink", "mmmu_pro", "mathvista"],
        "methods": ["direct_qwen", "tgvf_prompt_only_free", "tgvf_module_free", "tgvf_module_force"],
    },
    "core_medium_image": {
        "tier": "medium",
        "benchmarks": ["vstar_bench", "hr_bench_4k", "ocrbench_v2", "blink", "mmmu_pro", "mathvista"],
        "methods": ["direct_qwen", "tgvf_prompt_only_free", "tgvf_module_free", "tgvf_module_force"],
    },
    "video_light": {
        "tier": "light",
        "benchmarks": ["ovo_bench"],
        "methods": ["direct_qwen", "tgvf_module_free"],
    },
    "math_light": {
        "tier": "light",
        "benchmarks": ["mathvista", "mathverse"],
        "methods": ["direct_qwen", "tgvf_prompt_only_free", "tgvf_module_free", "tgvf_module_force"],
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a TGVF benchmark suite.")
    parser.add_argument("--suite", choices=sorted(SUITES), default="core_light_image")
    parser.add_argument("--benchmarks", default=None)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    known, rest = parser.parse_known_args(argv)
    base = parse_run_args(["--benchmark", "vstar_bench", *rest])
    for key, value in vars(known).items():
        setattr(base, key, value)
    return base


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    suite = SUITES[args.suite]
    benchmarks = args.benchmarks.split(",") if args.benchmarks else suite["benchmarks"]
    methods = args.methods.split(",") if args.methods else suite["methods"]
    for benchmark in benchmarks:
        for method in methods:
            run_args = copy(args)
            run_args.benchmark = benchmark
            run_args.method = method
            run_args.tier = suite["tier"]
            run_args.trigger_mode = None
            run_args.tgvf_mode = None
            if args.suite == "video_light" and method == "tgvf_module_free":
                for mode in ("per_frame_reencode", "nextframe_encode_no_reencode"):
                    video_args = copy(run_args)
                    video_args.video_foveation_mode = mode
                    run_benchmark(video_args)
            else:
                run_benchmark(run_args)


if __name__ == "__main__":
    main()
