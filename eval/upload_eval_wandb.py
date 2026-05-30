from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from revisit_vlm.wandb_logging import WandbLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload merged TGVF eval metrics and artifacts to W&B.")
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--group", default=None)
    parser.add_argument("--mode", default=None, choices=("online", "offline", "disabled"))
    parser.add_argument("--tags", default="")
    parser.add_argument("--artifact-name", default=None)
    parser.add_argument("--step", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_root = Path(args.eval_root)
    if not eval_root.exists():
        raise FileNotFoundError(f"Missing eval root: {eval_root}")

    tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()] or None
    logger = WandbLogger(
        project=args.project,
        entity=args.entity,
        name=args.run_name or eval_root.name,
        group=args.group,
        job_type="tgvf-evaluation",
        config={"eval_root": str(eval_root)},
        mode=args.mode,
        tags=tags,
    )
    metrics = collect_eval_metrics(eval_root)
    if metrics:
        logger.log(metrics, step=args.step)
    logger.log_artifact(
        name=args.artifact_name or safe_artifact_name(eval_root),
        artifact_type="tgvf-eval-output",
        paths=[eval_root],
        aliases=["latest"],
    )
    logger.finish()


def collect_eval_metrics(eval_root: Path) -> dict[str, float | int | bool | str]:
    reports = {
        "readout": eval_root / "readout" / "readout_eval_report.json",
        "query": eval_root / "query_sensitivity" / "query_sensitivity_report.json",
        "distribution": eval_root / "fvt_distribution" / "fvt_distribution_report.json",
    }
    for end2end_dir in sorted(eval_root.glob("end2end_*")):
        reports[end2end_dir.name] = end2end_dir / "end2end_eval_report.json"

    metrics: dict[str, float | int | bool | str] = {}
    for task, report_path in reports.items():
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text())
        flatten_numeric(report, prefix=f"eval/{task}", output=metrics)
    return metrics


def flatten_numeric(
    value: Any,
    *,
    prefix: str,
    output: dict[str, float | int | bool | str],
) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "config":
                continue
            flatten_numeric(item, prefix=f"{prefix}/{key}", output=output)
    elif isinstance(value, bool):
        output[prefix] = value
    elif isinstance(value, (int, float)) and value is not None:
        output[prefix] = value
    elif isinstance(value, str) and prefix.endswith("/mode"):
        output[prefix] = value


def safe_artifact_name(eval_root: Path) -> str:
    name = "-".join(eval_root.parts[-3:])
    safe = "".join(char if char.isalnum() or char in "._-" else "-" for char in name)
    return safe.strip("-") or "tgvf-eval-output"


if __name__ == "__main__":
    main()
