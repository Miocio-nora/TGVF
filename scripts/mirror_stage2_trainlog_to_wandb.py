#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mirror TGVF Stage2 JSON train.log metrics into a fresh W&B run.")
    parser.add_argument("--train-log", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--project", default="tgvf-v3")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--name", required=True)
    parser.add_argument("--group", default=None)
    parser.add_argument("--tags", default="")
    parser.add_argument("--poll-sec", type=float, default=30.0)
    parser.add_argument("--exit-when-idle-sec", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import wandb

    config: dict[str, Any] = {}
    if args.config and Path(args.config).exists():
        config = json.loads(Path(args.config).read_text())
    config["mirrored_from_train_log"] = str(Path(args.train_log))
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.name,
        group=args.group,
        job_type="tgvf-v3-stage2-trainlog-mirror",
        tags=[tag.strip() for tag in args.tags.split(",") if tag.strip()] or None,
        config=config,
    )
    seen_steps: set[int] = set()
    last_new = time.time()
    train_log = Path(args.train_log)
    while True:
        new_count = 0
        for record in parse_json_objects(train_log):
            step = record.get("step")
            if not isinstance(step, int) or step in seen_steps:
                continue
            metrics = to_wandb_metrics(record)
            if metrics:
                wandb.log(metrics, step=step)
                seen_steps.add(step)
                new_count += 1
        if new_count:
            last_new = time.time()
            print(json.dumps({"mirrored_new_steps": new_count, "latest_step": max(seen_steps)}, ensure_ascii=False), flush=True)
        if args.exit_when_idle_sec > 0 and time.time() - last_new >= args.exit_when_idle_sec:
            break
        time.sleep(args.poll_sec)
    run.finish()


def parse_json_objects(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(errors="ignore")
    decoder = json.JSONDecoder()
    out: list[dict[str, Any]] = []
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(obj, dict) and "step" in obj and "loss_total" in obj:
            out.append(obj)
        index = start + end
    return out


def to_wandb_metrics(record: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    passthrough = {
        "micro_step",
        "loss_total",
        "loss_focus",
        "loss_no_focus",
        "loss_visual_token_manifold",
        "grad_norm",
        "peak_memory_gb",
        "world_size",
        "effective_global_batch_size",
        "target_focus_sampling_ratio",
        "focus_count",
        "no_focus_count",
        "focus_ratio",
        "no_focus_ratio",
        "focus_sample_mask_active_rate",
        "no_focus_mask_active_rate",
        "value_span_match_rate",
    }
    for key in passthrough:
        value = record.get(key)
        if isinstance(value, (int, float)):
            metrics[f"train/{key}"] = value
    for key, value in record.items():
        if (key.startswith("protocol_c_boundary_acc_") or key.startswith("protocol_c_boundary_support_")) and isinstance(value, (int, float)):
            metrics[f"train/{key}"] = value
    for key in (
        "special_tokens_added",
        "normal_tokens_added",
        "tokenizer_resized",
        "markers_are_plain_text",
        "matrix_ce_enabled",
        "same_image_negative_enabled",
    ):
        if key in record:
            metrics[f"train/{key}"] = float(bool(record.get(key)))
    for index, lr in enumerate(record.get("learning_rates") or []):
        if isinstance(lr, (int, float)):
            metrics[f"train/lr_group_{index}"] = lr
    return metrics


if __name__ == "__main__":
    main()
