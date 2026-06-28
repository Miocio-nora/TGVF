"""Stepwise Stage3 GRPO runner with resume-safe sample scheduling."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from revisit_vlm_clean.cli.common import print_json
from revisit_vlm_clean.cli.train_stage3_grpo import (
    _git_identity,
    build_stage3_grpo_training_plan,
    write_stage3_grpo_training_plan,
)
from revisit_vlm_clean.stage3_grpo.data import load_stage3_sample_schedule, write_jsonl
from revisit_vlm_clean.stage3_grpo.schemas import Stage3GRPOConfig, now_iso, read_json, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage3 GRPO one judged step at a time.")
    parser.add_argument("--template-plan", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--start-step", type=int, default=None)
    parser.add_argument("--target-step", type=int, default=None)
    parser.add_argument("--max-new-steps", type=int, default=1)
    parser.add_argument("--pythonpath", default="revisit_vlm_clean/src")
    parser.add_argument("--torchrun", default="torchrun")
    parser.add_argument("--judge-backend", choices=("local_qwen_vl", "fake"), default="local_qwen_vl")
    parser.add_argument("--judge-model-preset", default="qwen3_vl_32b_thinking")
    parser.add_argument("--judge-devices", default="cuda:0")
    parser.add_argument("--judge-max-image-resolution", type=int, default=512)
    parser.add_argument("--judge-max-new-tokens", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not (args.dry_run or args.preflight_only or args.execute):
        raise SystemExit("use --dry-run, --preflight-only, or --execute")
    template_plan = read_json(args.template_plan)
    config = Stage3GRPOConfig.from_dict(dict(template_plan.get("config") or template_plan))
    output_root = Path(args.output_root)
    state_path = Path(args.state_path or output_root / "stage3_grpo_stepwise_state.json")
    state = _load_or_init_state(
        state_path=state_path,
        config=config,
        output_root=output_root,
        target_step=args.target_step,
    )
    preflight = _stepwise_preflight(config=config, state=state, output_root=output_root)
    if args.dry_run or args.preflight_only:
        print_json(preflight)
        return 0 if preflight["status"] == "passed" else 1
    if preflight["status"] != "passed":
        print_json(preflight)
        return 1
    result = execute_stepwise(
        config=config,
        output_root=output_root,
        state_path=state_path,
        state=state,
        start_step=args.start_step,
        target_step=args.target_step,
        max_new_steps=args.max_new_steps,
        pythonpath=args.pythonpath,
        torchrun=args.torchrun,
        judge_backend=args.judge_backend,
        judge_model_preset=args.judge_model_preset,
        judge_devices=_parse_csv(args.judge_devices),
        judge_max_image_resolution=args.judge_max_image_resolution,
        judge_max_new_tokens=args.judge_max_new_tokens,
    )
    print_json(result)
    return 0 if result["status"] in {"completed", "running"} else 1


def execute_stepwise(
    *,
    config: Stage3GRPOConfig,
    output_root: Path,
    state_path: Path,
    state: dict[str, Any],
    start_step: int | None,
    target_step: int | None,
    max_new_steps: int,
    pythonpath: str,
    torchrun: str,
    judge_backend: str,
    judge_model_preset: str,
    judge_devices: list[str],
    judge_max_image_resolution: int,
    judge_max_new_tokens: int,
) -> dict[str, Any]:
    next_step = int(start_step or state.get("next_step") or 1)
    final_step = int(target_step or state.get("target_steps") or next_step)
    max_new = max(1, int(max_new_steps))
    completed_now: list[int] = []
    state["status"] = "running"
    _write_state(state_path, state)
    for step in range(next_step, final_step + 1):
        if len(completed_now) >= max_new:
            break
        if step in set(int(item) for item in state.get("completed_steps") or []):
            state["next_step"] = step + 1
            _write_state(state_path, state)
            continue
        try:
            step_result = _execute_one_step(
                template_config=config,
                output_root=output_root,
                state=state,
                state_path=state_path,
                global_step=step,
                pythonpath=pythonpath,
                torchrun=torchrun,
                judge_backend=judge_backend,
                judge_model_preset=judge_model_preset,
                judge_devices=judge_devices,
                judge_max_image_resolution=judge_max_image_resolution,
                judge_max_new_tokens=judge_max_new_tokens,
            )
            completed_now.append(step)
            _append_event(output_root, {"event": "step_completed", "global_step": step, **step_result})
        except Exception as exc:
            failed = list(state.get("failed_steps") or [])
            failed.append({"global_step": step, "error": f"{type(exc).__name__}: {exc}", "created_at": now_iso()})
            state["failed_steps"] = failed
            state["status"] = "failed"
            _write_state(state_path, state)
            _append_event(output_root, {"event": "step_failed", "global_step": step, "error": str(exc)})
            raise
    if int(state.get("next_step") or 1) > final_step:
        state["status"] = "completed"
    else:
        state["status"] = "running"
    _write_state(state_path, state)
    return {
        "status": state["status"],
        "state_path": str(state_path),
        "completed_now": completed_now,
        "next_step": state.get("next_step"),
        "current_checkpoint": state.get("current_checkpoint"),
    }


def _execute_one_step(
    *,
    template_config: Stage3GRPOConfig,
    output_root: Path,
    state: dict[str, Any],
    state_path: Path,
    global_step: int,
    pythonpath: str,
    torchrun: str,
    judge_backend: str,
    judge_model_preset: str,
    judge_devices: list[str],
    judge_max_image_resolution: int,
    judge_max_new_tokens: int,
) -> dict[str, Any]:
    step_dir = output_root / f"step_{int(global_step):06d}"
    judge_dir = step_dir / "offline_judge_32b"
    current_checkpoint = str(state.get("current_checkpoint") or template_config.policy_checkpoint)
    step_config = replace(
        template_config,
        run_id=f"{template_config.run_id.rsplit('_step', 1)[0]}_step{int(global_step):06d}",
        output_dir=str(step_dir),
        policy_checkpoint=current_checkpoint,
        sample_schedule_start_step=int(global_step),
        judge=replace(
            template_config.judge,
            focus_cache_path=str(judge_dir / "focus_judge_cache.jsonl"),
            grounding_cache_path=str(judge_dir / "grounding_judge_cache.jsonl"),
        ),
        wandb=replace(
            template_config.wandb,
            name=(
                f"{template_config.wandb.name.rsplit('_step', 1)[0]}_step{int(global_step):06d}"
                if template_config.wandb.name
                else None
            ),
        ),
    )
    git_commit, dirty_worktree = _git_identity()
    plan = build_stage3_grpo_training_plan(
        step_config,
        git_commit=git_commit,
        dirty_worktree=dirty_worktree,
    )
    plan_paths = write_stage3_grpo_training_plan(step_dir, plan)
    plan_path = plan_paths["training_plan"]
    env = _subprocess_env(pythonpath)
    _run_command(
        _executor_command(plan_path, "--preflight-only"),
        cwd=Path.cwd(),
        env=env,
        log_path=step_dir / "preflight_stdout.log",
    )
    _run_command(
        _distributed_executor_command(
            torchrun=torchrun,
            world_size=int(step_config.train.world_size),
            plan_path=plan_path,
            action="--rollout-only",
        ),
        cwd=Path.cwd(),
        env=env,
        log_path=step_dir / "rollout_only_torchrun_stdout.log",
    )
    pending_path = _merge_pending_rows(step_dir)
    if step_config.judge.enabled and step_config.judge.mode != "disabled" and _jsonl_count(pending_path) > 0:
        _run_judge_shards(
            pending_path=pending_path,
            output_dir=judge_dir,
            focus_cache_path=Path(step_config.judge.focus_cache_path or judge_dir / "focus_judge_cache.jsonl"),
            grounding_cache_path=Path(step_config.judge.grounding_cache_path or judge_dir / "grounding_judge_cache.jsonl"),
            pythonpath=pythonpath,
            judge_backend=judge_backend,
            judge_model_preset=judge_model_preset,
            judge_devices=judge_devices,
            judge_max_image_resolution=judge_max_image_resolution,
            judge_max_new_tokens=judge_max_new_tokens,
        )
    _run_command(
        _executor_command(plan_path, "--preflight-only"),
        cwd=Path.cwd(),
        env=env,
        log_path=step_dir / "preflight_after_judge_stdout.log",
    )
    _run_command(
        _distributed_executor_command(
            torchrun=torchrun,
            world_size=int(step_config.train.world_size),
            plan_path=plan_path,
            action="--launch-training",
        ),
        cwd=Path.cwd(),
        env=env,
        log_path=step_dir / "torchrun_train_stdout.log",
    )
    checkpoint = step_dir / "checkpoint_step_1.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"expected checkpoint missing after step {global_step}: {checkpoint}")
    _mark_step_completed(
        state=state,
        state_path=state_path,
        config=step_config,
        checkpoint=checkpoint,
        global_step=global_step,
    )
    return {
        "step_dir": str(step_dir),
        "checkpoint": str(checkpoint),
        "pending_rows": _jsonl_count(pending_path),
    }


def _mark_step_completed(
    *,
    state: dict[str, Any],
    state_path: Path,
    config: Stage3GRPOConfig,
    checkpoint: Path,
    global_step: int,
) -> None:
    schedule_rows = load_stage3_sample_schedule(str(config.sample_schedule_path))
    consumed = [row for row in schedule_rows if int(row["global_step"]) == int(global_step)]
    completed = list(state.get("completed_steps") or [])
    if int(global_step) not in set(int(item) for item in completed):
        completed.append(int(global_step))
    used_sample_ids = list(state.get("used_sample_ids") or [])
    used_image_uids = list(state.get("used_image_uids") or [])
    used_sample_ids.extend(str(row["sample_id"]) for row in consumed)
    used_image_uids.extend(str(row.get("stable_image_uid") or "") for row in consumed if row.get("stable_image_uid"))
    if len(used_sample_ids) != len(set(used_sample_ids)):
        raise ValueError("stepwise state detected repeated sample_id consumption")
    if len(used_image_uids) != len(set(used_image_uids)):
        raise ValueError("stepwise state detected repeated image consumption")
    state["completed_steps"] = sorted(int(item) for item in completed)
    state["current_checkpoint"] = str(checkpoint)
    state["next_step"] = int(global_step) + 1
    state["used_sample_ids"] = used_sample_ids
    state["used_image_uids"] = used_image_uids
    state["updated_at"] = now_iso()
    _write_state(state_path, state)


def _run_judge_shards(
    *,
    pending_path: Path,
    output_dir: Path,
    focus_cache_path: Path,
    grounding_cache_path: Path,
    pythonpath: str,
    judge_backend: str,
    judge_model_preset: str,
    judge_devices: list[str],
    judge_max_image_resolution: int,
    judge_max_new_tokens: int,
) -> None:
    rows = _read_jsonl(pending_path)
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    devices = judge_devices or ["cuda:0"]
    shard_paths = []
    processes = []
    for shard_index, device in enumerate(devices):
        shard_rows = rows[shard_index:: len(devices)]
        if not shard_rows:
            continue
        shard_dir = output_dir / f"shard_{shard_index}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_pending = shard_dir / "judge_pending.jsonl"
        shard_focus = shard_dir / "focus_judge_cache.jsonl"
        shard_ground = shard_dir / "grounding_judge_cache.jsonl"
        write_jsonl(shard_pending, shard_rows)
        env = _subprocess_env(pythonpath)
        env["CUDA_VISIBLE_DEVICES"] = _cuda_visible_device(device)
        cmd = [
            sys.executable,
            "-m",
            "revisit_vlm_clean.cli.stage3_grpo_judge",
            "--pending-path",
            str(shard_pending),
            "--output-dir",
            str(shard_dir),
            "--focus-cache-path",
            str(shard_focus),
            "--grounding-cache-path",
            str(shard_ground),
            "--backend",
            judge_backend,
            "--model-preset",
            judge_model_preset,
            "--device",
            "cuda:0" if device.startswith("cuda") else device,
            "--device-map",
            "cuda:0" if device.startswith("cuda") else device,
            "--dtype",
            "bfloat16",
            "--attn-implementation",
            "sdpa",
            "--max-image-resolution",
            str(judge_max_image_resolution),
            "--max-new-tokens",
            str(judge_max_new_tokens),
            "--no-skip-existing",
            "--no-append",
            "--execute",
        ]
        log_handle = (shard_dir / "judge_stdout.log").open("w", encoding="utf-8")
        process = subprocess.Popen(cmd, cwd=Path.cwd(), env=env, stdout=log_handle, stderr=subprocess.STDOUT)
        processes.append((process, log_handle, cmd))
        shard_paths.append((shard_focus, shard_ground))
    failures = []
    for process, log_handle, cmd in processes:
        returncode = process.wait()
        log_handle.close()
        if returncode != 0:
            failures.append({"returncode": returncode, "cmd": cmd})
    if failures:
        raise RuntimeError(f"judge shard failed: {failures[:2]}")
    _merge_cache_shards([focus for focus, _ground in shard_paths], focus_cache_path)
    _merge_cache_shards([ground for _focus, ground in shard_paths], grounding_cache_path)


def _merge_pending_rows(step_dir: Path) -> Path:
    paths = [step_dir / "judge_pending.jsonl"]
    paths.extend(sorted(step_dir.glob("rank_*/judge_pending.jsonl")))
    rows = []
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in _read_jsonl(path):
            key = str(row.get("cache_key") or json.dumps(row, sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    out = step_dir / "judge_pending_all_ranks.jsonl"
    write_jsonl(out, rows)
    write_json(
        step_dir / "judge_pending_merge_summary.json",
        {
            "created_at": now_iso(),
            "input_paths": [str(path) for path in paths if path.exists()],
            "rows": len(rows),
            "judge_type": dict(Counter(str(row.get("judge_type") or "unknown") for row in rows)),
        },
    )
    return out


def _merge_cache_shards(paths: list[Path], output_path: Path) -> None:
    rows = []
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in _read_jsonl(path):
            key = str(row.get("cache_key") or json.dumps(row, sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    write_jsonl(output_path, rows)


def _load_or_init_state(
    *,
    state_path: Path,
    config: Stage3GRPOConfig,
    output_root: Path,
    target_step: int | None,
) -> dict[str, Any]:
    if state_path.exists():
        state = read_json(state_path)
        if not isinstance(state, dict):
            raise ValueError(f"state must be a JSON object: {state_path}")
        return state
    state = {
        "schema_version": "stage3_grpo_stepwise_state_v0",
        "created_at": now_iso(),
        "run_id": config.run_id.rsplit("_step", 1)[0],
        "status": "planned",
        "target_steps": int(target_step or config.train.max_steps),
        "completed_steps": [],
        "failed_steps": [],
        "next_step": 1,
        "current_checkpoint": None,
        "schedule_path": config.sample_schedule_path,
        "output_root": str(output_root),
        "used_sample_ids": [],
        "used_image_uids": [],
    }
    _write_state(state_path, state)
    return state


def _stepwise_preflight(
    *,
    config: Stage3GRPOConfig,
    state: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    errors = []
    warnings = []
    if not config.sample_schedule_path:
        errors.append("template config missing sample_schedule_path")
        schedule_rows = []
    elif not Path(config.sample_schedule_path).exists():
        errors.append(f"sample_schedule_path does not exist: {config.sample_schedule_path}")
        schedule_rows = []
    else:
        schedule_rows = load_stage3_sample_schedule(config.sample_schedule_path)
    sample_ids = [str(row["sample_id"]) for row in schedule_rows]
    image_uids = [str(row.get("stable_image_uid") or "") for row in schedule_rows if row.get("stable_image_uid")]
    if len(sample_ids) != len(set(sample_ids)):
        errors.append("sample schedule contains duplicate sample ids")
    if len(image_uids) != len(set(image_uids)):
        warnings.append("sample schedule contains duplicate image uids")
    completed = [int(item) for item in state.get("completed_steps") or []]
    used_sample_ids = list(state.get("used_sample_ids") or [])
    if len(used_sample_ids) != len(set(used_sample_ids)):
        errors.append("state contains duplicate used_sample_ids")
    return {
        "schema_version": "stage3_grpo_stepwise_preflight_v0",
        "created_at": now_iso(),
        "status": "failed" if errors else "passed",
        "errors": errors,
        "warnings": warnings,
        "output_root": str(output_root),
        "state": {
            "status": state.get("status"),
            "target_steps": state.get("target_steps"),
            "next_step": state.get("next_step"),
            "completed_count": len(completed),
            "current_checkpoint": state.get("current_checkpoint"),
        },
        "schedule": {
            "path": config.sample_schedule_path,
            "rows": len(schedule_rows),
            "step_count": len({int(row["global_step"]) for row in schedule_rows}),
            "rank_count": len({int(row["rank"]) for row in schedule_rows}),
        },
    }


def _distributed_executor_command(
    *,
    torchrun: str,
    world_size: int,
    plan_path: str,
    action: str,
) -> list[str]:
    if int(world_size) <= 1:
        return _executor_command(plan_path, action)
    return [
        torchrun,
        "--standalone",
        "--nproc_per_node",
        str(int(world_size)),
        "-m",
        "revisit_vlm_clean.training.stage3_grpo_executor",
        "--plan",
        str(plan_path),
        action,
    ]


def _executor_command(plan_path: str, action: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "revisit_vlm_clean.training.stage3_grpo_executor",
        "--plan",
        str(plan_path),
        action,
    ]


def _run_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        handle.flush()
        completed = subprocess.run(cmd, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
        elapsed = time.perf_counter() - started
        handle.write(f"\n[elapsed_seconds] {elapsed:.3f}\n")
    if completed.returncode != 0:
        raise RuntimeError(f"command failed rc={completed.returncode}: {' '.join(cmd)}")


def _subprocess_env(pythonpath: str) -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = pythonpath if not existing else f"{pythonpath}:{existing}"
    return env


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _cuda_visible_device(device: str) -> str:
    if device.startswith("cuda:"):
        return device.split(":", 1)[1]
    return device


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _jsonl_count(path: Path) -> int:
    return len(_read_jsonl(path))


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, state)


def _append_event(output_root: Path, row: dict[str, Any]) -> None:
    path = output_root / "stage3_grpo_stepwise_events.jsonl"
    rows = [{"created_at": now_iso(), **row}]
    write_jsonl(path, _read_jsonl(path) + rows)


if __name__ == "__main__":
    raise SystemExit(main())
