"""Diagnostic dynamic-queue benchmark runner.

This CLI is intentionally separate from the comparable fixed-shard runner.  It
schedules rows through a shared queue so faster GPU workers do not wait for
slower fixed shards.  It can optionally batch queued samples per worker and
writes partial rows as soon as each sample finishes.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from revisit_vlm_clean.benchmark_data import materialize_samples_from_manifest_payload
from revisit_vlm_clean.cli.benchmark import (
    _git_identity,
    _resolve_manifest_payload,
    _stage2_runtime_config,
)
from revisit_vlm_clean.defaults import DEFAULT_BENCHMARK_ROOT
from revisit_vlm_clean.outputs import (
    benchmark_source_manifest_reference,
    write_benchmark_source_manifest,
    write_executed_benchmark_output,
    write_run_config_text,
)
from revisit_vlm_clean.rendering import render_benchmark_inputs
from revisit_vlm_clean.runner import (
    BackendConfig,
    RUNNER_BACKENDS,
    STAGE2_LEGACY_BACKEND,
    STAGE2_NATIVE_BACKEND,
    backend_role_identity,
    make_backend,
    resolve_backend_name,
    summarize_executed_rows,
    _continuation_metadata,
    _d_shape,
    _final_output,
    _row_deepstack_execution,
    _trigger_policy,
)
from revisit_vlm_clean.schema import (
    DeepStackScope,
    DeepStackState,
    EvalFamily,
    EvalMode,
    ForwardMode,
    ParserScorerIdentity,
    RunConfig,
    ScoringBackend,
    _to_jsonable,
)
from revisit_vlm_clean.scoring import score_output_rows
from revisit_vlm_clean.tgvf_protocol import SUPPORTED_PROTOCOLS


SUPPORTED_DYNAMIC_BACKENDS = RUNNER_BACKENDS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnostic dynamic-queue clean benchmark runner."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Thinking")
    parser.add_argument("--processor-id", default=None)
    parser.add_argument(
        "--eval-family",
        choices=[EvalFamily.PROJECT_NATIVE_EXTERNAL.value, EvalFamily.INTERNAL_DIAGNOSTIC.value],
        default=EvalFamily.INTERNAL_DIAGNOSTIC.value,
    )
    parser.add_argument("--mode", choices=[item.value for item in EvalMode], default=EvalMode.ORIGINAL.value)
    parser.add_argument(
        "--post-tgvf-forward-mode",
        choices=[item.value for item in ForwardMode],
        default=ForwardMode.KV_CACHE.value,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--population-id")
    group.add_argument("--subset-id")
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--manifest-hash", default=None)
    parser.add_argument("--benchmark-root", default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-image-resolution", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-action-tokens", type=int, default=64)
    parser.add_argument("--max-answer-tokens", type=int, default=128)
    parser.add_argument(
        "--tgvf-protocol",
        choices=SUPPORTED_PROTOCOLS,
        default="protocol_c_tool_observation",
    )
    parser.add_argument(
        "--scoring-backend",
        choices=[item.value for item in ScoringBackend],
        default=ScoringBackend.AUTO.value,
    )
    parser.add_argument("--softforce-prompt-text", default="")
    parser.add_argument("--runner-backend", choices=SUPPORTED_DYNAMIC_BACKENDS, default="dry_run")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--original-no-thinking",
        action="store_true",
        help="For qwen3_original only: disable the thinking chat prefill and block think tags.",
    )
    parser.add_argument("--stage2-checkpoint", default="")
    parser.add_argument("--stage2-eval-jsonl", default="")
    parser.add_argument("--stage2-d-condition", default="correct_D")
    parser.add_argument("--force-prefix-mode", default="target_hint")
    parser.add_argument("--deepstack-enabled", action="store_true")
    parser.add_argument("--d-deepstack-enabled", action="store_true")
    parser.add_argument(
        "--deepstack-original-image-scope",
        choices=[item.value for item in DeepStackScope],
        default=DeepStackScope.OFF.value,
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="Comma-separated physical GPU ids. Use one worker per id.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of queued samples each GPU worker generates per model.generate call.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Diagnostic smoke only: evaluate the first N rows from the resolved manifest.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress every N completed rows.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    batch_size = int(args.batch_size)
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    max_samples = None if args.max_samples is None else int(args.max_samples)
    if max_samples is not None and max_samples < 1:
        raise ValueError("--max-samples must be >= 1 when provided")
    gpus = _parse_gpus(args.gpus)
    if not gpus:
        raise ValueError("--gpus must specify at least one GPU id")
    if args.d_deepstack_enabled and not args.deepstack_enabled:
        raise ValueError("--d-deepstack-enabled requires --deepstack-enabled")

    git_commit, dirty_worktree = _git_identity()
    deepstack_scope = DeepStackScope(args.deepstack_original_image_scope)
    if args.deepstack_enabled and deepstack_scope == DeepStackScope.OFF:
        deepstack_scope = DeepStackScope.NO_BLOCK
    config = RunConfig(
        run_id=args.run_id,
        checkpoint_path=args.checkpoint_path,
        started_at=datetime.now(timezone.utc).isoformat(),
        num_shards=len(gpus),
        shard_index=0,
        model_id=args.model_id,
        processor_id=args.processor_id,
        eval_family=EvalFamily(args.eval_family),
        mode=EvalMode(args.mode),
        population_id=args.population_id,
        subset_id=args.subset_id,
        manifest_path=args.manifest_path,
        manifest_hash=args.manifest_hash,
        benchmark_root=args.benchmark_root,
        max_image_resolution=args.max_image_resolution,
        max_tokens=args.max_tokens,
        max_action_tokens=args.max_action_tokens,
        max_answer_tokens=args.max_answer_tokens,
        tgvf_protocol=args.tgvf_protocol,
        post_tgvf_forward_mode=ForwardMode(args.post_tgvf_forward_mode),
        softforce_prompt_text=args.softforce_prompt_text,
        deepstack=DeepStackState(
            enabled=bool(args.deepstack_enabled),
            original_image_scope=deepstack_scope,
            d_features_enabled=bool(args.d_deepstack_enabled),
        ),
        parser_scorer=ParserScorerIdentity(
            scoring_backend=ScoringBackend(args.scoring_backend),
            fallback_allowed=ScoringBackend(args.scoring_backend) == ScoringBackend.AUTO,
        ),
        git_commit=git_commit,
        dirty_worktree=dirty_worktree,
    )
    config.validate()
    manifest = _resolve_manifest_payload(args)
    config = replace(
        config,
        manifest_hash=manifest.get("manifest_hash") or config.manifest_hash,
    )
    resolved_backend = resolve_backend_name(args.runner_backend)
    backend_config = BackendConfig(
        backend=args.runner_backend,
        dtype=args.dtype,
        device=args.device,
        device_map=None if args.device_map in {"", "none", "None", "null"} else args.device_map,
        attn_implementation=(
            None
            if args.attn_implementation in {"", "none", "None", "null"}
            else args.attn_implementation
        ),
        trust_remote_code=bool(args.trust_remote_code),
        original_no_thinking=bool(args.original_no_thinking),
        stage2=(
            _stage2_runtime_config(args, config)
            if resolved_backend in {STAGE2_NATIVE_BACKEND, STAGE2_LEGACY_BACKEND}
            else None
        ),
    )
    samples = materialize_samples_from_manifest_payload(
        manifest,
        benchmark_root=config.benchmark_root,
        metadata_only=args.runner_backend == "dry_run",
    )
    source_sample_count = len(samples)
    if max_samples is not None:
        samples = samples[:max_samples]
    rendered_inputs = render_benchmark_inputs(samples, config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_initial_artifacts(
        output_dir,
        config=config,
        manifest=manifest,
        backend_config=backend_config,
        gpus=gpus,
        batch_size=int(args.batch_size),
        max_samples=max_samples,
        source_sample_count=source_sample_count,
    )

    started = time.perf_counter()
    rows = _run_dynamic_queue(
        samples=samples,
        rendered_inputs=rendered_inputs,
        config=config,
        backend_config=backend_config,
        output_dir=output_dir,
        gpus=gpus,
        batch_size=batch_size,
        progress_every=max(1, int(args.progress_every)),
    )
    rows.sort(key=lambda row: int(row.get("manifest_index", 0)))

    scoring_error = None
    try:
        score_output_rows(
            rows,
            scoring_backend=config.parser_scorer.scoring_backend,
            benchmark_root=config.benchmark_root,
        )
    except Exception as exc:  # pragma: no cover - defensive parity with fixed runner
        scoring_error = f"{type(exc).__name__}: {exc}"
        for row in rows:
            row["scoring_completed"] = False
            row["scoring_error"] = scoring_error
    else:
        for row in rows:
            row["scoring_completed"] = True
            row["scoring_error"] = None

    summary = summarize_executed_rows(rows, config=config, manifest_hash=config.manifest_hash)
    if scoring_error is not None:
        summary = replace(
            summary,
            comparable=False,
            comparability_note=f"dynamic_queue_scoring_failed: {scoring_error}",
        )
    artifacts = write_executed_benchmark_output(
        output_dir,
        config=config,
        manifest=manifest,
        rows=rows,
        summary=summary,
        backend_config=backend_config,
    )
    _write_json(
        output_dir / "dynamic_summary.json",
        {
            "schema_version": "clean_dynamic_benchmark_summary_v1",
            "run_id": config.run_id,
            "runner_backend": backend_config.backend,
            "gpus": gpus,
            "batch_size": batch_size,
            "max_samples": max_samples,
            "source_sample_count": source_sample_count,
            "n_rows": len(rows),
            "elapsed_sec": time.perf_counter() - started,
            "completed_rows_path": str(output_dir / "partial_rows.jsonl"),
            "final_rows_path": artifacts["rows"],
            "timing": _timing_summary(rows),
        },
    )
    print(json.dumps(artifacts, sort_keys=True))
    return 0


def _run_dynamic_queue(
    *,
    samples: list[Any],
    rendered_inputs: list[Any],
    config: RunConfig,
    backend_config: BackendConfig,
    output_dir: Path,
    gpus: list[str],
    batch_size: int,
    progress_every: int,
) -> list[dict[str, Any]]:
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    for index in range(len(samples)):
        task_queue.put(index)
    for _ in gpus:
        task_queue.put(None)

    workers = []
    for worker_id, gpu in enumerate(gpus):
        proc = ctx.Process(
            target=_worker_main,
            kwargs={
                "worker_id": worker_id,
                "gpu": gpu,
                "task_queue": task_queue,
                "result_queue": result_queue,
                "samples": samples,
                "rendered_inputs": rendered_inputs,
                "config": config,
                "backend_config": backend_config,
                "batch_size": batch_size,
            },
        )
        proc.start()
        workers.append(proc)

    rows: list[dict[str, Any]] = []
    partial_path = output_dir / "partial_rows.jsonl"
    done_workers = 0
    with partial_path.open("w", encoding="utf-8") as handle:
        while done_workers < len(workers):
            message = result_queue.get()
            msg_type = message.get("type")
            if msg_type == "row":
                row = message["row"]
                rows.append(row)
                handle.write(json.dumps(_to_jsonable(row), sort_keys=True) + "\n")
                handle.flush()
                if len(rows) == 1 or len(rows) % progress_every == 0 or len(rows) == len(samples):
                    print(
                        (
                            "[clean_dynamic_progress] "
                            f"run_id={config.run_id} rows={len(rows)}/{len(samples)} "
                            f"worker={row.get('dynamic_worker_id')} gpu={row.get('dynamic_gpu')} "
                            f"batch_size={row.get('dynamic_batch_size')} "
                            f"sample_id={row.get('sample_id')}"
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
            elif msg_type == "worker_done":
                done_workers += 1
            elif msg_type == "worker_error":
                done_workers += 1
                print(
                    (
                        "[clean_dynamic_worker_error] "
                        f"run_id={config.run_id} worker={message.get('worker_id')} "
                        f"gpu={message.get('gpu')} error={message.get('error')}"
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            else:
                raise RuntimeError(f"unknown dynamic worker message: {message}")

    for proc in workers:
        proc.join(timeout=30)
        if proc.exitcode not in {0, None}:
            print(
                (
                    "[clean_dynamic_worker_exit] "
                    f"run_id={config.run_id} pid={proc.pid} exitcode={proc.exitcode}"
                ),
                file=sys.stderr,
                flush=True,
            )
    return rows


def _worker_main(
    *,
    worker_id: int,
    gpu: str,
    task_queue: Any,
    result_queue: Any,
    samples: list[Any],
    rendered_inputs: list[Any],
    config: RunConfig,
    backend_config: BackendConfig,
    batch_size: int,
) -> None:
    if backend_config.backend != "dry_run":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        backend = make_backend(backend_config, config=config)
        backend.prepare(config)
        resolved_backend = resolve_backend_name(backend_config.backend)
        batch_id = 0
        while True:
            batch_indices, stop_after_batch = _next_task_batch(task_queue, batch_size)
            if not batch_indices and not stop_after_batch:
                continue
            if not batch_indices and stop_after_batch:
                break
            batch_samples = [samples[index] for index in batch_indices]
            batch_rendered = [rendered_inputs[index] for index in batch_indices]
            started = time.perf_counter()
            results = backend.run_batch(batch_samples, batch_rendered, config)
            batch_wall_sec = time.perf_counter() - started
            for batch_index, (index, sample, rendered, result) in enumerate(
                zip(batch_indices, batch_samples, batch_rendered, results, strict=True)
            ):
                row = _row_from_result(
                    manifest_index=int(index),
                    worker_id=worker_id,
                    gpu=gpu,
                    sample=sample,
                    rendered=rendered,
                    result=result,
                    config=config,
                    backend_config=backend_config,
                    resolved_backend=resolved_backend,
                    queue_wait_sec=0.0,
                    worker_wall_sec=batch_wall_sec,
                    dynamic_batch_id=batch_id,
                    dynamic_batch_size=len(batch_indices),
                    dynamic_batch_index=batch_index,
                )
                result_queue.put({"type": "row", "row": row})
            batch_id += 1
            if stop_after_batch:
                break
    except Exception as exc:
        result_queue.put(
            {
                "type": "worker_error",
                "worker_id": worker_id,
                "gpu": gpu,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return
    result_queue.put({"type": "worker_done", "worker_id": worker_id, "gpu": gpu})


def _row_from_result(
    *,
    manifest_index: int,
    worker_id: int,
    gpu: str,
    sample: Any,
    rendered: Any,
    result: Any,
    config: RunConfig,
    backend_config: BackendConfig,
    resolved_backend: str,
    queue_wait_sec: float,
    worker_wall_sec: float,
    dynamic_batch_id: int,
    dynamic_batch_size: int,
    dynamic_batch_index: int,
) -> dict[str, Any]:
    backend_role = backend_role_identity(backend_config.backend)
    debug = dict(result.debug or {})
    debug["dynamic_queue"] = {
        "schema_version": "clean_dynamic_queue_row_v1",
        "manifest_index": manifest_index,
        "worker_id": worker_id,
        "gpu": gpu,
        "queue_wait_sec": queue_wait_sec,
        "worker_wall_sec": worker_wall_sec,
        "batch_id": dynamic_batch_id,
        "batch_size": dynamic_batch_size,
        "batch_index": dynamic_batch_index,
    }
    deepstack_execution = _row_deepstack_execution(
        config=config,
        backend_config=backend_config,
        resolved_backend=resolved_backend,
        debug=debug,
    )
    return {
        "sample_id": sample.sample_id,
        "benchmark": sample.benchmark,
        "population_id": sample.population_id,
        "subset_id": config.subset_id,
        "source_file": sample.source_file,
        "num_shards": config.num_shards,
        "shard_index": worker_id,
        "manifest_index": manifest_index,
        "dynamic_worker_id": worker_id,
        "dynamic_gpu": gpu,
        "dynamic_batch_id": dynamic_batch_id,
        "dynamic_batch_size": dynamic_batch_size,
        "dynamic_batch_index": dynamic_batch_index,
        "method": config.mode.value,
        "eval_family": config.eval_family.value,
        "tgvf_protocol": config.tgvf_protocol,
        "post_tgvf_continuation": config.post_tgvf_continuation.value,
        "post_tgvf_forward_mode": config.post_tgvf_forward_mode.value,
        "deepstack": config.deepstack.to_dict(),
        "deepstack_execution": deepstack_execution,
        "parser_scorer": config.parser_scorer.to_dict(),
        "d_condition": backend_config.stage2.d_condition if backend_config.stage2 is not None else None,
        "runner_backend": backend_config.backend,
        "resolved_runner_backend": resolved_backend,
        "runner_backend_final_clean": backend_role["final_clean_backend"],
        "runner_backend_diagnostic_bridge": backend_role["diagnostic_bridge"],
        "runner_backend_deprecated_alias": False,
        "runner_backend_stage2_generic_alias": False,
        "runner_backend_alias_target": (
            resolved_backend if backend_config.backend != resolved_backend else None
        ),
        "question": sample.question,
        "choices": list(sample.choices),
        "gold_answer": sample.gold_answer,
        "raw_output": result.raw_output,
        "final_output": _final_output(result),
        "parsed_answer": "",
        "score": None,
        "answer_parse_success": False,
        "scorer_name": "",
        "official_tool_used": False,
        "official_tool_path": None,
        "official_compatible": False,
        "malformed": bool(result.error),
        "trigger_policy": _trigger_policy(config),
        "trigger_focus_decision": result.triggered,
        "focus_valid": result.focus_valid,
        "focus_target": result.focus_target,
        "append_success": result.append_success,
        "d_shape": _d_shape(debug),
        "continuation_metadata": _continuation_metadata(
            result=result,
            config=config,
            backend_config=backend_config,
            resolved_backend=resolved_backend,
        ),
        "output_tokens": result.output_tokens,
        "wall_time_sec": result.wall_time_sec,
        "metadata": sample.metadata,
        "debug_metadata": debug,
        "error": result.error,
    }


def _write_initial_artifacts(
    output_dir: Path,
    *,
    config: RunConfig,
    manifest: dict[str, Any],
    backend_config: BackendConfig,
    gpus: list[str],
    batch_size: int,
    max_samples: int | None,
    source_sample_count: int,
) -> None:
    source_manifest = write_benchmark_source_manifest(
        output_dir / "benchmark_sources.json",
        config=config,
        manifest=manifest,
    )
    runtime_config = replace(
        config,
        benchmark_source_manifest=benchmark_source_manifest_reference(source_manifest),
        execution_backend={
            **backend_config.to_dict(),
            "dynamic_queue": {
                "schema_version": "clean_dynamic_queue_config_v1",
                "gpus": gpus,
                "batch_size": batch_size,
                "sample_selection": {
                    "rule": "first_n_manifest_rows" if max_samples is not None else "all_manifest_rows",
                    "max_samples": max_samples,
                    "source_sample_count": source_sample_count,
                },
                "partial_rows": "partial_rows.jsonl",
            },
        },
    )
    _write_json(output_dir / "run_config.json", runtime_config)
    _write_json(output_dir / "sample_manifest.json", manifest)
    write_run_config_text(
        output_dir / "run_config.txt",
        config=runtime_config,
        manifest=manifest,
        backend_config=backend_config,
    )
    (output_dir / "partial_rows.jsonl").write_text("", encoding="utf-8")
    _write_json(
        output_dir / "dynamic_config.json",
        {
            "schema_version": "clean_dynamic_queue_config_v1",
            "run_id": config.run_id,
            "runner_backend": backend_config.backend,
            "gpus": gpus,
            "batch_size": batch_size,
            "sample_selection": {
                "rule": "first_n_manifest_rows" if max_samples is not None else "all_manifest_rows",
                "max_samples": max_samples,
                "source_sample_count": source_sample_count,
            },
            "partial_rows": "partial_rows.jsonl",
            "manifest_hash": manifest.get("manifest_hash") or config.manifest_hash,
        },
    )


def _next_task_batch(task_queue: Any, batch_size: int) -> tuple[list[int], bool]:
    indices: list[int] = []
    stop_after_batch = False
    while len(indices) < batch_size:
        try:
            item = task_queue.get(timeout=1) if not indices else task_queue.get_nowait()
        except queue.Empty:
            break
        if item is None:
            stop_after_batch = True
            break
        indices.append(int(item))
    return indices, stop_after_batch


def _timing_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("wall_time_sec",)
    debug_keys = ("preprocess_sec", "h2d_sec", "generate_sec", "decode_sec")
    payload: dict[str, Any] = {}
    for key in keys:
        values = [float(row.get(key) or 0.0) for row in rows]
        payload[key] = _numeric_summary(values)
    for key in debug_keys:
        values = []
        for row in rows:
            timing = ((row.get("debug_metadata") or {}).get("timing") or {})
            if key in timing:
                values.append(float(timing[key] or 0.0))
        payload[key] = _numeric_summary(values)
    payload["output_tokens"] = _numeric_summary(
        [float(row.get("output_tokens") or 0.0) for row in rows]
    )
    return payload


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "min": None, "p50": None, "p90": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "n": len(values),
        "min": ordered[0],
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def _percentile(ordered: list[float], q: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    index = q * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _parse_gpus(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
