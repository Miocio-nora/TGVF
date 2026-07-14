"""Generate verified original-Qwen reasoning for direct Stage2 teacher rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from revisit_vlm_clean.benchmark_data import BenchmarkSample
from revisit_vlm_clean.reasoning_replay import (
    REPLAY_SCHEMA_VERSION,
    SourceRecord,
    build_replay_split,
    load_source_records,
    parse_original_reasoning,
    select_direct_records,
    sha256_file,
    token_statistics,
    validate_generation,
)
from revisit_vlm_clean.rendering import render_benchmark_inputs
from revisit_vlm_clean.runner import BackendConfig, make_backend
from revisit_vlm_clean.schema import EvalMode, ForwardMode, RunConfig


DEFAULT_SOURCE_ROOT = Path(
    "data/tgvf_teacher/generated/runs/"
    "tgvf_v4_teacher_50k_clean_imend_open_answer/splits"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay and verify original Qwen reasoning on direct Stage2 rows."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--source-train",
        default=str(DEFAULT_SOURCE_ROOT / "tgvf_v4_teacher_stage2_protocol_c.train.jsonl"),
    )
    parser.add_argument(
        "--source-test",
        default=str(DEFAULT_SOURCE_ROOT / "tgvf_v4_teacher_stage2_protocol_c.test.jsonl"),
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Thinking")
    parser.add_argument("--processor-id", default=None)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--max-replay-target-tokens",
        "--max-accepted-output-tokens",
        dest="max_replay_target_tokens",
        type=int,
        default=None,
        help="Optional extra cap on serialized <think>reasoning</think> + gold answer.",
    )
    parser.add_argument(
        "--max-sequence-tokens",
        type=int,
        default=2048,
        help="Maximum padded generation prompt plus serialized replay target length.",
    )
    parser.add_argument("--max-image-resolution", type=int, default=512)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--selection-seed", type=int, default=20260714)
    parser.add_argument(
        "--max-direct-per-split",
        type=int,
        default=None,
        help="Deterministic stratified pilot size per split. Omit for every direct row.",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--reuse-generation-rows",
        default=None,
        help="Revalidate an exact existing generation_rows.jsonl without loading a model.",
    )
    parser.add_argument("--publish-splits", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_paths = {
        "train": Path(args.source_train).resolve(),
        "test": Path(args.source_test).resolve(),
    }
    source_records = {
        split: load_source_records(path, split=split)
        for split, path in source_paths.items()
    }
    selected = {
        split: select_direct_records(
            records,
            max_records=args.max_direct_per_split,
            seed=int(args.selection_seed),
        )
        for split, records in source_records.items()
    }
    missing_images = [
        str(record.row.get("image") or "")
        for records in selected.values()
        for record in records
        if not Path(str(record.row.get("image") or "")).exists()
    ]
    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} selected rows have missing images; first={missing_images[0]!r}"
        )

    gpus = [] if args.reuse_generation_rows else _parse_gpus(args.gpus)
    identity = _build_identity(
        args=args,
        source_paths=source_paths,
        source_records=source_records,
        selected=selected,
        gpus=gpus,
    )
    identity_hash = _json_hash(identity)
    run_config = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "identity_sha256": identity_hash,
        "identity": identity,
        "git": _git_identity(),
    }
    _bind_or_write_run_config(output_dir, run_config, resume=bool(args.resume))
    _write_selection_manifest(output_dir / "selection_manifest.jsonl", selected)
    _write_json(
        output_dir / "selection_summary.json",
        _selection_summary(source_records, selected),
    )
    if args.plan_only:
        print(json.dumps({"status": "planned", "output_dir": str(output_dir)}, sort_keys=True))
        return 0

    existing_rows = _load_generation_rows(output_dir / "generation_rows.jsonl")
    selected_list = [record for split in ("train", "test") for record in selected[split]]
    selected_keys = {(record.split, record.uid) for record in selected_list}
    unexpected = sorted(set(existing_rows) - selected_keys)
    if unexpected:
        raise ValueError(f"resume output contains rows outside the bound selection: {unexpected[:3]}")
    if args.reuse_generation_rows and not existing_rows:
        existing_rows = _revalidate_generation_rows(
            source_path=Path(args.reuse_generation_rows).resolve(),
            selected=selected_list,
            generation_max_tokens=int(args.max_new_tokens),
            max_replay_target_tokens=args.max_replay_target_tokens,
            max_sequence_tokens=int(args.max_sequence_tokens),
        )
    pending = [
        record for record in selected_list if (record.split, record.uid) not in existing_rows
    ]

    started = time.perf_counter()
    if pending and args.reuse_generation_rows:
        raise ValueError(
            f"reused generation file is missing {len(pending)} selected rows"
        )
    if pending:
        new_rows = _run_dynamic_generation(
            records=pending,
            output_path=output_dir / "generation_rows.jsonl",
            args=args,
            gpus=gpus,
        )
        for row in new_rows:
            key = (str(row["split"]), str(row["v4_uid"]))
            if key in existing_rows:
                raise ValueError(f"duplicate generated row: {key}")
            existing_rows[key] = row
    generation_rows = [existing_rows[key] for key in sorted(existing_rows, key=_row_key)]
    _rewrite_generation_rows(output_dir / "generation_rows.jsonl", generation_rows)

    missing_keys = sorted(selected_keys - set(existing_rows))
    publish_stats: list[dict[str, Any]] = []
    if args.publish_splits:
        if args.max_direct_per_split is not None:
            raise ValueError("--publish-splits requires all direct rows; omit --max-direct-per-split")
        if missing_keys:
            raise RuntimeError(f"cannot publish with {len(missing_keys)} missing generations")
        split_dir = output_dir / "splits"
        replay_metadata = {
            "run_id": args.run_id,
            "model_id": args.model_id,
            "processor_id": args.processor_id or args.model_id,
            "generation_identity_sha256": identity_hash,
            "max_image_resolution": int(args.max_image_resolution),
            "generation_max_tokens": int(args.max_new_tokens),
            "max_replay_target_tokens": args.max_replay_target_tokens,
            "max_sequence_tokens": int(args.max_sequence_tokens),
        }
        for split in ("train", "test"):
            publish_stats.append(
                build_replay_split(
                    source_path=source_paths[split],
                    output_path=(
                        split_dir / f"tgvf_v4_teacher_stage2_protocol_c.{split}.jsonl"
                    ),
                    split=split,
                    generation_rows=generation_rows,
                    run_metadata=replay_metadata,
                )
            )

    report = _build_report(
        run_id=args.run_id,
        identity_hash=identity_hash,
        source_records=source_records,
        selected=selected,
        generation_rows=generation_rows,
        missing_keys=missing_keys,
        publish_stats=publish_stats,
        elapsed_sec=time.perf_counter() - started,
    )
    _write_json(output_dir / "report.json", report)
    (output_dir / "report.md").write_text(_render_report_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "complete" if not missing_keys else "incomplete",
                "selected": len(selected_keys),
                "generated": len(generation_rows),
                "accepted": report["accepted_rows"],
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )
    return 0 if not missing_keys else 2


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.batch_size) < 1:
        raise ValueError("--batch-size must be >= 1")
    if int(args.max_new_tokens) < 1:
        raise ValueError("--max-new-tokens must be >= 1")
    if args.max_replay_target_tokens is not None and int(args.max_replay_target_tokens) < 1:
        raise ValueError("--max-replay-target-tokens must be positive when provided")
    if int(args.max_sequence_tokens) < 1:
        raise ValueError("--max-sequence-tokens must be positive")
    if args.max_direct_per_split is not None and int(args.max_direct_per_split) < 1:
        raise ValueError("--max-direct-per-split must be >= 1")
    if args.reuse_generation_rows and not Path(args.reuse_generation_rows).exists():
        raise FileNotFoundError(
            f"reused generation rows do not exist: {args.reuse_generation_rows}"
        )
    model_path = Path(args.model_id)
    if model_path.is_absolute() and not model_path.exists():
        raise FileNotFoundError(f"model path does not exist: {model_path}")


def _build_identity(
    *,
    args: argparse.Namespace,
    source_paths: dict[str, Path],
    source_records: dict[str, list[SourceRecord]],
    selected: dict[str, list[SourceRecord]],
    gpus: list[str],
) -> dict[str, Any]:
    cli_path = Path(__file__).resolve()
    core_path = cli_path.parents[1] / "reasoning_replay.py"
    return {
        "implementation": {
            "cli_path": str(cli_path),
            "cli_sha256": sha256_file(cli_path),
            "core_path": str(core_path),
            "core_sha256": sha256_file(core_path),
        },
        "model_id": args.model_id,
        "processor_id": args.processor_id or args.model_id,
        "generation": {
            "mode": (
                "revalidate_existing_original_qwen_generations"
                if args.reuse_generation_rows
                else "original_qwen_thinking"
            ),
            "do_sample": False,
            "max_new_tokens": int(args.max_new_tokens),
            "max_replay_target_tokens": args.max_replay_target_tokens,
            "max_sequence_tokens": int(args.max_sequence_tokens),
            "max_image_resolution": int(args.max_image_resolution),
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "batch_size": int(args.batch_size),
            "gpus": gpus,
            "reused_generation_rows": (
                {
                    "path": str(Path(args.reuse_generation_rows).resolve()),
                    "sha256": sha256_file(args.reuse_generation_rows),
                }
                if args.reuse_generation_rows
                else None
            ),
        },
        "selection": {
            "rule": (
                "all_direct_rows"
                if args.max_direct_per_split is None
                else "deterministic_question_type_then_source_round_robin_hash"
            ),
            "seed": int(args.selection_seed),
            "max_direct_per_split": args.max_direct_per_split,
        },
        "acceptance": {
            "require_natural_stop": True,
            "require_single_think_end": True,
            "reject_tool_protocol_markers": True,
            "answer_match": (
                "normalized_exact_then_first_explicit_candidate_equivalence"
            ),
            "length_limit_surface": (
                "padded_generation_prompt_plus_serialized_replay_target"
            ),
            "rejected_direct_policy": "exclude_from_published_split",
            "focus_policy": "copy_original_jsonl_lines_exactly",
        },
        "output_policy": {
            "publish_splits": bool(args.publish_splits),
            "accepted_direct_only": True,
            "focus_rows_byte_identical": True,
        },
        "sources": {
            split: {
                "path": str(source_paths[split]),
                "sha256": sha256_file(source_paths[split]),
                "rows": len(source_records[split]),
                "direct_rows": sum(record.is_direct for record in source_records[split]),
                "focus_rows": sum(not record.is_direct for record in source_records[split]),
                "selected_direct_rows": len(selected[split]),
                "selected_uid_sha256": _uid_hash(selected[split]),
            }
            for split in ("train", "test")
        },
    }


def _run_dynamic_generation(
    *,
    records: list[SourceRecord],
    output_path: Path,
    args: argparse.Namespace,
    gpus: list[str],
) -> list[dict[str, Any]]:
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    for index in range(len(records)):
        task_queue.put(index)
    for _ in gpus:
        task_queue.put(None)

    workers = []
    worker_config = {
        "run_id": args.run_id,
        "model_id": args.model_id,
        "processor_id": args.processor_id,
        "max_image_resolution": int(args.max_image_resolution),
        "max_new_tokens": int(args.max_new_tokens),
        "max_replay_target_tokens": args.max_replay_target_tokens,
        "max_sequence_tokens": int(args.max_sequence_tokens),
        "batch_size": int(args.batch_size),
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "num_workers": len(gpus),
    }
    for worker_id, gpu in enumerate(gpus):
        process = ctx.Process(
            target=_worker_main,
            kwargs={
                "worker_id": worker_id,
                "gpu": gpu,
                "task_queue": task_queue,
                "result_queue": result_queue,
                "records": records,
                "worker_config": worker_config,
            },
        )
        process.start()
        workers.append(process)

    rows: list[dict[str, Any]] = []
    done_workers = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        while done_workers < len(workers):
            message = result_queue.get()
            if message["type"] == "row":
                row = message["row"]
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                if (
                    len(rows) == 1
                    or len(rows) % max(1, int(args.progress_every)) == 0
                    or len(rows) == len(records)
                ):
                    print(
                        "[reasoning_replay_progress] "
                        f"rows={len(rows)}/{len(records)} accepted="
                        f"{sum(bool(item['validation']['accepted']) for item in rows)} "
                        f"gpu={row['worker_gpu']} uid={row['v4_uid']}",
                        file=sys.stderr,
                        flush=True,
                    )
            elif message["type"] == "worker_done":
                done_workers += 1
            elif message["type"] == "worker_error":
                done_workers += 1
                print(
                    "[reasoning_replay_worker_error] "
                    f"worker={message['worker_id']} gpu={message['gpu']} "
                    f"error={message['error']}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                raise RuntimeError(f"unknown worker message: {message}")

    for process in workers:
        process.join(timeout=30)
        if process.exitcode not in {0, None}:
            print(
                f"[reasoning_replay_worker_exit] pid={process.pid} exitcode={process.exitcode}",
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
    records: list[SourceRecord],
    worker_config: dict[str, Any],
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        config = RunConfig(
            run_id=str(worker_config["run_id"]),
            checkpoint_path=str(worker_config["model_id"]),
            model_id=str(worker_config["model_id"]),
            processor_id=worker_config.get("processor_id"),
            population_id="tgvf_stage2_direct_reasoning_replay",
            benchmark_root=".",
            mode=EvalMode.ORIGINAL,
            post_tgvf_forward_mode=ForwardMode.KV_CACHE,
            num_shards=int(worker_config["num_workers"]),
            shard_index=worker_id,
            max_image_resolution=int(worker_config["max_image_resolution"]),
            max_tokens=int(worker_config["max_new_tokens"]),
            max_answer_tokens=int(worker_config["max_new_tokens"]),
        )
        config.validate()
        backend = make_backend(
            BackendConfig(
                backend="qwen3_original",
                dtype=str(worker_config["dtype"]),
                device="cuda:0",
                device_map="cuda:0",
                attn_implementation=str(worker_config["attn_implementation"]),
                original_no_thinking=False,
            ),
            config=config,
        )
        backend.prepare(config)
        _model, processor = backend._load()
        tokenizer = processor.tokenizer
        batch_id = 0
        while True:
            indices, stop_after_batch = _next_batch(
                task_queue, int(worker_config["batch_size"])
            )
            if not indices and stop_after_batch:
                break
            if not indices:
                continue
            batch_records = [records[index] for index in indices]
            samples = [_benchmark_sample(record) for record in batch_records]
            rendered = render_benchmark_inputs(samples, config)
            started = time.perf_counter()
            results = backend.run_batch(samples, rendered, config)
            batch_wall_sec = time.perf_counter() - started
            for batch_index, (record, result) in enumerate(
                zip(batch_records, results, strict=True)
            ):
                parsed = parse_original_reasoning(result.raw_output)
                reasoning_tokens = None
                replay_target_tokens = None
                if parsed.valid:
                    reasoning_tokens = len(
                        tokenizer.encode(parsed.reasoning, add_special_tokens=False)
                    )
                    replay_target_text = (
                        f"<think>\n{parsed.reasoning}\n</think>\n"
                        f"{str(record.row.get('answer') or '').strip()}<|im_end|>"
                    )
                    replay_target_tokens = len(
                        tokenizer.encode(replay_target_text, add_special_tokens=False)
                    )
                prompt_tokens = (
                    (result.debug or {}).get("batch_generation", {}).get(
                        "shared_prompt_len"
                    )
                )
                replay_sequence_tokens = (
                    int(prompt_tokens) + int(replay_target_tokens)
                    if prompt_tokens is not None and replay_target_tokens is not None
                    else None
                )
                validation = validate_generation(
                    record.row,
                    raw_output=result.raw_output,
                    output_tokens=int(result.output_tokens),
                    generation_max_tokens=int(worker_config["max_new_tokens"]),
                    max_replay_target_tokens=worker_config["max_replay_target_tokens"],
                    max_sequence_tokens=int(worker_config["max_sequence_tokens"]),
                    prompt_tokens=prompt_tokens,
                    replay_target_tokens=replay_target_tokens,
                    error=result.error,
                )
                result_queue.put(
                    {
                        "type": "row",
                        "row": {
                            "schema_version": REPLAY_SCHEMA_VERSION,
                            "split": record.split,
                            "source_index": record.source_index,
                            "source_row_sha256": record.row_sha256,
                            "v4_uid": record.uid,
                            "image_id": record.row.get("image_id"),
                            "source_dataset": record.row.get("source_dataset"),
                            "question_type": record.row.get("question_type"),
                            "question": record.row.get("question"),
                            "gold_answer": record.row.get("answer"),
                            "raw_output": result.raw_output,
                            "output_tokens": int(result.output_tokens),
                            "reasoning_tokens": reasoning_tokens,
                            "replay_target_tokens": replay_target_tokens,
                            "generation_prompt_tokens": prompt_tokens,
                            "replay_sequence_tokens": replay_sequence_tokens,
                            "wall_time_sec": float(result.wall_time_sec),
                            "batch_wall_sec": batch_wall_sec,
                            "worker_id": worker_id,
                            "worker_gpu": gpu,
                            "batch_id": batch_id,
                            "batch_size": len(indices),
                            "batch_index": batch_index,
                            "generation_error": result.error,
                            "generation_debug": result.debug,
                            "validation": validation,
                        },
                    }
                )
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


def _benchmark_sample(record: SourceRecord) -> BenchmarkSample:
    choices = tuple(str(choice) for choice in (record.row.get("choices") or []))
    question = str(record.row.get("question") or "").strip()
    if choices:
        options = "\n".join(
            f"{chr(ord('A') + index)}. {choice}" for index, choice in enumerate(choices)
        )
        question = f"{question}\n{options}"
    return BenchmarkSample(
        sample_id=record.uid,
        benchmark="tgvf_stage2_direct_reasoning_replay",
        population_id=record.split,
        source_file=record.split,
        question=question,
        media=({"kind": "path", "path": str(record.row["image"])},),
        choices=choices,
        gold_answer=str(record.row.get("answer") or ""),
        metadata={
            "source_dataset": record.row.get("source_dataset"),
            "question_type": record.row.get("question_type"),
        },
    )


def _next_batch(task_queue: Any, batch_size: int) -> tuple[list[int], bool]:
    first = task_queue.get()
    if first is None:
        return [], True
    indices = [int(first)]
    stop_after_batch = False
    while len(indices) < batch_size:
        try:
            item = task_queue.get_nowait()
        except queue.Empty:
            break
        if item is None:
            stop_after_batch = True
            break
        indices.append(int(item))
    return indices, stop_after_batch


def _load_generation_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("split") or ""), str(row.get("v4_uid") or ""))
            if not all(key):
                raise ValueError(f"invalid generation row at {path}:{line_number}")
            if key in rows:
                raise ValueError(f"duplicate generation row at {path}:{line_number}: {key}")
            rows[key] = row
    return rows


def _revalidate_generation_rows(
    *,
    source_path: Path,
    selected: list[SourceRecord],
    generation_max_tokens: int,
    max_replay_target_tokens: int | None,
    max_sequence_tokens: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    source_rows = _load_generation_rows(source_path)
    selected_by_key = {(record.split, record.uid): record for record in selected}
    if set(source_rows) != set(selected_by_key):
        missing = sorted(set(selected_by_key) - set(source_rows))
        extra = sorted(set(source_rows) - set(selected_by_key))
        raise ValueError(
            "reused generation selection mismatch: "
            f"missing={missing[:3]} extra={extra[:3]}"
        )
    revalidated: dict[tuple[str, str], dict[str, Any]] = {}
    source_sha256 = sha256_file(source_path)
    for key, old_row in source_rows.items():
        record = selected_by_key[key]
        if old_row.get("source_row_sha256") != record.row_sha256:
            raise ValueError(f"reused generation source-row hash mismatch: {key}")
        row = dict(old_row)
        previous_validation = dict(row.get("validation") or {})
        prompt_tokens = row.get("generation_prompt_tokens")
        if prompt_tokens is None:
            prompt_tokens = (
                (row.get("generation_debug") or {})
                .get("batch_generation", {})
                .get("shared_prompt_len")
            )
        replay_target_tokens = row.get("replay_target_tokens")
        row["generation_prompt_tokens"] = prompt_tokens
        row["replay_sequence_tokens"] = (
            int(prompt_tokens) + int(replay_target_tokens)
            if prompt_tokens is not None and replay_target_tokens is not None
            else None
        )
        row["validation"] = validate_generation(
            record.row,
            raw_output=str(row.get("raw_output") or ""),
            output_tokens=int(row.get("output_tokens") or 0),
            generation_max_tokens=generation_max_tokens,
            max_replay_target_tokens=max_replay_target_tokens,
            max_sequence_tokens=max_sequence_tokens,
            prompt_tokens=(int(prompt_tokens) if prompt_tokens is not None else None),
            replay_target_tokens=(
                int(replay_target_tokens)
                if replay_target_tokens is not None
                else None
            ),
            error=row.get("generation_error"),
        )
        row["revalidation"] = {
            "schema_version": "tgvf_reasoning_replay_revalidation_v1",
            "source_path": str(source_path),
            "source_sha256": source_sha256,
            "previous_accepted": previous_validation.get("accepted"),
            "previous_reason": previous_validation.get("reason"),
        }
        revalidated[key] = row
    return revalidated


def _rewrite_generation_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _build_report(
    *,
    run_id: str,
    identity_hash: str,
    source_records: dict[str, list[SourceRecord]],
    selected: dict[str, list[SourceRecord]],
    generation_rows: list[dict[str, Any]],
    missing_keys: list[tuple[str, str]],
    publish_stats: list[dict[str, Any]],
    elapsed_sec: float,
) -> dict[str, Any]:
    accepted = [row for row in generation_rows if row["validation"]["accepted"]]
    rejected = [row for row in generation_rows if not row["validation"]["accepted"]]
    by_split: dict[str, Any] = {}
    for split in ("train", "test"):
        split_rows = [row for row in generation_rows if row["split"] == split]
        split_accepted = [row for row in split_rows if row["validation"]["accepted"]]
        by_split[split] = {
            "source_rows": len(source_records[split]),
            "source_direct_rows": sum(record.is_direct for record in source_records[split]),
            "source_focus_rows": sum(not record.is_direct for record in source_records[split]),
            "selected_direct_rows": len(selected[split]),
            "generated_rows": len(split_rows),
            "accepted_rows": len(split_accepted),
            "acceptance_rate": (
                len(split_accepted) / len(split_rows) if split_rows else None
            ),
            "accepted_output_tokens": token_statistics(
                row["output_tokens"] for row in split_accepted
            ),
            "accepted_replay_target_tokens": token_statistics(
                row["replay_target_tokens"]
                for row in split_accepted
                if row.get("replay_target_tokens") is not None
            ),
            "accepted_replay_sequence_tokens": token_statistics(
                row["replay_sequence_tokens"]
                for row in split_accepted
                if row.get("replay_sequence_tokens") is not None
            ),
            "rejection_reasons": dict(
                sorted(
                    Counter(
                        row["validation"]["reason"]
                        for row in split_rows
                        if not row["validation"]["accepted"]
                    ).items()
                )
            ),
        }
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "run_id": run_id,
        "identity_sha256": identity_hash,
        "elapsed_sec": elapsed_sec,
        "selected_rows": sum(len(rows) for rows in selected.values()),
        "generated_rows": len(generation_rows),
        "missing_rows": len(missing_keys),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "acceptance_rate": len(accepted) / len(generation_rows) if generation_rows else None,
        "all_output_tokens": token_statistics(row["output_tokens"] for row in generation_rows),
        "accepted_output_tokens": token_statistics(row["output_tokens"] for row in accepted),
        "accepted_reasoning_tokens": token_statistics(
            row["reasoning_tokens"]
            for row in accepted
            if row.get("reasoning_tokens") is not None
        ),
        "accepted_replay_target_tokens": token_statistics(
            row["replay_target_tokens"]
            for row in accepted
            if row.get("replay_target_tokens") is not None
        ),
        "accepted_replay_sequence_tokens": token_statistics(
            row["replay_sequence_tokens"]
            for row in accepted
            if row.get("replay_sequence_tokens") is not None
        ),
        "accepted_match_types": dict(
            sorted(
                Counter(
                    str(row["validation"].get("match_type") or "unknown")
                    for row in accepted
                ).items()
            )
        ),
        "rejection_reasons": dict(
            sorted(Counter(row["validation"]["reason"] for row in rejected).items())
        ),
        "by_split": by_split,
        "accepted_by_question_type": dict(
            sorted(Counter(str(row.get("question_type") or "unknown") for row in accepted).items())
        ),
        "accepted_by_source_dataset": dict(
            sorted(Counter(str(row.get("source_dataset") or "unknown") for row in accepted).items())
        ),
        "publish_stats": publish_stats,
    }


def _render_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Direct Original-Reasoning Replay Report",
        "",
        f"- Run: `{report['run_id']}`",
        f"- Identity: `{report['identity_sha256']}`",
        f"- Selected/generated: `{report['selected_rows']}` / `{report['generated_rows']}`",
        f"- Accepted/rejected: `{report['accepted_rows']}` / `{report['rejected_rows']}`",
        f"- Acceptance rate: `{_format_percent(report['acceptance_rate'])}`",
        "",
        "## Split Summary",
        "",
        "| Split | Source direct | Selected | Generated | Accepted | Acceptance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "test"):
        item = report["by_split"][split]
        lines.append(
            f"| {split} | {item['source_direct_rows']} | {item['selected_direct_rows']} | "
            f"{item['generated_rows']} | {item['accepted_rows']} | "
            f"{_format_percent(item['acceptance_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Accepted Replay-Sequence Token Lengths",
            "",
            "| Count | Mean | Median | P90 | P95 | P99 | Max |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    stats = report["accepted_replay_sequence_tokens"]
    lines.append(
        f"| {stats['count']} | {_format_number(stats['mean'])} | "
        f"{_format_number(stats['median'])} | {_format_number(stats['p90'])} | "
        f"{_format_number(stats['p95'])} | {_format_number(stats['p99'])} | "
        f"{_format_number(stats['max'])} |"
    )
    lines.extend(["", "## Rejections", ""])
    if report["rejection_reasons"]:
        lines.extend(["| Reason | Rows |", "|---|---:|"])
        for reason, count in report["rejection_reasons"].items():
            lines.append(f"| `{reason}` | {count} |")
    else:
        lines.append("No rejected rows.")
    if report["publish_stats"]:
        lines.extend(["", "## Published Splits", ""])
        for item in report["publish_stats"]:
            lines.append(
                f"- `{item['split']}`: `{item['written_rows']}` rows, "
                f"`{item['written_direct_rows']}` replayed direct rows; focus payload "
                f"identity `{item['focus_payload_sha256']}`."
            )
    lines.append("")
    return "\n".join(lines)


def _selection_summary(
    source_records: dict[str, list[SourceRecord]],
    selected: dict[str, list[SourceRecord]],
) -> dict[str, Any]:
    return {
        split: {
            "source_rows": len(source_records[split]),
            "source_direct_rows": sum(record.is_direct for record in source_records[split]),
            "source_focus_rows": sum(not record.is_direct for record in source_records[split]),
            "selected_direct_rows": len(selected[split]),
            "selected_by_question_type": dict(
                sorted(
                    Counter(
                        str(record.row.get("question_type") or "unknown")
                        for record in selected[split]
                    ).items()
                )
            ),
            "selected_by_source_dataset": dict(
                sorted(
                    Counter(
                        str(record.row.get("source_dataset") or "unknown")
                        for record in selected[split]
                    ).items()
                )
            ),
        }
        for split in ("train", "test")
    }


def _write_selection_manifest(path: Path, selected: dict[str, list[SourceRecord]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for split in ("train", "test"):
            for selection_index, record in enumerate(selected[split]):
                handle.write(
                    json.dumps(
                        {
                            "split": split,
                            "selection_index": selection_index,
                            "source_index": record.source_index,
                            "source_row_sha256": record.row_sha256,
                            "v4_uid": record.uid,
                            "image_id": record.row.get("image_id"),
                            "source_dataset": record.row.get("source_dataset"),
                            "question_type": record.row.get("question_type"),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )


def _bind_or_write_run_config(
    output_dir: Path, run_config: dict[str, Any], *, resume: bool
) -> None:
    path = output_dir / "run_config.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("identity_sha256") != run_config["identity_sha256"]:
            raise ValueError(
                "output directory is bound to a different generation identity: "
                f"{existing.get('identity_sha256')} != {run_config['identity_sha256']}"
            )
        if not resume:
            raise FileExistsError(f"output already exists and --no-resume was requested: {path}")
        return
    _write_json(path, run_config)
    (output_dir / "run_config.txt").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _uid_hash(records: list[SourceRecord]) -> str:
    payload = "\n".join(record.uid for record in records) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_hash(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _row_key(key: tuple[str, str]) -> tuple[int, str]:
    return (0 if key[0] == "train" else 1, key[1])


def _parse_gpus(value: str) -> list[str]:
    gpus = [item.strip() for item in str(value).split(",") if item.strip()]
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("--gpus must contain unique comma-separated ids")
    return gpus


def _git_identity() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty_worktree": dirty}


def _format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def _format_number(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
