#!/usr/bin/env python3
"""Run isolated TGVF inference timing tests.

This script intentionally does not change the normal clean benchmark runtime.
It builds a small instrumented NativeStage2Engine subclass for latency tests
only, with an optional cached-feature prewarm branch and a fresh vision
reencode branch.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from revisit_vlm_clean.benchmark_data import (
    BenchmarkSample,
    load_manifest_payload,
    materialize_samples_from_manifest_payload,
)
from revisit_vlm_clean.rendering import RenderedBenchmarkInput, render_benchmark_inputs
from revisit_vlm_clean.runner import BackendConfig, Qwen3OriginalBackend
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
from revisit_vlm_clean.stage2_native import NativeStage2Engine, NativeStage2RunResult
from revisit_vlm_clean.stage2_runtime import Stage2RuntimeConfig, checkpoint_identity


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking"
DEFAULT_BENCHMARK_ROOT = "/home/dredvpn009/Flash_Storage/datasets/benchmarks"
DEFAULT_MANIFEST = "revisit_vlm_clean/benchmark_manifests/core_balanced_dev_2511_seed20260625.json"
DEFAULT_STAGE2_CKPT = (
    "outputs/clean_training/"
    "qwen3_stage2_ddeepstack_norm01_from_stage1_ddeepstack_8gpu_20260703_005210/"
    "stage2_micro4/clean_training_execution/checkpoint_step_1200.pt"
)
DEFAULT_STAGE2_EVAL_JSONL = (
    "data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/"
    "splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl"
)


class TimingNativeStage2Engine(NativeStage2Engine):
    """Instrumented engine used only by this timing script."""

    def __init__(
        self,
        *,
        stage2_config: Stage2RuntimeConfig,
        backend_options: dict[str, Any] | None = None,
        timing_variant: str = "current_runtime",
    ) -> None:
        super().__init__(stage2_config=stage2_config, backend_options=backend_options)
        self.timing_variant = timing_variant
        self._active_timing: dict[str, Any] = {}

    def run(
        self,
        sample: BenchmarkSample,
        rendered: RenderedBenchmarkInput,
        config: RunConfig,
    ) -> NativeStage2RunResult:
        self._active_timing = {
            "schema_version": "tgvf_inference_time_test_phase_timing_v1",
            "timing_variant": self.timing_variant,
            "cuda_synchronized": True,
        }
        result = super().run(sample, rendered, config)
        debug = dict(result.debug or {})
        debug["timing_test"] = dict(self._active_timing)
        return replace(result, debug=debug)

    def _run_force(self, sample: Any) -> NativeStage2RunResult:
        self._ensure_loaded(sample)
        if self.timing_variant == "cached_prewarm":
            self._prewarm_vision_features(sample)
        return super()._run_force(sample)

    def _run_free(self, sample: Any) -> NativeStage2RunResult:
        self._ensure_loaded(sample)
        if self.timing_variant == "cached_prewarm":
            self._prewarm_vision_features(sample)
        return super()._run_free(sample)

    def _prewarm_vision_features(self, sample: Any) -> None:
        with self._timed("prewarm_vision_features_sec"):
            super()._vision_features(sample)

    def _capture_generated_focus(self, sample: Any, *, force_prefix: bool) -> Any:
        with self._timed("focus_capture_sec"):
            return super()._capture_generated_focus(sample, force_prefix=force_prefix)

    def _capture_free_router(self, sample: Any) -> Any:
        with self._timed("focus_capture_sec"):
            return super()._capture_free_router(sample)

    def _parse_action(self, text: str) -> Any:
        with self._timed("parse_action_sec"):
            return super()._parse_action(text)

    def _d_from_capture(self, sample: Any, capture: Any, *, focus_source: str) -> Any:
        with self._timed("d_from_capture_total_sec"):
            return super()._d_from_capture(sample, capture, focus_source=focus_source)

    def _vision_features(self, sample: Any) -> Any:
        cache_hit = self._vision_feature_cache_hit(sample)
        if self.timing_variant == "reencode_after_focus":
            self.vision_cache.pop(self._vision_feature_cache_key(sample), None)
            cache_hit = False
        phase = "cached_vision_lookup_sec" if cache_hit else "reencode_vision_tap_sec"
        with self._timed(phase):
            result = super()._vision_features(sample)
        self._active_timing.setdefault("vision_feature_cache_hits", 0)
        self._active_timing.setdefault("vision_feature_cache_misses", 0)
        if cache_hit:
            self._active_timing["vision_feature_cache_hits"] += 1
        else:
            self._active_timing["vision_feature_cache_misses"] += 1
        return result

    def _append_visual_d(self, capture: Any, d: Any) -> Any:
        with self._timed("append_visual_d_sec"):
            return super()._append_visual_d(capture, d)

    def _append_visual_d_full_sequence(self, sample: Any, capture: Any, d: Any) -> Any:
        with self._timed("append_visual_d_full_sequence_sec"):
            return super()._append_visual_d_full_sequence(sample, capture, d)

    def _continue_generation(self, append_result: Any, *, capture: Any | None = None) -> Any:
        with self._timed("post_d_continue_sec"):
            return super()._continue_generation(append_result, capture=capture)

    def _vision_feature_cache_key(self, sample: Any) -> str:
        from revisit_vlm_clean.stage2_native import _image_identity_text

        config = self._run_config
        if config is None:
            raise RuntimeError("NativeStage2Engine.prepare must be called before vision cache key")
        return f"{_image_identity_text(sample.image)}|{config.max_image_resolution}"

    def _vision_feature_cache_hit(self, sample: Any) -> bool:
        try:
            return self._vision_feature_cache_key(sample) in self.vision_cache
        except Exception:
            return False

    @contextmanager
    def _timed(self, key: str) -> Any:
        _sync_cuda()
        started = time.perf_counter()
        try:
            yield
        finally:
            _sync_cuda()
            elapsed = time.perf_counter() - started
            self._active_timing[key] = float(self._active_timing.get(key, 0.0)) + elapsed
            self._active_timing[f"{key}_calls"] = int(
                self._active_timing.get(f"{key}_calls", 0)
            ) + 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isolated TGVF inference time test.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST)
    parser.add_argument("--benchmark-root", default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--stage2-checkpoint", default=DEFAULT_STAGE2_CKPT)
    parser.add_argument("--stage2-eval-jsonl", default=DEFAULT_STAGE2_EVAL_JSONL)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--processor-id", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--max-image-resolution", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--tgvf-protocol", default="protocol_c_tool_observation")
    parser.add_argument("--softforce-prompt-text", default="Use focus tool.")
    parser.add_argument(
        "--methods",
        default=(
            "original,"
            "tgvf_cached_prewarm_force,tgvf_reencode_force,"
            "tgvf_cached_prewarm_free,tgvf_reencode_free,"
            "tgvf_cached_prewarm_softforce,tgvf_reencode_softforce"
        ),
        help="Comma-separated timing methods.",
    )
    parser.add_argument(
        "--rows-per-benchmark",
        type=int,
        default=1,
        help=(
            "Deterministic first-N rows per benchmark. Default 1 keeps the timing "
            "test cheap; use larger values only when a stabler estimate is needed."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--warmup-samples", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or f"tgvf_inference_time_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest_payload(args.manifest_path)
    manifest_hash = str(manifest.get("manifest_hash") or "")
    all_samples = materialize_samples_from_manifest_payload(
        manifest,
        benchmark_root=args.benchmark_root,
        metadata_only=False,
    )
    selected = _select_samples(
        all_samples,
        rows_per_benchmark=int(args.rows_per_benchmark),
        max_samples=args.max_samples,
    )
    warmup = selected[: max(0, int(args.warmup_samples))]
    measured = list(selected)
    methods = _parse_methods(args.methods)

    run_identity = {
        "schema_version": "tgvf_inference_time_test_run_identity_v1",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": _git_identity(),
        "manifest_path": args.manifest_path,
        "manifest_hash": manifest_hash,
        "benchmark_root": args.benchmark_root,
        "source_sample_count": len(all_samples),
        "selected_sample_count": len(selected),
        "warmup_sample_count": len(warmup),
        "measured_sample_count": len(measured),
        "selection_rule": {
            "rows_per_benchmark": int(args.rows_per_benchmark),
            "max_samples": args.max_samples,
            "order": "manifest_order_first_n_per_benchmark",
        },
        "methods": methods,
        "model_id": args.model_id,
        "processor_id": args.processor_id,
        "device": args.device,
        "device_map": args.device_map,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "max_image_resolution": int(args.max_image_resolution),
        "max_tokens": int(args.max_tokens),
        "stage2_checkpoint_identity": checkpoint_identity(args.stage2_checkpoint),
        "stage2_eval_jsonl": args.stage2_eval_jsonl,
        "timing_note": (
            "tgvf_cached_prewarm excludes the synthetic vision prewarm from row wall time; "
            "tgvf_reencode forces a fresh post-focus Qwen3 vision tap before D construction."
        ),
    }
    _write_json(output_dir / "timing_run_identity.json", run_identity)
    _write_json(output_dir / "sample_manifest.json", [s.to_materialized_row() for s in selected])
    _write_text_config(output_dir / "run_config.txt", run_identity)

    rows: list[dict[str, Any]] = []
    rows_path = output_dir / "timing_rows.jsonl"
    rows_path.write_text("", encoding="utf-8")
    for method in methods:
        method_rows = _run_method(
            method=method,
            args=args,
            run_id=run_id,
            manifest_hash=manifest_hash,
            warmup_samples=warmup,
            measured_samples=measured,
            output_dir=output_dir,
            progress_every=max(1, int(args.progress_every)),
        )
        rows.extend(method_rows)
        with rows_path.open("a", encoding="utf-8") as handle:
            for row in method_rows:
                handle.write(json.dumps(_to_jsonable(row), sort_keys=True) + "\n")
        _collect_cuda()

    summary = _summarize_rows(rows)
    _write_json(output_dir / "timing_summary.json", summary)
    print(json.dumps({"rows": str(rows_path), "summary": str(output_dir / "timing_summary.json")}))
    return 0


def _run_method(
    *,
    method: str,
    args: argparse.Namespace,
    run_id: str,
    manifest_hash: str,
    warmup_samples: list[BenchmarkSample],
    measured_samples: list[BenchmarkSample],
    output_dir: Path,
    progress_every: int,
) -> list[dict[str, Any]]:
    parsed = _method_config(method)
    config = _run_config_for_method(
        args=args,
        run_id=run_id,
        manifest_hash=manifest_hash,
        mode=parsed["mode"],
    )
    warmup_rendered = render_benchmark_inputs(warmup_samples, config)
    measured_rendered = render_benchmark_inputs(measured_samples, config)
    if parsed["kind"] == "original":
        backend = Qwen3OriginalBackend(
            model_id=args.model_id,
            processor_id=args.processor_id,
            backend_config=BackendConfig(
                backend="qwen3_original",
                dtype=args.dtype,
                device=args.device,
                device_map=None if args.device_map in {"none", "None", ""} else args.device_map,
                attn_implementation=args.attn_implementation,
            ),
            max_image_resolution=int(args.max_image_resolution),
            max_answer_tokens=int(args.max_tokens),
        )
        backend.prepare(config)
        runner = lambda sample, rendered: backend.run(sample, rendered, config)
        cleanup = lambda: None
    else:
        stage2_config = Stage2RuntimeConfig(
            stage2_checkpoint=args.stage2_checkpoint,
            eval_jsonl=args.stage2_eval_jsonl,
            protocol=args.tgvf_protocol,
            d_condition="correct_D",
            force_prefix_mode="target_hint",
            append_forward_mode=ForwardMode.KV_CACHE,
            max_tokens=int(args.max_tokens),
        )
        engine = TimingNativeStage2Engine(
            stage2_config=stage2_config,
            backend_options={
                "dtype": args.dtype,
                "device": args.device,
                "device_map": None if args.device_map in {"none", "None", ""} else args.device_map,
                "attn_implementation": args.attn_implementation,
                "trust_remote_code": True,
            },
            timing_variant=parsed["timing_variant"],
        )
        engine.prepare(config)

        def runner(sample: BenchmarkSample, rendered: RenderedBenchmarkInput) -> Any:
            return engine.run(sample, rendered, config)

        cleanup = engine.cleanup_after_row

    for sample, rendered in zip(warmup_samples, warmup_rendered, strict=True):
        _ = runner(sample, rendered)
        cleanup()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, (sample, rendered) in enumerate(
        zip(measured_samples, measured_rendered, strict=True)
    ):
        row_started = time.perf_counter()
        result = runner(sample, rendered)
        row = _row_from_result(
            method=method,
            timing_variant=parsed.get("timing_variant"),
            sample=sample,
            rendered=rendered,
            result=result,
            measured_index=index,
            row_wall_sec=time.perf_counter() - row_started,
        )
        rows.append(row)
        cleanup()
        if (index + 1) == 1 or (index + 1) % progress_every == 0 or (index + 1) == len(measured_samples):
            print(
                (
                    "[tgvf_time_progress] "
                    f"method={method} rows={index + 1}/{len(measured_samples)} "
                    f"benchmark={sample.benchmark} sample_id={sample.sample_id} "
                    f"elapsed={time.perf_counter() - started:.2f}s"
                ),
                flush=True,
            )
    del runner
    _collect_cuda()
    return rows


def _row_from_result(
    *,
    method: str,
    timing_variant: str | None,
    sample: BenchmarkSample,
    rendered: RenderedBenchmarkInput,
    result: Any,
    measured_index: int,
    row_wall_sec: float,
) -> dict[str, Any]:
    debug = dict(getattr(result, "debug", {}) or {})
    timing = dict(debug.get("timing_test") or debug.get("timing") or {})
    token_budget = debug.get("token_budget") or {}
    wall_time = getattr(result, "wall_time_sec", None)
    if not isinstance(wall_time, (int, float)) or float(wall_time) == 0.0:
        wall_time = debug.get("wall_time_sec")
    return {
        "schema_version": "tgvf_inference_time_test_row_v1",
        "method": method,
        "mode": rendered.mode.value,
        "timing_variant": timing_variant,
        "measured_index": measured_index,
        "sample_id": sample.sample_id,
        "benchmark": sample.benchmark,
        "population_id": sample.population_id,
        "source_file": sample.source_file,
        "question": sample.question,
        "gold_answer": sample.gold_answer,
        "trigger_focus_decision": bool(getattr(result, "triggered", False)),
        "focus_valid": getattr(result, "focus_valid", None),
        "focus_target": getattr(result, "focus_target", ""),
        "append_success": getattr(result, "append_success", None),
        "answer_parse_success": debug.get("answer_parse_success"),
        "malformed": bool(debug.get("malformed") or getattr(result, "error", None)),
        "output_tokens": int(getattr(result, "output_tokens", 0) or 0),
        "action_tokens": token_budget.get("action_tokens"),
        "answer_tokens": token_budget.get("answer_tokens"),
        "wall_time_sec": float(wall_time or 0.0),
        "outer_row_wall_sec": float(row_wall_sec),
        "timing": timing,
        "d_shape": debug.get("D_shape") or debug.get("fvt_shape"),
        "mask_mode": debug.get("mask_mode"),
        "uses_deepstack_for_fvt": debug.get("uses_deepstack_for_fvt"),
        "second_full_forward_used": debug.get("second_full_forward_used"),
        "error": getattr(result, "error", None),
    }


def _method_config(method: str) -> dict[str, Any]:
    if method == "original":
        return {"kind": "original", "mode": EvalMode.ORIGINAL, "timing_variant": None}
    table = {
        "tgvf_cached_prewarm_force": (EvalMode.TGVF_FORCE, "cached_prewarm"),
        "tgvf_reencode_force": (EvalMode.TGVF_FORCE, "reencode_after_focus"),
        "tgvf_current_force": (EvalMode.TGVF_FORCE, "current_runtime"),
        "tgvf_cached_prewarm_free": (EvalMode.TGVF_FREE, "cached_prewarm"),
        "tgvf_reencode_free": (EvalMode.TGVF_FREE, "reencode_after_focus"),
        "tgvf_current_free": (EvalMode.TGVF_FREE, "current_runtime"),
        "tgvf_cached_prewarm_softforce": (EvalMode.TGVF_SOFTFORCE, "cached_prewarm"),
        "tgvf_reencode_softforce": (EvalMode.TGVF_SOFTFORCE, "reencode_after_focus"),
        "tgvf_current_softforce": (EvalMode.TGVF_SOFTFORCE, "current_runtime"),
    }
    if method not in table:
        raise ValueError(f"unknown timing method: {method}")
    mode, variant = table[method]
    return {"kind": "tgvf", "mode": mode, "timing_variant": variant}


def _run_config_for_method(
    *,
    args: argparse.Namespace,
    run_id: str,
    manifest_hash: str,
    mode: EvalMode,
) -> RunConfig:
    return RunConfig(
        run_id=f"{run_id}_{mode.value}",
        checkpoint_path=args.stage2_checkpoint,
        started_at=datetime.now(timezone.utc).isoformat(),
        eval_family=EvalFamily.PROJECT_NATIVE_EXTERNAL,
        mode=mode,
        model_id=args.model_id,
        processor_id=args.processor_id,
        subset_id=Path(args.manifest_path).stem,
        manifest_path=args.manifest_path,
        manifest_hash=manifest_hash,
        benchmark_root=args.benchmark_root,
        max_image_resolution=int(args.max_image_resolution),
        max_tokens=int(args.max_tokens),
        tgvf_protocol=args.tgvf_protocol,
        post_tgvf_forward_mode=ForwardMode.KV_CACHE,
        softforce_prompt_text=args.softforce_prompt_text if mode == EvalMode.TGVF_SOFTFORCE else "",
        deepstack=DeepStackState(
            enabled=mode != EvalMode.ORIGINAL,
            original_image_scope=DeepStackScope.NO_BLOCK if mode != EvalMode.ORIGINAL else DeepStackScope.OFF,
            d_features_enabled=mode != EvalMode.ORIGINAL,
        ),
        parser_scorer=ParserScorerIdentity(
            scoring_backend=ScoringBackend.AUTO,
            fallback_allowed=True,
        ),
    )


def _select_samples(
    samples: list[BenchmarkSample],
    *,
    rows_per_benchmark: int,
    max_samples: int | None,
) -> list[BenchmarkSample]:
    if max_samples is not None:
        return samples[: int(max_samples)]
    selected: list[BenchmarkSample] = []
    counts: dict[str, int] = {}
    for sample in samples:
        count = counts.get(sample.benchmark, 0)
        if count >= rows_per_benchmark:
            continue
        selected.append(sample)
        counts[sample.benchmark] = count + 1
    return selected


def _parse_methods(raw: str) -> list[str]:
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    if not methods:
        raise ValueError("--methods must include at least one method")
    for method in methods:
        _method_config(method)
    return methods


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    by_benchmark_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)
        key = f"{row['benchmark']}::{row['method']}"
        by_benchmark_method.setdefault(key, []).append(row)
    return {
        "schema_version": "tgvf_inference_time_test_summary_v1",
        "n_rows": len(rows),
        "by_method": {
            method: _summary_for_group(method_rows)
            for method, method_rows in sorted(by_method.items())
        },
        "by_benchmark_method": {
            key: _summary_for_group(group_rows)
            for key, group_rows in sorted(by_benchmark_method.items())
        },
    }


def _summary_for_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    triggered = [row for row in rows if row.get("trigger_focus_decision")]
    timing_keys = sorted(
        {
            key
            for row in rows
            for key, value in (row.get("timing") or {}).items()
            if isinstance(value, (int, float)) and not key.endswith("_calls")
        }
    )
    return {
        "n": len(rows),
        "triggered_n": len(triggered),
        "trigger_rate": len(triggered) / len(rows) if rows else None,
        "append_success_rate": _bool_rate(
            [row.get("append_success") for row in rows if row.get("append_success") is not None]
        ),
        "parse_rate": _bool_rate(
            [
                row.get("answer_parse_success")
                for row in rows
                if row.get("answer_parse_success") is not None
            ]
        ),
        "wall_time_sec": _numeric_summary([row.get("wall_time_sec") for row in rows]),
        "outer_row_wall_sec": _numeric_summary([row.get("outer_row_wall_sec") for row in rows]),
        "output_tokens": _numeric_summary([row.get("output_tokens") for row in rows]),
        "action_tokens": _numeric_summary([row.get("action_tokens") for row in rows]),
        "answer_tokens": _numeric_summary([row.get("answer_tokens") for row in rows]),
        "triggered_wall_time_sec": _numeric_summary(
            [row.get("wall_time_sec") for row in triggered]
        ),
        "timing": {
            key: _numeric_summary([(row.get("timing") or {}).get(key) for row in rows])
            for key in timing_keys
        },
        "triggered_timing": {
            key: _numeric_summary([(row.get("timing") or {}).get(key) for row in triggered])
            for key in timing_keys
        },
    }


def _numeric_summary(values: list[Any]) -> dict[str, Any]:
    nums = [float(value) for value in values if isinstance(value, (int, float))]
    if not nums:
        return {"n": 0, "mean": None, "min": None, "p50": None, "p90": None, "max": None}
    ordered = sorted(nums)
    return {
        "n": len(ordered),
        "mean": statistics.fmean(ordered),
        "min": ordered[0],
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "max": ordered[-1],
    }


def _percentile(ordered: list[float], q: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    index = q * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bool_rate(values: list[Any]) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if bool(value)) / len(values)


def _sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def _collect_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _git_identity() -> dict[str, Any]:
    def run(cmd: list[str]) -> str:
        return subprocess.check_output(cmd, cwd=REPO_ROOT, text=True).strip()

    try:
        commit = run(["git", "rev-parse", "HEAD"])
    except Exception:
        commit = None
    try:
        dirty = bool(run(["git", "status", "--porcelain"]))
    except Exception:
        dirty = None
    return {"commit": commit, "dirty_worktree": dirty}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _write_text_config(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# TGVF inference time test", ""]
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            lines.append(f"{key}: {json.dumps(_to_jsonable(value), sort_keys=True)}")
        else:
            lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
