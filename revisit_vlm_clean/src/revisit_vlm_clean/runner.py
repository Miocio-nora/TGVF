"""Executable clean benchmark runner backends."""

from __future__ import annotations

import base64
import io
import json
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .benchmark_data import BenchmarkSample
from .deepstack import deepstack_scope_contract
from .rendering import RenderedBenchmarkInput
from .schema import EvalFamily, EvalMode, EvalSummary, RunConfig
from .scoring import score_output_rows
from .stage2_native import (
    NativeStage2Engine,
    NativeStage2RunResult,
)
from .stage2_runtime import Stage2RuntimeConfig

STAGE2_NATIVE_BACKEND = "tgvf_stage2_qwen3_native"
STAGE2_LEGACY_BACKEND = "tgvf_stage2_qwen3_legacy"
STAGE2_GENERIC_BACKEND = "tgvf_stage2_qwen3"
RUNNER_BACKENDS = (
    "dry_run",
    "qwen3_original",
    STAGE2_GENERIC_BACKEND,
    STAGE2_NATIVE_BACKEND,
    STAGE2_LEGACY_BACKEND,
)


@dataclass(frozen=True)
class BackendConfig:
    backend: str = "dry_run"
    dtype: str = "bfloat16"
    device: str = "auto"
    device_map: str | None = "auto"
    attn_implementation: str | None = "sdpa"
    trust_remote_code: bool = True
    stage2: Stage2RuntimeConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        resolved_backend = resolve_backend_name(self.backend)
        return {
            "backend": self.backend,
            "resolved_backend": resolved_backend,
            "stage2_generic_alias": self.backend == STAGE2_GENERIC_BACKEND,
            "alias_target": resolved_backend if self.backend != resolved_backend else None,
            "deprecated_alias": False,
            "dtype": self.dtype,
            "device": self.device,
            "device_map": self.device_map,
            "attn_implementation": self.attn_implementation,
            "trust_remote_code": self.trust_remote_code,
            "stage2": None if self.stage2 is None else self.stage2.to_dict(),
        }


@dataclass(frozen=True)
class ModelRunResult:
    raw_output: str
    triggered: bool = False
    focus_target: str = ""
    focus_valid: bool | None = None
    append_success: bool | None = None
    output_tokens: int = 0
    wall_time_sec: float = 0.0
    error: str | None = None
    debug: dict[str, Any] = field(default_factory=dict)


class CleanRunnerBackend:
    def prepare(self, config: RunConfig) -> None:
        del config

    def run(
        self,
        sample: BenchmarkSample,
        rendered: RenderedBenchmarkInput,
        config: RunConfig,
    ) -> ModelRunResult:
        raise NotImplementedError


class DryRunBackend(CleanRunnerBackend):
    def run(
        self,
        sample: BenchmarkSample,
        rendered: RenderedBenchmarkInput,
        config: RunConfig,
    ) -> ModelRunResult:
        del rendered, config
        started = time.perf_counter()
        output = _dry_output(sample)
        return ModelRunResult(
            raw_output=output,
            triggered=False,
            wall_time_sec=time.perf_counter() - started,
            debug={"dry_run": True},
        )


class Qwen3OriginalBackend(CleanRunnerBackend):
    def __init__(
        self,
        *,
        model_id: str,
        processor_id: str | None,
        backend_config: BackendConfig,
        max_image_resolution: int,
        max_answer_tokens: int,
    ) -> None:
        self.model_id = model_id
        self.processor_id = processor_id
        self.backend_config = backend_config
        self.max_image_resolution = max_image_resolution
        self.max_answer_tokens = max_answer_tokens
        self._loaded: tuple[Any, Any] | None = None

    def prepare(self, config: RunConfig) -> None:
        if config.mode != EvalMode.ORIGINAL:
            raise NotImplementedError(
                "qwen3_original backend currently supports only mode='original'"
            )
        self._load()

    def run(
        self,
        sample: BenchmarkSample,
        rendered: RenderedBenchmarkInput,
        config: RunConfig,
    ) -> ModelRunResult:
        started = time.perf_counter()
        try:
            if config.mode != EvalMode.ORIGINAL:
                raise NotImplementedError(
                    "qwen3_original backend currently supports only mode='original'"
                )
            model, processor = self._load()
            messages = [
                {
                    "role": "user",
                    "content": [
                        *self._vision_content_items(sample.media),
                        {"type": "text", "text": rendered.user_prompt},
                    ],
                }
            ]
            inputs = self._build_inputs(processor, messages)
            device = _resolve_generation_device(model, self.backend_config.device)
            inputs = _move_tensors(inputs, device)
            prompt_len = int(inputs["input_ids"].shape[-1])
            generated = model.generate(
                **inputs,
                max_new_tokens=self.max_answer_tokens,
                do_sample=False,
            )
            new_ids = generated[0, prompt_len:].detach().cpu().tolist()
            raw = processor.tokenizer.decode(
                new_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return ModelRunResult(
                raw_output=raw,
                output_tokens=len(new_ids),
                wall_time_sec=time.perf_counter() - started,
            )
        except Exception as exc:
            return ModelRunResult(
                raw_output="",
                wall_time_sec=time.perf_counter() - started,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _load(self) -> tuple[Any, Any]:
        if self._loaded is not None:
            return self._loaded
        import torch
        from transformers import (
            AutoConfig,
            AutoProcessor,
            Qwen2VLForConditionalGeneration,
            Qwen3VLForConditionalGeneration,
        )

        dtype = _torch_dtype(torch, self.backend_config.dtype)
        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "trust_remote_code": self.backend_config.trust_remote_code,
        }
        if self.backend_config.device_map:
            model_kwargs["device_map"] = self.backend_config.device_map
        if self.backend_config.attn_implementation:
            model_kwargs["attn_implementation"] = self.backend_config.attn_implementation
        model_config = AutoConfig.from_pretrained(
            self.model_id,
            trust_remote_code=self.backend_config.trust_remote_code,
        )
        model_type = str(getattr(model_config, "model_type", ""))
        if model_type == "qwen3_vl":
            model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_id, **model_kwargs)
        elif model_type == "qwen2_vl":
            model = Qwen2VLForConditionalGeneration.from_pretrained(self.model_id, **model_kwargs)
        else:
            raise ValueError(f"unsupported VLM model_type for clean runner: {model_type!r}")
        processor = AutoProcessor.from_pretrained(
            self.processor_id or self.model_id,
            trust_remote_code=self.backend_config.trust_remote_code,
        )
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        self._loaded = (model, processor)
        return self._loaded

    def _build_inputs(self, processor: Any, messages: list[dict[str, Any]]) -> dict[str, Any]:
        from qwen_vl_utils import process_vision_info

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_patch_size = int(
            getattr(getattr(processor, "image_processor", None), "patch_size", 16) or 16
        )
        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=image_patch_size,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
        except TypeError:
            image_inputs, video_inputs = process_vision_info(messages)
            video_kwargs = {}
        video_metadatas = None
        if video_inputs is not None and video_inputs and isinstance(video_inputs[0], tuple):
            video_inputs, video_metadatas = zip(*video_inputs, strict=True)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        kwargs = dict(video_kwargs or {})
        if video_metadatas is not None:
            kwargs["video_metadata"] = video_metadatas
        return dict(
            processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                do_resize=False,
                padding=True,
                return_tensors="pt",
                **kwargs,
            )
        )

    def _vision_content_items(self, media: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        items = []
        max_pixels = int(self.max_image_resolution) ** 2
        for item in media:
            image = _media_to_image_input(item)
            if image is None:
                continue
            items.append({"type": "image", "image": image, "max_pixels": max_pixels})
        if not items:
            raise ValueError("sample has no loadable image media for qwen3_original backend")
        return items


class TGVFStage2Qwen3Backend(CleanRunnerBackend):
    def __init__(
        self,
        *,
        stage2_config: Stage2RuntimeConfig | None,
        backend_config: BackendConfig,
    ) -> None:
        if stage2_config is None:
            raise ValueError("Stage2 Qwen3 legacy backend requires Stage2RuntimeConfig")
        self.stage2_config = stage2_config
        self.backend_config = backend_config
        self._evaluator: Any | None = None

    def prepare(self, config: RunConfig) -> None:
        _reject_unported_deepstack_execution(config, backend=STAGE2_LEGACY_BACKEND)
        self.stage2_config.validate()
        from .legacy_stage2_adapter import build_legacy_stage2_args

        args = build_legacy_stage2_args(
            runtime=self.stage2_config,
            run_config=config,
            output_dir=f"/tmp/revisit_vlm_clean_stage2_runtime/{config.run_id}",
        )
        args.dtype = self.backend_config.dtype
        args.device = (
            self.backend_config.device if self.backend_config.device != "auto" else "cuda:0"
        )
        args.device_map = self.backend_config.device_map or args.device
        args.attn_implementation = self.backend_config.attn_implementation
        try:
            from eval.eval_v3_stage2_protocol import Stage2ProtocolEvaluator
        except Exception as exc:
            raise RuntimeError(
                "legacy Stage2 evaluator is unavailable; run with repository src/ on PYTHONPATH"
            ) from exc
        evaluator = Stage2ProtocolEvaluator(args)
        evaluator.load()
        self._evaluator = evaluator

    def run(
        self,
        sample: BenchmarkSample,
        rendered: RenderedBenchmarkInput,
        config: RunConfig,
    ) -> ModelRunResult:
        del config
        if self._evaluator is None:
            raise RuntimeError("TGVFStage2Qwen3Backend.prepare must be called before run")
        started = time.perf_counter()
        try:
            from .legacy_stage2_adapter import make_legacy_stage2_sample

            legacy_sample = make_legacy_stage2_sample(sample, rendered)
            if rendered.mode == EvalMode.TGVF_FORCE:
                result_row = self._run_force(legacy_sample)
            elif rendered.mode in {EvalMode.TGVF_FREE, EvalMode.TGVF_SOFTFORCE}:
                result_row = self._run_free(legacy_sample)
            else:
                raise ValueError(
                    f"Stage2 Qwen3 legacy backend does not support mode={rendered.mode.value!r}"
                )
            return ModelRunResult(
                raw_output=str(
                    result_row.get("final_raw_output") or result_row.get("raw_output") or ""
                ),
                triggered=bool(result_row.get("trigger_focus_decision")),
                focus_target=str(
                    result_row.get("parsed_focus_target") or result_row.get("focus_target") or ""
                ),
                focus_valid=result_row.get("focus_valid"),
                append_success=result_row.get("append_success"),
                wall_time_sec=time.perf_counter() - started,
                error=_row_error(result_row),
                debug=result_row,
            )
        except Exception as exc:
            return ModelRunResult(
                raw_output="",
                wall_time_sec=time.perf_counter() - started,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _run_force(self, legacy_sample: Any) -> dict[str, Any]:
        evaluator = self._evaluator
        assert evaluator is not None
        capture = evaluator._capture_generated_focus(legacy_sample, force_prefix=True)
        if not capture.capture_found:
            parsed = evaluator._capture_fields(
                capture,
                _legacy_parse_v3_action(capture.generated_text, evaluator.args.tgvf_protocol),
            )
            parsed.update(
                final_raw_output=capture.generated_text,
                trigger_focus_decision=False,
                append_success=False,
                errors=["focus_capture_not_found"],
            )
            return parsed
        correct_d = evaluator._d_from_capture(legacy_sample, capture, focus_source="clean_force")
        return evaluator._run_post_tgvf_condition(
            sample=legacy_sample,
            capture=capture,
            correct_d=correct_d,
            condition=self.stage2_config.d_condition,
            block="clean_force_end2end",
            focus_source="clean_force",
        )

    def _run_free(self, legacy_sample: Any) -> dict[str, Any]:
        evaluator = self._evaluator
        assert evaluator is not None
        capture = evaluator._capture_free_router(legacy_sample)
        parsed = _legacy_parse_v3_action(capture.generated_text, evaluator.args.tgvf_protocol)
        trigger = capture.capture_found
        if not trigger:
            row = evaluator._capture_fields(capture, parsed)
            row.update(
                final_raw_output=capture.generated_text,
                parsed_answer=parsed.answer,
                answer_parse_success=parsed.answer_valid,
                trigger_focus_decision=False,
                append_success=None,
            )
            return row
        correct_d = evaluator._d_from_capture(legacy_sample, capture, focus_source="clean_free")
        return evaluator._run_post_tgvf_condition(
            sample=legacy_sample,
            capture=capture,
            correct_d=correct_d,
            condition=self.stage2_config.d_condition,
            block="clean_free_end2end",
            focus_source="clean_free",
        )


class TGVFStage2Qwen3NativeBackend(CleanRunnerBackend):
    """Final clean-native Stage2 backend boundary."""

    def __init__(
        self,
        *,
        stage2_config: Stage2RuntimeConfig | None,
        backend_config: BackendConfig,
    ) -> None:
        if stage2_config is None:
            raise ValueError("Stage2 Qwen3 native backend requires Stage2RuntimeConfig")
        self.stage2_config = stage2_config
        self.backend_config = backend_config
        self._engine = NativeStage2Engine(
            stage2_config=stage2_config,
            backend_options={
                "dtype": backend_config.dtype,
                "device": backend_config.device,
                "device_map": backend_config.device_map,
                "attn_implementation": backend_config.attn_implementation,
                "trust_remote_code": backend_config.trust_remote_code,
            },
        )

    def prepare(self, config: RunConfig) -> None:
        _reject_unported_deepstack_execution(config, backend=STAGE2_NATIVE_BACKEND)
        self._engine.prepare(config)

    def run(
        self,
        sample: BenchmarkSample,
        rendered: RenderedBenchmarkInput,
        config: RunConfig,
    ) -> ModelRunResult:
        started = time.perf_counter()
        try:
            return _native_stage2_result_to_model_run_result(
                self._engine.run(sample, rendered, config),
                started=started,
            )
        except Exception as exc:
            return ModelRunResult(
                raw_output="",
                wall_time_sec=time.perf_counter() - started,
                error=f"{type(exc).__name__}: {exc}",
                debug={"native_stage2": self._engine.identity()},
            )


def _native_stage2_result_to_model_run_result(
    result: NativeStage2RunResult,
    *,
    started: float,
) -> ModelRunResult:
    return ModelRunResult(
        raw_output=result.raw_output,
        triggered=result.triggered,
        focus_target=result.focus_target,
        focus_valid=result.focus_valid,
        append_success=result.append_success,
        output_tokens=result.output_tokens,
        wall_time_sec=time.perf_counter() - started,
        error=result.error,
        debug=result.debug,
    )


def resolve_backend_name(name: str) -> str:
    if name == STAGE2_GENERIC_BACKEND:
        return STAGE2_NATIVE_BACKEND
    return name


def make_backend(
    backend_config: BackendConfig,
    *,
    config: RunConfig,
) -> CleanRunnerBackend:
    backend_name = resolve_backend_name(backend_config.backend)
    if backend_name == "dry_run":
        return DryRunBackend()
    if backend_name == "qwen3_original":
        return Qwen3OriginalBackend(
            model_id=config.model_id,
            processor_id=config.processor_id,
            backend_config=backend_config,
            max_image_resolution=config.max_image_resolution,
            max_answer_tokens=config.max_answer_tokens,
        )
    if backend_name == STAGE2_NATIVE_BACKEND:
        return TGVFStage2Qwen3NativeBackend(
            stage2_config=backend_config.stage2,
            backend_config=backend_config,
        )
    if backend_name == STAGE2_LEGACY_BACKEND:
        if config.eval_family != EvalFamily.INTERNAL_DIAGNOSTIC:
            raise ValueError(
                "tgvf_stage2_qwen3_legacy is diagnostic-only; set "
                "eval_family='internal_diagnostic' or use tgvf_stage2_qwen3/native"
            )
        return TGVFStage2Qwen3Backend(
            stage2_config=backend_config.stage2,
            backend_config=backend_config,
        )
    raise ValueError(f"unknown clean runner backend: {backend_config.backend}")


def _reject_unported_deepstack_execution(config: RunConfig, *, backend: str) -> None:
    if not config.deepstack.enabled:
        return
    plan = build_deepstack_execution_plan(config, backend=backend)
    raise NotImplementedError(
        "DeepStack execution is not implemented for clean Stage2 benchmark backends yet "
        f"(backend={backend}, scope={config.deepstack.original_image_scope}, "
        f"answer_restore={plan['original_image_deepstack']['restore_for_answer']}). "
        "The run config may record DeepStack identity, but real evaluation must not "
        "claim DeepStack behavior until original-image DeepStack injection/masking is ported. "
        f"deepstack_execution_plan={json.dumps(plan, sort_keys=True)}"
    )


def build_deepstack_execution_plan(config: RunConfig, *, backend: str) -> dict[str, Any]:
    state = config.deepstack
    blockers = [
        "native Qwen3 DeepStack feature injection for original image is not ported",
        "post-D DeepStack masking/restoration by scope is not ported",
        "equivalence with no-DeepStack native Stage2 path is not proven",
    ]
    scope_contract = deepstack_scope_contract(
        state,
        surface="benchmark_eval",
        backend=backend,
        execution_supported=False,
        blocking_items=blockers,
    )
    return {
        "backend": backend,
        "enabled": bool(state.enabled),
        "execution_supported": False,
        "status": "not_ported",
        "original_image_scope": str(state.original_image_scope),
        "original_image_deepstack": scope_contract["original_image_deepstack"],
        "d_deepstack_features": scope_contract["d_deepstack_features"],
        "fvt_visual_token_path": scope_contract["fvt_visual_token_path"],
        "scope_contract": scope_contract,
        "blocking_items": blockers,
    }


def run_benchmark_rows(
    samples: list[BenchmarkSample],
    rendered_inputs: list[RenderedBenchmarkInput],
    *,
    config: RunConfig,
    backend_config: BackendConfig,
) -> tuple[list[dict[str, Any]], EvalSummary]:
    if len(samples) != len(rendered_inputs):
        raise ValueError("sample/rendered input count mismatch")
    backend = make_backend(backend_config, config=config)
    backend.prepare(config)
    rows = []
    resolved_backend = resolve_backend_name(backend_config.backend)
    for sample, rendered in zip(samples, rendered_inputs, strict=True):
        result = backend.run(sample, rendered, config)
        deepstack_execution = _row_deepstack_execution(
            config=config,
            backend_config=backend_config,
            resolved_backend=resolved_backend,
            debug=result.debug,
        )
        rows.append(
            {
                "sample_id": sample.sample_id,
                "benchmark": sample.benchmark,
                "population_id": sample.population_id,
                "subset_id": config.subset_id,
                "source_file": sample.source_file,
                "num_shards": config.num_shards,
                "shard_index": config.shard_index,
                "method": config.mode.value,
                "eval_family": config.eval_family.value,
                "tgvf_protocol": config.tgvf_protocol,
                "post_tgvf_continuation": config.post_tgvf_continuation.value,
                "post_tgvf_forward_mode": config.post_tgvf_forward_mode.value,
                "deepstack": config.deepstack.to_dict(),
                "deepstack_execution": deepstack_execution,
                "parser_scorer": config.parser_scorer.to_dict(),
                "d_condition": (
                    backend_config.stage2.d_condition if backend_config.stage2 is not None else None
                ),
                "runner_backend": backend_config.backend,
                "resolved_runner_backend": resolved_backend,
                "runner_backend_deprecated_alias": False,
                "runner_backend_stage2_generic_alias": (
                    backend_config.backend == STAGE2_GENERIC_BACKEND
                ),
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
                "d_shape": _d_shape(result.debug),
                "continuation_metadata": _continuation_metadata(
                    result=result,
                    config=config,
                    backend_config=backend_config,
                    resolved_backend=resolved_backend,
                ),
                "output_tokens": result.output_tokens,
                "wall_time_sec": result.wall_time_sec,
                "metadata": sample.metadata,
                "debug_metadata": result.debug,
                "error": result.error,
            }
        )
    score_output_rows(
        rows,
        scoring_backend=config.parser_scorer.scoring_backend,
        benchmark_root=config.benchmark_root,
    )
    summary = summarize_executed_rows(rows, config=config, manifest_hash=config.manifest_hash)
    return rows, summary


def summarize_executed_rows(
    rows: list[dict[str, Any]],
    *,
    config: RunConfig,
    manifest_hash: str | None,
) -> EvalSummary:
    scored = [float(row["score"]) for row in rows if row.get("score") is not None]
    parse_values = [bool(row.get("answer_parse_success")) for row in rows]
    malformed = [bool(row.get("malformed")) for row in rows]
    trigger_values = [
        bool(row.get("trigger_focus_decision"))
        for row in rows
        if row.get("trigger_focus_decision") is not None
    ]
    focus_values = [
        bool(row.get("focus_valid")) for row in rows if row.get("focus_valid") is not None
    ]
    append_values = [
        bool(row.get("append_success")) for row in rows if row.get("append_success") is not None
    ]
    return EvalSummary(
        run_id=config.run_id,
        n_rows=len(rows),
        n_scored=len(scored),
        accuracy=_mean(scored),
        answer_parse_rate=_mean_bool(parse_values),
        malformed_rate=_mean_bool(malformed),
        trigger_rate=_mean_bool(trigger_values) if trigger_values else None,
        focus_valid_rate=_mean_bool(focus_values) if focus_values else None,
        append_success_rate=_mean_bool(append_values) if append_values else None,
        manifest_hash=manifest_hash,
        comparable=True,
        comparability_note="clean executable runner output",
    )


def summarize_deepstack_execution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    executions = [row.get("deepstack_execution") or {} for row in rows]
    requested_enabled_values = [
        bool(item.get("requested_enabled"))
        for item in executions
        if item.get("requested_enabled") is not None
    ]
    fvt_values = [
        bool(item.get("fvt_append_uses_deepstack"))
        for item in executions
        if item.get("fvt_append_uses_deepstack") is not None
    ]
    unsupported_requested = [
        item
        for item in executions
        if item.get("requested_enabled")
        and item.get("execution_supported_for_requested_state") is False
    ]
    return {
        "schema_version": "clean_deepstack_execution_summary_v1",
        "n_rows": len(rows),
        "requested_enabled_rows": sum(requested_enabled_values),
        "fvt_append_reported_rows": len(fvt_values),
        "fvt_append_uses_deepstack_rows": sum(fvt_values),
        "any_fvt_append_uses_deepstack": any(fvt_values) if fvt_values else False,
        "all_reported_fvt_append_uses_deepstack": (
            all(fvt_values) if fvt_values else None
        ),
        "unsupported_requested_rows": len(unsupported_requested),
        "deepstack_caution_rows": sum(
            bool(item.get("deepstack_caution")) for item in executions
        ),
    }


def summarize_result_breakdowns(rows: list[dict[str, Any]]) -> dict[str, Any]:
    choice_rows = [row for row in rows if row.get("choices")]
    return {
        "schema_version": "clean_benchmark_result_breakdowns_v1",
        "n_rows": len(rows),
        "by_benchmark": _grouped_metric_summary(rows, "benchmark"),
        "by_population_id": _grouped_metric_summary(rows, "population_id"),
        "by_method": _grouped_metric_summary(rows, "method"),
        "by_d_condition": _grouped_metric_summary(rows, "d_condition"),
        "choice_counts": {
            "choice_row_count": len(choice_rows),
            "prediction_counts": _count_values(
                row.get("parsed_answer") for row in choice_rows if row.get("parsed_answer")
            ),
            "gold_counts": _count_values(
                row.get("gold_answer") for row in choice_rows if row.get("gold_answer")
            ),
        },
        "official_scoring": {
            "official_tool_used_rows": sum(bool(row.get("official_tool_used")) for row in rows),
            "official_compatible_rows": sum(bool(row.get("official_compatible")) for row in rows),
            "scorer_names": _count_values(row.get("scorer_name") for row in rows),
            "official_tool_paths": sorted(
                {
                    str(row.get("official_tool_path"))
                    for row in rows
                    if row.get("official_tool_path")
                }
            ),
        },
    }


def _final_output(result: ModelRunResult) -> str:
    return str(result.debug.get("final_raw_output") or result.raw_output or "")


def _trigger_policy(config: RunConfig) -> dict[str, Any]:
    if config.mode == EvalMode.ORIGINAL:
        policy = "none_original"
    elif config.mode == EvalMode.TGVF_FORCE:
        policy = "forced_focus_prefix"
    elif config.mode == EvalMode.TGVF_SOFTFORCE:
        policy = "softforce_prompted_router"
    else:
        policy = "free_router"
    return {
        "schema_version": "clean_benchmark_trigger_policy_v1",
        "policy": policy,
        "mode": config.mode.value,
        "requires_focus": config.mode == EvalMode.TGVF_FORCE,
        "allows_no_focus": config.mode in {
            EvalMode.ORIGINAL,
            EvalMode.TGVF_FREE,
            EvalMode.TGVF_SOFTFORCE,
        },
        "softforce_prompt_text": config.softforce_prompt_text,
    }


def _d_shape(debug: dict[str, Any]) -> list[int] | None:
    value = debug.get("D_shape")
    if value is None:
        value = debug.get("fvt_shape")
    if value is None:
        return None
    if isinstance(value, int):
        return [int(value)]
    try:
        return [int(item) for item in value]
    except Exception:
        return None


def _continuation_metadata(
    *,
    result: ModelRunResult,
    config: RunConfig,
    backend_config: BackendConfig,
    resolved_backend: str,
) -> dict[str, Any]:
    stage2 = backend_config.stage2
    return {
        "schema_version": "clean_benchmark_continuation_metadata_v1",
        "post_tgvf_continuation": config.post_tgvf_continuation.value,
        "post_tgvf_forward_mode": config.post_tgvf_forward_mode.value,
        "stage2_append_forward_mode": (
            stage2.append_forward_mode.value if stage2 is not None else None
        ),
        "runner_backend": backend_config.backend,
        "resolved_runner_backend": resolved_backend,
        "output_tokens": result.output_tokens,
        "wall_time_sec": result.wall_time_sec,
        "triggered": result.triggered,
        "append_success": result.append_success,
        "continuation_not_im_end": _optional_bool(
            result.debug.get("continuation_not_im_end")
        ),
        "second_full_forward_used": _optional_bool(
            result.debug.get("second_full_forward_used")
        ),
        "final_output_source": (
            "post_tgvf_continuation"
            if result.triggered and result.append_success
            else "direct_or_capture_output"
        ),
    }


def _row_deepstack_execution(
    *,
    config: RunConfig,
    backend_config: BackendConfig,
    resolved_backend: str,
    debug: dict[str, Any],
) -> dict[str, Any]:
    requested_enabled = bool(config.deepstack.enabled)
    return {
        "schema_version": "clean_deepstack_execution_row_v1",
        "requested": config.deepstack.to_dict(),
        "requested_enabled": requested_enabled,
        "requested_scope": str(config.deepstack.original_image_scope),
        "execution_supported_for_requested_state": not requested_enabled,
        "backend": backend_config.backend,
        "resolved_backend": resolved_backend,
        "fvt_append_path": debug.get("mask_mode") or debug.get("fvt_append_path"),
        "fvt_position_mode": debug.get("fvt_position_mode"),
        "fvt_append_uses_deepstack": _optional_bool(debug.get("uses_deepstack_for_fvt")),
        "deepstack_caution": debug.get("deepstack_caution"),
        "notes": (
            ["DeepStack was requested but clean Stage2 execution rejects it before rows"]
            if requested_enabled
            else []
        ),
    }


def _dry_output(sample: BenchmarkSample) -> str:
    if sample.gold_answer not in (None, ""):
        return str(sample.gold_answer)
    if sample.choices:
        return "A"
    return "dry_run_answer"


def _media_to_image_input(media: dict[str, Any]) -> Any | None:
    kind = media.get("kind")
    if kind == "path":
        path = str(media.get("path") or "")
        return path if path and Path(path).exists() else None
    if kind in {"image_struct", "embedded_image_struct", "embedded_bytes"}:
        bytes_value = media.get("bytes")
        if isinstance(bytes_value, (bytes, bytearray)):
            return _image_from_bytes(bytes(bytes_value))
        path = str(media.get("path") or "")
        if path and Path(path).exists():
            return path
        return None
    if kind == "embedded_base64":
        value = str(media.get("value") or "")
        if value:
            return _image_from_base64(value)
    return None


def _image_from_base64(value: str) -> Any:
    payload = value.split(",", 1)[1] if value.startswith("data:image") and "," in value else value
    return _image_from_bytes(base64.b64decode(payload))


def _image_from_bytes(value: bytes) -> Any:
    from PIL import Image

    with Image.open(io.BytesIO(value)) as image:
        return image.convert("RGB").copy()


def _torch_dtype(torch: Any, name: str) -> Any:
    if name == "auto":
        return "auto"
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _resolve_generation_device(model: Any, requested: str) -> Any:
    import torch

    if requested != "auto":
        return torch.device(requested)
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _move_tensors(value: Any, device: Any) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tensors(item, device) for item in value]
    return value


def _legacy_parse_v3_action(text: str, protocol: str) -> Any:
    from revisit_vlm.qwen3_vl_tgvf import parse_v3_action

    return parse_v3_action(text, protocol=protocol)


def _row_error(row: dict[str, Any]) -> str | None:
    errors = row.get("errors")
    if isinstance(errors, list) and errors:
        return "; ".join(str(item) for item in errors)
    if row.get("append_success") is False and row.get("trigger_focus_decision"):
        return "append_failed"
    return None


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return None if not values else sum(values) / len(values)


def _mean_bool(values: Iterable[bool]) -> float | None:
    values = list(values)
    return None if not values else sum(1.0 for item in values if item) / len(values)


def _grouped_metric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_summary_key(row.get(key)), []).append(row)
    return {
        group_key: _metric_summary(group_rows)
        for group_key, group_rows in sorted(groups.items())
    }


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [float(row["score"]) for row in rows if row.get("score") is not None]
    parse_values = [bool(row.get("answer_parse_success")) for row in rows]
    malformed = [bool(row.get("malformed")) for row in rows]
    trigger_values = [
        bool(row.get("trigger_focus_decision"))
        for row in rows
        if row.get("trigger_focus_decision") is not None
    ]
    focus_values = [
        bool(row.get("focus_valid")) for row in rows if row.get("focus_valid") is not None
    ]
    append_values = [
        bool(row.get("append_success")) for row in rows if row.get("append_success") is not None
    ]
    return {
        "n_rows": len(rows),
        "n_scored": len(scored),
        "accuracy": _mean(scored),
        "answer_parse_rate": _mean_bool(parse_values),
        "malformed_rate": _mean_bool(malformed),
        "trigger_rate": _mean_bool(trigger_values) if trigger_values else None,
        "focus_valid_rate": _mean_bool(focus_values) if focus_values else None,
        "append_success_rate": _mean_bool(append_values) if append_values else None,
    }


def _summary_key(value: Any) -> str:
    if value in (None, ""):
        return "none"
    return str(value)


def _count_values(values: Iterable[Any]) -> dict[str, int]:
    counts = Counter(_summary_key(value) for value in values if value not in (None, ""))
    return dict(sorted(counts.items()))


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)
