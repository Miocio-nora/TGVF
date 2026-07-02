"""Clean-native Stage2 engine boundary.

This module owns the final backend contract for Qwen3 Stage2 TGVF execution.
The engine uses lower-level Qwen3/TGVF primitives directly and keeps the native
backend independent from historical evaluator classes.
"""

from __future__ import annotations

import base64
import gc
import hashlib
import io
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .benchmark_data import BenchmarkSample
from .data_generation import file_identity
from .deepstack import (
    build_cached_chunk_original_image_key_block_attention_mask,
    build_original_image_key_block_attention_mask,
    build_qwen3_deepstack_payload,
    build_single_query_original_image_key_block_attention_mask,
    capture_qwen3_original_image_deepstack_features,
)
from .rendering import RenderedBenchmarkInput
from .schema import DeepStackScope, EvalMode, ForwardMode, RunConfig
from .stage2_runtime import Stage2RuntimeConfig, eval_jsonl_identity

SUPPORTED_NATIVE_STAGE2_MODES = frozenset(
    {
        EvalMode.TGVF_FORCE,
        EvalMode.TGVF_FREE,
        EvalMode.TGVF_SOFTFORCE,
    }
)


@dataclass(frozen=True)
class NativeStage2RunResult:
    raw_output: str
    triggered: bool = False
    focus_target: str = ""
    focus_valid: bool | None = None
    append_success: bool | None = None
    output_tokens: int = 0
    error: str | None = None
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NativeStage2Sample:
    image: Any
    question: str
    answer: str
    need_focus: bool
    evidence_state: str
    trajectory_type: str
    target: str = ""
    evidence_description: str = ""
    image_id: str | None = None
    choices: Any | None = None
    answer_format: str | None = None
    value_span_text: str | None = None
    evidence_type: str | None = None
    target_style: str | None = None
    target_cues: list[str] = field(default_factory=list)
    source_dataset: str | None = None
    source_profile: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_question(self) -> str:
        if not self.choices:
            return self.question
        if isinstance(self.choices, dict):
            choice_lines = [f"{key}. {value}" for key, value in self.choices.items()]
        elif isinstance(self.choices, (list, tuple)):
            choice_lines = [str(choice) for choice in self.choices]
        else:
            choice_lines = [str(self.choices)]
        return self.question.rstrip() + "\nChoices:\n" + "\n".join(choice_lines)


@dataclass(frozen=True)
class _AlignedCaptureCache:
    past_key_values: Any
    attention_mask: Any
    input_ids: Any
    tail_token_ids: Any
    debug: dict[str, Any]


class NativeStage2Engine:
    """Clean-native Stage2 runtime owner.

    The engine deliberately accepts only plain backend options so it does not
    depend on the runner's backend dataclass.
    """

    def __init__(
        self,
        *,
        stage2_config: Stage2RuntimeConfig,
        backend_options: dict[str, Any] | None = None,
    ) -> None:
        self.stage2_config = stage2_config
        self.backend_options = dict(backend_options or {})
        self._prepared = False
        self._identity: dict[str, Any] | None = None
        self._run_config: RunConfig | None = None
        self._checkpoint: dict[str, Any] | None = None
        self._loaded: Any | None = None
        self.model: Any | None = None
        self.utility_model: Any | None = None
        self.processor: Any | None = None
        self.foveal_module: Any | None = None
        self.device: Any | None = None
        self.vision_cache: dict[str, tuple[Any, Any, Any, list[Any], list[Any]]] = {}
        self.deepstack_cache: dict[str, list[Any]] = {}

    def prepare(self, config: RunConfig) -> None:
        self.stage2_config.validate()
        _validate_run_alignment(self.stage2_config, config)
        self._identity = {
            "engine": "NativeStage2Engine",
            "capture_append_ported": True,
            "heavy_runtime_loaded": False,
            "stage2_runtime": self.stage2_config.to_dict(),
            "run_config": {
                "run_id": config.run_id,
                "mode": config.mode.value,
                "post_tgvf_forward_mode": config.post_tgvf_forward_mode.value,
                "post_tgvf_continuation": config.post_tgvf_continuation.value,
                "tgvf_protocol": config.tgvf_protocol,
                "max_image_resolution": config.max_image_resolution,
                "token_budget_policy": (
                    "unified_max_tokens" if config.max_tokens is not None else "legacy_split"
                ),
                "max_tokens": config.max_tokens,
                "legacy_split_limits": {
                    "active": config.max_tokens is None,
                    "max_action_tokens": config.max_action_tokens,
                    "max_answer_tokens": config.max_answer_tokens,
                },
                "deepstack": config.deepstack.to_dict(),
                "parser_scorer": config.parser_scorer.to_dict(),
            },
            "checkpoint_file": file_identity(self.stage2_config.stage2_checkpoint).to_dict(),
            "eval_jsonl": eval_jsonl_identity(self.stage2_config.eval_jsonl),
            "backend_options": dict(self.backend_options),
        }
        self._prepared = True
        self._run_config = config

    def run(
        self,
        sample: BenchmarkSample,
        rendered: RenderedBenchmarkInput,
        config: RunConfig,
    ) -> NativeStage2RunResult:
        if not self._prepared:
            raise RuntimeError("NativeStage2Engine.prepare must be called before run")
        _validate_run_alignment(self.stage2_config, config)
        if rendered.mode != config.mode:
            raise ValueError(
                "rendered input mode does not match run config: "
                f"{rendered.mode.value!r} != {config.mode.value!r}"
            )
        if rendered.sample_id != sample.sample_id:
            raise ValueError(
                "rendered input sample_id does not match sample: "
                f"{rendered.sample_id!r} != {sample.sample_id!r}"
            )
        runtime_sample = stage2_sample_from_clean_sample(sample, rendered)
        if rendered.mode == EvalMode.TGVF_FORCE:
            return self._run_force(runtime_sample)
        if rendered.mode in {EvalMode.TGVF_FREE, EvalMode.TGVF_SOFTFORCE}:
            return self._run_free(runtime_sample)
        raise NotImplementedError(f"unsupported clean-native Stage2 mode: {rendered.mode.value}")

    def identity(self) -> dict[str, Any]:
        if self._identity is None:
            return {
                "engine": "NativeStage2Engine",
                "prepared": False,
                "stage2_runtime": self.stage2_config.to_dict(),
                "backend_options": dict(self.backend_options),
            }
        return dict(self._identity)

    def cleanup_after_row(self) -> dict[str, Any]:
        """Release per-sample caches that can otherwise accumulate GPU tensors."""
        vision_entries = len(self.vision_cache)
        deepstack_entries = len(self.deepstack_cache)
        self.vision_cache.clear()
        self.deepstack_cache.clear()
        _collect_cuda_garbage()
        return {
            "vision_cache_entries_cleared": vision_entries,
            "deepstack_cache_entries_cleared": deepstack_entries,
        }

    def recover_after_fatal_error(self, error: str | None) -> dict[str, Any]:
        cleanup = self.cleanup_after_row()
        fatal = _is_cuda_fatal_error(error)
        unloaded = False
        if fatal:
            self._unload_runtime()
            unloaded = True
        return {
            **cleanup,
            "fatal_cuda_error": fatal,
            "runtime_unloaded": unloaded,
        }

    def _unload_runtime(self) -> None:
        self._loaded = None
        self._checkpoint = None
        self.model = None
        self.utility_model = None
        self.processor = None
        self.foveal_module = None
        self.device = None
        if self._identity is not None:
            self._identity["heavy_runtime_loaded"] = False
            self._identity["runtime_unloaded_after_error"] = True
        _collect_cuda_garbage()

    def _run_force(self, sample: NativeStage2Sample) -> NativeStage2RunResult:
        self._ensure_loaded(sample)
        started = time.perf_counter()
        capture = self._capture_generated_focus(sample, force_prefix=True)
        parsed = self._parse_action(capture.generated_text)
        if not bool(getattr(capture, "capture_found", False)):
            debug = {
                **self._base_debug(sample, block="force_end2end", method="correct_D"),
                **self._capture_fields(capture, parsed),
                "final_raw_output": capture.generated_text,
                "trigger_focus_decision": False,
                "append_success": False,
                "errors": ["focus_capture_not_found"],
                "token_budget": self._token_budget_debug(capture, None),
                "wall_time_sec": time.perf_counter() - started,
            }
            return NativeStage2RunResult(
                raw_output=capture.generated_text,
                triggered=False,
                focus_target=str(debug.get("parsed_focus_target") or ""),
                focus_valid=debug.get("focus_valid"),
                append_success=False,
                debug=debug,
            )
        d = self._d_from_capture(sample, capture, focus_source="clean_native_force")
        return self._run_post_tgvf_condition(
            sample=sample,
            capture=capture,
            correct_d=d,
            block="clean_native_force_end2end",
            focus_source="clean_native_force",
        )

    def _run_free(self, sample: NativeStage2Sample) -> NativeStage2RunResult:
        self._ensure_loaded(sample)
        started = time.perf_counter()
        capture = self._capture_free_router(sample)
        parsed = self._parse_action(capture.generated_text)
        trigger = bool(getattr(capture, "capture_found", False)) and (
            parsed.evidence_state == self._need_local_evidence()
            or self.stage2_config.protocol
            in {
                "protocol_c_thinking_special",
                "protocol_c_tool_observation",
                "protocol_c_tool_observation_qwen2_no_think",
            }
        )
        if not trigger:
            debug = {
                **self._base_debug(sample, block="free_router_end2end", method="direct_or_miss"),
                **self._capture_fields(capture, parsed),
                "final_raw_output": capture.generated_text,
                "parsed_answer": parsed.answer,
                "answer_parse_success": parsed.answer_valid,
                "trigger_focus_decision": False,
                "focus_miss": sample.need_focus,
                "no_focus_false_trigger": False,
                "continuation_not_im_end": _continuation_not_im_end(capture.generated_text),
                "token_budget": self._token_budget_debug(capture, None),
                "wall_time_sec": time.perf_counter() - started,
            }
            return NativeStage2RunResult(
                raw_output=capture.generated_text,
                triggered=False,
                focus_target=str(debug.get("parsed_focus_target") or ""),
                focus_valid=debug.get("focus_valid"),
                append_success=None,
                output_tokens=len(getattr(capture, "generated_ids", []) or []),
                debug=debug,
            )
        d = self._d_from_capture(sample, capture, focus_source="clean_native_free")
        return self._run_post_tgvf_condition(
            sample=sample,
            capture=capture,
            correct_d=d,
            block="clean_native_free_end2end",
            focus_source="clean_native_free",
        )

    def _run_post_tgvf_condition(
        self,
        *,
        sample: NativeStage2Sample,
        capture: Any,
        correct_d: Any,
        block: str,
        focus_source: str,
    ) -> NativeStage2RunResult:
        started = time.perf_counter()
        parsed_focus = self._parse_action(capture.generated_text)
        debug = {
            **self._base_debug(sample, block=block, method=self.stage2_config.d_condition),
            **self._capture_fields(capture, parsed_focus),
            "focus_source": focus_source,
        }
        try:
            if self.stage2_config.d_condition != "correct_D":
                raise ValueError(
                    "clean-native Stage2 currently supports only d_condition='correct_D'"
                )
            if self.stage2_config.append_forward_mode == ForwardMode.NO_KV_FULL_SEQUENCE:
                append_result = self._append_visual_d_full_sequence(sample, capture, correct_d)
            else:
                try:
                    append_result = self._append_visual_d(capture, correct_d)
                except (RuntimeError, ValueError) as exc:
                    if not _should_retry_full_sequence_multi_image_append(capture, exc):
                        raise
                    append_result = self._append_visual_d_full_sequence(
                        sample,
                        capture,
                        correct_d,
                    )
                    append_result.debug_metadata[
                        "kv_cache_append_retry_error"
                    ] = f"{type(exc).__name__}: {exc}"
                    append_result.debug_metadata[
                        "kv_cache_append_retry_reason"
                    ] = "multi_image_native_source_grid_position_ids"
            continuation = self._continue_generation(append_result, capture=capture)
            full_text = _full_protocol_text(
                capture.generated_text,
                continuation.generated_text,
                protocol=self.stage2_config.protocol,
            )
            parsed_full = self._parse_action(full_text)
            debug.update(
                final_raw_output=continuation.generated_text,
                full_protocol_text=full_text,
                parsed_evidence_state=parsed_full.evidence_state or parsed_focus.evidence_state,
                parsed_focus_target=parsed_focus.focus_target,
                parsed_evidence=_parsed_evidence_text(full_text, self.stage2_config.protocol),
                parsed_answer=parsed_full.answer,
                malformed=bool(parsed_focus.malformed or parsed_full.malformed),
                answer_parse_success=parsed_full.answer_valid,
                trigger_focus_decision=True,
                continuation_not_im_end=_continuation_not_im_end(continuation.generated_text),
                append_success=True,
                fvt_shape=append_result.debug_metadata.get("fvt_shape"),
                D_shape=append_result.debug_metadata.get("fvt_shape"),
                mask_mode=append_result.debug_metadata.get("fvt_append_path"),
                fvt_position_mode=append_result.debug_metadata.get("fvt_position_mode"),
                uses_deepstack_for_fvt=append_result.debug_metadata.get(
                    "uses_deepstack_for_fvt"
                ),
                deepstack_caution=append_result.debug_metadata.get("deepstack_caution"),
                kv_cache_input_len=append_result.debug_metadata.get("kv_cache_input_len"),
                kv_cache_initial_seq_len=append_result.debug_metadata.get(
                    "kv_cache_initial_seq_len"
                ),
                kv_cache_tail_prefill_tokens=append_result.debug_metadata.get(
                    "kv_cache_tail_prefill_tokens"
                ),
                kv_cache_aligned_seq_len=append_result.debug_metadata.get(
                    "kv_cache_aligned_seq_len"
                ),
                kv_cache_tail_prefill_used=append_result.debug_metadata.get(
                    "kv_cache_tail_prefill_used"
                ),
                kv_cache_tail_in_append_chunk=append_result.debug_metadata.get(
                    "kv_cache_tail_in_append_chunk"
                ),
                kv_cache_tail_chunk_tokens=append_result.debug_metadata.get(
                    "kv_cache_tail_chunk_tokens"
                ),
                model_append_chunk_length=append_result.debug_metadata.get(
                    "model_append_chunk_length"
                ),
                kv_cache_append_retry_error=append_result.debug_metadata.get(
                    "kv_cache_append_retry_error"
                ),
                kv_cache_append_retry_reason=append_result.debug_metadata.get(
                    "kv_cache_append_retry_reason"
                ),
                second_full_forward_used=bool(
                    getattr(capture, "second_full_forward_used", False)
                    or append_result.debug_metadata.get("second_full_forward_used")
                ),
                focus_generated_ids=list(getattr(capture, "generated_ids", []) or []),
                focus_generated_logprobs=list(
                    getattr(capture, "generated_logprobs", []) or []
                ),
                continuation_generated_ids=list(
                    getattr(continuation, "generated_ids", []) or []
                ),
                continuation_generated_logprobs=list(
                    getattr(continuation, "generated_logprobs", []) or []
                ),
                token_budget=self._token_budget_debug(capture, continuation),
                wall_time_sec=time.perf_counter() - started,
            )
            return NativeStage2RunResult(
                raw_output=continuation.generated_text,
                triggered=True,
                focus_target=str(parsed_focus.focus_target or getattr(capture, "target_text", "")),
                focus_valid=debug.get("focus_valid"),
                append_success=True,
                output_tokens=len(getattr(continuation, "generated_ids", []) or []),
                debug=debug,
            )
        except Exception as exc:
            debug.update(
                errors=[f"{type(exc).__name__}:{exc}"],
                append_success=False,
                wall_time_sec=time.perf_counter() - started,
            )
            return NativeStage2RunResult(
                raw_output="",
                triggered=True,
                focus_target=str(debug.get("parsed_focus_target") or ""),
                focus_valid=debug.get("focus_valid"),
                append_success=False,
                error=f"{type(exc).__name__}: {exc}",
                debug=debug,
            )

    def _ensure_loaded(self, sample: NativeStage2Sample) -> None:
        if self.model is not None and self.processor is not None and self.foveal_module is not None:
            return
        config = self._run_config
        if config is None:
            raise RuntimeError("NativeStage2Engine.prepare must be called before runtime load")

        import torch
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

        from revisit_vlm.qwen3_vl_tgvf import (
            PROTOCOL_C_THINKING_SPECIAL,
            PROTOCOL_C_TOOL_OBSERVATION,
            PROTOCOL_C_TOOL_OBSERVATION_QWEN2_NO_THINK,
            PROTOCOL_E_ACTION_EVIDENCE_SPECIAL,
            ensure_tgvf_protocol_tokens,
            load_qwen3_vl,
        )
        from revisit_vlm.tgvf_training import build_tgvf_module
        from revisit_vlm.qwen3_vl_tgvf import llm_hidden_dim, tap_qwen3_vision_features
        from revisit_vlm.tgvf_v3_stage1 import freeze_qwen_backbone

        self.device = _resolve_runtime_device(torch, self.backend_options.get("device"))
        checkpoint = torch.load(self.stage2_config.stage2_checkpoint, map_location="cpu")
        self._checkpoint = checkpoint
        checkpoint_config = checkpoint.get("config") or {}
        checkpoint_protocol = checkpoint_config.get("tgvf_protocol")
        if checkpoint_protocol and self.stage2_config.protocol != checkpoint_protocol:
            raise ValueError(
                "Stage2 checkpoint protocol mismatch: "
                f"checkpoint={checkpoint_protocol!r} requested={self.stage2_config.protocol!r}"
            )
        processor_id = config.processor_id or checkpoint_config.get("processor_id")
        loaded = load_qwen3_vl(
            config.model_id,
            processor_id=processor_id,
            dtype=str(self.backend_options.get("dtype") or "bfloat16"),
            device_map=self.backend_options.get("device_map") or self.backend_options.get("device"),
            attn_implementation=self.backend_options.get("attn_implementation"),
        )
        base_model = loaded.model
        processor = loaded.processor
        if getattr(processor.tokenizer, "pad_token", None) is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        protocol_token_info: dict[str, Any] = {}
        if self.stage2_config.protocol in {
            PROTOCOL_C_THINKING_SPECIAL,
            PROTOCOL_C_TOOL_OBSERVATION,
            PROTOCOL_C_TOOL_OBSERVATION_QWEN2_NO_THINK,
            PROTOCOL_E_ACTION_EVIDENCE_SPECIAL,
        }:
            protocol_token_info = ensure_tgvf_protocol_tokens(
                processor.tokenizer,
                base_model,
                protocol=self.stage2_config.protocol,
            )
        freeze_qwen_backbone(base_model)
        qwen_lora_state = checkpoint["qwen_lora"]
        _validate_qwen_lora_protocol_token_payload(qwen_lora_state)
        checkpoint_has_trainable_token_adapter = any(
            "trainable_tokens" in key or "token_adapter" in key for key in qwen_lora_state
        )
        lora_cfg = checkpoint_config.get("lora") or {}
        modules_to_save = lora_cfg.get("modules_to_save")
        lora_config = LoraConfig(
            r=int(lora_cfg.get("rank", 64)),
            lora_alpha=int(lora_cfg.get("alpha", 256)),
            target_modules=list(
                lora_cfg.get("target_modules")
                or "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj".split(",")
            ),
            lora_dropout=float(lora_cfg.get("dropout", 0.0)),
            bias=str(lora_cfg.get("bias", "none")),
            task_type="CAUSAL_LM",
            modules_to_save=list(modules_to_save) if modules_to_save else None,
            ensure_weight_tying=False,
            trainable_token_indices=(
                list(protocol_token_info.get("tgvf_protocol_token_ids", {}).values())
                if checkpoint_has_trainable_token_adapter and not modules_to_save
                else None
            ),
        )
        model = get_peft_model(base_model, lora_config)
        _validate_peft_load_result(set_peft_model_state_dict(model, qwen_lora_state))
        model.eval()
        if hasattr(model, "config"):
            model.config.use_cache = True
        utility_model = model.get_base_model() if hasattr(model, "get_base_model") else model
        tap, v_pre, _v_merge = tap_qwen3_vision_features(
            utility_model,
            processor,
            image=self._image(sample),
            question=sample.prompt_question,
            device=self.device,
        )
        if v_pre is None:
            raise RuntimeError(f"Could not infer V_pre shape: {tap.errors}")
        dims = {
            "d_lm": int(llm_hidden_dim(utility_model)),
            "d_v": int(v_pre.shape[-1]),
            "spatial_merge_size": int(tap.spatial_merge_size or tap.merge_size or 2),
        }
        tgvf_cfg = checkpoint_config.get("tgvf") or {}
        _validate_d_deepstack_checkpoint_support(config, checkpoint_config)
        foveal_module = build_tgvf_module(
            variant=str(tgvf_cfg.get("variant") or "tgvf_v2_bidirectional"),
            d_lm=dims["d_lm"],
            d_v=dims["d_v"],
            num_foveated_tokens=tgvf_cfg.get("num_foveated_tokens"),
            spatial_merge_size=int(
                tgvf_cfg.get("spatial_merge_size") or dims["spatial_merge_size"]
            ),
            attn_dim=tgvf_cfg.get("attn_dim"),
            encoder_adapter_layers=tuple(tgvf_cfg.get("encoder_adapter_layers") or (8, 16, 24)),
            encoder_adapter_gate_init=float(tgvf_cfg.get("encoder_adapter_gate_init", 0.0)),
            encoder_adapter_share_weights=bool(
                tgvf_cfg.get("encoder_adapter_share_weights", False)
            ),
            encoder_adapter_layer_index_base=int(
                tgvf_cfg.get("encoder_adapter_layer_index_base", 0)
            ),
            encoder_reencode_deepstack_compatible=bool(
                tgvf_cfg.get("encoder_reencode_deepstack_compatible", False)
            ),
            d_deepstack_enabled=bool(tgvf_cfg.get("d_deepstack_enabled", False)),
            d_deepstack_branch_layers=tuple(
                tgvf_cfg.get("d_deepstack_branch_layers") or (8, 16, 24)
            ),
        ).to(device=self.device, dtype=next(model.parameters()).dtype)
        foveal_module.load_state_dict(checkpoint["tgvf_module"], strict=True)
        foveal_module.eval()
        self._loaded = loaded
        self.model = model
        self.utility_model = utility_model
        self.processor = processor
        self.foveal_module = foveal_module
        if self._identity is not None:
            self._identity["heavy_runtime_loaded"] = True
            self._identity["checkpoint_global_step"] = checkpoint.get("global_step")
            self._identity["processor_id"] = processor_id or config.model_id

    def _capture_generated_focus(self, sample: NativeStage2Sample, *, force_prefix: bool) -> Any:
        from revisit_vlm.qwen3_vl_tgvf import (
            build_direct_messages,
            build_qwen3_inputs,
            capture_focus_single_pass_from_inputs_qwen3,
            capture_focus_single_pass_qwen3,
        )

        if self.model is None or self.processor is None:
            raise RuntimeError("native Stage2 model is not loaded")
        image = self._image(sample)
        forced_text = self._force_prefix_text(sample) if force_prefix else None
        if forced_text is not None:
            inputs = build_qwen3_inputs(
                self.processor,
                build_direct_messages(image, sample.prompt_question),
            )
            return capture_focus_single_pass_from_inputs_qwen3(
                model=self.model,
                tokenizer=self.processor.tokenizer,
                inputs=inputs,
                max_new_tokens=self._action_token_budget(),
                device=self.device,
                forced_prefix_text=forced_text,
                protocol=self.stage2_config.protocol,
                **self._sampling_options(),
            )
        return capture_focus_single_pass_qwen3(
            self.model,
            self.processor,
            image=image,
            question=sample.prompt_question,
            messages=build_direct_messages(image, sample.prompt_question),
            max_new_tokens=self._action_token_budget(),
            device=self.device,
            protocol=self.stage2_config.protocol,
            **self._sampling_options(),
        )

    def _capture_free_router(self, sample: NativeStage2Sample) -> Any:
        from revisit_vlm.qwen3_vl_tgvf import build_direct_messages, capture_focus_single_pass_qwen3

        if self.model is None or self.processor is None:
            raise RuntimeError("native Stage2 model is not loaded")
        image = self._image(sample)
        messages = build_direct_messages(image, sample.prompt_question)
        return capture_focus_single_pass_qwen3(
            self.model,
            self.processor,
            image=image,
            question=sample.prompt_question,
            messages=messages,
            max_new_tokens=self._action_token_budget(),
            device=self.device,
            force_action_prefix=False,
            protocol=self.stage2_config.protocol,
            **self._sampling_options(),
        )

    def _d_from_capture(
        self,
        sample: NativeStage2Sample,
        capture: Any,
        *,
        focus_source: str,
    ) -> Any:
        from revisit_vlm.tgvf_foveal import finalize_tgvf_output_with_frozen_qwen_merger

        if self.foveal_module is None or self.utility_model is None or self.processor is None:
            raise RuntimeError("native Stage2 foveal runtime is not loaded")
        tap, v_pre, _v_merge, deepstack_pre, _deepstack_features = self._vision_features(sample)
        if v_pre is None:
            raise RuntimeError(f"Qwen3 V_pre tap failed: {tap.errors}")
        output = self.foveal_module(
            target_hidden_states=capture.target_hidden_states.to(self.device),
            pre_merge_visual_tokens=v_pre.to(self.device),
            metadata={
                "target": capture.target_text,
                "stage": "clean_native_stage2_eval",
                "focus_source": focus_source,
                "qwen_model": self.utility_model,
                "processor": self.processor,
                "image": self._image(sample),
                "question": sample.prompt_question,
                "device": self.device,
                "deepstack_pre_merge_visual_tokens": [
                    item.to(self.device) for item in deepstack_pre
                ],
            },
        )
        output = finalize_tgvf_output_with_frozen_qwen_merger(self.utility_model, output)
        tokens = output.foveated_visual_tokens
        if self.backend_options.get("detach_d", True):
            tokens = tokens.detach()
            deepstack = (
                None
                if not output.deepstack_visual_embeds
                else [item.detach() for item in output.deepstack_visual_embeds]
            )
        else:
            deepstack = output.deepstack_visual_embeds
        return SimpleNamespace(tokens=tokens, deepstack_visual_embeds=deepstack)

    def _vision_features(self, sample: NativeStage2Sample) -> tuple[Any, Any, Any, list[Any], list[Any]]:
        from revisit_vlm.qwen3_vl_tgvf import tap_qwen3_vision_features_with_deepstack_premerge

        if self.utility_model is None or self.processor is None:
            raise RuntimeError("native Stage2 utility model is not loaded")
        config = self._run_config
        if config is None:
            raise RuntimeError("NativeStage2Engine.prepare must be called before vision tap")
        key = f"{_image_identity_text(sample.image)}|{config.max_image_resolution}"
        if key not in self.vision_cache:
            self.vision_cache[key] = tap_qwen3_vision_features_with_deepstack_premerge(
                self.utility_model,
                self.processor,
                image=self._image(sample),
                question=sample.prompt_question,
                device=self.device,
            )
        return self.vision_cache[key]

    def _append_visual_d(self, capture: Any, d: Any) -> Any:
        import torch

        from revisit_vlm.qwen3_vl_tgvf import (
            EVIDENCE_START,
            THINK_START,
            _append_source_image_grid,
            _bracketed_visual_token_ids,
            _chunk_position_ids_inherit_source_visual_positions,
            _chunk_position_ids_native_source_grid,
            _compute_qwen3_position_ids_for_sequence,
            _encode_text,
            _extend_attention,
            _full_mm_token_type_ids_for_append,
            _fvt_mm_token_type_ids,
            _next_position_ids_after_prefill,
            protocol_uses_evidence_tags,
            protocol_uses_think_tags,
            protocol_uses_tool_observation,
            render_tgvf_prefix_suffix,
        )

        if self.model is None or self.processor is None or self.utility_model is None:
            raise RuntimeError("native Stage2 model is not loaded")
        tokenizer = self.processor.tokenizer
        d_deepstack_visual_embeds = (
            getattr(d, "deepstack_visual_embeds", None)
            if self._d_deepstack_enabled()
            else None
        )
        d = getattr(d, "tokens", d)
        if not capture.capture_found:
            raise ValueError("capture must contain a valid focus span before FVT append")
        if d.ndim != 2:
            raise ValueError("foveated visual tokens must have shape [M, hidden_dim]")
        source_geometry = capture.source_visual_geometry
        if source_geometry is None:
            raise ValueError("capture is missing source visual geometry")
        source_token_count = int(source_geometry.source_visual_token_count)
        if int(d.shape[0]) != source_token_count:
            raise ValueError(
                f"FVT token count {int(d.shape[0])} != source token count {source_token_count}"
            )
        embed = self.model.get_input_embeddings()
        hidden_dim = int(embed.weight.shape[-1])
        if int(d.shape[-1]) != hidden_dim:
            raise ValueError(f"FVT dim {int(d.shape[-1])} != Qwen hidden dim {hidden_dim}")
        prefix, suffix = render_tgvf_prefix_suffix(
            protocol=self.stage2_config.protocol,
            include_leading_im_end=not protocol_uses_tool_observation(self.stage2_config.protocol),
        )
        if protocol_uses_think_tags(self.stage2_config.protocol):
            suffix += f"{THINK_START}\n"
        elif protocol_uses_evidence_tags(self.stage2_config.protocol):
            suffix += "<|evidence_start|>"
        else:
            suffix += EVIDENCE_START
        token_ids = _bracketed_visual_token_ids(
            self.processor,
            self.utility_model,
            num_fvt_tokens=int(d.shape[0]),
            prefix=prefix,
            suffix=suffix,
            device=self.device,
        )
        prefix_ids = _encode_text(tokenizer, prefix, self.device)
        fvt_token_start = int(prefix_ids.shape[0]) + 1
        fvt_token_end = fvt_token_start + int(d.shape[0])
        d_chunk_token_ids = token_ids.view(1, -1).to(self.device)
        d_chunk_embeds = embed(d_chunk_token_ids).detach().clone()
        d_chunk_embeds[0, fvt_token_start:fvt_token_end] = d.to(
            device=self.device,
            dtype=d_chunk_embeds.dtype,
        )
        base_attention_mask = (
            None if capture.attention_mask is None else capture.attention_mask.to(self.device)
        )
        base_input_ids = None if capture.input_ids is None else capture.input_ids.to(self.device)
        past_key_values = capture.past_key_values
        cache_alignment_debug: dict[str, Any] = {
            "kv_cache_input_len": (
                None if base_input_ids is None else int(base_input_ids.shape[-1])
            ),
            "kv_cache_initial_seq_len": _past_key_values_sequence_length(past_key_values),
            "kv_cache_tail_prefill_tokens": 0,
            "kv_cache_aligned_seq_len": _past_key_values_sequence_length(past_key_values),
            "kv_cache_tail_prefill_used": False,
            "kv_cache_tail_in_append_chunk": False,
        }
        tail_token_ids = None
        if self._deepstack_enabled():
            aligned_cache = self._align_capture_cache_to_input_ids(capture)
            past_key_values = aligned_cache.past_key_values
            base_attention_mask = aligned_cache.attention_mask
            base_input_ids = aligned_cache.input_ids
            tail_token_ids = aligned_cache.tail_token_ids
            cache_alignment_debug = aligned_cache.debug
        attention_mask = base_attention_mask
        if attention_mask is not None:
            attention_mask = _extend_attention(attention_mask, int(d_chunk_token_ids.shape[-1]))
        input_ids = base_input_ids
        if input_ids is not None:
            input_ids = torch.cat([input_ids, d_chunk_token_ids], dim=-1)
        mm_token_type_ids = _fvt_mm_token_type_ids(
            chunk_length=int(d_chunk_token_ids.shape[-1]),
            fvt_token_start=fvt_token_start,
            fvt_token_end=fvt_token_end,
            device=self.device,
        )
        model_token_ids = d_chunk_token_ids
        model_embeds = d_chunk_embeds
        model_mm_token_type_ids = mm_token_type_ids
        tail_len = 0 if tail_token_ids is None else int(tail_token_ids.shape[-1])
        if tail_len > 0:
            tail_token_ids = tail_token_ids.to(self.device)
            tail_embeds = embed(tail_token_ids).detach().clone()
            model_token_ids = torch.cat([tail_token_ids, d_chunk_token_ids], dim=-1)
            model_embeds = torch.cat([tail_embeds.to(dtype=d_chunk_embeds.dtype), d_chunk_embeds], dim=1)
            tail_mm_token_type_ids = torch.zeros(
                (1, tail_len),
                dtype=model_mm_token_type_ids.dtype,
                device=self.device,
            )
            model_mm_token_type_ids = torch.cat(
                [tail_mm_token_type_ids, model_mm_token_type_ids],
                dim=-1,
            )
        position_ids = _chunk_position_ids_native_source_grid(
            model=self.utility_model,
            capture=capture,
            token_ids=token_ids,
            attention_mask=attention_mask,
            chunk_mm_token_type_ids=mm_token_type_ids,
            fvt_token_start=fvt_token_start,
            fvt_token_end=fvt_token_end,
            source_geometry=source_geometry,
            device=self.device,
        )
        if position_ids is None and source_geometry.source_visual_position_ids is not None:
            position_ids = _chunk_position_ids_inherit_source_visual_positions(
                attention_mask=attention_mask,
                chunk_length=int(token_ids.shape[0]),
                visual_token_start=fvt_token_start,
                visual_token_end=fvt_token_end,
                source_visual_position_ids=source_geometry.source_visual_position_ids,
                device=self.device,
            )
        cache_seq_len = cache_alignment_debug.get("kv_cache_initial_seq_len")
        if self._deepstack_enabled() and cache_seq_len is not None:
            full_mm_token_type_ids = _full_mm_token_type_ids_for_append(
                model=self.utility_model,
                capture_input_ids=base_input_ids.to(self.device),
                chunk_mm_token_type_ids=mm_token_type_ids.to(self.device),
                device=self.device,
            )
            image_grid_thw = _append_source_image_grid(
                capture.image_grid_thw,
                source_geometry.image_grid_thw,
                device=self.device,
            )
            full_position_ids = _compute_qwen3_position_ids_for_sequence(
                model=self.utility_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_grid_thw=image_grid_thw,
                video_grid_thw=capture.video_grid_thw,
                mm_token_type_ids=full_mm_token_type_ids,
            )
            if full_position_ids is None:
                raise ValueError("Qwen3 model does not expose compute_3d_position_ids")
            position_ids = full_position_ids[:, :, int(cache_seq_len) :].to(device=self.device)
        append_attention = attention_mask
        block_original_image_keys = False
        original_positions = None
        if self._deepstack_blocks_original_image_keys():
            if past_key_values is None:
                raise ValueError("KV DeepStack append requires capture.past_key_values")
            if attention_mask is None or input_ids is None:
                raise ValueError("KV DeepStack append requires attention_mask and input_ids")
            if source_geometry.source_visual_token_indices is None:
                raise ValueError("KV DeepStack append requires source visual token indices")
            original_positions = source_geometry.source_visual_token_indices.to(self.device)
            chunk_length = int(model_token_ids.shape[-1])
            query_start = int(attention_mask.shape[-1]) - chunk_length
            if cache_seq_len is not None:
                query_start = int(cache_seq_len)
            cached_attention = attention_mask
            if cache_seq_len is not None and tail_len > 0:
                cached_prefix_attention = base_attention_mask[:, : int(cache_seq_len)]
                chunk_attention = torch.ones(
                    (cached_prefix_attention.shape[0], chunk_length),
                    dtype=cached_prefix_attention.dtype,
                    device=cached_prefix_attention.device,
                )
                cached_attention = torch.cat([cached_prefix_attention, chunk_attention], dim=-1)
            append_attention = build_cached_chunk_original_image_key_block_attention_mask(
                attention_mask_2d=cached_attention,
                original_image_token_indices=original_positions,
                query_start=query_start,
                query_length=chunk_length,
                block_query_offset=tail_len,
                dtype=model_embeds.dtype,
            )
            block_original_image_keys = True
        d_deepstack_payload = None
        if d_deepstack_visual_embeds:
            d_visual_pos_masks = torch.zeros(
                (1, int(model_token_ids.shape[-1])),
                dtype=torch.bool,
                device=self.device,
            )
            d_visual_pos_masks[0, tail_len + fvt_token_start : tail_len + fvt_token_end] = True
            d_deepstack_payload = {
                "visual_pos_masks": d_visual_pos_masks,
                "deepstack_visual_embeds": [
                    item.to(device=self.device, dtype=model_embeds.dtype)
                    for item in d_deepstack_visual_embeds
                ],
            }
            outputs = self._forward_qwen3_language_with_deepstack(
                inputs_embeds=model_embeds,
                attention_mask=append_attention,
                position_ids=position_ids,
                past_key_values=past_key_values,
                visual_pos_masks=d_deepstack_payload["visual_pos_masks"],
                deepstack_visual_embeds=d_deepstack_payload["deepstack_visual_embeds"],
                use_cache=True,
            )
        else:
            outputs = self.model(
                inputs_embeds=model_embeds,
                past_key_values=past_key_values,
                attention_mask=append_attention,
                position_ids=position_ids,
                mm_token_type_ids=model_mm_token_type_ids,
                use_cache=True,
                return_dict=True,
            )
        model_kwargs = dict(getattr(capture, "model_kwargs", {}) or {})
        next_position_ids = _next_position_ids_after_prefill(position_ids)
        if next_position_ids is not None:
            model_kwargs["tgvf_next_position_ids"] = next_position_ids.detach().cpu()
        if block_original_image_keys:
            model_kwargs["tgvf_block_original_image_keys"] = True
            model_kwargs["tgvf_original_image_token_indices"] = original_positions.detach().cpu()
            model_kwargs["tgvf_deepstack_scope"] = self._deepstack_scope().value
            model_kwargs["tgvf_deepstack_restore_for_answer"] = (
                self._deepstack_scope() == DeepStackScope.EVIDENCE_ONLY
            )
        return self._append_result_cls()(
            past_key_values=outputs.past_key_values,
            attention_mask=attention_mask,
            cache_position=None,
            input_ids=input_ids,
            last_logits=outputs.logits,
            appended_token_ids=token_ids.detach().cpu(),
            appended_inputs_embeds=model_embeds.detach().cpu(),
            fvt_token_start=fvt_token_start + tail_len,
            fvt_token_end=fvt_token_end + tail_len,
            model_kwargs=model_kwargs,
            debug_metadata={
                "fvt_append_path": "clean_native_qwen3_visual_special_tokens_embedding_replace",
                "tgvf_protocol": self.stage2_config.protocol,
                "fvt_shape": list(d.shape),
                "num_fvt_tokens": int(d.shape[0]),
                "model_append_chunk_length": int(model_token_ids.shape[-1]),
                "kv_cache_tail_chunk_tokens": tail_len,
                "source_visual_token_count": source_token_count,
                "fvt_position_mode": "native_source_grid",
                "position_ids_shape": (
                    list(position_ids.shape) if position_ids is not None else None
                ),
                "mm_token_type_ids_shape": list(model_mm_token_type_ids.shape),
                "append_attention_mask_shape": (
                    list(append_attention.shape) if append_attention is not None else None
                ),
                "native_qwen3_position_compute_used": True,
                "second_full_forward_used": False,
                "past_key_values_preserved": (
                    past_key_values is not None and outputs.past_key_values is not None
                ),
                **cache_alignment_debug,
                "deepstack_caution": None
                if self._deepstack_enabled()
                else (
                    "clean-native FVT append uses Qwen3 visual special tokens and real "
                    "3D positions, but does not provide native Qwen3 DeepStack visual "
                    "features."
                ),
                "uses_deepstack_for_fvt": bool(self._deepstack_enabled()),
                "uses_d_deepstack_for_fvt": bool(d_deepstack_payload is not None),
                "d_deepstack_feature_shapes": (
                    None
                    if d_deepstack_payload is None
                    else [list(item.shape) for item in d_deepstack_payload["deepstack_visual_embeds"]]
                ),
                "deepstack_cached_prefix_used": bool(self._deepstack_enabled()),
                "deepstack_scope": (
                    None if not self._deepstack_enabled() else self._deepstack_scope().value
                ),
                "deepstack_answer_restore_policy": (
                    None
                    if not self._deepstack_enabled()
                    else "not_blocked"
                    if not block_original_image_keys
                    else (
                        "restore_after_answer_boundary"
                        if self._deepstack_scope() == DeepStackScope.EVIDENCE_ONLY
                        else "blocked_through_answer"
                    )
                ),
                "deepstack_append_attention_mask": (
                    None
                    if not self._deepstack_enabled()
                    else "2d_no_original_image_key_block"
                    if not block_original_image_keys
                    else "4d_cached_chunk_original_image_key_block"
                ),
                "deepstack_prefix_source": (
                    None
                    if not self._deepstack_enabled()
                    else "native_qwen3_capture_past_key_values"
                ),
                "deepstack_original_image_key_block": bool(block_original_image_keys),
            },
        )

    def _align_capture_cache_to_input_ids(self, capture: Any) -> _AlignedCaptureCache:
        """Prepare a cached prefix plus any uncached tail tokens for D append.

        `generate(..., return_dict_in_generate=True)` may return a cache whose
        sequence length is one token shorter than `sequences`. The clean KV
        append keeps the cache unchanged and includes the missing tail token(s)
        in the same `inputs_embeds` chunk as D. Tail queries keep access to the
        original image; D queries use the configured original-image key block
        only for blocking DeepStack scopes.
        """

        if self.model is None:
            raise RuntimeError("native Stage2 model is not loaded")
        if capture.past_key_values is None:
            raise ValueError("KV DeepStack append requires capture.past_key_values")
        if capture.attention_mask is None or capture.input_ids is None:
            raise ValueError("KV DeepStack append requires attention_mask and input_ids")

        past_key_values = capture.past_key_values
        input_ids = capture.input_ids.to(self.device)
        attention_mask = capture.attention_mask.to(self.device)
        input_len = int(input_ids.shape[-1])
        attention_len = int(attention_mask.shape[-1])
        if attention_len != input_len:
            raise ValueError(
                "KV DeepStack append requires attention_mask/input_ids length match: "
                f"attention={attention_len} input_ids={input_len}"
            )
        initial_seq_len = _past_key_values_sequence_length(past_key_values)
        debug = {
            "kv_cache_input_len": input_len,
            "kv_cache_initial_seq_len": initial_seq_len,
            "kv_cache_tail_prefill_tokens": 0,
            "kv_cache_aligned_seq_len": initial_seq_len,
            "kv_cache_tail_prefill_used": False,
            "kv_cache_tail_in_append_chunk": False,
        }
        if initial_seq_len is None:
            return _AlignedCaptureCache(
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                input_ids=input_ids,
                tail_token_ids=None,
                debug=debug,
            )
        if initial_seq_len > input_len:
            raise ValueError(
                "KV DeepStack cache is longer than captured input_ids: "
                f"cache={initial_seq_len} input_ids={input_len}"
            )
        if initial_seq_len == input_len:
            debug["kv_cache_aligned_seq_len"] = input_len
            return _AlignedCaptureCache(
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                input_ids=input_ids,
                tail_token_ids=None,
                debug=debug,
            )

        missing = input_len - initial_seq_len
        tail_token_ids = input_ids[:, initial_seq_len:]
        debug.update(
            kv_cache_tail_prefill_tokens=missing,
            kv_cache_aligned_seq_len=initial_seq_len,
            kv_cache_tail_prefill_used=False,
            kv_cache_tail_in_append_chunk=True,
        )
        return _AlignedCaptureCache(
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            input_ids=input_ids,
            tail_token_ids=tail_token_ids,
            debug=debug,
        )

    def _append_visual_d_full_sequence(
        self,
        sample: NativeStage2Sample,
        capture: Any,
        d: Any,
    ) -> Any:
        import torch

        from revisit_vlm.qwen3_vl_tgvf import (
            EVIDENCE_START,
            THINK_START,
            _append_source_image_grid,
            _bracketed_visual_token_ids,
            _chunk_position_ids_inherit_source_visual_positions,
            _compute_qwen3_position_ids_for_sequence,
            _encode_text,
            _full_mm_token_type_ids_for_append,
            _fvt_mm_token_type_ids,
            _next_position_ids_after_prefill,
            protocol_uses_evidence_tags,
            protocol_uses_think_tags,
            protocol_uses_tool_observation,
            render_tgvf_prefix_suffix,
        )

        if self.model is None or self.utility_model is None or self.processor is None:
            raise RuntimeError("native Stage2 model is not loaded")
        d_deepstack_visual_embeds = (
            getattr(d, "deepstack_visual_embeds", None)
            if self._d_deepstack_enabled()
            else None
        )
        d = getattr(d, "tokens", d)
        if capture.input_ids is None:
            raise ValueError("capture.input_ids is required for full-sequence prefill")
        source_geometry = capture.source_visual_geometry
        if source_geometry is None or source_geometry.source_visual_token_indices is None:
            raise ValueError(
                "capture source visual token indices are required for full-sequence prefill"
            )
        if d.ndim != 2:
            raise ValueError("foveated visual tokens must have shape [M, hidden_dim]")
        source_token_count = int(source_geometry.source_visual_token_count)
        if int(d.shape[0]) != source_token_count:
            raise ValueError(
                f"FVT token count {int(d.shape[0])} != source token count {source_token_count}"
            )
        embed = self.model.get_input_embeddings()
        hidden_dim = int(embed.weight.shape[-1])
        if int(d.shape[-1]) != hidden_dim:
            raise ValueError(f"FVT dim {int(d.shape[-1])} != Qwen hidden dim {hidden_dim}")
        prefix, suffix = render_tgvf_prefix_suffix(
            protocol=self.stage2_config.protocol,
            include_leading_im_end=not protocol_uses_tool_observation(self.stage2_config.protocol),
        )
        if protocol_uses_think_tags(self.stage2_config.protocol):
            suffix += f"{THINK_START}\n"
        elif protocol_uses_evidence_tags(self.stage2_config.protocol):
            suffix += "<|evidence_start|>"
        else:
            suffix += EVIDENCE_START
        token_ids = _bracketed_visual_token_ids(
            self.processor,
            self.utility_model,
            num_fvt_tokens=int(d.shape[0]),
            prefix=prefix,
            suffix=suffix,
            device=self.device,
        ).view(1, -1)
        prefix_ids = _encode_text(self.processor.tokenizer, prefix, self.device)
        fvt_token_start = int(prefix_ids.shape[0]) + 1
        fvt_token_end = fvt_token_start + int(d.shape[0])
        capture_input_ids = capture.input_ids.to(self.device)
        full_input_ids = torch.cat([capture_input_ids, token_ids.to(self.device)], dim=-1)
        full_attention = torch.ones_like(full_input_ids)
        embeds = embed(full_input_ids).detach().clone()
        _tap, _v_pre, v_merge, _deepstack_pre, _deepstack_features = self._vision_features(sample)
        if v_merge is None:
            raise RuntimeError("source merged visual features are unavailable")
        original_positions = source_geometry.source_visual_token_indices.to(self.device)
        if int(original_positions.numel()) != int(v_merge.shape[0]):
            raise ValueError(
                f"source visual token count mismatch: positions={int(original_positions.numel())} "
                f"v_merge={int(v_merge.shape[0])}"
            )
        embeds[0, original_positions, :] = v_merge.to(device=self.device, dtype=embeds.dtype)
        chunk_mm_token_type_ids = _fvt_mm_token_type_ids(
            chunk_length=int(token_ids.shape[-1]),
            fvt_token_start=fvt_token_start,
            fvt_token_end=fvt_token_end,
            device=self.device,
        )
        full_fvt_start = int(capture_input_ids.shape[-1]) + int(fvt_token_start)
        full_fvt_end = int(capture_input_ids.shape[-1]) + int(fvt_token_end)
        embeds[0, full_fvt_start:full_fvt_end, :] = d.to(device=self.device, dtype=embeds.dtype)
        image_grid_thw = _append_source_image_grid(
            capture.image_grid_thw,
            source_geometry.image_grid_thw,
            device=self.device,
        )
        full_mm_token_type_ids = _full_mm_token_type_ids_for_append(
            model=self.utility_model,
            capture_input_ids=capture_input_ids,
            chunk_mm_token_type_ids=chunk_mm_token_type_ids,
            device=self.device,
        )
        position_ids_mode = "native_source_grid"
        position_ids_fallback_error = None
        try:
            position_ids = _compute_qwen3_position_ids_for_sequence(
                model=self.utility_model,
                input_ids=full_input_ids,
                attention_mask=full_attention,
                image_grid_thw=image_grid_thw,
                video_grid_thw=capture.video_grid_thw,
                mm_token_type_ids=full_mm_token_type_ids,
            )
        except RuntimeError as exc:
            if (
                _image_grid_count(source_geometry.image_grid_thw) > 1
                and source_geometry.source_visual_position_ids is not None
            ):
                base_attention = torch.ones_like(capture_input_ids)
                capture_model_kwargs = dict(getattr(capture, "model_kwargs", {}) or {})
                base_mm_token_type_ids = capture_model_kwargs.get("mm_token_type_ids")
                if hasattr(base_mm_token_type_ids, "to"):
                    base_mm_token_type_ids = base_mm_token_type_ids.to(self.device)
                    base_len = int(base_mm_token_type_ids.shape[-1])
                    target_len = int(capture_input_ids.shape[-1])
                    if base_len < target_len:
                        zeros = torch.zeros(
                            (base_mm_token_type_ids.shape[0], target_len - base_len),
                            dtype=base_mm_token_type_ids.dtype,
                            device=base_mm_token_type_ids.device,
                        )
                        base_mm_token_type_ids = torch.cat([base_mm_token_type_ids, zeros], dim=-1)
                    elif base_len > target_len:
                        raise ValueError(
                            "capture mm_token_type_ids length exceeds capture input length: "
                            f"{base_len} > {target_len}"
                        )
                base_position_ids = _compute_qwen3_position_ids_for_sequence(
                    model=self.utility_model,
                    input_ids=capture_input_ids,
                    attention_mask=base_attention,
                    image_grid_thw=capture.image_grid_thw,
                    video_grid_thw=capture.video_grid_thw,
                    mm_token_type_ids=base_mm_token_type_ids,
                )
                if base_position_ids is None:
                    raise ValueError(
                        "base position id computation failed for multi-image full-sequence prefill"
                    ) from exc
                chunk_position_ids = _chunk_position_ids_inherit_source_visual_positions(
                    attention_mask=full_attention,
                    chunk_length=int(token_ids.shape[-1]),
                    visual_token_start=fvt_token_start,
                    visual_token_end=fvt_token_end,
                    source_visual_position_ids=source_geometry.source_visual_position_ids,
                    device=self.device,
                )
                if chunk_position_ids is None:
                    raise ValueError(
                        "chunk position id computation failed for multi-image full-sequence prefill"
                    ) from exc
                position_ids = torch.cat(
                    [
                        base_position_ids.to(device=self.device),
                        chunk_position_ids.to(device=self.device),
                    ],
                    dim=-1,
                )
                position_ids_mode = "multi_image_inherit_source_visual_positions"
                position_ids_fallback_error = f"{type(exc).__name__}: {exc}"
            else:
                raise
        if position_ids is None:
            raise ValueError("position id computation failed for full-sequence prefill")
        deepstack_payload = None
        prefill_attention = full_attention
        block_original_image_keys = False
        if self._deepstack_enabled():
            scope = self._deepstack_scope()
            deepstack_features = self._original_image_deepstack_features(sample)
            d_token_indices = torch.arange(
                full_fvt_start,
                full_fvt_end,
                dtype=torch.long,
                device=self.device,
            )
            deepstack_payload = build_qwen3_deepstack_payload(
                sequence_length=int(full_input_ids.shape[-1]),
                original_image_token_indices=original_positions,
                original_deepstack_features=deepstack_features,
                d_token_indices=d_token_indices if d_deepstack_visual_embeds else None,
                d_deepstack_features=d_deepstack_visual_embeds,
                device=self.device,
                dtype=embeds.dtype,
                visual_pos_mask_policy=(
                    "original_and_d_tokens"
                    if d_deepstack_visual_embeds
                    else "original_image_tokens_only"
                ),
            )
            if self._deepstack_blocks_original_image_keys():
                prefill_attention = build_original_image_key_block_attention_mask(
                    attention_mask_2d=full_attention,
                    original_image_token_indices=original_positions,
                    block_query_start=int(capture_input_ids.shape[-1]),
                    dtype=embeds.dtype,
                    block_query_end=None,
                )
                block_original_image_keys = True
            outputs = self._forward_qwen3_language_with_deepstack(
                inputs_embeds=embeds,
                attention_mask=prefill_attention,
                position_ids=position_ids,
                past_key_values=None,
                visual_pos_masks=deepstack_payload.visual_pos_masks,
                deepstack_visual_embeds=deepstack_payload.deepstack_visual_embeds,
                use_cache=True,
            )
        else:
            outputs = self.model(
                inputs_embeds=embeds,
                attention_mask=full_attention,
                position_ids=position_ids,
                mm_token_type_ids=full_mm_token_type_ids,
                use_cache=True,
                return_dict=True,
            )
        model_kwargs: dict[str, Any] = {}
        next_position_ids = _next_position_ids_after_prefill(position_ids)
        if next_position_ids is not None:
            model_kwargs["tgvf_next_position_ids"] = next_position_ids.detach().cpu()
        if block_original_image_keys:
            model_kwargs["tgvf_block_original_image_keys"] = True
            model_kwargs["tgvf_original_image_token_indices"] = original_positions.detach().cpu()
            model_kwargs["tgvf_deepstack_scope"] = scope.value
            model_kwargs["tgvf_deepstack_restore_for_answer"] = (
                scope == DeepStackScope.EVIDENCE_ONLY
            )
        return self._append_result_cls()(
            past_key_values=outputs.past_key_values,
            attention_mask=full_attention,
            cache_position=None,
            input_ids=full_input_ids,
            last_logits=outputs.logits,
            appended_token_ids=token_ids.detach().cpu(),
            appended_inputs_embeds=torch.empty(0),
            fvt_token_start=fvt_token_start,
            fvt_token_end=fvt_token_end,
            model_kwargs=model_kwargs,
            debug_metadata={
                "fvt_append_path": "clean_native_full_sequence_prefill",
                "tgvf_protocol": self.stage2_config.protocol,
                "uses_deepstack_for_fvt": bool(deepstack_payload is not None),
                "uses_d_deepstack_for_fvt": bool(
                    deepstack_payload is not None and d_deepstack_visual_embeds
                ),
                "fvt_shape": list(d.shape),
                "num_fvt_tokens": int(d.shape[0]),
                "source_visual_token_count": int(source_geometry.source_visual_token_count),
                "fvt_position_mode": position_ids_mode,
                "position_ids_shape": list(position_ids.shape),
                "position_ids_fallback_error": position_ids_fallback_error,
                "mm_token_type_ids_shape": list(full_mm_token_type_ids.shape),
                "native_qwen3_position_compute_used": position_ids_fallback_error is None,
                "second_full_forward_used": True,
                "past_key_values_preserved": False,
                "deepstack_caution": (
                    None
                    if deepstack_payload is not None
                    else (
                        "clean-native full-sequence FVT append uses Qwen3 visual special "
                        "tokens and real 3D positions, but does not provide native Qwen3 "
                        "DeepStack visual features."
                    )
                ),
                "deepstack_payload": (
                    None if deepstack_payload is None else deepstack_payload.to_debug_dict()
                ),
                "deepstack_scope": (
                    None if deepstack_payload is None else self._deepstack_scope().value
                ),
                "deepstack_answer_restore_policy": (
                    None
                    if deepstack_payload is None
                    else "not_blocked"
                    if not block_original_image_keys
                    else (
                        "restore_after_answer_boundary"
                        if self._deepstack_scope() == DeepStackScope.EVIDENCE_ONLY
                        else "blocked_through_answer"
                    )
                ),
                "deepstack_prefill_attention_mask": (
                    None
                    if deepstack_payload is None
                    else "2d_no_original_image_key_block"
                    if not block_original_image_keys
                    else "4d_original_image_key_block"
                ),
                "deepstack_original_image_key_block": bool(block_original_image_keys),
            },
        )

    def _continue_generation(self, append_result: Any, *, capture: Any | None = None) -> Any:
        from revisit_vlm.qwen3_vl_tgvf import continue_generation_qwen3

        if self.model is None or self.processor is None:
            raise RuntimeError("native Stage2 model is not loaded")
        max_new_tokens = self._answer_token_budget(capture)
        if max_new_tokens <= 0:
            return self._empty_continuation(append_result, stop_reason="max_tokens_exhausted")
        if bool((append_result.model_kwargs or {}).get("tgvf_block_original_image_keys")):
            return self._continue_generation_blocking_original_image_keys(
                append_result,
                max_new_tokens=max_new_tokens,
            )
        return continue_generation_qwen3(
            self.model,
            self.processor,
            append_result,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **self._sampling_options(),
        )

    def _action_token_budget(self) -> int:
        return int(self.stage2_config.max_tokens or self.stage2_config.max_action_tokens)

    def _answer_token_budget(self, capture: Any | None) -> int:
        if self.stage2_config.max_tokens is None:
            return int(self.stage2_config.max_answer_tokens)
        action_tokens = len(getattr(capture, "generated_ids", []) or [])
        return max(0, int(self.stage2_config.max_tokens) - int(action_tokens))

    def _token_budget_debug(self, capture: Any | None, continuation: Any | None) -> dict[str, Any]:
        action_tokens = len(getattr(capture, "generated_ids", []) or [])
        answer_tokens = len(getattr(continuation, "generated_ids", []) or [])
        max_tokens = self.stage2_config.max_tokens
        return {
            "schema_version": "clean_stage2_unified_token_budget_v1",
            "token_budget_policy": (
                "unified_max_tokens" if max_tokens is not None else "legacy_split"
            ),
            "max_tokens": max_tokens,
            "legacy_split_limits": {
                "active": max_tokens is None,
                "max_action_tokens": self.stage2_config.max_action_tokens,
                "max_answer_tokens": self.stage2_config.max_answer_tokens,
            },
            "action_budget": self._action_token_budget(),
            "action_tokens": action_tokens,
            "answer_budget": self._answer_token_budget(capture),
            "answer_tokens": answer_tokens,
            "total_generated_tokens": action_tokens + answer_tokens,
            "unified_budget_active": max_tokens is not None,
        }

    def _empty_continuation(self, append_result: Any, *, stop_reason: str) -> Any:
        from revisit_vlm.qwen3_vl_tgvf import Qwen3Continuation

        return Qwen3Continuation(
            generated_ids=[],
            generated_text="",
            past_key_values=append_result.past_key_values,
            attention_mask=append_result.attention_mask,
            input_ids=append_result.input_ids,
            last_logits=append_result.last_logits,
            stop_reason=stop_reason,
            generated_logprobs=[],
        )

    def _sampling_options(self) -> dict[str, Any]:
        return {
            "do_sample": bool(self.backend_options.get("do_sample", False)),
            "temperature": float(self.backend_options.get("temperature", 1.0)),
            "top_p": float(self.backend_options.get("top_p", 1.0)),
        }

    def _continue_generation_blocking_original_image_keys(
        self,
        append_result: Any,
        *,
        max_new_tokens: int | None = None,
    ) -> Any:
        import torch

        from revisit_vlm.qwen3_vl_tgvf import Qwen3Continuation, _select_next_token

        if self.model is None or self.processor is None:
            raise RuntimeError("native Stage2 model is not loaded")
        tokenizer = self.processor.tokenizer
        logits = append_result.last_logits
        past_key_values = append_result.past_key_values
        attention_mask = append_result.attention_mask
        input_ids = append_result.input_ids
        model_kwargs = dict(append_result.model_kwargs or {})
        next_position_ids = model_kwargs.get("tgvf_next_position_ids")
        original_indices = model_kwargs.get("tgvf_original_image_token_indices")
        restore_for_answer = bool(model_kwargs.get("tgvf_deepstack_restore_for_answer"))
        if logits is None or attention_mask is None or input_ids is None:
            raise ValueError("DeepStack continuation requires logits, input_ids, and 2D attention")
        if next_position_ids is None or original_indices is None:
            raise ValueError("DeepStack continuation is missing position or original-key metadata")
        generated_ids: list[int] = []
        generated_logprobs: list[float] = []
        stop_reason = "max_new_tokens"
        eos_token_id = tokenizer.eos_token_id
        device = logits.device
        param_dtype = next(self.model.parameters()).dtype
        sampling = self._sampling_options()
        blocked_focus_start_ids: list[int] = []
        for marker in (
            "<|focus_start|>",
            "<|focus_end|>",
            "<FOCUS>",
            "</FOCUS>",
            "<tool_call>",
            "</tool_call>",
        ):
            ids = tokenizer.encode(marker, add_special_tokens=False)
            if len(ids) == 1:
                blocked_focus_start_ids.append(int(ids[0]))
        if max_new_tokens is None:
            max_new_tokens = int(self.stage2_config.max_answer_tokens)
        if max_new_tokens <= 0:
            return self._empty_continuation(append_result, stop_reason="max_tokens_exhausted")
        for _ in range(max_new_tokens):
            step_logits = logits[:, -1, :]
            if blocked_focus_start_ids:
                step_logits = step_logits.clone()
                step_logits[:, blocked_focus_start_ids] = -torch.inf
            next_token, selected_logprob = _select_next_token(step_logits, **sampling)
            restore_original_image_keys = restore_for_answer and _generated_answer_has_started(
                tokenizer,
                generated_ids,
                protocol=self.stage2_config.protocol,
            )
            token_id = int(next_token[0, 0].detach().cpu().item())
            generated_ids.append(token_id)
            generated_logprobs.append(float(selected_logprob))
            input_ids = torch.cat([input_ids.to(device), next_token.to(device)], dim=-1)
            next_attention = torch.ones(
                (attention_mask.shape[0], 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([attention_mask.to(device), next_attention], dim=-1)
            position_ids = next_position_ids.to(device=next_token.device) + (len(generated_ids) - 1)
            cache_position = torch.arange(
                attention_mask.shape[-1] - 1,
                attention_mask.shape[-1],
                device=next_token.device,
                dtype=torch.long,
            )
            step_attention = (
                attention_mask
                if restore_original_image_keys
                else build_single_query_original_image_key_block_attention_mask(
                    attention_mask_2d=attention_mask,
                    original_image_token_indices=original_indices,
                    dtype=param_dtype,
                )
            )
            outputs = self.model(
                input_ids=next_token.to(device),
                past_key_values=past_key_values,
                attention_mask=step_attention,
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits
            if eos_token_id is not None and token_id == eos_token_id:
                stop_reason = "eos_token"
                break
        return Qwen3Continuation(
            generated_ids=generated_ids,
            generated_text=tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ),
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            input_ids=input_ids,
            last_logits=logits,
            stop_reason=stop_reason,
            generated_logprobs=generated_logprobs,
        )

    def teacher_forced_continue_logprobs(
        self,
        append_result: Any,
        *,
        generated_token_ids: list[int],
    ) -> Any:
        from revisit_vlm.qwen3_vl_tgvf import teacher_forced_continue_logprobs_qwen3

        if self.model is None or self.processor is None:
            raise RuntimeError("native Stage2 model is not loaded")
        if bool((append_result.model_kwargs or {}).get("tgvf_block_original_image_keys")):
            return self._teacher_forced_continue_logprobs_blocking_original_image_keys(
                append_result,
                generated_token_ids=generated_token_ids,
            )
        return teacher_forced_continue_logprobs_qwen3(
            self.model,
            self.processor,
            append_result,
            generated_token_ids=generated_token_ids,
        )

    def _teacher_forced_continue_logprobs_blocking_original_image_keys(
        self,
        append_result: Any,
        *,
        generated_token_ids: list[int],
    ) -> Any:
        import torch

        if self.model is None or self.processor is None:
            raise RuntimeError("native Stage2 model is not loaded")
        tokenizer = self.processor.tokenizer
        logits = append_result.last_logits
        if logits is None:
            raise ValueError("DeepStack teacher-forced continuation replay requires state.last_logits")
        past_key_values = append_result.past_key_values
        attention_mask = append_result.attention_mask
        input_ids = append_result.input_ids
        model_kwargs = dict(append_result.model_kwargs or {})
        next_position_ids = model_kwargs.get("tgvf_next_position_ids")
        original_indices = model_kwargs.get("tgvf_original_image_token_indices")
        restore_for_answer = bool(model_kwargs.get("tgvf_deepstack_restore_for_answer"))
        if attention_mask is None or input_ids is None:
            raise ValueError("DeepStack teacher-forced continuation requires input_ids and 2D attention")
        if next_position_ids is None or original_indices is None:
            raise ValueError("DeepStack teacher-forced continuation is missing position or original-key metadata")
        device = logits.device
        param_dtype = next(self.model.parameters()).dtype
        blocked_focus_start_ids: list[int] = []
        for marker in (
            "<|focus_start|>",
            "<|focus_end|>",
            "<FOCUS>",
            "</FOCUS>",
            "<tool_call>",
            "</tool_call>",
        ):
            ids = tokenizer.encode(marker, add_special_tokens=False)
            if len(ids) == 1:
                blocked_focus_start_ids.append(int(ids[0]))
        generated_ids: list[int] = []
        logprob_rows: list[torch.Tensor] = []
        for index, token_id in enumerate(generated_token_ids):
            step_logits = logits[:, -1, :]
            if blocked_focus_start_ids:
                step_logits = step_logits.clone()
                step_logits[:, blocked_focus_start_ids] = -torch.inf
            token = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
            logprob_rows.append(
                torch.log_softmax(step_logits.float(), dim=-1).gather(-1, token).view(())
            )
            restore_original_image_keys = restore_for_answer and _generated_answer_has_started(
                tokenizer,
                generated_ids,
                protocol=self.stage2_config.protocol,
            )
            generated_ids.append(int(token_id))
            input_ids = torch.cat([input_ids.to(device), token.to(device)], dim=-1)
            next_attention = torch.ones(
                (attention_mask.shape[0], 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([attention_mask.to(device), next_attention], dim=-1)
            position_ids = next_position_ids.to(device=token.device) + index
            cache_position = torch.arange(
                attention_mask.shape[-1] - 1,
                attention_mask.shape[-1],
                device=token.device,
                dtype=torch.long,
            )
            step_attention = (
                attention_mask
                if restore_original_image_keys
                else build_single_query_original_image_key_block_attention_mask(
                    attention_mask_2d=attention_mask,
                    original_image_token_indices=original_indices,
                    dtype=param_dtype,
                )
            )
            outputs = self.model(
                input_ids=token.to(device),
                past_key_values=past_key_values,
                attention_mask=step_attention,
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits
        if not logprob_rows:
            return torch.empty((0,), dtype=torch.float32, device=device)
        return torch.stack(logprob_rows)

    def _parse_action(self, text: str) -> Any:
        from revisit_vlm.qwen3_vl_tgvf import parse_v3_action

        return parse_v3_action(text, protocol=self.stage2_config.protocol)

    def _image(self, sample: NativeStage2Sample) -> Any:
        from revisit_vlm.tgvf_v3_stage1 import _image_input

        config = self._run_config
        if config is None:
            raise RuntimeError("NativeStage2Engine.prepare must be called before image load")
        if isinstance(sample.image, (list, tuple)):
            return [
                _image_input(str(image), max_image_resolution=config.max_image_resolution)
                for image in sample.image
            ]
        return _image_input(str(sample.image), max_image_resolution=config.max_image_resolution)

    def _force_prefix_text(self, sample: NativeStage2Sample) -> str:
        from revisit_vlm.qwen3_vl_tgvf import render_force_focus_prefix

        return render_force_focus_prefix(
            protocol=self.stage2_config.protocol,
            target_hint=(
                sample.target if self.stage2_config.force_prefix_mode == "target_hint" else None
            ),
        )

    def _append_result_cls(self) -> Any:
        from revisit_vlm.qwen3_vl_tgvf import Qwen3AppendResult

        return Qwen3AppendResult

    def _deepstack_enabled(self) -> bool:
        config = self._run_config
        return bool(config is not None and config.deepstack.enabled)

    def _d_deepstack_enabled(self) -> bool:
        config = self._run_config
        return bool(config is not None and config.deepstack.d_features_enabled)

    def _deepstack_scope(self) -> DeepStackScope:
        config = self._run_config
        if config is None:
            return DeepStackScope.OFF
        return config.deepstack.original_image_scope

    def _deepstack_blocks_original_image_keys(self) -> bool:
        return self._deepstack_scope() in {
            DeepStackScope.THROUGH_ANSWER,
            DeepStackScope.EVIDENCE_ONLY,
        }

    def _original_image_deepstack_features(self, sample: NativeStage2Sample) -> list[Any]:
        from revisit_vlm.qwen3_vl_tgvf import build_direct_messages, build_qwen3_inputs

        if self.utility_model is None or self.processor is None:
            raise RuntimeError("native Stage2 utility model is not loaded")
        config = self._run_config
        if config is None:
            raise RuntimeError("NativeStage2Engine.prepare must be called before DeepStack capture")
        key = f"{sample.image}|{config.max_image_resolution}|deepstack"
        if key not in self.deepstack_cache:
            image = self._image(sample)
            inputs = build_qwen3_inputs(
                self.processor,
                build_direct_messages(image, sample.prompt_question),
            )
            model_inputs = {
                name: value.to(self.device) if hasattr(value, "to") else value
                for name, value in dict(inputs).items()
            }
            self.deepstack_cache[key] = [
                item.detach().cpu()
                for item in capture_qwen3_original_image_deepstack_features(
                    self.utility_model,
                    model_inputs,
                    detach=True,
                )
            ]
        return self.deepstack_cache[key]

    def _forward_qwen3_language_with_deepstack(
        self,
        *,
        inputs_embeds: Any,
        attention_mask: Any,
        position_ids: Any,
        past_key_values: Any,
        visual_pos_masks: Any,
        deepstack_visual_embeds: list[Any],
        use_cache: bool,
    ) -> Any:
        if self.model is None:
            raise RuntimeError("native Stage2 model is not loaded")
        causal_lm = _unwrap_qwen3_causal_lm(self.model)
        vl_model = getattr(causal_lm, "model", None)
        language_model = getattr(vl_model, "language_model", None)
        lm_head = getattr(causal_lm, "lm_head", None)
        if language_model is None or lm_head is None:
            raise RuntimeError("Qwen3 DeepStack runtime requires model.language_model and lm_head")
        outputs = language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        logits = lm_head(outputs.last_hidden_state)
        return SimpleNamespace(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=getattr(outputs, "hidden_states", None),
        )

    def _base_debug(self, sample: NativeStage2Sample, *, block: str, method: str) -> dict[str, Any]:
        return {
            "id": _sample_uid(sample),
            "block": block,
            "method": method,
            "tgvf_protocol": self.stage2_config.protocol,
            "image": sample.image,
            "image_id": sample.image_id,
            "source_dataset": sample.source_dataset,
            "source_profile": sample.source_profile,
            "need_focus": sample.need_focus,
            "question": sample.prompt_question,
            "expected_target": sample.target,
            "expected_evidence": sample.evidence_description,
            "expected_answer": sample.answer,
            "answer_format": sample.answer_format,
            "evidence_type": sample.evidence_type,
            "target_style": sample.target_style,
            "target_cues": sample.target_cues,
            "raw_output": "",
            "focus_raw_output": "",
            "final_raw_output": "",
            "parsed_evidence_state": None,
            "parsed_focus_target": "",
            "parsed_evidence": "",
            "parsed_answer": "",
            "malformed": False,
            "answer_parse_success": False,
            "target_length": 0,
            "generic_target_flag": False,
            "target_answer_leakage_flag": False,
            "trigger_focus_decision": False,
            "focus_miss": False,
            "no_focus_false_trigger": False,
            "continuation_not_im_end": False,
            "second_full_forward_used": False,
            "H_q_shape": None,
            "D_shape": None,
            "mask_mode": None,
            "append_success": False,
            "errors": [],
        }

    def _capture_fields(self, capture: Any, parsed: Any) -> dict[str, Any]:
        target = getattr(capture, "target_text", "") or parsed.focus_target
        target_hidden_states = getattr(capture, "target_hidden_states", None)
        source_geometry = getattr(capture, "source_visual_geometry", None)
        return {
            "focus_raw_output": capture.generated_text,
            "raw_output": capture.generated_text,
            "parsed_evidence_state": parsed.evidence_state,
            "parsed_focus_target": target,
            "malformed": bool(getattr(capture, "malformed", False) or parsed.malformed),
            "target_length": len(str(target).split()),
            "generic_target_flag": False,
            "target_answer_leakage_flag": False,
            "focus_valid": bool(
                getattr(capture, "capture_found", False)
                and not getattr(capture, "malformed", False)
            ),
            "second_full_forward_used": bool(getattr(capture, "second_full_forward_used", False)),
            "H_q_shape": _shape_list(target_hidden_states),
            "target_token_count": len(getattr(capture, "target_token_ids", []) or []),
            "focus_generated_ids": list(getattr(capture, "generated_ids", []) or []),
            "focus_generated_logprobs": list(getattr(capture, "generated_logprobs", []) or []),
            "source_visual_token_count": (
                source_geometry.source_visual_token_count if source_geometry is not None else 0
            ),
            "capture_stop_reason": getattr(capture, "stop_reason", ""),
            "capture_errors": list(getattr(capture, "errors", []) or []),
        }

    @staticmethod
    def _need_local_evidence() -> str:
        return "need_local_visual_evidence"


def stage2_sample_from_clean_sample(
    sample: BenchmarkSample,
    rendered: RenderedBenchmarkInput,
) -> NativeStage2Sample:
    image_paths, media_report = _loadable_image_paths(sample)
    image: str | list[str] = image_paths[0] if len(image_paths) == 1 else image_paths
    return NativeStage2Sample(
        image=image,
        question=rendered.user_prompt,
        answer=sample.gold_answer or "",
        need_focus=True,
        evidence_state="need_local_visual_evidence",
        trajectory_type="single_focus",
        target="",
        evidence_description="",
        image_id=sample.sample_id,
        choices=None,
        answer_format="multiple_choice" if sample.choices else "open",
        source_dataset=sample.benchmark,
        source_profile=str(sample.metadata.get("category") or sample.population_id),
        metadata={
            "population_id": sample.population_id,
            "source_file": sample.source_file,
            "choices": list(sample.choices),
            "image_input_count": len(image_paths),
            "image_input_mode": "single_image" if len(image_paths) == 1 else "multi_image",
            "image_materialization": media_report,
            **sample.metadata,
        },
    )


def _validate_run_alignment(stage2_config: Stage2RuntimeConfig, config: RunConfig) -> None:
    config.validate()
    if config.mode not in SUPPORTED_NATIVE_STAGE2_MODES:
        raise NotImplementedError(
            f"clean-native Stage2 backend supports only TGVF modes, got {config.mode.value!r}"
        )
    if stage2_config.protocol != config.tgvf_protocol:
        raise ValueError(
            "Stage2 runtime protocol does not match run config: "
            f"{stage2_config.protocol!r} != {config.tgvf_protocol!r}"
        )
    if stage2_config.append_forward_mode != config.post_tgvf_forward_mode:
        raise ValueError(
            "Stage2 append_forward_mode does not match run config: "
            f"{stage2_config.append_forward_mode.value!r} != "
            f"{config.post_tgvf_forward_mode.value!r}"
        )


def _validate_d_deepstack_checkpoint_support(
    config: RunConfig,
    checkpoint_config: dict[str, Any],
) -> None:
    if not config.deepstack.d_features_enabled:
        return
    tgvf_cfg = checkpoint_config.get("tgvf") or {}
    if bool(tgvf_cfg.get("d_deepstack_enabled", False)):
        return
    raise ValueError(
        "RunConfig requests D DeepStack features, but the Stage2 checkpoint "
        "was not trained with d_deepstack_enabled=true"
    )


def _loadable_image_paths(sample: BenchmarkSample) -> tuple[list[str], list[dict[str, Any]]]:
    resolved_images: list[tuple[str, dict[str, Any]]] = []
    for media_index, media in _coalesce_duplicate_decoded_image_media_refs(sample.media):
        resolved = _loadable_image_path(sample, media, media_index=media_index)
        if resolved is None:
            continue
        path, report = resolved
        resolved_images.append((path, report))
    resolved_images = _coalesce_duplicate_decoded_image_media(resolved_images)
    image_paths = [path for path, _report in resolved_images]
    media_report = [report for _path, report in resolved_images]
    if not image_paths:
        raise ValueError(
            "clean-native Stage2 requires loadable image media; "
            f"sample {sample.sample_id} has media kinds "
            f"{[item.get('kind') for item in sample.media]}"
        )
    return image_paths, media_report


def _coalesce_duplicate_decoded_image_media_refs(
    media_refs: tuple[dict[str, Any], ...],
) -> list[tuple[int, dict[str, Any]]]:
    existing_path_basenames = {
        Path(str(item.get("path"))).name
        for item in media_refs
        if item.get("source_key") == "image"
        and item.get("kind") == "path"
        and item.get("path")
        and (item.get("exists") is True or Path(str(item.get("path"))).exists())
    }
    if not existing_path_basenames:
        return list(enumerate(media_refs))
    coalesced: list[tuple[int, dict[str, Any]]] = []
    for media_index, media in enumerate(media_refs):
        if (
            media.get("source_key") == "decoded_image"
            and Path(str(media.get("path_hint") or media.get("path") or "")).name
            in existing_path_basenames
        ):
            continue
        coalesced.append((media_index, media))
    return coalesced


def _coalesce_duplicate_decoded_image_media(
    resolved_images: list[tuple[str, dict[str, Any]]],
) -> list[tuple[str, dict[str, Any]]]:
    existing_path_basenames = {
        Path(path).name
        for path, report in resolved_images
        if report.get("source_key") == "image"
        and report.get("materialization_source") == "existing_path"
    }
    if not existing_path_basenames:
        return resolved_images
    coalesced: list[tuple[str, dict[str, Any]]] = []
    for path, report in resolved_images:
        if (
            report.get("source_key") == "decoded_image"
            and Path(str(report.get("path_hint") or path)).name in existing_path_basenames
        ):
            continue
        coalesced.append((path, report))
    return coalesced


def _loadable_image_path(
    sample: BenchmarkSample,
    media: dict[str, Any],
    *,
    media_index: int,
) -> tuple[str, dict[str, Any]] | None:
    kind = str(media.get("kind") or "")
    path = str(media.get("path") or "")
    if path and (media.get("exists") is True or Path(path).exists()):
        return path, _image_media_report(
            media,
            media_index=media_index,
            path=path,
            materialized=False,
            source="existing_path",
        )
    if kind in {"image_struct", "embedded_image_struct", "embedded_bytes"}:
        bytes_value = media.get("bytes")
        if isinstance(bytes_value, (bytes, bytearray)):
            materialized = _materialize_image_bytes(
                bytes(bytes_value),
                sample=sample,
                media=media,
                media_index=media_index,
            )
            return materialized, _image_media_report(
                media,
                media_index=media_index,
                path=materialized,
                materialized=True,
                source="embedded_bytes",
            )
        return None
    if kind == "embedded_base64":
        value = str(media.get("value") or "")
        if not value:
            return None
        materialized = _materialize_image_bytes(
            _decode_base64_image(value),
            sample=sample,
            media=media,
            media_index=media_index,
        )
        return materialized, _image_media_report(
            media,
            media_index=media_index,
            path=materialized,
            materialized=True,
            source="embedded_base64",
        )
    return None


def _materialize_image_bytes(
    payload: bytes,
    *,
    sample: BenchmarkSample,
    media: dict[str, Any],
    media_index: int,
) -> str:
    if not payload:
        raise ValueError(f"empty embedded image payload for sample {sample.sample_id}")
    image_format = _image_format(payload)
    suffix = _image_suffix(image_format, media.get("path_hint") or media.get("path"))
    digest = hashlib.sha256()
    digest.update(sample.sample_id.encode("utf-8", errors="replace"))
    digest.update(b"\0")
    digest.update(str(media_index).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(media.get("kind") or "").encode("utf-8", errors="replace"))
    digest.update(b"\0")
    digest.update(payload)
    cache_dir = _stage2_media_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_cache_stem(f"{sample.benchmark}_{sample.sample_id}")
    path = cache_dir / f"{stem}_{digest.hexdigest()[:20]}{suffix}"
    if not path.exists():
        tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        tmp_path.write_bytes(payload)
        os.replace(tmp_path, path)
    return str(path)


def _decode_base64_image(value: str) -> bytes:
    payload = value.split(",", 1)[1] if value.startswith("data:image") and "," in value else value
    return base64.b64decode(payload)


def _image_format(payload: bytes) -> str:
    from PIL import Image

    with Image.open(io.BytesIO(payload)) as image:
        image.verify()
        return str(image.format or "").lower()


def _image_suffix(image_format: str, path_hint: Any) -> str:
    hint_suffix = Path(str(path_hint or "")).suffix.lower()
    if hint_suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}:
        return hint_suffix
    if image_format in {"jpeg", "jpg"}:
        return ".jpg"
    if image_format in {"png", "webp", "bmp", "gif"}:
        return f".{image_format}"
    return ".img"


def _image_media_report(
    media: dict[str, Any],
    *,
    media_index: int,
    path: str,
    materialized: bool,
    source: str,
) -> dict[str, Any]:
    return {
        "media_index": media_index,
        "kind": media.get("kind"),
        "source_key": media.get("source_key"),
        "path": path,
        "materialized": materialized,
        "materialization_source": source,
        "path_hint": media.get("path_hint"),
        "byte_length": media.get("byte_length"),
    }


def _stage2_media_cache_dir() -> Path:
    configured = os.environ.get("REVISIT_VLM_CLEAN_STAGE2_MEDIA_CACHE")
    return Path(configured or "outputs/clean_media_cache/stage2_native").resolve()


def _safe_cache_stem(value: str) -> str:
    chars = [char if char.isalnum() else "_" for char in value]
    stem = "_".join(part for part in "".join(chars).split("_") if part)
    return (stem or "sample")[:96]


def _validate_peft_load_result(load_result: Any) -> None:
    unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
    missing = list(getattr(load_result, "missing_keys", []) or [])
    adapter_markers = ("lora_", "modules_to_save", "token_adapter", "trainable_tokens")
    adapter_missing = [key for key in missing if any(marker in key for marker in adapter_markers)]
    if unexpected or adapter_missing:
        raise RuntimeError(
            "Incomplete Stage2 LoRA load: "
            f"unexpected_keys={unexpected[:20]} "
            f"adapter_missing_keys={adapter_missing[:20]}"
        )


def _validate_qwen_lora_protocol_token_payload(qwen_lora_state: Any) -> None:
    if not isinstance(qwen_lora_state, dict):
        raise RuntimeError("Incomplete Stage2 LoRA load: qwen_lora_state_not_mapping")
    keys = [str(key) for key in qwen_lora_state]
    embed_keys = [key for key in keys if "embed_tokens" in key]
    lm_head_keys = [key for key in keys if "lm_head" in key]
    token_adapter_keys = [
        key for key in keys if "token_adapter" in key or "trainable_tokens" in key
    ]
    if (embed_keys and lm_head_keys) or token_adapter_keys:
        return
    raise RuntimeError(
        "Incomplete Stage2 LoRA load: missing_protocol_token_payload "
        "expected embed_tokens+lm_head or token_adapter/trainable_tokens keys"
    )


def _resolve_runtime_device(torch: Any, requested: Any) -> Any:
    requested_text = str(requested or "auto")
    if requested_text == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_text)


def _unwrap_qwen3_causal_lm(model: Any) -> Any:
    if hasattr(model, "get_base_model"):
        try:
            return model.get_base_model()
        except Exception:
            pass
    base_model = getattr(model, "base_model", None)
    nested = getattr(base_model, "model", None)
    if nested is not None:
        return nested
    return model


def _sample_uid(sample: NativeStage2Sample) -> str:
    base = "|".join(
        [
            sample.image_id or _image_identity_text(sample.image),
            sample.question,
            sample.target,
            sample.answer,
        ]
    )
    return str(abs(hash(base)))


def _image_identity_text(image: Any) -> str:
    if isinstance(image, (list, tuple)):
        return "[" + ",".join(str(item) for item in image) + "]"
    return str(image)


def _image_grid_count(image_grid_thw: Any) -> int:
    if image_grid_thw is None:
        return 0
    try:
        return int(image_grid_thw.detach().cpu().view(-1, 3).shape[0])
    except Exception:
        return 0


def _should_retry_full_sequence_multi_image_append(capture: Any, exc: Exception) -> bool:
    source_geometry = getattr(capture, "source_visual_geometry", None)
    if source_geometry is None or _image_grid_count(getattr(source_geometry, "image_grid_thw", None)) <= 1:
        return False
    message = str(exc)
    retry_markers = (
        "native_source_grid",
        "shape mismatch",
        "position id",
        "compute_3d_position_ids",
    )
    return any(marker in message for marker in retry_markers)


def _full_protocol_text(action_text: str, continuation: str, *, protocol: str) -> str:
    uses_think, uses_evidence = _protocol_text_modes(protocol)
    if uses_think:
        return (
            f"{action_text}\n"
            "<|tgvf_start|>\n[visual embeddings]\n<|tgvf_end|>\n"
            f"<think>\n{continuation}"
        )
    if uses_evidence:
        return (
            f"{action_text}\n"
            "<|tgvf_start|>\n[visual embeddings]\n<|tgvf_end|>\n"
            f"<|evidence_start|>{continuation}"
        )
    return (
        f"{action_text}\n"
        "<|tgvf_start|>\n[visual embeddings]\n<|tgvf_end|>\n"
        f"<EVIDENCE>{continuation}"
    )


def _parsed_evidence_text(text: str, protocol: str) -> str:
    import re

    uses_think, uses_evidence = _protocol_text_modes(protocol)
    if uses_think:
        matches = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        return matches[-1].strip() if matches else ""
    if uses_evidence:
        return _extract_tag(text, "<|evidence_start|>", "<|evidence_end|>") or ""
    return _extract_tag(text, "<EVIDENCE>", "</EVIDENCE>") or ""


def _generated_answer_has_started(
    tokenizer: Any,
    generated_ids: list[int],
    *,
    protocol: str,
) -> bool:
    if not generated_ids:
        return False
    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    uses_think, uses_evidence = _protocol_text_modes(protocol)
    if uses_think:
        return "</think>" in text
    if uses_evidence:
        return "<|evidence_end|>" in text
    return "<ANSWER>" in text


def _past_key_values_sequence_length(past_key_values: Any) -> int | None:
    if past_key_values is None:
        return None
    get_seq_length = getattr(past_key_values, "get_seq_length", None)
    if callable(get_seq_length):
        value = get_seq_length()
        return None if value is None else int(value)
    try:
        layer0 = past_key_values[0]
        key = layer0[0] if isinstance(layer0, (tuple, list)) else getattr(layer0, "key", None)
        if key is not None and hasattr(key, "shape") and len(key.shape) >= 3:
            return int(key.shape[-2])
    except Exception:
        return None
    return None


def _collect_cuda_garbage() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        return


def _is_cuda_fatal_error(error: str | None) -> bool:
    text = str(error or "").lower()
    return any(
        marker in text
        for marker in (
            "outofmemoryerror",
            "cuda out of memory",
            "mha_graph.execute",
            "cublas",
            "cuda error",
        )
    )


def _protocol_text_modes(protocol: str) -> tuple[bool, bool]:
    try:
        from revisit_vlm.qwen3_vl_tgvf import (
            protocol_uses_evidence_tags,
            protocol_uses_think_tags,
        )

        return protocol_uses_think_tags(protocol), protocol_uses_evidence_tags(protocol)
    except Exception:
        return (
            protocol in {"protocol_c_thinking_special", "protocol_c_tool_observation"},
            protocol
            in {
                "protocol_c_tool_observation_qwen2_no_think",
                "protocol_e_action_evidence_special",
            },
        )


def _extract_tag(text: str, start: str, end: str) -> str | None:
    start_index = text.find(start)
    if start_index < 0:
        return None
    inner_start = start_index + len(start)
    end_index = text.find(end, inner_start)
    if end_index < 0:
        return None
    return text[inner_start:end_index].strip()


def _continuation_not_im_end(text: str) -> bool:
    import re

    stripped = re.sub(r"\s+", "", text or "")
    if not stripped:
        return False
    return stripped not in {"<|im_end|>", "<|endoftext|>"}


def _shape_list(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(item) for item in shape]
