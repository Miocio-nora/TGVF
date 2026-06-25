"""Clean-native Stage2 engine boundary.

This module owns the final backend contract for Qwen3 Stage2 TGVF execution.
The engine uses lower-level Qwen3/TGVF primitives directly and keeps the native
backend independent from historical evaluator classes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .benchmark_data import BenchmarkSample
from .data_generation import file_identity
from .rendering import RenderedBenchmarkInput
from .schema import EvalMode, ForwardMode, RunConfig
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
    image: str
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
        self.vision_cache: dict[str, tuple[Any, Any, Any]] = {}

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
                "max_action_tokens": config.max_action_tokens,
                "max_answer_tokens": config.max_answer_tokens,
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
                append_result = self._append_visual_d(capture, correct_d)
            continuation = self._continue_generation(append_result)
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
                second_full_forward_used=bool(
                    getattr(capture, "second_full_forward_used", False)
                    or append_result.debug_metadata.get("second_full_forward_used")
                ),
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
        from revisit_vlm.tgvf_v3_stage1 import freeze_qwen_backbone, infer_qwen3_stage1_dims

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
        dims = infer_qwen3_stage1_dims(
            model=utility_model,
            processor=processor,
            sample=sample,  # type: ignore[arg-type]
            device=self.device,
            max_image_resolution=config.max_image_resolution,
        )
        tgvf_cfg = checkpoint_config.get("tgvf") or {}
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
                max_new_tokens=self.stage2_config.max_action_tokens,
                device=self.device,
                forced_prefix_text=forced_text,
                protocol=self.stage2_config.protocol,
            )
        return capture_focus_single_pass_qwen3(
            self.model,
            self.processor,
            image=image,
            question=sample.prompt_question,
            messages=build_direct_messages(image, sample.prompt_question),
            max_new_tokens=self.stage2_config.max_action_tokens,
            device=self.device,
            protocol=self.stage2_config.protocol,
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
            max_new_tokens=self.stage2_config.max_action_tokens,
            device=self.device,
            force_action_prefix=False,
            protocol=self.stage2_config.protocol,
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
        tap, v_pre, _v_merge = self._vision_features(sample)
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
            },
        )
        output = finalize_tgvf_output_with_frozen_qwen_merger(self.utility_model, output)
        return output.foveated_visual_tokens.detach()

    def _vision_features(self, sample: NativeStage2Sample) -> tuple[Any, Any, Any]:
        from revisit_vlm.qwen3_vl_tgvf import tap_qwen3_vision_features

        if self.utility_model is None or self.processor is None:
            raise RuntimeError("native Stage2 utility model is not loaded")
        config = self._run_config
        if config is None:
            raise RuntimeError("NativeStage2Engine.prepare must be called before vision tap")
        key = f"{sample.image}|{config.max_image_resolution}"
        if key not in self.vision_cache:
            self.vision_cache[key] = tap_qwen3_vision_features(
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
            _bracketed_visual_token_ids,
            _chunk_position_ids_inherit_source_visual_positions,
            _chunk_position_ids_native_source_grid,
            _encode_text,
            _extend_attention,
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
        embeds = embed(token_ids.view(1, -1).to(self.device)).detach().clone()
        embeds[0, fvt_token_start:fvt_token_end] = d.to(device=self.device, dtype=embeds.dtype)
        attention_mask = capture.attention_mask
        if attention_mask is not None:
            attention_mask = _extend_attention(
                attention_mask.to(self.device),
                int(token_ids.shape[0]),
            )
        input_ids = capture.input_ids
        if input_ids is not None:
            input_ids = torch.cat(
                [input_ids.to(self.device), token_ids.view(1, -1).to(self.device)],
                dim=-1,
            )
        mm_token_type_ids = _fvt_mm_token_type_ids(
            chunk_length=int(token_ids.shape[0]),
            fvt_token_start=fvt_token_start,
            fvt_token_end=fvt_token_end,
            device=self.device,
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
        outputs = self.model(
            inputs_embeds=embeds,
            past_key_values=capture.past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            mm_token_type_ids=mm_token_type_ids,
            use_cache=True,
            return_dict=True,
        )
        model_kwargs = dict(getattr(capture, "model_kwargs", {}) or {})
        next_position_ids = _next_position_ids_after_prefill(position_ids)
        if next_position_ids is not None:
            model_kwargs["tgvf_next_position_ids"] = next_position_ids.detach().cpu()
        return self._append_result_cls()(
            past_key_values=outputs.past_key_values,
            attention_mask=attention_mask,
            cache_position=None,
            input_ids=input_ids,
            last_logits=outputs.logits,
            appended_token_ids=token_ids.detach().cpu(),
            appended_inputs_embeds=embeds.detach().cpu(),
            fvt_token_start=fvt_token_start,
            fvt_token_end=fvt_token_end,
            model_kwargs=model_kwargs,
            debug_metadata={
                "fvt_append_path": "clean_native_qwen3_visual_special_tokens_embedding_replace",
                "tgvf_protocol": self.stage2_config.protocol,
                "uses_deepstack_for_fvt": False,
                "fvt_shape": list(d.shape),
                "num_fvt_tokens": int(d.shape[0]),
                "source_visual_token_count": source_token_count,
                "fvt_position_mode": "native_source_grid",
                "position_ids_shape": (
                    list(position_ids.shape) if position_ids is not None else None
                ),
                "mm_token_type_ids_shape": list(mm_token_type_ids.shape),
                "native_qwen3_position_compute_used": True,
                "second_full_forward_used": False,
                "past_key_values_preserved": (
                    capture.past_key_values is not None and outputs.past_key_values is not None
                ),
            },
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
        _tap, _v_pre, v_merge = self._vision_features(sample)
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
        position_ids = _compute_qwen3_position_ids_for_sequence(
            model=self.utility_model,
            input_ids=full_input_ids,
            attention_mask=full_attention,
            image_grid_thw=image_grid_thw,
            video_grid_thw=capture.video_grid_thw,
            mm_token_type_ids=full_mm_token_type_ids,
        )
        if position_ids is None:
            raise ValueError("position id computation failed for full-sequence prefill")
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
                "uses_deepstack_for_fvt": False,
                "fvt_shape": list(d.shape),
                "num_fvt_tokens": int(d.shape[0]),
                "source_visual_token_count": int(source_geometry.source_visual_token_count),
                "fvt_position_mode": "native_source_grid",
                "position_ids_shape": list(position_ids.shape),
                "mm_token_type_ids_shape": list(full_mm_token_type_ids.shape),
                "native_qwen3_position_compute_used": True,
                "second_full_forward_used": True,
                "past_key_values_preserved": False,
            },
        )

    def _continue_generation(self, append_result: Any) -> Any:
        from revisit_vlm.qwen3_vl_tgvf import continue_generation_qwen3

        if self.model is None or self.processor is None:
            raise RuntimeError("native Stage2 model is not loaded")
        return continue_generation_qwen3(
            self.model,
            self.processor,
            append_result,
            max_new_tokens=self.stage2_config.max_answer_tokens,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )

    def _parse_action(self, text: str) -> Any:
        from revisit_vlm.qwen3_vl_tgvf import parse_v3_action

        return parse_v3_action(text, protocol=self.stage2_config.protocol)

    def _image(self, sample: NativeStage2Sample) -> Any:
        from revisit_vlm.tgvf_v3_stage1 import _image_input

        config = self._run_config
        if config is None:
            raise RuntimeError("NativeStage2Engine.prepare must be called before image load")
        return _image_input(sample.image, max_image_resolution=config.max_image_resolution)

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
    image = _primary_path_media(sample)
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
            **sample.metadata,
        },
    )


def _validate_run_alignment(stage2_config: Stage2RuntimeConfig, config: RunConfig) -> None:
    config.validate()
    if config.mode not in SUPPORTED_NATIVE_STAGE2_MODES:
        raise NotImplementedError(
            "clean-native Stage2 backend supports only TGVF modes, got "
            f"{config.mode.value!r}"
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


def _primary_path_media(sample: BenchmarkSample) -> str:
    for media in sample.media:
        if media.get("kind") == "path" and media.get("path") and media.get("exists") is True:
            return str(media["path"])
    raise ValueError(
        "clean-native Stage2 requires path-backed image media; "
        f"sample {sample.sample_id} has media kinds {[item.get('kind') for item in sample.media]}"
    )


def _validate_peft_load_result(load_result: Any) -> None:
    unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
    missing = list(getattr(load_result, "missing_keys", []) or [])
    adapter_markers = ("lora_", "modules_to_save", "token_adapter", "trainable_tokens")
    adapter_missing = [
        key for key in missing if any(marker in key for marker in adapter_markers)
    ]
    if unexpected or adapter_missing:
        raise RuntimeError(
            "Incomplete Stage2 LoRA load: "
            f"unexpected_keys={unexpected[:20]} "
            f"adapter_missing_keys={adapter_missing[:20]}"
        )


def _resolve_runtime_device(torch: Any, requested: Any) -> Any:
    requested_text = str(requested or "auto")
    if requested_text == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_text)


def _sample_uid(sample: NativeStage2Sample) -> str:
    base = "|".join(
        [sample.image_id or sample.image, sample.question, sample.target, sample.answer]
    )
    return str(abs(hash(base)))


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
