from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from tgvf_eval.adapters import BenchmarkSample
from tgvf_eval.config import MethodConfig, image_budget_kwargs
from tgvf_eval.parsing import extract_foveation_target
from tgvf_eval.prompts import build_direct_prompt, build_focused_target_answer_prompt, build_prompt
from tgvf_eval.target_rules import (
    build_vstar_target_spec,
    target_spec_debug,
    validate_target_text,
)


@dataclass
class ModelRunResult:
    raw_output: str
    triggered: bool = False
    foveation_target: str = ""
    num_foveations: int = 0
    second_full_forward_used: bool = False
    visual_token_count: int = 0
    output_tokens: int = 0
    wall_time_sec: float = 0.0
    error: str | None = None
    debug: dict[str, Any] | None = None


class DryRunModelRunner:
    def prepare(self, config: MethodConfig) -> None:
        del config

    def run(self, sample: BenchmarkSample, config: MethodConfig) -> ModelRunResult:
        start = time.perf_counter()
        if config.trigger_mode in {"force", "free"} and config.tgvf_mode != "none":
            target = "relevant local visual evidence"
            raw = sample.gold_answer or "A"
            return ModelRunResult(
                raw_output=raw,
                triggered=True,
                foveation_target=target,
                num_foveations=1,
                second_full_forward_used=False,
                wall_time_sec=time.perf_counter() - start,
                debug={"dry_run": True},
            )
        return ModelRunResult(
            raw_output=sample.gold_answer or "A",
            triggered=False,
            second_full_forward_used=False,
            wall_time_sec=time.perf_counter() - start,
            debug={"dry_run": True},
        )


class QwenTGVFModelRunner:
    def __init__(
        self,
        *,
        model_path: str,
        processor_path: str | None = None,
        tgvf_checkpoint: str | None = None,
        tgvf_variant: str = "target_slot_foveal_cross_merger",
        dtype: str = "bf16",
        device: str = "auto",
        attn_implementation: str = "flash_attention_2",
        answer_max_new_tokens: int = 128,
        capture_max_new_tokens: int = 128,
        capture_layer: int = -1,
        num_foveated_tokens: int | None = 16,
        spatial_merge_size: str = "auto",
        repeat_options_in_continuation: bool = False,
        force_target_mode: str = "generated",
        fixed_force_target: str | None = None,
        force_target_source: str = "llm",
        suppress_im_end_first_token_after_fvt: bool = False,
        fvt_append_mode: str = "qwen_native_pseudo_image",
        fresh_answer_context: str = "direct",
    ) -> None:
        self.model_path = model_path
        self.processor_path = processor_path
        self.tgvf_checkpoint = tgvf_checkpoint
        self.tgvf_variant = tgvf_variant
        self.dtype = dtype
        self.device_arg = device
        self.attn_implementation = attn_implementation
        self.answer_max_new_tokens = answer_max_new_tokens
        self.capture_max_new_tokens = capture_max_new_tokens
        self.capture_layer = capture_layer
        self.num_foveated_tokens = num_foveated_tokens
        self.spatial_merge_size = spatial_merge_size
        self.repeat_options_in_continuation = repeat_options_in_continuation
        self.force_target_mode = force_target_mode
        self.fixed_force_target = fixed_force_target
        self.force_target_source = force_target_source
        self.suppress_im_end_first_token_after_fvt = suppress_im_end_first_token_after_fvt
        self.fvt_append_mode = fvt_append_mode
        if fresh_answer_context not in {"direct", "focused_target"}:
            raise ValueError(
                "fresh_answer_context must be one of: direct, focused_target"
            )
        self.fresh_answer_context = fresh_answer_context
        self._loaded = None

    def prepare(self, config: MethodConfig) -> None:
        needs_module = config.tgvf_mode == "module" and config.control_condition not in {
            "no_D",
            "original_D",
        }
        self._load(need_tgvf=needs_module)

    def _image_kwargs(self, config: MethodConfig) -> dict[str, int]:
        kwargs = image_budget_kwargs(config.image_budget)
        if config.video_nframes is not None:
            kwargs["nframes"] = int(config.video_nframes)
        return kwargs

    def _single_user_messages(
        self,
        sample: BenchmarkSample,
        prompt: str,
        config: MethodConfig,
    ) -> list[dict[str, Any]]:
        from revisit_vlm.tgvf_capture import _vision_content_items

        media = sample.primary_media
        if media is None:
            raise ValueError("sample has no image/video media path")
        return [
            {
                "role": "user",
                "content": [
                    *_vision_content_items(media, self._image_kwargs(config)),
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _option_letters(self, sample: BenchmarkSample) -> list[str] | None:
        if not sample.choices:
            return None
        return [chr(ord("A") + index) for index in range(len(sample.choices))]

    def _force_marker_capture_model(
        self,
        model: Any,
        processor: Any,
        *,
        fixed_target_text: str | None = None,
    ) -> tuple[Any, int]:
        from revisit_vlm.tgvf import FOVEATE_END, FOVEATE_START
        from revisit_vlm.tgvf_training import ForcedFoveationBracketWrapper, ForcedFoveationWrapper

        start_ids = processor.tokenizer.encode(FOVEATE_START, add_special_tokens=False)
        end_ids = processor.tokenizer.encode(FOVEATE_END, add_special_tokens=False)
        if fixed_target_text is not None:
            target_ids = processor.tokenizer.encode(str(fixed_target_text), add_special_tokens=False)
            forced_ids = [*start_ids, *target_ids, *end_ids]
            return ForcedFoveationWrapper(model, forced_ids), len(forced_ids)
        max_target_tokens = 32
        suppress_ids = (
            [processor.tokenizer.eos_token_id]
            if processor.tokenizer.eos_token_id is not None
            else []
        )
        wrapped = ForcedFoveationBracketWrapper(
            model,
            start_ids=start_ids,
            end_ids=end_ids,
            max_target_tokens=max_target_tokens,
            suppress_ids=suppress_ids,
        )
        return wrapped, len(start_ids) + max_target_tokens + len(end_ids)

    def _capture_force_request(
        self,
        *,
        model: Any,
        processor: Any,
        sample: BenchmarkSample,
        prompt: str,
        messages: list[dict[str, Any]],
        config: MethodConfig,
        device: Any,
    ) -> tuple[Any, dict[str, Any]]:
        from revisit_vlm.tgvf_capture import trim_capture_target_at_delimiters

        rule_spec = self._rule_target_spec(sample)
        configured_source = self.force_target_source
        fixed_target_text: str | None = None
        initial_target_source = configured_source
        if self.force_target_mode == "fixed":
            fixed_target_text = self.fixed_force_target
            initial_target_source = "fixed"
        elif configured_source == "rule" and rule_spec is not None:
            fixed_target_text = rule_spec.target_text
            initial_target_source = "rule"
        elif configured_source == "rule" and rule_spec is None:
            initial_target_source = "llm_rule_unavailable"

        capture = self._capture_once(
            model=model,
            processor=processor,
            prompt=prompt,
            messages=messages,
            config=config,
            device=device,
            fixed_target_text=fixed_target_text,
        )
        if capture.capture_found:
            trim_capture_target_at_delimiters(capture, processor.tokenizer)

        validation = validate_target_text(
            capture.target_text if capture.capture_found else "",
            spec=rule_spec,
            choices=sample.choices,
        )
        debug = {
            "configured_target_source": configured_source,
            "target_source": initial_target_source,
            "fallback_used": False,
            "raw_target_text": capture.raw_target_text if capture.capture_found else capture.generated_text,
            "final_target_text": capture.target_text if capture.capture_found else "",
            **validation.to_debug(),
            **target_spec_debug(rule_spec),
        }

        if (
            configured_source == "llm_with_rule_fallback"
            and rule_spec is not None
            and (not capture.capture_found or not validation.valid)
        ):
            fallback_capture = self._capture_once(
                model=model,
                processor=processor,
                prompt=prompt,
                messages=messages,
                config=config,
                device=device,
                fixed_target_text=rule_spec.target_text,
            )
            if fallback_capture.capture_found:
                trim_capture_target_at_delimiters(fallback_capture, processor.tokenizer)
            fallback_validation = validate_target_text(
                fallback_capture.target_text if fallback_capture.capture_found else "",
                spec=rule_spec,
                choices=sample.choices,
            )
            debug = {
                "configured_target_source": configured_source,
                "target_source": "rule",
                "fallback_used": True,
                "raw_target_text": capture.raw_target_text if capture.capture_found else capture.generated_text,
                "final_target_text": fallback_capture.target_text if fallback_capture.capture_found else "",
                **fallback_validation.to_debug(),
                "invalid_reason": validation.invalid_reason or "capture_not_found",
                **target_spec_debug(rule_spec),
            }
            capture = fallback_capture

        return capture, debug

    def _capture_once(
        self,
        *,
        model: Any,
        processor: Any,
        prompt: str,
        messages: list[dict[str, Any]],
        config: MethodConfig,
        device: Any,
        fixed_target_text: str | None,
    ) -> Any:
        from revisit_vlm.tgvf_capture import capture_tgvf_single_pass

        capture_model, forced_token_count = self._force_marker_capture_model(
            model,
            processor,
            fixed_target_text=fixed_target_text,
        )
        return capture_tgvf_single_pass(
            capture_model,
            processor,
            image=None,
            question=prompt,
            messages=messages,
            max_new_tokens=(
                forced_token_count + 4 if forced_token_count else self.capture_max_new_tokens
            ),
            device=device,
            hidden_state_index=self.capture_layer,
            eos_token_id=processor.tokenizer.eos_token_id,
            image_kwargs=self._image_kwargs(config),
        )

    def _rule_target_spec(self, sample: BenchmarkSample) -> Any:
        if sample.benchmark != "vstar_bench":
            return None
        return build_vstar_target_spec(
            sample.question,
            category=str(sample.metadata.get("category") or ""),
            choices=sample.choices,
        )

    def run(self, sample: BenchmarkSample, config: MethodConfig) -> ModelRunResult:
        start = time.perf_counter()
        try:
            if not sample.primary_media:
                raise ValueError("sample has no image/video media path")
            if config.tgvf_mode == "module":
                result = self._run_module(sample, config)
            elif config.tgvf_mode == "prompt_only":
                result = self._run_prompt_only(sample, config)
            else:
                result = self._run_direct(sample, config)
            result.wall_time_sec = time.perf_counter() - start
            return result
        except Exception as exc:
            return ModelRunResult(
                raw_output="",
                wall_time_sec=time.perf_counter() - start,
                second_full_forward_used=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _load(self, need_tgvf: bool) -> tuple[Any, Any, Any, Any]:
        if self._loaded is not None:
            return self._loaded
        import argparse

        from transformers import AutoProcessor

        from eval.common import load_qwen_and_tgvf, resolve_device
        from revisit_vlm.models.qwen2vl import load_qwen2vl
        from revisit_vlm.tgvf_training import freeze_qwen2vl

        device = resolve_device(self.device_arg)
        if need_tgvf:
            if not self.tgvf_checkpoint:
                raise ValueError("module TGVF mode requires --tgvf-checkpoint")
            args = argparse.Namespace(
                model_path=self.model_path,
                processor_path=self.processor_path,
                tgvf_checkpoint=self.tgvf_checkpoint,
                variant=self.tgvf_variant,
                device=self.device_arg,
                dtype=self.dtype,
                attn_implementation=self.attn_implementation,
                num_foveated_tokens=self.num_foveated_tokens,
                spatial_merge_size=self.spatial_merge_size,
            )
            model, processor, foveal_module, device, _info = load_qwen_and_tgvf(args)
            self._loaded = (model, processor, foveal_module, device)
            return self._loaded

        loaded = load_qwen2vl(
            self.model_path,
            torch_dtype=self.dtype,
            attn_implementation=self.attn_implementation,
            device_map={"": device} if str(device).startswith("cuda") else None,
            trust_remote_code=False,
        )
        processor = loaded.processor
        if self.processor_path:
            processor = AutoProcessor.from_pretrained(self.processor_path, trust_remote_code=False)
            if processor.tokenizer.pad_token is None:
                processor.tokenizer.pad_token = processor.tokenizer.eos_token
        freeze_qwen2vl(loaded.model)
        self._loaded = (loaded.model, processor, None, device)
        return self._loaded

    def _run_direct(self, sample: BenchmarkSample, config: MethodConfig) -> ModelRunResult:
        from revisit_vlm.tgvf_capture import build_qwen2vl_tgvf_inputs

        model, processor, _module, device = self._load(need_tgvf=False)
        prompt = build_prompt(sample.question, config).prompt
        messages = self._single_user_messages(sample, prompt, config)
        inputs = build_qwen2vl_tgvf_inputs(
            processor,
            image=sample.primary_media,
            question=prompt,
            messages=messages,
        )
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        outputs = model.generate(
            **inputs,
            max_new_tokens=self.answer_max_new_tokens,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
        prompt_len = inputs["input_ids"].shape[-1]
        generated = outputs[0, prompt_len:].detach().cpu().tolist()
        answer = processor.tokenizer.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return ModelRunResult(
            raw_output=answer,
            output_tokens=len(generated),
            second_full_forward_used=False,
        )

    def _run_prompt_only(self, sample: BenchmarkSample, config: MethodConfig) -> ModelRunResult:
        if config.reasoning_mode == "visible_tool_cot":
            return self._run_visible_tool_cot_prompt_only(sample, config)

        from revisit_vlm.tgvf_capture import capture_tgvf_single_pass
        from revisit_vlm.tgvf_foveal import (
            append_answer_turn_and_open,
            continue_generation_from_state,
        )

        model, processor, _module, device = self._load(need_tgvf=False)
        prompt = build_prompt(sample.question, config).prompt
        messages = self._single_user_messages(sample, prompt, config)
        target_debug: dict[str, Any] = {}
        if config.trigger_mode == "force":
            capture, target_debug = self._capture_force_request(
                model=model,
                processor=processor,
                sample=sample,
                prompt=prompt,
                messages=messages,
                config=config,
                device=device,
            )
        else:
            capture = capture_tgvf_single_pass(
                model,
                processor,
                image=sample.primary_media,
                question=prompt,
                messages=messages,
                max_new_tokens=self.capture_max_new_tokens,
                device=device,
                hidden_state_index=self.capture_layer,
                eos_token_id=processor.tokenizer.eos_token_id,
                image_kwargs=self._image_kwargs(config),
            )
        if not capture.capture_found:
            return ModelRunResult(
                raw_output=capture.generated_text,
                triggered=False,
                second_full_forward_used=False,
                output_tokens=len(capture.generated_ids),
                debug={"stop_reason": capture.stop_reason, **target_debug},
            )
        answer_state = append_answer_turn_and_open(
            model=model,
            tokenizer_or_processor=processor,
            generation_state=capture,
            benchmark_answer_format="multiple_choice" if sample.choices else None,
            option_letters=self._option_letters(sample),
            original_question=sample.question,
            option_texts=sample.choices or None,
            repeat_options_in_continuation=(
                self.repeat_options_in_continuation and bool(sample.choices)
            ),
            device=device,
        )
        continuation = continue_generation_from_state(
            model=model,
            tokenizer_or_processor=processor,
            generation_state=answer_state,
            max_new_tokens=self.answer_max_new_tokens,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
        return ModelRunResult(
            raw_output=continuation.generated_text,
            triggered=True,
            foveation_target=capture.target_text,
            num_foveations=1,
            second_full_forward_used=False,
            output_tokens=len(continuation.generated_ids),
            debug={
                "capture_text": capture.generated_text,
                "stop_reason": continuation.stop_reason,
                **target_debug,
                **answer_state.debug_metadata,
            },
        )

    def _run_module(self, sample: BenchmarkSample, config: MethodConfig) -> ModelRunResult:
        if config.reasoning_mode == "visible_tool_cot":
            return self._run_visible_tool_cot_module(sample, config)
        if config.control_condition is None:
            return self._run_correct_module(sample, config)
        return self._run_module_control(sample, config)

    def _run_visible_tool_cot_prompt_only(
        self,
        sample: BenchmarkSample,
        config: MethodConfig,
    ) -> ModelRunResult:
        from revisit_vlm.tgvf_capture import (
            build_qwen2vl_tgvf_inputs,
            generate_until_event_from_inputs,
        )

        model, processor, _module, device = self._load(need_tgvf=False)
        prompt = build_prompt(sample.question, config).prompt
        messages = self._single_user_messages(sample, prompt, config)
        inputs = build_qwen2vl_tgvf_inputs(
            processor,
            image=sample.primary_media,
            question=prompt,
            messages=messages,
            image_kwargs=self._image_kwargs(config),
        )
        event = generate_until_event_from_inputs(
            model,
            processor.tokenizer,
            inputs,
            max_new_tokens=self.answer_max_new_tokens,
            device=device,
            hidden_state_index=self.capture_layer,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
        return ModelRunResult(
            raw_output=event.generated_text,
            triggered=event.event_type == "foveate",
            foveation_target=event.target_text,
            num_foveations=1 if event.event_type == "foveate" else 0,
            second_full_forward_used=False,
            output_tokens=len(event.generated_ids),
            debug={
                "reasoning_mode": config.reasoning_mode,
                "visible_cot_transcript": event.generated_text,
                "visible_cot_event_type": event.event_type,
                "visible_cot_final_answer": event.final_answer,
                "prompt_only_no_d": True,
            },
        )

    def _visible_foveation_result_instruction(
        self,
        sample: BenchmarkSample,
        target_text: str,
        *,
        has_visual_tokens: bool = True,
    ) -> str:
        target = target_text.strip() or "the requested visual evidence"
        if sample.choices:
            letters = ", ".join(self._option_letters(sample) or [])
            answer_line = f"When ready, write Final answer: one of {letters}."
        else:
            answer_line = "When ready, write Final answer: followed by your answer."
        evidence_line = (
            "Use the focused visual evidence above to continue your reasoning."
            if has_visual_tokens
            else "No focused visual evidence was provided; continue from the original context."
        )
        return (
            f"Foveation result for: {target}\n"
            f"{evidence_line}\n"
            f"{answer_line}"
        )

    def _run_visible_tool_cot_module(
        self,
        sample: BenchmarkSample,
        config: MethodConfig,
    ) -> ModelRunResult:
        import torch

        from revisit_vlm.tgvf_capture import (
            build_qwen2vl_tgvf_inputs,
            generate_until_event_from_inputs,
            generate_until_event_from_state,
            trim_capture_target_at_delimiters,
        )
        from revisit_vlm.tgvf_foveal import (
            Qwen2VLPreMergeVisualHook,
            append_answer_turn_and_open,
            append_fvt_result_and_open_answer_turn,
            finalize_tgvf_output_with_frozen_qwen_merger,
        )
        from revisit_vlm.tgvf_training import Qwen2VLMergedVisualHook

        if config.control_condition == "conditioned_D_fresh":
            raise ValueError("visible_tool_cot is a non-fresh mode and is incompatible with conditioned_D_fresh")

        needs_module = config.control_condition not in {"no_D", "original_D"}
        model, processor, module, device = self._load(need_tgvf=needs_module)
        prompt = build_prompt(sample.question, config).prompt
        messages = self._single_user_messages(sample, prompt, config)
        inputs = build_qwen2vl_tgvf_inputs(
            processor,
            image=sample.primary_media,
            question=prompt,
            messages=messages,
            image_kwargs=self._image_kwargs(config),
        )
        target_debug: dict[str, Any] = {}

        with Qwen2VLPreMergeVisualHook(model) as visual_hook, Qwen2VLMergedVisualHook(model) as merged_hook:
            if config.trigger_mode == "force":
                event, target_debug = self._capture_force_request(
                    model=model,
                    processor=processor,
                    sample=sample,
                    prompt=prompt,
                    messages=messages,
                    config=config,
                    device=device,
                )
                event.event_type = "foveate" if event.capture_found else event.stop_reason
                event.final_answer = None
                event.debug_metadata = target_debug
            else:
                event = generate_until_event_from_inputs(
                    model,
                    processor.tokenizer,
                    inputs,
                    max_new_tokens=self.capture_max_new_tokens,
                    device=device,
                    hidden_state_index=self.capture_layer,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )
            if event.capture_found:
                trim_capture_target_at_delimiters(event, processor.tokenizer)
            pre_merge_visual_tokens = (
                None
                if visual_hook.pre_merge_visual_tokens is None
                else visual_hook.pre_merge_visual_tokens.to(device)
            )
            original_merged_visual_tokens = (
                None
                if merged_hook.merged_visual_tokens is None
                else merged_hook.merged_visual_tokens.to(device)
            )

        transcript_parts = [event.generated_text]
        foveation_events: list[dict[str, Any]] = []
        visual_token_count = 0
        last_fvt_output = None
        last_append_debug: dict[str, Any] = {}
        triggered = event.event_type == "foveate"
        foveation_target = event.target_text if event.event_type == "foveate" else ""
        source_image_grid = event.image_grid_thw
        total_output_tokens = len(event.generated_ids)

        for foveation_index in range(max(0, int(config.max_foveations))):
            if event.event_type != "foveate":
                break
            triggered = True
            foveation_target = event.target_text
            d = None
            use_source_image_grid = False

            if config.control_condition == "no_D":
                generation_state = append_answer_turn_and_open(
                    model=model,
                    tokenizer_or_processor=processor,
                    generation_state=event,
                    append_as_new_user_turn=False,
                    instruction_text=self._visible_foveation_result_instruction(
                        sample,
                        event.target_text,
                        has_visual_tokens=False,
                    ),
                    device=device,
                )
            elif config.control_condition == "original_D":
                if original_merged_visual_tokens is None:
                    raise RuntimeError("Merged visual tokens were not captured during visible original_D inference")
                d = original_merged_visual_tokens
                if d.ndim == 3:
                    d = d.reshape(-1, d.shape[-1])
                if d.ndim != 2:
                    raise ValueError("Merged visual tokens must have shape [N, d_lm] for visible original_D")
                visual_token_count = int(d.shape[0])
                use_source_image_grid = source_image_grid is not None
            else:
                if pre_merge_visual_tokens is None:
                    raise RuntimeError("Pre-merge visual tokens were not captured during visible TGVF inference")
                if module is None:
                    raise RuntimeError("TGVF module is required for visible_tool_cot module inference")
                last_fvt_output = module(
                    target_hidden_states=event.target_hidden_states.to(device),
                    pre_merge_visual_tokens=pre_merge_visual_tokens,
                    metadata={
                        "target_text": event.target_text,
                        "control_condition": config.control_condition or "visible_tool_cot",
                        "reasoning_mode": config.reasoning_mode,
                    },
                )
                last_fvt_output = finalize_tgvf_output_with_frozen_qwen_merger(model, last_fvt_output)
                d = last_fvt_output.foveated_visual_tokens
                if config.control_condition == "random_D":
                    d = torch.randn_like(d)
                elif config.control_condition == "wrong_D":
                    d = torch.flip(d, dims=[0]) if d.shape[0] > 1 else torch.randn_like(d)
                visual_token_count = int(d.shape[0])
                use_source_image_grid = last_fvt_output.debug_metadata.get("tgvf_version") == "v2"

            if d is not None:
                generation_state = append_fvt_result_and_open_answer_turn(
                    model=model,
                    tokenizer_or_processor=processor,
                    generation_state=event,
                    foveated_visual_tokens=d,
                    target_text=event.target_text,
                    append_as_new_user_turn=False,
                    instruction_text=self._visible_foveation_result_instruction(
                        sample,
                        event.target_text,
                        has_visual_tokens=True,
                    ),
                    fvt_append_mode=config.fvt_append_mode,
                    image_grid_thw=source_image_grid if use_source_image_grid else None,
                    device=device,
                )

            last_append_debug = dict(generation_state.debug_metadata or {})
            foveation_events.append(
                {
                    "index": foveation_index,
                    "target_text": event.target_text,
                    "raw_foveation_text": event.generated_text,
                    "target_hidden_shape": list(event.target_hidden_states.shape),
                    "generated_token_count": len(event.generated_ids),
                    "num_fvt_tokens": int(visual_token_count),
                    "append_as_new_user_turn": False,
                    "position_ids_source": last_append_debug.get("position_ids_source"),
                    "cache_preserved": bool(last_append_debug.get("cache_preserved")),
                    "control_condition": config.control_condition,
                }
            )
            transcript_parts.append(f"\n[Foveation result inserted for: {event.target_text}]\n")
            event = generate_until_event_from_state(
                model,
                processor.tokenizer,
                generation_state,
                max_new_tokens=self.answer_max_new_tokens,
                hidden_state_index=self.capture_layer,
                eos_token_id=processor.tokenizer.eos_token_id,
                suppress_first_token_ids=(
                    [processor.tokenizer.eos_token_id]
                    if self.suppress_im_end_first_token_after_fvt
                    and processor.tokenizer.eos_token_id is not None
                    else None
                ),
            )
            if event.capture_found:
                trim_capture_target_at_delimiters(event, processor.tokenizer)
            transcript_parts.append(event.generated_text)
            total_output_tokens += len(event.generated_ids)

        if event.event_type == "foveate" and len(foveation_events) >= int(config.max_foveations):
            stop_reason = "max_foveations_reached"
        else:
            stop_reason = event.stop_reason
        transcript = "".join(transcript_parts)
        return ModelRunResult(
            raw_output=transcript,
            triggered=triggered,
            foveation_target=foveation_target,
            num_foveations=len(foveation_events),
            second_full_forward_used=False,
            output_tokens=total_output_tokens,
            visual_token_count=visual_token_count,
            debug={
                "reasoning_mode": config.reasoning_mode,
                "visible_cot_transcript": transcript,
                "visible_cot_event_type": event.event_type,
                "visible_cot_final_answer": event.final_answer,
                "visible_cot_stop_reason": stop_reason,
                "foveation_events": foveation_events,
                "num_foveations_used": len(foveation_events),
                "variant_name": (
                    None
                    if last_fvt_output is None
                    else last_fvt_output.debug_metadata.get("variant_name")
                ),
                "tgvf_debug": None if last_fvt_output is None else last_fvt_output.debug_metadata,
                "pre_merge_visual_shape": (
                    None if pre_merge_visual_tokens is None else list(pre_merge_visual_tokens.shape)
                ),
                **target_debug,
                **last_append_debug,
            },
        )

    def _run_correct_module(self, sample: BenchmarkSample, config: MethodConfig) -> ModelRunResult:
        from revisit_vlm.tgvf_inference import run_tgvf_inference

        model, processor, module, device = self._load(need_tgvf=True)
        prompt = build_prompt(sample.question, config).prompt
        messages = self._single_user_messages(sample, prompt, config)
        effective_force_target_mode = self.force_target_mode
        effective_fixed_force_target = self.fixed_force_target
        target_debug: dict[str, Any] = {}
        if config.trigger_mode == "force" and self.force_target_mode != "fixed":
            rule_spec = self._rule_target_spec(sample)
            if self.force_target_source == "rule" and rule_spec is not None:
                effective_force_target_mode = "fixed"
                effective_fixed_force_target = rule_spec.target_text
                validation = validate_target_text(rule_spec.target_text, spec=rule_spec, choices=sample.choices)
                target_debug = {
                    "configured_target_source": self.force_target_source,
                    "target_source": "rule",
                    "fallback_used": False,
                    "raw_target_text": rule_spec.target_text,
                    "final_target_text": rule_spec.target_text,
                    **validation.to_debug(),
                    **target_spec_debug(rule_spec),
                }
        result = run_tgvf_inference(
            model=model,
            processor=processor,
            foveal_module=module,
            image=sample.primary_media,
            question=prompt,
            messages=messages,
            device=device,
            capture_max_new_tokens=self.capture_max_new_tokens,
            answer_max_new_tokens=self.answer_max_new_tokens,
            hidden_state_index=self.capture_layer,
            eos_token_id=processor.tokenizer.eos_token_id,
            force_foveation_markers=config.trigger_mode == "force",
            image_kwargs=self._image_kwargs(config),
            benchmark_answer_format="multiple_choice" if sample.choices else None,
            option_letters=self._option_letters(sample),
            original_question=sample.question,
            option_texts=sample.choices or None,
            repeat_options_in_continuation=(
                self.repeat_options_in_continuation and bool(sample.choices)
            ),
            force_target_mode=effective_force_target_mode,
            fixed_force_target=effective_fixed_force_target,
            suppress_im_end_first_token_after_fvt=self.suppress_im_end_first_token_after_fvt,
            fvt_append_mode=config.fvt_append_mode,
        )
        if not result.capture.capture_found:
            return ModelRunResult(
                raw_output=result.foveation_request,
                triggered=False,
                num_foveations=0,
                second_full_forward_used=False,
                output_tokens=len(result.capture.generated_ids),
                debug={**result.debug_metadata, **target_debug},
            )
        return ModelRunResult(
            raw_output=result.answer,
            triggered=True,
            foveation_target=result.target_text
            or extract_foveation_target(result.foveation_request),
            num_foveations=1,
            second_full_forward_used=False,
            output_tokens=len(result.generated_ids),
            visual_token_count=(
                0
                if result.fvt_output is None
                else int(result.fvt_output.foveated_visual_tokens.shape[0])
            ),
            debug={**result.debug_metadata, **target_debug},
        )

    def _run_module_control(self, sample: BenchmarkSample, config: MethodConfig) -> ModelRunResult:
        import torch

        from revisit_vlm.tgvf_capture import (
            capture_tgvf_single_pass,
            trim_capture_target_at_delimiters,
        )
        from revisit_vlm.tgvf_foveal import (
            Qwen2VLPreMergeVisualHook,
            append_answer_turn_and_open,
            append_fvt_result_and_open_answer_turn,
            continue_generation_from_state,
            finalize_tgvf_output_with_frozen_qwen_merger,
            prefill_fvt_first_pass_answer_turn,
        )
        from revisit_vlm.tgvf_training import Qwen2VLMergedVisualHook

        needs_module = config.control_condition not in {"no_D", "original_D"}
        model, processor, module, device = self._load(need_tgvf=needs_module)
        prompt = build_prompt(sample.question, config).prompt
        messages = self._single_user_messages(sample, prompt, config)
        target_debug: dict[str, Any] = {}
        with Qwen2VLPreMergeVisualHook(model) as visual_hook, Qwen2VLMergedVisualHook(model) as merged_hook:
            if config.trigger_mode == "force":
                capture, target_debug = self._capture_force_request(
                    model=model,
                    processor=processor,
                    sample=sample,
                    prompt=prompt,
                    messages=messages,
                    config=config,
                    device=device,
                )
            else:
                capture = capture_tgvf_single_pass(
                    model,
                    processor,
                    image=sample.primary_media,
                    question=prompt,
                    messages=messages,
                    max_new_tokens=self.capture_max_new_tokens,
                    device=device,
                    hidden_state_index=self.capture_layer,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    image_kwargs=self._image_kwargs(config),
                )
                if capture.capture_found:
                    trim_capture_target_at_delimiters(capture, processor.tokenizer)
        if not capture.capture_found:
            return ModelRunResult(
                raw_output=capture.generated_text,
                triggered=False,
                num_foveations=0,
                second_full_forward_used=False,
                output_tokens=len(capture.generated_ids),
                debug={
                    "control_condition": config.control_condition,
                    "stop_reason": capture.stop_reason,
                    **target_debug,
                },
            )

        d = None
        fvt_output = None
        visual_token_count = 0
        use_source_image_grid = False
        if config.control_condition == "original_D":
            if merged_hook.merged_visual_tokens is None:
                raise RuntimeError("Merged visual tokens were not captured during original_D inference")
            d = merged_hook.merged_visual_tokens.to(device)
            if d.ndim == 3:
                d = d.reshape(-1, d.shape[-1])
            if d.ndim != 2:
                raise ValueError("Merged visual tokens must have shape [N, d_lm] for original_D")
            visual_token_count = int(d.shape[0])
            use_source_image_grid = capture.image_grid_thw is not None
        elif config.control_condition != "no_D":
            if visual_hook.pre_merge_visual_tokens is None:
                raise RuntimeError(
                    "Pre-merge visual tokens were not captured during control inference"
                )
            if module is None:
                raise RuntimeError("TGVF module is required for this control condition")
            fvt_output = module(
                target_hidden_states=capture.target_hidden_states.to(device),
                pre_merge_visual_tokens=visual_hook.pre_merge_visual_tokens.to(device),
                metadata={
                    "target_text": capture.target_text,
                    "control_condition": config.control_condition,
                },
            )
            fvt_output = finalize_tgvf_output_with_frozen_qwen_merger(model, fvt_output)
            d = fvt_output.foveated_visual_tokens
            if config.control_condition == "random_D":
                d = torch.randn_like(d)
            elif config.control_condition == "wrong_D":
                d = torch.flip(d, dims=[0]) if d.shape[0] > 1 else torch.randn_like(d)
            visual_token_count = int(d.shape[0])
            use_source_image_grid = fvt_output.debug_metadata.get("tgvf_version") == "v2"

        if d is None:
            generation_state = append_answer_turn_and_open(
                model=model,
                tokenizer_or_processor=processor,
                generation_state=capture,
                benchmark_answer_format="multiple_choice" if sample.choices else None,
                option_letters=self._option_letters(sample),
                original_question=sample.question,
                option_texts=sample.choices or None,
                repeat_options_in_continuation=(
                    self.repeat_options_in_continuation and bool(sample.choices)
                ),
                device=device,
            )
        elif config.control_condition == "conditioned_D_fresh":
            if self.fresh_answer_context == "focused_target":
                answer_prompt = build_focused_target_answer_prompt(
                    sample.question,
                    capture.target_text,
                    cot_enabled=config.cot_enabled,
                )
            else:
                answer_prompt = build_direct_prompt(sample.question, cot_enabled=config.cot_enabled)
            generation_state = prefill_fvt_first_pass_answer_turn(
                model=model,
                tokenizer_or_processor=processor,
                foveated_visual_tokens=d,
                answer_prompt=answer_prompt,
                fvt_append_mode=config.fvt_append_mode,
                image_grid_thw=capture.image_grid_thw if use_source_image_grid else None,
                device=device,
            )
            generation_state.debug_metadata["fresh_answer_context"] = self.fresh_answer_context
        else:
            generation_state = append_fvt_result_and_open_answer_turn(
                model=model,
                tokenizer_or_processor=processor,
                generation_state=capture,
                foveated_visual_tokens=d,
                target_text=capture.target_text,
                benchmark_answer_format="multiple_choice" if sample.choices else None,
                option_letters=self._option_letters(sample),
                original_question=sample.question,
                option_texts=sample.choices or None,
                repeat_options_in_continuation=(
                    self.repeat_options_in_continuation and bool(sample.choices)
                ),
                append_as_new_user_turn=True,
                fvt_append_mode=config.fvt_append_mode,
                image_grid_thw=capture.image_grid_thw if use_source_image_grid else None,
                device=device,
            )
        continuation = continue_generation_from_state(
            model=model,
            tokenizer_or_processor=processor,
            generation_state=generation_state,
            max_new_tokens=self.answer_max_new_tokens,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
        return ModelRunResult(
            raw_output=continuation.generated_text,
            triggered=True,
            foveation_target=capture.target_text,
            num_foveations=1,
            second_full_forward_used=config.control_condition == "conditioned_D_fresh",
            output_tokens=len(continuation.generated_ids),
            visual_token_count=visual_token_count,
            debug={
                "control_condition": config.control_condition,
                "capture_text": capture.generated_text,
                "stop_reason": continuation.stop_reason,
                "variant_name": (
                    None if fvt_output is None else fvt_output.debug_metadata.get("variant_name")
                ),
                "target_hidden_shape": list(capture.target_hidden_states.shape),
                "pre_merge_visual_shape": (
                    None
                    if visual_hook.pre_merge_visual_tokens is None
                    else list(visual_hook.pre_merge_visual_tokens.shape)
                ),
                "wrong_D_source": "local_token_shuffle"
                if config.control_condition == "wrong_D"
                else None,
                "original_D_source": (
                    "qwen_visual_merger_output"
                    if config.control_condition == "original_D"
                    else None
                ),
                "tgvf_debug": None if fvt_output is None else fvt_output.debug_metadata,
                **target_debug,
                **generation_state.debug_metadata,
            },
        )
