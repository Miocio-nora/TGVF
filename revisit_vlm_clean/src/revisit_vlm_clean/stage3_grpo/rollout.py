"""Rollout engine interfaces for Stage3 GRPO."""

from __future__ import annotations

import hashlib
import math
from contextlib import nullcontext
from typing import Protocol

from revisit_vlm_clean.scoring import extract_final_answer
from revisit_vlm_clean.tgvf_protocol import PROTOCOL_C_FOCUS_END, PROTOCOL_C_FOCUS_START

from .schemas import RolloutConfig, RolloutRecord, Stage3GRPOConfig, Stage3Sample


class RolloutEngine(Protocol):
    def free_rollout(self, sample: Stage3Sample, *, rollout_id: int) -> RolloutRecord:
        ...

    def forced_off(self, sample: Stage3Sample, *, rollout_id: int) -> RolloutRecord:
        ...

    def forced_on_clean(self, sample: Stage3Sample, *, rollout_id: int) -> RolloutRecord:
        ...


class FakeRolloutEngine:
    """Deterministic fake engine for CLI smoke and tests.

    This does not claim model quality. It exercises the same record/reward/GRPO
    surfaces that the native single-focus engine will fill with real policy
    generations.
    """

    def __init__(self, config: RolloutConfig) -> None:
        self.config = config

    def free_rollout(self, sample: Stage3Sample, *, rollout_id: int) -> RolloutRecord:
        helpful = str(sample.tool_need_hint or "").lower() in {"likely_required", "useful_tool"}
        use_tool = helpful and rollout_id % 2 == 0 and self.config.max_tool_calls > 0
        correct = rollout_id % 3 != 1
        answer = sample.gold_answer if correct else "unknown"
        if use_tool:
            target = sample.target_text or "the relevant visible detail"
            raw = (
                f"<think>\nI need visual focus before answering.\n</think>\n"
                f"{PROTOCOL_C_FOCUS_START}{target}{PROTOCOL_C_FOCUS_END}<|im_end|>\n"
                f"<|im_start|>tool\n<|tgvf_start|>\n[D]\n<|tgvf_end|><|im_end|>\n"
                f"<|im_start|>assistant\n<think>\nThe focused evidence supports the answer.\n</think>\n{answer}"
            )
            targets = (target,)
            num_tool_calls = 1
            post_tool_reasoning = "The focused evidence supports the answer."
        else:
            raw = f"<think>\nThe answer is directly visible from the image.\n</think>\n{answer}"
            targets = ()
            num_tool_calls = 0
            post_tool_reasoning = ""
        token_ids, old_logprobs = _fake_tokens_and_logprobs(sample.sample_id, rollout_id, raw)
        return RolloutRecord(
            sample_id=sample.sample_id,
            rollout_id=rollout_id,
            rollout_type="free",
            raw_output=raw,
            final_answer=answer,
            used_tool=use_tool,
            num_tool_calls=num_tool_calls,
            targets=targets,
            post_tool_reasoning=post_tool_reasoning,
            token_ids=tuple(token_ids),
            old_logprobs=tuple(old_logprobs),
            ref_logprobs=tuple(value - 0.01 for value in old_logprobs),
            loss_mask=tuple(1 for _ in token_ids),
            runtime={"backend": "fake"},
        )

    def forced_off(self, sample: Stage3Sample, *, rollout_id: int) -> RolloutRecord:
        answer = sample.gold_answer if str(sample.tool_need_hint or "") == "no_tool" else "unknown"
        token_ids, old_logprobs = _fake_tokens_and_logprobs(sample.sample_id, rollout_id, answer)
        return RolloutRecord(
            sample_id=sample.sample_id,
            rollout_id=rollout_id,
            rollout_type="forced_off",
            raw_output=answer,
            final_answer=answer,
            used_tool=False,
            num_tool_calls=0,
            token_ids=tuple(token_ids),
            old_logprobs=tuple(old_logprobs),
            ref_logprobs=tuple(old_logprobs),
            loss_mask=tuple(1 for _ in token_ids),
            runtime={"backend": "fake"},
        )

    def forced_on_clean(self, sample: Stage3Sample, *, rollout_id: int) -> RolloutRecord:
        target = sample.target_text or "the relevant visible detail"
        answer = sample.gold_answer
        raw = f"{PROTOCOL_C_FOCUS_START}{target}{PROTOCOL_C_FOCUS_END}\n{answer}"
        token_ids, old_logprobs = _fake_tokens_and_logprobs(sample.sample_id, rollout_id, raw)
        return RolloutRecord(
            sample_id=sample.sample_id,
            rollout_id=rollout_id,
            rollout_type="forced_on_clean",
            raw_output=raw,
            final_answer=answer,
            used_tool=True,
            num_tool_calls=1,
            targets=(target,),
            post_tool_reasoning="The focused evidence supports the answer.",
            token_ids=tuple(token_ids),
            old_logprobs=tuple(old_logprobs),
            ref_logprobs=tuple(old_logprobs),
            loss_mask=tuple(1 for _ in token_ids),
            runtime={"backend": "fake"},
        )


class NativeSingleFocusRolloutEngine:
    """Native Stage2/TGVF-backed rollout engine.

    This path performs real Qwen/TGVF free and forced trajectories through the
    clean-native Stage2 runtime and exposes logprob replay for GRPO training
    updates.
    """

    def __init__(self, config: Stage3GRPOConfig) -> None:
        self.config = config
        self._engine = None
        self._run_config = None

    def free_rollout(self, sample: Stage3Sample, *, rollout_id: int) -> RolloutRecord:
        engine = self._ensure_engine(sample)
        runtime_sample = self._stage2_sample(sample)
        result = engine._run_free(runtime_sample)  # noqa: SLF001 - Stage3 wraps clean-native runtime.
        return self._record_from_native_result(
            sample,
            rollout_id=rollout_id,
            rollout_type="free",
            result=result,
            forced_target=None,
        )

    def forced_off(self, sample: Stage3Sample, *, rollout_id: int) -> RolloutRecord:
        engine = self._ensure_engine(sample)
        runtime_sample = self._stage2_sample(sample)
        raw = self._generate_direct(engine, runtime_sample)
        return RolloutRecord(
            sample_id=sample.sample_id,
            rollout_id=rollout_id,
            rollout_type="forced_off",
            raw_output=raw,
            final_answer=extract_final_answer(raw) or raw,
            used_tool=False,
            num_tool_calls=0,
            targets=(),
            runtime=self._runtime_metadata(extra={"forced_probe": "off"}),
        )

    def forced_on_clean(self, sample: Stage3Sample, *, rollout_id: int) -> RolloutRecord:
        engine = self._ensure_engine(sample)
        runtime_sample = self._stage2_sample(sample, target_override=sample.target_text)
        capture = engine._capture_generated_focus(runtime_sample, force_prefix=True)  # noqa: SLF001
        if not bool(getattr(capture, "capture_found", False)):
            raw = str(getattr(capture, "generated_text", "") or "")
            return RolloutRecord(
                sample_id=sample.sample_id,
                rollout_id=rollout_id,
                rollout_type="forced_on_clean",
                raw_output=raw,
                final_answer=extract_final_answer(raw) or raw,
                used_tool=False,
                num_tool_calls=0,
                targets=(),
                protocol={"native_errors": ["focus_capture_not_found"]},
                runtime=self._runtime_metadata(extra={"forced_probe": "on_clean"}),
            )
        d = engine._d_from_capture(runtime_sample, capture, focus_source="stage3_forced_on_clean")  # noqa: SLF001
        result = engine._run_post_tgvf_condition(  # noqa: SLF001
            sample=runtime_sample,
            capture=capture,
            correct_d=d,
            block="stage3_forced_on_clean",
            focus_source="stage3_forced_on_clean",
        )
        return self._record_from_native_result(
            sample,
            rollout_id=rollout_id,
            rollout_type="forced_on_clean",
            result=result,
            forced_target=sample.target_text,
        )

    def replay_rollout_logprobs(
        self,
        sample: Stage3Sample,
        rollout: RolloutRecord,
        *,
        reference: bool = False,
    ):
        import torch
        from revisit_vlm.qwen3_vl_tgvf import (
            build_direct_messages,
            build_qwen3_inputs,
            capture_focus_single_pass_from_inputs_qwen3,
        )
        from revisit_vlm_clean.stage3_grpo.native_replay import replay_focus_segment_logprobs

        engine = self._ensure_engine(sample)
        if engine.model is None or engine.processor is None:  # type: ignore[attr-defined]
            raise RuntimeError("native Stage2 model is not loaded")
        focus_ids = [int(item) for item in (rollout.protocol or {}).get("focus_generated_ids") or []]
        continuation_ids = [
            int(item) for item in (rollout.protocol or {}).get("continuation_generated_ids") or []
        ]
        if not focus_ids and not continuation_ids:
            raise ValueError(f"rollout {rollout.sample_id}/{rollout.rollout_id} has no replay tokens")
        model = engine.model  # type: ignore[attr-defined]
        processor = engine.processor  # type: ignore[attr-defined]
        ctx = _reference_model_context(model) if reference else nullcontext()
        with ctx:
            focus_logprobs = replay_focus_segment_logprobs(
                model=model,
                processor=processor,
                sample=sample,
                generated_token_ids=focus_ids,
                config=self.config,
                device=engine.device,  # type: ignore[attr-defined]
            )
            if not continuation_ids:
                return focus_logprobs
            runtime_sample = self._stage2_sample(sample)
            image = engine._image(runtime_sample)  # noqa: SLF001
            inputs = build_qwen3_inputs(
                processor,
                build_direct_messages(image, runtime_sample.prompt_question),
            )
            forced_prefix_text = processor.tokenizer.decode(
                focus_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            capture = capture_focus_single_pass_from_inputs_qwen3(
                model=model,
                tokenizer=processor.tokenizer,
                inputs=inputs,
                max_new_tokens=self.config.rollout.max_action_tokens,
                device=engine.device,  # type: ignore[attr-defined]
                forced_prefix_text=forced_prefix_text,
                protocol=self.config.protocol,
            )
            if not bool(getattr(capture, "capture_found", False)):
                raise ValueError(
                    f"teacher-forced replay could not recover focus span for {rollout.sample_id}"
                )
            d = engine._d_from_capture(runtime_sample, capture, focus_source="stage3_replay")  # noqa: SLF001
            append_result = engine._append_visual_d(capture, d)  # noqa: SLF001
            continuation_logprobs = engine.teacher_forced_continue_logprobs(  # type: ignore[attr-defined]
                append_result,
                generated_token_ids=continuation_ids,
            )
            return torch.cat([focus_logprobs, continuation_logprobs], dim=0)

    def _ensure_engine(self, sample: Stage3Sample):
        if self._engine is not None:
            return self._engine
        from revisit_vlm_clean.schema import EvalMode, ForwardMode, RunConfig
        from revisit_vlm_clean.stage2_native import NativeStage2Engine
        from revisit_vlm_clean.stage2_runtime import Stage2RuntimeConfig

        stage2_config = Stage2RuntimeConfig(
            stage2_checkpoint=self.config.policy_checkpoint,
            eval_jsonl=self.config.rl_data_path,
            protocol=self.config.protocol,
            d_condition="correct_D",
            force_prefix_mode="target_hint",
            append_forward_mode=ForwardMode.KV_CACHE,
            max_action_tokens=self.config.rollout.max_action_tokens,
            max_answer_tokens=self.config.rollout.max_answer_tokens,
        )
        engine = NativeStage2Engine(
            stage2_config=stage2_config,
            backend_options={
                "dtype": self.config.dtype,
                "device": self.config.device,
                "device_map": _resolve_native_device_map(self.config.device_map),
                "attn_implementation": self.config.attn_implementation,
                "do_sample": True,
                "temperature": self.config.rollout.temperature,
                "top_p": self.config.rollout.top_p,
                "detach_d": False,
            },
        )
        run_config = RunConfig(
            run_id=self.config.run_id,
            checkpoint_path=self.config.policy_checkpoint,
            post_tgvf_forward_mode=ForwardMode.KV_CACHE,
            mode=EvalMode.TGVF_FREE,
            model_id=self.config.model_id,
            processor_id=self.config.processor_id,
            population_id="stage3_rl",
            benchmark_root="/",
            max_image_resolution=self.config.max_image_resolution,
            max_action_tokens=self.config.rollout.max_action_tokens,
            max_answer_tokens=self.config.rollout.max_answer_tokens,
            tgvf_protocol=self.config.protocol,
            deepstack=self.config.deepstack,
            execution_backend={
                "name": "stage3_native_single_focus",
                "rollout_only": False,
                "sampled_generation_ready": True,
                "logprob_replay_ready": True,
                "deepstack": self.config.deepstack.to_dict(),
            },
        )
        engine.prepare(run_config)
        engine._ensure_loaded(self._stage2_sample(sample))  # noqa: SLF001
        self._engine = engine
        self._run_config = run_config
        return engine

    def _stage2_sample(
        self,
        sample: Stage3Sample,
        *,
        target_override: str | None = None,
    ):
        from revisit_vlm_clean.stage2_native import NativeStage2Sample

        return NativeStage2Sample(
            image=sample.image_path,
            question=sample.question,
            answer=sample.gold_answer,
            need_focus=True,
            evidence_state="need_local_visual_evidence",
            trajectory_type="single_focus",
            target=target_override or sample.target_text,
            evidence_description="",
            image_id=sample.sample_id,
            choices=None,
            answer_format=sample.answer_type,
            value_span_text=sample.gold_answer,
            evidence_type=sample.evidence_type,
            source_dataset=sample.source_dataset,
            source_profile=sample.source_profile,
            metadata={
                "stage3_sample_id": sample.sample_id,
                "stable_image_uid": sample.stable_image_uid,
                "tool_need_hint": sample.tool_need_hint,
            },
        )

    def _generate_direct(self, engine: object, sample: object) -> str:
        from revisit_vlm.qwen3_vl_tgvf import generate_direct_qwen3

        if engine.model is None or engine.processor is None:  # type: ignore[attr-defined]
            raise RuntimeError("native Stage2 model is not loaded")
        direct = generate_direct_qwen3(
            engine.model,  # type: ignore[attr-defined]
            engine.processor,  # type: ignore[attr-defined]
            image=engine._image(sample),  # noqa: SLF001
            question=sample.prompt_question,  # type: ignore[attr-defined]
            max_new_tokens=self.config.rollout.max_answer_tokens,
            device=engine.device,  # type: ignore[attr-defined]
            protocol=self.config.protocol,
            do_sample=True,
            temperature=self.config.rollout.temperature,
            top_p=self.config.rollout.top_p,
        )
        return str(direct.get("raw_output") or direct.get("parsed_answer") or "")

    def _record_from_native_result(
        self,
        sample: Stage3Sample,
        *,
        rollout_id: int,
        rollout_type: str,
        result: object,
        forced_target: str | None,
    ) -> RolloutRecord:
        debug = dict(getattr(result, "debug", {}) or {})
        full_text = str(debug.get("full_protocol_text") or getattr(result, "raw_output", "") or "")
        final = str(debug.get("parsed_answer") or extract_final_answer(full_text) or "")
        target = str(forced_target or getattr(result, "focus_target", "") or debug.get("parsed_focus_target") or "")
        used_tool = bool(getattr(result, "triggered", False))
        post_tool_reasoning = str(debug.get("parsed_evidence") or debug.get("final_raw_output") or "")
        token_ids = [
            *[int(item) for item in (debug.get("focus_generated_ids") or [])],
            *[int(item) for item in (debug.get("continuation_generated_ids") or [])],
        ]
        focus_ids = [int(item) for item in (debug.get("focus_generated_ids") or [])]
        continuation_ids = [
            int(item) for item in (debug.get("continuation_generated_ids") or [])
        ]
        focus_logprobs = [
            float(item)
            for item in (debug.get("focus_generated_logprobs") or [])
            if _is_finite_float(item)
        ]
        continuation_logprobs = [
            float(item)
            for item in (debug.get("continuation_generated_logprobs") or [])
            if _is_finite_float(item)
        ]
        old_logprobs = [
            *focus_logprobs,
            *continuation_logprobs,
        ]
        if len(old_logprobs) != len(token_ids):
            old_logprobs = []
        return RolloutRecord(
            sample_id=sample.sample_id,
            rollout_id=rollout_id,
            rollout_type=rollout_type,
            raw_output=full_text,
            final_answer=final,
            used_tool=used_tool,
            num_tool_calls=1 if used_tool else 0,
            targets=(target,) if target else (),
            post_tool_reasoning=post_tool_reasoning,
            token_ids=tuple(token_ids),
            old_logprobs=tuple(old_logprobs),
            loss_mask=tuple(1 for _ in token_ids),
            protocol={
                "focus_valid": getattr(result, "focus_valid", None),
                "append_success": getattr(result, "append_success", None),
                "native_errors": debug.get("errors") or [],
                "focus_generated_ids": focus_ids,
                "focus_generated_logprobs": focus_logprobs,
                "continuation_generated_ids": continuation_ids,
                "continuation_generated_logprobs": continuation_logprobs,
                "replay_segments": {
                    "focus": len(focus_ids),
                    "continuation": len(continuation_ids),
                },
            },
            runtime=self._runtime_metadata(
                extra={
                    "output_tokens": getattr(result, "output_tokens", 0),
                    "rollout_type": rollout_type,
                }
            ),
        )

    def _runtime_metadata(self, *, extra: dict[str, object] | None = None) -> dict[str, object]:
        return {
            "backend": "native_single_focus",
            "stage2_checkpoint": self.config.policy_checkpoint,
            "model_id": self.config.model_id,
            "processor_id": self.config.processor_id,
            "device": self.config.device,
            "device_map": self.config.device_map,
            "deepstack": self.config.deepstack.to_dict(),
            "sampled_generation_ready": True,
            "logprob_replay_ready": True,
            "grpo_update_ready": True,
            **dict(extra or {}),
        }


def build_rollout_engine(config: Stage3GRPOConfig | RolloutConfig) -> RolloutEngine:
    rollout_config = config.rollout if isinstance(config, Stage3GRPOConfig) else config
    if rollout_config.runtime_backend == "fake":
        return FakeRolloutEngine(rollout_config)
    if not isinstance(config, Stage3GRPOConfig):
        raise ValueError("native_single_focus rollout requires the full Stage3GRPOConfig")
    return NativeSingleFocusRolloutEngine(config)


def _resolve_native_device_map(device_map: object) -> object:
    if isinstance(device_map, str) and (
        device_map.startswith("cuda") or device_map == "cpu"
    ):
        return {"": device_map}
    return device_map


def _fake_tokens_and_logprobs(sample_id: str, rollout_id: int, text: str) -> tuple[list[int], list[float]]:
    digest = hashlib.sha1(f"{sample_id}:{rollout_id}:{text}".encode("utf-8")).digest()
    length = max(3, min(16, len(text.split()) + 2))
    token_ids = [int(digest[index % len(digest)]) for index in range(length)]
    logprobs = [-1.0 - (token_id % 17) / 100.0 for token_id in token_ids]
    return token_ids, logprobs


def _is_finite_float(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _reference_model_context(model: object):
    # Stage3 currently continues training the Stage2 LoRA adapter directly.
    # Disabling adapters would make the reference path the base Qwen model, not
    # the frozen Stage2 policy. Native updates therefore use detached
    # teacher-forced replay logprobs as the behavior/reference baseline instead.
    return nullcontext()
