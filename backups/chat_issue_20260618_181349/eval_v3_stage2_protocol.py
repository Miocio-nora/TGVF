#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

from revisit_vlm.qwen3_vl_tgvf import (
    ANSWER_END,
    ANSWER_START,
    EVIDENCE_END,
    EVIDENCE_START,
    EVIDENCE_STATE_END,
    EVIDENCE_STATE_START,
    FOCUS_END,
    FOCUS_START,
    NEED_LOCAL_EVIDENCE,
    SUFFICIENT_EVIDENCE,
    TGVF_END,
    TGVF_START,
    PROTOCOL_C_TOOL_OBSERVATION,
    PROTOCOL_C_THINKING_SPECIAL,
    PROTOCOL_D_QWEN_TOOL,
    PROTOCOL_E_ACTION_EVIDENCE_SPECIAL,
    TGVF_PROTOCOL_CHOICES,
    Qwen3AppendResult,
    Qwen3FocusCapture,
    build_direct_messages,
    build_qwen3_inputs,
    capture_focus_single_pass_from_inputs_qwen3,
    capture_focus_single_pass_qwen3,
    continue_generation_qwen3,
    generate_direct_qwen3,
    ensure_tgvf_protocol_tokens,
    is_generic_target,
    load_qwen3_vl,
    make_smoke_d,
    parse_v3_action,
    peak_memory_gb,
    render_focus_action_text,
    render_force_focus_prefix,
    render_tgvf_prefix_suffix,
    THINK_END,
    THINK_START,
    tap_qwen3_vision_features,
    write_json,
    write_jsonl,
    _chunk_position_ids_1d,
    _chunk_position_ids_inherit_source_visual_positions,
    _chunk_position_ids_native_source_grid,
    _bracketed_visual_token_ids,
    _decode,
    _encode_text,
    _extend_attention,
    _fvt_mm_token_type_ids,
)
from revisit_vlm.tgvf_foveal import finalize_tgvf_output_with_frozen_qwen_merger
from revisit_vlm.tgvf_training import TGVF_VARIANTS, build_tgvf_module
from revisit_vlm.tgvf_v3_stage1 import _image_input, freeze_qwen_backbone, infer_qwen3_stage1_dims
from revisit_vlm.tgvf_v3_stage2 import TGVFv3Stage2Dataset, TGVFv3Stage2Sample, dataset_stage2_stats


DEFAULT_VAL = "data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl"
DEFAULT_CKPT = (
    "outputs/tgvf_v3_protocol_c/"
    "protocol_c_toolobs_stage2_v4data_clean_rowonly_gpu0_3_multifocus_focus_imend_from_clean_rowonly_focus_imend_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step_20260617/"
    "checkpoint_step_1200.pt"
)
DEFAULT_MODEL = "/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking"


def _validate_peft_load_result(load_result: Any) -> None:
    unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
    missing = list(getattr(load_result, "missing_keys", []) or [])
    adapter_markers = ("lora_", "modules_to_save", "token_adapter", "trainable_tokens")
    adapter_missing = [
        key
        for key in missing
        if any(marker in key for marker in adapter_markers)
    ]
    if unexpected or adapter_missing:
        raise RuntimeError(
            "Incomplete Stage2 LoRA load: "
            f"unexpected_keys={unexpected[:20]} "
            f"adapter_missing_keys={adapter_missing[:20]}"
        )


class Stage2ProtocolEvaluator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint = torch.load(args.stage2_checkpoint, map_location="cpu")
        checkpoint_config = self.checkpoint.get("config") or {}
        checkpoint_protocol = checkpoint_config.get("tgvf_protocol")
        if checkpoint_protocol and args.tgvf_protocol == "legacy_v3_tags":
            args.tgvf_protocol = checkpoint_protocol
        elif checkpoint_protocol and args.tgvf_protocol != checkpoint_protocol:
            raise ValueError(
                "Stage2 checkpoint protocol mismatch: "
                f"checkpoint={checkpoint_protocol!r} requested={args.tgvf_protocol!r}"
            )
        checkpoint_processor = checkpoint_config.get("processor_id")
        if checkpoint_processor and args.processor_id is None:
            args.processor_id = str(checkpoint_processor)
        self.dataset = TGVFv3Stage2Dataset(args.eval_jsonl, min_confidence=args.min_confidence)
        self.focus_samples_all = [sample for sample in self.dataset.samples if sample.need_focus]
        self.no_focus_samples_all = [sample for sample in self.dataset.samples if not sample.need_focus]
        self.focus_samples = self._select_shard(self.focus_samples_all, args.max_focus)
        self.no_focus_samples = self._select_shard(self.no_focus_samples_all, args.max_no_focus)
        self.focus_groups = self._group_focus_samples(self.focus_samples_all)
        self.focus_index = {id(sample): index for index, sample in enumerate(self.focus_samples_all)}
        self.teacher_cache: dict[int, dict[str, Any]] = {}
        self.vision_cache: dict[str, tuple[Any, torch.Tensor, torch.Tensor]] = {}
        self.loaded = None
        self.model = None
        self.utility_model = None
        self.processor = None
        self.foveal_module = None

    def load(self) -> None:
        loaded = load_qwen3_vl(
            self.args.model_id,
            processor_id=self.args.processor_id,
            dtype=self.args.dtype,
            device_map=self.args.device_map,
            attn_implementation=self.args.attn_implementation,
        )
        base_model = loaded.model
        processor = loaded.processor
        if getattr(processor.tokenizer, "pad_token", None) is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        protocol_token_info: dict[str, Any] = {}
        if self.args.tgvf_protocol in {PROTOCOL_C_THINKING_SPECIAL, PROTOCOL_C_TOOL_OBSERVATION, PROTOCOL_E_ACTION_EVIDENCE_SPECIAL}:
            protocol_token_info = ensure_tgvf_protocol_tokens(processor.tokenizer, base_model, protocol=self.args.tgvf_protocol)
        freeze_qwen_backbone(base_model)
        config = self.checkpoint["config"]
        qwen_lora_state = self.checkpoint["qwen_lora"]
        checkpoint_has_trainable_token_adapter = any(
            "trainable_tokens" in key or "token_adapter" in key
            for key in qwen_lora_state
        )
        lora_cfg = config.get("lora") or {}
        lora_modules_to_save = lora_cfg.get("modules_to_save")
        lora_config = LoraConfig(
            r=int(lora_cfg.get("rank", self.args.lora_rank)),
            lora_alpha=int(lora_cfg.get("alpha", self.args.lora_alpha)),
            target_modules=list(lora_cfg.get("target_modules") or self.args.lora_target_modules.split(",")),
            lora_dropout=float(lora_cfg.get("dropout", 0.0)),
            bias=str(lora_cfg.get("bias", "none")),
            task_type="CAUSAL_LM",
            modules_to_save=list(lora_modules_to_save) if lora_modules_to_save else None,
            ensure_weight_tying=False,
            trainable_token_indices=(
                list(protocol_token_info.get("tgvf_protocol_token_ids", {}).values())
                if self.args.tgvf_protocol in {PROTOCOL_C_THINKING_SPECIAL, PROTOCOL_C_TOOL_OBSERVATION, PROTOCOL_E_ACTION_EVIDENCE_SPECIAL}
                and checkpoint_has_trainable_token_adapter
                and not lora_modules_to_save
                else None
            ),
        )
        model = get_peft_model(base_model, lora_config)
        lora_load_result = set_peft_model_state_dict(model, qwen_lora_state)
        _validate_peft_load_result(lora_load_result)
        model.eval()
        if hasattr(model, "config"):
            model.config.use_cache = True
        utility_model = model.get_base_model() if hasattr(model, "get_base_model") else model
        sample = self.focus_samples_all[0] if self.focus_samples_all else self.dataset.samples[0]
        dims = infer_qwen3_stage1_dims(
            model=utility_model,
            processor=processor,
            sample=sample,  # type: ignore[arg-type]
            device=self.device,
            max_image_resolution=self.args.max_image_resolution,
        )
        tgvf_cfg = config.get("tgvf") or {}
        self.foveal_module = build_tgvf_module(
            variant=str(tgvf_cfg.get("variant") or self.args.variant),
            d_lm=dims["d_lm"],
            d_v=dims["d_v"],
            num_foveated_tokens=tgvf_cfg.get("num_foveated_tokens", self.args.num_foveated_tokens),
            spatial_merge_size=int(tgvf_cfg.get("spatial_merge_size") or dims["spatial_merge_size"]),
            attn_dim=tgvf_cfg.get("attn_dim"),
            encoder_adapter_layers=tuple(tgvf_cfg.get("encoder_adapter_layers") or (8, 16, 24)),
            encoder_adapter_gate_init=float(tgvf_cfg.get("encoder_adapter_gate_init", 0.0)),
            encoder_adapter_share_weights=bool(tgvf_cfg.get("encoder_adapter_share_weights", False)),
            encoder_adapter_layer_index_base=int(tgvf_cfg.get("encoder_adapter_layer_index_base", 0)),
            encoder_reencode_deepstack_compatible=bool(tgvf_cfg.get("encoder_reencode_deepstack_compatible", False)),
        ).to(device=self.device, dtype=next(model.parameters()).dtype)
        self.foveal_module.load_state_dict(self.checkpoint["tgvf_module"], strict=True)
        self.foveal_module.eval()
        self.loaded = loaded
        self.model = model
        self.utility_model = utility_model
        self.processor = processor
        write_json(
            self.output_dir / "config.json",
            {
                "stage": "tgvf_v3_stage2_protocol_eval",
                "tgvf_protocol": self.args.tgvf_protocol,
                "model_id": self.args.model_id,
                "processor_id": self.args.processor_id or self.args.model_id,
                "stage2_checkpoint": self.args.stage2_checkpoint,
                "checkpoint_global_step": self.checkpoint.get("global_step"),
                "eval_jsonl": self.args.eval_jsonl,
                "dataset": dataset_stage2_stats(self.dataset),
                "selected_focus": len(self.focus_samples),
                "selected_no_focus": len(self.no_focus_samples),
                "num_shards": self.args.num_shards,
                "shard_index": self.args.shard_index,
                "max_image_resolution": self.args.max_image_resolution,
                "max_action_tokens": self.args.max_action_tokens,
                "max_answer_tokens": self.args.max_answer_tokens,
                "position_mode": self.args.fvt_position_mode,
                "blocks": self.args.blocks,
                "d_conditions": self.args.d_conditions,
                "second_full_forward_used": False,
                "protocol_c_special_token_ids": protocol_token_info.get("protocol_c_special_token_ids"),
                "tgvf_protocol_token_ids": protocol_token_info.get("tgvf_protocol_token_ids"),
            },
        )

    def run(self) -> None:
        assert self.model is not None and self.processor is not None and self.foveal_module is not None
        started = time.perf_counter()
        block_summaries: dict[str, Any] = {}
        blocks = {item.strip() for item in self.args.blocks.split(",") if item.strip()}
        if "no_focus_direct" in blocks:
            rows = self.eval_no_focus_direct()
            block_summaries["no_focus_direct"] = self._finish_block("no_focus_direct", rows)
        if "force_focus_targets" in blocks:
            rows = self.eval_force_focus_targets()
            block_summaries["force_focus_targets"] = self._finish_block("force_focus_targets", rows)
        if "teacher_forced_post_tgvf" in blocks:
            rows = self.eval_teacher_forced_post_tgvf()
            block_summaries["teacher_forced_post_tgvf"] = self._finish_block("teacher_forced_post_tgvf", rows)
        if "force_end2end" in blocks:
            rows = self.eval_force_end2end()
            block_summaries["force_end2end"] = self._finish_block("force_end2end", rows)
        if "free_router_end2end" in blocks:
            rows = self.eval_free_router_end2end()
            block_summaries["free_router_end2end"] = self._finish_block("free_router_end2end", rows)
        write_json(
            self.output_dir / "summary.json",
            {
                "wall_time_sec": time.perf_counter() - started,
                "blocks": block_summaries,
                "peak_memory_gb": peak_memory_gb(),
            },
        )

    @torch.no_grad()
    def eval_no_focus_direct(self) -> list[dict[str, Any]]:
        rows = []
        for sample in self._iter(self.no_focus_samples, "no_focus_direct"):
            start = time.perf_counter()
            image = self._image(sample)
            direct = generate_direct_qwen3(
                self.model,
                self.processor,
                image=image,
                question=sample.prompt_question,
                max_new_tokens=self.args.max_answer_tokens,
                device=self.device,
                protocol=self.args.tgvf_protocol,
            )
            parsed = parse_v3_action(direct["raw_output"], protocol=self.args.tgvf_protocol)
            row = self._base_row(sample, "no_focus_direct", "direct")
            row.update(
                raw_output=direct["raw_output"],
                final_raw_output=direct["raw_output"],
                parsed_evidence_state=parsed.evidence_state,
                parsed_focus_target=parsed.focus_target,
                parsed_evidence=_parsed_evidence_text(direct["raw_output"], self.args.tgvf_protocol),
                parsed_answer=parsed.answer if parsed.answer_valid else direct["parsed_answer"],
                malformed=parsed.malformed,
                answer_parse_success=bool(parsed.answer_valid or direct.get("parsed_answer")),
                trigger_focus_decision=parsed.evidence_state == NEED_LOCAL_EVIDENCE or parsed.focus_valid,
                no_focus_false_trigger=parsed.evidence_state == NEED_LOCAL_EVIDENCE or parsed.focus_valid,
                continuation_not_im_end=_continuation_not_im_end(direct["raw_output"]),
                second_full_forward_used=False,
                wall_time_sec=time.perf_counter() - start,
            )
            self._add_answer_metrics(row, sample)
            rows.append(row)
        return rows

    @torch.no_grad()
    def eval_force_focus_targets(self) -> list[dict[str, Any]]:
        rows = []
        for sample in self._iter(self.focus_samples, "force_focus_targets"):
            start = time.perf_counter()
            capture = self._capture_generated_focus(sample, force_prefix=True)
            parsed = parse_v3_action(capture.generated_text, protocol=self.args.tgvf_protocol)
            row = self._base_row(sample, "force_focus_targets", "force_generated_target")
            row.update(self._capture_fields(capture, parsed))
            row.update(
                force_prefix_mode=self._force_prefix_mode_name(),
                final_raw_output=capture.generated_text,
                continuation_not_im_end=_continuation_not_im_end(capture.generated_text),
                trigger_focus_decision=capture.capture_found,
                wall_time_sec=time.perf_counter() - start,
            )
            rows.append(row)
        return rows

    @torch.no_grad()
    def eval_teacher_forced_post_tgvf(self) -> list[dict[str, Any]]:
        rows = []
        conditions = self._d_conditions()
        for sample in self._iter(self.focus_samples, "teacher_forced_post_tgvf"):
            item = self._teacher_item(sample)
            for condition in conditions:
                row = self._run_post_tgvf_condition(
                    sample=sample,
                    capture=item["capture"],
                    correct_d=item["d"],
                    condition=condition,
                    block="teacher_forced_post_tgvf",
                    focus_source="teacher_forced",
                )
                rows.append(row)
        return rows

    @torch.no_grad()
    def eval_force_end2end(self) -> list[dict[str, Any]]:
        rows = []
        conditions = self._d_conditions()
        for sample in self._iter(self.focus_samples, "force_end2end"):
            capture = self._capture_generated_focus(sample, force_prefix=True)
            if not capture.capture_found:
                row = self._base_row(sample, "force_end2end", "correct_D")
                parsed = parse_v3_action(capture.generated_text, protocol=self.args.tgvf_protocol)
                row.update(self._capture_fields(capture, parsed))
                row.update(final_raw_output=capture.generated_text, focus_miss=True)
                rows.append(row)
                continue
            correct_d = self._d_from_capture(sample, capture, focus_source="force_generated")
            for condition in conditions:
                rows.append(
                    self._run_post_tgvf_condition(
                        sample=sample,
                        capture=capture,
                        correct_d=correct_d,
                        condition=condition,
                        block="force_end2end",
                        focus_source="force_generated",
                    )
                )
        return rows

    @torch.no_grad()
    def eval_free_router_end2end(self) -> list[dict[str, Any]]:
        rows = []
        focus_conditions = self._d_conditions()
        samples = [*self.focus_samples, *self.no_focus_samples]
        for sample in self._iter(samples, "free_router_end2end"):
            capture = self._capture_free_router(sample)
            parsed = parse_v3_action(capture.generated_text, protocol=self.args.tgvf_protocol)
            trigger = capture.capture_found and (
                parsed.evidence_state == NEED_LOCAL_EVIDENCE
                or self.args.tgvf_protocol in {PROTOCOL_C_THINKING_SPECIAL, PROTOCOL_C_TOOL_OBSERVATION}
            )
            if not trigger:
                row = self._base_row(sample, "free_router_end2end", "direct_or_miss")
                row.update(self._capture_fields(capture, parsed))
                row.update(
                    final_raw_output=capture.generated_text,
                    parsed_answer=parsed.answer,
                    answer_parse_success=parsed.answer_valid,
                    trigger_focus_decision=False,
                    focus_miss=sample.need_focus,
                    no_focus_false_trigger=False,
                    continuation_not_im_end=_continuation_not_im_end(capture.generated_text),
                )
                self._add_answer_metrics(row, sample)
                rows.append(row)
                continue
            correct_d = self._d_from_capture(sample, capture, focus_source="free_router")
            for condition in focus_conditions:
                rows.append(
                    self._run_post_tgvf_condition(
                        sample=sample,
                        capture=capture,
                        correct_d=correct_d,
                        condition=condition,
                        block="free_router_end2end",
                        focus_source="free_router",
                    )
                )
        return rows

    def _run_post_tgvf_condition(
        self,
        *,
        sample: TGVFv3Stage2Sample,
        capture: Qwen3FocusCapture,
        correct_d: torch.Tensor,
        condition: str,
        block: str,
        focus_source: str,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        row = self._base_row(sample, block, condition)
        row["focus_source"] = focus_source
        parsed_focus = parse_v3_action(capture.generated_text, protocol=self.args.tgvf_protocol)
        row.update(self._capture_fields(capture, parsed_focus))
        try:
            d = self._condition_d(sample, correct_d, condition)
            if d is None:
                append_result = self._append_text_only_no_d(capture)
            else:
                append_result = self._append_visual_d(capture, d.to(self.device))
            continuation = continue_generation_qwen3(
                self.model,
                self.processor,
                append_result,
                max_new_tokens=self.args.max_answer_tokens,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
            full_text = _full_protocol_text(
                capture.generated_text,
                continuation.generated_text,
                protocol=self.args.tgvf_protocol,
            )
            parsed_full = parse_v3_action(full_text, protocol=self.args.tgvf_protocol)
            row.update(
                final_raw_output=continuation.generated_text,
                full_protocol_text=full_text,
                parsed_evidence_state=parsed_full.evidence_state or parsed_focus.evidence_state,
                parsed_focus_target=parsed_focus.focus_target,
                parsed_evidence=_parsed_evidence_text(full_text, self.args.tgvf_protocol),
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
                    capture.second_full_forward_used
                    or append_result.debug_metadata.get("second_full_forward_used")
                ),
                wall_time_sec=time.perf_counter() - start,
            )
        except Exception as exc:
            row.update(
                errors=[f"{type(exc).__name__}:{exc}"],
                append_success=False,
                continuation_not_im_end=False,
                wall_time_sec=time.perf_counter() - start,
            )
        self._add_answer_metrics(row, sample)
        return row

    def _condition_d(
        self,
        sample: TGVFv3Stage2Sample,
        correct_d: torch.Tensor,
        condition: str,
    ) -> torch.Tensor | None:
        if condition == "correct_D":
            return correct_d
        if condition == "no_D":
            return None
        if condition == "random_D":
            return make_smoke_d(
                source="random_calibrated",
                num_fvt_tokens=int(correct_d.shape[0]),
                hidden_dim=int(correct_d.shape[-1]),
                reference=correct_d.detach().cpu(),
                device=self.device,
                dtype=correct_d.dtype,
            )
        if condition == "wrong_same_image_D":
            wrong = self._wrong_sample(sample, same_image=True, required_tokens=int(correct_d.shape[0]))
            if wrong is None:
                raise RuntimeError("wrong_same_image_D_unavailable")
            return self._teacher_item(wrong)["d"]
        if condition == "wrong_diff_image_D":
            wrong = self._wrong_sample(sample, same_image=False, required_tokens=int(correct_d.shape[0]))
            if wrong is None:
                raise RuntimeError("wrong_diff_image_D_unavailable")
            return self._teacher_item(wrong)["d"]
        raise ValueError(f"unknown D condition: {condition}")

    def _teacher_item(self, sample: TGVFv3Stage2Sample) -> dict[str, Any]:
        index = self.focus_index[id(sample)]
        if index in self.teacher_cache:
            return self.teacher_cache[index]
        capture = self._capture_teacher_forced(sample)
        d = self._d_from_capture(sample, capture, focus_source="teacher_forced")
        item = {"capture": capture, "d": d.detach()}
        self.teacher_cache[index] = item
        return item

    def _d_from_capture(
        self,
        sample: TGVFv3Stage2Sample,
        capture: Qwen3FocusCapture,
        *,
        focus_source: str,
    ) -> torch.Tensor:
        tap, v_pre, _v_merge = self._vision_features(sample)
        if v_pre is None:
            raise RuntimeError(f"Qwen3 V_pre tap failed: {tap.errors}")
        output = self.foveal_module(
            target_hidden_states=capture.target_hidden_states.to(self.device),
            pre_merge_visual_tokens=v_pre.to(self.device),
            metadata={
                "target": capture.target_text,
                "stage": "tgvf_v3_stage2_protocol_eval",
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

    def _vision_features(self, sample: TGVFv3Stage2Sample):
        key = f"{sample.image}|{self.args.max_image_resolution}"
        if key not in self.vision_cache:
            image = self._image(sample)
            self.vision_cache[key] = tap_qwen3_vision_features(
                self.utility_model,
                self.processor,
                image=image,
                question=sample.prompt_question,
                device=self.device,
            )
        return self.vision_cache[key]

    def _capture_teacher_forced(self, sample: TGVFv3Stage2Sample) -> Qwen3FocusCapture:
        image = self._image(sample)
        return self._capture_with_forced_text(
            image=image,
            question=sample.prompt_question,
            forced_text=render_focus_action_text(sample.target, protocol=self.args.tgvf_protocol),
        )

    def _capture_generated_focus(self, sample: TGVFv3Stage2Sample, *, force_prefix: bool) -> Qwen3FocusCapture:
        image = self._image(sample)
        forced_text = self._force_prefix_text(sample) if force_prefix else None
        if forced_text is not None:
            return self._capture_with_forced_text(
                image=image,
                question=sample.prompt_question,
                forced_text=forced_text,
            )
        return capture_focus_single_pass_qwen3(
            self.model,
            self.processor,
            image=image,
            question=sample.prompt_question,
            messages=build_direct_messages(image, sample.prompt_question),
            max_new_tokens=self.args.max_action_tokens,
            device=self.device,
            protocol=self.args.tgvf_protocol,
        )

    def _force_prefix_mode_name(self) -> str:
        if self.args.tgvf_protocol not in {PROTOCOL_C_THINKING_SPECIAL, PROTOCOL_C_TOOL_OBSERVATION, PROTOCOL_E_ACTION_EVIDENCE_SPECIAL}:
            return "legacy_action_prefix"
        return self.args.force_prefix_mode

    def _force_prefix_text(self, sample: TGVFv3Stage2Sample) -> str:
        if self.args.tgvf_protocol == "legacy_v3_tags":
            return render_force_focus_prefix(protocol=self.args.tgvf_protocol)
        if self.args.tgvf_protocol == PROTOCOL_D_QWEN_TOOL:
            return render_force_focus_prefix(protocol=self.args.tgvf_protocol)
        mode = self.args.force_prefix_mode
        if mode == "target_hint":
            return render_force_focus_prefix(
                protocol=self.args.tgvf_protocol,
                target_hint=sample.target,
            )
        if mode == "empty_think_then_focus_start":
            return render_force_focus_prefix(
                protocol=self.args.tgvf_protocol,
                target_hint=None,
            )
        if mode == "think_open_only":
            return f"{THINK_START}\n"
        if mode == "generic_need_then_focus_start":
            return (
                f"{THINK_START}\n"
                "I need to inspect local visual evidence before answering.\n"
                f"{THINK_END}\n"
                "<|focus_start|>"
            )
        if mode == "specific_region_think_open":
            return f"{THINK_START}\nI need to inspect a specific local region before answering."
        if mode == "inspect_think_open":
            return f"{THINK_START}\nI need to inspect"
        raise ValueError(f"unsupported force prefix mode: {mode}")

    def _capture_with_forced_text(
        self,
        *,
        image: dict[str, Any],
        question: str,
        forced_text: str,
    ) -> Qwen3FocusCapture:
        inputs = build_qwen3_inputs(self.processor, build_direct_messages(image, question))
        return capture_focus_single_pass_from_inputs_qwen3(
            max_new_tokens=self.args.max_action_tokens,
            model=self.model,
            tokenizer=self.processor.tokenizer,
            inputs=inputs,
            device=self.device,
            forced_prefix_text=forced_text,
            protocol=self.args.tgvf_protocol,
        )

    def _capture_free_router(self, sample: TGVFv3Stage2Sample) -> Qwen3FocusCapture:
        image = self._image(sample)
        messages = build_direct_messages(image, sample.prompt_question)
        return capture_focus_single_pass_qwen3(
            self.model,
            self.processor,
            image=image,
            question=sample.prompt_question,
            messages=messages,
            max_new_tokens=self.args.max_action_tokens,
            device=self.device,
            force_action_prefix=False,
            protocol=self.args.tgvf_protocol,
        )

    def _append_text_only_no_d(self, capture: Qwen3FocusCapture) -> Qwen3AppendResult:
        tokenizer = self.processor.tokenizer
        start, end = render_tgvf_prefix_suffix(protocol=self.args.tgvf_protocol)
        text = f"{start}{end}"
        if self.args.tgvf_protocol in {PROTOCOL_C_THINKING_SPECIAL, PROTOCOL_C_TOOL_OBSERVATION, PROTOCOL_D_QWEN_TOOL}:
            text += f"{THINK_START}\n"
        else:
            text += EVIDENCE_START
        token_ids = _encode_text(tokenizer, text, self.device)
        attention_mask = capture.attention_mask
        if attention_mask is not None:
            attention_mask = _extend_attention(attention_mask.to(self.device), int(token_ids.shape[0]))
        input_ids = capture.input_ids
        if input_ids is not None:
            input_ids = torch.cat([input_ids.to(self.device), token_ids.view(1, -1)], dim=-1)
        position_ids = _chunk_position_ids_1d(
            attention_mask=attention_mask,
            chunk_length=int(token_ids.shape[0]),
            device=self.device,
        )
        outputs = self.model(
            input_ids=token_ids.view(1, -1).to(self.device),
            past_key_values=capture.past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )
        return Qwen3AppendResult(
            past_key_values=outputs.past_key_values,
            attention_mask=attention_mask,
            cache_position=None,
            input_ids=input_ids,
            last_logits=outputs.logits,
            appended_token_ids=token_ids.detach().cpu(),
            appended_inputs_embeds=torch.empty(0),
            fvt_token_start=-1,
            fvt_token_end=-1,
            model_kwargs={},
            debug_metadata={
                "fvt_append_path": "text_only_no_D",
                "tgvf_protocol": self.args.tgvf_protocol,
                "fvt_shape": None,
                "num_fvt_tokens": 0,
                "fvt_position_mode": "text_only_no_D",
                "second_full_forward_used": False,
            },
        )

    def _append_visual_d(self, capture: Qwen3FocusCapture, d: torch.Tensor) -> Qwen3AppendResult:
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
            raise ValueError(f"FVT token count {int(d.shape[0])} != source token count {source_token_count}")
        embed = self.model.get_input_embeddings()
        hidden_dim = int(embed.weight.shape[-1])
        if int(d.shape[-1]) != hidden_dim:
            raise ValueError(f"FVT dim {int(d.shape[-1])} != Qwen hidden dim {hidden_dim}")
        prefix, suffix = render_tgvf_prefix_suffix(protocol=self.args.tgvf_protocol)
        suffix += f"{THINK_START}\n" if self.args.tgvf_protocol in {PROTOCOL_C_THINKING_SPECIAL, PROTOCOL_C_TOOL_OBSERVATION, PROTOCOL_D_QWEN_TOOL} else EVIDENCE_START
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
            attention_mask = _extend_attention(attention_mask.to(self.device), int(token_ids.shape[0]))
        input_ids = capture.input_ids
        if input_ids is not None:
            input_ids = torch.cat([input_ids.to(self.device), token_ids.view(1, -1).to(self.device)], dim=-1)
        mm_token_type_ids = _fvt_mm_token_type_ids(
            chunk_length=int(token_ids.shape[0]),
            fvt_token_start=fvt_token_start,
            fvt_token_end=fvt_token_end,
            device=self.device,
        )
        if self.args.fvt_position_mode == "native_source_grid":
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
        else:
            if source_geometry.source_visual_position_ids is None:
                raise ValueError("source visual position ids are unavailable")
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
        return Qwen3AppendResult(
            past_key_values=outputs.past_key_values,
            attention_mask=attention_mask,
            cache_position=None,
            input_ids=input_ids,
            last_logits=outputs.logits,
            appended_token_ids=token_ids.detach().cpu(),
            appended_inputs_embeds=embeds.detach().cpu(),
            fvt_token_start=fvt_token_start,
            fvt_token_end=fvt_token_end,
            model_kwargs={},
            debug_metadata={
                "fvt_append_path": "qwen3_visual_special_tokens_embedding_replace_lora_forward",
                "tgvf_protocol": self.args.tgvf_protocol,
                "uses_deepstack_for_fvt": False,
                "fvt_shape": list(d.shape),
                "num_fvt_tokens": int(d.shape[0]),
                "source_visual_token_count": source_token_count,
                "fvt_position_mode": self.args.fvt_position_mode,
                "position_ids_shape": list(position_ids.shape) if position_ids is not None else None,
                "mm_token_type_ids_shape": list(mm_token_type_ids.shape),
                "native_qwen3_position_compute_used": self.args.fvt_position_mode == "native_source_grid",
                "second_full_forward_used": False,
                "past_key_values_preserved": capture.past_key_values is not None and outputs.past_key_values is not None,
                "deepstack_caution": "FVT append uses visual special tokens and Qwen3 3D positions, but no native DeepStack features for FVT.",
            },
        )

    def _base_row(self, sample: TGVFv3Stage2Sample, block: str, method: str) -> dict[str, Any]:
        return {
            "id": _sample_uid(sample),
            "block": block,
            "method": method,
            "tgvf_protocol": self.args.tgvf_protocol,
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

    def _capture_fields(self, capture: Qwen3FocusCapture, parsed: Any) -> dict[str, Any]:
        target = capture.target_text or parsed.focus_target
        return {
            "focus_raw_output": capture.generated_text,
            "raw_output": capture.generated_text,
            "parsed_evidence_state": parsed.evidence_state,
            "parsed_focus_target": target,
            "malformed": bool(capture.malformed or parsed.malformed),
            "target_length": len(target.split()),
            "generic_target_flag": is_generic_target(target) if target else False,
            "target_answer_leakage_flag": False,
            "focus_valid": bool(capture.capture_found and not capture.malformed),
            "second_full_forward_used": bool(capture.second_full_forward_used),
            "H_q_shape": list(capture.target_hidden_states.shape),
            "target_token_count": len(capture.target_token_ids),
            "source_visual_token_count": (
                capture.source_visual_geometry.source_visual_token_count
                if capture.source_visual_geometry is not None
                else 0
            ),
            "capture_stop_reason": capture.stop_reason,
            "capture_errors": list(capture.errors),
        }

    def _add_answer_metrics(self, row: dict[str, Any], sample: TGVFv3Stage2Sample) -> None:
        row["answer_score"] = answer_score(row.get("parsed_answer") or "", sample.answer)
        target = row.get("parsed_focus_target") or ""
        row["target_answer_leakage_flag"] = target_leaks_answer(target, sample.answer)
        if sample.value_span_text:
            row["value_span_text"] = sample.value_span_text

    def _finish_block(self, block: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        out = self.output_dir / f"{block}.jsonl"
        write_jsonl(out, rows)
        summary = summarize_rows(rows)
        write_json(self.output_dir / f"{block}.summary.json", summary)
        return summary

    def _wrong_sample(
        self,
        sample: TGVFv3Stage2Sample,
        *,
        same_image: bool,
        required_tokens: int,
    ) -> TGVFv3Stage2Sample | None:
        if same_image:
            candidates = [item for item in self.focus_groups[_group_key(sample)] if item is not sample]
        else:
            candidates = [item for item in self.focus_samples_all if _group_key(item) != _group_key(sample)]
        for candidate in candidates[: self.args.wrong_search_limit]:
            try:
                item = self._teacher_item(candidate)
                if int(item["d"].shape[0]) == int(required_tokens):
                    return candidate
            except Exception:
                continue
        return None

    def _group_focus_samples(self, samples: list[TGVFv3Stage2Sample]) -> dict[str, list[TGVFv3Stage2Sample]]:
        groups: dict[str, list[TGVFv3Stage2Sample]] = defaultdict(list)
        for sample in samples:
            groups[_group_key(sample)].append(sample)
        return groups

    def _d_conditions(self) -> list[str]:
        return [item.strip() for item in self.args.d_conditions.split(",") if item.strip()]

    def _select_shard(self, samples: list[TGVFv3Stage2Sample], limit: int | None) -> list[TGVFv3Stage2Sample]:
        if limit is not None:
            samples = samples[:limit]
        if self.args.num_shards <= 1:
            return samples
        return [sample for index, sample in enumerate(samples) if index % self.args.num_shards == self.args.shard_index]

    def _iter(self, samples: list[TGVFv3Stage2Sample], label: str):
        total = len(samples)
        for index, sample in enumerate(samples, start=1):
            if self.args.progress and (index == 1 or index % self.args.log_every == 0 or index == total):
                print(json.dumps({"block": label, "index": index, "total": total}, ensure_ascii=False), flush=True)
            yield sample

    def _image(self, sample: TGVFv3Stage2Sample) -> dict[str, Any]:
        return _image_input(sample.image, max_image_resolution=self.args.max_image_resolution)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    triggered = [row for row in rows if row.get("trigger_focus_decision")]
    no_focus_rows = [row for row in rows if not row.get("need_focus")]
    focus_rows = [row for row in rows if row.get("need_focus")]
    by_method: dict[str, Any] = {}
    for method in sorted({str(row.get("method")) for row in rows}):
        method_rows = [row for row in rows if row.get("method") == method]
        by_method[method] = _summary_core(method_rows)
    summary = _summary_core(rows)
    summary.update(
        focus_valid_rate=_mean(row.get("focus_valid") for row in rows),
        malformed_rate=_mean(row.get("malformed") for row in rows),
        free_trigger_rate=_mean(row.get("trigger_focus_decision") for row in rows),
        no_focus_false_trigger_rate=_mean(row.get("no_focus_false_trigger") for row in no_focus_rows),
        focus_miss_rate=_mean(row.get("focus_miss") for row in focus_rows),
        generic_target_rate=_mean(row.get("generic_target_flag") for row in rows if row.get("parsed_focus_target")),
        answer_parse_rate=_mean(row.get("answer_parse_success") for row in rows),
        continuation_not_im_end_rate=_mean(row.get("continuation_not_im_end") for row in rows),
        accuracy=_mean(row.get("answer_score") for row in rows),
        triggered_accuracy=_mean(row.get("answer_score") for row in triggered),
        second_full_forward_used_any=any(bool(row.get("second_full_forward_used")) for row in rows),
        by_method=by_method,
    )
    return summary


def _summary_core(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "answer_parse_rate": _mean(row.get("answer_parse_success") for row in rows),
        "continuation_not_im_end_rate": _mean(row.get("continuation_not_im_end") for row in rows),
        "accuracy": _mean(row.get("answer_score") for row in rows),
        "malformed_rate": _mean(row.get("malformed") for row in rows),
        "avg_target_len": _mean(row.get("target_length") for row in rows if row.get("target_length")),
        "avg_wall_time_sec": _mean(row.get("wall_time_sec") for row in rows),
        "append_success_rate": _mean(row.get("append_success") for row in rows if row.get("method") != "direct"),
    }


def _full_protocol_text(action_text: str, continuation: str, *, protocol: str) -> str:
    if protocol in {PROTOCOL_C_THINKING_SPECIAL, PROTOCOL_C_TOOL_OBSERVATION}:
        return (
            f"{action_text}\n"
            "<|tgvf_start|>\n[visual embeddings]\n<|tgvf_end|>\n"
            f"{THINK_START}\n{continuation}"
        )
    if protocol == PROTOCOL_D_QWEN_TOOL:
        return (
            f"{action_text}\n"
            "<tool_response>\n[visual embeddings]\n</tool_response>\n"
            f"{THINK_START}\n{continuation}"
        )
    return f"{action_text}\n{TGVF_START}\n[visual embeddings]\n{TGVF_END}\n{EVIDENCE_START}{continuation}"


def _parsed_evidence_text(text: str, protocol: str) -> str:
    if protocol in {PROTOCOL_C_THINKING_SPECIAL, PROTOCOL_C_TOOL_OBSERVATION, PROTOCOL_D_QWEN_TOOL}:
        matches = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        return matches[-1].strip() if matches else ""
    return _extract_tag(text, EVIDENCE_START, EVIDENCE_END) or ""


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
    stripped = re.sub(r"\s+", "", text or "")
    if not stripped:
        return False
    return stripped not in {"<|im_end|>", "<|endoftext|>"}


def target_leaks_answer(target: str, answer: str) -> bool:
    norm_target = normalize_answer(target)
    norm_answer = normalize_answer(answer)
    return bool(norm_answer and norm_answer in norm_target)


def answer_score(prediction: str, answer: str) -> float:
    pred = normalize_answer(prediction)
    gold = normalize_answer(answer)
    if not pred or not gold:
        return 0.0
    if pred == gold:
        return 1.0
    if gold in pred or pred in gold:
        return 1.0
    return 0.0


def normalize_answer(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _sample_uid(sample: TGVFv3Stage2Sample) -> str:
    base = "|".join([sample.image_id or sample.image, sample.question, sample.target, sample.answer])
    return str(abs(hash(base)))


def _group_key(sample: TGVFv3Stage2Sample) -> str:
    return sample.image_id or sample.image


def _mean(values) -> float | None:
    vals = [float(value) for value in values if value is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TGVF-v3 Stage2 protocol inference.")
    parser.add_argument("--stage2-checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--eval-jsonl", default=DEFAULT_VAL)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--processor-id", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--tgvf-protocol", choices=TGVF_PROTOCOL_CHOICES, default="legacy_v3_tags")
    parser.add_argument("--variant", choices=TGVF_VARIANTS, default="tgvf_v2_bidirectional")
    parser.add_argument("--num-foveated-tokens", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=256)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--max-image-resolution", type=int, default=512)
    parser.add_argument("--max-action-tokens", type=int, default=96)
    parser.add_argument("--max-answer-tokens", type=int, default=128)
    parser.add_argument("--fvt-position-mode", choices=("native_source_grid", "inherit_source_visual_positions"), default="native_source_grid")
    parser.add_argument("--blocks", default="no_focus_direct,force_focus_targets,teacher_forced_post_tgvf,force_end2end,free_router_end2end")
    parser.add_argument("--d-conditions", default="correct_D,no_D,random_D,wrong_same_image_D,wrong_diff_image_D")
    parser.add_argument("--max-focus", type=int, default=None)
    parser.add_argument("--max-no-focus", type=int, default=None)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--wrong-search-limit", type=int, default=256)
    parser.add_argument(
        "--force-prefix-mode",
        choices=(
            "target_hint",
            "empty_think_then_focus_start",
            "think_open_only",
            "generic_need_then_focus_start",
            "specific_region_think_open",
            "inspect_think_open",
        ),
        default="target_hint",
    )
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    if args.num_shards < 1:
        parser.error("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must satisfy 0 <= shard-index < num-shards")
    return args


def main() -> None:
    args = parse_args()
    evaluator = Stage2ProtocolEvaluator(args)
    evaluator.load()
    evaluator.run()


if __name__ == "__main__":
    main()
