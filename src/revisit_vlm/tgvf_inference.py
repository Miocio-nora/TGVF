from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from revisit_vlm.tgvf import FOVEATE_END, FOVEATE_START
from revisit_vlm.tgvf_capture import (
    TGVFCaptureResult,
    capture_tgvf_single_pass,
    trim_capture_target_at_delimiters,
)
from revisit_vlm.tgvf_foveal import (
    BracketedFVTAppendResult,
    FovealCrossAttentionOutput,
    Qwen2VLPreMergeVisualHook,
    append_fvt_result_and_open_answer_turn,
    continue_generation_from_state,
    finalize_tgvf_output_with_frozen_qwen_merger,
)
from revisit_vlm.tgvf_training import ForcedFoveationBracketWrapper, ForcedFoveationWrapper

DEFAULT_CONTINUATION_INSTRUCTION = (
    "The newly provided visual tokens are focused evidence for the target you requested. "
    "Use them to answer the user's original question. Do not mention the foveation process."
)


@dataclass
class TGVFInferenceResult:
    target_text: str
    foveation_request: str
    answer: str
    capture: TGVFCaptureResult
    fvt_output: FovealCrossAttentionOutput | None
    append_result: BracketedFVTAppendResult | None
    generated_ids: list[int]
    stop_reason: str
    debug_metadata: dict[str, Any]


@torch.no_grad()
def run_tgvf_inference(
    *,
    model: Any,
    processor: Any,
    foveal_module: nn.Module,
    image: Any,
    question: str,
    messages: list[dict[str, Any]] | None = None,
    device: torch.device | str | None = None,
    capture_max_new_tokens: int = 128,
    answer_max_new_tokens: int = 64,
    hidden_state_index: int = -1,
    eos_token_id: int | None = None,
    continuation_instruction: str | None = DEFAULT_CONTINUATION_INSTRUCTION,
    force_foveation_markers: bool = False,
    forced_target_max_tokens: int = 32,
    image_kwargs: dict[str, Any] | None = None,
    benchmark_answer_format: str | None = None,
    option_letters: list[str] | tuple[str, ...] | None = None,
    original_question: str | None = None,
    option_texts: list[str] | tuple[str, ...] | None = None,
    repeat_options_in_continuation: bool = False,
    force_target_mode: str = "generated",
    fixed_force_target: str | None = None,
    suppress_im_end_first_token_after_fvt: bool = False,
    fvt_append_mode: str = "qwen_native_pseudo_image",
) -> TGVFInferenceResult:
    """Run structural TGVF inference without full-context replay after foveation.

    Inference flow:
    single-pass capture -> FVT generation -> bracketed FVT append -> continuation.

    This is intentionally different from training's fresh readout branch.
    """

    if device is None:
        device = _infer_model_device(model) or torch.device("cpu")
    model.eval()
    foveal_module.eval()

    capture_model = model
    forced_marker_token_count = 0
    if force_foveation_markers:
        start_ids = processor.tokenizer.encode(FOVEATE_START, add_special_tokens=False)
        end_ids = processor.tokenizer.encode(FOVEATE_END, add_special_tokens=False)
        if force_target_mode == "fixed":
            if not fixed_force_target:
                raise ValueError("force_target_mode='fixed' requires fixed_force_target")
            target_ids = processor.tokenizer.encode(
                str(fixed_force_target), add_special_tokens=False
            )
            forced_ids = [*start_ids, *target_ids, *end_ids]
            forced_marker_token_count = len(forced_ids)
            capture_model = ForcedFoveationWrapper(model, forced_ids)
        elif force_target_mode == "generated":
            forced_marker_token_count = (
                len(start_ids) + int(forced_target_max_tokens) + len(end_ids)
            )
            suppress_ids = (
                [processor.tokenizer.eos_token_id]
                if processor.tokenizer.eos_token_id is not None
                else []
            )
            capture_model = ForcedFoveationBracketWrapper(
                model,
                start_ids=start_ids,
                end_ids=end_ids,
                max_target_tokens=int(forced_target_max_tokens),
                suppress_ids=suppress_ids,
            )
        else:
            raise ValueError("force_target_mode must be 'generated' or 'fixed'")

    with Qwen2VLPreMergeVisualHook(model) as visual_hook:
        capture = capture_tgvf_single_pass(
            capture_model,
            processor,
            image=image,
            question=question,
            messages=messages,
            max_new_tokens=(
                forced_marker_token_count + 4 if force_foveation_markers else capture_max_new_tokens
            ),
            device=device,
            hidden_state_index=hidden_state_index,
            eos_token_id=(
                eos_token_id if eos_token_id is not None else processor.tokenizer.eos_token_id
            ),
            image_kwargs=image_kwargs,
        )

    if capture.capture_found:
        trim_capture_target_at_delimiters(capture, processor.tokenizer)

    if not capture.capture_found:
        return TGVFInferenceResult(
            target_text="",
            foveation_request=capture.generated_text,
            answer="",
            capture=capture,
            fvt_output=None,
            append_result=None,
            generated_ids=[],
            stop_reason=capture.stop_reason,
            debug_metadata={
                "capture_found": False,
                "second_full_forward_used": False,
                "continue_generation_success": False,
                "hard_force_trigger_used": bool(force_foveation_markers),
                "force_foveation_markers_used": bool(force_foveation_markers),
            },
        )
    if visual_hook.pre_merge_visual_tokens is None:
        raise RuntimeError("Pre-merge visual tokens were not captured during inference")

    fvt_output = foveal_module(
        target_hidden_states=capture.target_hidden_states.to(device),
        pre_merge_visual_tokens=visual_hook.pre_merge_visual_tokens.to(device),
        metadata={"target_text": capture.target_text},
    )
    fvt_output = finalize_tgvf_output_with_frozen_qwen_merger(model, fvt_output)
    append_result = append_fvt_result_and_open_answer_turn(
        model=model,
        tokenizer_or_processor=processor,
        generation_state=capture,
        foveated_visual_tokens=fvt_output.foveated_visual_tokens,
        target_text=capture.target_text,
        benchmark_answer_format=benchmark_answer_format,
        option_letters=option_letters,
        original_question=original_question,
        option_texts=option_texts,
        repeat_options_in_continuation=repeat_options_in_continuation,
        append_as_new_user_turn=True,
        fvt_append_mode=fvt_append_mode,
        image_grid_thw=(
            capture.image_grid_thw
            if fvt_output.debug_metadata.get("tgvf_version") == "v2"
            else None
        ),
        device=device,
    )
    continuation = continue_generation_from_state(
        model=model,
        tokenizer_or_processor=processor,
        generation_state=append_result,
        max_new_tokens=answer_max_new_tokens,
        eos_token_id=eos_token_id if eos_token_id is not None else processor.tokenizer.eos_token_id,
        suppress_first_token_ids=(
            [processor.tokenizer.eos_token_id]
            if suppress_im_end_first_token_after_fvt
            and processor.tokenizer.eos_token_id is not None
            else None
        ),
    )
    return TGVFInferenceResult(
        target_text=capture.target_text,
        foveation_request=capture.generated_text,
        answer=continuation.generated_text,
        capture=capture,
        fvt_output=fvt_output,
        append_result=append_result,
        generated_ids=continuation.generated_ids,
        stop_reason=continuation.stop_reason,
        debug_metadata={
            "variant_name": fvt_output.debug_metadata.get("variant_name"),
            "target_text": capture.target_text,
            "target_hidden_shape": list(capture.target_hidden_states.shape),
            "pre_merge_visual_shape": list(visual_hook.pre_merge_visual_tokens.shape),
            "foveated_visual_tokens_shape": list(fvt_output.foveated_visual_tokens.shape),
            "attention_shape": fvt_output.debug_metadata.get("attention_shape"),
            "append_token_count": append_result.debug_metadata["append_token_count"],
            "continuation_instruction_appended": bool(continuation_instruction),
            "instruction_token_count": append_result.debug_metadata.get(
                "instruction_token_count", 0
            ),
            "bracketed_append_used": True,
            "second_full_forward_used": False,
            "cache_preserved": append_result.debug_metadata["cache_preserved"],
            "continue_generation_success": len(continuation.generated_ids) > 0,
            "first_generated_token_after_append": (
                _token_debug(processor.tokenizer, continuation.generated_ids[0])
                if continuation.generated_ids
                else None
            ),
            "stopped_before_generating_answer": (
                len(continuation.generated_ids) == 0
                or (
                    processor.tokenizer.eos_token_id is not None
                    and continuation.generated_ids[0] == processor.tokenizer.eos_token_id
                )
            ),
            "repeat_options_in_continuation": bool(repeat_options_in_continuation),
            "force_target_mode": force_target_mode,
            "fixed_force_target_used": bool(force_target_mode == "fixed"),
            "suppress_im_end_first_token_after_fvt": bool(suppress_im_end_first_token_after_fvt),
            "fvt_append_mode": fvt_append_mode,
            "hard_force_trigger_used": bool(force_foveation_markers),
            "force_foveation_markers_used": bool(force_foveation_markers),
            **append_result.debug_metadata,
        },
    )


def format_tgvf_inference_debug(result: TGVFInferenceResult) -> str:
    return (
        "foveation_request:\n"
        f"{result.foveation_request}\n\n"
        "target_text:\n"
        f"{result.target_text}\n\n"
        "answer:\n"
        f"{result.answer}\n\n"
        "debug_metadata:\n"
        f"{result.debug_metadata}"
    )


def _infer_model_device(model: Any) -> torch.device | None:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return None


def _token_debug(tokenizer: Any, token_id: int) -> dict[str, Any]:
    return {
        "token_id": int(token_id),
        "token_text": tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
    }
