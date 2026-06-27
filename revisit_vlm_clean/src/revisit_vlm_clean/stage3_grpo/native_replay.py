"""Native Stage3 policy/ref logprob replay helpers."""

from __future__ import annotations

from typing import Any

from .schemas import Stage3GRPOConfig, Stage3Sample


def selected_token_logprobs_from_logits(
    logits: Any,
    *,
    generated_token_ids: list[int],
    prompt_len: int,
) -> Any:
    """Gather next-token logprobs for generated ids from full-sequence logits.

    For a prompt length P and generated ids y_0..y_n, the probability of y_0 is
    read from logits at position P-1, y_1 from P, etc. This helper is intentionally
    small and framework-owned so both focus replay and post-D continuation replay
    use exactly the same alignment rule.
    """
    import torch

    if not generated_token_ids:
        return torch.empty((0,), dtype=logits.dtype, device=logits.device)
    if logits.ndim != 3 or int(logits.shape[0]) != 1:
        raise ValueError("logits must have shape [1, seq, vocab]")
    start = int(prompt_len) - 1
    end = start + len(generated_token_ids)
    if start < 0 or end > int(logits.shape[1]):
        raise ValueError(
            "logits sequence is too short for generated token replay: "
            f"prompt_len={prompt_len} generated={len(generated_token_ids)} logits_seq={int(logits.shape[1])}"
        )
    positions = logits[:, start:end, :].float()
    ids = torch.tensor(generated_token_ids, dtype=torch.long, device=logits.device).view(1, -1, 1)
    return torch.log_softmax(positions, dim=-1).gather(-1, ids).view(-1)


def replay_focus_segment_logprobs(
    *,
    model: Any,
    processor: Any,
    sample: Stage3Sample,
    generated_token_ids: list[int],
    config: Stage3GRPOConfig,
    device: Any,
) -> Any:
    """Replay the initial image+question -> action/direct-answer segment.

    This is the first differentiable replay piece needed for real native GRPO.
    It covers both direct no-tool outputs and the focus/action prefix generated
    before D append. Post-D continuation replay is implemented separately because
    it must rebuild the TGVF append state.
    """
    import torch
    from revisit_vlm.qwen3_vl_tgvf import build_direct_messages, build_qwen3_inputs
    from revisit_vlm.tgvf_v3_stage1 import _image_input

    if not generated_token_ids:
        return torch.empty((0,), dtype=torch.float32, device=device)
    image = _image_input(sample.image_path, max_image_resolution=config.max_image_resolution)
    inputs = build_qwen3_inputs(processor, build_direct_messages(image, sample.question))
    model_inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    prompt_ids = model_inputs["input_ids"]
    generated = torch.tensor([generated_token_ids], dtype=torch.long, device=prompt_ids.device)
    replay_inputs = replay_inputs_with_generated_text(model_inputs, generated)
    outputs = model(**replay_inputs, use_cache=False, return_dict=True)
    return selected_token_logprobs_from_logits(
        outputs.logits,
        generated_token_ids=generated_token_ids,
        prompt_len=int(prompt_ids.shape[-1]),
    )


def replay_inputs_with_generated_text(model_inputs: dict[str, Any], generated_ids: Any) -> dict[str, Any]:
    """Append generated text ids to Qwen-VL replay inputs.

    Qwen3-VL uses `mm_token_type_ids` when computing 3D position ids. If we
    append text tokens to `input_ids` and `attention_mask` without extending
    `mm_token_type_ids`, the model sees mismatched sequence lengths during
    replay. Generated text tokens are ordinary language tokens, so their
    multimodal token type is zero.
    """
    import torch

    if "input_ids" not in model_inputs:
        raise ValueError("model_inputs must contain input_ids")
    input_ids = model_inputs["input_ids"]
    generated = generated_ids.view(1, -1).to(device=input_ids.device)
    generated_len = int(generated.shape[-1])
    appended = dict(model_inputs)
    appended["input_ids"] = torch.cat([input_ids, generated], dim=-1)
    attention_mask = model_inputs.get("attention_mask")
    if isinstance(attention_mask, torch.Tensor):
        ones = torch.ones(
            (attention_mask.shape[0], generated_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        appended["attention_mask"] = torch.cat([attention_mask, ones], dim=-1)
    mm_token_type_ids = model_inputs.get("mm_token_type_ids")
    if isinstance(mm_token_type_ids, torch.Tensor):
        zeros = torch.zeros(
            (mm_token_type_ids.shape[0], generated_len),
            dtype=mm_token_type_ids.dtype,
            device=mm_token_type_ids.device,
        )
        appended["mm_token_type_ids"] = torch.cat([mm_token_type_ids, zeros], dim=-1)
    appended.pop("position_ids", None)
    appended.pop("cache_position", None)
    appended.pop("rope_deltas", None)
    return appended
