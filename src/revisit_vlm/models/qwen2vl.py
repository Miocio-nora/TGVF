from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


@dataclass
class LoadedQwen2VL:
    model: Qwen2VLForConditionalGeneration
    processor: Any


def load_qwen2vl(
    model_name_or_path: str,
    *,
    torch_dtype: str = "auto",
    attn_implementation: str | None = "flash_attention_2",
    device_map: str | dict[str, Any] | None = None,
    trust_remote_code: bool = True,
) -> LoadedQwen2VL:
    """Load Qwen2-VL in one place so research changes stay centralized."""

    dtype = _resolve_dtype(torch_dtype)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    if device_map:
        model_kwargs["device_map"] = device_map

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return LoadedQwen2VL(model=model, processor=processor)


def _resolve_dtype(torch_dtype: str) -> torch.dtype | str:
    if torch_dtype == "auto":
        return "auto"
    if torch_dtype in {"float16", "fp16"}:
        return torch.float16
    if torch_dtype in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if torch_dtype in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")
