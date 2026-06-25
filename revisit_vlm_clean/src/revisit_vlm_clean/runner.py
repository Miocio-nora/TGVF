"""Executable clean benchmark runner backends."""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .benchmark_data import BenchmarkSample
from .rendering import RenderedBenchmarkInput
from .schema import EvalMode, EvalSummary, RunConfig, ScoringBackend
from .scoring import parse_and_score


@dataclass(frozen=True)
class BackendConfig:
    backend: str = "dry_run"
    dtype: str = "bfloat16"
    device: str = "auto"
    device_map: str | None = "auto"
    attn_implementation: str | None = "sdpa"
    trust_remote_code: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "dtype": self.dtype,
            "device": self.device,
            "device_map": self.device_map,
            "attn_implementation": self.attn_implementation,
            "trust_remote_code": self.trust_remote_code,
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

    def run(self, sample: BenchmarkSample, rendered: RenderedBenchmarkInput, config: RunConfig) -> ModelRunResult:
        raise NotImplementedError


class DryRunBackend(CleanRunnerBackend):
    def run(self, sample: BenchmarkSample, rendered: RenderedBenchmarkInput, config: RunConfig) -> ModelRunResult:
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
            raise NotImplementedError("qwen3_original backend currently supports only mode='original'")
        self._load()

    def run(self, sample: BenchmarkSample, rendered: RenderedBenchmarkInput, config: RunConfig) -> ModelRunResult:
        started = time.perf_counter()
        try:
            if config.mode != EvalMode.ORIGINAL:
                raise NotImplementedError("qwen3_original backend currently supports only mode='original'")
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
        from transformers import AutoConfig, AutoProcessor, Qwen2VLForConditionalGeneration, Qwen3VLForConditionalGeneration

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
        image_patch_size = int(getattr(getattr(processor, "image_processor", None), "patch_size", 16) or 16)
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
            video_inputs, video_metadatas = zip(*video_inputs)
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


def make_backend(
    backend_config: BackendConfig,
    *,
    config: RunConfig,
) -> CleanRunnerBackend:
    if backend_config.backend == "dry_run":
        return DryRunBackend()
    if backend_config.backend == "qwen3_original":
        return Qwen3OriginalBackend(
            model_id=config.model_id,
            processor_id=config.processor_id,
            backend_config=backend_config,
            max_image_resolution=config.max_image_resolution,
            max_answer_tokens=config.max_answer_tokens,
        )
    raise ValueError(f"unknown clean runner backend: {backend_config.backend}")


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
    for sample, rendered in zip(samples, rendered_inputs, strict=True):
        result = backend.run(sample, rendered, config)
        if result.error:
            parsed_answer = ""
            score = None
            answer_parse_success = False
        else:
            parsed = parse_and_score(
                result.raw_output,
                choices=list(sample.choices),
                gold_answer=sample.gold_answer,
                scoring_backend=ScoringBackend.PROJECT,
            )
            parsed_answer = parsed.parsed_answer
            score = parsed.score
            answer_parse_success = parsed.answer_parse_success
        rows.append(
            {
                "sample_id": sample.sample_id,
                "benchmark": sample.benchmark,
                "population_id": sample.population_id,
                "subset_id": config.subset_id,
                "source_file": sample.source_file,
                "method": config.mode.value,
                "runner_backend": backend_config.backend,
                "question": sample.question,
                "choices": list(sample.choices),
                "gold_answer": sample.gold_answer,
                "raw_output": result.raw_output,
                "parsed_answer": parsed_answer,
                "score": score,
                "answer_parse_success": answer_parse_success,
                "malformed": bool(result.error),
                "trigger_focus_decision": result.triggered,
                "focus_valid": result.focus_valid,
                "focus_target": result.focus_target,
                "append_success": result.append_success,
                "output_tokens": result.output_tokens,
                "wall_time_sec": result.wall_time_sec,
                "metadata": sample.metadata,
                "debug_metadata": result.debug,
                "error": result.error,
            }
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
    focus_values = [bool(row.get("focus_valid")) for row in rows if row.get("focus_valid") is not None]
    append_values = [bool(row.get("append_success")) for row in rows if row.get("append_success") is not None]
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


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return None if not values else sum(values) / len(values)


def _mean_bool(values: Iterable[bool]) -> float | None:
    values = list(values)
    return None if not values else sum(1.0 for item in values if item) / len(values)
