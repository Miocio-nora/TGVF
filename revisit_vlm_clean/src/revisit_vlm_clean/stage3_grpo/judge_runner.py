"""Offline Stage3 judge runner with local Qwen-VL support."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from revisit_vlm_clean.schema import _to_jsonable

from .schemas import STAGE3_GRPO_JUDGE_SCHEMA_VERSION, now_iso, write_json

JUDGE_RUN_SCHEMA_VERSION = "stage3_grpo_offline_judge_run_v0"
DEFAULT_MODEL_ROOT = os.environ.get("TGVF_JUDGE_MODEL_ROOT", "/nvmesv/dredvpn009/models/hf")

MODEL_PRESETS = {
    "qwen3_vl_2b_thinking": "Qwen/Qwen3-VL-2B-Thinking",
    "qwen3_vl_4b_thinking": "Qwen/Qwen3-VL-4B-Thinking",
    "qwen3_vl_8b_thinking": "Qwen/Qwen3-VL-8B-Thinking",
    "qwen3_vl_30b_a3b_thinking": "Qwen/Qwen3-VL-30B-A3B-Thinking",
    "qwen3_vl_32b_thinking": "Qwen/Qwen3-VL-32B-Thinking",
    "qwen3_vl_32b_thinking_fp8": "Qwen/Qwen3-VL-32B-Thinking-FP8",
    "qwen3_vl_235b_a22b_thinking": "Qwen/Qwen3-VL-235B-A22B-Thinking",
    "qwen3_vl_235b_a22b_thinking_fp8": "Qwen/Qwen3-VL-235B-A22B-Thinking-FP8",
    "qwen25_vl_72b_instruct": "Qwen/Qwen2.5-VL-72B-Instruct",
    "qvq_72b_preview": "Qwen/QVQ-72B-Preview",
}


@dataclass(frozen=True)
class OfflineJudgeConfig:
    pending_path: str
    output_dir: str
    focus_cache_path: str | None = None
    grounding_cache_path: str | None = None
    backend: str = "local_qwen_vl"
    model_id: str = "Qwen/Qwen3-VL-32B-Thinking"
    processor_id: str | None = None
    model_root: str = DEFAULT_MODEL_ROOT
    require_local_model: bool = True
    dtype: str = "bfloat16"
    device: str = "auto"
    device_map: str | None = "auto"
    attn_implementation: str | None = "sdpa"
    trust_remote_code: bool = False
    max_image_resolution: int = 768
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    limit: int | None = None
    skip_existing: bool = True
    append: bool = True
    prompt_version: str = "stage3_grpo_local_judge_v0"

    def validate(self) -> None:
        if not self.pending_path:
            raise ValueError("pending_path is required")
        if not self.output_dir:
            raise ValueError("output_dir is required")
        if self.backend not in {"local_qwen_vl", "fake"}:
            raise ValueError("judge backend must be local_qwen_vl or fake")
        if not self.model_id:
            raise ValueError("model_id is required")
        if int(self.max_image_resolution) < 1:
            raise ValueError("max_image_resolution must be >= 1")
        if int(self.max_new_tokens) < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if float(self.temperature) < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < float(self.top_p) <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.limit is not None and int(self.limit) < 1:
            raise ValueError("limit must be >= 1 when set")

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


class JudgeBackend(Protocol):
    def score(self, row: dict[str, Any]) -> dict[str, Any]:
        ...


def run_offline_judge(config: OfflineJudgeConfig, *, preflight_only: bool = False) -> dict[str, Any]:
    config.validate()
    pending_path = Path(config.pending_path)
    if not pending_path.exists():
        raise FileNotFoundError(f"pending_path does not exist: {pending_path}")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    focus_cache_path = Path(config.focus_cache_path or output_dir / "focus_judge_cache.jsonl")
    grounding_cache_path = Path(
        config.grounding_cache_path or output_dir / "grounding_judge_cache.jsonl"
    )
    pending_rows = load_pending_rows(pending_path, limit=config.limit)
    existing_keys = set()
    if config.skip_existing:
        existing_keys |= read_cache_keys(focus_cache_path)
        existing_keys |= read_cache_keys(grounding_cache_path)
    rows_to_score = [
        row for row in pending_rows if str(row.get("cache_key") or "") not in existing_keys
    ]
    missing_images = [
        row.get("image_path")
        for row in rows_to_score
        if row.get("image_path") and not Path(str(row["image_path"])).exists()
    ][:20]
    model_report = judge_model_preflight(config)
    errors: list[str] = []
    if missing_images:
        errors.append(f"missing_images:{len(missing_images)}")
    if config.backend == "local_qwen_vl" and model_report["status"] != "ready":
        errors.extend(str(item) for item in model_report.get("errors", []))
    preflight = {
        "schema_version": "stage3_grpo_offline_judge_preflight_v0",
        "created_at": now_iso(),
        "status": "failed" if errors else "passed",
        "errors": errors,
        "pending_path": str(pending_path),
        "output_dir": str(output_dir),
        "focus_cache_path": str(focus_cache_path),
        "grounding_cache_path": str(grounding_cache_path),
        "backend": config.backend,
        "model_id": config.model_id,
        "processor_id": config.processor_id,
        "model_root": config.model_root,
        "require_local_model": config.require_local_model,
        "model": model_report,
        "pending_rows": len(pending_rows),
        "existing_cache_keys": len(existing_keys),
        "rows_to_score": len(rows_to_score),
        "missing_images": missing_images,
    }
    write_json(output_dir / "judge_preflight_report.json", preflight)
    if preflight_only:
        return {**preflight, "preflight_only": True}
    if errors:
        raise RuntimeError(f"offline judge preflight failed: {errors}")
    backend = build_judge_backend(config)
    scored_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    raw_output_path = output_dir / "judge_raw_outputs.jsonl"
    if not config.append:
        for path in (focus_cache_path, grounding_cache_path, raw_output_path):
            if path.exists():
                path.unlink()
    started = time.perf_counter()
    for row in rows_to_score:
        try:
            scored = backend.score(row)
            scored_rows.append(scored)
            append_jsonl(_cache_path_for_row(scored, focus_cache_path, grounding_cache_path), scored)
            append_jsonl(raw_output_path, _raw_output_row(scored))
        except Exception as exc:
            failure = {
                "schema_version": JUDGE_RUN_SCHEMA_VERSION,
                "created_at": now_iso(),
                "cache_key": row.get("cache_key"),
                "kind": row.get("kind"),
                "sample_id": row.get("sample_id"),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            append_jsonl(output_dir / "judge_failures.jsonl", failure)
    summary = {
        "schema_version": JUDGE_RUN_SCHEMA_VERSION,
        "created_at": now_iso(),
        "status": "completed",
        "pending_path": str(pending_path),
        "output_dir": str(output_dir),
        "focus_cache_path": str(focus_cache_path),
        "grounding_cache_path": str(grounding_cache_path),
        "backend": config.backend,
        "model_id": config.model_id,
        "processor_id": config.processor_id,
        "resolved_model_id": model_report.get("resolved_model_id"),
        "model_root": config.model_root,
        "prompt_version": config.prompt_version,
        "pending_rows": len(pending_rows),
        "skipped_existing": len(pending_rows) - len(rows_to_score),
        "scored_rows": len(scored_rows),
        "failed_rows": len(failures),
        "focus_rows": sum(1 for row in scored_rows if row.get("kind") == "focus"),
        "grounding_rows": sum(1 for row in scored_rows if row.get("kind") == "grounding"),
        "wall_time_sec": time.perf_counter() - started,
        "score_distribution": score_distribution(scored_rows),
    }
    write_json(output_dir / "judge_run_summary.json", summary)
    return summary


def build_judge_backend(config: OfflineJudgeConfig) -> JudgeBackend:
    if config.backend == "fake":
        return FakeJudgeBackend(config)
    return LocalQwenVLJudgeBackend(config)


class FakeJudgeBackend:
    def __init__(self, config: OfflineJudgeConfig) -> None:
        self.config = config

    def score(self, row: dict[str, Any]) -> dict[str, Any]:
        kind = str(row.get("kind") or "")
        target = str(row.get("target") or "")
        if kind == "focus":
            score = 2 if target and len(target.split()) >= 3 else 1 if target else 0
            parsed = {"focus_score": score, "reason": "fake backend deterministic score"}
        elif kind == "grounding":
            reasoning = str(row.get("post_tool_reasoning") or "")
            answer = str(row.get("final_answer") or "")
            score = 2 if reasoning and answer else 1 if answer else 0
            parsed = {"grounding_score": score, "reason": "fake backend deterministic score"}
        else:
            raise ValueError(f"unsupported judge kind: {kind!r}")
        return build_scored_cache_row(
            pending=row,
            parsed=parsed,
            raw_output=json.dumps(parsed),
            backend=self.config.backend,
            model_id=self.config.model_id,
            prompt_version=self.config.prompt_version,
        )


class LocalQwenVLJudgeBackend:
    def __init__(self, config: OfflineJudgeConfig) -> None:
        self.config = config
        self._loaded: tuple[Any, Any] | None = None

    def score(self, row: dict[str, Any]) -> dict[str, Any]:
        prompt = judge_prompt_for_row(row)
        raw_output = self.generate(row, prompt)
        parse_fallback = False
        try:
            parsed = parse_judge_json(raw_output, kind=str(row.get("kind") or ""))
        except Exception as exc:
            parsed = parse_judge_text_fallback(raw_output, kind=str(row.get("kind") or ""))
            parse_fallback = True
        return build_scored_cache_row(
            pending=row,
            parsed=parsed,
            raw_output=raw_output,
            backend=self.config.backend,
            model_id=self.config.model_id,
            prompt_version=self.config.prompt_version,
            parse_fallback=parse_fallback,
        )

    def generate(self, row: dict[str, Any], prompt: str) -> str:
        import torch

        model, processor = self._load()
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a JSON-only visual scoring function. "
                            "Return exactly one compact JSON object and no prose, "
                            "no markdown, no hidden reasoning, no bullet list."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str(row["image_path"]),
                        "max_pixels": int(self.config.max_image_resolution) ** 2,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = build_qwen_vl_inputs(processor, messages)
        device = resolve_generation_device(model, self.config.device)
        model_inputs = move_tensors(inputs, device)
        prompt_len = int(model_inputs["input_ids"].shape[-1])
        generate_kwargs: dict[str, Any] = {
            **model_inputs,
            "max_new_tokens": int(self.config.max_new_tokens),
            "do_sample": float(self.config.temperature) > 0,
        }
        if float(self.config.temperature) > 0:
            generate_kwargs["temperature"] = float(self.config.temperature)
            generate_kwargs["top_p"] = float(self.config.top_p)
        eos_id = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None)
        if eos_id is not None:
            generate_kwargs["eos_token_id"] = eos_id
        with torch.no_grad():
            generated = model.generate(**generate_kwargs)
        new_ids = generated[0, prompt_len:].detach().cpu().tolist()
        tokenizer = getattr(processor, "tokenizer", processor)
        return tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def _load(self) -> tuple[Any, Any]:
        if self._loaded is not None:
            return self._loaded
        import torch
        from transformers import (
            AutoConfig,
            AutoProcessor,
            Qwen2VLForConditionalGeneration,
            Qwen3VLForConditionalGeneration,
            Qwen3VLMoeForConditionalGeneration,
        )

        dtype = torch_dtype(torch, self.config.dtype)
        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "trust_remote_code": self.config.trust_remote_code,
        }
        if self.config.device_map:
            model_kwargs["device_map"] = normalized_device_map(self.config.device_map)
        if self.config.attn_implementation:
            model_kwargs["attn_implementation"] = self.config.attn_implementation
        model_config = AutoConfig.from_pretrained(
            resolved_model_id_for_loading(self.config),
            trust_remote_code=self.config.trust_remote_code,
        )
        model_type = str(getattr(model_config, "model_type", ""))
        if model_type == "qwen3_vl":
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                resolved_model_id_for_loading(self.config),
                **model_kwargs,
            )
        elif model_type == "qwen3_vl_moe":
            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                resolved_model_id_for_loading(self.config),
                **model_kwargs,
            )
        elif model_type == "qwen2_vl":
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                resolved_model_id_for_loading(self.config),
                **model_kwargs,
            )
        else:
            raise ValueError(f"unsupported local judge VLM model_type: {model_type!r}")
        processor = AutoProcessor.from_pretrained(
            resolved_processor_id_for_loading(self.config),
            trust_remote_code=self.config.trust_remote_code,
        )
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        self._loaded = (model, processor)
        return self._loaded


def resolved_model_id_for_loading(config: OfflineJudgeConfig) -> str:
    resolved = resolve_local_or_remote_model_id(
        config.model_id,
        model_root=config.model_root,
        require_local_model=config.require_local_model,
    )
    if resolved["status"] != "ready":
        raise FileNotFoundError("; ".join(str(item) for item in resolved["errors"]))
    return str(resolved["resolved_model_id"])


def resolved_processor_id_for_loading(config: OfflineJudgeConfig) -> str:
    processor_id = config.processor_id or config.model_id
    resolved = resolve_local_or_remote_model_id(
        processor_id,
        model_root=config.model_root,
        require_local_model=config.require_local_model,
    )
    if resolved["status"] != "ready":
        raise FileNotFoundError("; ".join(str(item) for item in resolved["errors"]))
    return str(resolved["resolved_model_id"])


def judge_model_preflight(config: OfflineJudgeConfig) -> dict[str, Any]:
    if config.backend == "fake":
        return {
            "status": "ready",
            "backend": "fake",
            "model_id": config.model_id,
            "resolved_model_id": config.model_id,
            "errors": [],
            "warnings": ["fake judge backend does not load a model"],
        }
    model = resolve_local_or_remote_model_id(
        config.model_id,
        model_root=config.model_root,
        require_local_model=config.require_local_model,
    )
    processor = resolve_local_or_remote_model_id(
        config.processor_id or config.model_id,
        model_root=config.model_root,
        require_local_model=config.require_local_model,
    )
    errors = [*model["errors"], *[f"processor:{item}" for item in processor["errors"]]]
    status = "ready" if not errors else "missing"
    return {
        "status": status,
        "model_id": config.model_id,
        "processor_id": config.processor_id,
        "resolved_model_id": model.get("resolved_model_id"),
        "resolved_processor_id": processor.get("resolved_model_id"),
        "model_root": config.model_root,
        "require_local_model": config.require_local_model,
        "model": model,
        "processor": processor,
        "errors": errors,
        "download_commands": model_download_commands(config.model_id, model_root=config.model_root),
    }


def resolve_local_or_remote_model_id(
    model_id: str,
    *,
    model_root: str,
    require_local_model: bool,
) -> dict[str, Any]:
    requested = MODEL_PRESETS.get(str(model_id), str(model_id))
    path = Path(requested).expanduser()
    candidates = []
    if path.exists():
        candidates.append(path)
    root = Path(model_root).expanduser()
    if "/" in requested and not requested.startswith("/"):
        org, name = requested.split("/", 1)
        candidates.extend([root / name, root / f"{org}--{name.replace('/', '--')}"])
    else:
        candidates.append(root / requested)
    for candidate in candidates:
        if candidate.exists():
            identity = local_model_identity(candidate)
            errors = []
            if not identity["has_required_files"]:
                errors.append(f"local_model_missing_required_files:{candidate}")
            model_type = str(identity.get("model_type") or "")
            if model_type not in {"qwen3_vl", "qwen3_vl_moe", "qwen2_vl"}:
                errors.append(f"unsupported_local_model_type:{model_type or 'missing'}")
            return {
                "status": "ready" if not errors else "missing",
                "requested": model_id,
                "hf_id": requested,
                "resolved_model_id": str(candidate),
                "local": True,
                "identity": identity,
                "errors": errors,
            }
    if require_local_model:
        return {
            "status": "missing",
            "requested": model_id,
            "hf_id": requested,
            "resolved_model_id": None,
            "local": False,
            "candidate_paths": [str(item) for item in candidates],
            "errors": [f"local_model_missing:{requested}"],
        }
    return {
        "status": "ready",
        "requested": model_id,
        "hf_id": requested,
        "resolved_model_id": requested,
        "local": False,
        "candidate_paths": [str(item) for item in candidates],
        "errors": [],
        "warnings": ["remote model loading allowed; Hugging Face cache/network may be used"],
    }


def local_model_identity(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    config_path = path / "config.json"
    tokenizer_path = path / "tokenizer_config.json"
    preprocessor_path = path / "preprocessor_config.json"
    processor_path = path / "processor_config.json"
    model_type = None
    architectures = None
    if config_path.exists():
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            model_type = payload.get("model_type")
            architectures = payload.get("architectures")
        except Exception:
            model_type = "unreadable"
    return {
        "path": str(path),
        "exists": path.exists(),
        "config_json": _small_file_identity(config_path),
        "tokenizer_config_json": _small_file_identity(tokenizer_path),
        "preprocessor_config_json": _small_file_identity(preprocessor_path),
        "processor_config_json": _small_file_identity(processor_path),
        "model_type": model_type,
        "architectures": architectures,
        "has_required_files": bool(config_path.exists() and tokenizer_path.exists() and preprocessor_path.exists()),
    }


def model_download_commands(model_id: str, *, model_root: str) -> list[str]:
    requested = MODEL_PRESETS.get(str(model_id), str(model_id))
    if requested.startswith("/") or Path(requested).expanduser().exists():
        return []
    name = requested.split("/")[-1]
    local_dir = str(Path(model_root).expanduser() / name)
    return [
        f"hf download {requested} --local-dir {local_dir}",
    ]


def _small_file_identity(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"exists": False, "path": str(path)}
    stat = path.stat()
    return {"exists": True, "path": str(path), "size_bytes": int(stat.st_size)}


def judge_prompt_for_row(row: dict[str, Any]) -> str:
    kind = str(row.get("kind") or "")
    common = {
        "question": str(row.get("question") or ""),
        "choices": row.get("choices") or [],
        "target": str(row.get("target") or ""),
        "post_tool_reasoning": str(row.get("post_tool_reasoning") or ""),
        "final_answer": str(row.get("final_answer") or ""),
    }
    if kind == "focus":
        return FOCUS_JUDGE_PROMPT.format(**common)
    if kind == "grounding":
        return GROUNDING_JUDGE_PROMPT.format(**common)
    raise ValueError(f"unsupported judge kind: {kind!r}")


FOCUS_JUDGE_PROMPT = """You are a strict visual-focus judge for TGVF Stage3 RL.

Inspect the image directly. Do not use hidden gold answers. Judge only whether
the generated visual focus target is real, visually executable, and relevant to
the question.

Question:
{question}

Choices, if any:
{choices}

Generated target:
{target}

Score:
2 = target corresponds to real visual content, is specific/executable, and matches evidence needed for the question.
1 = target is related but too broad, vague, partly incomplete, or mildly ambiguous.
0 = target is irrelevant, non-visual, not present, impossible to execute, or directly answers the question instead of guiding observation.

Your entire response must be exactly one JSON object on one line. Do not
explain, do not describe the image, and do not use markdown:
{{"focus_score": 0|1|2, "reason": "short reason"}}
"""


GROUNDING_JUDGE_PROMPT = """You are a strict grounding judge for TGVF Stage3 RL.

Inspect the image directly. Do not use hidden gold answers. Judge whether the
post-tool reasoning makes visually true statements and whether those statements
support the final answer.

Question:
{question}

Choices, if any:
{choices}

Generated target:
{target}

Post-tool reasoning:
{post_tool_reasoning}

Final answer:
{final_answer}

Score:
2 = visual statements are correct and sufficiently support the final answer.
1 = visual statements are mostly correct but incomplete, weakly connected, or underspecified.
0 = reasoning contains visual hallucination, contradicts the image, ignores relevant evidence, or cannot support the final answer.

Your entire response must be exactly one JSON object on one line. Do not
explain, do not describe the image, and do not use markdown:
{{"grounding_score": 0|1|2, "reason": "short reason"}}
"""


def parse_judge_json(raw_output: str, *, kind: str) -> dict[str, Any]:
    text = str(raw_output or "").strip()
    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(item.strip() for item in fenced)
    brace = _extract_first_json_object(text)
    if brace:
        candidates.append(brace)
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception as exc:
            last_error = exc
            continue
        if not isinstance(parsed, dict):
            continue
        return validate_judge_payload(parsed, kind=kind)
    raise ValueError(f"could not parse judge JSON: {last_error}")


def parse_judge_text_fallback(raw_output: str, *, kind: str) -> dict[str, Any]:
    text = str(raw_output or "").strip()
    lower = text.lower()
    if not text:
        raise ValueError("empty judge output")
    field = "focus_score" if kind == "focus" else "grounding_score"
    zero_markers = (
        "score: 0",
        "score 0",
        f"{field}: 0",
        "should be 0",
        "would be 0",
    )
    two_markers = (
        "score: 2",
        "score 2",
        f"{field}: 2",
        "should be 2",
        "would be 2",
    )
    one_markers = (
        "score: 1",
        "score 1",
        f"{field}: 1",
        "should be 1",
        "would be 1",
    )
    if any(marker in lower for marker in zero_markers):
        score = 0
    elif any(marker in lower for marker in two_markers):
        score = 2
    elif any(marker in lower for marker in one_markers):
        score = 1
    elif kind == "focus":
        score = _fallback_focus_score(lower)
    else:
        score = _fallback_grounding_score(lower)
    if score is None:
        raise ValueError(f"could not infer {field} from non-JSON judge output")
    reason = _compact_reason(text)
    return {field: int(score), "score": int(score), "reason": f"parse_fallback: {reason}"}


def _fallback_focus_score(lower: str) -> int | None:
    if any(
        phrase in lower
        for phrase in (
            "not relevant",
            "irrelevant",
            "not present",
            "does not correspond",
            "doesn't correspond",
            "impossible to execute",
            "directly answers the question",
        )
    ):
        return 0
    if any(phrase in lower for phrase in ("too broad", "too vague", "ambiguous")):
        return 1
    if (
        any(phrase in lower for phrase in ("relevant", "matches", "corresponds", "appropriate"))
        and any(phrase in lower for phrase in ("specific", "executable", "visual content", "target"))
    ):
        return 2
    return None


def _fallback_grounding_score(lower: str) -> int | None:
    if any(
        phrase in lower
        for phrase in (
            "hallucination",
            "contradicts",
            "contradict",
            "incorrect",
            "not support",
            "does not support",
            "cannot support",
        )
    ):
        return 0
    if any(phrase in lower for phrase in ("incomplete", "weakly connected", "underspecified")):
        return 1
    if (
        any(phrase in lower for phrase in ("correct", "true", "accurate"))
        and any(phrase in lower for phrase in ("support", "supports", "sufficient"))
    ):
        return 2
    return None


def _compact_reason(text: str) -> str:
    clean = " ".join(str(text or "").split())
    return clean[:300]


def validate_judge_payload(payload: dict[str, Any], *, kind: str) -> dict[str, Any]:
    field = "focus_score" if kind == "focus" else "grounding_score"
    if field not in payload and "score" in payload:
        payload[field] = payload["score"]
    try:
        score = int(payload[field])
    except Exception as exc:
        raise ValueError(f"judge payload missing integer {field}") from exc
    if score not in {0, 1, 2}:
        raise ValueError(f"{field} must be 0, 1, or 2")
    reason = str(payload.get("reason") or "").strip()
    return {field: score, "score": score, "reason": reason[:500]}


def build_scored_cache_row(
    *,
    pending: dict[str, Any],
    parsed: dict[str, Any],
    raw_output: str,
    backend: str,
    model_id: str,
    prompt_version: str,
    parse_fallback: bool = False,
) -> dict[str, Any]:
    kind = str(pending.get("kind") or "")
    payload = {
        "schema_version": STAGE3_GRPO_JUDGE_SCHEMA_VERSION,
        "created_at": now_iso(),
        "status": "scored",
        "kind": kind,
        "cache_key": pending.get("cache_key"),
        "sample_id": pending.get("sample_id"),
        "image_path": pending.get("image_path"),
        "image_sha256": pending.get("image_sha256"),
        "question": pending.get("question"),
        "choices": pending.get("choices") or [],
        "target": pending.get("target") or "",
        "post_tool_reasoning": pending.get("post_tool_reasoning") or "",
        "final_answer": pending.get("final_answer") or "",
        "judge_backend": backend,
        "judge_model": model_id,
        "prompt_version": prompt_version,
        "raw_output": raw_output,
        "parsed_json": dict(parsed),
        "parse_fallback": bool(parse_fallback),
        "reason": parsed.get("reason") or "",
    }
    payload.update(parsed)
    return payload


def build_qwen_vl_inputs(processor: Any, messages: list[dict[str, Any]]) -> dict[str, Any]:
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
    return dict(
        processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            do_resize=False,
            padding=True,
            return_tensors="pt",
            **dict(video_kwargs or {}),
        )
    )


def load_pending_rows(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("kind") or "") not in {"focus", "grounding"}:
                continue
            rows.append(row)
            if limit is not None and len(rows) >= int(limit):
                break
    return rows


def read_cache_keys(path: str | Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()
    keys: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("cache_key") or "")
            if key:
                keys.add(key)
    return keys


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_to_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")


def score_distribution(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out = {"focus": {}, "grounding": {}}
    for row in rows:
        kind = str(row.get("kind") or "")
        field = "focus_score" if kind == "focus" else "grounding_score"
        score = str(row.get(field, "missing"))
        if kind in out:
            out[kind][score] = out[kind].get(score, 0) + 1
    return out


def torch_dtype(torch: Any, name: str) -> Any:
    value = str(name or "bfloat16").lower()
    if value in {"auto"}:
        return "auto"
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def normalized_device_map(value: str | None) -> Any:
    text = str(value or "").strip()
    if text.startswith("cuda:"):
        return {"": text}
    if text == "cuda":
        return {"": "cuda:0"}
    return value


def resolve_generation_device(model: Any, requested: str) -> Any:
    import torch

    if requested and requested != "auto":
        return torch.device(requested)
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def move_tensors(value: Any, device: Any) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tensors(item, device) for item in value]
    return value


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start=start):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _cache_path_for_row(row: dict[str, Any], focus_path: Path, grounding_path: Path) -> Path:
    if row.get("kind") == "focus":
        return focus_path
    if row.get("kind") == "grounding":
        return grounding_path
    raise ValueError(f"unsupported scored row kind: {row.get('kind')!r}")


def _raw_output_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": JUDGE_RUN_SCHEMA_VERSION,
        "created_at": row.get("created_at"),
        "kind": row.get("kind"),
        "cache_key": row.get("cache_key"),
        "sample_id": row.get("sample_id"),
        "judge_backend": row.get("judge_backend"),
        "judge_model": row.get("judge_model"),
        "raw_output": row.get("raw_output"),
        "parsed_json": row.get("parsed_json"),
    }
