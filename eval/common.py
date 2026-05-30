from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor

from revisit_vlm.models.qwen2vl import load_qwen2vl
from revisit_vlm.tgvf import FOVEATE_END, FOVEATE_START
from revisit_vlm.tgvf_capture import capture_tgvf_single_pass
from revisit_vlm.tgvf_foveal import (
    Qwen2VLPreMergeVisualHook,
    append_fvt_result_and_open_answer_turn,
    continue_generation_from_state,
    finalize_tgvf_output_with_frozen_qwen_merger,
)
from revisit_vlm.tgvf_training import (
    IGNORE_INDEX,
    TGVF_DYNAMIC_NUM_FVT_VARIANTS,
    TGVF_VARIANTS,
    ForcedFoveationWrapper,
    TeacherGuideSample,
    build_tgvf_module,
    collect_training_features,
    compute_readout_lm_loss,
    freeze_qwen2vl,
    infer_tgvf_dims,
    load_tgvf_module_checkpoint,
    prepare_readout_inputs,
    readout_prompt_parts,
)


@dataclass
class EvalSample:
    uid: str
    image: str
    question: str
    target: str
    evidence_description: str
    stable_image_uid: str | None = None
    image_id: str | None = None
    source_dataset: str | None = None
    source_profile: str | None = None
    short_answer: str | None = None
    evidence_type: str | None = None
    locality: str | None = None
    answer_type: str | None = None
    visual_difficulty: str | None = None
    visibility: str | None = None
    confidence: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def group_id(self) -> str:
        return self.stable_image_uid or self.image_id or self.image

    def to_teacher_sample(self) -> TeacherGuideSample:
        return TeacherGuideSample(
            image=self.image,
            question=self.question,
            target=self.target,
            evidence_description=self.evidence_description,
            image_id=self.group_id,
            short_answer=self.short_answer,
            evidence_type=self.evidence_type,
            confidence=self.confidence,
            source_dataset=self.source_dataset,
            metadata=self.raw.get("metadata") or {},
        )


@dataclass
class EvalFeatureCacheItem:
    sample: EvalSample
    target_hidden_states: torch.Tensor
    pre_merge_visual_tokens: torch.Tensor
    merged_visual_tokens: torch.Tensor
    foveated_visual_tokens: torch.Tensor
    image_grid_thw: torch.Tensor | None
    capture_text: str
    shapes: dict[str, list[int]]


def add_common_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--processor-path", default=None)
    parser.add_argument("--tgvf-checkpoint", required=True)
    parser.add_argument(
        "--variant",
        choices=TGVF_VARIANTS,
        default="foveal_cross_merger",
    )
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1, help="Reserved for future tensor batching; current eval uses sharding for speed.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-key", choices=["image", "sample"], default="image")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--capture-layer", type=int, default=-1)
    parser.add_argument("--num-foveated-tokens", type=_parse_optional_positive_int, default=16)
    parser.add_argument("--spatial-merge-size", default="auto")
    parser.add_argument("--source-profile-filter", default=None)
    parser.add_argument("--evidence-type-filter", default=None)
    parser.add_argument("--debug-examples", type=int, default=10)
    parser.add_argument("--use-fvt-cache", action="store_true")
    parser.add_argument("--fvt-cache-dir", default=None)
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--no-progress", action="store_false", dest="progress")
    parser.add_argument("--progress-file", default=None)



def progress_iter(
    iterable: Any,
    *,
    desc: str,
    total: int | None = None,
    enabled: bool = True,
    progress_file: str | Path | None = None,
) -> Any:
    progress_path = Path(progress_file) if progress_file else None
    if progress_path is not None:
        _write_progress(progress_path, desc=desc, current=0, total=total, done=False)

    iterator = iterable
    if enabled:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(iterable, desc=desc, total=total, dynamic_ncols=True)
        except Exception:
            iterator = iterable

    for index, item in enumerate(iterator, start=1):
        yield item
        if progress_path is not None:
            done = total is not None and index >= total
            _write_progress(progress_path, desc=desc, current=index, total=total, done=done)


def _write_progress(
    path: Path,
    *,
    desc: str,
    current: int,
    total: int | None,
    done: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "desc": desc,
        "current": int(current),
        "total": None if total is None else int(total),
        "done": bool(done),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)

def _parse_optional_positive_int(value: str) -> int | None:
    if value.lower() in {"none", "null"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer or 'none'")
    return parsed


def validate_common_eval_args(args: argparse.Namespace) -> None:
    if (
        getattr(args, "variant", None) not in TGVF_DYNAMIC_NUM_FVT_VARIANTS
        and getattr(args, "num_foveated_tokens", 16) is None
    ):
        variants = ", ".join(TGVF_DYNAMIC_NUM_FVT_VARIANTS)
        raise ValueError(f"--num-foveated-tokens none is supported only with --variant in: {variants}")

def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _load_qwen2vl_with_retries(model_path: str, **kwargs: Any) -> Any:
    retries = max(1, int(os.environ.get("QWEN_LOAD_RETRIES", "3")))
    delay = max(0.0, float(os.environ.get("QWEN_LOAD_RETRY_DELAY", "10")))
    for attempt in range(1, retries + 1):
        try:
            return load_qwen2vl(model_path, **kwargs)
        except OSError as exc:
            if attempt >= retries:
                raise
            print(
                f"[eval] Qwen load failed on attempt {attempt}/{retries}: {exc}. Retrying in {delay:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)


def load_eval_samples(
    path: str | Path,
    *,
    max_samples: int | None = None,
    source_profile_filter: str | None = None,
    evidence_type_filter: str | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
    shard_key: str = "image",
) -> list[EvalSample]:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    samples = []
    with Path(path).open() as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            if source_profile_filter and record.get("source_profile") != source_profile_filter:
                continue
            if evidence_type_filter and record.get("evidence_type") != evidence_type_filter:
                continue
            required = ("image", "question", "target", "evidence_description")
            missing = [key for key in required if not record.get(key)]
            if missing:
                raise ValueError(f"{path}:{index + 1} missing required fields: {missing}")
            uid = record.get("uid") or f"{record.get('stable_image_uid') or record.get('image_id') or index}:{index}"
            samples.append(
                EvalSample(
                    uid=uid,
                    image=record["image"],
                    question=record["question"],
                    target=record["target"],
                    evidence_description=record["evidence_description"],
                    stable_image_uid=record.get("stable_image_uid"),
                    image_id=record.get("image_id"),
                    source_dataset=record.get("source_dataset"),
                    source_profile=record.get("source_profile"),
                    short_answer=record.get("short_answer"),
                    evidence_type=record.get("evidence_type"),
                    locality=record.get("locality"),
                    answer_type=record.get("answer_type"),
                    visual_difficulty=record.get("visual_difficulty"),
                    visibility=record.get("visibility"),
                    confidence=record.get("confidence"),
                    raw=record,
                )
            )
            if max_samples is not None and len(samples) >= max_samples:
                break
    if num_shards > 1:
        samples = shard_eval_samples(
            samples,
            num_shards=num_shards,
            shard_index=shard_index,
            shard_key=shard_key,
        )
    return samples



def shard_eval_samples(
    samples: list[EvalSample],
    *,
    num_shards: int,
    shard_index: int,
    shard_key: str = "image",
) -> list[EvalSample]:
    selected = []
    for index, sample in enumerate(samples):
        if shard_key == "sample":
            keep = index % num_shards == shard_index
        else:
            digest = int(hashlib.sha1(sample.group_id.encode()).hexdigest(), 16)
            keep = digest % num_shards == shard_index
        if keep:
            selected.append(sample)
    return selected

def make_output_dir(base: str | Path, *, prefix: str, overwrite: bool = False) -> Path:
    path = Path(base)
    if path.exists() and any(path.iterdir()) and not overwrite:
        run_id = datetime.now(timezone.utc).strftime(f"{prefix}_%Y%m%d_%H%M%S")
        path = path / run_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "examples").mkdir(exist_ok=True)
    return path


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return to_jsonable(asdict(value))
    return value


def load_qwen_and_tgvf(args: argparse.Namespace) -> tuple[Any, Any, torch.nn.Module, torch.device, dict[str, Any]]:
    validate_common_eval_args(args)
    device = resolve_device(args.device)
    loaded = _load_qwen2vl_with_retries(
        args.model_path,
        torch_dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        device_map={"": device} if str(device).startswith("cuda") else None,
        trust_remote_code=False,
    )
    model = loaded.model
    processor = loaded.processor
    if args.processor_path:
        processor = AutoProcessor.from_pretrained(args.processor_path, trust_remote_code=False)
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
    freeze_qwen2vl(model)
    d_lm, d_v, inferred_merge = infer_tgvf_dims(model)
    spatial_merge_size = inferred_merge if args.spatial_merge_size == "auto" else int(args.spatial_merge_size)
    module = build_tgvf_module(
        variant=args.variant,
        d_lm=d_lm,
        d_v=d_v,
        num_foveated_tokens=args.num_foveated_tokens,
        spatial_merge_size=spatial_merge_size,
    ).to(device=device, dtype=next(model.parameters()).dtype)
    checkpoint = load_tgvf_module_checkpoint(module, args.tgvf_checkpoint, strict=True)
    module.eval()
    return model, processor, module, device, {
        "d_lm": d_lm,
        "d_v": d_v,
        "spatial_merge_size": spatial_merge_size,
        "checkpoint_global_step": checkpoint.get("global_step"),
        "checkpoint_optimizer_step": checkpoint.get("optimizer_step"),
    }


@torch.no_grad()
def compute_eval_item(
    *,
    model: Any,
    processor: Any,
    foveal_module: torch.nn.Module,
    sample: EvalSample,
    device: torch.device | str,
    capture_layer: int = -1,
) -> EvalFeatureCacheItem:
    features = collect_training_features(
        model=model,
        processor=processor,
        sample=sample.to_teacher_sample(),
        device=device,
        hidden_state_index=capture_layer,
    )
    output = foveal_module(
        target_hidden_states=features.target_hidden_states.to(device),
        pre_merge_visual_tokens=features.pre_merge_visual_tokens.to(device),
        metadata={"target": sample.target},
    )
    output = finalize_tgvf_output_with_frozen_qwen_merger(model, output)
    d = output.foveated_visual_tokens.detach()
    return EvalFeatureCacheItem(
        sample=sample,
        target_hidden_states=features.target_hidden_states.detach().cpu(),
        pre_merge_visual_tokens=features.pre_merge_visual_tokens.detach().cpu(),
        merged_visual_tokens=features.merged_visual_tokens.detach().cpu(),
        foveated_visual_tokens=d.detach().cpu(),
        image_grid_thw=(
            features.image_grid_thw.detach().cpu()
            if output.debug_metadata.get("tgvf_version") == "v2"
            and features.image_grid_thw is not None
            else None
        ),
        capture_text=features.capture.generated_text,
        shapes={
            "H_q": list(features.target_hidden_states.shape),
            "V_pre": list(features.pre_merge_visual_tokens.shape),
            "V_merge": list(features.merged_visual_tokens.shape),
            "D": list(d.shape),
        },
    )



@torch.no_grad()
def compute_or_load_eval_item(
    *,
    model: Any,
    processor: Any,
    foveal_module: torch.nn.Module,
    sample: EvalSample,
    device: torch.device | str,
    capture_layer: int = -1,
    cache_dir: str | Path | None = None,
    checkpoint_path: str | None = None,
    variant: str | None = None,
    use_cache: bool = False,
) -> EvalFeatureCacheItem:
    if not use_cache:
        return compute_eval_item(
            model=model,
            processor=processor,
            foveal_module=foveal_module,
            sample=sample,
            device=device,
            capture_layer=capture_layer,
        )
    if cache_dir is None:
        raise ValueError("--use-fvt-cache requires --fvt-cache-dir")
    cache_path = _fvt_cache_path(
        cache_dir=Path(cache_dir),
        sample=sample,
        checkpoint_path=checkpoint_path or "",
        variant=variant or "",
        capture_layer=capture_layer,
    )
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        return EvalFeatureCacheItem(
            sample=sample,
            target_hidden_states=payload["target_hidden_states"],
            pre_merge_visual_tokens=payload["pre_merge_visual_tokens"],
            merged_visual_tokens=payload["merged_visual_tokens"],
            foveated_visual_tokens=payload["foveated_visual_tokens"],
            image_grid_thw=payload.get("image_grid_thw"),
            capture_text=payload.get("capture_text", ""),
            shapes=payload["shapes"],
        )
    item = compute_eval_item(
        model=model,
        processor=processor,
        foveal_module=foveal_module,
        sample=sample,
        device=device,
        capture_layer=capture_layer,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "target_hidden_states": item.target_hidden_states.detach().cpu(),
            "pre_merge_visual_tokens": item.pre_merge_visual_tokens.detach().cpu(),
            "merged_visual_tokens": item.merged_visual_tokens.detach().cpu(),
            "foveated_visual_tokens": item.foveated_visual_tokens.detach().cpu(),
            "image_grid_thw": (
                None if item.image_grid_thw is None else item.image_grid_thw.detach().cpu()
            ),
            "capture_text": item.capture_text,
            "shapes": item.shapes,
            "metadata": {
                "uid": sample.uid,
                "target": sample.target,
                "image": sample.image,
                "checkpoint_path": checkpoint_path,
                "variant": variant,
                "capture_layer": capture_layer,
            },
        },
        cache_path,
    )
    return item


def _fvt_cache_path(
    *,
    cache_dir: Path,
    sample: EvalSample,
    checkpoint_path: str,
    variant: str,
    capture_layer: int,
) -> Path:
    key = "|".join(
        [
            checkpoint_path,
            variant,
            str(capture_layer),
            sample.uid,
            sample.target,
            sample.image,
        ]
    )
    return cache_dir / f"{short_hash(key)}.pt"

def compute_readout_nll(
    *,
    model: Any,
    tokenizer_or_processor: Any,
    target_text: str,
    evidence_description: str,
    foveated_visual_tokens: torch.Tensor | None,
    device: torch.device | str,
    image_grid_thw: torch.Tensor | None = None,
) -> dict[str, Any]:
    if foveated_visual_tokens is None:
        readout_inputs = prepare_target_only_readout_inputs(
            model=model,
            tokenizer_or_processor=tokenizer_or_processor,
            target=target_text,
            evidence_description=evidence_description,
            device=device,
        )
    else:
        readout_inputs = prepare_readout_inputs(
            model=model,
            tokenizer_or_processor=tokenizer_or_processor,
            target=target_text,
            evidence_description=evidence_description,
            foveated_visual_tokens=foveated_visual_tokens.to(device),
            image_grid_thw=image_grid_thw,
            device=device,
        )
    loss, log_likelihood = compute_readout_lm_loss(model=model, readout_inputs=readout_inputs)
    token_count = int(readout_inputs["answer_token_count"])
    total_nll = -float(log_likelihood.detach().cpu())
    return {
        "avg_nll": float(loss.detach().cpu()),
        "total_nll": total_nll,
        "log_likelihood": float(log_likelihood.detach().cpu()),
        "answer_token_count": token_count,
    }


def prepare_target_only_readout_inputs(
    *,
    model: Any,
    tokenizer_or_processor: Any,
    target: str,
    evidence_description: str,
    device: torch.device | str,
) -> dict[str, Any]:
    tokenizer = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)
    before, after = readout_prompt_parts(target)
    prefix_ids = tokenizer.encode(before + after, add_special_tokens=False)
    answer_ids = tokenizer.encode(evidence_description, add_special_tokens=False)
    input_ids = torch.tensor([prefix_ids + answer_ids], dtype=torch.long, device=device)
    embeds = model.get_input_embeddings()(input_ids)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    labels[:, len(prefix_ids) :] = input_ids[:, len(prefix_ids) :]
    return {
        "input_ids": input_ids,
        "inputs_embeds": embeds,
        "labels": labels,
        "attention_mask": torch.ones_like(input_ids),
        "answer_token_count": len(answer_ids),
        "answer_start": len(prefix_ids),
    }


def random_d_like(d: torch.Tensor, *, reference: torch.Tensor | None = None, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    base = reference.detach().float().cpu() if reference is not None and reference.numel() else d.detach().float().cpu()
    mean = base.mean()
    std = base.std(unbiased=False).clamp_min(1e-6)
    return torch.randn(d.shape, generator=generator, dtype=torch.float32) * std + mean


def group_indices_by_image(items: list[EvalFeatureCacheItem] | list[EvalSample]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        sample = item.sample if isinstance(item, EvalFeatureCacheItem) else item
        groups.setdefault(sample.group_id, []).append(index)
    return groups


def different_image_index(items: list[EvalFeatureCacheItem], index: int) -> int | None:
    source_group = items[index].sample.group_id
    for other_index, item in enumerate(items):
        if other_index != index and item.sample.group_id != source_group:
            return other_index
    return None


def same_image_wrong_index(groups: dict[str, list[int]], item: EvalFeatureCacheItem, index: int) -> int | None:
    indices = groups.get(item.sample.group_id, [])
    for other in indices:
        if other != index:
            return other
    return None


def summarize_config(args: argparse.Namespace, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    config = vars(args).copy()
    config.update(extra or {})
    config["timestamp"] = datetime.now(timezone.utc).isoformat()
    return config


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def generate_direct_qwen_answer(
    *,
    model: Any,
    processor: Any,
    image: str,
    question: str,
    device: torch.device | str,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> str:
    from revisit_vlm.tgvf_capture import build_qwen2vl_tgvf_inputs

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    inputs = build_qwen2vl_tgvf_inputs(processor, image=image, question=question, messages=messages)
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=eos_token_id)
    prompt_len = inputs["input_ids"].shape[-1]
    generated = outputs[0, prompt_len:].detach().cpu().tolist()
    return processor.tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)


@torch.no_grad()
def run_forced_target_end2end(
    *,
    model: Any,
    processor: Any,
    foveal_module: torch.nn.Module,
    sample: EvalSample,
    device: torch.device | str,
    capture_layer: int,
    answer_max_new_tokens: int,
    condition: str = "correct",
    replacement_d: torch.Tensor | None = None,
    replacement_image_grid_thw: torch.Tensor | None = None,
) -> dict[str, Any]:
    forced_text = f"{FOVEATE_START}{sample.target}{FOVEATE_END}"
    forced_ids = processor.tokenizer.encode(forced_text, add_special_tokens=False)
    wrapped = ForcedFoveationWrapper(model, forced_ids)
    with Qwen2VLPreMergeVisualHook(model) as pre_hook:
        capture = capture_tgvf_single_pass(
            wrapped,
            processor,
            image=sample.image,
            question=sample.question,
            max_new_tokens=len(forced_ids) + 8,
            device=device,
            hidden_state_index=capture_layer,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    if not capture.capture_found:
        return {"answer": "", "capture_found": False, "stop_reason": capture.stop_reason}

    d = None
    if condition != "no_D":
        if replacement_d is not None:
            d = replacement_d.to(device)
        else:
            if pre_hook.pre_merge_visual_tokens is None:
                raise RuntimeError("Pre-merge visual tokens were not captured")
            fvt_output = foveal_module(
                target_hidden_states=capture.target_hidden_states.to(device),
                pre_merge_visual_tokens=pre_hook.pre_merge_visual_tokens.to(device),
                metadata={"target": sample.target},
            )
            fvt_output = finalize_tgvf_output_with_frozen_qwen_merger(model, fvt_output)
            d = fvt_output.foveated_visual_tokens
        image_grid_thw = (
            replacement_image_grid_thw
            if replacement_d is not None
            else (
                capture.image_grid_thw
                if fvt_output.debug_metadata.get("tgvf_version") == "v2"
                else None
            )
        )
        append_result = append_fvt_result_and_open_answer_turn(
            model=model,
            tokenizer_or_processor=processor,
            generation_state=capture,
            foveated_visual_tokens=d,
            target_text=sample.target,
            image_grid_thw=image_grid_thw,
            device=device,
        )
        continuation = continue_generation_from_state(
            model=model,
            tokenizer_or_processor=processor,
            generation_state=append_result,
            max_new_tokens=answer_max_new_tokens,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
        return {
            "answer": continuation.generated_text,
            "capture_found": True,
            "target_text": capture.target_text,
            "foveation_request": capture.generated_text,
            "D_shape": list(d.shape),
            "stop_reason": continuation.stop_reason,
            "second_full_forward_used": False,
        }

    continuation = continue_generation_from_state(
        model=model,
        tokenizer_or_processor=processor,
        generation_state=capture,
        max_new_tokens=answer_max_new_tokens,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    return {
        "answer": continuation.generated_text,
        "capture_found": True,
        "target_text": capture.target_text,
        "foveation_request": capture.generated_text,
        "D_shape": None,
        "stop_reason": continuation.stop_reason,
        "second_full_forward_used": False,
    }


def save_summary(path: str | Path, title: str, lines: list[str]) -> None:
    Path(path).write_text("# " + title + "\n\n" + "\n".join(lines) + "\n")
