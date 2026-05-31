from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from eval.common import (
    make_output_dir,
    progress_iter,
    random_d_like,
    resolve_device,
    save_summary,
    set_seed,
    short_hash,
    summarize_config,
    write_json,
    write_jsonl,
)
from revisit_vlm.qwen3_vl_tgvf import (
    EVIDENCE_END,
    EVIDENCE_START,
    NEED_LOCAL_EVIDENCE,
    Qwen3FocusCapture,
    Qwen3SourceVisualGeometry,
    _compute_qwen3_position_ids_for_sequence,
    _encode_text,
    load_qwen3_vl,
)
from revisit_vlm.tgvf_foveal import finalize_tgvf_output_with_frozen_qwen_merger
from revisit_vlm.tgvf_training import (
    IGNORE_INDEX,
    TGVF_DYNAMIC_NUM_FVT_VARIANTS,
    TGVF_VARIANTS,
    build_tgvf_module,
    load_tgvf_module_checkpoint,
)
from revisit_vlm.tgvf_v3_stage1 import (
    PositionMode,
    TGVFv3Stage1Dataset,
    TGVFv3Stage1Sample,
    build_weak_strict_attention_mask,
    collect_v3_stage1_features,
    compute_v3_stage1_lm_loss,
    freeze_qwen_backbone,
    infer_qwen3_stage1_dims,
    prepare_v3_stage1_readout_inputs,
    summarize_weak_strict_mask,
)


@dataclass
class V3EvalFeatureCacheItem:
    sample: TGVFv3Stage1Sample
    uid: str
    target_hidden_states: torch.Tensor
    pre_merge_visual_tokens: torch.Tensor
    merged_visual_tokens: torch.Tensor
    foveated_visual_tokens: torch.Tensor
    capture: Qwen3FocusCapture
    shapes: dict[str, list[int]]
    readout_metadata: dict[str, Any]


def add_v3_common_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Thinking")
    parser.add_argument("--processor-id", default=None)
    parser.add_argument("--tgvf-checkpoint", required=True)
    parser.add_argument("--variant", choices=TGVF_VARIANTS, default="tgvf_v2_bidirectional")
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1, help="Reserved; v3 eval is sample-wise.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-key", choices=["image", "sample"], default="image")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--capture-layer", type=int, default=-1)
    parser.add_argument("--num-foveated-tokens", type=_parse_optional_positive_int, default=None)
    parser.add_argument("--max-image-resolution", type=int, default=512)
    parser.add_argument(
        "--fvt-position-mode",
        choices=["native_source_grid", "inherit_source_visual_positions"],
        default="native_source_grid",
    )
    parser.add_argument("--mask-original-image-after-tgvf", action="store_true", default=True)
    parser.add_argument("--no-mask-original-image-after-tgvf", action="store_false", dest="mask_original_image_after_tgvf")
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--source-profile-filter", default=None)
    parser.add_argument("--evidence-type-filter", default=None)
    parser.add_argument("--debug-examples", type=int, default=10)
    parser.add_argument("--use-fvt-cache", action="store_true")
    parser.add_argument("--fvt-cache-dir", default=None)
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--no-progress", action="store_false", dest="progress")
    parser.add_argument("--progress-file", default=None)


def validate_v3_common_eval_args(args: argparse.Namespace) -> None:
    if (
        getattr(args, "variant", None) not in TGVF_DYNAMIC_NUM_FVT_VARIANTS
        and getattr(args, "num_foveated_tokens", None) is None
    ):
        variants = ", ".join(TGVF_DYNAMIC_NUM_FVT_VARIANTS)
        raise ValueError(f"--num-foveated-tokens none is supported only with --variant in: {variants}")
    if getattr(args, "num_shards", 1) < 1:
        raise ValueError("--num-shards must be >= 1")
    if getattr(args, "shard_index", 0) < 0 or getattr(args, "shard_index", 0) >= getattr(args, "num_shards", 1):
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")


def load_v3_eval_samples(
    path: str | Path,
    *,
    max_samples: int | None = None,
    min_confidence: float | None = None,
    source_profile_filter: str | None = None,
    evidence_type_filter: str | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
    shard_key: str = "image",
) -> list[TGVFv3Stage1Sample]:
    dataset = TGVFv3Stage1Dataset(path, focus_only=True, min_confidence=min_confidence)
    samples = []
    for sample in dataset.samples:
        if source_profile_filter and sample.source_profile != source_profile_filter:
            continue
        if evidence_type_filter and sample.evidence_type != evidence_type_filter:
            continue
        samples.append(sample)
        if max_samples is not None and len(samples) >= max_samples:
            break
    if num_shards > 1:
        samples = shard_v3_eval_samples(
            samples,
            num_shards=num_shards,
            shard_index=shard_index,
            shard_key=shard_key,
        )
    return samples


def shard_v3_eval_samples(
    samples: list[TGVFv3Stage1Sample],
    *,
    num_shards: int,
    shard_index: int,
    shard_key: str = "image",
) -> list[TGVFv3Stage1Sample]:
    selected = []
    for index, sample in enumerate(samples):
        if shard_key == "sample":
            keep = index % num_shards == shard_index
        else:
            digest = int(hashlib.sha1(v3_group_id(sample).encode()).hexdigest(), 16)
            keep = digest % num_shards == shard_index
        if keep:
            selected.append(sample)
    return selected


def load_qwen3_and_tgvf(
    args: argparse.Namespace,
    *,
    example_sample: TGVFv3Stage1Sample,
) -> tuple[Any, Any, torch.nn.Module, torch.device, dict[str, Any]]:
    validate_v3_common_eval_args(args)
    device = resolve_device(args.device)
    loaded = load_qwen3_vl(
        args.model_id,
        processor_id=args.processor_id,
        dtype=args.dtype,
        device_map=_resolve_device_map(args.device_map),
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    model = loaded.model
    processor = loaded.processor
    freeze_qwen_backbone(model)
    dims = infer_qwen3_stage1_dims(
        model=model,
        processor=processor,
        sample=example_sample,
        device=device,
        max_image_resolution=args.max_image_resolution,
    )
    module = build_tgvf_module(
        variant=args.variant,
        d_lm=dims["d_lm"],
        d_v=dims["d_v"],
        num_foveated_tokens=args.num_foveated_tokens,
        spatial_merge_size=dims["spatial_merge_size"],
    ).to(device=device, dtype=next(model.parameters()).dtype)
    checkpoint = load_tgvf_module_checkpoint(module, args.tgvf_checkpoint, strict=True)
    module.eval()
    return model, processor, module, device, {
        "model_id": args.model_id,
        "processor_id": args.processor_id or args.model_id,
        "d_lm": dims["d_lm"],
        "d_v": dims["d_v"],
        "spatial_merge_size": dims["spatial_merge_size"],
        "checkpoint_global_step": checkpoint.get("global_step"),
        "checkpoint_optimizer_step": checkpoint.get("optimizer_step"),
        "stage": "tgvf_v3_stage1_eval",
        "second_full_forward_used": False,
    }


@torch.no_grad()
def compute_v3_eval_item(
    *,
    model: Any,
    processor: Any,
    foveal_module: torch.nn.Module,
    sample: TGVFv3Stage1Sample,
    device: torch.device | str,
    capture_layer: int = -1,
    max_image_resolution: int | None = 512,
    position_mode: PositionMode = "native_source_grid",
    mask_original_image_after_tgvf: bool = True,
) -> V3EvalFeatureCacheItem:
    features = collect_v3_stage1_features(
        model=model,
        processor=processor,
        sample=sample,
        device=device,
        hidden_state_index=capture_layer,
        max_image_resolution=max_image_resolution,
    )
    output = foveal_module(
        target_hidden_states=features.target_hidden_states.to(device),
        pre_merge_visual_tokens=features.pre_merge_visual_tokens.to(device),
        metadata={
            "target": sample.target,
            "stage": "tgvf_v3_eval",
            "evidence_state": NEED_LOCAL_EVIDENCE,
        },
    )
    output = finalize_tgvf_output_with_frozen_qwen_merger(model, output)
    d = output.foveated_visual_tokens.detach()
    readout_inputs = prepare_v3_stage1_readout_inputs(
        model=model,
        tokenizer_or_processor=processor,
        capture=features.capture,
        evidence_description=sample.evidence_description,
        foveated_visual_tokens=d,
        device=device,
        mask_original_image_after_tgvf=mask_original_image_after_tgvf,
        position_mode=position_mode,
    )
    return V3EvalFeatureCacheItem(
        sample=sample,
        uid=v3_sample_uid(sample),
        target_hidden_states=features.target_hidden_states.detach().cpu(),
        pre_merge_visual_tokens=features.pre_merge_visual_tokens.detach().cpu(),
        merged_visual_tokens=features.merged_visual_tokens.detach().cpu(),
        foveated_visual_tokens=d.detach().cpu(),
        capture=_cpu_capture_for_cache(features.capture),
        shapes={
            "H_q": list(features.target_hidden_states.shape),
            "V_pre": list(features.pre_merge_visual_tokens.shape),
            "V_merge": list(features.merged_visual_tokens.shape),
            "D": list(d.shape),
        },
        readout_metadata={
            "mask_mode": readout_inputs.get("mask_mode"),
            "position_mode": readout_inputs.get("position_mode"),
            "position_ids_source": readout_inputs.get("position_ids_source"),
            "original_image_token_count": readout_inputs.get("original_image_token_count"),
            "blocked_original_image_keys_for_post_tgvf": readout_inputs.get(
                "blocked_original_image_keys_for_post_tgvf"
            ),
            "pre_tgvf_queries_keep_original_image_keys": readout_inputs.get(
                "pre_tgvf_queries_keep_original_image_keys"
            ),
            "second_full_forward_used": False,
        },
    )


@torch.no_grad()
def compute_or_load_v3_eval_item(
    *,
    model: Any,
    processor: Any,
    foveal_module: torch.nn.Module,
    sample: TGVFv3Stage1Sample,
    device: torch.device | str,
    capture_layer: int = -1,
    max_image_resolution: int | None = 512,
    position_mode: PositionMode = "native_source_grid",
    mask_original_image_after_tgvf: bool = True,
    cache_dir: str | Path | None = None,
    checkpoint_path: str | None = None,
    variant: str | None = None,
    use_cache: bool = False,
) -> V3EvalFeatureCacheItem:
    if not use_cache:
        return compute_v3_eval_item(
            model=model,
            processor=processor,
            foveal_module=foveal_module,
            sample=sample,
            device=device,
            capture_layer=capture_layer,
            max_image_resolution=max_image_resolution,
            position_mode=position_mode,
            mask_original_image_after_tgvf=mask_original_image_after_tgvf,
        )
    if cache_dir is None:
        raise ValueError("--use-fvt-cache requires --fvt-cache-dir")
    cache_path = _v3_fvt_cache_path(
        cache_dir=Path(cache_dir),
        sample=sample,
        checkpoint_path=checkpoint_path or "",
        variant=variant or "",
        capture_layer=capture_layer,
        max_image_resolution=max_image_resolution,
        position_mode=position_mode,
    )
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        return V3EvalFeatureCacheItem(
            sample=sample,
            uid=payload.get("uid", v3_sample_uid(sample)),
            target_hidden_states=payload["target_hidden_states"],
            pre_merge_visual_tokens=payload["pre_merge_visual_tokens"],
            merged_visual_tokens=payload["merged_visual_tokens"],
            foveated_visual_tokens=payload["foveated_visual_tokens"],
            capture=_capture_from_payload(payload["capture"]),
            shapes=payload["shapes"],
            readout_metadata=payload.get("readout_metadata", {}),
        )
    item = compute_v3_eval_item(
        model=model,
        processor=processor,
        foveal_module=foveal_module,
        sample=sample,
        device=device,
        capture_layer=capture_layer,
        max_image_resolution=max_image_resolution,
        position_mode=position_mode,
        mask_original_image_after_tgvf=mask_original_image_after_tgvf,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "uid": item.uid,
            "target_hidden_states": item.target_hidden_states.detach().cpu(),
            "pre_merge_visual_tokens": item.pre_merge_visual_tokens.detach().cpu(),
            "merged_visual_tokens": item.merged_visual_tokens.detach().cpu(),
            "foveated_visual_tokens": item.foveated_visual_tokens.detach().cpu(),
            "capture": _capture_payload(item.capture),
            "shapes": item.shapes,
            "readout_metadata": item.readout_metadata,
            "metadata": {
                "uid": item.uid,
                "target": sample.target,
                "image": sample.image,
                "checkpoint_path": checkpoint_path,
                "variant": variant,
                "capture_layer": capture_layer,
                "max_image_resolution": max_image_resolution,
                "position_mode": position_mode,
            },
        },
        cache_path,
    )
    return item


def compute_v3_readout_nll(
    *,
    model: Any,
    tokenizer_or_processor: Any,
    capture: Qwen3FocusCapture,
    evidence_description: str,
    foveated_visual_tokens: torch.Tensor | None,
    device: torch.device | str,
    mask_original_image_after_tgvf: bool = True,
    position_mode: PositionMode = "native_source_grid",
) -> dict[str, Any]:
    if foveated_visual_tokens is None:
        readout_inputs = prepare_v3_target_only_readout_inputs(
            model=model,
            tokenizer_or_processor=tokenizer_or_processor,
            capture=capture,
            evidence_description=evidence_description,
            device=device,
            mask_original_image_after_tgvf=mask_original_image_after_tgvf,
        )
    else:
        readout_inputs = prepare_v3_stage1_readout_inputs(
            model=model,
            tokenizer_or_processor=tokenizer_or_processor,
            capture=capture,
            evidence_description=evidence_description,
            foveated_visual_tokens=foveated_visual_tokens.to(device),
            device=device,
            mask_original_image_after_tgvf=mask_original_image_after_tgvf,
            position_mode=position_mode,
        )
    loss, log_likelihood = compute_v3_stage1_lm_loss(model=model, readout_inputs=readout_inputs)
    token_count = int(readout_inputs["answer_token_count"])
    total_nll = -float(log_likelihood.detach().cpu())
    return {
        "avg_nll": float(loss.detach().cpu()),
        "total_nll": total_nll,
        "log_likelihood": float(log_likelihood.detach().cpu()),
        "answer_token_count": token_count,
        "mask_mode": readout_inputs.get("mask_mode"),
        "position_mode": readout_inputs.get("position_mode"),
        "position_ids_source": readout_inputs.get("position_ids_source"),
    }


def prepare_v3_target_only_readout_inputs(
    *,
    model: Any,
    tokenizer_or_processor: Any,
    capture: Qwen3FocusCapture,
    evidence_description: str,
    device: torch.device | str,
    mask_original_image_after_tgvf: bool = True,
) -> dict[str, Any]:
    if not capture.capture_found:
        raise ValueError("capture must contain a valid focus span")
    if capture.input_ids is None:
        raise ValueError("capture input_ids are required for target-only v3 readout")
    if capture.source_visual_geometry is None:
        raise ValueError("capture is missing source visual geometry")
    source_geometry = capture.source_visual_geometry
    original_image_indices = source_geometry.source_visual_token_indices
    if original_image_indices is None:
        raise RuntimeError("source visual token indices are unavailable")

    tokenizer = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)
    base_input_ids = capture.input_ids.to(device)
    base_len = int(base_input_ids.shape[-1])
    evidence_prefix_ids = _encode_text(tokenizer, f"\n{EVIDENCE_START}", device).view(1, -1)
    evidence_ids = _encode_text(tokenizer, f"{evidence_description}{EVIDENCE_END}", device).view(1, -1)
    input_ids = torch.cat([base_input_ids, evidence_prefix_ids, evidence_ids], dim=-1)
    evidence_start = base_len + int(evidence_prefix_ids.shape[-1])

    embeds = model.get_input_embeddings()(input_ids).detach()
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    labels[:, evidence_start:] = input_ids[:, evidence_start:]
    attention_mask_2d = torch.ones_like(input_ids)
    mm_token_type_ids = _original_image_mm_token_type_ids(
        model=model,
        input_ids=input_ids,
        source_geometry=source_geometry,
        device=device,
    )
    image_grid_thw = _stage1_image_grid_thw_for_eval(capture, source_geometry, device)
    position_ids = _compute_qwen3_position_ids_for_sequence(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask_2d,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None if capture.video_grid_thw is None else capture.video_grid_thw.to(device),
        mm_token_type_ids=mm_token_type_ids,
    )
    if position_ids is None:
        raise RuntimeError("Qwen3 position id computation is unavailable")
    original_image_indices = original_image_indices.to(device=device, dtype=torch.long)
    if mask_original_image_after_tgvf:
        attention_mask = build_weak_strict_attention_mask(
            attention_mask_2d=attention_mask_2d,
            original_image_token_indices=original_image_indices,
            block_query_start=base_len,
            dtype=embeds.dtype,
        )
        mask_mode = "weak_strict_original_image_keys_4d_target_only"
    else:
        attention_mask = attention_mask_2d
        mask_mode = "standard_2d_causal_target_only"
    mask_summary = summarize_weak_strict_mask(
        attention_mask=attention_mask,
        original_image_token_indices=original_image_indices,
        block_query_start=base_len,
    )
    return {
        "input_ids": input_ids,
        "inputs_embeds": embeds,
        "labels": labels,
        "attention_mask": attention_mask,
        "attention_mask_2d": attention_mask_2d,
        "position_ids": position_ids,
        "image_grid_thw": image_grid_thw,
        "mm_token_type_ids": mm_token_type_ids,
        "evidence_start": evidence_start,
        "answer_token_count": int(evidence_ids.shape[-1]),
        "mask_mode": mask_mode,
        "position_mode": "target_only_no_D",
        "position_ids_source": "qwen3_native_source_grid_full_trajectory",
        "original_image_token_count": int(original_image_indices.numel()),
        "original_image_token_span_detection": "source_image_token_id_scan",
        "block_query_start": base_len,
        "blocked_original_image_keys_for_post_tgvf": mask_summary[
            "blocked_original_image_keys_for_post_tgvf"
        ],
        "pre_tgvf_queries_keep_original_image_keys": mask_summary[
            "pre_tgvf_queries_keep_original_image_keys"
        ],
        "mask_summary": mask_summary,
    }


def can_score_fvt_for_item(item: V3EvalFeatureCacheItem, d: torch.Tensor) -> bool:
    geometry = item.capture.source_visual_geometry
    if geometry is None:
        return False
    return int(d.shape[0]) == int(geometry.source_visual_token_count)


def group_indices_by_v3_image(
    items: list[V3EvalFeatureCacheItem] | list[TGVFv3Stage1Sample],
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        sample = item.sample if isinstance(item, V3EvalFeatureCacheItem) else item
        groups.setdefault(v3_group_id(sample), []).append(index)
    return groups


def same_image_wrong_index_v3(
    groups: dict[str, list[int]],
    item: V3EvalFeatureCacheItem,
    index: int,
) -> int | None:
    indices = groups.get(v3_group_id(item.sample), [])
    for other in indices:
        if other != index:
            return other
    return None


def different_image_index_v3(items: list[V3EvalFeatureCacheItem], index: int) -> int | None:
    source_group = v3_group_id(items[index].sample)
    for other_index, item in enumerate(items):
        if other_index != index and v3_group_id(item.sample) != source_group:
            return other_index
    return None


def v3_group_id(sample: TGVFv3Stage1Sample) -> str:
    return str(sample.image_id or sample.image)


def v3_sample_uid(sample: TGVFv3Stage1Sample) -> str:
    metadata_uid = sample.metadata.get("uid") if sample.metadata else None
    if metadata_uid:
        return str(metadata_uid)
    key = "|".join([v3_group_id(sample), sample.question, sample.target, sample.evidence_description])
    return f"{v3_group_id(sample)}:{short_hash(key)}"


def sample_metadata_row(sample: TGVFv3Stage1Sample) -> dict[str, Any]:
    return {
        "uid": v3_sample_uid(sample),
        "stable_image_uid": v3_group_id(sample),
        "image": sample.image,
        "image_id": sample.image_id,
        "target": sample.target,
        "target_style": sample.target_style,
        "target_cues": sample.target_cues,
        "evidence_description": sample.evidence_description,
        "evidence_type": sample.evidence_type,
        "source_dataset": sample.source_dataset,
        "source_profile": sample.source_profile,
        "answer_type": sample.metadata.get("answer_type") if sample.metadata else None,
        "answer_format": sample.answer_format,
        "visual_difficulty": sample.metadata.get("visual_difficulty") if sample.metadata else None,
        "visibility": sample.metadata.get("visibility") if sample.metadata else None,
        "confidence": sample.metadata.get("confidence") if sample.metadata else None,
    }


def _parse_optional_positive_int(value: str) -> int | None:
    if value.lower() in {"none", "null"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer or 'none'")
    return parsed


def _resolve_device_map(device_map: str | None) -> str | dict[str, str] | None:
    if device_map is None or device_map == "" or device_map.lower() == "none":
        return None
    if device_map == "auto":
        return "auto"
    if device_map.startswith("cuda") or device_map == "cpu":
        return {"": device_map}
    return device_map


def _v3_fvt_cache_path(
    *,
    cache_dir: Path,
    sample: TGVFv3Stage1Sample,
    checkpoint_path: str,
    variant: str,
    capture_layer: int,
    max_image_resolution: int | None,
    position_mode: str,
) -> Path:
    key = "|".join(
        [
            checkpoint_path,
            variant,
            str(capture_layer),
            str(max_image_resolution),
            position_mode,
            v3_sample_uid(sample),
            sample.target,
            sample.image,
        ]
    )
    return cache_dir / f"{short_hash(key)}.pt"


def _cpu_capture_for_cache(capture: Qwen3FocusCapture) -> Qwen3FocusCapture:
    return Qwen3FocusCapture(
        target_text=capture.target_text,
        target_token_ids=list(capture.target_token_ids),
        target_hidden_states=capture.target_hidden_states.detach().cpu(),
        generated_ids=list(capture.generated_ids),
        generated_text=capture.generated_text,
        generated_hidden_states=torch.empty(0),
        past_key_values=None,
        attention_mask=_detach_cpu(capture.attention_mask),
        cache_position=None,
        input_ids=_detach_cpu(capture.input_ids),
        last_logits=None,
        model_kwargs={},
        image_grid_thw=_detach_cpu(capture.image_grid_thw),
        video_grid_thw=_detach_cpu(capture.video_grid_thw),
        source_visual_geometry=capture.source_visual_geometry,
        target_token_start=capture.target_token_start,
        target_token_end=capture.target_token_end,
        stop_reason=capture.stop_reason,
        capture_found=capture.capture_found,
        second_full_forward_used=False,
        malformed=capture.malformed,
        errors=list(capture.errors),
    )


def _capture_payload(capture: Qwen3FocusCapture) -> dict[str, Any]:
    return {
        "target_text": capture.target_text,
        "target_token_ids": list(capture.target_token_ids),
        "target_hidden_states": capture.target_hidden_states.detach().cpu(),
        "generated_ids": list(capture.generated_ids),
        "generated_text": capture.generated_text,
        "attention_mask": _detach_cpu(capture.attention_mask),
        "input_ids": _detach_cpu(capture.input_ids),
        "image_grid_thw": _detach_cpu(capture.image_grid_thw),
        "video_grid_thw": _detach_cpu(capture.video_grid_thw),
        "source_visual_geometry": (
            None if capture.source_visual_geometry is None else asdict(capture.source_visual_geometry)
        ),
        "target_token_start": capture.target_token_start,
        "target_token_end": capture.target_token_end,
        "stop_reason": capture.stop_reason,
        "capture_found": capture.capture_found,
        "malformed": capture.malformed,
        "errors": list(capture.errors),
    }


def _capture_from_payload(payload: dict[str, Any]) -> Qwen3FocusCapture:
    geometry_payload = payload.get("source_visual_geometry")
    geometry = None
    if geometry_payload is not None:
        geometry = Qwen3SourceVisualGeometry(**geometry_payload)
    return Qwen3FocusCapture(
        target_text=payload.get("target_text", ""),
        target_token_ids=list(payload.get("target_token_ids") or []),
        target_hidden_states=payload["target_hidden_states"],
        generated_ids=list(payload.get("generated_ids") or []),
        generated_text=payload.get("generated_text", ""),
        generated_hidden_states=torch.empty(0),
        past_key_values=None,
        attention_mask=payload.get("attention_mask"),
        cache_position=None,
        input_ids=payload.get("input_ids"),
        last_logits=None,
        model_kwargs={},
        image_grid_thw=payload.get("image_grid_thw"),
        video_grid_thw=payload.get("video_grid_thw"),
        source_visual_geometry=geometry,
        target_token_start=payload.get("target_token_start"),
        target_token_end=payload.get("target_token_end"),
        stop_reason=payload.get("stop_reason", "cached"),
        capture_found=bool(payload.get("capture_found", True)),
        second_full_forward_used=False,
        malformed=bool(payload.get("malformed", False)),
        errors=list(payload.get("errors") or []),
    )


def _detach_cpu(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else value.detach().cpu()


def _original_image_mm_token_type_ids(
    *,
    model: Any,
    input_ids: torch.Tensor,
    source_geometry: Qwen3SourceVisualGeometry,
    device: torch.device | str,
) -> torch.Tensor:
    token_type_ids = torch.zeros_like(input_ids, device=device)
    image_token_id = source_geometry.image_token_id
    if image_token_id is None:
        image_token_id = getattr(getattr(model, "config", None), "image_token_id", None)
    if image_token_id is not None:
        token_type_ids = token_type_ids.masked_fill(input_ids.to(device) == int(image_token_id), 1)
    return token_type_ids


def _stage1_image_grid_thw_for_eval(
    capture: Qwen3FocusCapture,
    source_geometry: Qwen3SourceVisualGeometry,
    device: torch.device | str,
) -> torch.Tensor | None:
    image_grid_thw = source_geometry.image_grid_thw
    if image_grid_thw is None:
        image_grid_thw = capture.image_grid_thw
    return None if image_grid_thw is None else image_grid_thw.to(device)


__all__ = [
    "V3EvalFeatureCacheItem",
    "add_v3_common_eval_args",
    "can_score_fvt_for_item",
    "compute_or_load_v3_eval_item",
    "compute_v3_readout_nll",
    "different_image_index_v3",
    "group_indices_by_v3_image",
    "load_qwen3_and_tgvf",
    "load_v3_eval_samples",
    "make_output_dir",
    "progress_iter",
    "random_d_like",
    "same_image_wrong_index_v3",
    "sample_metadata_row",
    "save_summary",
    "set_seed",
    "summarize_config",
    "v3_group_id",
    "v3_sample_uid",
    "write_json",
    "write_jsonl",
]
