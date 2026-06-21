#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from PIL import Image, ImageDraw

from revisit_vlm.qwen3_vl_tgvf import (
    PROTOCOL_C_TOOL_OBSERVATION,
    PROTOCOL_C_THINKING_SPECIAL,
    TGVF_PROTOCOL_CHOICES,
    ensure_protocol_c_special_tokens,
    load_qwen3_vl,
)
from revisit_vlm.tgvf_foveal import finalize_tgvf_output_with_frozen_qwen_merger
from revisit_vlm.tgvf_training import TGVF_VARIANTS, build_tgvf_module
from revisit_vlm.tgvf_v3_stage1 import (
    TGVFv3Stage1Dataset,
    TGVFv3Stage1Sample,
    _image_input,
    capture_v3_stage1_focus_teacher_forced,
    freeze_qwen_backbone,
    format_question_with_choices,
)
from revisit_vlm.qwen3_vl_tgvf import tap_qwen3_vision_features


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_config = checkpoint.get("config") or {}
    checkpoint_tgvf = checkpoint_config.get("tgvf") or {}
    protocol = args.tgvf_protocol or checkpoint_config.get("tgvf_protocol") or "legacy_v3_tags"
    variant = args.variant or checkpoint_tgvf.get("variant") or "tgvf_v2_bidirectional"
    num_fvt = _resolve_num_fvt(args.num_foveated_tokens, checkpoint_tgvf)
    spatial_merge_size_arg = args.spatial_merge_size or checkpoint_tgvf.get("spatial_merge_size") or "auto"
    attn_dim = args.attn_dim if args.attn_dim is not None else checkpoint_tgvf.get("attn_dim")

    loaded = load_qwen3_vl(
        args.model_id,
        processor_id=args.processor_id,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )
    model = loaded.model
    processor = loaded.processor
    if getattr(processor.tokenizer, "pad_token", None) is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    token_info: dict[str, Any] = {}
    if protocol in {PROTOCOL_C_THINKING_SPECIAL, PROTOCOL_C_TOOL_OBSERVATION}:
        token_info = ensure_protocol_c_special_tokens(processor.tokenizer, model)
    freeze_qwen_backbone(model)

    model, lora_info = _maybe_load_lora(
        model=model,
        checkpoint=checkpoint,
        checkpoint_config=checkpoint_config,
        protocol=protocol,
        args=args,
    )
    model.eval()
    utility_model = model.get_base_model() if hasattr(model, "get_base_model") else model

    dataset = TGVFv3Stage1Dataset(args.samples, focus_only=True, min_confidence=args.min_confidence)
    selected = _select_samples(
        dataset.samples,
        num_samples=args.num_samples,
        same_image_groups_only=args.same_image_groups_only,
        seed=args.seed,
    )
    if not selected:
        raise RuntimeError("No focus samples selected for visualization")

    dims = _infer_dims_from_first_sample(
        model=utility_model,
        processor=processor,
        sample=selected[0],
        device=device,
        max_image_resolution=args.max_image_resolution,
    )
    spatial_merge_size = (
        int(dims["spatial_merge_size"])
        if str(spatial_merge_size_arg) == "auto"
        else int(spatial_merge_size_arg)
    )
    foveal_module = build_tgvf_module(
        variant=variant,
        d_lm=int(dims["d_lm"]),
        d_v=int(dims["d_v"]),
        num_foveated_tokens=num_fvt,
        spatial_merge_size=spatial_merge_size,
        attn_dim=attn_dim,
    ).to(device=device, dtype=next(model.parameters()).dtype)
    foveal_module.load_state_dict(checkpoint["tgvf_module"], strict=True)
    foveal_module.eval()

    summaries = []
    for index, sample in enumerate(selected):
        sample_dir = output_dir / f"sample_{index:04d}_{_slug(sample.image_id or Path(sample.image).stem)}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        summary = _run_sample(
            model=model,
            utility_model=utility_model,
            processor=processor,
            foveal_module=foveal_module,
            sample=sample,
            sample_dir=sample_dir,
            device=device,
            protocol=protocol,
            max_image_resolution=args.max_image_resolution,
            hidden_state_index=args.hidden_state_index,
        )
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))

    report = {
        "checkpoint": str(args.checkpoint),
        "model_id": args.model_id,
        "processor_id": loaded.processor_id,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "attn_implementation": args.attn_implementation,
        "tgvf_protocol": protocol,
        "variant": variant,
        "num_foveated_tokens": num_fvt,
        "spatial_merge_size": spatial_merge_size,
        "attn_dim": attn_dim,
        "token_info": token_info,
        "lora_info": lora_info,
        "num_samples": len(summaries),
        "summaries": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_json_safe(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _run_sample(
    *,
    model: Any,
    utility_model: Any,
    processor: Any,
    foveal_module: torch.nn.Module,
    sample: TGVFv3Stage1Sample,
    sample_dir: Path,
    device: torch.device,
    protocol: str,
    max_image_resolution: int | None,
    hidden_state_index: int,
) -> dict[str, Any]:
    image_input = _image_input(sample.image, max_image_resolution=max_image_resolution)
    capture = capture_v3_stage1_focus_teacher_forced(
        model=model,
        processor=processor,
        image=image_input,
        question=sample.prompt_question,
        target=sample.target,
        device=device,
        hidden_state_index=hidden_state_index,
        protocol=protocol,
    )
    if not capture.capture_found:
        raise RuntimeError(f"Could not capture target span for {sample.image_id}: {capture.errors}")
    tap, v_pre, v_merge = tap_qwen3_vision_features(
        utility_model,
        processor,
        image=image_input,
        question=sample.prompt_question,
        device=device,
    )
    if v_pre is None:
        raise RuntimeError(f"Qwen3 V_pre tap failed for {sample.image_id}: {tap.errors}")
    with torch.no_grad():
        output = foveal_module(
            target_hidden_states=capture.target_hidden_states.to(device),
            pre_merge_visual_tokens=v_pre.to(device),
            metadata={
                "target": sample.target,
                "image_id": sample.image_id,
                "question": sample.question,
                "stage": "tgvf_attention_visualization",
            },
        )
        finalized = finalize_tgvf_output_with_frozen_qwen_merger(utility_model, output)

    image = Image.open(sample.image).convert("RGB")
    image.save(sample_dir / "image.png")

    grid_shape = _premerge_grid_shape(capture.image_grid_thw, int(v_pre.shape[0]))
    heatmaps: dict[str, torch.Tensor] = {}
    heatmaps.update(_attention_heatmaps(output.attention_debug, visual_token_count=int(v_pre.shape[0])))
    files: dict[str, str] = {}
    stats: dict[str, Any] = {}
    for name, scores in heatmaps.items():
        grid = _scores_to_grid(scores, grid_shape)
        if grid is None:
            continue
        torch.save(scores.detach().cpu(), sample_dir / f"{name}.pt")
        torch.save(grid.detach().cpu(), sample_dir / f"{name}_grid.pt")
        overlay_path = sample_dir / f"{name}_overlay.png"
        heat_path = sample_dir / f"{name}_heatmap.png"
        _save_heatmap_overlay(image, grid, overlay_path, label_text=sample.target)
        _save_heatmap_image(grid, heat_path)
        files[f"{name}_overlay"] = str(overlay_path)
        files[f"{name}_heatmap"] = str(heat_path)
        stats[name] = _heatmap_stats(scores, grid_shape)

    debug = {
        "image": sample.image,
        "image_id": sample.image_id,
        "source_dataset": sample.source_dataset,
        "source_profile": sample.source_profile,
        "question": sample.question,
        "choices": sample.choices,
        "target": sample.target,
        "captured_target_text": capture.target_text,
        "evidence_description": sample.evidence_description,
        "answer": sample.answer or sample.short_answer,
        "target_style": sample.target_style,
        "target_cues": sample.target_cues,
        "target_hidden_shape": list(capture.target_hidden_states.shape),
        "target_token_ids": capture.target_token_ids,
        "v_pre_shape": list(v_pre.shape),
        "v_merge_shape": None if v_merge is None else list(v_merge.shape),
        "d_shape": list(finalized.foveated_visual_tokens.shape),
        "image_grid_thw": None
        if capture.image_grid_thw is None
        else capture.image_grid_thw.detach().cpu().tolist(),
        "premerge_grid_shape": None if grid_shape is None else list(grid_shape),
        "attention_debug_shapes": {
            key: list(value.shape) if isinstance(value, torch.Tensor) else None
            for key, value in output.attention_debug.items()
        },
        "heatmap_files": files,
        "heatmap_stats": stats,
        "second_full_forward_used": bool(capture.second_full_forward_used),
    }
    (sample_dir / "debug.json").write_text(
        json.dumps(_json_safe(debug), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "sample_dir": str(sample_dir),
        "image_id": sample.image_id,
        "question": sample.question,
        "target": sample.target,
        "captured_target_text": capture.target_text,
        "target_hidden_shape": list(capture.target_hidden_states.shape),
        "v_pre_shape": list(v_pre.shape),
        "d_shape": list(finalized.foveated_visual_tokens.shape),
        "premerge_grid_shape": None if grid_shape is None else list(grid_shape),
        "heatmap_files": files,
        "heatmap_stats": stats,
        "second_full_forward_used": bool(capture.second_full_forward_used),
    }


def _attention_heatmaps(
    attention_debug: dict[str, Any],
    *,
    visual_token_count: int,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    target_to_visual = attention_debug.get("target_to_visual_attention")
    if isinstance(target_to_visual, torch.Tensor) and target_to_visual.numel():
        # [target_tokens, visual_tokens]
        if target_to_visual.ndim == 2 and target_to_visual.shape[-1] == visual_token_count:
            result["target_to_visual_mean"] = target_to_visual.float().mean(dim=0)
            result["target_to_visual_max"] = target_to_visual.float().max(dim=0).values
    visual_salience = attention_debug.get("attention_weights")
    if isinstance(visual_salience, torch.Tensor) and visual_salience.numel():
        scores = visual_salience.float()
        if scores.ndim == 1 and scores.shape[0] == visual_token_count:
            result["visual_salience"] = scores
        elif scores.ndim == 2 and scores.shape[-1] == visual_token_count:
            result["visual_salience"] = scores.mean(dim=0)
    subslot = attention_debug.get("sub_slot_attention_weights")
    if isinstance(subslot, torch.Tensor) and subslot.numel() and subslot.shape[-1] == visual_token_count:
        result["subslot_visual_attention"] = subslot.float().reshape(-1, visual_token_count).mean(dim=0)
    return result


def _premerge_grid_shape(image_grid_thw: torch.Tensor | None, token_count: int) -> tuple[int, int] | None:
    if image_grid_thw is not None:
        values = image_grid_thw.detach().cpu().reshape(-1).tolist()
        if len(values) >= 3:
            t, h, w = int(values[-3]), int(values[-2]), int(values[-1])
            if t == 1 and h * w == token_count:
                return h, w
            if t > 0 and h * w * t == token_count:
                return h * t, w
    root = int(round(math.sqrt(token_count)))
    if root * root == token_count:
        return root, root
    return None


def _scores_to_grid(scores: torch.Tensor, grid_shape: tuple[int, int] | None) -> torch.Tensor | None:
    if grid_shape is None:
        return None
    h, w = grid_shape
    if int(scores.numel()) != h * w:
        return None
    return scores.detach().float().reshape(h, w)


def _heatmap_stats(scores: torch.Tensor, grid_shape: tuple[int, int] | None) -> dict[str, Any]:
    values = scores.detach().float().flatten()
    total = values.sum().clamp_min(1e-12)
    probs = values.clamp_min(0) / values.clamp_min(0).sum().clamp_min(1e-12)
    entropy = float((-(probs * probs.clamp_min(1e-12).log()).sum()).item())
    topk = min(10, int(values.numel()))
    top_values, top_indices = torch.topk(values, k=topk)
    top_coords = []
    if grid_shape is not None:
        h, w = grid_shape
        top_coords = [[int(idx.item()) // w, int(idx.item()) % w] for idx in top_indices]
    return {
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "std": float(values.std().item()) if values.numel() > 1 else 0.0,
        "sum": float(total.item()),
        "entropy": entropy,
        "top_indices": [int(idx.item()) for idx in top_indices],
        "top_coords_yx": top_coords,
        "top_values": [float(item) for item in top_values.detach().cpu().tolist()],
    }


def _save_heatmap_overlay(
    image: Image.Image,
    grid_scores: torch.Tensor,
    path: Path,
    *,
    label_text: str | None = None,
) -> None:
    heat = _scores_to_heatmap(grid_scores)
    heat = heat.resize(image.size, Image.Resampling.BILINEAR)
    overlay = Image.blend(image.convert("RGBA"), heat, alpha=0.45)
    if label_text:
        _draw_label(overlay, label_text)
    overlay.save(path)


def _save_heatmap_image(grid_scores: torch.Tensor, path: Path) -> None:
    heat = _scores_to_heatmap(grid_scores)
    heat = heat.resize((grid_scores.shape[1] * 24, grid_scores.shape[0] * 24), Image.Resampling.NEAREST)
    heat.save(path)


def _scores_to_heatmap(grid_scores: torch.Tensor) -> Image.Image:
    values = grid_scores.float()
    values = (values - values.min()) / (values.max() - values.min()).clamp_min(1e-6)
    arr = (values * 255).to(torch.uint8).cpu()
    red = arr
    green = torch.zeros_like(arr)
    blue = 255 - arr
    alpha = torch.full_like(arr, 180)
    rgba = torch.stack([red, green, blue, alpha], dim=-1).numpy()
    return Image.fromarray(rgba, mode="RGBA")


def _draw_label(image: Image.Image, text: str) -> None:
    draw = ImageDraw.Draw(image)
    max_chars = max(24, min(64, image.size[0] // 8))
    lines = _wrap_text(f"target: {text}", max_chars=max_chars)[:4]
    line_height = 14
    padding = 6
    width = min(image.size[0] - 8, max(draw.textlength(line) for line in lines) + padding * 2)
    height = len(lines) * line_height + padding * 2
    draw.rectangle((4, 4, 4 + width, 4 + height), fill=(0, 0, 0, 230))
    y = 4 + padding
    for line in lines:
        draw.text((4 + padding, y), line, fill=(0, 255, 80, 255))
        y += line_height


def _wrap_text(text: str, *, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [text]


def _maybe_load_lora(
    *,
    model: Any,
    checkpoint: dict[str, Any],
    checkpoint_config: dict[str, Any],
    protocol: str,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    if not args.load_lora or "qwen_lora" not in checkpoint:
        return model, {"loaded": False, "reason": "disabled_or_no_qwen_lora"}
    lora_cfg = checkpoint_config.get("lora") or {}
    target_modules = lora_cfg.get("target_modules") or args.lora_target_modules.split(",")
    target_modules = [str(item).strip() for item in target_modules if str(item).strip()]
    modules_to_save = lora_cfg.get("modules_to_save")
    if modules_to_save is None and protocol in {PROTOCOL_C_THINKING_SPECIAL, PROTOCOL_C_TOOL_OBSERVATION}:
        modules_to_save = ["embed_tokens", "lm_head"]
    config = LoraConfig(
        r=int(lora_cfg.get("rank") or args.lora_rank),
        lora_alpha=int(lora_cfg.get("alpha") or args.lora_alpha),
        target_modules=target_modules,
        lora_dropout=float(lora_cfg.get("dropout") if lora_cfg.get("dropout") is not None else args.lora_dropout),
        bias=str(lora_cfg.get("bias") or args.lora_bias),
        task_type="CAUSAL_LM",
        modules_to_save=modules_to_save,
        ensure_weight_tying=False,
    )
    model = get_peft_model(model, config)
    result = set_peft_model_state_dict(model, checkpoint["qwen_lora"])
    return model, {
        "loaded": True,
        "missing_keys": list(getattr(result, "missing_keys", []) or []),
        "unexpected_keys": list(getattr(result, "unexpected_keys", []) or []),
        "target_modules": target_modules,
        "modules_to_save": modules_to_save,
    }


def _infer_dims_from_first_sample(
    *,
    model: Any,
    processor: Any,
    sample: TGVFv3Stage1Sample,
    device: torch.device,
    max_image_resolution: int | None,
) -> dict[str, int]:
    image_input = _image_input(sample.image, max_image_resolution=max_image_resolution)
    tap, v_pre, _ = tap_qwen3_vision_features(
        model,
        processor,
        image=image_input,
        question=sample.prompt_question,
        device=device,
    )
    if v_pre is None:
        raise RuntimeError(f"Could not infer V_pre shape: {tap.errors}")
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    d_lm = getattr(text_config, "hidden_size", None) or getattr(config, "hidden_size")
    return {
        "d_lm": int(d_lm),
        "d_v": int(v_pre.shape[-1]),
        "spatial_merge_size": int(tap.spatial_merge_size or tap.merge_size or 2),
    }


def _resolve_num_fvt(cli_value: int | None, tgvf_config: dict[str, Any]) -> int | None:
    if cli_value is not None:
        return cli_value
    for key in ("num_foveated_tokens", "num_fvt_tokens", "num_fvt"):
        if tgvf_config.get(key) is not None:
            return int(tgvf_config[key])
    return None


def _select_samples(
    samples: list[TGVFv3Stage1Sample],
    *,
    num_samples: int,
    same_image_groups_only: bool,
    seed: int,
) -> list[TGVFv3Stage1Sample]:
    candidates = list(samples)
    if same_image_groups_only:
        counts: dict[str, int] = {}
        for sample in candidates:
            key = sample.image_id or sample.image
            counts[key] = counts.get(key, 0) + 1
        candidates = [sample for sample in candidates if counts.get(sample.image_id or sample.image, 0) > 1]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:num_samples]


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:80] or "sample"


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _optional_int(value: str) -> int | None:
    if value.lower() in {"none", "null", "auto"}:
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize TGVF target-to-visual attention heatmaps.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", required=True, help="v3 teacher train/val JSONL; focus rows are used.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Thinking")
    parser.add_argument("--processor-id", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=("auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"))
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--tgvf-protocol", choices=TGVF_PROTOCOL_CHOICES, default=None)
    parser.add_argument("--variant", choices=TGVF_VARIANTS, default=None)
    parser.add_argument("--num-foveated-tokens", type=_optional_int, default=None)
    parser.add_argument("--spatial-merge-size", default=None)
    parser.add_argument("--attn-dim", type=int, default=None)
    parser.add_argument("--max-image-resolution", type=_optional_int, default=512)
    parser.add_argument("--hidden-state-index", type=int, default=-1)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--same-image-groups-only", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=256)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-bias", default="none", choices=("none", "all", "lora_only"))
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    return parser.parse_args()


if __name__ == "__main__":
    main()
