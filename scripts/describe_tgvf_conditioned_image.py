from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from eval.common import load_qwen_and_tgvf
from revisit_vlm.tgvf import FOVEATE_END, FOVEATE_START
from revisit_vlm.tgvf_capture import (
    build_qwen2vl_tgvf_inputs,
    capture_tgvf_single_pass,
    trim_capture_target_at_delimiters,
)
from revisit_vlm.tgvf_foveal import (
    Qwen2VLPreMergeVisualHook,
    finalize_tgvf_output_with_frozen_qwen_merger,
)
from revisit_vlm.tgvf_training import (
    TGVF_DYNAMIC_NUM_FVT_VARIANTS,
    TGVF_VARIANTS,
    ForcedFoveationBracketWrapper,
    Qwen2VLMergedVisualHook,
    generate_readout_from_fvt,
    prepare_readout_prefix_inputs,
)
from tgvf_eval.adapters import BenchmarkRegistry
from tgvf_eval.config import (
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_SEED,
    image_budget_kwargs,
    method_config_from_name,
)
from tgvf_eval.prompts import build_prompt


def main() -> None:
    args = parse_args()
    device = f"cuda:{args.device_index}" if args.device_index is not None else args.device
    load_args = argparse.Namespace(
        model_path=args.model_path,
        processor_path=args.processor_path,
        tgvf_checkpoint=args.tgvf_checkpoint,
        variant=args.tgvf_variant,
        device=device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        num_foveated_tokens=args.num_foveated_tokens,
        spatial_merge_size=args.spatial_merge_size,
    )
    model, processor, foveal_module, device_obj, _info = load_qwen_and_tgvf(load_args)
    model.eval()
    foveal_module.eval()

    adapter = BenchmarkRegistry.get(args.benchmark, benchmark_root=args.benchmark_root)
    samples = adapter.sample(tier=args.tier, limit=args.limit, seed=args.seed)
    if args.sample_ids:
        wanted = {item.strip() for item in args.sample_ids.split(",") if item.strip()}
        samples = [sample for sample in samples if sample.sample_id in wanted]
    samples = samples[: args.max_samples]

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for sample in samples:
            row = describe_sample(
                args=args,
                model=model,
                processor=processor,
                foveal_module=foveal_module,
                device=device_obj,
                sample=sample,
            )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps(_brief_row(row), ensure_ascii=False), flush=True)


@torch.no_grad()
def describe_sample(
    *,
    args: argparse.Namespace,
    model: Any,
    processor: Any,
    foveal_module: torch.nn.Module,
    device: torch.device,
    sample: Any,
) -> dict[str, Any]:
    native_description = generate_native_description(
        model=model,
        processor=processor,
        image=sample.primary_media,
        prompt=args.description_prompt,
        image_kwargs=image_budget_kwargs(args.image_budget),
        max_new_tokens=args.max_new_tokens,
        eos_token_id=processor.tokenizer.eos_token_id,
        device=device,
    )

    config = method_config_from_name(
        "tgvf_module_force",
        image_budget=args.image_budget,
        fvt_append_mode=args.fvt_append_mode,
    )
    prompt = build_prompt(sample.question, config).prompt
    image_kwargs = image_budget_kwargs(args.image_budget)
    messages = _single_user_messages(sample.primary_media, prompt, image_kwargs)
    capture_model, forced_token_count = _force_marker_capture_model(
        model,
        processor,
        max_target_tokens=args.forced_target_max_tokens,
    )
    with Qwen2VLPreMergeVisualHook(model) as visual_hook, Qwen2VLMergedVisualHook(model) as merged_hook:
        capture = capture_tgvf_single_pass(
            capture_model,
            processor,
            image=sample.primary_media,
            question=prompt,
            messages=messages,
            max_new_tokens=forced_token_count + 4,
            device=device,
            hidden_state_index=args.capture_layer,
            eos_token_id=processor.tokenizer.eos_token_id,
            image_kwargs=image_kwargs,
        )
    if capture.capture_found:
        trim_capture_target_at_delimiters(capture, processor.tokenizer)
    if not capture.capture_found:
        return {
            "sample_id": sample.sample_id,
            "question": sample.question,
            "media": sample.media,
            "choices": sample.choices,
            "gold_answer": sample.gold_answer,
            "capture_found": False,
            "native_image_description": native_description,
            "generated_text": capture.generated_text,
            "stop_reason": capture.stop_reason,
        }
    if visual_hook.pre_merge_visual_tokens is None:
        raise RuntimeError("Pre-merge visual tokens were not captured")
    if merged_hook.merged_visual_tokens is None:
        raise RuntimeError("Merged visual tokens were not captured")

    original_d = merged_hook.merged_visual_tokens.to(device)
    if original_d.ndim == 3:
        original_d = original_d.reshape(-1, original_d.shape[-1])
    if original_d.ndim != 2:
        raise ValueError("Merged visual tokens must have shape [N, d_lm]")

    fvt_output = foveal_module(
        target_hidden_states=capture.target_hidden_states.to(device),
        pre_merge_visual_tokens=visual_hook.pre_merge_visual_tokens.to(device),
        metadata={"target_text": capture.target_text, "diagnostic": "conditioned_image_readout"},
    )
    fvt_output = finalize_tgvf_output_with_frozen_qwen_merger(model, fvt_output)
    use_source_grid = bool(fvt_output.debug_metadata.get("tgvf_version") == "v2")
    if use_source_grid and capture.image_grid_thw is None:
        raise RuntimeError("v2 conditioned-image readout requires source image_grid_thw")
    image_grid_thw = capture.image_grid_thw if use_source_grid else None

    conditional_d_description = generate_readout_from_fvt(
        model=model,
        tokenizer_or_processor=processor,
        target=None,
        foveated_visual_tokens=fvt_output.foveated_visual_tokens,
        image_grid_thw=image_grid_thw,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=processor.tokenizer.eos_token_id,
        device=device,
    )
    conditional_d_target_description = generate_readout_from_fvt(
        model=model,
        tokenizer_or_processor=processor,
        target=capture.target_text,
        foveated_visual_tokens=fvt_output.foveated_visual_tokens,
        image_grid_thw=image_grid_thw,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=processor.tokenizer.eos_token_id,
        device=device,
    )
    original_d_description = generate_readout_from_fvt(
        model=model,
        tokenizer_or_processor=processor,
        target=None,
        foveated_visual_tokens=original_d,
        image_grid_thw=capture.image_grid_thw,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=processor.tokenizer.eos_token_id,
        device=device,
    )
    original_d_target_description = generate_readout_from_fvt(
        model=model,
        tokenizer_or_processor=processor,
        target=capture.target_text,
        foveated_visual_tokens=original_d,
        image_grid_thw=capture.image_grid_thw,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=processor.tokenizer.eos_token_id,
        device=device,
    )
    prefix_debug = prepare_readout_prefix_inputs(
        model=model,
        tokenizer_or_processor=processor,
        target=None,
        foveated_visual_tokens=fvt_output.foveated_visual_tokens,
        image_grid_thw=image_grid_thw,
        device=device,
    )

    return {
        "sample_id": sample.sample_id,
        "question": sample.question,
        "media": sample.media,
        "choices": sample.choices,
        "gold_answer": sample.gold_answer,
        "metadata": sample.metadata,
        "capture_found": True,
        "native_image_description": native_description,
        "foveation_request": capture.generated_text,
        "target_text": capture.target_text,
        "raw_target_text": capture.raw_target_text,
        "target_token_count": len(capture.target_token_ids),
        "conditional_d_description": conditional_d_description,
        "conditional_d_description_with_target": conditional_d_target_description,
        "original_d_description": original_d_description,
        "original_d_description_with_target": original_d_target_description,
        "conditioned_image_description_no_target": conditional_d_description,
        "conditioned_image_description_with_target": conditional_d_target_description,
        "fvt_shape": list(fvt_output.foveated_visual_tokens.shape),
        "original_d_shape": list(original_d.shape),
        "pre_merge_visual_shape": list(visual_hook.pre_merge_visual_tokens.shape),
        "merged_visual_shape": list(original_d.shape),
        "target_hidden_shape": list(capture.target_hidden_states.shape),
        "tgvf_debug": fvt_output.debug_metadata,
        "readout_debug": {
            "readout_append_mode": prefix_debug["readout_append_mode"],
            "position_ids_source": prefix_debug["position_ids_source"],
            "fake_image_grid_thw": prefix_debug["fake_image_grid_thw"],
            "source_image_grid_thw": prefix_debug["source_image_grid_thw"],
            "fvt_grid_thw": prefix_debug["fvt_grid_thw"],
            "fvt_grid_source": prefix_debug["fvt_grid_source"],
            "actual_image_pad_token_count": prefix_debug["actual_image_pad_token_count"],
            "expected_llm_image_tokens": prefix_debug["expected_llm_image_tokens"],
            "image_pad_mm_type_is_image": prefix_debug["image_pad_mm_type_is_image"],
            "image_position_ids_are_3d": prefix_debug["image_position_ids_are_3d"],
            "text_position_ids_are_1d": prefix_debug["text_position_ids_are_1d"],
            "visual_tower_called_for_fvt": prefix_debug["visual_tower_called_for_fvt"],
        },
        "source_image_grid_thw": None
        if capture.image_grid_thw is None
        else capture.image_grid_thw.detach().cpu().tolist(),
        "stop_reason": capture.stop_reason,
    }


@torch.no_grad()
def generate_native_description(
    *,
    model: Any,
    processor: Any,
    image: Any,
    prompt: str,
    image_kwargs: dict[str, Any],
    max_new_tokens: int,
    eos_token_id: int | None,
    device: torch.device,
) -> str:
    messages = _single_user_messages(image, prompt, image_kwargs)
    inputs = build_qwen2vl_tgvf_inputs(
        processor,
        image=image,
        question=prompt,
        messages=messages,
        image_kwargs=image_kwargs,
    )
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=eos_token_id,
    )
    prompt_len = int(inputs["input_ids"].shape[-1])
    generated = outputs[0, prompt_len:].detach().cpu().tolist()
    return processor.tokenizer.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def _force_marker_capture_model(
    model: Any, processor: Any, *, max_target_tokens: int
) -> tuple[Any, int]:
    start_ids = processor.tokenizer.encode(FOVEATE_START, add_special_tokens=False)
    end_ids = processor.tokenizer.encode(FOVEATE_END, add_special_tokens=False)
    suppress_ids = (
        [processor.tokenizer.eos_token_id] if processor.tokenizer.eos_token_id is not None else []
    )
    wrapped = ForcedFoveationBracketWrapper(
        model,
        start_ids=start_ids,
        end_ids=end_ids,
        max_target_tokens=max_target_tokens,
        suppress_ids=suppress_ids,
    )
    return wrapped, len(start_ids) + int(max_target_tokens) + len(end_ids)


def _single_user_messages(
    media: Any, prompt: str, image_kwargs: dict[str, Any]
) -> list[dict[str, Any]]:
    from revisit_vlm.tgvf_capture import _vision_content_items

    return [
        {
            "role": "user",
            "content": [
                *_vision_content_items(media, image_kwargs),
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _brief_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row.get("sample_id"),
        "target_text": row.get("target_text"),
        "native": row.get("native_image_description"),
        "conditional_d": row.get("conditional_d_description"),
        "conditional_d_with_target": row.get("conditional_d_description_with_target"),
        "original_d": row.get("original_d_description"),
        "original_d_with_target": row.get("original_d_description_with_target"),
        "readout_debug": row.get("readout_debug"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Describe TGVF conditioned pseudo-images with Qwen2-VL."
    )
    parser.add_argument("--benchmark", default="vstar_bench", choices=BenchmarkRegistry.names())
    parser.add_argument("--benchmark-root", default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--tier", choices=["light", "medium", "full"], default="full")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--sample-ids", default=None, help="Comma-separated benchmark sample ids.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model-path", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--processor-path", default=None)
    parser.add_argument("--tgvf-checkpoint", required=True)
    parser.add_argument("--tgvf-variant", choices=TGVF_VARIANTS, required=True)
    parser.add_argument("--num-foveated-tokens", type=_optional_positive_int, default=16)
    parser.add_argument("--spatial-merge-size", default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-index", type=int, default=None)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--image-budget", choices=["low", "mid", "high"], default="mid")
    parser.add_argument("--capture-layer", type=int, default=-1)
    parser.add_argument("--forced-target-max-tokens", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--description-prompt",
        default=(
            "Describe the image content in detail. Mention visible objects, colors, "
            "text, and spatial relationships. Do not answer any multiple-choice question."
        ),
    )
    parser.add_argument(
        "--fvt-append-mode",
        choices=["qwen_native_pseudo_image"],
        default="qwen_native_pseudo_image",
    )
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()
    if args.tgvf_variant not in TGVF_DYNAMIC_NUM_FVT_VARIANTS and args.num_foveated_tokens is None:
        raise ValueError("--num-foveated-tokens none is supported only for dynamic/v2 variants")
    return args


def _optional_positive_int(value: str) -> int | None:
    if value.lower() in {"none", "null"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer or 'none'")
    return parsed


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
