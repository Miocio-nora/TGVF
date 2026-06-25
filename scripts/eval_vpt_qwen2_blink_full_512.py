#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import time
import types
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor

from revisit_vlm.tgvf_capture import _vision_content_items
from tgvf_eval.adapters import BenchmarkRegistry, BenchmarkSample
from tgvf_eval.config import DEFAULT_SEED
from tgvf_eval.progress import configure_quiet_external_progress, iter_progress
from tgvf_eval.prompts import build_direct_prompt
from tgvf_eval.results import ResultWriter, make_run_paths, summarize_rows, write_config_yaml


VPT_MODELING_PATH = (
    Path(__file__).resolve().parents[1]
    / "example_project/VisualPerceptionToken/transformers/src/transformers/models/qwen2_vl_vpt/modeling_qwen2_vl_vpt.py"
)
VPT_REGION_TRIGGER_PROMPT = "Identify the region that can help you answer the question, and then answer the question"
VPT_REENCODE_TRIGGER_PROMPT = "Require additional perception features, and then answer the question"
REGION_X_TOKENS = [f"<|x_{index}|>" for index in range(8)]
REGION_Y_TOKENS = [f"<|y_{index}|>" for index in range(8)]
REGION_PATTERN = re.compile(r"<\|region_token_start\|>(<\|[xy]_[01234567]\|>)+<\|region_token_end\|>")


class VPTQwen2CLIPRunner:
    def __init__(
        self,
        *,
        model_id: str,
        base_model_id: str,
        device: str,
        dtype: str,
        max_image_resolution: int,
        max_new_tokens: int,
        vpt_second_round: str,
        vpt_trigger_prompt: str,
        vpt_force_action: str,
    ) -> None:
        self.model_id = model_id
        self.base_model_id = base_model_id
        self.device = device
        self.dtype = _resolve_dtype(dtype)
        self.max_image_resolution = max_image_resolution
        self.max_new_tokens = max_new_tokens
        self.vpt_second_round = vpt_second_round
        self.vpt_trigger_prompt = vpt_trigger_prompt
        self.vpt_force_action = vpt_force_action
        self.model: Any | None = None
        self.processor: Qwen2VLProcessor | None = None
        self.load_debug: dict[str, Any] = {}

    def prepare(self) -> None:
        module = _load_vpt_modeling_module()
        raw_config = json.loads(Path(hf_hub_download(self.model_id, filename="config.json")).read_text())
        config = _build_compat_config(
            module,
            raw_config=raw_config,
            base_model_id=self.base_model_id,
        )
        processor = Qwen2VLProcessor.from_pretrained(self.model_id, trust_remote_code=False)
        if not getattr(processor, "chat_template", None):
            base_processor = Qwen2VLProcessor.from_pretrained(self.base_model_id, trust_remote_code=False)
            processor.chat_template = base_processor.chat_template
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

        weights_path = hf_hub_download(self.model_id, filename="pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location="cpu", mmap=True)
        remapped_state_dict = {_map_vpt_key(key): value for key, value in state_dict.items()}

        start = time.perf_counter()
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)
        try:
            model = module.VPT_Qwen2VLForConditionalGeneration(config)
        finally:
            torch.set_default_dtype(previous_dtype)
        load_result = model.load_state_dict(remapped_state_dict, strict=False)
        missing_keys = list(load_result.missing_keys)
        unexpected_keys = list(load_result.unexpected_keys)
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "VPT checkpoint did not load cleanly after compat remap: "
                f"missing={len(missing_keys)} unexpected={len(unexpected_keys)} "
                f"missing_sample={missing_keys[:5]} unexpected_sample={unexpected_keys[:5]}"
            )
        model.to(self.device)
        model.eval()
        self.model = model
        self.processor = processor
        self.load_debug = {
            "modeling_path": str(VPT_MODELING_PATH),
            "base_model_id": self.base_model_id,
            "raw_model_id": self.model_id,
            "compat_key_remap": [
                "visual.* -> model.visual.*",
                "model.* -> model.language_model.*",
                "depth_projector.* -> clip_projector.*",
            ],
            "load_wall_time_sec": time.perf_counter() - start,
            "missing_keys": 0,
            "unexpected_keys": 0,
            "vision_encoder_ls": raw_config.get("vision_encoder_ls"),
            "num_inner_forward_run": raw_config.get("num_inner_forward_run"),
            "para_mask_id": raw_config.get("para_mask_id"),
            "para_mask_ratio": raw_config.get("para_mask_ratio"),
        }

    def run(self, sample: BenchmarkSample) -> dict[str, Any]:
        if self.model is None or self.processor is None:
            raise RuntimeError("runner.prepare() must be called before run()")
        base_prompt = build_direct_prompt(sample.question, cot_enabled=False)
        prompt = _apply_vpt_trigger_prompt(base_prompt, self.vpt_trigger_prompt)
        first_messages = self._build_direct_messages(sample, prompt)
        forced_action_text = _forced_action_text(self.vpt_force_action)
        forced_first_round = bool(forced_action_text)
        if forced_action_text:
            first_raw = forced_action_text
            first_tokens = 0
        else:
            first_raw, first_tokens = self._generate(first_messages)
        clip_action_text = _extract_clip_action(first_raw)
        region_actions = _extract_region_actions(first_raw)
        action_kind_count = int(bool(clip_action_text)) + int(bool(region_actions))
        second_round_used = False
        second_round_type = ""
        final_raw = first_raw
        second_raw = ""
        region_debug: dict[str, Any] = {}
        if clip_action_text and action_kind_count == 1 and self.vpt_second_round == "auto":
            second_messages, clip_media = self._build_second_round_messages(
                sample,
                prompt=prompt,
                action_text=clip_action_text,
            )
            second_raw, _second_tokens = self._generate(second_messages, clip_media=clip_media)
            if second_raw.strip():
                final_raw = second_raw
                second_round_used = True
                second_round_type = "clip_reencode"
        elif region_actions and action_kind_count == 1 and self.vpt_second_round == "auto":
            crops, region_debug = _crop_region_media(
                self._media_items(sample),
                region_actions,
                max_image_resolution=self.max_image_resolution,
            )
            second_messages = self._build_region_second_round_messages(
                sample,
                prompt=prompt,
                action_text=first_raw,
                crops=crops,
            )
            second_raw, _second_tokens = self._generate(second_messages)
            if second_raw.strip():
                final_raw = second_raw
                second_round_used = True
                second_round_type = "region_crop"
        return {
            "raw_output": final_raw,
            "debug": {
                "first_round_raw": first_raw,
                "first_round_tokens": first_tokens,
                "trigger_prompt": self.vpt_trigger_prompt,
                "force_action": self.vpt_force_action,
                "forced_first_round": forced_first_round,
                "clip_action_found": bool(clip_action_text),
                "clip_action_text": clip_action_text,
                "region_action_found": bool(region_actions),
                "region_actions": region_actions,
                "ambiguous_multi_action": action_kind_count > 1,
                "second_round_used": second_round_used,
                "second_round_type": second_round_type,
                "second_round_raw": second_raw,
                "num_input_media": len(self._media_items(sample)),
                **region_debug,
            },
        }

    def _media_items(self, sample: BenchmarkSample) -> list[Any]:
        return [item for item in sample.media if item is not None and not (isinstance(item, str) and not item)]

    def _build_direct_messages(self, sample: BenchmarkSample, prompt: str) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        image_kwargs = {"max_pixels": int(self.max_image_resolution) ** 2}
        for media in self._media_items(sample):
            content.extend(_vision_content_items(media, image_kwargs))
        if not content:
            raise ValueError("sample has no image/video media")
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _build_second_round_messages(
        self,
        sample: BenchmarkSample,
        *,
        prompt: str,
        action_text: str,
    ) -> tuple[list[dict[str, Any]], list[Any]]:
        first_messages = self._build_direct_messages(sample, prompt)
        clip_media = self._media_items(sample)
        clip_prompt = "\n".join("<clip_image>" for _ in clip_media)
        messages = [
            *first_messages,
            {"role": "assistant", "content": action_text},
            {"role": "user", "content": clip_prompt},
        ]
        return messages, clip_media

    def _build_region_second_round_messages(
        self,
        sample: BenchmarkSample,
        *,
        prompt: str,
        action_text: str,
        crops: list[Image.Image],
    ) -> list[dict[str, Any]]:
        first_messages = self._build_direct_messages(sample, prompt)
        content: list[dict[str, Any]] = []
        image_kwargs = {"max_pixels": int(self.max_image_resolution) ** 2}
        if len(crops) == 1:
            content.extend(_vision_content_items(crops[0], image_kwargs))
        else:
            for index, crop in enumerate(crops):
                content.append({"type": "text", "text": f"Region {index}: "})
                content.extend(_vision_content_items(crop, image_kwargs))
                content.append({"type": "text", "text": "\n"})
        return [
            *first_messages,
            {"role": "assistant", "content": action_text},
            {"role": "user", "content": content},
        ]

    def _generate(
        self,
        messages: list[dict[str, Any]],
        *,
        clip_media: list[Any] | None = None,
    ) -> tuple[str, int]:
        assert self.model is not None
        assert self.processor is not None
        inputs = self._build_inputs(messages, clip_media=clip_media)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=False,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[-1]
        generated = outputs[0, prompt_len:].detach().cpu().tolist()
        text = self.processor.tokenizer.decode(
            generated,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return _strip_eos(text), len(generated)

    def _build_inputs(
        self,
        messages: list[dict[str, Any]],
        *,
        clip_media: list[Any] | None = None,
    ) -> dict[str, Any]:
        assert self.processor is not None
        if clip_media:
            text, image_inputs, video_inputs, clip_inputs = self._render_vpt_second_round(
                messages,
                clip_media=clip_media,
            )
        else:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            clip_inputs = {}
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs.pop("mm_token_type_ids", None)
        inputs.update(clip_inputs)
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def _render_vpt_second_round(
        self,
        messages: list[dict[str, Any]],
        *,
        clip_media: list[Any],
    ) -> tuple[str, list[Any] | None, list[Any] | None, dict[str, torch.Tensor]]:
        assert self.processor is not None
        first_text = self.processor.apply_chat_template(
            [messages[0]],
            tokenize=False,
            add_generation_prompt=False,
        )
        image_inputs, video_inputs = process_vision_info([messages[0]])
        action_text = str(messages[1]["content"])
        clip_message = {"role": "user", "content": [_vision_content_items(item, {"max_pixels": int(self.max_image_resolution) ** 2})[0] for item in clip_media]}
        clip_image_inputs, _clip_video_inputs = process_vision_info([clip_message])
        clip_processor_kwargs = {
            "images": clip_image_inputs,
            "max_pixels": int(self.max_image_resolution) ** 2,
            "return_tensors": "pt",
        }
        try:
            clip_inputs_raw = self.processor.image_processor(
                **clip_processor_kwargs,
                videos=None,
            )
        except TypeError as exc:
            if "unexpected keyword argument 'videos'" not in str(exc):
                raise
            clip_inputs_raw = self.processor.image_processor(**clip_processor_kwargs)
        clip_grid = clip_inputs_raw["image_grid_thw"]
        merge_length = int(getattr(self.processor.image_processor, "merge_size", 2)) ** 2
        clip_chunks = []
        for grid in clip_grid:
            num_tokens = int(grid.prod().item()) // merge_length
            clip_chunks.append(
                "<|vision_start|>" + ("<|clip_image_pad|>" * num_tokens) + "<|vision_end|>"
            )
        clip_text = "\n".join(clip_chunks)
        text = (
            first_text
            + "<|im_start|>assistant\n"
            + action_text
            + "<|im_end|>\n"
            + "<|im_start|>user\n"
            + clip_text
            + "<|im_end|>\n"
            + "<|im_start|>assistant\n"
        )
        clip_inputs = {
            "clip_pixel_values": clip_inputs_raw["pixel_values"],
            "clip_images_grid_thw": clip_inputs_raw["image_grid_thw"],
        }
        return text, image_inputs, video_inputs, clip_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VPT Qwen2-VL-2B CLIP on BLINK full val.")
    parser.add_argument("--model-id", default="rp-yu/Qwen2-VL-2b-VPT-CLIP")
    parser.add_argument("--base-model-id", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--benchmark-root", default="/nvmesv/dredvpn009/datasets/benchmarks")
    parser.add_argument("--tools-root", default="/nvmesv/dredvpn009/datasets/benchmarks/_tools")
    parser.add_argument("--output-root", default="eval_outputs/vpt_qwen2vl2b_clip_blink_full_512")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--max-image-resolution", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--method", default="vpt_clip_direct")
    parser.add_argument("--vpt-second-round", choices=["auto", "off"], default="auto")
    parser.add_argument("--vpt-trigger-prompt", choices=["none", "region", "reencode"], default="none")
    parser.add_argument("--vpt-force-action", choices=["none", "clip", "region_full"], default="none")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")
    configure_quiet_external_progress()
    run_id = args.run_id or f"vpt_clip_blink_{int(time.time())}"
    paths = make_run_paths(args.output_root, run_id=run_id)
    config_payload = {
        "run_id": run_id,
        "benchmark": "blink",
        "tier": "full",
        "method": args.method,
        "model": {
            "model_id": args.model_id,
            "base_model_id": args.base_model_id,
            "dtype": args.dtype,
            "device": args.device,
            "max_new_tokens": args.max_new_tokens,
            "vpt_second_round": args.vpt_second_round,
            "vpt_trigger_prompt": args.vpt_trigger_prompt,
            "vpt_force_action": args.vpt_force_action,
        },
        "image": {
            "max_image_resolution": args.max_image_resolution,
            "max_pixels": int(args.max_image_resolution) ** 2,
        },
        "eval": {
            "benchmark_root": args.benchmark_root,
            "tools_root": args.tools_root,
            "seed": args.seed,
            "limit": args.limit,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "scoring_backend": "project",
            "official_llm_mode": "disabled",
        },
        "legacy_safety": {
            "shared_runner_modified": False,
            "example_project_files_modified": False,
            "loader_scope": "this script only",
        },
    }
    write_config_yaml(paths.root / "config.yaml", config_payload)
    adapter = BenchmarkRegistry.get(
        "blink",
        benchmark_root=args.benchmark_root,
        tools_root=args.tools_root,
        scoring_backend="project",
        official_llm_mode="disabled",
    )
    samples = adapter.sample(tier="full", limit=args.limit, seed=args.seed)
    before_shard = len(samples)
    if args.num_shards > 1:
        samples = [sample for index, sample in enumerate(samples) if index % args.num_shards == args.shard_index]
    config_payload["eval"]["num_samples_before_shard"] = before_shard
    config_payload["eval"]["num_samples_after_shard"] = len(samples)
    write_config_yaml(paths.root / "config.yaml", config_payload)

    runner = VPTQwen2CLIPRunner(
        model_id=args.model_id,
        base_model_id=args.base_model_id,
        device=args.device,
        dtype=args.dtype,
        max_image_resolution=args.max_image_resolution,
        max_new_tokens=args.max_new_tokens,
        vpt_second_round=args.vpt_second_round,
        vpt_trigger_prompt=args.vpt_trigger_prompt,
        vpt_force_action=args.vpt_force_action,
    )
    runner.prepare()
    config_payload["model"]["load_debug"] = runner.load_debug
    write_config_yaml(paths.root / "config.yaml", config_payload)

    writer = ResultWriter(paths, benchmark="blink", method=args.method)
    for sample in iter_progress(
        samples,
        desc=f"blink {args.method}",
        total=len(samples),
        enabled=not args.no_progress,
    ):
        start = time.perf_counter()
        error = None
        result: dict[str, Any]
        try:
            result = runner.run(sample)
            raw_output = str(result["raw_output"])
        except Exception as exc:
            raw_output = ""
            result = {"debug": {}}
            error = f"{type(exc).__name__}: {exc}"
        parsed = adapter.parse_prediction(raw_output, sample) if not error else ""
        score = None
        if sample.gold_answer is not None and not error:
            score = adapter.score_predictions(
                [
                    {
                        "sample_id": sample.sample_id,
                        "question": sample.question,
                        "raw_output": raw_output,
                        "parsed_answer": parsed,
                        "gold_answer": sample.gold_answer,
                        "choices": sample.choices,
                        "metadata": sample.metadata,
                    }
                ]
            ).get("score")
        row = {
            "benchmark": "blink",
            "sample_id": sample.sample_id,
            "method": args.method,
            "trigger_mode": (
                f"force_{args.vpt_force_action}"
                if args.vpt_force_action != "none"
                else args.vpt_trigger_prompt
            ),
            "tgvf_mode": "none",
            "cot_enabled": False,
            "reasoning_mode": "final_only",
            "max_image_resolution": args.max_image_resolution,
            "question": sample.question,
            "media": sample.media,
            "choices": sample.choices,
            "raw_output": raw_output,
            "parsed_answer": parsed,
            "gold_answer": sample.gold_answer,
            "score": score,
            "metadata": sample.metadata,
            "triggered": bool(
                result.get("debug", {}).get("clip_action_found")
                or result.get("debug", {}).get("region_action_found")
            ),
            "foveation_target": json.dumps(
                result.get("debug", {}).get("region_boxes", []),
                ensure_ascii=False,
            ),
            "num_foveations": len(result.get("debug", {}).get("region_boxes", [])),
            "second_full_forward_used": bool(result.get("debug", {}).get("second_round_used")),
            "wall_time_sec": time.perf_counter() - start,
            "visual_token_count": 0,
            "output_tokens": result.get("debug", {}).get("first_round_tokens", 0),
            "debug_metadata": result.get("debug", {}),
            "official_tool_used": adapter.tool_info.official_tool_used,
            "official_tool_path": adapter.tool_info.official_tool_path,
            "scorer_name": adapter.tool_info.scorer_name,
            "prompt_source": (
                f"project_direct_no_cot_vpt_trigger_{args.vpt_trigger_prompt}"
                f"_force_{args.vpt_force_action}"
            ),
            "official_compatible": adapter.tool_info.official_compatible,
            "llm_judge_required": adapter.tool_info.llm_judge_required,
            "error": error,
        }
        writer.append_row(row)

    rows = writer.read_rows()
    scorer_payload = adapter.score_predictions(rows)
    summary = summarize_rows(
        rows,
        run_id=run_id,
        benchmark="blink",
        tier="full",
        method=args.method,
        cot_enabled=False,
        scorer_payload=scorer_payload,
    )
    writer.write_summary(summary)
    (paths.scores / f"blink__{args.method}.scores.json").write_text(
        json.dumps(scorer_payload, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _load_vpt_modeling_module() -> types.ModuleType:
    source = VPT_MODELING_PATH.read_text()
    source = source.replace(
        "AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor",
        "AutoModelForCausalLM, AutoProcessor",
    )
    source = source.replace(
        "from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast",
        (
            "try:\n"
            "    from transformers import Qwen2TokenizerFast\n"
            "except Exception:\n"
            "    from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer as Qwen2TokenizerFast"
        ),
    )
    source = source.replace(
        "if cache_position[0] != 0:",
        "if cache_position is not None and cache_position[0] != 0:",
    )
    source = source.replace(
        "if inputs_embeds is not None and cache_position[0] == 0:",
        "if inputs_embeds is not None and (cache_position is None or cache_position[0] == 0):",
    )
    source = source.replace("self.model.embed_tokens", "self.model.language_model.embed_tokens")
    source = source.replace("self.visual", "self.model.visual")
    source = source.replace(
        "class Projector(nn.Module):",
        (
            "def _vpt_tensor_output(value):\n"
            "    if hasattr(value, 'to'):\n"
            "        return value\n"
            "    if hasattr(value, 'pooler_output') and value.pooler_output is not None:\n"
            "        return value.pooler_output\n"
            "    if hasattr(value, 'last_hidden_state'):\n"
            "        return value.last_hidden_state\n"
            "    return value[0]\n\n"
            "class Projector(nn.Module):"
        ),
    )
    source = source.replace(
        "image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)",
        "image_embeds = _vpt_tensor_output(self.model.visual(pixel_values, grid_thw=image_grid_thw))",
    )
    source = source.replace(
        "clip_embedding = self.model.visual(clip_pixel_values, grid_thw=clip_images_grid_thw)",
        "clip_embedding = _vpt_tensor_output(self.model.visual(clip_pixel_values, grid_thw=clip_images_grid_thw))",
    )
    module = types.ModuleType("vpt_modeling_qwen2_vl_compat")
    exec(compile(source, str(VPT_MODELING_PATH), "exec"), module.__dict__)
    return module


def _build_compat_config(module: types.ModuleType, *, raw_config: dict[str, Any], base_model_id: str) -> Any:
    base_config = Qwen2VLConfig.from_pretrained(base_model_id)
    config = module.VPT_Qwen2VLConfig(
        **base_config.to_dict(),
        detection_image_token_id=raw_config.get("detection_image_token_id"),
        detection_action_id=raw_config.get("detection_action_id"),
        detection_action_start_id=raw_config.get("detection_action_start_id"),
        para_start_id=raw_config.get("para_start_id"),
        para_end_id=raw_config.get("para_end_id"),
        clip_image_token_id=raw_config.get("clip_image_token_id"),
        clip_action_id=raw_config.get("clip_action_id"),
        clip_action_start_id=raw_config.get("clip_action_start_id"),
        seg_image_token_id=raw_config.get("seg_image_token_id"),
        seg_action_id=raw_config.get("seg_action_id"),
        seg_action_start_id=raw_config.get("seg_action_start_id"),
        num_inner_forward_run=raw_config.get("num_inner_forward_run", 2),
        projector_scale=raw_config.get("projector_scale", 1),
        para_mask_id=raw_config.get("para_mask_id", 0),
        para_mask_ratio=raw_config.get("para_mask_ratio", 0.0),
        alignment=raw_config.get("alignment", False),
        vision_encoder_ls=raw_config.get("vision_encoder_ls") or [],
    )
    config.text_config.vocab_size = int(raw_config["vocab_size"])
    config.vocab_size = int(raw_config["vocab_size"])
    config.hidden_size = int(raw_config.get("hidden_size") or config.text_config.hidden_size)
    return config


def _map_vpt_key(key: str) -> str:
    if key.startswith("depth_projector."):
        return "clip_projector." + key[len("depth_projector.") :]
    if key.startswith("visual."):
        return "model.visual." + key[len("visual.") :]
    if key.startswith("model."):
        return "model.language_model." + key[len("model.") :]
    return key


def _extract_clip_action(text: str) -> str:
    start = text.find("<|clip_action_start|>")
    if start < 0:
        return ""
    end_token = "<|clip_action_end|>"
    end = text.find(end_token, start)
    if end < 0:
        return ""
    return text[start : end + len(end_token)]


def _extract_region_actions(text: str) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for match in REGION_PATTERN.finditer(text):
        match_text = match.group()
        x_tokens = [token for token in REGION_X_TOKENS if token in match_text]
        y_tokens = [token for token in REGION_Y_TOKENS if token in match_text]
        if x_tokens and y_tokens:
            actions.append(
                {
                    "text": match_text,
                    "x_tokens": x_tokens,
                    "y_tokens": y_tokens,
                }
            )
    return actions


def _crop_region_media(
    media_items: list[Any],
    region_actions: list[dict[str, Any]],
    *,
    max_image_resolution: int,
    num_cut: int = 8,
) -> tuple[list[Image.Image], dict[str, Any]]:
    if not media_items:
        raise ValueError("region action requested but sample has no image media")
    image = _open_media_as_image(media_items[0])
    width, height = image.size
    crops: list[Image.Image] = []
    boxes: list[dict[str, Any]] = []
    for action in region_actions:
        x_indices = [_region_token_index(token, "x") for token in action["x_tokens"]]
        y_indices = [_region_token_index(token, "y") for token in action["y_tokens"]]
        min_x = max(min(x_indices) - 1, 0)
        max_x = min(max(x_indices) + 1, num_cut - 1)
        min_y = max(min(y_indices) - 1, 0)
        max_y = min(max(y_indices) + 1, num_cut - 1)
        grid_width = width / num_cut
        grid_height = height / num_cut
        bbox = (
            min_x * grid_width,
            min_y * grid_height,
            (max_x + 1) * grid_width,
            (max_y + 1) * grid_height,
        )
        crops.append(image.crop(bbox).convert("RGB"))
        boxes.append(
            {
                "grid": [min_x, min_y, max_x, max_y],
                "pixel_bbox": [int(round(value)) for value in bbox],
                "source_media_index": 0,
            }
        )
    return crops, {
        "region_boxes": boxes,
        "region_num_crops": len(crops),
        "region_uses_first_media_only": True,
        "region_num_original_media": len(media_items),
        "region_max_image_resolution": max_image_resolution,
    }


def _open_media_as_image(media: Any) -> Image.Image:
    if isinstance(media, Image.Image):
        return media.convert("RGB").copy()
    if isinstance(media, (str, Path)):
        media_text = str(media)
        if media_text.startswith(("http://", "https://")):
            raise ValueError("region crop does not support remote image URLs in this evaluator")
        with Image.open(media_text) as image:
            return image.convert("RGB").copy()
    if isinstance(media, dict) and media.get("bytes"):
        import io

        with Image.open(io.BytesIO(media["bytes"])) as image:
            return image.convert("RGB").copy()
    raise TypeError(f"Unsupported media type for region crop: {type(media).__name__}")


def _region_token_index(token: str, axis: str) -> int:
    return int(token.replace(f"<|{axis}_", "").replace("|>", ""))


def _apply_vpt_trigger_prompt(prompt: str, mode: str) -> str:
    if mode == "none":
        return prompt
    if mode == "region":
        return f"{VPT_REGION_TRIGGER_PROMPT}\n{prompt}"
    if mode == "reencode":
        return f"{VPT_REENCODE_TRIGGER_PROMPT}\n{prompt}"
    raise ValueError(f"Unsupported VPT trigger prompt: {mode}")


def _forced_action_text(mode: str) -> str:
    if mode == "none":
        return ""
    if mode == "clip":
        return "<|clip_action_start|><|clip_action|><|clip_action_end|>"
    if mode == "region_full":
        return (
            "<|region_token_start|>"
            + "".join(REGION_X_TOKENS)
            + "".join(REGION_Y_TOKENS)
            + "<|region_token_end|>"
        )
    raise ValueError(f"Unsupported VPT force action: {mode}")


def _strip_eos(text: str) -> str:
    return text.replace("<|im_end|>", "").strip()


def _resolve_dtype(dtype: str) -> torch.dtype:
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp16", "float16"}:
        return torch.float16
    if dtype in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


if __name__ == "__main__":
    main()
