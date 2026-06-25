"""Adapter contracts for the historical Stage2 evaluator.

This is a narrow bridge used while the clean runner is being brought up. It
keeps clean benchmark identity and mode names on this side, and exposes only
the single-step Stage2 configuration we intend to preserve.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .benchmark_data import BenchmarkSample
from .rendering import RenderedBenchmarkInput
from .schema import ForwardMode, RunConfig
from .stage2_runtime import Stage2RuntimeConfig


def legacy_append_prefill_mode(mode: ForwardMode) -> str:
    if mode == ForwardMode.KV_CACHE:
        return "kv_cache"
    if mode == ForwardMode.NO_KV_FULL_SEQUENCE:
        return "full_sequence"
    raise ValueError(f"unsupported forward mode for legacy Stage2 adapter: {mode}")


def build_legacy_stage2_args(
    *,
    runtime: Stage2RuntimeConfig,
    run_config: RunConfig,
    output_dir: str | Path,
) -> argparse.Namespace:
    runtime.validate()
    return argparse.Namespace(
        stage2_checkpoint=runtime.stage2_checkpoint,
        eval_jsonl=runtime.eval_jsonl,
        output_dir=str(output_dir),
        model_id=run_config.model_id,
        processor_id=run_config.processor_id,
        dtype="bfloat16",
        device="cuda:0",
        device_map="cuda:0",
        attn_implementation="sdpa",
        tgvf_protocol=runtime.protocol,
        variant="tgvf_v2_bidirectional",
        num_foveated_tokens=None,
        lora_rank=64,
        lora_alpha=256,
        lora_target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        max_image_resolution=run_config.max_image_resolution,
        max_action_tokens=runtime.max_action_tokens,
        max_answer_tokens=runtime.max_answer_tokens,
        fvt_position_mode="native_source_grid",
        append_prefill_mode=legacy_append_prefill_mode(runtime.append_forward_mode),
        blocks=_legacy_blocks_for_mode(run_config),
        d_conditions=runtime.d_condition,
        max_focus=1,
        max_no_focus=0,
        min_confidence=None,
        num_shards=1,
        shard_index=0,
        wrong_search_limit=1,
        force_prefix_mode=runtime.force_prefix_mode,
        progress=False,
        log_every=8,
    )


def stage2_record_from_clean_sample(
    sample: BenchmarkSample,
    rendered: RenderedBenchmarkInput,
) -> dict[str, Any]:
    image = _primary_path_media(sample)
    return {
        "image": image,
        "question": rendered.user_prompt,
        "answer": sample.gold_answer or "",
        "need_focus": True,
        "evidence_state": "need_local_visual_evidence",
        "trajectory_type": "single_focus",
        "target": "",
        "evidence_description": "",
        "image_id": sample.sample_id,
        "choices": None,
        "answer_format": "multiple_choice" if sample.choices else "open",
        "source_dataset": sample.benchmark,
        "source_profile": str(sample.metadata.get("category") or sample.population_id),
        "metadata": {
            "population_id": sample.population_id,
            "source_file": sample.source_file,
            "choices": list(sample.choices),
            **sample.metadata,
        },
    }


def make_legacy_stage2_sample(sample: BenchmarkSample, rendered: RenderedBenchmarkInput) -> Any:
    from revisit_vlm.tgvf_v3_stage2 import TGVFv3Stage2Sample

    return TGVFv3Stage2Sample(**stage2_record_from_clean_sample(sample, rendered))


def _legacy_blocks_for_mode(run_config: RunConfig) -> str:
    if run_config.mode.value == "tgvf_free":
        return "free_router_end2end"
    if run_config.mode.value in {"tgvf_force", "tgvf_softforce"}:
        return "force_end2end"
    raise ValueError(f"Stage2 legacy adapter does not support mode={run_config.mode.value!r}")


def _primary_path_media(sample: BenchmarkSample) -> str:
    for media in sample.media:
        if media.get("kind") == "path" and media.get("path") and media.get("exists") is True:
            return str(media["path"])
    raise ValueError(
        "legacy Stage2 adapter requires path-backed image media; "
        f"sample {sample.sample_id} has media kinds {[item.get('kind') for item in sample.media]}"
    )
