import json

import pytest

from revisit_vlm_clean.benchmark_data import BenchmarkSample
from revisit_vlm_clean.legacy_stage2_adapter import (
    build_legacy_stage2_args,
    legacy_append_prefill_mode,
    stage2_record_from_clean_sample,
)
from revisit_vlm_clean.rendering import render_benchmark_input
from revisit_vlm_clean.schema import EvalMode, ForwardMode, RunConfig
from revisit_vlm_clean.stage2_runtime import Stage2RuntimeConfig


def _run_config(mode: EvalMode, forward_mode: ForwardMode = ForwardMode.KV_CACHE) -> RunConfig:
    return RunConfig(
        run_id="legacy",
        checkpoint_path="outputs/checkpoint.pt",
        mode=mode,
        post_tgvf_forward_mode=forward_mode,
        subset_id="core_smoke_256_seed20260625",
    )


def _sample(kind: str = "path") -> BenchmarkSample:
    media = (
        {"kind": "path", "path": "/tmp/image.jpg", "exists": True}
        if kind == "path"
        else {"kind": "embedded_base64", "payload_loaded": True, "char_length": 10}
    )
    return BenchmarkSample(
        sample_id="sample-1",
        benchmark="vstar_bench",
        population_id="vstar_test_questions_191",
        source_file="vstar_bench/snapshot/test_questions.jsonl",
        question="What color?\n(A) red\n(B) blue",
        media=(media,),
        choices=("red", "blue"),
        gold_answer="B",
        metadata={"category": "toy"},
    )


def _runtime(tmp_path, forward_mode: ForwardMode = ForwardMode.KV_CACHE) -> Stage2RuntimeConfig:
    ckpt = tmp_path / "ckpt.pt"
    jsonl = tmp_path / "eval.jsonl"
    ckpt.write_bytes(b"placeholder")
    jsonl.write_text(json.dumps({"image": "x.jpg", "need_focus": True}) + "\n")
    return Stage2RuntimeConfig(
        stage2_checkpoint=str(ckpt),
        eval_jsonl=str(jsonl),
        append_forward_mode=forward_mode,
    )


def test_legacy_append_prefill_mode_mapping() -> None:
    assert legacy_append_prefill_mode(ForwardMode.KV_CACHE) == "kv_cache"
    assert legacy_append_prefill_mode(ForwardMode.NO_KV_FULL_SEQUENCE) == "full_sequence"


def test_build_legacy_stage2_args_maps_clean_mode_and_forward(tmp_path) -> None:
    args = build_legacy_stage2_args(
        runtime=_runtime(tmp_path, ForwardMode.NO_KV_FULL_SEQUENCE),
        run_config=_run_config(EvalMode.TGVF_FORCE, ForwardMode.NO_KV_FULL_SEQUENCE),
        output_dir=tmp_path / "out",
    )

    assert args.blocks == "force_end2end"
    assert args.append_prefill_mode == "full_sequence"
    assert args.d_conditions == "correct_D"


def test_stage2_record_from_clean_sample_requires_path_media() -> None:
    sample = _sample("path")
    rendered = render_benchmark_input(sample, _run_config(EvalMode.TGVF_FREE))
    record = stage2_record_from_clean_sample(sample, rendered)

    assert record["image"] == "/tmp/image.jpg"
    assert record["need_focus"] is True
    assert record["answer_format"] == "multiple_choice"


def test_stage2_record_from_clean_sample_rejects_embedded_media() -> None:
    sample = _sample("embedded_base64")
    rendered = render_benchmark_input(sample, _run_config(EvalMode.TGVF_FREE))

    with pytest.raises(ValueError, match="path-backed image media"):
        stage2_record_from_clean_sample(sample, rendered)
