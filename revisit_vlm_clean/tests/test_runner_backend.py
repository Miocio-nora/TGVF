import pytest
from revisit_vlm_clean.runner import (
    STAGE2_LEGACY_BACKEND,
    STAGE2_NATIVE_BACKEND,
    BackendConfig,
    TGVFStage2Qwen3Backend,
    TGVFStage2Qwen3NativeBackend,
    make_backend,
    resolve_backend_name,
)
from revisit_vlm_clean.schema import EvalMode, ForwardMode, RunConfig
from revisit_vlm_clean.stage2_runtime import Stage2RuntimeConfig


def test_make_tgvf_stage2_backend_without_prepare(tmp_path) -> None:
    runtime = Stage2RuntimeConfig(
        stage2_checkpoint=str(tmp_path / "ckpt.pt"),
        eval_jsonl=str(tmp_path / "eval.jsonl"),
    )
    config = RunConfig(
        run_id="stage2",
        checkpoint_path="outputs/checkpoint.pt",
        mode=EvalMode.TGVF_FORCE,
        post_tgvf_forward_mode=ForwardMode.KV_CACHE,
        subset_id="core_smoke_256_seed20260625",
    )
    backend = make_backend(
        BackendConfig(backend=STAGE2_LEGACY_BACKEND, stage2=runtime),
        config=config,
    )

    assert isinstance(backend, TGVFStage2Qwen3Backend)


def test_stage2_legacy_alias_is_explicit(tmp_path) -> None:
    runtime = Stage2RuntimeConfig(
        stage2_checkpoint=str(tmp_path / "ckpt.pt"),
        eval_jsonl=str(tmp_path / "eval.jsonl"),
    )
    config = RunConfig(
        run_id="stage2",
        checkpoint_path="outputs/checkpoint.pt",
        mode=EvalMode.TGVF_FORCE,
        post_tgvf_forward_mode=ForwardMode.KV_CACHE,
        subset_id="core_smoke_256_seed20260625",
    )
    backend_config = BackendConfig(backend="tgvf_stage2_qwen3", stage2=runtime)
    backend = make_backend(backend_config, config=config)

    assert resolve_backend_name("tgvf_stage2_qwen3") == STAGE2_LEGACY_BACKEND
    assert isinstance(backend, TGVFStage2Qwen3Backend)
    assert backend_config.to_dict()["deprecated_alias"] is True
    assert backend_config.to_dict()["resolved_backend"] == STAGE2_LEGACY_BACKEND


def test_stage2_native_backend_fails_fast_until_ported(tmp_path) -> None:
    ckpt = tmp_path / "ckpt.pt"
    jsonl = tmp_path / "eval.jsonl"
    ckpt.write_bytes(b"placeholder")
    jsonl.write_text('{"image": "x.jpg", "need_focus": true}\n', encoding="utf-8")
    runtime = Stage2RuntimeConfig(
        stage2_checkpoint=str(ckpt),
        eval_jsonl=str(jsonl),
    )
    config = RunConfig(
        run_id="stage2",
        checkpoint_path="outputs/checkpoint.pt",
        mode=EvalMode.TGVF_FORCE,
        post_tgvf_forward_mode=ForwardMode.KV_CACHE,
        subset_id="core_smoke_256_seed20260625",
    )
    backend = make_backend(
        BackendConfig(backend=STAGE2_NATIVE_BACKEND, stage2=runtime),
        config=config,
    )

    assert isinstance(backend, TGVFStage2Qwen3NativeBackend)
    with pytest.raises(NotImplementedError, match="final clean-native"):
        backend.prepare(config)
