import inspect

import pytest
import revisit_vlm_clean.runner as runner_module
import revisit_vlm_clean.stage2_native as stage2_native
from revisit_vlm_clean.benchmark_data import BenchmarkSample
from revisit_vlm_clean.rendering import render_benchmark_input
from revisit_vlm_clean.runner import (
    STAGE2_LEGACY_BACKEND,
    STAGE2_NATIVE_BACKEND,
    BackendConfig,
    ModelRunResult,
    TGVFStage2Qwen3Backend,
    TGVFStage2Qwen3NativeBackend,
    make_backend,
    resolve_backend_name,
)
from revisit_vlm_clean.schema import EvalMode, ForwardMode, RunConfig
from revisit_vlm_clean.stage2_native import (
    NativeStage2Engine,
    NativeStage2ExecutionNotPortedError,
    NativeStage2RunResult,
)
from revisit_vlm_clean.stage2_runtime import Stage2RuntimeConfig


def _run_config(mode: EvalMode = EvalMode.TGVF_FORCE) -> RunConfig:
    return RunConfig(
        run_id="stage2",
        checkpoint_path="outputs/checkpoint.pt",
        mode=mode,
        post_tgvf_forward_mode=ForwardMode.KV_CACHE,
        subset_id="core_smoke_256_seed20260625",
    )


def _runtime(tmp_path) -> Stage2RuntimeConfig:
    ckpt = tmp_path / "ckpt.pt"
    jsonl = tmp_path / "eval.jsonl"
    ckpt.write_bytes(b"placeholder")
    jsonl.write_text('{"image": "x.jpg", "need_focus": true}\n', encoding="utf-8")
    return Stage2RuntimeConfig(
        stage2_checkpoint=str(ckpt),
        eval_jsonl=str(jsonl),
    )


def _sample() -> BenchmarkSample:
    return BenchmarkSample(
        sample_id="sample-1",
        benchmark="vstar",
        population_id="core_smoke_256_seed20260625",
        source_file="vstar/test.jsonl",
        question="What is shown?",
        media=({"kind": "path", "path": "/tmp/image.jpg"},),
        choices=("A", "B"),
        gold_answer="A",
    )


def test_make_tgvf_stage2_backend_without_prepare(tmp_path) -> None:
    runtime = Stage2RuntimeConfig(
        stage2_checkpoint=str(tmp_path / "ckpt.pt"),
        eval_jsonl=str(tmp_path / "eval.jsonl"),
    )
    config = _run_config()
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
    config = _run_config()
    backend_config = BackendConfig(backend="tgvf_stage2_qwen3", stage2=runtime)
    backend = make_backend(backend_config, config=config)

    assert resolve_backend_name("tgvf_stage2_qwen3") == STAGE2_LEGACY_BACKEND
    assert isinstance(backend, TGVFStage2Qwen3Backend)
    assert backend_config.to_dict()["deprecated_alias"] is True
    assert backend_config.to_dict()["resolved_backend"] == STAGE2_LEGACY_BACKEND


def test_stage2_native_backend_uses_clean_native_engine(monkeypatch, tmp_path) -> None:
    runtime = _runtime(tmp_path)
    config = _run_config()

    class FakeNativeEngine:
        def __init__(self, *, stage2_config, backend_options) -> None:
            self.stage2_config = stage2_config
            self.backend_options = backend_options
            self.prepared_config = None

        def prepare(self, config) -> None:
            self.prepared_config = config

        def run(self, sample, rendered, config) -> NativeStage2RunResult:
            assert self.prepared_config is config
            assert sample.sample_id == rendered.sample_id == "sample-1"
            return NativeStage2RunResult(
                raw_output="B",
                triggered=True,
                focus_target="needle",
                focus_valid=True,
                append_success=True,
                output_tokens=7,
                debug={"fake_native": True},
            )

        def identity(self) -> dict[str, bool]:
            return {"fake_native": True}

    monkeypatch.setattr(runner_module, "NativeStage2Engine", FakeNativeEngine)
    backend = make_backend(
        BackendConfig(backend=STAGE2_NATIVE_BACKEND, stage2=runtime),
        config=config,
    )

    assert isinstance(backend, TGVFStage2Qwen3NativeBackend)
    backend.prepare(config)
    result = backend.run(_sample(), render_benchmark_input(_sample(), config), config)

    assert isinstance(result, ModelRunResult)
    assert result.raw_output == "B"
    assert result.triggered is True
    assert result.focus_target == "needle"
    assert result.append_success is True
    assert result.debug == {"fake_native": True}


def test_stage2_native_engine_records_identity_and_marks_execution_unported(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    config = _run_config()
    engine = NativeStage2Engine(
        stage2_config=runtime,
        backend_options={"dtype": "bfloat16", "device": "cuda:0"},
    )

    engine.prepare(config)
    identity = engine.identity()

    assert identity["capture_append_ported"] is False
    assert identity["checkpoint_file"]["exists"] is True
    assert identity["checkpoint_file"]["sha256"]
    assert identity["eval_jsonl"]["n_rows"] == 1
    assert identity["backend_options"] == {"dtype": "bfloat16", "device": "cuda:0"}
    with pytest.raises(NativeStage2ExecutionNotPortedError, match="capture/append"):
        engine.run(_sample(), render_benchmark_input(_sample(), config), config)


def test_stage2_native_backend_reports_unported_execution_as_row_error(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    config = _run_config()
    backend = make_backend(
        BackendConfig(backend=STAGE2_NATIVE_BACKEND, stage2=runtime),
        config=config,
    )

    backend.prepare(config)
    result = backend.run(_sample(), render_benchmark_input(_sample(), config), config)

    assert result.raw_output == ""
    assert "NativeStage2ExecutionNotPortedError" in str(result.error)
    assert result.debug["native_stage2_execution_ported"] is False
    assert result.debug["native_stage2"]["capture_append_ported"] is False


def test_stage2_native_module_does_not_depend_on_legacy_evaluator() -> None:
    source = inspect.getsource(stage2_native)

    assert "eval_v3_stage2_protocol" not in source
    assert "Stage2ProtocolEvaluator" not in source
