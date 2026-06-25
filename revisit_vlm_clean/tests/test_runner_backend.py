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
from revisit_vlm_clean.schema import (
    DeepStackScope,
    DeepStackState,
    EvalMode,
    ForwardMode,
    RunConfig,
)
from revisit_vlm_clean.stage2_native import (
    NativeStage2Engine,
    NativeStage2RunResult,
    stage2_sample_from_clean_sample,
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


def _deepstack_run_config() -> RunConfig:
    return RunConfig(
        run_id="stage2_deepstack",
        checkpoint_path="outputs/checkpoint.pt",
        mode=EvalMode.TGVF_FORCE,
        post_tgvf_forward_mode=ForwardMode.KV_CACHE,
        subset_id="core_smoke_256_seed20260625",
        deepstack=DeepStackState(
            enabled=True,
            original_image_scope=DeepStackScope.THROUGH_ANSWER,
        ),
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
        media=({"kind": "path", "path": "/tmp/image.jpg", "exists": True},),
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


def test_stage2_native_backend_rejects_unported_deepstack_execution(tmp_path) -> None:
    backend = make_backend(
        BackendConfig(backend=STAGE2_NATIVE_BACKEND, stage2=_runtime(tmp_path)),
        config=_deepstack_run_config(),
    )

    with pytest.raises(NotImplementedError, match="DeepStack execution is not implemented"):
        backend.prepare(_deepstack_run_config())


def test_stage2_legacy_backend_rejects_unported_deepstack_execution(tmp_path) -> None:
    backend = make_backend(
        BackendConfig(backend=STAGE2_LEGACY_BACKEND, stage2=_runtime(tmp_path)),
        config=_deepstack_run_config(),
    )

    with pytest.raises(NotImplementedError, match="DeepStack execution is not implemented"):
        backend.prepare(_deepstack_run_config())


def test_stage2_native_engine_records_identity_without_loading_runtime(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    config = _run_config()
    engine = NativeStage2Engine(
        stage2_config=runtime,
        backend_options={"dtype": "bfloat16", "device": "cuda:0"},
    )

    engine.prepare(config)
    identity = engine.identity()

    assert identity["capture_append_ported"] is True
    assert identity["heavy_runtime_loaded"] is False
    assert identity["checkpoint_file"]["exists"] is True
    assert identity["checkpoint_file"]["sha256"]
    assert identity["eval_jsonl"]["n_rows"] == 1
    assert identity["backend_options"] == {"dtype": "bfloat16", "device": "cuda:0"}


def test_stage2_sample_from_clean_sample_uses_rendered_prompt() -> None:
    sample = _sample()
    config = RunConfig(
        run_id="stage2",
        checkpoint_path="outputs/checkpoint.pt",
        mode=EvalMode.TGVF_SOFTFORCE,
        post_tgvf_forward_mode=ForwardMode.KV_CACHE,
        subset_id="core_smoke_256_seed20260625",
        softforce_prompt_text="Use focus tool.",
    )
    rendered = render_benchmark_input(sample, config)

    converted = stage2_sample_from_clean_sample(sample, rendered)

    assert converted.image == "/tmp/image.jpg"
    assert converted.question.endswith("Use focus tool.")
    assert converted.choices is None
    assert converted.metadata["choices"] == ["A", "B"]


def test_stage2_native_engine_force_flow_with_fake_runtime(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    config = _run_config(EvalMode.TGVF_FORCE)
    engine = _FakeFlowEngine(stage2_config=runtime)

    engine.prepare(config)
    result = engine.run(_sample(), render_benchmark_input(_sample(), config), config)

    assert result.raw_output == "B"
    assert result.triggered is True
    assert result.focus_target == "needle"
    assert result.focus_valid is True
    assert result.append_success is True
    assert result.output_tokens == 1
    assert result.debug["block"] == "clean_native_force_end2end"
    assert result.debug["mask_mode"] == "fake_append"


def test_stage2_native_engine_free_flow_with_fake_runtime(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    config = _run_config(EvalMode.TGVF_FREE)
    engine = _FakeFlowEngine(stage2_config=runtime)

    engine.prepare(config)
    result = engine.run(_sample(), render_benchmark_input(_sample(), config), config)

    assert result.raw_output == "B"
    assert result.triggered is True
    assert result.debug["block"] == "clean_native_free_end2end"
    assert engine.calls[:4] == ["ensure_loaded", "capture_free", "parse", "d_from_capture"]


def test_stage2_native_backend_reports_runtime_errors_as_row_error(monkeypatch, tmp_path) -> None:
    runtime = _runtime(tmp_path)
    config = _run_config()

    class FakeErrorEngine:
        def __init__(self, *, stage2_config, backend_options) -> None:
            del stage2_config, backend_options

        def prepare(self, config) -> None:
            del config

        def run(self, sample, rendered, config):
            del sample, rendered, config
            raise ValueError("fake runtime failure")

        def identity(self) -> dict[str, bool]:
            return {"fake_error_engine": True}

    monkeypatch.setattr(runner_module, "NativeStage2Engine", FakeErrorEngine)
    backend = make_backend(
        BackendConfig(backend=STAGE2_NATIVE_BACKEND, stage2=runtime),
        config=config,
    )

    backend.prepare(config)
    result = backend.run(_sample(), render_benchmark_input(_sample(), config), config)

    assert result.raw_output == ""
    assert result.error == "ValueError: fake runtime failure"
    assert result.debug["native_stage2"] == {"fake_error_engine": True}


def test_stage2_native_module_does_not_depend_on_legacy_evaluator() -> None:
    source = inspect.getsource(stage2_native)

    assert "eval_v3_stage2_protocol" not in source
    assert "Stage2ProtocolEvaluator" not in source


class _FakeParsed:
    evidence_state = "need_local_visual_evidence"
    focus_target = "needle"
    answer = "B"
    answer_valid = True
    malformed = False


class _FakeShape:
    shape = (1, 2, 3)


class _FakeGeometry:
    source_visual_token_count = 4


class _FakeCapture:
    target_text = "needle"
    target_token_ids = [1, 2]
    target_hidden_states = _FakeShape()
    generated_ids = [10, 11]
    generated_text = "<|focus_start|>needle<|focus_end|>"
    capture_found = True
    malformed = False
    second_full_forward_used = False
    source_visual_geometry = _FakeGeometry()
    stop_reason = "focus_end"
    errors = []


class _FakeAppend:
    debug_metadata = {
        "fvt_shape": [4, 8],
        "fvt_append_path": "fake_append",
        "fvt_position_mode": "native_source_grid",
        "second_full_forward_used": False,
    }


class _FakeContinuation:
    generated_text = "B"
    generated_ids = [42]


class _FakeFlowEngine(NativeStage2Engine):
    def __init__(self, *, stage2_config) -> None:
        super().__init__(stage2_config=stage2_config)
        self.calls = []

    def _ensure_loaded(self, sample) -> None:
        del sample
        self.calls.append("ensure_loaded")

    def _capture_generated_focus(self, sample, *, force_prefix):
        del sample
        self.calls.append(f"capture_force={force_prefix}")
        return _FakeCapture()

    def _capture_free_router(self, sample):
        del sample
        self.calls.append("capture_free")
        return _FakeCapture()

    def _d_from_capture(self, sample, capture, *, focus_source):
        del sample, capture, focus_source
        self.calls.append("d_from_capture")
        return object()

    def _append_visual_d(self, capture, d):
        del capture, d
        self.calls.append("append_visual_d")
        return _FakeAppend()

    def _continue_generation(self, append_result):
        del append_result
        self.calls.append("continue")
        return _FakeContinuation()

    def _parse_action(self, text):
        del text
        self.calls.append("parse")
        return _FakeParsed()
