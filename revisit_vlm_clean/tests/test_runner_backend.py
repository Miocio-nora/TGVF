import base64
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import revisit_vlm_clean.runner as runner_module
import revisit_vlm_clean.stage2_native as stage2_native
from revisit_vlm_clean.benchmark_data import BenchmarkSample
from revisit_vlm_clean.rendering import render_benchmark_input
from revisit_vlm_clean.runner import (
    STAGE2_GENERIC_BACKEND,
    STAGE2_LEGACY_BACKEND,
    STAGE2_NATIVE_BACKEND,
    BackendConfig,
    ModelRunResult,
    TGVFStage2Qwen3Backend,
    TGVFStage2Qwen3NativeBackend,
    backend_role_identity,
    build_deepstack_execution_plan,
    make_backend,
    resolve_backend_name,
)
from revisit_vlm_clean.schema import (
    DeepStackScope,
    DeepStackState,
    EvalFamily,
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


def _diagnostic_run_config(mode: EvalMode = EvalMode.TGVF_FORCE) -> RunConfig:
    return RunConfig(
        run_id="stage2_diagnostic",
        checkpoint_path="outputs/checkpoint.pt",
        eval_family=EvalFamily.INTERNAL_DIAGNOSTIC,
        mode=mode,
        post_tgvf_forward_mode=ForwardMode.KV_CACHE,
        subset_id="diagnostic_vstar_first_1_20260626",
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


def _deepstack_full_sequence_run_config() -> RunConfig:
    return RunConfig(
        run_id="stage2_deepstack_full_sequence",
        checkpoint_path="outputs/checkpoint.pt",
        mode=EvalMode.TGVF_FORCE,
        post_tgvf_forward_mode=ForwardMode.NO_KV_FULL_SEQUENCE,
        subset_id="core_smoke_256_seed20260625",
        deepstack=DeepStackState(
            enabled=True,
            original_image_scope=DeepStackScope.THROUGH_ANSWER,
        ),
    )


def _deepstack_full_sequence_evidence_run_config() -> RunConfig:
    return RunConfig(
        run_id="stage2_deepstack_full_sequence_evidence",
        checkpoint_path="outputs/checkpoint.pt",
        mode=EvalMode.TGVF_FORCE,
        post_tgvf_forward_mode=ForwardMode.NO_KV_FULL_SEQUENCE,
        subset_id="core_smoke_256_seed20260625",
        deepstack=DeepStackState(
            enabled=True,
            original_image_scope=DeepStackScope.EVIDENCE_ONLY,
        ),
    )


def _runtime(
    tmp_path,
    *,
    append_forward_mode: ForwardMode = ForwardMode.KV_CACHE,
) -> Stage2RuntimeConfig:
    ckpt = tmp_path / "ckpt.pt"
    jsonl = tmp_path / "eval.jsonl"
    ckpt.write_bytes(b"placeholder")
    jsonl.write_text('{"image": "x.jpg", "need_focus": true}\n', encoding="utf-8")
    return Stage2RuntimeConfig(
        stage2_checkpoint=str(ckpt),
        eval_jsonl=str(jsonl),
        append_forward_mode=append_forward_mode,
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


def test_stage2_legacy_backend_requires_internal_diagnostic_family(tmp_path) -> None:
    runtime = Stage2RuntimeConfig(
        stage2_checkpoint=str(tmp_path / "ckpt.pt"),
        eval_jsonl=str(tmp_path / "eval.jsonl"),
    )
    with pytest.raises(ValueError, match="diagnostic-only"):
        make_backend(
            BackendConfig(backend=STAGE2_LEGACY_BACKEND, stage2=runtime),
            config=_run_config(),
        )


def test_stage2_backend_role_identity_marks_legacy_as_diagnostic_bridge() -> None:
    assert backend_role_identity(STAGE2_GENERIC_BACKEND) == {
        "final_clean_backend": True,
        "diagnostic_bridge": False,
        "requires_internal_diagnostic_family": False,
    }
    assert backend_role_identity(STAGE2_NATIVE_BACKEND) == {
        "final_clean_backend": True,
        "diagnostic_bridge": False,
        "requires_internal_diagnostic_family": False,
    }
    assert backend_role_identity(STAGE2_LEGACY_BACKEND) == {
        "final_clean_backend": False,
        "diagnostic_bridge": True,
        "requires_internal_diagnostic_family": True,
    }


def test_runner_does_not_top_level_import_legacy_stage2_adapter() -> None:
    source = inspect.getsource(runner_module)
    top_level_import_block = "\n".join(source.splitlines()[:40])

    assert "legacy_stage2_adapter import" not in top_level_import_block


def test_make_tgvf_stage2_legacy_backend_without_prepare_for_diagnostic(tmp_path) -> None:
    runtime = Stage2RuntimeConfig(
        stage2_checkpoint=str(tmp_path / "ckpt.pt"),
        eval_jsonl=str(tmp_path / "eval.jsonl"),
    )
    backend = make_backend(
        BackendConfig(backend=STAGE2_LEGACY_BACKEND, stage2=runtime),
        config=_diagnostic_run_config(),
    )

    assert isinstance(backend, TGVFStage2Qwen3Backend)


def test_stage2_generic_backend_resolves_to_clean_native(monkeypatch, tmp_path) -> None:
    runtime = Stage2RuntimeConfig(
        stage2_checkpoint=str(tmp_path / "ckpt.pt"),
        eval_jsonl=str(tmp_path / "eval.jsonl"),
    )
    config = _run_config()

    class FakeNativeEngine:
        def __init__(self, *, stage2_config, backend_options) -> None:
            self.stage2_config = stage2_config
            self.backend_options = backend_options

    monkeypatch.setattr(runner_module, "NativeStage2Engine", FakeNativeEngine)
    backend_config = BackendConfig(backend=STAGE2_GENERIC_BACKEND, stage2=runtime)
    backend = make_backend(backend_config, config=config)

    assert resolve_backend_name(STAGE2_GENERIC_BACKEND) == STAGE2_NATIVE_BACKEND
    assert isinstance(backend, TGVFStage2Qwen3NativeBackend)
    assert backend_config.to_dict()["stage2_generic_alias"] is True
    assert backend_config.to_dict()["deprecated_alias"] is False
    assert backend_config.to_dict()["final_clean_backend"] is True
    assert backend_config.to_dict()["diagnostic_bridge"] is False
    assert backend_config.to_dict()["alias_target"] == STAGE2_NATIVE_BACKEND
    assert backend_config.to_dict()["resolved_backend"] == STAGE2_NATIVE_BACKEND


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

    with pytest.raises(NotImplementedError, match="deepstack_execution_plan"):
        backend.prepare(_deepstack_run_config())


def test_stage2_native_backend_accepts_supported_full_sequence_deepstack(tmp_path) -> None:
    config = _deepstack_full_sequence_run_config()
    backend = make_backend(
        BackendConfig(
            backend=STAGE2_NATIVE_BACKEND,
            stage2=_runtime(
                tmp_path,
                append_forward_mode=ForwardMode.NO_KV_FULL_SEQUENCE,
            ),
        ),
        config=config,
    )

    backend.prepare(config)


def test_stage2_native_backend_accepts_supported_full_sequence_evidence_only_deepstack(
    tmp_path,
) -> None:
    config = _deepstack_full_sequence_evidence_run_config()
    backend = make_backend(
        BackendConfig(
            backend=STAGE2_NATIVE_BACKEND,
            stage2=_runtime(
                tmp_path,
                append_forward_mode=ForwardMode.NO_KV_FULL_SEQUENCE,
            ),
        ),
        config=config,
    )

    backend.prepare(config)


def test_deepstack_execution_plan_records_scope_semantics() -> None:
    disabled = build_deepstack_execution_plan(
        _run_config(),
        backend=STAGE2_NATIVE_BACKEND,
    )
    assert disabled["enabled"] is False
    assert disabled["execution_supported"] is True
    assert disabled["status"] == "disabled_noop"
    assert disabled["original_image_scope"] == "off"
    assert disabled["blocking_items"] == []
    assert disabled["runtime_hooks"]["all_required_hooks_implemented"] is True
    assert all(
        hook["status"] == "not_required"
        for hook in disabled["runtime_hooks"]["hooks"].values()
    )
    assert disabled["scope_contract"]["execution_supported"] is True
    assert disabled["scope_contract"]["blocking_items"] == []

    through_answer = build_deepstack_execution_plan(
        _deepstack_run_config(),
        backend=STAGE2_NATIVE_BACKEND,
    )
    assert through_answer["execution_supported"] is False
    assert through_answer["original_image_scope"] == "through_answer"
    assert through_answer["runtime_hooks"]["all_required_hooks_implemented"] is False
    assert through_answer["runtime_hooks"]["hooks"][
        "capture_original_image_deepstack_features"
    ]["status"] == "not_ported"
    assert through_answer["runtime_hooks"]["hooks"][
        "apply_post_tgvf_deepstack_scope_mask"
    ]["required"] is True
    assert through_answer["runtime_hooks"]["hooks"][
        "restore_deepstack_for_answer_when_scope_requires"
    ]["required"] is False
    assert through_answer["blocking_items"] == through_answer["runtime_hooks"][
        "blocking_items"
    ]
    assert through_answer["original_image_deepstack"]["block_after_tgvf_append"] is True
    assert through_answer["original_image_deepstack"]["restore_for_answer"] is False
    assert through_answer["d_deepstack_features"]["required_for_current_mainline"] is False
    assert through_answer["scope_contract"]["surface"] == "benchmark_eval"
    assert through_answer["scope_contract"]["backend"] == STAGE2_NATIVE_BACKEND
    assert through_answer["scope_contract"]["original_image_deepstack"] == (
        through_answer["original_image_deepstack"]
    )
    assert through_answer["scope_contract"]["original_image_deepstack"][
        "block_query_end"
    ] is None

    full_sequence = build_deepstack_execution_plan(
        _deepstack_full_sequence_run_config(),
        backend=STAGE2_NATIVE_BACKEND,
    )
    assert full_sequence["execution_supported"] is True
    assert full_sequence["status"] == "supported_full_sequence_through_answer"
    assert full_sequence["supported_forward_mode"] == "no_kv_full_sequence"
    assert full_sequence["runtime_hooks"]["all_required_hooks_implemented"] is True
    assert full_sequence["runtime_hooks"]["hooks"][
        "capture_original_image_deepstack_features"
    ]["status"] == "ported"
    assert full_sequence["runtime_hooks"]["hooks"][
        "apply_post_tgvf_deepstack_scope_mask"
    ]["status"] == "ported"
    assert full_sequence["scope_contract"]["runtime_hooks"] == full_sequence["runtime_hooks"]

    evidence_only_config = RunConfig(
        run_id="stage2_deepstack_evidence",
        checkpoint_path="outputs/checkpoint.pt",
        mode=EvalMode.TGVF_FORCE,
        post_tgvf_forward_mode=ForwardMode.KV_CACHE,
        subset_id="core_smoke_256_seed20260625",
        deepstack=DeepStackState(
            enabled=True,
            original_image_scope=DeepStackScope.EVIDENCE_ONLY,
        ),
    )
    evidence_only = build_deepstack_execution_plan(
        evidence_only_config,
        backend=STAGE2_NATIVE_BACKEND,
    )
    assert evidence_only["original_image_scope"] == "evidence_only"
    assert evidence_only["original_image_deepstack"]["restore_for_answer"] is True
    assert evidence_only["runtime_hooks"]["hooks"][
        "restore_deepstack_for_answer_when_scope_requires"
    ]["required"] is True
    assert evidence_only["scope_contract"]["original_image_deepstack"][
        "block_query_end"
    ] == "answer_start"

    full_sequence_evidence = build_deepstack_execution_plan(
        _deepstack_full_sequence_evidence_run_config(),
        backend=STAGE2_NATIVE_BACKEND,
    )
    assert full_sequence_evidence["execution_supported"] is True
    assert full_sequence_evidence["status"] == "supported_full_sequence_evidence_only"
    assert full_sequence_evidence["supported_forward_mode"] == "no_kv_full_sequence"
    assert full_sequence_evidence["runtime_hooks"]["all_required_hooks_implemented"] is True
    assert full_sequence_evidence["runtime_hooks"]["hooks"][
        "restore_deepstack_for_answer_when_scope_requires"
    ]["status"] == "ported"
    assert full_sequence_evidence["scope_contract"]["runtime_hooks"] == (
        full_sequence_evidence["runtime_hooks"]
    )


def test_stage2_legacy_backend_rejects_unported_deepstack_execution(tmp_path) -> None:
    backend = make_backend(
        BackendConfig(backend=STAGE2_LEGACY_BACKEND, stage2=_runtime(tmp_path)),
        config=_diagnostic_run_config(),
    )

    with pytest.raises(NotImplementedError, match="DeepStack execution is not implemented"):
        diagnostic_deepstack = RunConfig(
            run_id="stage2_deepstack",
            checkpoint_path="outputs/checkpoint.pt",
            eval_family=EvalFamily.INTERNAL_DIAGNOSTIC,
            mode=EvalMode.TGVF_FORCE,
            post_tgvf_forward_mode=ForwardMode.KV_CACHE,
            subset_id="diagnostic_vstar_first_1_20260626",
            deepstack=DeepStackState(
                enabled=True,
                original_image_scope=DeepStackScope.THROUGH_ANSWER,
            ),
        )
        backend.prepare(diagnostic_deepstack)


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


def test_stage2_sample_from_clean_sample_materializes_embedded_base64(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("REVISIT_VLM_CLEAN_STAGE2_MEDIA_CACHE", str(tmp_path / "media_cache"))
    image_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ"
        "/pLvAAAAAElFTkSuQmCC"
    )
    sample = BenchmarkSample(
        sample_id="hr/sample/1",
        benchmark="hr_bench_4k",
        population_id="hr_bench_4k_800",
        source_file="hr_bench_4k/snapshot/hr_bench_4k.parquet",
        question="What is the number?",
        media=(
            {
                "kind": "embedded_base64",
                "source_key": "image",
                "value": base64.b64encode(image_bytes).decode("ascii"),
                "payload_loaded": True,
                "char_length": len(image_bytes),
            },
        ),
        gold_answer="7",
    )
    config = _run_config(EvalMode.TGVF_FREE)
    rendered = render_benchmark_input(sample, config)

    converted = stage2_sample_from_clean_sample(sample, rendered)

    assert isinstance(converted.image, str)
    assert converted.image.endswith(".png")
    assert (tmp_path / "media_cache").is_dir()
    assert converted.metadata["image_input_count"] == 1
    assert converted.metadata["image_input_mode"] == "single_image"
    assert converted.metadata["image_materialization"][0]["materialized"] is True
    assert converted.metadata["image_materialization"][0]["materialization_source"] == "embedded_base64"


def test_stage2_sample_from_clean_sample_materializes_multiple_image_structs(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("REVISIT_VLM_CLEAN_STAGE2_MEDIA_CACHE", str(tmp_path / "media_cache"))
    image_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ"
        "/pLvAAAAAElFTkSuQmCC"
    )
    sample = BenchmarkSample(
        sample_id="blink/sample/1",
        benchmark="blink",
        population_id="blink_val_all_subtasks_1901",
        source_file="blink/snapshot/Art_Style/val-00000-of-00001.parquet",
        question="Which image matches?",
        media=(
            {
                "kind": "image_struct",
                "source_key": "image_1",
                "path_hint": "a.png",
                "payload_loaded": True,
                "byte_length": len(image_bytes),
                "bytes": image_bytes,
            },
            {
                "kind": "image_struct",
                "source_key": "image_2",
                "path_hint": "b.png",
                "payload_loaded": True,
                "byte_length": len(image_bytes),
                "bytes": image_bytes,
            },
        ),
        choices=("A", "B"),
        gold_answer="A",
    )
    config = _run_config(EvalMode.TGVF_FREE)
    rendered = render_benchmark_input(sample, config)

    converted = stage2_sample_from_clean_sample(sample, rendered)

    assert isinstance(converted.image, list)
    assert len(converted.image) == 2
    assert all(Path(path).exists() for path in converted.image)
    assert converted.metadata["image_input_count"] == 2
    assert converted.metadata["image_input_mode"] == "multi_image"
    assert [item["source_key"] for item in converted.metadata["image_materialization"]] == [
        "image_1",
        "image_2",
    ]


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
    assert result.debug["uses_deepstack_for_fvt"] is False
    assert "does not provide" in result.debug["deepstack_caution"]


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


def test_stage2_native_deepstack_evidence_only_restores_attention_after_answer_boundary(
    tmp_path,
) -> None:
    import torch

    class FakeTokenizer:
        eos_token_id = 3
        text = {
            1: "</think>",
            2: "B",
            3: "<|im_end|>",
        }

        def encode(self, text, add_special_tokens=False):
            del text, add_special_tokens
            return []

        def decode(
            self,
            ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ):
            del skip_special_tokens, clean_up_tokenization_spaces
            return "".join(self.text.get(int(item), "") for item in ids)

    def logits_for(token_id: int) -> torch.Tensor:
        logits = torch.full((1, 1, 8), -1000.0)
        logits[0, 0, token_id] = 1000.0
        return logits

    class RecordingModel:
        def __init__(self) -> None:
            self.attention_dims = []
            self.param = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
            self.next_tokens = [2, 3, 3]

        def parameters(self):
            yield self.param

        def __call__(self, **kwargs):
            attention_mask = kwargs["attention_mask"]
            self.attention_dims.append(int(attention_mask.ndim))
            token_id = self.next_tokens.pop(0)
            return SimpleNamespace(
                past_key_values=kwargs.get("past_key_values"),
                logits=logits_for(token_id),
            )

    runtime = _runtime(
        tmp_path,
        append_forward_mode=ForwardMode.NO_KV_FULL_SEQUENCE,
    )
    engine = NativeStage2Engine(stage2_config=runtime)
    engine.model = RecordingModel()
    engine.processor = SimpleNamespace(tokenizer=FakeTokenizer())
    append_result = SimpleNamespace(
        last_logits=logits_for(1),
        past_key_values=object(),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        input_ids=torch.tensor([[10, 11]], dtype=torch.long),
        model_kwargs={
            "tgvf_next_position_ids": torch.zeros((3, 1, 1), dtype=torch.long),
            "tgvf_original_image_token_indices": torch.tensor([0], dtype=torch.long),
            "tgvf_deepstack_scope": "evidence_only",
            "tgvf_deepstack_restore_for_answer": True,
        },
    )

    continuation = engine._continue_generation_blocking_original_image_keys(append_result)

    assert continuation.generated_ids == [1, 2, 3]
    assert continuation.stop_reason == "eos_token"
    assert engine.model.attention_dims == [4, 2, 2]


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
        "uses_deepstack_for_fvt": False,
        "deepstack_caution": "fake append does not provide DeepStack features",
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
