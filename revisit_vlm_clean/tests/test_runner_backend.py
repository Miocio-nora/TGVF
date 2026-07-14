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
    Qwen3OriginalBackend,
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
    _validate_d_deepstack_checkpoint_support,
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


def _d_deepstack_run_config() -> RunConfig:
    return RunConfig(
        run_id="stage2_d_deepstack",
        checkpoint_path="outputs/checkpoint.pt",
        mode=EvalMode.TGVF_FORCE,
        post_tgvf_forward_mode=ForwardMode.KV_CACHE,
        subset_id="core_smoke_256_seed20260625",
        deepstack=DeepStackState(
            enabled=True,
            original_image_scope=DeepStackScope.THROUGH_ANSWER,
            d_features_enabled=True,
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


def test_stage2_native_unified_token_budget_tracks_remaining_answer_tokens(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    runtime = Stage2RuntimeConfig(
        stage2_checkpoint=runtime.stage2_checkpoint,
        eval_jsonl=runtime.eval_jsonl,
        append_forward_mode=runtime.append_forward_mode,
        max_tokens=5,
        max_action_tokens=64,
        max_answer_tokens=512,
    )
    engine = NativeStage2Engine(stage2_config=runtime)
    capture = SimpleNamespace(generated_ids=[1, 2, 3])
    continuation = SimpleNamespace(generated_ids=[4, 5])

    assert engine._action_token_budget() == 5
    assert engine._answer_token_budget(capture) == 2
    assert engine._token_budget_debug(capture, continuation) == {
        "schema_version": "clean_stage2_unified_token_budget_v1",
        "token_budget_policy": "unified_max_tokens",
        "max_tokens": 5,
        "legacy_split_limits": {
            "active": False,
            "max_action_tokens": 64,
            "max_answer_tokens": 512,
        },
        "action_budget": 5,
        "action_tokens": 3,
        "answer_budget": 2,
        "answer_tokens": 2,
        "total_generated_tokens": 5,
        "unified_budget_active": True,
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


def test_qwen3_original_backend_run_batch_splits_generated_outputs(monkeypatch, tmp_path) -> None:
    import torch

    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"not a real image; processor is mocked")
    samples = [
        BenchmarkSample(
            sample_id=f"sample-{index}",
            benchmark="vstar_bench",
            population_id="vstar_test_questions_191",
            source_file="vstar/test.jsonl",
            question=f"Question {index}?",
            media=({"kind": "path", "path": str(image_path), "exists": True},),
            choices=("A", "B"),
            gold_answer="A",
        )
        for index in range(2)
    ]
    config = _run_config(EvalMode.ORIGINAL)
    rendered = [render_benchmark_input(sample, config) for sample in samples]

    class FakeTokenizer:
        pad_token_id = 0

        def decode(self, ids, **kwargs):
            del kwargs
            return " ".join(str(item) for item in ids)

    class FakeProcessor:
        tokenizer = FakeTokenizer()

    class FakeModel:
        def generate(self, **kwargs):
            assert int(kwargs["input_ids"].shape[0]) == 2
            return torch.tensor(
                [
                    [1, 2, 3, 10, 11, 0],
                    [4, 5, 0, 12, 0, 0],
                ]
            )

    backend = Qwen3OriginalBackend(
        model_id="fake",
        processor_id=None,
        backend_config=BackendConfig(backend="qwen3_original", device="cpu", device_map=None),
        max_image_resolution=512,
        max_answer_tokens=8,
    )
    monkeypatch.setattr(backend, "_load", lambda: (FakeModel(), FakeProcessor()))
    monkeypatch.setattr(
        backend,
        "_build_batch_inputs",
        lambda processor, messages: {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
        },
    )

    results = backend.run_batch(samples, rendered, config)

    assert [result.raw_output for result in results] == ["10 11", "12"]
    assert [result.output_tokens for result in results] == [2, 1]
    assert [result.debug["batch_generation"]["batch_size"] for result in results] == [2, 2]


def test_qwen3_original_batch_inputs_force_left_padding(monkeypatch) -> None:
    import sys
    import types

    import torch

    fake_qwen_vl_utils = types.ModuleType("qwen_vl_utils")

    def fake_process_vision_info(*args, **kwargs):
        del args, kwargs
        return None, None, {}

    fake_qwen_vl_utils.process_vision_info = fake_process_vision_info
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", fake_qwen_vl_utils)

    class FakeTokenizer:
        padding_side = "right"

    class FakeProcessor:
        tokenizer = FakeTokenizer()

        def apply_chat_template(self, messages, **kwargs):
            del kwargs
            return messages[0]["content"][0]["text"]

        def __call__(self, **kwargs):
            assert self.tokenizer.padding_side == "left"
            assert kwargs["padding"] is True
            return {
                "input_ids": torch.tensor([[1, 2], [0, 3]]),
                "attention_mask": torch.ones(2, 2, dtype=torch.long),
            }

    backend = Qwen3OriginalBackend(
        model_id="fake",
        processor_id=None,
        backend_config=BackendConfig(backend="qwen3_original", device="cpu", device_map=None),
        max_image_resolution=512,
        max_answer_tokens=8,
    )

    inputs = backend._build_batch_inputs(
        FakeProcessor(),
        [
            [{"role": "user", "content": [{"type": "text", "text": "one"}]}],
            [{"role": "user", "content": [{"type": "text", "text": "two"}]}],
        ],
    )

    assert inputs["input_ids"].shape == (2, 2)


def test_qwen3_original_no_thinking_blocks_think_tokens(monkeypatch, tmp_path) -> None:
    import torch

    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"not a real image; processor is mocked")
    sample = BenchmarkSample(
        sample_id="sample",
        benchmark="vstar_bench",
        population_id="vstar_test_questions_191",
        source_file="vstar/test.jsonl",
        question="Question?",
        media=({"kind": "path", "path": str(image_path), "exists": True},),
        choices=("A", "B"),
        gold_answer="A",
    )
    config = _run_config(EvalMode.ORIGINAL)
    rendered = render_benchmark_input(sample, config)

    class FakeTokenizer:
        pad_token_id = 0

        def encode(self, text, **kwargs):
            del kwargs
            return {"<think>": [101], "</think>": [102]}.get(text, [])

        def decode(self, ids, **kwargs):
            del kwargs
            return " ".join(str(item) for item in ids)

    class FakeProcessor:
        tokenizer = FakeTokenizer()

    class FakeModel:
        def generate(self, **kwargs):
            assert kwargs["bad_words_ids"] == [[101], [102]]
            return torch.tensor([[1, 2, 10, 0]])

    backend = Qwen3OriginalBackend(
        model_id="fake",
        processor_id=None,
        backend_config=BackendConfig(
            backend="qwen3_original",
            device="cpu",
            device_map=None,
            original_no_thinking=True,
        ),
        max_image_resolution=512,
        max_answer_tokens=8,
    )
    monkeypatch.setattr(backend, "_load", lambda: (FakeModel(), FakeProcessor()))
    monkeypatch.setattr(
        backend,
        "_build_batch_inputs",
        lambda processor, messages: {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        },
    )

    result = backend.run(sample, rendered, config)

    assert result.raw_output == "10"
    assert result.debug["original_generation"]["no_thinking"] is True


def test_qwen3_original_no_thinking_strips_think_prefill(monkeypatch) -> None:
    import sys
    import types

    import torch

    fake_qwen_vl_utils = types.ModuleType("qwen_vl_utils")

    def fake_process_vision_info(*args, **kwargs):
        del args, kwargs
        return None, None, {}

    fake_qwen_vl_utils.process_vision_info = fake_process_vision_info
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", fake_qwen_vl_utils)

    class FakeTokenizer:
        padding_side = "right"

    class FakeProcessor:
        tokenizer = FakeTokenizer()

        def apply_chat_template(self, messages, **kwargs):
            del messages
            if "enable_thinking" in kwargs:
                raise TypeError("old processor")
            return "prompt<|im_start|>assistant\n<think>\n"

        def __call__(self, **kwargs):
            assert kwargs["text"] == ["prompt<|im_start|>assistant\n"]
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.ones(1, 2, dtype=torch.long),
            }

    backend = Qwen3OriginalBackend(
        model_id="fake",
        processor_id=None,
        backend_config=BackendConfig(
            backend="qwen3_original",
            device="cpu",
            device_map=None,
            original_no_thinking=True,
        ),
        max_image_resolution=512,
        max_answer_tokens=8,
    )

    inputs = backend._build_batch_inputs(
        FakeProcessor(),
        [[{"role": "user", "content": [{"type": "text", "text": "one"}]}]],
    )

    assert inputs["input_ids"].shape == (1, 2)


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


def test_stage2_native_backend_cleans_cache_after_success(monkeypatch, tmp_path) -> None:
    runtime = _runtime(tmp_path)
    config = _run_config()

    class FakeNativeEngine:
        instances = []

        def __init__(self, *, stage2_config, backend_options) -> None:
            self.stage2_config = stage2_config
            self.backend_options = backend_options
            self.cleaned = False
            FakeNativeEngine.instances.append(self)

        def prepare(self, config) -> None:
            self.prepared_config = config

        def run(self, sample, rendered, config) -> NativeStage2RunResult:
            return NativeStage2RunResult(raw_output="A", debug={"ok": True})

        def cleanup_after_row(self) -> dict:
            self.cleaned = True
            return {"vision_cache_entries_cleared": 1}

    monkeypatch.setattr(runner_module, "NativeStage2Engine", FakeNativeEngine)
    backend = make_backend(
        BackendConfig(backend=STAGE2_NATIVE_BACKEND, stage2=runtime),
        config=config,
    )

    backend.prepare(config)
    result = backend.run(_sample(), render_benchmark_input(_sample(), config), config)

    assert result.error is None
    assert FakeNativeEngine.instances[-1].cleaned is True
    assert result.debug["native_stage2_recovery"] == {"vision_cache_entries_cleared": 1}


def test_stage2_native_backend_recovers_after_cuda_error(monkeypatch, tmp_path) -> None:
    runtime = _runtime(tmp_path)
    config = _run_config()

    class FakeNativeEngine:
        instances = []

        def __init__(self, *, stage2_config, backend_options) -> None:
            self.stage2_config = stage2_config
            self.backend_options = backend_options
            self.recovered_error = None
            FakeNativeEngine.instances.append(self)

        def prepare(self, config) -> None:
            self.prepared_config = config

        def run(self, sample, rendered, config) -> NativeStage2RunResult:
            return NativeStage2RunResult(
                raw_output="",
                error="OutOfMemoryError: CUDA out of memory",
                debug={"failed": True},
            )

        def recover_after_fatal_error(self, error: str) -> dict:
            self.recovered_error = error
            return {"fatal_cuda_error": True, "runtime_unloaded": True}

    monkeypatch.setattr(runner_module, "NativeStage2Engine", FakeNativeEngine)
    backend = make_backend(
        BackendConfig(backend=STAGE2_NATIVE_BACKEND, stage2=runtime),
        config=config,
    )

    backend.prepare(config)
    result = backend.run(_sample(), render_benchmark_input(_sample(), config), config)

    assert result.error == "OutOfMemoryError: CUDA out of memory"
    assert FakeNativeEngine.instances[-1].recovered_error == result.error
    assert result.debug["native_stage2_recovery"] == {
        "fatal_cuda_error": True,
        "runtime_unloaded": True,
    }


def test_native_stage2_engine_cleanup_clears_per_sample_caches(tmp_path) -> None:
    engine = NativeStage2Engine(
        stage2_config=_runtime(tmp_path),
        backend_options={},
    )
    engine.vision_cache["image"] = ("tap", "v_pre", "v_merge")
    engine.deepstack_cache["image"] = ["feature"]

    report = engine.cleanup_after_row()

    assert report["vision_cache_entries_cleared"] == 1
    assert report["deepstack_cache_entries_cleared"] == 1
    assert engine.vision_cache == {}
    assert engine.deepstack_cache == {}


def test_stage2_native_backend_accepts_supported_kv_deepstack(tmp_path) -> None:
    config = _deepstack_run_config()
    backend = make_backend(
        BackendConfig(backend=STAGE2_NATIVE_BACKEND, stage2=_runtime(tmp_path)),
        config=config,
    )

    backend.prepare(config)


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


def test_d_deepstack_runtime_request_requires_matching_checkpoint_config() -> None:
    _validate_d_deepstack_checkpoint_support(
        _deepstack_run_config(),
        {"tgvf": {"d_deepstack_enabled": False}},
    )
    _validate_d_deepstack_checkpoint_support(
        _d_deepstack_run_config(),
        {"tgvf": {"d_deepstack_enabled": True}},
    )
    with pytest.raises(ValueError, match="d_deepstack_enabled=true"):
        _validate_d_deepstack_checkpoint_support(
            _d_deepstack_run_config(),
            {"tgvf": {"d_deepstack_enabled": False}},
        )


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
    assert through_answer["execution_supported"] is True
    assert through_answer["status"] == "supported_kv_cache_through_answer"
    assert through_answer["supported_forward_mode"] == "kv_cache"
    assert through_answer["original_image_scope"] == "through_answer"
    assert through_answer["runtime_hooks"]["all_required_hooks_implemented"] is True
    assert through_answer["runtime_hooks"]["hooks"][
        "capture_original_image_deepstack_features"
    ]["status"] == "ported"
    assert through_answer["runtime_hooks"]["hooks"][
        "apply_post_tgvf_deepstack_scope_mask"
    ]["required"] is True
    assert through_answer["runtime_hooks"]["hooks"][
        "restore_deepstack_for_answer_when_scope_requires"
    ]["required"] is False
    assert through_answer["blocking_items"] == []
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

    no_block_config = RunConfig(
        run_id="stage2_deepstack_no_block",
        checkpoint_path="outputs/checkpoint.pt",
        mode=EvalMode.TGVF_FORCE,
        post_tgvf_forward_mode=ForwardMode.KV_CACHE,
        subset_id="core_smoke_256_seed20260625",
        deepstack=DeepStackState(
            enabled=True,
            original_image_scope=DeepStackScope.NO_BLOCK,
        ),
    )
    no_block = build_deepstack_execution_plan(
        no_block_config,
        backend=STAGE2_NATIVE_BACKEND,
    )
    assert no_block["execution_supported"] is True
    assert no_block["status"] == "supported_kv_cache_no_block"
    assert no_block["original_image_scope"] == "no_block"
    assert no_block["original_image_deepstack"]["block_after_tgvf_append"] is False
    assert no_block["original_image_deepstack"]["restore_for_answer"] is False
    assert no_block["runtime_hooks"]["hooks"][
        "apply_post_tgvf_deepstack_scope_mask"
    ]["required"] is False

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

    full_sequence_no_block = build_deepstack_execution_plan(
        RunConfig(
            run_id="stage2_deepstack_full_no_block",
            checkpoint_path="outputs/checkpoint.pt",
            mode=EvalMode.TGVF_FORCE,
            post_tgvf_forward_mode=ForwardMode.NO_KV_FULL_SEQUENCE,
            subset_id="core_smoke_256_seed20260625",
            deepstack=DeepStackState(
                enabled=True,
                original_image_scope=DeepStackScope.NO_BLOCK,
            ),
        ),
        backend=STAGE2_NATIVE_BACKEND,
    )
    assert full_sequence_no_block["execution_supported"] is True
    assert full_sequence_no_block["status"] == "supported_full_sequence_no_block"
    assert full_sequence_no_block["runtime_hooks"]["hooks"][
        "apply_post_tgvf_deepstack_scope_mask"
    ]["status"] == "not_required"

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
    assert evidence_only["execution_supported"] is True
    assert evidence_only["status"] == "supported_kv_cache_evidence_only"
    assert evidence_only["supported_forward_mode"] == "kv_cache"
    assert evidence_only["original_image_scope"] == "evidence_only"
    assert evidence_only["original_image_deepstack"]["restore_for_answer"] is True
    assert evidence_only["runtime_hooks"]["hooks"][
        "restore_deepstack_for_answer_when_scope_requires"
    ]["status"] == "ported"
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


def test_stage2_sample_from_clean_sample_coalesces_decoded_image_duplicate(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("REVISIT_VLM_CLEAN_STAGE2_MEDIA_CACHE", str(tmp_path / "media_cache"))
    image_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ"
        "/pLvAAAAAElFTkSuQmCC"
    )
    image_path = tmp_path / "images" / "555.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(image_bytes)
    sample = BenchmarkSample(
        sample_id="mathvista/sample/555",
        benchmark="mathvista",
        population_id="mathvista_testmini_1000",
        source_file="mathvista/snapshot/data/testmini.parquet",
        question="Is this nest larger than a fist?",
        media=(
            {
                "kind": "path",
                "source_key": "image",
                "path": str(image_path),
                "exists": True,
                "path_hint": "images/555.jpg",
            },
            {
                "kind": "image_struct",
                "source_key": "decoded_image",
                "path_hint": "555.jpg",
                "payload_loaded": True,
                "byte_length": len(image_bytes),
                "bytes": image_bytes,
            },
        ),
        choices=("Yes", "No"),
        gold_answer="No",
    )
    config = _run_config(EvalMode.TGVF_FREE)
    rendered = render_benchmark_input(sample, config)

    converted = stage2_sample_from_clean_sample(sample, rendered)

    assert converted.image == str(image_path)
    assert converted.metadata["image_input_count"] == 1
    assert converted.metadata["image_input_mode"] == "single_image"
    assert [item["source_key"] for item in converted.metadata["image_materialization"]] == [
        "image"
    ]
    assert not (tmp_path / "media_cache").exists()


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
            "tgvf_block_original_image_keys": True,
            "tgvf_next_position_ids": torch.zeros((3, 1, 1), dtype=torch.long),
            "tgvf_original_image_token_indices": torch.tensor([0], dtype=torch.long),
            "tgvf_deepstack_scope": "evidence_only",
            "tgvf_deepstack_restore_for_answer": True,
        },
    )

    continuation = engine._continue_generation_blocking_original_image_keys(append_result)

    assert continuation.generated_ids == [1, 2, 3]
    assert len(continuation.generated_logprobs) == 3
    assert continuation.stop_reason == "eos_token"
    assert engine.model.attention_dims == [4, 2, 2]


def test_stage2_native_deepstack_teacher_forced_replay_masks_like_decode(tmp_path) -> None:
    import torch

    class FakeTokenizer:
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

    runtime = _runtime(tmp_path, append_forward_mode=ForwardMode.KV_CACHE)
    engine = NativeStage2Engine(stage2_config=runtime)
    engine.model = RecordingModel()
    engine.processor = SimpleNamespace(tokenizer=FakeTokenizer())
    initial_logits = logits_for(1).requires_grad_()
    append_result = SimpleNamespace(
        last_logits=initial_logits,
        past_key_values=object(),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        input_ids=torch.tensor([[10, 11]], dtype=torch.long),
        model_kwargs={
            "tgvf_block_original_image_keys": True,
            "tgvf_next_position_ids": torch.zeros((3, 1, 1), dtype=torch.long),
            "tgvf_original_image_token_indices": torch.tensor([0], dtype=torch.long),
            "tgvf_deepstack_scope": "evidence_only",
            "tgvf_deepstack_restore_for_answer": True,
        },
    )

    logprobs = engine.teacher_forced_continue_logprobs(
        append_result,
        generated_token_ids=[1, 2, 3],
    )
    (-logprobs.sum()).backward()

    assert tuple(logprobs.shape) == (3,)
    assert initial_logits.grad is not None
    assert engine.model.attention_dims == [4, 2, 2]


def test_stage2_native_deepstack_blocked_decode_uses_sampling_options(
    monkeypatch,
    tmp_path,
) -> None:
    import torch
    import revisit_vlm.qwen3_vl_tgvf as qwen3_vl_tgvf

    class FakeTokenizer:
        eos_token_id = 99

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
            return "".join(str(int(item)) for item in ids)

    class RecordingModel:
        def __init__(self) -> None:
            self.param = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))

        def parameters(self):
            yield self.param

        def __call__(self, **kwargs):
            return SimpleNamespace(
                past_key_values=kwargs.get("past_key_values"),
                logits=torch.zeros((1, 1, 8), dtype=torch.float32),
            )

    calls = []

    def fake_select_next_token(logits, *, do_sample, temperature, top_p):
        calls.append(
            {
                "shape": tuple(logits.shape),
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        return torch.tensor([[4]], dtype=torch.long), -0.5

    monkeypatch.setattr(qwen3_vl_tgvf, "_select_next_token", fake_select_next_token)
    runtime = _runtime(tmp_path, append_forward_mode=ForwardMode.KV_CACHE)
    runtime = Stage2RuntimeConfig(
        stage2_checkpoint=runtime.stage2_checkpoint,
        eval_jsonl=runtime.eval_jsonl,
        append_forward_mode=runtime.append_forward_mode,
        max_answer_tokens=1,
    )
    engine = NativeStage2Engine(
        stage2_config=runtime,
        backend_options={"do_sample": True, "temperature": 0.7, "top_p": 0.8},
    )
    engine.model = RecordingModel()
    engine.processor = SimpleNamespace(tokenizer=FakeTokenizer())
    append_result = SimpleNamespace(
        last_logits=torch.zeros((1, 1, 8), dtype=torch.float32),
        past_key_values=object(),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        input_ids=torch.tensor([[10, 11]], dtype=torch.long),
        model_kwargs={
            "tgvf_next_position_ids": torch.zeros((3, 1, 1), dtype=torch.long),
            "tgvf_original_image_token_indices": torch.tensor([0], dtype=torch.long),
            "tgvf_deepstack_scope": "through_answer",
            "tgvf_deepstack_restore_for_answer": False,
        },
    )

    continuation = engine._continue_generation_blocking_original_image_keys(append_result)

    assert continuation.generated_ids == [4]
    assert continuation.generated_logprobs == [-0.5]
    assert calls == [{"shape": (1, 8), "do_sample": True, "temperature": 0.7, "top_p": 0.8}]


def test_stage2_native_kv_deepstack_append_uses_cached_chunk_mask(
    monkeypatch,
    tmp_path,
) -> None:
    import torch
    import revisit_vlm.qwen3_vl_tgvf as qwen3_vl_tgvf

    class FakeEmbedding:
        weight = torch.zeros((32, 4), dtype=torch.float32)

        def __call__(self, token_ids):
            return torch.zeros((*token_ids.shape, 4), dtype=torch.float32)

    class RecordingModel:
        def __init__(self) -> None:
            self.calls = []
            self.embed = FakeEmbedding()

        def get_input_embeddings(self):
            return self.embed

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                past_key_values="new-cache",
                logits=torch.zeros((1, 1, 8), dtype=torch.float32),
            )

    monkeypatch.setattr(
        qwen3_vl_tgvf,
        "render_tgvf_prefix_suffix",
        lambda **kwargs: ("P", "S"),
    )
    monkeypatch.setattr(qwen3_vl_tgvf, "protocol_uses_tool_observation", lambda protocol: True)
    monkeypatch.setattr(qwen3_vl_tgvf, "protocol_uses_think_tags", lambda protocol: False)
    monkeypatch.setattr(qwen3_vl_tgvf, "protocol_uses_evidence_tags", lambda protocol: False)
    monkeypatch.setattr(
        qwen3_vl_tgvf,
        "_bracketed_visual_token_ids",
        lambda *args, **kwargs: torch.tensor([10, 11, 12, 13], dtype=torch.long),
    )
    monkeypatch.setattr(
        qwen3_vl_tgvf,
        "_encode_text",
        lambda tokenizer, text, device: torch.empty((0,), dtype=torch.long),
    )
    monkeypatch.setattr(
        qwen3_vl_tgvf,
        "_fvt_mm_token_type_ids",
        lambda *, chunk_length, fvt_token_start, fvt_token_end, device: torch.zeros(
            (1, chunk_length),
            dtype=torch.long,
        ),
    )
    monkeypatch.setattr(
        qwen3_vl_tgvf,
        "_chunk_position_ids_native_source_grid",
        lambda **kwargs: torch.zeros((3, 1, 4), dtype=torch.long),
    )
    monkeypatch.setattr(
        qwen3_vl_tgvf,
        "_next_position_ids_after_prefill",
        lambda position_ids: torch.zeros((3, 1, 1), dtype=torch.long),
    )

    runtime = _runtime(tmp_path, append_forward_mode=ForwardMode.KV_CACHE)
    config = _deepstack_run_config()
    engine = NativeStage2Engine(stage2_config=runtime)
    engine.prepare(config)
    model = RecordingModel()
    engine.model = model
    engine.utility_model = model
    engine.processor = SimpleNamespace(tokenizer=object())
    engine.device = torch.device("cpu")
    capture = SimpleNamespace(
        capture_found=True,
        past_key_values="prefix-cache",
        attention_mask=torch.ones((1, 5), dtype=torch.long),
        input_ids=torch.arange(5, dtype=torch.long).view(1, -1),
        model_kwargs={},
        source_visual_geometry=SimpleNamespace(
            source_visual_token_count=2,
            source_visual_token_indices=torch.tensor([1, 2], dtype=torch.long),
            source_visual_position_ids=torch.zeros((3, 2), dtype=torch.long),
        ),
    )

    result = engine._append_visual_d(capture, torch.ones((2, 4), dtype=torch.float32))

    call = model.calls[0]
    attention_mask = call["attention_mask"]
    blocked = torch.finfo(torch.float32).min
    assert list(attention_mask.shape) == [1, 1, 4, 9]
    assert attention_mask[0, 0, 0, 1].item() == blocked
    assert attention_mask[0, 0, 3, 2].item() == blocked
    assert attention_mask[0, 0, 0, 6].item() == blocked
    assert attention_mask[0, 0, 1, 6].item() == 0.0
    assert result.attention_mask.shape == (1, 9)
    assert result.model_kwargs["tgvf_block_original_image_keys"] is True
    assert result.model_kwargs["tgvf_original_image_token_indices"].tolist() == [1, 2]
    assert result.model_kwargs["tgvf_deepstack_scope"] == "through_answer"
    assert result.model_kwargs["tgvf_deepstack_restore_for_answer"] is False
    assert result.debug_metadata["uses_deepstack_for_fvt"] is True
    assert result.debug_metadata["deepstack_caution"] is None
    assert result.debug_metadata["deepstack_prefix_source"] == (
        "native_qwen3_capture_past_key_values"
    )


def test_stage2_native_kv_deepstack_no_block_keeps_original_image_keys_visible(
    monkeypatch,
    tmp_path,
) -> None:
    import torch
    import revisit_vlm.qwen3_vl_tgvf as qwen3_vl_tgvf

    class FakeEmbedding:
        weight = torch.zeros((32, 4), dtype=torch.float32)

        def __call__(self, token_ids):
            return torch.zeros((*token_ids.shape, 4), dtype=torch.float32)

    class RecordingModel:
        def __init__(self) -> None:
            self.calls = []
            self.embed = FakeEmbedding()

        def get_input_embeddings(self):
            return self.embed

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                past_key_values="new-cache",
                logits=torch.zeros((1, 1, 8), dtype=torch.float32),
            )

    monkeypatch.setattr(qwen3_vl_tgvf, "render_tgvf_prefix_suffix", lambda **kwargs: ("P", "S"))
    monkeypatch.setattr(qwen3_vl_tgvf, "protocol_uses_tool_observation", lambda protocol: True)
    monkeypatch.setattr(qwen3_vl_tgvf, "protocol_uses_think_tags", lambda protocol: False)
    monkeypatch.setattr(qwen3_vl_tgvf, "protocol_uses_evidence_tags", lambda protocol: False)
    monkeypatch.setattr(
        qwen3_vl_tgvf,
        "_bracketed_visual_token_ids",
        lambda *args, **kwargs: torch.tensor([10, 11, 12, 13], dtype=torch.long),
    )
    monkeypatch.setattr(
        qwen3_vl_tgvf,
        "_encode_text",
        lambda tokenizer, text, device: torch.empty((0,), dtype=torch.long),
    )
    monkeypatch.setattr(
        qwen3_vl_tgvf,
        "_fvt_mm_token_type_ids",
        lambda *, chunk_length, fvt_token_start, fvt_token_end, device: torch.zeros(
            (1, chunk_length),
            dtype=torch.long,
        ),
    )
    monkeypatch.setattr(
        qwen3_vl_tgvf,
        "_chunk_position_ids_native_source_grid",
        lambda **kwargs: torch.zeros((3, 1, 4), dtype=torch.long),
    )
    monkeypatch.setattr(
        qwen3_vl_tgvf,
        "_next_position_ids_after_prefill",
        lambda position_ids: torch.zeros((3, 1, 1), dtype=torch.long),
    )

    runtime = _runtime(tmp_path, append_forward_mode=ForwardMode.KV_CACHE)
    config = RunConfig(
        run_id="stage2_deepstack_no_block",
        checkpoint_path="outputs/checkpoint.pt",
        mode=EvalMode.TGVF_FORCE,
        post_tgvf_forward_mode=ForwardMode.KV_CACHE,
        subset_id="core_smoke_256_seed20260625",
        deepstack=DeepStackState(
            enabled=True,
            original_image_scope=DeepStackScope.NO_BLOCK,
        ),
    )
    engine = NativeStage2Engine(stage2_config=runtime)
    engine.prepare(config)
    model = RecordingModel()
    engine.model = model
    engine.utility_model = model
    engine.processor = SimpleNamespace(tokenizer=object())
    engine.device = torch.device("cpu")
    capture = SimpleNamespace(
        capture_found=True,
        past_key_values="prefix-cache",
        attention_mask=torch.ones((1, 5), dtype=torch.long),
        input_ids=torch.arange(5, dtype=torch.long).view(1, -1),
        model_kwargs={},
        source_visual_geometry=SimpleNamespace(
            source_visual_token_count=2,
            source_visual_token_indices=torch.tensor([1, 2], dtype=torch.long),
            source_visual_position_ids=torch.zeros((3, 2), dtype=torch.long),
        ),
    )

    result = engine._append_visual_d(capture, torch.ones((2, 4), dtype=torch.float32))

    attention_mask = model.calls[0]["attention_mask"]
    assert list(attention_mask.shape) == [1, 9]
    assert "tgvf_block_original_image_keys" not in result.model_kwargs
    assert result.debug_metadata["uses_deepstack_for_fvt"] is True
    assert result.debug_metadata["deepstack_scope"] == "no_block"
    assert result.debug_metadata["deepstack_answer_restore_policy"] == "not_blocked"
    assert result.debug_metadata["deepstack_append_attention_mask"] == (
        "2d_no_original_image_key_block"
    )
    assert result.debug_metadata["deepstack_original_image_key_block"] is False


def test_stage2_native_retries_full_sequence_for_multi_image_kv_position_failure(
    monkeypatch,
    tmp_path,
) -> None:
    import torch

    runtime = _runtime(tmp_path, append_forward_mode=ForwardMode.KV_CACHE)
    engine = NativeStage2Engine(stage2_config=runtime)
    sample = SimpleNamespace(
        image="image-a.jpg",
        image_id="sample",
        source_dataset="blink",
        source_profile="blink_val_all_subtasks_1901",
        need_focus=True,
        question="Question?",
        prompt_question="Question?",
        target="target",
        evidence_description="",
        answer="A",
        answer_format="multiple_choice",
        evidence_type="visual",
        target_style="object",
        target_cues=[],
    )
    capture = SimpleNamespace(
        generated_text="<think>\n<|focus_start|>target<|focus_end|>",
        generated_ids=[1, 2, 3],
        generated_logprobs=[],
        capture_found=True,
        malformed=False,
        target_text="target",
        target_token_ids=[2],
        target_hidden_states=torch.zeros((1, 4)),
        source_visual_geometry=SimpleNamespace(
            image_grid_thw=torch.tensor([[1, 1, 2], [1, 1, 3]], dtype=torch.long),
            source_visual_token_count=5,
        ),
    )
    parsed = SimpleNamespace(
        evidence_state=None,
        focus_target="target",
        malformed=False,
        answer="A",
        answer_valid=True,
    )
    append_result = SimpleNamespace(
        debug_metadata={
            "fvt_shape": [5, 4],
            "fvt_append_path": "clean_native_full_sequence_prefill",
            "fvt_position_mode": "multi_image_inherit_source_visual_positions",
            "uses_deepstack_for_fvt": True,
            "deepstack_caution": None,
            "second_full_forward_used": True,
        },
    )
    calls = {"kv": 0, "full": 0}

    def fake_kv_append(capture_arg, d_arg):
        del capture_arg, d_arg
        calls["kv"] += 1
        raise RuntimeError(
            "shape mismatch: value tensor of shape [3, 804] cannot be broadcast "
            "to indexing result of shape [3, 1032]"
        )

    def fake_full_append(sample_arg, capture_arg, d_arg):
        del sample_arg, capture_arg, d_arg
        calls["full"] += 1
        return append_result

    monkeypatch.setattr(engine, "_append_visual_d", fake_kv_append)
    monkeypatch.setattr(engine, "_append_visual_d_full_sequence", fake_full_append)
    monkeypatch.setattr(
        engine,
        "_continue_generation",
        lambda append_result_arg, *, capture: SimpleNamespace(
            generated_text="A",
            generated_ids=[4],
            generated_logprobs=[],
        ),
    )
    monkeypatch.setattr(engine, "_parse_action", lambda _text: parsed)

    result = engine._run_post_tgvf_condition(
        sample=sample,
        capture=capture,
        correct_d=torch.zeros((5, 4)),
        block="unit",
        focus_source="unit",
    )

    assert result.error is None
    assert result.append_success is True
    assert calls == {"kv": 1, "full": 1}
    assert result.debug["second_full_forward_used"] is True
    assert result.debug["mask_mode"] == "clean_native_full_sequence_prefill"
    assert result.debug["kv_cache_append_retry_reason"] == (
        "multi_image_native_source_grid_position_ids"
    )


def test_stage2_native_kv_deepstack_prefills_generate_cache_tail(
    monkeypatch,
    tmp_path,
) -> None:
    import torch
    import revisit_vlm.qwen3_vl_tgvf as qwen3_vl_tgvf

    class FakeCache:
        def __init__(self, seq_len: int) -> None:
            self.seq_len = seq_len

        def get_seq_length(self):
            return self.seq_len

    class FakeEmbedding:
        weight = torch.zeros((32, 4), dtype=torch.float32)

        def __call__(self, token_ids):
            return torch.zeros((*token_ids.shape, 4), dtype=torch.float32)

    class RecordingModel:
        def __init__(self) -> None:
            self.calls = []
            self.embed = FakeEmbedding()

        def get_input_embeddings(self):
            return self.embed

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                past_key_values=FakeCache(9),
                logits=torch.zeros((1, 1, 8), dtype=torch.float32),
            )

    monkeypatch.setattr(
        qwen3_vl_tgvf,
        "render_tgvf_prefix_suffix",
        lambda **kwargs: ("P", "S"),
    )
    monkeypatch.setattr(qwen3_vl_tgvf, "protocol_uses_tool_observation", lambda protocol: True)
    monkeypatch.setattr(qwen3_vl_tgvf, "protocol_uses_think_tags", lambda protocol: False)
    monkeypatch.setattr(qwen3_vl_tgvf, "protocol_uses_evidence_tags", lambda protocol: False)
    monkeypatch.setattr(
        qwen3_vl_tgvf,
        "_bracketed_visual_token_ids",
        lambda *args, **kwargs: torch.tensor([10, 11, 12, 13], dtype=torch.long),
    )
    monkeypatch.setattr(
        qwen3_vl_tgvf,
        "_encode_text",
        lambda tokenizer, text, device: torch.empty((0,), dtype=torch.long),
    )
    monkeypatch.setattr(
        qwen3_vl_tgvf,
        "_fvt_mm_token_type_ids",
        lambda *, chunk_length, fvt_token_start, fvt_token_end, device: torch.zeros(
            (1, chunk_length),
            dtype=torch.long,
        ),
    )
    monkeypatch.setattr(
        qwen3_vl_tgvf,
        "_chunk_position_ids_native_source_grid",
        lambda **kwargs: torch.zeros((3, 1, 4), dtype=torch.long),
    )
    monkeypatch.setattr(
        qwen3_vl_tgvf,
        "_compute_qwen3_position_ids_for_sequence",
        lambda **kwargs: torch.zeros((3, 1, 9), dtype=torch.long),
    )
    monkeypatch.setattr(
        qwen3_vl_tgvf,
        "_next_position_ids_after_prefill",
        lambda position_ids: torch.zeros((3, 1, 1), dtype=torch.long),
    )

    runtime = _runtime(tmp_path, append_forward_mode=ForwardMode.KV_CACHE)
    config = _deepstack_run_config()
    engine = NativeStage2Engine(stage2_config=runtime)
    engine.prepare(config)
    model = RecordingModel()
    engine.model = model
    engine.utility_model = model
    engine.processor = SimpleNamespace(tokenizer=object())
    engine.device = torch.device("cpu")
    capture = SimpleNamespace(
        capture_found=True,
        past_key_values=FakeCache(4),
        attention_mask=torch.ones((1, 5), dtype=torch.long),
        input_ids=torch.arange(5, dtype=torch.long).view(1, -1),
        cache_position=None,
        model_kwargs={},
        image_grid_thw=None,
        video_grid_thw=None,
        source_visual_geometry=SimpleNamespace(
            source_visual_token_count=2,
            source_visual_token_indices=torch.tensor([1, 2], dtype=torch.long),
            source_visual_position_ids=torch.zeros((3, 2), dtype=torch.long),
            image_grid_thw=torch.tensor([[1, 1, 2]], dtype=torch.long),
        ),
    )

    result = engine._append_visual_d(capture, torch.ones((2, 4), dtype=torch.float32))

    assert len(model.calls) == 1
    assert "inputs_embeds" in model.calls[0]
    assert "input_ids" not in model.calls[0]
    assert model.calls[0]["past_key_values"].get_seq_length() == 4
    assert list(model.calls[0]["inputs_embeds"].shape) == [1, 5, 4]
    attention_mask = model.calls[0]["attention_mask"]
    blocked = torch.finfo(torch.float32).min
    assert list(attention_mask.shape) == [1, 1, 5, 9]
    assert attention_mask[0, 0, 0, 1].item() == 0.0
    assert attention_mask[0, 0, 1, 1].item() == blocked
    assert attention_mask[0, 0, 0, 5].item() == blocked
    assert attention_mask[0, 0, 1, 5].item() == 0.0
    assert result.debug_metadata["kv_cache_initial_seq_len"] == 4
    assert result.debug_metadata["kv_cache_input_len"] == 5
    assert result.debug_metadata["kv_cache_tail_prefill_tokens"] == 1
    assert result.debug_metadata["kv_cache_aligned_seq_len"] == 4
    assert result.debug_metadata["kv_cache_tail_prefill_used"] is False
    assert result.debug_metadata["kv_cache_tail_in_append_chunk"] is True
    assert result.debug_metadata["kv_cache_tail_chunk_tokens"] == 1


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

    def _continue_generation(self, append_result, *, capture=None):
        del append_result, capture
        self.calls.append("continue")
        return _FakeContinuation()

    def _parse_action(self, text):
        del text
        self.calls.append("parse")
        return _FakeParsed()
