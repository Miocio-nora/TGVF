from revisit_vlm_clean.benchmark_data import BenchmarkSample
from revisit_vlm_clean.rendering import (
    PROTOCOL_C_FOCUS_START,
    PROTOCOL_C_TOOL_OBSERVATION_QWEN2_NO_THINK,
    protocol_render_spec,
    render_benchmark_input,
)
from revisit_vlm_clean.schema import EvalMode, ForwardMode, RunConfig


def _sample() -> BenchmarkSample:
    return BenchmarkSample(
        sample_id="sample-1",
        benchmark="vstar_bench",
        population_id="vstar_test_questions_191",
        source_file="vstar_bench/snapshot/test_questions.jsonl",
        question="What color is the toy?\n(A) red\n(B) blue",
        media=({"kind": "path", "path": "/tmp/toy.jpg", "exists": True},),
        choices=("red", "blue"),
        gold_answer="B",
        metadata={"category": "toy"},
    )


def _config(mode: EvalMode, **kwargs) -> RunConfig:
    return RunConfig(
        run_id=f"render-{mode.value}",
        checkpoint_path="outputs/checkpoint.pt",
        mode=mode,
        post_tgvf_forward_mode=ForwardMode.KV_CACHE,
        subset_id="core_smoke_256_seed20260625",
        **kwargs,
    )


def test_original_and_free_do_not_add_prompt_text() -> None:
    sample = _sample()
    original = render_benchmark_input(sample, _config(EvalMode.ORIGINAL))
    free = render_benchmark_input(sample, _config(EvalMode.TGVF_FREE))

    assert original.user_prompt == sample.question
    assert free.user_prompt == sample.question
    assert original.requires_tgvf_controller is False
    assert free.requires_tgvf_controller is True
    assert free.force_action_prefix == ""


def test_softforce_appends_only_configured_phrase() -> None:
    rendered = render_benchmark_input(
        _sample(),
        _config(EvalMode.TGVF_SOFTFORCE, softforce_prompt_text="Use focus tool."),
    )

    assert rendered.user_prompt.endswith("\n\nUse focus tool.")
    assert "Before answering" not in rendered.user_prompt
    assert rendered.softforce_prompt_text == "Use focus tool."


def test_force_uses_control_prefix_without_prompt_pollution() -> None:
    sample = _sample()
    rendered = render_benchmark_input(sample, _config(EvalMode.TGVF_FORCE))

    assert rendered.user_prompt == sample.question
    assert rendered.force_action_prefix.endswith(PROTOCOL_C_FOCUS_START)
    assert rendered.requires_tgvf_controller is True


def test_qwen2_no_think_protocol_prefix() -> None:
    spec = protocol_render_spec(PROTOCOL_C_TOOL_OBSERVATION_QWEN2_NO_THINK)

    assert spec.force_action_prefix == PROTOCOL_C_FOCUS_START
