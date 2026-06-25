"""Clean benchmark prompt/input rendering contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .benchmark_data import BenchmarkSample
from .schema import EvalMode, RunConfig, _to_jsonable


THINK_START = "<think>"
THINK_END = "</think>"
PROTOCOL_C_FOCUS_START = "<|focus_start|>"
PROTOCOL_C_FOCUS_END = "<|focus_end|>"
PROTOCOL_C_TGVF_START = "<|tgvf_start|>"
PROTOCOL_C_TGVF_END = "<|tgvf_end|>"
PROTOCOL_C_TOOL_OBSERVATION = "protocol_c_tool_observation"
PROTOCOL_C_TOOL_OBSERVATION_QWEN2_NO_THINK = "protocol_c_tool_observation_qwen2_no_think"


@dataclass(frozen=True)
class ProtocolRenderSpec:
    protocol: str
    focus_start: str
    focus_end: str
    tgvf_start: str
    tgvf_end: str
    force_action_prefix: str
    tool_observation_prefix: str
    tool_observation_suffix: str

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class RenderedBenchmarkInput:
    sample_id: str
    benchmark: str
    population_id: str
    source_file: str
    mode: EvalMode
    question: str
    user_prompt: str
    media: list[dict[str, Any]]
    choices: list[str]
    gold_answer: str | None
    answer_format: str
    tgvf_protocol: str
    requires_tgvf_controller: bool
    force_action_prefix: str = ""
    softforce_prompt_text: str = ""
    prompt_suffix: str = ""
    metadata: dict[str, Any] | None = None

    def to_row(self) -> dict[str, Any]:
        return _to_jsonable(self)


def protocol_render_spec(protocol: str) -> ProtocolRenderSpec:
    if protocol == PROTOCOL_C_TOOL_OBSERVATION_QWEN2_NO_THINK:
        return ProtocolRenderSpec(
            protocol=protocol,
            focus_start=PROTOCOL_C_FOCUS_START,
            focus_end=PROTOCOL_C_FOCUS_END,
            tgvf_start=PROTOCOL_C_TGVF_START,
            tgvf_end=PROTOCOL_C_TGVF_END,
            force_action_prefix=PROTOCOL_C_FOCUS_START,
            tool_observation_prefix=f"<|im_start|>tool\n{PROTOCOL_C_TGVF_START}\n",
            tool_observation_suffix=f"\n{PROTOCOL_C_TGVF_END}<|im_end|>\n<|im_start|>assistant\n",
        )
    if protocol == PROTOCOL_C_TOOL_OBSERVATION:
        return ProtocolRenderSpec(
            protocol=protocol,
            focus_start=PROTOCOL_C_FOCUS_START,
            focus_end=PROTOCOL_C_FOCUS_END,
            tgvf_start=PROTOCOL_C_TGVF_START,
            tgvf_end=PROTOCOL_C_TGVF_END,
            force_action_prefix=f"{THINK_START}\n{THINK_END}\n{PROTOCOL_C_FOCUS_START}",
            tool_observation_prefix=f"<|im_start|>tool\n{PROTOCOL_C_TGVF_START}\n",
            tool_observation_suffix=f"\n{PROTOCOL_C_TGVF_END}<|im_end|>\n<|im_start|>assistant\n",
        )
    raise ValueError(f"unsupported clean TGVF protocol: {protocol}")


def render_benchmark_input(sample: BenchmarkSample, config: RunConfig) -> RenderedBenchmarkInput:
    config.validate()
    spec = protocol_render_spec(config.tgvf_protocol)
    user_prompt = _prompt_for_mode(sample.question, config)
    return RenderedBenchmarkInput(
        sample_id=sample.sample_id,
        benchmark=sample.benchmark,
        population_id=sample.population_id,
        source_file=sample.source_file,
        mode=config.mode,
        question=sample.question,
        user_prompt=user_prompt,
        media=[item for item in sample.to_materialized_row()["media"]],
        choices=list(sample.choices),
        gold_answer=sample.gold_answer,
        answer_format="multiple_choice" if sample.choices else "open",
        tgvf_protocol=config.tgvf_protocol,
        requires_tgvf_controller=config.mode in {
            EvalMode.TGVF_FREE,
            EvalMode.TGVF_FORCE,
            EvalMode.TGVF_SOFTFORCE,
        },
        force_action_prefix=spec.force_action_prefix if config.mode == EvalMode.TGVF_FORCE else "",
        softforce_prompt_text=config.softforce_prompt_text if config.mode == EvalMode.TGVF_SOFTFORCE else "",
        prompt_suffix=config.prompt_suffix,
        metadata=sample.metadata,
    )


def render_benchmark_inputs(
    samples: Iterable[BenchmarkSample],
    config: RunConfig,
) -> list[RenderedBenchmarkInput]:
    return [render_benchmark_input(sample, config) for sample in samples]


def _prompt_for_mode(question: str, config: RunConfig) -> str:
    prompt = question.strip()
    if config.prompt_suffix:
        prompt = _append_prompt_text(prompt, config.prompt_suffix)
    if config.mode == EvalMode.TGVF_SOFTFORCE:
        prompt = _append_prompt_text(prompt, config.softforce_prompt_text)
    return prompt


def _append_prompt_text(prompt: str, suffix: str) -> str:
    suffix = suffix.strip()
    if not suffix:
        return prompt
    if not prompt:
        return suffix
    return f"{prompt}\n\n{suffix}"
