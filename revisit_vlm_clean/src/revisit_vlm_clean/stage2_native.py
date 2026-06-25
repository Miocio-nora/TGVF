"""Clean-native Stage2 engine boundary.

This module owns the final backend contract for Qwen3 Stage2 TGVF execution.
The heavy capture/append implementation is intentionally not ported here yet;
this phase only establishes the clean runtime identity and keeps the native
backend independent from historical evaluator classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .benchmark_data import BenchmarkSample
from .data_generation import file_identity
from .rendering import RenderedBenchmarkInput
from .schema import EvalMode, RunConfig
from .stage2_runtime import Stage2RuntimeConfig, eval_jsonl_identity

SUPPORTED_NATIVE_STAGE2_MODES = frozenset(
    {
        EvalMode.TGVF_FORCE,
        EvalMode.TGVF_FREE,
        EvalMode.TGVF_SOFTFORCE,
    }
)


class NativeStage2ExecutionNotPortedError(NotImplementedError):
    """Raised when the native engine is prepared but execution is not ported."""


@dataclass(frozen=True)
class NativeStage2RunResult:
    raw_output: str
    triggered: bool = False
    focus_target: str = ""
    focus_valid: bool | None = None
    append_success: bool | None = None
    output_tokens: int = 0
    error: str | None = None
    debug: dict[str, Any] = field(default_factory=dict)


class NativeStage2Engine:
    """Clean-native Stage2 runtime owner.

    The engine deliberately accepts only plain backend options so it does not
    depend on the runner's config dataclasses. Later phases should add model
    loading, focus capture, D construction, and post-TGVF continuation here.
    """

    def __init__(
        self,
        *,
        stage2_config: Stage2RuntimeConfig,
        backend_options: dict[str, Any] | None = None,
    ) -> None:
        self.stage2_config = stage2_config
        self.backend_options = dict(backend_options or {})
        self._prepared = False
        self._identity: dict[str, Any] | None = None

    def prepare(self, config: RunConfig) -> None:
        self.stage2_config.validate()
        _validate_run_alignment(self.stage2_config, config)
        self._identity = {
            "engine": "NativeStage2Engine",
            "capture_append_ported": False,
            "stage2_runtime": self.stage2_config.to_dict(),
            "run_config": {
                "run_id": config.run_id,
                "mode": config.mode.value,
                "post_tgvf_forward_mode": config.post_tgvf_forward_mode.value,
                "post_tgvf_continuation": config.post_tgvf_continuation.value,
                "tgvf_protocol": config.tgvf_protocol,
                "max_image_resolution": config.max_image_resolution,
                "max_action_tokens": config.max_action_tokens,
                "max_answer_tokens": config.max_answer_tokens,
                "deepstack": config.deepstack.to_dict(),
                "parser_scorer": config.parser_scorer.to_dict(),
            },
            "checkpoint_file": file_identity(self.stage2_config.stage2_checkpoint).to_dict(),
            "eval_jsonl": eval_jsonl_identity(self.stage2_config.eval_jsonl),
            "backend_options": dict(self.backend_options),
        }
        self._prepared = True

    def run(
        self,
        sample: BenchmarkSample,
        rendered: RenderedBenchmarkInput,
        config: RunConfig,
    ) -> NativeStage2RunResult:
        if not self._prepared:
            raise RuntimeError("NativeStage2Engine.prepare must be called before run")
        _validate_run_alignment(self.stage2_config, config)
        if rendered.mode != config.mode:
            raise ValueError(
                "rendered input mode does not match run config: "
                f"{rendered.mode.value!r} != {config.mode.value!r}"
            )
        if rendered.sample_id != sample.sample_id:
            raise ValueError(
                "rendered input sample_id does not match sample: "
                f"{rendered.sample_id!r} != {sample.sample_id!r}"
            )
        raise NativeStage2ExecutionNotPortedError(
            "clean-native Stage2 capture/append execution is not ported yet; "
            "the native engine currently validates runtime identity only"
        )

    def identity(self) -> dict[str, Any]:
        if self._identity is None:
            return {
                "engine": "NativeStage2Engine",
                "prepared": False,
                "stage2_runtime": self.stage2_config.to_dict(),
                "backend_options": dict(self.backend_options),
            }
        return dict(self._identity)


def _validate_run_alignment(stage2_config: Stage2RuntimeConfig, config: RunConfig) -> None:
    config.validate()
    if config.mode not in SUPPORTED_NATIVE_STAGE2_MODES:
        raise NotImplementedError(
            "clean-native Stage2 backend supports only TGVF modes, got "
            f"{config.mode.value!r}"
        )
    if stage2_config.protocol != config.tgvf_protocol:
        raise ValueError(
            "Stage2 runtime protocol does not match run config: "
            f"{stage2_config.protocol!r} != {config.tgvf_protocol!r}"
        )
    if stage2_config.append_forward_mode != config.post_tgvf_forward_mode:
        raise ValueError(
            "Stage2 append_forward_mode does not match run config: "
            f"{stage2_config.append_forward_mode.value!r} != "
            f"{config.post_tgvf_forward_mode.value!r}"
        )
