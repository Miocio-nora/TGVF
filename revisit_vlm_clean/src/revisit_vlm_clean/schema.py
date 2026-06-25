"""Dataclass and JSON schema contracts for clean TGVF runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from .defaults import (
    DEFAULT_CHOICE_PARSER_IDENTITY,
    DEFAULT_CONTINUATION,
    DEFAULT_MAX_IMAGE_RESOLUTION,
    DEFAULT_MODEL_ID,
    DEFAULT_PARSER_IDENTITY,
    DEFAULT_PROTOCOL,
    DEFAULT_SCORING_BACKEND,
)


class StrEnum(str, Enum):
    """String enum with stable JSON representation."""

    def __str__(self) -> str:
        return self.value


class EvalFamily(StrEnum):
    INTERNAL_DIAGNOSTIC = "internal_diagnostic"
    PROJECT_NATIVE_EXTERNAL = "project_native_external"
    VALKIT = "valkit"


class EvalMode(StrEnum):
    ORIGINAL = "original"
    TGVF_FREE = "tgvf_free"
    TGVF_FORCE = "tgvf_force"
    TGVF_SOFTFORCE = "tgvf_softforce"


class ContinuationMode(StrEnum):
    NATURAL_CONTINUE = "natural_continue"


class ForwardMode(StrEnum):
    KV_CACHE = "kv_cache"
    NO_KV_FULL_SEQUENCE = "no_kv_full_sequence"


class DeepStackScope(StrEnum):
    OFF = "off"
    THROUGH_ANSWER = "through_answer"
    EVIDENCE_ONLY = "evidence_only"


class ScoringBackend(StrEnum):
    AUTO = "auto"
    OFFICIAL = "official"
    PROJECT = "project"


@dataclass(frozen=True)
class DeepStackState:
    enabled: bool = False
    original_image_scope: DeepStackScope = DeepStackScope.OFF
    d_features_enabled: bool = False

    def validate(self) -> None:
        if self.d_features_enabled:
            raise ValueError("D DeepStack-like features are not a clean default")
        if self.enabled and self.original_image_scope == DeepStackScope.OFF:
            raise ValueError("enabled DeepStack requires a non-off original_image_scope")
        if not self.enabled and self.original_image_scope != DeepStackScope.OFF:
            raise ValueError("disabled DeepStack must use original_image_scope='off'")


@dataclass(frozen=True)
class ParserScorerIdentity:
    model_output_parser: str = DEFAULT_PARSER_IDENTITY
    choice_parser: str = DEFAULT_CHOICE_PARSER_IDENTITY
    scoring_backend: ScoringBackend = ScoringBackend.AUTO
    official_scorer_source: str = "src/tgvf_eval/official_tools.py"
    fallback_allowed: bool = True

    def validate(self) -> None:
        if not self.model_output_parser:
            raise ValueError("model_output_parser is required")
        if self.scoring_backend != ScoringBackend.AUTO and self.fallback_allowed:
            raise ValueError("fallback_allowed should be true only for scoring_backend='auto'")


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    checkpoint_path: str
    post_tgvf_forward_mode: ForwardMode
    eval_family: EvalFamily = EvalFamily.PROJECT_NATIVE_EXTERNAL
    mode: EvalMode = EvalMode.ORIGINAL
    model_id: str = DEFAULT_MODEL_ID
    processor_id: str | None = None
    population_id: str | None = None
    subset_id: str | None = None
    manifest_path: str | None = None
    manifest_hash: str | None = None
    max_image_resolution: int = DEFAULT_MAX_IMAGE_RESOLUTION
    max_action_tokens: int = 64
    max_answer_tokens: int = 128
    tgvf_protocol: str = DEFAULT_PROTOCOL
    post_tgvf_continuation: ContinuationMode = ContinuationMode.NATURAL_CONTINUE
    prompt_suffix: str = ""
    softforce_prompt_text: str = ""
    deepstack: DeepStackState = DeepStackState()
    parser_scorer: ParserScorerIdentity = ParserScorerIdentity()
    git_commit: str | None = None
    dirty_worktree: bool | None = None

    def validate(self) -> None:
        if not self.run_id:
            raise ValueError("run_id is required")
        if not self.checkpoint_path:
            raise ValueError("checkpoint_path is required")
        if bool(self.population_id) == bool(self.subset_id):
            raise ValueError("exactly one of population_id or subset_id is required")
        if self.post_tgvf_continuation != ContinuationMode.NATURAL_CONTINUE:
            raise ValueError("clean benchmark continuation must be natural_continue")
        if self.mode in {EvalMode.ORIGINAL, EvalMode.TGVF_FREE} and (
            self.prompt_suffix or self.softforce_prompt_text
        ):
            raise ValueError("original and tgvf_free must not include prompt suffixes")
        if self.mode == EvalMode.TGVF_SOFTFORCE and not self.softforce_prompt_text:
            raise ValueError("tgvf_softforce requires softforce_prompt_text")
        self.deepstack.validate()
        self.parser_scorer.validate()

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunConfig":
        data = dict(payload)
        data["eval_family"] = _coerce_enum(EvalFamily, data.get("eval_family", EvalFamily.PROJECT_NATIVE_EXTERNAL))
        data["mode"] = _coerce_enum(EvalMode, data.get("mode", EvalMode.ORIGINAL))
        data["post_tgvf_forward_mode"] = _coerce_enum(ForwardMode, data["post_tgvf_forward_mode"])
        data["post_tgvf_continuation"] = _coerce_enum(
            ContinuationMode,
            data.get("post_tgvf_continuation", DEFAULT_CONTINUATION),
        )
        data["deepstack"] = _deepstack_from_dict(data.get("deepstack", {}))
        data["parser_scorer"] = _parser_scorer_from_dict(data.get("parser_scorer", {}))
        config = cls(**data)
        config.validate()
        return config

    @classmethod
    def from_json(cls, text: str) -> "RunConfig":
        return cls.from_dict(json.loads(text))


@dataclass(frozen=True)
class EvalRow:
    sample_id: str
    benchmark: str
    population_id: str
    method: str
    raw_output: str
    parsed_answer: str
    score: float | None
    subset_id: str | None = None
    source_file: str | None = None
    gold_answer: str | None = None
    answer_parse_success: bool = False
    malformed: bool = False
    trigger_focus_decision: bool | None = None
    focus_valid: bool | None = None
    focus_target: str = ""
    d_condition: str | None = None
    append_success: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class EvalSummary:
    run_id: str
    n_rows: int
    n_scored: int
    accuracy: float | None
    answer_parse_rate: float | None
    malformed_rate: float | None
    trigger_rate: float | None = None
    focus_valid_rate: float | None = None
    append_success_rate: float | None = None
    manifest_hash: str | None = None
    comparable: bool = True
    comparability_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


T = TypeVar("T", bound=Enum)


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def read_run_config(path: str | Path) -> RunConfig:
    return RunConfig.from_json(Path(path).read_text())


def _coerce_enum(enum_type: type[T], value: Any) -> T:
    if isinstance(value, enum_type):
        return value
    return enum_type(str(value))


def _deepstack_from_dict(payload: dict[str, Any] | DeepStackState) -> DeepStackState:
    if isinstance(payload, DeepStackState):
        return payload
    data = dict(payload)
    data["original_image_scope"] = _coerce_enum(
        DeepStackScope,
        data.get("original_image_scope", DeepStackScope.OFF),
    )
    state = DeepStackState(**data)
    state.validate()
    return state


def _parser_scorer_from_dict(payload: dict[str, Any] | ParserScorerIdentity) -> ParserScorerIdentity:
    if isinstance(payload, ParserScorerIdentity):
        return payload
    data = dict(payload)
    data["scoring_backend"] = _coerce_enum(
        ScoringBackend,
        data.get("scoring_backend", DEFAULT_SCORING_BACKEND),
    )
    identity = ParserScorerIdentity(**data)
    identity.validate()
    return identity


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value
