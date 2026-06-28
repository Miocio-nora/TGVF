"""Schemas for Stage3 GRPO training, rollout, reward, and caches."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from revisit_vlm_clean.schema import _to_jsonable

STAGE3_GRPO_PLAN_SCHEMA_VERSION = "stage3_grpo_training_plan_v0"
STAGE3_GRPO_EXECUTION_SCHEMA_VERSION = "stage3_grpo_execution_bundle_v0"
STAGE3_GRPO_ROLLOUT_SCHEMA_VERSION = "stage3_grpo_rollout_v0"
STAGE3_GRPO_REWARD_SCHEMA_VERSION = "stage3_grpo_reward_breakdown_v0"
STAGE3_GRPO_PROBE_SCHEMA_VERSION = "stage3_grpo_probe_cache_v0"
STAGE3_GRPO_JUDGE_SCHEMA_VERSION = "stage3_grpo_judge_cache_v0"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RolloutConfig:
    group_size: int = 8
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 224
    max_action_tokens: int = 96
    max_answer_tokens: int = 128
    max_tool_calls: int = 1
    runtime_backend: str = "fake"
    dynamic_filter_action: str = "log"
    low_variance_eps: float = 1e-6

    def validate(self) -> None:
        if int(self.group_size) < 2:
            raise ValueError("rollout.group_size must be >= 2")
        if float(self.temperature) <= 0:
            raise ValueError("rollout.temperature must be > 0")
        if not 0 < float(self.top_p) <= 1:
            raise ValueError("rollout.top_p must be in (0, 1]")
        for name in ("max_new_tokens", "max_action_tokens", "max_answer_tokens"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"rollout.{name} must be >= 1")
        if int(self.max_tool_calls) < 0:
            raise ValueError("rollout.max_tool_calls must be >= 0")
        if self.runtime_backend not in {"fake", "native_single_focus"}:
            raise ValueError("rollout.runtime_backend must be fake or native_single_focus")
        if self.dynamic_filter_action not in {"log", "skip", "downweight"}:
            raise ValueError("rollout.dynamic_filter_action must be log, skip, or downweight")
        if float(self.low_variance_eps) < 0:
            raise ValueError("rollout.low_variance_eps must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class ProbeConfig:
    enabled: bool = True
    cache_path: str | None = None
    num_off: int = 4
    num_on_clean: int = 4
    tau: float = 0.25
    missing_policy: str = "teacher_hint"
    hint_label_weight: float = 0.5

    def validate(self) -> None:
        if int(self.num_off) < 1 or int(self.num_on_clean) < 1:
            raise ValueError("probe num_off/num_on_clean must be >= 1")
        if float(self.tau) < 0:
            raise ValueError("probe.tau must be >= 0")
        if self.missing_policy not in {"teacher_hint", "unknown"}:
            raise ValueError("probe.missing_policy must be teacher_hint or unknown")
        if not 0 <= float(self.hint_label_weight) <= 1:
            raise ValueError("probe.hint_label_weight must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class RewardConfig:
    w_answer: float = 2.0
    w_tool: float = 1.0
    w_focus: float = 1.0
    w_ground: float = 1.0
    lambda_call: float = 0.05
    protocol_penalty: float = -1.0
    focus_zero_reward: float = 0.0
    grounding_zero_reward: float = -1.0
    reward_normalization: bool = True

    def validate(self) -> None:
        for name in ("w_answer", "w_tool", "w_focus", "w_ground", "lambda_call"):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"reward.{name} must be >= 0")
        if float(self.protocol_penalty) > 0:
            raise ValueError("reward.protocol_penalty should be <= 0")
        if self.focus_zero_reward > 0:
            raise ValueError("reward.focus_zero_reward should be <= 0 or 0")
        if self.grounding_zero_reward > 0:
            raise ValueError("reward.grounding_zero_reward should be <= 0 or 0")

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class JudgeConfig:
    enabled: bool = True
    mode: str = "cache_only"
    model: str = "offline_cache"
    focus_cache_path: str | None = None
    grounding_cache_path: str | None = None
    pending_path: str | None = None
    prompt_version: str = "stage3_grpo_judge_v0"
    cache_miss_reward: float = 0.0

    def validate(self) -> None:
        if self.mode not in {"offline", "cache_only", "log_only", "disabled"}:
            raise ValueError("judge.mode must be offline, cache_only, log_only, or disabled")
        if float(self.cache_miss_reward) < -1 or float(self.cache_miss_reward) > 1:
            raise ValueError("judge.cache_miss_reward must be in [-1, 1]")

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class TrainConfig:
    algorithm: str = "grpo"
    optimizer: str = "adamw"
    max_steps: int = 1
    world_size: int = 1
    per_device_prompt_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-6
    kl_coef: float = 0.02
    clip_range: float = 0.2
    eps: float = 1e-6
    lora: bool = True
    save_steps: int = 1
    eval_steps: int = 0
    max_grad_norm: float = 1.0
    seed: int = 20260627

    def validate(self) -> None:
        if self.algorithm != "grpo":
            raise ValueError("train.algorithm must be grpo")
        if self.optimizer not in {"adamw", "manual_sgd"}:
            raise ValueError("train.optimizer must be adamw or manual_sgd")
        for name in (
            "max_steps",
            "world_size",
            "per_device_prompt_batch_size",
            "gradient_accumulation_steps",
            "save_steps",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"train.{name} must be >= 1")
        if int(self.eval_steps) < 0:
            raise ValueError("train.eval_steps must be >= 0")
        for name in ("learning_rate", "kl_coef", "clip_range", "eps", "max_grad_norm"):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"train.{name} must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class WandbConfig:
    project: str | None = None
    entity: str | None = None
    mode: str | None = None
    name: str | None = None
    group: str | None = None
    job_type: str = "stage3_grpo"
    tags: tuple[str, ...] = ()
    log_artifacts: bool = True
    log_checkpoint_artifact: bool = False

    def validate(self) -> None:
        if self.mode is not None and self.mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb.mode must be online, offline, disabled, or null")
        if not self.job_type:
            raise ValueError("wandb.job_type is required")

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class Stage3GRPOConfig:
    run_id: str
    rl_data_path: str
    output_dir: str
    policy_checkpoint: str
    sample_schedule_path: str | None = None
    sample_schedule_start_step: int = 1
    model_id: str = "Qwen/Qwen3-VL-8B-Thinking"
    processor_id: str | None = None
    protocol: str = "protocol_c_tool_observation"
    max_image_resolution: int = 512
    dtype: str = "bfloat16"
    device: str = "auto"
    device_map: str | None = "auto"
    attn_implementation: str = "sdpa"
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    schema_version: str = STAGE3_GRPO_PLAN_SCHEMA_VERSION

    def validate(self) -> None:
        if not self.run_id:
            raise ValueError("run_id is required")
        if not self.rl_data_path:
            raise ValueError("rl_data_path is required")
        if not self.output_dir:
            raise ValueError("output_dir is required")
        if not self.policy_checkpoint:
            raise ValueError("policy_checkpoint is required")
        if int(self.sample_schedule_start_step) < 1:
            raise ValueError("sample_schedule_start_step must be >= 1")
        if int(self.max_image_resolution) < 1:
            raise ValueError("max_image_resolution must be >= 1")
        self.rollout.validate()
        self.probe.validate()
        self.reward.validate()
        self.judge.validate()
        self.train.validate()
        self.wandb.validate()

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Stage3GRPOConfig":
        data = dict(payload)
        data["rollout"] = RolloutConfig(**dict(data.get("rollout") or {}))
        data["probe"] = ProbeConfig(**dict(data.get("probe") or {}))
        data["reward"] = RewardConfig(**dict(data.get("reward") or {}))
        data["judge"] = JudgeConfig(**dict(data.get("judge") or {}))
        data["train"] = TrainConfig(**dict(data.get("train") or {}))
        data["wandb"] = WandbConfig(**dict(data.get("wandb") or {}))
        config = cls(**data)
        config.validate()
        return config


@dataclass(frozen=True)
class Stage3Sample:
    sample_id: str
    image_path: str
    question: str
    gold_answer: str
    source_dataset: str
    stable_image_uid: str = ""
    image_sha256: str | None = None
    choices: tuple[str, ...] = ()
    original_choices: tuple[str, ...] = ()
    answer_aliases: tuple[str, ...] = ()
    answer_type: str = "short_text"
    eval_metric: str = "normalized_exact_match"
    evidence_type: str = "unknown"
    difficulty: str = "unknown"
    question_family: str | None = None
    tool_need_hint: str | None = None
    reference_target: str | None = None
    target_spec: dict[str, Any] | None = None
    source_profile: str | None = None
    source_split: str | None = None
    source_metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def target_text(self) -> str:
        spec = self.target_spec or {}
        return str(spec.get("target_text") or self.reference_target or "")

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "Stage3Sample":
        return cls(
            sample_id=str(record.get("sample_id") or ""),
            image_path=str(record.get("image_path") or ""),
            image_sha256=record.get("image_sha256"),
            stable_image_uid=str(record.get("stable_image_uid") or ""),
            question=str(record.get("question") or ""),
            choices=tuple(str(item) for item in (record.get("choices") or [])),
            original_choices=tuple(str(item) for item in (record.get("original_choices") or [])),
            gold_answer=_string_answer(record.get("gold_answer")),
            answer_aliases=tuple(str(item) for item in (record.get("answer_aliases") or [])),
            answer_type=str(record.get("answer_type") or record.get("answer_format") or "short_text"),
            eval_metric=str(record.get("eval_metric") or "normalized_exact_match"),
            evidence_type=str(record.get("evidence_type") or "unknown"),
            difficulty=str(record.get("difficulty") or "unknown"),
            question_family=record.get("question_family"),
            source_dataset=str(record.get("source_dataset") or ""),
            source_profile=record.get("source_profile"),
            source_split=record.get("source_split"),
            source_metadata=dict(record.get("source_metadata") or {}),
            tool_need_hint=record.get("tool_need_hint")
            or ((record.get("source_metadata") or {}).get("qa_metadata") or {}).get("tool_need_hint"),
            reference_target=record.get("reference_target"),
            target_spec=dict(record.get("target_spec") or {}) or None,
            raw=dict(record),
        )

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class RolloutRecord:
    sample_id: str
    rollout_id: int
    rollout_type: str
    raw_output: str
    final_answer: str
    used_tool: bool
    num_tool_calls: int
    targets: tuple[str, ...] = ()
    post_tool_reasoning: str = ""
    token_ids: tuple[int, ...] = ()
    old_logprobs: tuple[float, ...] = ()
    ref_logprobs: tuple[float, ...] = ()
    loss_mask: tuple[int, ...] = ()
    protocol: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    reward: dict[str, Any] | None = None
    schema_version: str = STAGE3_GRPO_ROLLOUT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class RewardBreakdown:
    sample_id: str
    rollout_id: int
    reward_total: float
    reward_answer: float
    reward_tool: float
    reward_focus: float
    reward_ground: float
    reward_protocol: float
    answer_correct: bool
    tool_label: str = "unknown"
    tool_label_source: str = "unknown"
    tool_label_weight: float = 0.0
    focus_judge: dict[str, Any] | None = None
    grounding_judge: dict[str, Any] | None = None
    protocol_reasons: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = STAGE3_GRPO_REWARD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _string_answer(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0] if value else "")
    return str(value or "")
