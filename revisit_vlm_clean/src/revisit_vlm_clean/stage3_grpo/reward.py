"""Deterministic and cache-backed reward functions for Stage3 GRPO."""

from __future__ import annotations

import math
import re
from typing import Any

from revisit_vlm_clean.scoring import extract_final_answer, normalize_open_answer
from revisit_vlm_clean.tgvf_protocol import (
    PROTOCOL_C_FOCUS_END,
    PROTOCOL_C_FOCUS_START,
    is_generic_target,
    parse_tgvf_action,
)

from .judge import JudgeBundle, judge_score_value
from .probe import ProbeCache, ToolDecisionLabel
from .schemas import RewardBreakdown, RewardConfig, RolloutRecord, Stage3Sample


def score_rollout_reward(
    *,
    sample: Stage3Sample,
    rollout: RolloutRecord,
    reward_config: RewardConfig,
    probe_cache: ProbeCache,
    judge_bundle: JudgeBundle,
    tau: float,
    missing_probe_policy: str,
    hint_label_weight: float,
    protocol: str,
    max_tool_calls: int,
) -> RewardBreakdown:
    protocol_reward, protocol_reasons = protocol_gate_reward(
        rollout,
        reward_config=reward_config,
        protocol=protocol,
        max_tool_calls=max_tool_calls,
    )
    answer_correct = answer_is_correct(
        rollout.final_answer or extract_final_answer(rollout.raw_output),
        sample,
    )
    answer_reward = 1.0 if answer_correct else 0.0
    tool_label = probe_cache.label_for(
        sample,
        tau=tau,
        missing_policy=missing_probe_policy,
        hint_label_weight=hint_label_weight,
    )
    tool_reward = tool_decision_reward(
        rollout,
        label=tool_label,
        lambda_call=reward_config.lambda_call,
    )
    focus_row = judge_bundle.focus_score(sample, rollout) if rollout.used_tool else None
    ground_row = judge_bundle.grounding_score(sample, rollout) if rollout.used_tool else None
    focus_reward = (
        judge_score_value(
            focus_row,
            field="focus_score",
            zero_reward=reward_config.focus_zero_reward,
            missing_reward=reward_config.focus_zero_reward,
        )
        if rollout.used_tool
        else 0.0
    )
    ground_reward = (
        judge_score_value(
            ground_row,
            field="grounding_score",
            zero_reward=reward_config.grounding_zero_reward,
            missing_reward=reward_config.grounding_zero_reward,
        )
        if rollout.used_tool
        else 0.0
    )
    total = (
        reward_config.w_answer * answer_reward
        + reward_config.w_tool * tool_reward
        + reward_config.w_focus * focus_reward
        + reward_config.w_ground * ground_reward
        + protocol_reward
    )
    return RewardBreakdown(
        sample_id=sample.sample_id,
        rollout_id=rollout.rollout_id,
        reward_total=float(total),
        reward_answer=float(answer_reward),
        reward_tool=float(tool_reward),
        reward_focus=float(focus_reward),
        reward_ground=float(ground_reward),
        reward_protocol=float(protocol_reward),
        answer_correct=answer_correct,
        tool_label=tool_label.label,
        tool_label_source=tool_label.source,
        tool_label_weight=tool_label.weight,
        focus_judge=focus_row,
        grounding_judge=ground_row,
        protocol_reasons=tuple(protocol_reasons),
        metadata={
            "used_tool": rollout.used_tool,
            "num_tool_calls": rollout.num_tool_calls,
            "targets": list(rollout.targets),
        },
    )


def protocol_gate_reward(
    rollout: RolloutRecord,
    *,
    reward_config: RewardConfig,
    protocol: str,
    max_tool_calls: int,
) -> tuple[float, list[str]]:
    reasons: list[str] = []
    text = rollout.raw_output or ""
    if text.count(PROTOCOL_C_FOCUS_START) != text.count(PROTOCOL_C_FOCUS_END):
        reasons.append("unbalanced_focus_tags")
    if int(rollout.num_tool_calls) > int(max_tool_calls):
        reasons.append("max_tool_calls_exceeded")
    if int(rollout.num_tool_calls) < 0:
        reasons.append("negative_tool_calls")
    if rollout.used_tool and not rollout.targets:
        reasons.append("missing_target")
    for target in rollout.targets:
        if not str(target).strip():
            reasons.append("empty_target")
        elif is_generic_target(str(target)):
            reasons.append("generic_target")
    if rollout.used_tool and text.count(PROTOCOL_C_FOCUS_START) > int(rollout.num_tool_calls):
        reasons.append("extra_focus_after_append")
    if _looks_looped(text):
        reasons.append("loop_detected")
    try:
        parsed = parse_tgvf_action(text, protocol=protocol)
        reasons.extend(str(item) for item in parsed.malformed_reasons)
    except Exception as exc:
        reasons.append(f"parse_error:{type(exc).__name__}")
    reasons = sorted(set(reasons))
    if reasons:
        return float(reward_config.protocol_penalty), reasons
    return 0.0, []


def answer_is_correct(prediction: str, sample: Stage3Sample) -> bool:
    pred = normalize_open_answer(extract_final_answer(prediction) or prediction)
    candidates = [sample.gold_answer, *sample.answer_aliases]
    if any(pred == normalize_open_answer(candidate) for candidate in candidates if candidate):
        return True
    if sample.answer_type in {"number", "count"} or sample.eval_metric == "numeric":
        pred_num = _parse_float(prediction)
        for candidate in candidates:
            gold_num = _parse_float(candidate)
            if pred_num is not None and gold_num is not None and math.isclose(
                pred_num,
                gold_num,
                rel_tol=1e-3,
                abs_tol=1e-3,
            ):
                return True
    return False


def tool_decision_reward(
    rollout: RolloutRecord,
    *,
    label: ToolDecisionLabel,
    lambda_call: float,
) -> float:
    if label.label == "unknown" or float(label.weight) <= 0:
        return -extra_call_penalty(rollout.num_tool_calls, expected_calls=0, lambda_call=lambda_call)
    used = bool(rollout.used_tool)
    if label.label == "tool_needed":
        base = 1.0 if used else -1.0
        expected = 1
    elif label.label == "tool_unnecessary":
        base = 1.0 if not used else -0.5
        expected = 0
    else:
        base = 0.0
        expected = 0
    return float(label.weight) * base - extra_call_penalty(
        rollout.num_tool_calls,
        expected_calls=expected,
        lambda_call=lambda_call,
    )


def extra_call_penalty(num_tool_calls: int, *, expected_calls: int, lambda_call: float) -> float:
    extra = max(0, int(num_tool_calls) - int(expected_calls))
    return float(lambda_call) * float(extra)


def _parse_float(value: Any) -> float | None:
    text = str(value or "").replace(",", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _looks_looped(text: str) -> bool:
    tokens = re.findall(r"\w+|[^\w\s]", str(text or "").lower())
    if len(tokens) < 12:
        return False
    for size in range(3, 7):
        tail = tokens[-size:]
        if len(tokens) >= size * 4 and tokens[-size * 2 : -size] == tail and tokens[-size * 3 : -size * 2] == tail:
            return True
    return False
