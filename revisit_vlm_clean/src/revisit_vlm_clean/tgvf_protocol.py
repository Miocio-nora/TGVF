"""Clean TGVF protocol parsing helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

LEGACY_EVIDENCE_STATE_START = "<EVIDENCE_STATE>"
LEGACY_EVIDENCE_START = "<EVIDENCE>"
LEGACY_ANSWER_START = "<ANSWER>"
LEGACY_FOCUS_START = "<FOCUS>"
PROTOCOL_C_FOCUS_START = "<|focus_start|>"
PROTOCOL_C_FOCUS_END = "<|focus_end|>"
PROTOCOL_E_EVIDENCE_END = "<|evidence_end|>"
THINK_START = "<think>"
THINK_END = "</think>"

SUPPORTED_PROTOCOLS = (
    "protocol_c_tool_observation",
    "protocol_c_tool_observation_qwen2_no_think",
)

GENERIC_TARGETS = {
    "the image",
    "image",
    "the scene",
    "scene",
    "the object",
    "object",
    "the answer",
    "answer",
    "something",
    "visual target description",
    "target",
    "local visual evidence",
    "relevant visual evidence",
    "the relevant visual evidence",
    "specific local region",
}


@dataclass(frozen=True)
class TGVFActionParse:
    raw_text: str
    protocol: str
    focus_target: str = ""
    answer: str = ""
    has_focus_open: bool = False
    has_focus_close: bool = False
    focus_valid: bool = False
    answer_valid: bool = False
    malformed: bool = False
    malformed_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "protocol": self.protocol,
            "focus_target": self.focus_target,
            "answer": self.answer,
            "has_focus_open": self.has_focus_open,
            "has_focus_close": self.has_focus_close,
            "focus_valid": self.focus_valid,
            "answer_valid": self.answer_valid,
            "malformed": self.malformed,
            "malformed_reasons": list(self.malformed_reasons),
        }


def parse_tgvf_action(text: str, *, protocol: str) -> TGVFActionParse:
    if protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"unsupported clean TGVF protocol: {protocol}")
    raw_text = str(text or "")
    focus = _extract_between(raw_text, PROTOCOL_C_FOCUS_START, PROTOCOL_C_FOCUS_END)
    answer = _extract_protocol_c_answer(raw_text)
    has_focus_open = PROTOCOL_C_FOCUS_START in raw_text
    has_focus_close = PROTOCOL_C_FOCUS_END in raw_text
    reasons: list[str] = []
    if any(
        tag in raw_text
        for tag in (
            LEGACY_EVIDENCE_STATE_START,
            LEGACY_EVIDENCE_START,
            LEGACY_ANSWER_START,
            LEGACY_FOCUS_START,
        )
    ):
        reasons.append("legacy_tag_in_protocol_c")
    if raw_text.count(PROTOCOL_C_FOCUS_START) > 1 or raw_text.count(PROTOCOL_C_FOCUS_END) > 1:
        reasons.append("nested_or_repeated_focus")
    if has_focus_open and not has_focus_close:
        reasons.append("missing_closing_focus")
    if has_focus_close and not has_focus_open:
        reasons.append("missing_opening_focus")
    if focus is not None and not focus.strip():
        reasons.append("empty_focus")
    if focus is not None and is_generic_target(focus):
        reasons.append("generic_target")

    focus_valid = focus is not None and not any(
        reason
        in {
            "missing_closing_focus",
            "missing_opening_focus",
            "empty_focus",
            "generic_target",
            "nested_or_repeated_focus",
        }
        for reason in reasons
    )
    answer_valid = answer is not None and bool(answer.strip())
    return TGVFActionParse(
        raw_text=raw_text,
        protocol=protocol,
        focus_target=focus.strip() if focus else "",
        answer=answer.strip() if answer else "",
        has_focus_open=has_focus_open,
        has_focus_close=has_focus_close,
        focus_valid=focus_valid,
        answer_valid=answer_valid,
        malformed=bool(reasons) or (has_focus_open and not focus_valid),
        malformed_reasons=reasons,
    )


def is_generic_target(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
    return normalized in GENERIC_TARGETS or normalized.startswith("visual target description")


def _extract_between(text: str, start: str, end: str) -> str | None:
    start_index = text.find(start)
    if start_index < 0:
        return None
    inner_start = start_index + len(start)
    end_index = text.find(end, inner_start)
    if end_index < 0:
        return None
    return text[inner_start:end_index].strip()


def _extract_protocol_c_answer(text: str) -> str | None:
    if not text:
        return None
    evidence_end = text.rfind(PROTOCOL_E_EVIDENCE_END)
    if evidence_end >= 0:
        return _strip_im_end(text[evidence_end + len(PROTOCOL_E_EVIDENCE_END) :])
    last_think_end = text.rfind(THINK_END)
    last_think_start = text.rfind(THINK_START)
    if last_think_start > last_think_end:
        return None
    if last_think_end >= 0:
        return _strip_im_end(text[last_think_end + len(THINK_END) :])
    if PROTOCOL_C_FOCUS_START in text:
        return None
    return _strip_im_end(text)


def _strip_im_end(text: str) -> str | None:
    stripped = re.sub(r"<\|im_end\|>\s*$", "", str(text or "")).strip()
    return stripped or None
