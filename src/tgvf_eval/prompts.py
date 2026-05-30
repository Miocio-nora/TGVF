from __future__ import annotations

from dataclasses import dataclass

from revisit_vlm.tgvf import FOVEATE_END, FOVEATE_START
from tgvf_eval.config import MethodConfig

CONTINUATION_INSTRUCTION = (
    "The newly provided visual tokens are focused evidence for the target you requested. "
    "Use them to answer the user's original question. Do not mention the foveation process."
)


@dataclass(frozen=True)
class PromptBundle:
    prompt: str
    prompt_source: str
    cot_prompt_source: str


def build_force_prompt(question: str, *, cot_enabled: bool = False) -> str:
    cot_line = _cot_line(cot_enabled)
    return (
        f"{question}\n\n"
        "Before answering, select one local visual object or region to inspect. "
        "Name the thing to look at, not the answer value.\n"
        "Good targets are short noun phrases like: the glove, the dustpan, "
        "the motorcycle helmet, the Apple logo, the two mentioned objects.\n"
        "Bad targets include answer options, colors, JSON, pipes, or the words "
        "target/visual target description.\n"
        "Output exactly one foveation request in this format:\n"
        f"{FOVEATE_START}<object or region phrase>{FOVEATE_END}\n"
        f"Stop immediately after {FOVEATE_END}. Do not answer before foveating.\n"
        "Do not output reasoning, JSON, separators, or tool metadata.\n"
        f"{cot_line}"
    ).strip()


def build_free_prompt(question: str, *, cot_enabled: bool = False) -> str:
    cot_line = _cot_line(cot_enabled)
    return (
        f"{question}\n\n"
        "If local visual evidence is needed, output exactly one foveation request in this format:\n"
        f"{FOVEATE_START}target{FOVEATE_END}\n"
        f"Then stop immediately after {FOVEATE_END}. If no local visual evidence is needed, answer directly.\n"
        f"{cot_line}"
    ).strip()


def build_direct_prompt(question: str, *, cot_enabled: bool = False) -> str:
    return f"{question}\n\n{_cot_line(cot_enabled)}".strip()


def build_visible_tool_cot_prompt(question: str, *, force_foveation: bool = False) -> str:
    if force_foveation:
        trigger_text = (
            "Before answering, you must request focused visual evidence.\n"
            f"Your first output must start with {FOVEATE_START}; do not output an option letter first.\n"
            "Output exactly one foveation request:\n"
            f"{FOVEATE_START}local visual target{FOVEATE_END}\n"
            "Do not answer before the foveation result is provided."
        )
    else:
        trigger_text = (
            "If focused visual evidence is needed, output exactly one foveation request:\n"
            f"{FOVEATE_START}local visual target{FOVEATE_END}"
        )
    return (
        f"{question}\n\n"
        "Answer the question. You may reason briefly.\n"
        f"{trigger_text}\n"
        "After the foveation result is provided, continue from the same reasoning context.\n"
        "When ready, finish with:\n"
        "Final answer: <one option letter>\n"
        "Do not put answer values or option words inside the foveation target."
    ).strip()


def build_focused_target_answer_prompt(
    question: str,
    target: str,
    *,
    cot_enabled: bool = False,
) -> str:
    cot_line = _cot_line(cot_enabled)
    clean_target = target.strip() or "the requested visual evidence"
    return (
        "The visual tokens above are focused evidence for the target:\n"
        f"{clean_target}\n\n"
        "Now answer the original multiple-choice question.\n"
        f"{question}\n\n"
        f"{cot_line}"
    ).strip()


def build_prompt(question: str, config: MethodConfig) -> PromptBundle:
    if config.reasoning_mode == "visible_tool_cot":
        prompt = build_visible_tool_cot_prompt(
            question,
            force_foveation=config.trigger_mode == "force",
        )
    elif config.trigger_mode == "force":
        prompt = build_force_prompt(question, cot_enabled=config.cot_enabled)
    elif config.trigger_mode == "free":
        prompt = build_free_prompt(question, cot_enabled=config.cot_enabled)
    else:
        prompt = build_direct_prompt(question, cot_enabled=config.cot_enabled)
    return PromptBundle(
        prompt=prompt,
        prompt_source="project",
        cot_prompt_source=(
            "visible_tool_cot"
            if config.reasoning_mode == "visible_tool_cot"
            else ("minimal" if config.cot_enabled else "none")
        ),
    )


def continuation_prompt() -> str:
    return CONTINUATION_INSTRUCTION


def strip_answer_choices(question: str) -> str:
    lines = []
    for line in question.splitlines():
        stripped = line.strip()
        if len(stripped) > 3 and stripped[0] == "(" and stripped[2] == ")" and stripped[1].isalpha():
            continue
        if len(stripped) > 2 and stripped[0].isalpha() and stripped[1] == ".":
            continue
        lowered = stripped.lower()
        if lowered.startswith("answer with") or lowered.startswith("answer only with"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _cot_line(enabled: bool) -> str:
    if not enabled:
        return ""
    return "Think step by step internally, then provide the final answer in the required format."
