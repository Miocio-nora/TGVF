"""Clean Stage2 runtime identity contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import ForwardMode
from .tgvf_protocol import SUPPORTED_PROTOCOLS


@dataclass(frozen=True)
class Stage2RuntimeConfig:
    stage2_checkpoint: str
    eval_jsonl: str
    protocol: str = "protocol_c_tool_observation"
    d_condition: str = "correct_D"
    force_prefix_mode: str = "target_hint"
    append_forward_mode: ForwardMode = ForwardMode.KV_CACHE
    max_action_tokens: int = 64
    max_answer_tokens: int = 128

    def validate(self) -> None:
        if self.protocol not in SUPPORTED_PROTOCOLS:
            raise ValueError(f"unsupported clean Stage2 protocol: {self.protocol}")
        if self.d_condition != "correct_D":
            raise ValueError("clean Stage2 benchmark runner only supports d_condition='correct_D'")
        if self.force_prefix_mode != "target_hint":
            raise ValueError(
                "clean Stage2 benchmark runner only supports force_prefix_mode='target_hint'"
            )
        if self.max_action_tokens <= 0 or self.max_answer_tokens <= 0:
            raise ValueError("max_action_tokens and max_answer_tokens must be positive")
        if not Path(self.stage2_checkpoint).exists():
            raise FileNotFoundError(f"Stage2 checkpoint does not exist: {self.stage2_checkpoint}")
        if not Path(self.eval_jsonl).exists():
            raise FileNotFoundError(f"Stage2 eval_jsonl does not exist: {self.eval_jsonl}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage2_checkpoint": self.stage2_checkpoint,
            "eval_jsonl": self.eval_jsonl,
            "protocol": self.protocol,
            "d_condition": self.d_condition,
            "force_prefix_mode": self.force_prefix_mode,
            "append_forward_mode": self.append_forward_mode.value,
            "max_action_tokens": self.max_action_tokens,
            "max_answer_tokens": self.max_answer_tokens,
        }


def checkpoint_identity(path: str | Path) -> dict[str, Any]:
    """Read lightweight Stage2 checkpoint identity metadata."""

    import torch

    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint.get("config") or {}
    return {
        "path": str(path),
        "global_step": checkpoint.get("global_step"),
        "protocol": config.get("tgvf_protocol"),
        "processor_id": config.get("processor_id"),
        "model_id": config.get("model_id"),
        "tgvf": config.get("tgvf"),
        "lora": config.get("lora"),
        "keys": sorted(str(key) for key in checkpoint.keys()),
    }


def eval_jsonl_identity(path: str | Path, *, max_rows: int = 5) -> dict[str, Any]:
    path = Path(path)
    count = 0
    need_focus = 0
    no_focus = 0
    examples = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            count += 1
            if len(examples) < max_rows:
                record = json.loads(line)
                examples.append(
                    {
                        "image": record.get("image"),
                        "need_focus": record.get("need_focus"),
                        "trajectory_type": record.get("trajectory_type"),
                        "target": record.get("target"),
                    }
                )
            else:
                record = json.loads(line)
            if bool(record.get("need_focus", True)):
                need_focus += 1
            else:
                no_focus += 1
    return {
        "path": str(path),
        "n_rows": count,
        "need_focus": need_focus,
        "no_focus": no_focus,
        "examples": examples,
    }
