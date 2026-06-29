"""Frozen policy reference helpers for Stage3 GRPO."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator


STAGE3_POLICY_ADAPTER_MARKERS = (
    "lora_",
    "modules_to_save",
    "trainable_tokens",
    "token_adapter",
)


@dataclass
class PolicyReferenceSnapshot:
    """A frozen copy of the trainable Stage2 policy adapter parameters."""

    state: dict[str, Any]
    source: str

    def summary(self) -> dict[str, Any]:
        parameter_count = sum(int(tensor.numel()) for tensor in self.state.values())
        device_counts: dict[str, int] = {}
        dtype_counts: dict[str, int] = {}
        for tensor in self.state.values():
            device_counts[str(tensor.device)] = device_counts.get(str(tensor.device), 0) + 1
            dtype_counts[str(tensor.dtype)] = dtype_counts.get(str(tensor.dtype), 0) + 1
        return {
            "source": self.source,
            "tensor_count": len(self.state),
            "parameter_count": parameter_count,
            "device_counts": device_counts,
            "dtype_counts": dtype_counts,
            "name_samples": list(self.state)[:8],
        }


def is_stage3_policy_adapter_parameter(name: str) -> bool:
    return any(marker in name for marker in STAGE3_POLICY_ADAPTER_MARKERS)


def capture_frozen_policy_reference(
    model: Any,
    *,
    source: str = "stage2_policy_before_stage3_updates",
) -> PolicyReferenceSnapshot:
    """Capture the current trainable policy adapter parameters as a reference.

    This intentionally snapshots only the Stage3-trainable adapter/token
    parameters. The base Qwen weights and TGVF module are frozen during Stage3,
    so duplicating them would waste memory without changing reference semantics.
    """
    state: dict[str, Any] = {}
    for name, param in model.named_parameters():
        if not bool(getattr(param, "requires_grad", False)):
            continue
        if not is_stage3_policy_adapter_parameter(name):
            raise ValueError(f"unexpected Stage3 trainable policy parameter: {name}")
        state[name] = param.detach().clone()
    if not state:
        raise RuntimeError("cannot capture frozen Stage2 reference: no trainable policy adapter parameters")
    return PolicyReferenceSnapshot(state=state, source=source)


@contextmanager
def swapped_policy_reference(
    model: Any,
    snapshot: PolicyReferenceSnapshot | None,
) -> Iterator[None]:
    """Temporarily replace trainable adapter params with the frozen reference."""
    if snapshot is None:
        yield
        return

    import torch

    params = dict(model.named_parameters())
    missing = sorted(set(snapshot.state) - set(params))
    if missing:
        raise RuntimeError(
            "frozen Stage2 reference snapshot contains parameters not present in model: "
            + ", ".join(missing[:5])
        )
    current: dict[str, Any] = {}
    with torch.no_grad():
        for name, ref_tensor in snapshot.state.items():
            param = params[name]
            if tuple(param.shape) != tuple(ref_tensor.shape):
                raise RuntimeError(
                    f"frozen Stage2 reference shape mismatch for {name}: "
                    f"model={tuple(param.shape)} ref={tuple(ref_tensor.shape)}"
                )
            current[name] = param.detach().clone()
            param.copy_(ref_tensor.to(device=param.device, dtype=param.dtype))
    try:
        yield
    finally:
        with torch.no_grad():
            for name, value in current.items():
                param = params[name]
                param.copy_(value.to(device=param.device, dtype=param.dtype))
