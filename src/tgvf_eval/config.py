from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

TriggerMode = Literal["direct", "force", "free"]
TGVFMode = Literal["none", "prompt_only", "module"]
VideoFoveationMode = Literal["off", "per_frame_reencode", "nextframe_encode_no_reencode"]
FVTAppendMode = Literal["legacy_text_positions", "qwen_native_pseudo_image"]
ReasoningMode = Literal["final_only", "internal_cot", "visible_tool_cot"]


DEFAULT_BENCHMARK_ROOT = "/home/dredvpn009/Flash_Storage/datasets/benchmarks"
DEFAULT_TOOLS_ROOT = "/home/dredvpn009/Flash_Storage/datasets/benchmarks/_tools"
DEFAULT_OUTPUT_ROOT = "eval_outputs/tgvf_benchmarks"
DEFAULT_SEED = 20260525


IMAGE_BUDGET_PIXELS: dict[str, dict[str, int]] = {
    "low": {"max_pixels": 1280 * 28 * 28},
    "mid": {"max_pixels": 4096 * 28 * 28},
    "high": {"max_pixels": 8192 * 28 * 28},
}


def image_budget_kwargs(budget: str) -> dict[str, int]:
    if budget not in IMAGE_BUDGET_PIXELS:
        raise ValueError(f"Unknown image budget '{budget}'. Expected one of {sorted(IMAGE_BUDGET_PIXELS)}")
    return dict(IMAGE_BUDGET_PIXELS[budget])


@dataclass(frozen=True)
class MethodConfig:
    method: str
    trigger_mode: TriggerMode = "direct"
    tgvf_mode: TGVFMode = "none"
    cot_enabled: bool = False
    cot_prompt_source: str = "none"
    reasoning_mode: ReasoningMode = "final_only"
    max_foveations: int = 1
    image_budget: str = "mid"
    video_nframes: int | None = None
    video_foveation_mode: VideoFoveationMode = "off"
    nextframe_reuse_ttl: int | None = None
    control_condition: str | None = None
    continuation_instruction: bool = True
    second_full_forward_allowed: bool = False
    fvt_append_mode: FVTAppendMode = "qwen_native_pseudo_image"

    def validate(self) -> None:
        if self.trigger_mode == "direct" and self.tgvf_mode != "none":
            raise ValueError("direct trigger mode requires tgvf_mode='none'")
        if self.tgvf_mode == "none" and self.trigger_mode != "direct":
            raise ValueError("tgvf_mode='none' requires trigger_mode='direct'")
        if self.max_foveations < 0:
            raise ValueError("max_foveations must be >= 0")
        validate_video_foveation_mode(self.video_foveation_mode)
        validate_fvt_append_mode(self.fvt_append_mode)
        validate_reasoning_mode(self.reasoning_mode)
        if self.video_foveation_mode == "nextframe_encode_no_reencode":
            if self.nextframe_reuse_ttl is None or self.nextframe_reuse_ttl < 1:
                raise ValueError("nextframe_encode_no_reencode requires nextframe_reuse_ttl >= 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


METHOD_MAP: dict[str, tuple[TriggerMode, TGVFMode, str | None]] = {
    "direct_qwen": ("direct", "none", None),
    "tgvf_prompt_only_force": ("force", "prompt_only", None),
    "tgvf_prompt_only_free": ("free", "prompt_only", None),
    "tgvf_module_force": ("force", "module", None),
    "tgvf_module_free": ("free", "module", None),
    "tgvf_conditioned_D_fresh": ("force", "module", "conditioned_D_fresh"),
    "tgvf_no_D": ("force", "module", "no_D"),
    "tgvf_original_D": ("force", "module", "original_D"),
    "tgvf_random_D": ("force", "module", "random_D"),
    "tgvf_wrong_D": ("force", "module", "wrong_D"),
}


def method_config_from_name(
    method: str,
    *,
    trigger_mode: str | None = None,
    tgvf_mode: str | None = None,
    cot_enabled: bool = False,
    max_foveations: int = 1,
    image_budget: str = "mid",
    video_nframes: int | None = None,
    video_foveation_mode: str = "off",
    nextframe_reuse_ttl: int | None = None,
    fvt_append_mode: str = "qwen_native_pseudo_image",
    reasoning_mode: str = "final_only",
) -> MethodConfig:
    if method not in METHOD_MAP:
        raise ValueError(f"Unknown method '{method}'. Expected one of {sorted(METHOD_MAP)}")
    default_trigger, default_tgvf, control = METHOD_MAP[method]
    effective_reuse_ttl = (
        nextframe_reuse_ttl if video_foveation_mode == "nextframe_encode_no_reencode" else None
    )
    config = MethodConfig(
        method=method,
        trigger_mode=(trigger_mode or default_trigger),  # type: ignore[arg-type]
        tgvf_mode=(tgvf_mode or default_tgvf),  # type: ignore[arg-type]
        cot_enabled=bool(cot_enabled or reasoning_mode == "internal_cot"),
        cot_prompt_source=(
            "visible_tool_cot"
            if reasoning_mode == "visible_tool_cot"
            else ("minimal" if cot_enabled or reasoning_mode == "internal_cot" else "none")
        ),
        reasoning_mode=reasoning_mode,  # type: ignore[arg-type]
        max_foveations=max_foveations,
        image_budget=image_budget,
        video_nframes=video_nframes,
        video_foveation_mode=video_foveation_mode,  # type: ignore[arg-type]
        nextframe_reuse_ttl=effective_reuse_ttl,
        control_condition=control,
        fvt_append_mode=fvt_append_mode,  # type: ignore[arg-type]
    )
    config.validate()
    return config


def validate_fvt_append_mode(value: str) -> None:
    allowed = {"legacy_text_positions", "qwen_native_pseudo_image"}
    if value not in allowed:
        raise ValueError(f"Invalid fvt_append_mode {value!r}. Expected one of {sorted(allowed)}")


def validate_video_foveation_mode(value: str) -> None:
    allowed = {"off", "per_frame_reencode", "nextframe_encode_no_reencode"}
    if value not in allowed:
        raise ValueError(f"Invalid video_foveation_mode '{value}'. Expected one of {sorted(allowed)}")


def validate_reasoning_mode(value: str) -> None:
    allowed = {"final_only", "internal_cot", "visible_tool_cot"}
    if value not in allowed:
        raise ValueError(f"Invalid reasoning_mode {value!r}. Expected one of {sorted(allowed)}")


def tier_video_nframes(tier: str) -> int:
    return {"light": 8, "medium": 16, "full": 32}.get(tier, 8)


def project_default_output_root(project_root: str | Path | None = None) -> Path:
    root = Path(project_root) if project_root else Path.cwd()
    return root / DEFAULT_OUTPUT_ROOT
