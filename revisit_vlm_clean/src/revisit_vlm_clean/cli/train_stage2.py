"""Clean Stage2 training CLI skeleton."""

from __future__ import annotations

import argparse

from revisit_vlm_clean.cli.common import exit_not_implemented, print_json
from revisit_vlm_clean.defaults import (
    DEFAULT_MAX_IMAGE_RESOLUTION,
    DEFAULT_MODEL_ID,
    DEFAULT_PROTOCOL,
    DEFAULT_STAGE2_GLOBAL_BATCH,
    DEFAULT_STAGE2_MAX_STEPS,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean TGVF Stage2 training skeleton.")
    parser.add_argument("--print-defaults", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    defaults = {
        "model_id": DEFAULT_MODEL_ID,
        "protocol": DEFAULT_PROTOCOL,
        "stage2_path": "fast_batched",
        "target_focus_ratio": 0.8,
        "max_image_resolution": DEFAULT_MAX_IMAGE_RESOLUTION,
        "global_batch": DEFAULT_STAGE2_GLOBAL_BATCH,
        "max_steps": DEFAULT_STAGE2_MAX_STEPS,
        "mask_original_image_after_tgvf": True,
        "mask_original_image_after_tgvf_prob": 1.0,
        "mask_original_image_after_tgvf_scope": "through_answer",
        "deepstack_enabled": False,
        "deepstack_supported": True,
        "weighted_span_loss": {
            "evidence_state": 0.2,
            "focus_target": 1.5,
            "evidence": 1.0,
            "value_span": 1.0,
            "answer": 1.0,
            "no_focus_evidence_state": 0.2,
            "no_focus_answer": 1.0,
        },
    }
    if args.print_defaults:
        print_json(defaults)
        return 0
    return exit_not_implemented("Stage2 training is a later phase; phase 1 only exposes defaults")


if __name__ == "__main__":
    raise SystemExit(main())
