"""Clean Stage1 training CLI skeleton."""

from __future__ import annotations

import argparse

from revisit_vlm_clean.cli.common import exit_not_implemented, print_json
from revisit_vlm_clean.defaults import (
    DEFAULT_MAX_IMAGE_RESOLUTION,
    DEFAULT_MODEL_ID,
    DEFAULT_PROTOCOL,
    DEFAULT_STAGE1_GLOBAL_BATCH,
    DEFAULT_STAGE1_MAX_STEPS,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean TGVF Stage1 training skeleton.")
    parser.add_argument("--print-defaults", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    defaults = {
        "model_id": DEFAULT_MODEL_ID,
        "protocol": DEFAULT_PROTOCOL,
        "variant": "tgvf_v2_bidirectional",
        "token_row_mode": "row_only",
        "capture_mode": "teacher_forced",
        "fvt_position_mode": "native_source_grid",
        "max_image_resolution": DEFAULT_MAX_IMAGE_RESOLUTION,
        "global_batch": DEFAULT_STAGE1_GLOBAL_BATCH,
        "max_steps": DEFAULT_STAGE1_MAX_STEPS,
        "same_image_negative": "matrix_ce",
        "visual_token_manifold_loss": 0.1,
    }
    if args.print_defaults:
        print_json(defaults)
        return 0
    return exit_not_implemented("Stage1 training is a later phase; phase 1 only exposes defaults")


if __name__ == "__main__":
    raise SystemExit(main())
