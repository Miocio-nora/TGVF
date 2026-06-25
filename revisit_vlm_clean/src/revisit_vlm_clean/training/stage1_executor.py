"""Stage1 clean training executor entrypoint."""

from __future__ import annotations

from revisit_vlm_clean.training.executor import main_for_stage
from revisit_vlm_clean.training_plan import TrainingStage


def main(argv: list[str] | None = None) -> int:
    return main_for_stage(TrainingStage.STAGE1, argv)


if __name__ == "__main__":
    raise SystemExit(main())

