from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Iterator
from contextlib import suppress
from typing import TypeVar

T = TypeVar("T")


def configure_quiet_external_progress() -> None:
    """Hide dependency loading bars while keeping harness task bars visible."""

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    for name in ("transformers", "huggingface_hub", "datasets", "accelerate"):
        logging.getLogger(name).setLevel(logging.ERROR)

    with suppress(Exception):
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    with suppress(Exception):
        from datasets import disable_progress_bar

        disable_progress_bar()
    with suppress(Exception):
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()


def iter_progress(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    enabled: bool = True,
) -> Iterator[T]:
    if not enabled:
        yield from iterable
        return
    try:
        from tqdm.auto import tqdm

        yield from tqdm(
            iterable,
            desc=desc,
            total=total,
            dynamic_ncols=True,
            leave=True,
            unit="sample",
        )
    except Exception:
        yield from iterable
