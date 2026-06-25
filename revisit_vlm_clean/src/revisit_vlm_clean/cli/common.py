"""Shared CLI helpers."""

from __future__ import annotations

import json
import sys
from typing import Any

from revisit_vlm_clean.schema import _to_jsonable


def print_json(payload: Any) -> None:
    print(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True))


def exit_not_implemented(message: str) -> int:
    print(message, file=sys.stderr)
    return 2
