from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate TGVF eval shard progress files.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--progress-files", nargs="+", required=True)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=28)
    args = parser.parse_args()

    files = [Path(item) for item in args.progress_files]
    try:
        while True:
            states = [_read_json(path) for path in files]
            line = render_progress(args.label, states, width=args.width)
            sys.stderr.write("\r" + line)
            sys.stderr.flush()
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stderr.write("\n")
        sys.stderr.flush()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text())
    except Exception:
        return None


def render_progress(label: str, states: list[dict[str, Any] | None], *, width: int = 28) -> str:
    missing = sum(state is None for state in states)
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"current": 0, "total": 0, "workers": 0})
    for state in states:
        if not state:
            continue
        desc = str(state.get("desc") or "starting")
        current = int(state.get("current") or 0)
        total = state.get("total")
        groups[desc]["current"] += current
        groups[desc]["total"] += int(total or 0)
        groups[desc]["workers"] += 1

    if not groups:
        return f"[{label}] waiting for {len(states)} workers..."

    parts = []
    for desc, info in sorted(groups.items()):
        current = info["current"]
        total = info["total"]
        workers = info["workers"]
        if total > 0:
            ratio = max(0.0, min(1.0, current / total))
            filled = int(width * ratio)
            bar = "#" * filled + "-" * (width - filled)
            parts.append(f"{desc} {current}/{total} [{bar}] {ratio * 100:5.1f}% ({workers}w)")
        else:
            parts.append(f"{desc} {current} ({workers}w)")
    if missing:
        parts.append(f"starting {missing}w")
    return f"[{label}] " + " | ".join(parts)


if __name__ == "__main__":
    main()
