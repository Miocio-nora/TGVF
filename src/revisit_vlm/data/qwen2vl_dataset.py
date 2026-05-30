from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


class Qwen2VLJsonlDataset(Dataset):
    """JSONL dataset using Qwen2-VL message format."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.root = self.path.parent
        self.samples = self._load_jsonl(self.path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = deepcopy(self.samples[index])
        sample["messages"] = self._normalize_media_paths(sample["messages"])
        return sample

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if "messages" not in item:
                    raise ValueError(f"{path}:{line_no} is missing required key 'messages'")
                samples.append(item)
        if not samples:
            raise ValueError(f"{path} contains no training samples")
        return samples

    def _normalize_media_paths(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for message in messages:
            content = message.get("content", [])
            if isinstance(content, str):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                for key in ("image", "video"):
                    value = block.get(key)
                    if isinstance(value, str) and self._is_local_relative_path(value):
                        block[key] = str((self.root / value).resolve())
        return messages

    @staticmethod
    def _is_local_relative_path(value: str) -> bool:
        return not (
            value.startswith("http://")
            or value.startswith("https://")
            or value.startswith("file://")
            or Path(value).is_absolute()
        )
