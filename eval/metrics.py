from __future__ import annotations

import math
import re
import statistics
from collections import Counter, defaultdict
from typing import Any, Iterable

import torch


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(ref_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def char_f1(prediction: str, reference: str) -> float:
    pred_chars = list(normalize_text(prediction).replace(" ", ""))
    ref_chars = list(normalize_text(reference).replace(" ", ""))
    if not pred_chars and not ref_chars:
        return 1.0
    if not pred_chars or not ref_chars:
        return 0.0
    overlap = Counter(pred_chars) & Counter(ref_chars)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(pred_chars)
    recall = common / len(ref_chars)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def substring_match(prediction: str, reference: str) -> float:
    pred = normalize_text(prediction)
    ref = normalize_text(reference)
    if not pred or not ref:
        return 0.0
    return float(ref in pred or pred in ref)


def finite_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def mean(values: Iterable[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if finite_float(value) is not None]
    if not cleaned:
        return None
    return float(sum(cleaned) / len(cleaned))


def median(values: Iterable[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if finite_float(value) is not None]
    if not cleaned:
        return None
    return float(statistics.median(cleaned))


def pct_positive(values: Iterable[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if finite_float(value) is not None]
    if not cleaned:
        return None
    return float(sum(value > 0 for value in cleaned) / len(cleaned))


def grouped_means(
    rows: list[dict[str, Any]],
    *,
    group_key: str,
    metric_keys: list[str],
) -> dict[str, dict[str, float | None]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_key) or "unknown")].append(row)
    return {
        group: {metric: mean(row.get(metric) for row in group_rows) for metric in metric_keys}
        for group, group_rows in sorted(groups.items())
    }


def tensor_distribution_stats(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.detach().float()
    finite = torch.isfinite(values)
    flat = values[finite]
    if flat.numel() == 0:
        return {
            "shape": list(values.shape),
            "finite_rate": 0.0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "mean_token_norm": None,
        }
    token_norms = values.norm(dim=-1) if values.ndim >= 2 else values.abs()
    return {
        "shape": list(values.shape),
        "finite_rate": float(finite.float().mean().item()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "mean_token_norm": float(token_norms.float().mean().item()),
        "std_token_norm": float(token_norms.float().std(unbiased=False).item()),
    }
