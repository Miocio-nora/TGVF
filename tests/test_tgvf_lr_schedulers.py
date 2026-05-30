from __future__ import annotations

import importlib.util
import sys

import pytest
import torch

from revisit_vlm.lr_schedulers import (
    build_lr_scheduler,
    build_warmup_cosine_scheduler,
    estimate_optimizer_steps,
)
from revisit_vlm.tgvf_training import save_tgvf_checkpoint


def _optimizer(lr: float = 1e-4) -> torch.optim.Optimizer:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    return torch.optim.SGD([parameter], lr=lr)


def test_no_scheduler_returns_none_and_lr_stays_constant() -> None:
    optimizer = _optimizer(lr=1e-4)
    scheduler = build_lr_scheduler(
        optimizer,
        lr_scheduler="none",
        num_training_steps=10,
        warmup_ratio=0.1,
        warmup_steps=0,
        min_lr_ratio=0.1,
    )

    assert scheduler is None
    before = optimizer.param_groups[0]["lr"]
    optimizer.step()
    after = optimizer.param_groups[0]["lr"]
    assert before == pytest.approx(1e-4)
    assert after == pytest.approx(before)


def test_warmup_cosine_increases_then_decays_to_min_lr() -> None:
    optimizer = _optimizer(lr=1e-4)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        num_training_steps=100,
        warmup_ratio=0.1,
        warmup_steps=0,
        min_lr_ratio=0.1,
    )

    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(100):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    assert lrs[0] == pytest.approx(1e-5)
    assert lrs[5] > lrs[0]
    assert max(lrs) == pytest.approx(1e-4)
    assert lrs[-1] == pytest.approx(1e-5, rel=1e-4)
    assert lrs[-1] < lrs[20]


def test_estimated_optimizer_steps_uses_gradient_accumulation_ceiling() -> None:
    assert estimate_optimizer_steps(max_steps=8, gradient_accumulation_steps=4) == 2
    assert estimate_optimizer_steps(max_steps=9, gradient_accumulation_steps=4) == 3


def test_scheduler_steps_only_on_optimizer_steps_with_gradient_accumulation() -> None:
    optimizer = _optimizer(lr=1e-4)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        num_training_steps=2,
        warmup_ratio=0.5,
        warmup_steps=0,
        min_lr_ratio=0.1,
    )
    scheduler_step_count = 0

    for micro_step in range(1, 9):
        if micro_step % 4 == 0:
            optimizer.step()
            scheduler.step()
            scheduler_step_count += 1

    assert scheduler_step_count == 2
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)


def test_checkpoint_resume_restores_scheduler_lr(tmp_path) -> None:
    module = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(module.parameters(), lr=1e-4)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        num_training_steps=10,
        warmup_ratio=0.2,
        warmup_steps=0,
        min_lr_ratio=0.1,
    )
    for _ in range(4):
        optimizer.step()
        scheduler.step()
    expected_lr = optimizer.param_groups[0]["lr"]

    path = tmp_path / "checkpoint.pt"
    save_tgvf_checkpoint(
        path=path,
        foveal_module=module,
        config={"lr_scheduler": "warmup_cosine"},
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=8,
        optimizer_step=4,
    )

    new_module = torch.nn.Linear(1, 1)
    new_optimizer = torch.optim.SGD(new_module.parameters(), lr=1e-4)
    new_scheduler = build_warmup_cosine_scheduler(
        new_optimizer,
        num_training_steps=10,
        warmup_ratio=0.2,
        warmup_steps=0,
        min_lr_ratio=0.1,
    )
    checkpoint = torch.load(path, map_location="cpu")
    new_module.load_state_dict(checkpoint["tgvf_module"])
    new_optimizer.load_state_dict(checkpoint["optimizer"])
    new_scheduler.load_state_dict(checkpoint["scheduler"])

    assert checkpoint["optimizer_step"] == 4
    assert checkpoint["global_step"] == 8
    assert new_optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)
    assert new_scheduler.state_dict()["last_epoch"] == scheduler.state_dict()["last_epoch"]


def test_train_script_cli_parses_scheduler_args(monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location("train_tgvf_fvt", "scripts/train_tgvf_fvt.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_tgvf_fvt.py",
            "--train-file",
            "items.jsonl",
            "--output-dir",
            "out",
            "--lr-scheduler",
            "warmup_cosine",
            "--warmup-ratio",
            "0.03",
            "--min-lr-ratio",
            "0.1",
        ],
    )
    args = module.parse_args()

    assert args.lr_scheduler == "warmup_cosine"
    assert args.warmup_ratio == pytest.approx(0.03)
    assert args.min_lr_ratio == pytest.approx(0.1)
