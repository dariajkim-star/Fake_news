"""seed 고정 재현성 테스트 (Story 1.1 AC3, AC7)."""

from __future__ import annotations

import random

import numpy as np
import torch

from src.data.registry import build_dataloaders
from src.utils.config import Config
from src.utils.seed import set_seed, torch_generator


def _draw() -> tuple[float, float, float]:
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_same_seed_reproduces_all_rng_sources():
    set_seed(123)
    first = _draw()
    set_seed(123)
    second = _draw()
    assert first == second


def test_different_seed_changes_draw():
    set_seed(123)
    first = _draw()
    set_seed(456)
    assert first != _draw()


def test_deterministic_flag_sets_cudnn():
    set_seed(7, deterministic=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_torch_generator_is_reproducible():
    a = torch.randperm(10, generator=torch_generator(5))
    b = torch.randperm(10, generator=torch_generator(5))
    assert torch.equal(a, b)


def test_dataloader_shuffle_order_is_reproducible():
    cfg = Config({"seed": 42, "data": {"name": "dummy", "batch_size": 8, "num_workers": 0}})

    def first_batch_labels():
        set_seed(42)
        loader = build_dataloaders(cfg, splits=("train",))["train"]
        return next(iter(loader))["label"]

    assert torch.equal(first_batch_labels(), first_batch_labels())
