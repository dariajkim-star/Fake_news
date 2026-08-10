"""데이터셋 레지스트리 — config의 `data.name`으로 Dataset을 갈아끼운다.

등록된 데이터셋: `dummy`(스캐폴딩 smoke test용), `fakeddit`(Story 1.2).
"""

from __future__ import annotations

from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.config import Config
from src.utils.seed import seed_worker, torch_generator

DatasetBuilder = Callable[[Config, str], Dataset]

_REGISTRY: dict[str, DatasetBuilder] = {}


def register_dataset(name: str) -> Callable[[DatasetBuilder], DatasetBuilder]:
    """`@register_dataset("fakeddit")` 형태로 빌더를 등록한다."""

    def decorator(builder: DatasetBuilder) -> DatasetBuilder:
        _REGISTRY[name] = builder
        return builder

    return decorator


class DummyDataset(Dataset):
    """스캐폴딩 smoke test용 합성 데이터 (image/text 텐서 + 라벨 0=REAL, 1=FAKE)."""

    def __init__(self, n_samples: int = 64, image_size: int = 32, seq_len: int = 16, seed: int = 0):
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.randn(n_samples, 3, image_size, image_size, generator=generator)
        self.input_ids = torch.randint(0, 100, (n_samples, seq_len), generator=generator)
        self.labels = torch.randint(0, 2, (n_samples,), generator=generator)
        # 라벨과 상관된 신호를 조금 넣어 학습이 실제로 동작하는지 확인 가능하게 한다.
        self.images += self.labels.view(-1, 1, 1, 1) * 0.5

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "image": self.images[idx],
            "input_ids": self.input_ids[idx],
            "label": self.labels[idx],
        }


@register_dataset("dummy")
def _build_dummy(cfg: Config, split: str) -> Dataset:
    sizes = {"train": 64, "val": 32, "test": 32}
    seeds = {"train": 0, "val": 1, "test": 2}
    return DummyDataset(
        n_samples=int(cfg.get(f"data.sizes.{split}", sizes.get(split, 32))),
        seed=int(cfg.get("seed", 42)) + seeds.get(split, 0),
    )


def _ensure_builtin_datasets() -> None:
    """등록 side-effect를 위한 지연 import (순환 import 회피)."""
    import importlib

    for module in ("src.data.fakeddit",):
        importlib.import_module(module)


def build_dataset(cfg: Config, split: str) -> Dataset:
    name = str(cfg.get("data.name", "dummy"))
    if name not in _REGISTRY:
        _ensure_builtin_datasets()
    if name not in _REGISTRY:
        raise KeyError(f"등록되지 않은 dataset: {name!r} (사용 가능: {sorted(_REGISTRY)})")
    return _REGISTRY[name](cfg, split)


def build_dataloaders(cfg: Config, splits: tuple[str, ...] = ("train", "val")) -> dict[str, DataLoader]:
    """split별 DataLoader를 만든다. shuffle/worker seed까지 고정한다(NFR3)."""
    seed = int(cfg.get("seed", 42))
    batch_size = int(cfg.get("data.batch_size", 16))
    num_workers = int(cfg.get("data.num_workers", 0))

    loaders: dict[str, DataLoader] = {}
    for split in splits:
        dataset = build_dataset(cfg, split)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=torch_generator(seed),
            drop_last=False,
            # Dataset이 자체 collate를 제공하면 사용한다 (가변 길이 토큰 padding 등)
            collate_fn=getattr(dataset, "collate_fn", None),
        )
    return loaders
