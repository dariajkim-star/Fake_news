"""모델 레지스트리 — config의 `model.name`으로 모델을 교체한다.

ablation(NFR6)이 성립하려면 "학습 코드는 그대로 두고 모델만 바꾼다"가
가능해야 한다. 모든 모델은 `forward(batch: dict) -> logits [B, 2]` 계약을 지킨다.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from src.utils.config import Config

ModelBuilder = Callable[[Config], nn.Module]

_REGISTRY: dict[str, ModelBuilder] = {}


def register_model(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    """`@register_model("late_fusion")` 형태로 빌더를 등록한다."""

    def decorator(builder: ModelBuilder) -> ModelBuilder:
        _REGISTRY[name] = builder
        return builder

    return decorator


class DummyClassifier(nn.Module):
    """스캐폴딩 검증용 최소 모델 — image 텐서를 평균내어 2-class logits을 낸다."""

    def __init__(self, hidden_dim: int = 32, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        pooled = batch["image"].mean(dim=(2, 3))  # [B, 3]
        return self.net(pooled)


@register_model("dummy")
def _build_dummy(cfg: Config) -> nn.Module:
    return DummyClassifier(hidden_dim=int(cfg.get("model.hidden_dim", 32)))


def build_model(cfg: Config) -> nn.Module:
    name = str(cfg.get("model.name", "dummy"))
    if name not in _REGISTRY:
        raise KeyError(f"등록되지 않은 model: {name!r} (사용 가능: {sorted(_REGISTRY)})")
    return _REGISTRY[name](cfg)
