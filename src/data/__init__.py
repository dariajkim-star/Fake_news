"""데이터셋 모듈 (Fakeddit 파이프라인: 전처리 -> 분할 -> subset -> Dataset)."""

from src.data.labels import (
    FAKEDDIT_2WAY_TO_INTERNAL,
    LABEL_FAKE,
    LABEL_REAL,
    map_fakeddit_2way_label,
)
from src.data.registry import build_dataloaders, build_dataset, register_dataset

__all__ = [
    "FAKEDDIT_2WAY_TO_INTERNAL",
    "LABEL_FAKE",
    "LABEL_REAL",
    "build_dataloaders",
    "build_dataset",
    "map_fakeddit_2way_label",
    "register_dataset",
]
