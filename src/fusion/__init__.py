"""분류 모델 모듈.

- `encoders.py`: 재사용 가능한 단일 모달 인코더 (Story 1.3, 1.4+에서 공유)
- `baseline.py`: Phase 1 baseline (BERT only / ResNet only / late fusion, Story 1.3~1.4)
- `registry.py`: config `model.name` -> 모델 빌더
"""

from src.fusion.encoders import (
    ClassificationHead,
    EncoderError,
    ImageEncoder,
    TextEncoder,
)
from src.fusion.registry import build_model, register_model

__all__ = [
    "ClassificationHead",
    "EncoderError",
    "ImageEncoder",
    "LateFusionClassifier",
    "SingleModalClassifier",
    "TextEncoder",
    "build_model",
    "predict_single",
    "register_model",
]


def __getattr__(name: str):
    """baseline 모델은 지연 import (registry 순환 import 회피)."""
    if name in ("LateFusionClassifier", "SingleModalClassifier", "predict_single"):
        import importlib

        return getattr(importlib.import_module("src.fusion.baseline"), name)
    raise AttributeError(f"module 'src.fusion'에 '{name}'가 없습니다")
