"""분류 모델 모듈. Phase 1 baseline은 `baseline.py`, Phase 4 최종 모델은 이후 추가된다."""

from src.fusion.registry import build_model, register_model

__all__ = ["build_model", "register_model"]
