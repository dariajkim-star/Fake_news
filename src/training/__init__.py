"""공통 학습 루프 — baseline부터 최종 fusion 모델까지 동일 Trainer를 재사용한다."""

from src.training.trainer import Trainer, TrainerConfig

__all__ = ["Trainer", "TrainerConfig"]
