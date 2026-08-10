"""모델에 독립적인 공통 Trainer (AC4).

인터페이스 계약(NFR6 — 모듈 교체 가능성):
- batch는 dict이며 정답 라벨은 `label` 키에 담긴다.
- model은 `model(batch) -> logits [B, 2]`를 만족하면 무엇이든 된다
  (Phase 1 late-fusion baseline도, Phase 4 cross-attention 모델도 동일).

이 계약만 지키면 Epic 2~5의 어떤 변형도 학습 루프를 다시 짜지 않는다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_metrics
from src.utils.config import Config
from src.utils.logging import ExperimentLogger, setup_console_logging

LOGGER = setup_console_logging()


@dataclass
class TrainerConfig:
    """학습 루프 하이퍼파라미터 (config yaml의 `train` 블록과 대응)."""

    epochs: int = 10
    lr: float = 2e-5
    weight_decay: float = 0.01
    grad_clip: float | None = 1.0
    amp: bool = False
    monitor: str = "f1"  # best checkpoint 기준 지표
    monitor_mode: str = "max"
    early_stopping_patience: int | None = None
    log_every: int = 50

    @classmethod
    def from_config(cls, cfg: Config) -> "TrainerConfig":
        train_cfg = cfg.get("train", None)
        values = train_cfg.to_dict() if isinstance(train_cfg, Config) else {}
        known = {field: values[field] for field in cls.__dataclass_fields__ if field in values}
        return cls(**known)


def _move_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


class Trainer:
    """train/val loop + best checkpoint 저장 + early stopping + metric 로깅."""

    def __init__(
        self,
        model: nn.Module,
        cfg: TrainerConfig,
        out_dir: str | Path,
        device: torch.device | str = "cpu",
        criterion: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        logger: ExperimentLogger | None = None,
        metric_fn: Callable[[np.ndarray, np.ndarray], dict[str, float]] = compute_metrics,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.logger = logger
        self.metric_fn = metric_fn
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and self.device.type == "cuda")
        self.best_score: float | None = None
        self.best_epoch: int | None = None
        self.checkpoint_path = self.out_dir / "best.pt"

    # -- 내부 헬퍼 --------------------------------------------------------
    def _is_better(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.cfg.monitor_mode == "max":
            return score > self.best_score
        return score < self.best_score

    def _forward_loss(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        labels = batch["label"]
        logits = self.model(batch)
        loss = self.criterion(logits, labels)
        return loss, logits

    # -- 공개 API ---------------------------------------------------------
    def train_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss, n_batches = 0.0, 0
        for step, raw_batch in enumerate(loader, start=1):
            batch = _move_to_device(raw_batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=self.device.type, enabled=self.cfg.amp and self.device.type == "cuda"
            ):
                loss, _ = self._forward_loss(batch)

            self.scaler.scale(loss).backward()
            if self.cfg.grad_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.detach())
            n_batches += 1
            if self.cfg.log_every and step % self.cfg.log_every == 0:
                LOGGER.info("epoch %d step %d loss %.4f", epoch, step, total_loss / n_batches)

        return {"train_loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, prefix: str = "val") -> dict[str, Any]:
        """평가 후 지표와 raw 예측(확률/라벨)을 함께 돌려준다."""
        self.model.eval()
        total_loss, n_batches = 0.0, 0
        probs: list[np.ndarray] = []
        labels: list[np.ndarray] = []

        for raw_batch in loader:
            batch = _move_to_device(raw_batch, self.device)
            loss, logits = self._forward_loss(batch)
            total_loss += float(loss.detach())
            n_batches += 1
            probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
            labels.append(batch["label"].cpu().numpy())

        y_prob = np.concatenate(probs) if probs else np.array([])
        y_true = np.concatenate(labels) if labels else np.array([])
        metrics = self.metric_fn(y_true, y_prob)
        result: dict[str, Any] = {f"{prefix}_{key}": value for key, value in metrics.items()}
        result[f"{prefix}_loss"] = total_loss / max(n_batches, 1)
        result["y_true"] = y_true
        result["y_prob"] = y_prob
        return result

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict[str, Any]:
        """전체 학습을 수행하고 best checkpoint 정보를 반환한다."""
        patience_left = self.cfg.early_stopping_patience
        history: list[dict[str, float]] = []

        for epoch in range(1, self.cfg.epochs + 1):
            started = time.time()
            train_metrics = self.train_epoch(train_loader, epoch)
            val_result = self.evaluate(val_loader, prefix="val")
            scalars = {
                **train_metrics,
                **{k: v for k, v in val_result.items() if not isinstance(v, np.ndarray)},
                "epoch_time_sec": round(time.time() - started, 2),
            }
            history.append(scalars)
            if self.logger:
                self.logger.log(epoch, scalars)
            LOGGER.info("epoch %d | %s", epoch, scalars)

            score = scalars.get(f"val_{self.cfg.monitor}")
            if score is None:
                raise KeyError(
                    f"monitor 지표 'val_{self.cfg.monitor}'가 평가 결과에 없습니다: {sorted(scalars)}"
                )
            if self._is_better(float(score)):
                self.best_score, self.best_epoch = float(score), epoch
                self.save_checkpoint(epoch, scalars)
                patience_left = self.cfg.early_stopping_patience
            elif patience_left is not None:
                patience_left -= 1
                if patience_left <= 0:
                    LOGGER.info("early stopping (epoch %d, best epoch %d)", epoch, self.best_epoch)
                    break

        if self.logger:
            self.logger.close()
        return {
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "monitor": f"val_{self.cfg.monitor}",
            "checkpoint": str(self.checkpoint_path),
            "history": history,
        }

    def save_checkpoint(self, epoch: int, metrics: Mapping[str, float]) -> Path:
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "metrics": dict(metrics),
            },
            self.checkpoint_path,
        )
        return self.checkpoint_path

    def load_checkpoint(self, path: str | Path | None = None) -> dict[str, Any]:
        payload = torch.load(path or self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(payload["model_state"])
        return payload
