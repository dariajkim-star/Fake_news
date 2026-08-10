"""더미 데이터로 학습 루프가 끝까지 도는지 검증한다 (Story 1.1 AC4, Testing 절)."""

from __future__ import annotations

import json

import pytest
import torch

from scripts.train import main as train_main
from src.data.registry import build_dataloaders
from src.fusion.registry import build_model
from src.training import Trainer, TrainerConfig
from src.utils.config import Config
from src.utils.logging import CSVLogger

BASE_CFG = {
    "exp_name": "smoke",
    "seed": 42,
    "data": {"name": "dummy", "batch_size": 8, "num_workers": 0},
    "model": {"name": "dummy", "hidden_dim": 8},
}


def test_fit_saves_checkpoint_and_history(tmp_path):
    cfg = Config(BASE_CFG)
    loaders = build_dataloaders(cfg, splits=("train", "val"))
    trainer = Trainer(
        model=build_model(cfg),
        cfg=TrainerConfig(epochs=2, lr=0.01, log_every=0),
        out_dir=tmp_path,
        logger=CSVLogger(tmp_path),
    )
    summary = trainer.fit(loaders["train"], loaders["val"])

    assert summary["best_epoch"] in (1, 2)
    assert len(summary["history"]) == 2
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "history.csv").is_file()

    payload = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)
    assert "model_state" in payload and payload["epoch"] == summary["best_epoch"]


def test_unknown_monitor_metric_raises(tmp_path):
    cfg = Config(BASE_CFG)
    loaders = build_dataloaders(cfg, splits=("train", "val"))
    trainer = Trainer(
        model=build_model(cfg),
        cfg=TrainerConfig(epochs=1, monitor="nonexistent", log_every=0),
        out_dir=tmp_path,
    )
    with pytest.raises(KeyError, match="nonexistent"):
        trainer.fit(loaders["train"], loaders["val"])


def test_train_script_end_to_end(tmp_path):
    """`train.py --config`가 config 사본 / checkpoint / metrics.json을 남긴다."""
    summary = train_main(
        [
            "--config",
            "configs/base.yaml",
            "--set",
            "train.epochs=1",
            "exp_name=pytest_smoke",
            f"output_dir={tmp_path.as_posix()}",
            "device=cpu",
        ]
    )
    out_dir = tmp_path / "pytest_smoke"
    assert summary["best_epoch"] == 1
    assert (out_dir / "config.yaml").is_file()
    assert (out_dir / "best.pt").is_file()

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["exp_name"] == "pytest_smoke"
    assert set(metrics) >= {"val_accuracy", "val_f1", "val_auroc", "seed"}
