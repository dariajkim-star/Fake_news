"""학습 엔트리포인트.

    python scripts/train.py --config configs/base.yaml
    python scripts/train.py --config configs/base.yaml --set train.epochs=1 exp_name=smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import build_dataloaders  # noqa: E402
from src.evaluation import save_metrics  # noqa: E402
from src.fusion import build_model  # noqa: E402
from src.training import Trainer, TrainerConfig  # noqa: E402
from src.utils import build_logger, load_config, set_seed  # noqa: E402
from src.utils.logging import setup_console_logging  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FinFact 학습")
    parser.add_argument("--config", required=True, help="실험 config yaml 경로")
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="config 값 오버라이드 (예: train.epochs=1)",
    )
    return parser.parse_args(argv)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    cfg = load_config(args.config, overrides=args.overrides)
    logger = setup_console_logging()

    set_seed(int(cfg.seed), deterministic=bool(cfg.get("deterministic", True)))
    device = resolve_device(str(cfg.get("device", "auto")))

    out_dir = Path(cfg.get("output_dir", "outputs")) / str(cfg.exp_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / "config.yaml")  # 결과와 설정을 항상 함께 보관 (NFR3)
    logger.info("실험 '%s' 시작 | device=%s | out=%s", cfg.exp_name, device, out_dir)

    loaders = build_dataloaders(cfg, splits=("train", "val"))
    model = build_model(cfg)
    trainer = Trainer(
        model=model,
        cfg=TrainerConfig.from_config(cfg),
        out_dir=out_dir,
        device=device,
        logger=build_logger(out_dir, str(cfg.get("logging.backend", "csv"))),
    )

    summary = trainer.fit(loaders["train"], loaders["val"])
    val_result = trainer.evaluate(loaders["val"], prefix="val")
    metrics = {k: v for k, v in val_result.items() if not hasattr(v, "shape")}
    save_metrics(
        metrics,
        out_dir,
        extra={
            "exp_name": str(cfg.exp_name),
            "seed": int(cfg.seed),
            "best_epoch": summary["best_epoch"],
            "monitor": summary["monitor"],
            "split": "val",
        },
    )
    logger.info("완료 | best %s=%s (epoch %s)", summary["monitor"], summary["best_score"], summary["best_epoch"])
    return summary


if __name__ == "__main__":
    main()
