"""평가 엔트리포인트 — 저장된 checkpoint로 지정 split의 지표를 계산한다.

    python scripts/evaluate.py --config outputs/<exp_name>/config.yaml --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import build_dataloaders  # noqa: E402
from src.evaluation import save_metrics  # noqa: E402
from src.fusion import build_model  # noqa: E402
from src.training import Trainer, TrainerConfig  # noqa: E402
from src.utils import load_config, set_seed  # noqa: E402
from src.utils.logging import setup_console_logging  # noqa: E402

from scripts.train import resolve_device  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FinFact 평가")
    parser.add_argument("--config", required=True, help="실험 config yaml 경로")
    parser.add_argument("--split", default="test", help="평가할 split (기본: test)")
    parser.add_argument("--checkpoint", default=None, help="checkpoint 경로 (기본: outputs/<exp>/best.pt)")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    cfg = load_config(args.config, overrides=args.overrides)
    logger = setup_console_logging()

    set_seed(int(cfg.seed), deterministic=bool(cfg.get("deterministic", True)))
    device = resolve_device(str(cfg.get("device", "auto")))
    out_dir = Path(cfg.get("output_dir", "outputs")) / str(cfg.exp_name)

    loaders = build_dataloaders(cfg, splits=(args.split,))
    trainer = Trainer(
        model=build_model(cfg),
        cfg=TrainerConfig.from_config(cfg),
        out_dir=out_dir,
        device=device,
    )
    trainer.load_checkpoint(args.checkpoint)

    result = trainer.evaluate(loaders[args.split], prefix=args.split)
    metrics = {k: v for k, v in result.items() if not hasattr(v, "shape")}
    path = save_metrics(
        metrics,
        out_dir,
        filename=f"metrics_{args.split}.json",
        extra={"exp_name": str(cfg.exp_name), "seed": int(cfg.seed), "split": args.split},
    )
    logger.info("평가 완료 | %s | %s", metrics, path)
    return metrics


if __name__ == "__main__":
    main()
