"""단일 샘플 추론 엔트리포인트 (Story 1.4 AC6 / NFR2 사전 검증).

    python scripts/predict.py --config outputs/late_fusion/config.yaml \
        --image data/fakeddit/images/abc.jpg --text "tesla stock soars after earnings"

checkpoint(기본 `outputs/<exp_name>/best.pt`)를 로딩해 fake probability와
추론 소요 시간을 출력한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.fakeddit import build_tokenizer  # noqa: E402
from src.fusion.baseline import predict_single  # noqa: E402
from src.fusion.registry import build_model  # noqa: E402
from src.utils import load_config, set_seed  # noqa: E402

from scripts.train import resolve_device  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FinFact 단일 샘플 추론")
    parser.add_argument("--config", required=True, help="실험 config yaml 경로")
    parser.add_argument("--image", default=None, help="이미지 파일 경로")
    parser.add_argument("--text", default=None, help="기사/제목 텍스트")
    parser.add_argument("--checkpoint", default=None, help="checkpoint 경로 (기본: outputs/<exp>/best.pt)")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    if args.image is None and args.text is None:
        raise SystemExit("--image 또는 --text 중 최소 하나가 필요합니다")

    cfg = load_config(args.config, overrides=args.overrides)
    set_seed(int(cfg.seed), deterministic=bool(cfg.get("deterministic", True)))
    device = resolve_device(str(cfg.get("device", "auto")))

    model = build_model(cfg)
    checkpoint = args.checkpoint or Path(cfg.get("output_dir", "outputs")) / str(cfg.exp_name) / "best.pt"
    if Path(checkpoint).is_file():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload.get("model_state", payload))

    result = predict_single(
        model,
        image_path=args.image,
        text=args.text,
        tokenizer=build_tokenizer(cfg),
        image_size=int(cfg.get("data.image_size", 224)),
        max_length=int(cfg.get("data.tokenizer.max_length", 128)),
        device=device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
