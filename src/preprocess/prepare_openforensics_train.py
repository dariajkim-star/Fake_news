"""OpenForensics Train → D2 학습 준비: 압축해제 검증 → seed 42 표본 → YOLO 변환.

프로토콜 (README §5.6 / §7.1.1):
  - Train 44,122장 전량은 47분/epoch라 기각. **~10k 부분표본**(seed 42)으로 학습한다.
  - Val은 D1·D2 공통 평가 전용 — 여기서 손대지 않는다.
  - 추출 목록을 results/에 기록해 재현 가능하게 한다.

사용:
    python -m src.preprocess.prepare_openforensics_train --n 10000
"""
from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path

from src.preprocess.openforensics_to_yolo import convert

RAW = Path("data/model/raw/openforensics")
OUT = Path("data/model/processed/openforensics_yolo/train")


def extract_all() -> Path:
    """Train_part_1~5.zip 전부를 검증 후 해제한다. 이미 해제돼 있으면 스킵."""
    img_dir = RAW / "Train"
    parts = sorted(RAW.glob("Train_part_*.zip"))
    if len(parts) != 5:
        raise SystemExit(f"Train zip이 5개가 아니다: {[p.name for p in parts]}")
    for p in parts:
        z = zipfile.ZipFile(p)
        bad = z.testzip()
        if bad:
            raise SystemExit(f"{p.name} 손상: {bad}")
        names = [n for n in z.namelist() if n.lower().endswith(".jpg")]
        have = sum((img_dir / Path(n).name).exists() for n in names[:50])
        if have < 50:                                  # 대충 미해제로 판단
            print(f"  해제: {p.name} ({len(names):,}장)")
            z.extractall(RAW)
        else:
            print(f"  스킵: {p.name} (이미 해제됨)")
    n = len(list(img_dir.glob("*.jpg")))
    print(f"Train 이미지 총 {n:,}장")
    if n < 44_000:
        raise SystemExit(f"이미지 수 부족: {n} (기대 44,122)")
    return img_dir


def subsample(img_dir: Path, n: int, seed: int) -> Path:
    """seed 고정 표본을 뽑아 하드링크 디렉토리를 만든다."""
    import os
    all_imgs = sorted(p.name for p in img_dir.glob("*.jpg"))
    rng = random.Random(seed)
    picked = sorted(rng.sample(all_imgs, min(n, len(all_imgs))))
    sub = RAW / f"Train_sub{len(picked)}"
    sub.mkdir(exist_ok=True)
    for name in picked:
        dst = sub / name
        if not dst.exists():
            os.link(img_dir / name, dst)
    Path("results").mkdir(exist_ok=True)
    Path("results/d2_train_sample.json").write_text(json.dumps(
        {"seed": seed, "n": len(picked), "source": "OpenForensics Train",
         "files": picked}, ensure_ascii=False), encoding="utf-8")
    print(f"표본 {len(picked):,}장 (seed={seed}) → {sub}")
    return sub


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    img_dir = extract_all()
    sub = subsample(img_dir, args.n, args.seed)
    report = convert(RAW / "Train_poly.json", sub, OUT)
    # 표본에 없는 이미지의 라벨은 convert가 missing으로 걸러낸다.
    print({k: v for k, v in report.items()})

    # ultralytics 규약: images/ ↔ labels/ 형제 디렉토리. 표본을 images/로 하드링크.
    import os
    img_out = OUT / "images"
    img_out.mkdir(exist_ok=True)
    for p in sub.glob("*.jpg"):
        dst = img_out / p.name
        if not dst.exists():
            os.link(p, dst)
    val_images = Path("data/model/processed/openforensics_yolo/val/images").resolve()
    (OUT / "data.yaml").write_text(
        f"path: {OUT.resolve().as_posix()}\n"
        f"train: images\n"
        f"val: {val_images.as_posix()}\n"        # 검증은 항상 공식 Val
        "names:\n  0: face\n", encoding="utf-8")
    print(f"images/ 하드링크 {len(list(img_out.glob('*.jpg'))):,}장, data.yaml 갱신")
    print("\n다음: bbox QA 30장 → D2 fine-tuning")


if __name__ == "__main__":
    main()
