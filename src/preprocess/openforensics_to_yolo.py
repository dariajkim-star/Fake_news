"""OpenForensics COCO 어노테이션 → YOLO 학습 포맷 변환.

설계 (README §7.1.1):
  - 검출 클래스는 FACE 하나다. Real(0)/Fake(1) bbox를 전부 class 0으로 합친다.
    YOLO는 "어디에 얼굴이 있는가"만 답하고, 진위는 EfficientNet의 몫이다.
  - 단 GT의 real/forged 정보는 버리지 않는다 — forged-face recall을 따로 계산해야
    하므로 클래스별 원본 라벨을 sidecar(`*.forged.json`)에 보존한다.

주의 (실측으로 확인된 함정):
  - COCO bbox는 [x, y, w, h] 절대 픽셀, YOLO는 [cx, cy, w, h] 정규화.
  - json의 file_name은 "Images/Val/xxx.jpg"인데 zip을 풀면 "Val/xxx.jpg"다 —
    경로는 basename으로만 매칭한다.

사용:
    python -m src.preprocess.openforensics_to_yolo \
        --json data/model/raw/openforensics/Val_poly.json \
        --images data/model/raw/openforensics/Val \
        --out data/model/processed/openforensics_yolo/val
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def convert(json_path: Path, images_dir: Path, out_dir: Path) -> dict:
    coco = json.loads(json_path.read_text(encoding="utf-8"))

    images = {im["id"]: im for im in coco["images"]}
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    per_image: dict[int, list[str]] = {}
    forged: dict[str, list[int]] = {}          # stem → 원본 category_id 목록 (순서 = 라벨 행 순서)
    stats = Counter()
    bad = 0

    for a in coco["annotations"]:
        im = images[a["image_id"]]
        W, H = im["width"], im["height"]
        x, y, w, h = a["bbox"]
        if w <= 0 or h <= 0 or x < -1 or y < -1 or x + w > W + 1 or y + h > H + 1:
            bad += 1
            continue
        # 경계 클램프 후 정규화
        x, y = max(x, 0.0), max(y, 0.0)
        w, h = min(w, W - x), min(h, H - y)
        cx, cy = (x + w / 2) / W, (y + h / 2) / H
        line = f"0 {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}"   # class 0 = FACE (단일 클래스)
        per_image.setdefault(a["image_id"], []).append(line)
        stem = Path(im["file_name"]).stem
        forged.setdefault(stem, []).append(a["category_id"])
        stats["Real" if a["category_id"] == 0 else "Fake"] += 1

    n_missing_img = 0
    for img_id, lines in per_image.items():
        stem = Path(images[img_id]["file_name"]).stem
        if not (images_dir / f"{stem}.jpg").exists():
            n_missing_img += 1
            continue
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (out_dir / "forged_labels.json").write_text(
        json.dumps(forged, ensure_ascii=False), encoding="utf-8")

    # ultralytics data.yaml — 이미지 디렉토리는 원본 위치를 그대로 가리킨다(복사 없음).
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.resolve().as_posix()}\n"
        f"train: {images_dir.resolve().as_posix()}\n"
        f"val: {images_dir.resolve().as_posix()}\n"
        "names:\n  0: face\n",
        encoding="utf-8")

    report = {
        "images_in_json": len(images),
        "images_labeled": len(per_image) - n_missing_img,
        "faces": sum(stats.values()),
        "faces_real": stats["Real"],
        "faces_fake": stats["Fake"],
        "bad_boxes_skipped": bad,
        "missing_image_files": n_missing_img,
    }
    (out_dir / "convert_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report = convert(args.json, args.images, args.out)
    for k, v in report.items():
        print(f"  {k}: {v:,}")


if __name__ == "__main__":
    main()
