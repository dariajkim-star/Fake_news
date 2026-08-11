"""DFDC 프레임 → YOLO 얼굴 검출 → 224² face crop (README §8.1).

- 프레임당 confidence 최고 얼굴 하나만 사용한다 (다중 얼굴은 Nice-to-Have).
- margin 1.0 (tight). bbox를 정사각형으로 만든 뒤 224²로 리사이즈한다.
- 얼굴이 한 프레임도 안 잡힌 영상은 `detection_failed`로 기록하고 분류 평가에서
  제외하되 **Detection Success Rate 분모에는 포함**한다 — 조용히 버리면 성능이
  낙관적으로 왜곡된다 (§8.1).
- detector 가중치를 인자로 받는다: pretrained(D1)로 먼저 뚫고, D2가 나오면
  같은 스크립트로 재생성한다 (--weights 교체 + --out 분리).

사용:
    python -m src.preprocess.detect_faces \
        --weights models/face_detector/yolov8n-face-lindevs.pt \
        --out data/model/processed/faces
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

CROP_SIZE = 224
CONF_THRES = 0.25


def square_crop(frame, xyxy, margin: float):
    """bbox를 margin 배율의 정사각형으로 확장해 자른다. 경계는 클램프."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * margin
    half = side / 2
    a, b = int(max(0, cx - half)), int(max(0, cy - half))
    c, d = int(min(W, cx + half)), int(min(H, cy + half))
    if c - a < 8 or d - b < 8:
        return None
    return cv2.resize(frame[b:d, a:c], (CROP_SIZE, CROP_SIZE))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path,
                    default=Path("models/face_detector/yolov8n-face-lindevs.pt"))
    ap.add_argument("--frames", type=Path, default=Path("data/model/processed/frames"))
    ap.add_argument("--out", type=Path, default=Path("data/model/processed/faces"))
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(str(args.weights))

    video_dirs = sorted(d for d in args.frames.iterdir() if d.is_dir())
    print(f"대상 영상 {len(video_dirs):,}개  (detector: {args.weights.name})")

    t0 = time.time()
    per_video: dict[str, dict] = {}
    for i, vd in enumerate(video_dirs, 1):
        frames = sorted(vd.glob("f_*.jpg"))
        out_dir = args.out / vd.name
        done = list(out_dir.glob("f_*.jpg")) if out_dir.exists() else []
        if done and len(done) == len(frames):        # 재실행 스킵
            per_video[vd.name] = {"frames": len(frames), "detected": len(done), "skipped": True}
            continue

        results = model.predict([str(f) for f in frames], conf=CONF_THRES,
                                verbose=False, batch=args.batch)
        n_det = 0
        for f, r in zip(frames, results):
            if len(r.boxes) == 0:
                continue
            best = int(r.boxes.conf.argmax())
            crop = square_crop(cv2.imread(str(f)),
                               r.boxes.xyxy[best].tolist(), args.margin)
            if crop is None:
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / f.name), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            n_det += 1
        per_video[vd.name] = {"frames": len(frames), "detected": n_det}
        if i % 200 == 0:
            print(f"  {i}/{len(video_dirs)}  ({time.time()-t0:.0f}s)", flush=True)

    failed = [v for v, s in per_video.items() if s["detected"] == 0]
    n = len(per_video)
    report = {
        "detector": args.weights.name,
        "margin": args.margin,
        "videos": n,
        "detection_failed": failed,
        "detection_success_rate": round((n - len(failed)) / n, 4) if n else 0,
        "frames_total": sum(s["frames"] for s in per_video.values()),
        "crops_total": sum(s["detected"] for s in per_video.values()),
        "elapsed_sec": round(time.time() - t0),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "detect_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"완료: Success Rate {report['detection_success_rate']:.2%} "
          f"({n-len(failed)}/{n}), crop {report['crops_total']:,}장, "
          f"{report['elapsed_sec']}초")


if __name__ == "__main__":
    main()
