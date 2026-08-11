"""DFDC 영상 → 프레임 샘플링 (README §8.1).

    sample_fps = 2, max_frames = 20   (10초/30fps 영상에서 정확히 20장)

split CSV(train/val/test)에 등재된 영상만 처리한다 — K 상한으로 제외된 FAKE까지
디코딩하는 낭비를 막는다. 프레임은 video-level 집계를 위해 영상별 디렉토리에 저장한다.

디코딩 실패 영상은 조용히 버리지 않고 report에 기록한다 — 검출 실패와 마찬가지로,
누락을 숨기면 성능이 낙관적으로 왜곡된다 (§8.1).

사용:
    python -m src.preprocess.extract_frames --workers 6
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2

SAMPLE_FPS = 2
MAX_FRAMES = 20
JPEG_QUALITY = 95


def extract_one(video_path: str, out_dir: str) -> tuple[str, int, str]:
    """한 영상에서 프레임을 뽑는다. 반환: (영상명, 추출 수, 오류메시지)."""
    src = Path(video_path)
    dst = Path(out_dir)
    existing = list(dst.glob("f_*.jpg"))
    if len(existing) >= MAX_FRAMES:            # 재실행 시 스킵 (idempotent)
        return src.name, len(existing), ""

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return src.name, 0, "open_failed"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / SAMPLE_FPS))

    dst.mkdir(parents=True, exist_ok=True)
    got = idx = 0
    err = ""
    while got < MAX_FRAMES:
        ok, frame = cap.read()
        if not ok:
            if got == 0:
                err = "decode_failed"
            break
        if idx % step == 0:
            cv2.imwrite(str(dst / f"f_{got:02d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            got += 1
        idx += 1
    cap.release()
    return src.name, got, err


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits-dir", type=Path, default=Path("data/model/splits"))
    ap.add_argument("--video-root", type=Path,
                    default=Path("data/model/raw/dfdc/dfdc_train_part_02/dfdc_train_part_2"))
    ap.add_argument("--out", type=Path, default=Path("data/model/processed/frames"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    videos: list[str] = []
    for split in ("train", "val", "test"):
        with (args.splits_dir / f"{split}.csv").open(encoding="utf-8") as f:
            videos += [r["video"] for r in csv.DictReader(f)]
    print(f"대상 영상 {len(videos):,}개 (split CSV 등재분만)")

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(extract_one, str(args.video_root / v),
                          str(args.out / Path(v).stem)) for v in videos]
        for i, fu in enumerate(as_completed(futs), 1):
            results.append(fu.result())
            if i % 200 == 0:
                print(f"  {i}/{len(videos)}  ({time.time()-t0:.0f}s)", flush=True)

    failed = [(n, e) for n, g, e in results if e]
    short = [(n, g) for n, g, e in results if not e and g < MAX_FRAMES]
    report = {
        "videos": len(videos),
        "ok": len(results) - len(failed),
        "failed": dict(failed),
        "short_videos": dict(short),           # 20장 미만 (짧은 영상 — 오류 아님)
        "frames_total": sum(g for _, g, _ in results),
        "elapsed_sec": round(time.time() - t0),
        "sample_fps": SAMPLE_FPS, "max_frames": MAX_FRAMES,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "extract_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"완료: {report['ok']:,}/{len(videos):,} 영상, "
          f"{report['frames_total']:,} 프레임, {report['elapsed_sec']}초 "
          f"(실패 {len(failed)}, 20장 미만 {len(short)})")


if __name__ == "__main__":
    main()
