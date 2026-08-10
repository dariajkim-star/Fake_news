"""Fakeddit 전처리 + 고정 분할 생성 (Story 1.2 AC2, AC3).

    # 공식 분할 tsv(train/validate/test)를 그대로 사용
    python scripts/preprocess_fakeddit.py --root data/fakeddit

    # 단일 tsv에서 seed 고정 stratified split 생성
    python scripts/preprocess_fakeddit.py --root data/fakeddit \\
        --metadata data/fakeddit/raw/multimodal_train.tsv --split-mode stratified --seed 42

산출물 (`<root>/processed/`):
    manifest.csv, train.csv, val.csv, test.csv, split_report.json, preprocess_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.preprocess import (  # noqa: E402
    build_manifest,
    format_report,
    load_raw_metadata,
    manifest_report,
    write_manifest,
)
from src.data.splits import (  # noqa: E402
    DEFAULT_RATIOS,
    assert_disjoint,
    split_by_official,
    split_reports,
    stratified_split,
    write_splits,
)
from src.utils.logging import setup_console_logging  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

LOGGER = setup_console_logging()

#: 공식 분할 tsv 파일명 -> 내부 split 이름
OFFICIAL_SPLIT_FILES: dict[str, str] = {
    "train": "multimodal_train.tsv",
    "val": "multimodal_validate.tsv",
    "test": "multimodal_test_public.tsv",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fakeddit 전처리 및 분할")
    parser.add_argument("--root", default="data/fakeddit", help="데이터 루트")
    parser.add_argument("--metadata", default=None, help="단일 tsv 경로 (stratified 모드)")
    parser.add_argument("--out-dir", default=None, help="산출물 디렉토리 (기본 <root>/processed)")
    parser.add_argument("--image-dir", default=None, help="이미지 디렉토리 (기본 <root>/images)")
    parser.add_argument(
        "--split-mode",
        choices=("official", "stratified"),
        default="official",
        help="official: 공식 분할 tsv 사용 / stratified: seed 고정 층화 분할",
    )
    parser.add_argument("--seed", type=int, default=42, help="분할 seed (NFR3)")
    parser.add_argument("--allow-missing-image", action="store_true", help="이미지 없는 샘플도 유지")
    parser.add_argument("--no-verify-images", action="store_true", help="PIL 손상 검증 생략(빠름)")
    parser.add_argument("--limit", type=int, default=None, help="원본에서 앞 N행만 처리(디버깅)")
    return parser.parse_args(argv)


def _load_frames(args: argparse.Namespace, raw_dir: Path) -> dict[str, pd.DataFrame]:
    """split-mode에 따라 원본 DataFrame(들)을 읽는다."""
    if args.metadata:
        return {"all": load_raw_metadata(args.metadata)}
    if args.split_mode == "official":
        frames: dict[str, pd.DataFrame] = {}
        for split, filename in OFFICIAL_SPLIT_FILES.items():
            path = raw_dir / filename
            if path.is_file():
                frames[split] = load_raw_metadata(path)
            else:
                LOGGER.warning("공식 분할 파일 없음: %s", path)
        if not frames:
            raise SystemExit(
                f"{raw_dir}에 공식 분할 tsv가 없습니다. "
                "`python scripts/download_fakeddit.py --check`로 배치 절차를 확인하세요."
            )
        return frames
    raise SystemExit("stratified 모드에서는 --metadata 로 원본 tsv를 지정해야 합니다")


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    set_seed(int(args.seed))

    root = Path(args.root)
    raw_dir = root / "raw"
    image_dir = Path(args.image_dir) if args.image_dir else root / "images"
    out_dir = Path(args.out_dir) if args.out_dir else root / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = _load_frames(args, raw_dir)
    manifest_kwargs = dict(
        image_dir=image_dir,
        require_image=not args.allow_missing_image,
        verify_images=not args.no_verify_images,
        relative_to=root,
    )

    manifests: dict[str, pd.DataFrame] = {}
    stats: dict[str, dict[str, int]] = {}
    for name, frame in frames.items():
        if args.limit:
            frame = frame.head(int(args.limit))
        manifests[name], stats[name] = build_manifest(frame, **manifest_kwargs)
        LOGGER.info("[%s] 전처리 통계: %s", name, stats[name])

    full = pd.concat(manifests.values(), ignore_index=True).drop_duplicates("sample_id")
    manifest_path = write_manifest(full, out_dir / "manifest.csv")
    LOGGER.info("클린 manifest 저장: %s (%d건)", manifest_path, len(full))

    if "all" in manifests or args.split_mode == "stratified":
        splits = stratified_split(full, DEFAULT_RATIOS, seed=int(args.seed))
        split_mode = "stratified"
    else:
        splits = split_by_official(
            full, {name: frame["sample_id"].tolist() for name, frame in manifests.items()}
        )
        split_mode = "official"
    assert_disjoint(splits)

    paths = write_splits(splits, out_dir)
    reports = split_reports(splits)
    LOGGER.info("분할(%s) 저장: %s", split_mode, {k: str(v) for k, v in paths.items()})
    print(format_report([manifest_report(full, "manifest"), *reports]))

    summary: dict[str, object] = {
        "split_mode": split_mode,
        "seed": int(args.seed),
        "preprocess_stats": stats,
        "manifest": manifest_report(full, "manifest"),
        "splits": reports,
        "label_convention": "0=REAL, 1=FAKE (Fakeddit 2_way_label 1->0, 0->1)",
    }
    with (out_dir / "preprocess_report.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    return summary


if __name__ == "__main__":
    main()
