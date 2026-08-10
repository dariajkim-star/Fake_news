"""Fakeddit 메타데이터/이미지 준비 도우미 (Story 1.2 AC1).

**본 스크립트는 원본 데이터를 재배포하지 않는다.** Fakeddit은 비상업적 학술
연구용 공개 데이터셋이므로(architecture.md#데이터 라이선스), 사용자가 공식
배포처에서 직접 내려받은 tsv를 표준 디렉토리에 배치하는 것을 돕고, 필요한
subset의 이미지만 `image_url`에서 내려받는다 (전량 다운로드는 비현실적).

표준 디렉토리 구조 (git 제외 — `.gitignore`의 `data/`):

    data/fakeddit/
    ├── raw/                 # 공식 배포 tsv (multimodal_train.tsv 등)
    ├── images/              # <id>.jpg
    └── processed/           # 본 파이프라인 산출물 (manifest, split, subset)

사용법:

    # 1) 디렉토리 생성 + 배치 안내 출력 (네트워크 접근 없음)
    python scripts/download_fakeddit.py --root data/fakeddit --check

    # 2) 배치한 tsv에서 앞 N건의 이미지만 내려받기 (명시적 opt-in)
    python scripts/download_fakeddit.py --root data/fakeddit \\
        --metadata data/fakeddit/raw/multimodal_train.tsv --limit 500 --download-images
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging import setup_console_logging  # noqa: E402

LOGGER = setup_console_logging()

#: 공식 배포처 (직접 내려받아 raw/에 배치할 것)
FAKEDDIT_HOMEPAGE = "https://github.com/entitize/Fakeddit"
FAKEDDIT_PAPER = "https://arxiv.org/abs/1911.03854"
EXPECTED_TSV = ("multimodal_train.tsv", "multimodal_validate.tsv", "multimodal_test_public.tsv")

INSTRUCTIONS = f"""
[Fakeddit 데이터 준비 절차]

1. 공식 배포처에서 multimodal 메타데이터 tsv를 내려받는다:
     {FAKEDDIT_HOMEPAGE}   (논문: {FAKEDDIT_PAPER})
   라이선스: 비상업적 학술 연구용. 원본 재배포 금지 - 본 repo는 tsv를 포함하지 않는다.

2. 내려받은 파일을 아래 위치에 둔다 (파일명 유지 권장):
     <root>/raw/{EXPECTED_TSV[0]}
     <root>/raw/{EXPECTED_TSV[1]}
     <root>/raw/{EXPECTED_TSV[2]}

3. 이미지는 tsv의 `image_url`에서 내려받는다. 100만+ 전량은 비현실적이므로
   실험에 쓸 subset만 준비한다:
     python scripts/download_fakeddit.py --root <root> \\
         --metadata <root>/raw/{EXPECTED_TSV[0]} --limit 5000 --download-images
   결과: <root>/images/<id>.jpg

4. 클린 manifest + 고정 분할 생성:
     python scripts/preprocess_fakeddit.py --root <root>

5. 금융 subset 생성 (FR11):
     python scripts/filter_financial.py --config configs/fakeddit.yaml
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fakeddit 데이터 준비")
    parser.add_argument("--root", default="data/fakeddit", help="데이터 루트 디렉토리")
    parser.add_argument("--metadata", default=None, help="이미지를 내려받을 원본 tsv 경로")
    parser.add_argument("--limit", type=int, default=None, help="내려받을 최대 이미지 수")
    parser.add_argument(
        "--download-images",
        action="store_true",
        help="image_url에서 실제 이미지를 내려받는다 (네트워크 필요, 명시적 opt-in)",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="이미지 요청 타임아웃(초)")
    parser.add_argument("--check", action="store_true", help="디렉토리/파일 배치 상태만 점검")
    return parser.parse_args(argv)


def ensure_layout(root: str | Path) -> dict[str, Path]:
    """표준 디렉토리를 생성하고 경로 dict를 돌려준다."""
    root = Path(root)
    paths = {
        "root": root,
        "raw": root / "raw",
        "images": root / "images",
        "processed": root / "processed",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def check_layout(paths: dict[str, Path]) -> dict[str, object]:
    """배치 상태 점검 리포트 (네트워크 접근 없음)."""
    found = sorted(path.name for path in paths["raw"].glob("*.tsv"))
    n_images = sum(1 for _ in paths["images"].glob("*.*"))
    return {
        "raw_tsv": found,
        "missing_expected_tsv": [name for name in EXPECTED_TSV if name not in found],
        "n_images": n_images,
        "processed_files": sorted(path.name for path in paths["processed"].glob("*")),
    }


def download_images(
    metadata_path: str | Path,
    image_dir: str | Path,
    limit: int | None = None,
    timeout: float = 10.0,
) -> dict[str, int]:
    """tsv의 `image_url`에서 이미지를 내려받는다 (이미 있는 파일은 건너뜀)."""
    import urllib.request

    import pandas as pd

    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(metadata_path, sep="\t", dtype={"id": str}, low_memory=False)
    if "image_url" not in frame.columns:
        raise ValueError(f"tsv에 image_url 컬럼이 없습니다: {metadata_path}")
    if "hasImage" in frame.columns:
        frame = frame[frame["hasImage"] == True]  # noqa: E712 - pandas mask
    if limit:
        frame = frame.head(int(limit))

    stats = {"requested": len(frame), "downloaded": 0, "skipped": 0, "failed": 0}
    for record in frame.to_dict(orient="records"):
        target = image_dir / f"{record['id']}.jpg"
        if target.exists():
            stats["skipped"] += 1
            continue
        try:
            urllib.request.urlretrieve(str(record["image_url"]), target)  # noqa: S310
            stats["downloaded"] += 1
        except Exception as exc:  # noqa: BLE001 - 개별 URL 실패는 치명적이지 않다
            LOGGER.warning("이미지 다운로드 실패 %s: %s", record["id"], exc)
            target.unlink(missing_ok=True)
            stats["failed"] += 1
    return stats


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    paths = ensure_layout(args.root)
    LOGGER.info("데이터 디렉토리 준비 완료: %s", paths["root"])

    report = check_layout(paths)
    LOGGER.info("배치 상태: %s", report)
    if report["missing_expected_tsv"]:
        print(INSTRUCTIONS)

    if args.download_images:
        if not args.metadata:
            raise SystemExit("--download-images 사용 시 --metadata 가 필요합니다")
        stats = download_images(args.metadata, paths["images"], args.limit, args.timeout)
        LOGGER.info("이미지 다운로드 결과: %s", stats)
        report["download"] = stats
    elif not args.check:
        LOGGER.info("이미지 다운로드는 --download-images 로 명시적으로 요청해야 합니다.")
    return report


if __name__ == "__main__":
    main()
