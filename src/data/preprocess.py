"""Fakeddit 원본 메타데이터 -> 클린 manifest 전처리 (Story 1.2 AC2).

manifest 스키마는 architecture.md ``NewsSample``과 호환된다:
``sample_id, image_path, title, body, text, label, source``

Fakeddit은 ``clean_title``만 제공하므로 ``body``는 빈 문자열이며,
``text``는 학습 입력용으로 title/body를 합친 필드다.

이 모듈의 모든 함수는 DataFrame을 입출력으로 받아 **실데이터 없이도**
합성 manifest로 단위 테스트가 가능하다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from src.data.labels import (
    label_distribution,
    map_fakeddit_2way_label,
    validate_internal_labels,
)

SOURCE_NAME = "fakeddit"

#: 클린 manifest 컬럼 순서 (모든 하위 스크립트/Dataset이 이 스키마를 가정한다)
MANIFEST_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "image_path",
    "title",
    "body",
    "text",
    "label",
    "source",
)

#: Fakeddit 원본 tsv에서 사용하는 컬럼
RAW_ID_COLUMN = "id"
RAW_TEXT_COLUMN = "clean_title"
RAW_LABEL_COLUMN = "2_way_label"

#: 이미지 파일 탐색 확장자 (Fakeddit 다운로더 산출물은 보통 .jpg)
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


class PreprocessError(ValueError):
    """원본 메타데이터가 기대 스키마를 만족하지 않을 때."""


def load_raw_metadata(path: str | Path) -> pd.DataFrame:
    """Fakeddit 메타데이터 tsv를 읽는다 (필수 컬럼 존재 검증 포함)."""
    path = Path(path)
    if not path.is_file():
        raise PreprocessError(f"메타데이터 파일을 찾을 수 없습니다: {path}")
    frame = pd.read_csv(path, sep="\t", dtype={RAW_ID_COLUMN: str}, low_memory=False)
    missing = [
        column
        for column in (RAW_ID_COLUMN, RAW_TEXT_COLUMN, RAW_LABEL_COLUMN)
        if column not in frame.columns
    ]
    if missing:
        raise PreprocessError(f"원본 tsv에 필수 컬럼이 없습니다: {missing} ({path})")
    return frame


def clean_text(value: object) -> str:
    """텍스트 필드 정리 — NaN -> 빈 문자열, 공백 정규화."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return " ".join(str(value).split())


def resolve_image_path(
    image_dir: str | Path,
    sample_id: str,
    extensions: Sequence[str] = IMAGE_EXTENSIONS,
) -> Path | None:
    """``<image_dir>/<sample_id>.<ext>`` 중 실제 존재하는 첫 경로를 돌려준다."""
    image_dir = Path(image_dir)
    for extension in extensions:
        candidate = image_dir / f"{sample_id}{extension}"
        if candidate.is_file():
            return candidate
    return None


def is_loadable_image(path: str | Path) -> bool:
    """PIL로 열리는 정상 이미지인지 검증한다 (손상 파일 제외 — AC2).

    ``verify()``는 스트림을 소모하므로 검증 후 재오픈해 실제 디코딩까지 확인한다.
    """
    from PIL import Image  # 지역 import — PIL 미설치 환경에서 모듈 import는 되게 한다

    path = Path(path)
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.convert("RGB").load()
    except Exception:  # noqa: BLE001 - 손상 파일의 예외 종류는 다양하다
        return False
    return True


def build_manifest(
    raw: pd.DataFrame,
    image_dir: str | Path | None = None,
    *,
    require_image: bool = True,
    verify_images: bool = True,
    relative_to: str | Path | None = None,
    extensions: Sequence[str] = IMAGE_EXTENSIONS,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """원본 DataFrame -> 클린 manifest DataFrame + 필터링 통계.

    필터 순서: 텍스트 결측 -> 라벨 매핑 -> 이미지 존재 -> 이미지 로딩 검증.

    Args:
        require_image: False면 이미지 없는 샘플도 유지한다 (text-only 실험용).
        verify_images: False면 존재 여부만 확인하고 PIL 디코딩 검증을 건너뛴다.
        relative_to: 지정 시 manifest의 image_path를 이 경로 기준 상대경로로 저장.
    """
    stats = {
        "raw": len(raw),
        "dropped_empty_text": 0,
        "dropped_missing_image": 0,
        "dropped_corrupt_image": 0,
        "kept": 0,
    }
    rows: list[dict[str, object]] = []

    for record in raw.to_dict(orient="records"):
        sample_id = str(record[RAW_ID_COLUMN])
        title = clean_text(record.get(RAW_TEXT_COLUMN))
        if not title:
            stats["dropped_empty_text"] += 1
            continue

        label = map_fakeddit_2way_label(record[RAW_LABEL_COLUMN])

        image_path: Path | None = None
        if image_dir is not None:
            image_path = resolve_image_path(image_dir, sample_id, extensions)
            if image_path is None:
                if require_image:
                    stats["dropped_missing_image"] += 1
                    continue
            elif verify_images and not is_loadable_image(image_path):
                stats["dropped_corrupt_image"] += 1
                continue
        elif require_image:
            raise PreprocessError("require_image=True인데 image_dir이 주어지지 않았습니다")

        stored_path = ""
        if image_path is not None:
            stored_path = str(
                image_path.relative_to(Path(relative_to)) if relative_to else image_path
            ).replace("\\", "/")

        rows.append(
            {
                "sample_id": sample_id,
                "image_path": stored_path,
                "title": title,
                "body": "",
                "text": title,
                "label": label,
                "source": SOURCE_NAME,
            }
        )

    stats["kept"] = len(rows)
    manifest = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
    if not manifest.empty:
        manifest["label"] = manifest["label"].astype("int64")
        validate_internal_labels(manifest["label"])
    return manifest, stats


def write_manifest(manifest: pd.DataFrame, path: str | Path) -> Path:
    """manifest를 csv로 저장한다 (컬럼 순서 고정)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.reindex(columns=list(MANIFEST_COLUMNS)).to_csv(path, index=False, encoding="utf-8")
    return path


def read_manifest(path: str | Path) -> pd.DataFrame:
    """manifest csv를 읽고 스키마/라벨 규약을 검증한다."""
    path = Path(path)
    if not path.is_file():
        raise PreprocessError(
            f"manifest를 찾을 수 없습니다: {path}\n"
            "먼저 `python scripts/preprocess_fakeddit.py`를 실행하세요 (docs/DATA.md 참조)."
        )
    frame = pd.read_csv(path, dtype={"sample_id": str}, keep_default_na=False)
    missing = [column for column in MANIFEST_COLUMNS if column not in frame.columns]
    if missing:
        raise PreprocessError(f"manifest에 필수 컬럼이 없습니다: {missing} ({path})")
    frame["label"] = frame["label"].astype("int64")
    validate_internal_labels(frame["label"])
    return frame


def manifest_report(manifest: pd.DataFrame, name: str = "manifest") -> dict[str, object]:
    """샘플 수 / 라벨 분포 리포트 (AC3의 로그·문서 기록용)."""
    distribution = label_distribution(manifest["label"]) if len(manifest) else {"REAL": 0, "FAKE": 0}
    total = max(len(manifest), 1)
    return {
        "name": name,
        "n_samples": int(len(manifest)),
        "label_counts": distribution,
        "label_ratio": {key: round(value / total, 4) for key, value in distribution.items()},
    }


def format_report(reports: Iterable[dict[str, object]]) -> str:
    """리포트 dict들을 사람이 읽는 표로 포맷한다."""
    lines = [f"{'name':<16}{'n':>10}{'REAL':>10}{'FAKE':>10}{'fake_ratio':>12}"]
    for report in reports:
        counts = report["label_counts"]  # type: ignore[index]
        total = max(int(report["n_samples"]), 1)  # type: ignore[arg-type]
        lines.append(
            f"{str(report['name']):<16}{int(report['n_samples']):>10}"  # type: ignore[arg-type]
            f"{counts['REAL']:>10}{counts['FAKE']:>10}{counts['FAKE'] / total:>12.4f}"
        )
    return "\n".join(lines)
