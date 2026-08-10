"""라벨 규약 단일 진실 공급원 (single source of truth).

**치명적 주의**: 라벨 반전은 전 실험을 무효화한다. 내부 규약과 Fakeddit 원본
인코딩이 서로 반대이므로, 매핑은 반드시 이 모듈의 상수/함수만 거쳐야 한다.

- 내부 규약 [Source: architecture.md#Data Models `NewsSample.label`]
    ``0 = REAL``, ``1 = FAKE``
- Fakeddit 원본 ``2_way_label`` (공식 배포 README 기준)
    ``1 = true(real)``, ``0 = fake``

따라서 매핑은 항등이 아니라 **반전**이다: ``2_way_label 1 -> 0(REAL)``,
``2_way_label 0 -> 1(FAKE)``.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

# -- 내부 규약 --------------------------------------------------------------
LABEL_REAL: int = 0
LABEL_FAKE: int = 1

LABEL_NAMES: dict[int, str] = {LABEL_REAL: "REAL", LABEL_FAKE: "FAKE"}

# -- Fakeddit 원본 -> 내부 규약 ---------------------------------------------
FAKEDDIT_2WAY_TO_INTERNAL: dict[int, int] = {
    1: LABEL_REAL,  # Fakeddit 1 = true  -> 내부 0 = REAL
    0: LABEL_FAKE,  # Fakeddit 0 = fake  -> 내부 1 = FAKE
}


class LabelMappingError(ValueError):
    """Fakeddit 원본 라벨이 기대한 인코딩(0/1)을 벗어났을 때."""


def map_fakeddit_2way_label(raw_label: int | str | float) -> int:
    """Fakeddit ``2_way_label`` 하나를 내부 라벨(0=REAL, 1=FAKE)로 변환한다."""
    try:
        key = int(raw_label)
    except (TypeError, ValueError) as exc:
        raise LabelMappingError(f"2_way_label을 정수로 해석할 수 없습니다: {raw_label!r}") from exc
    if key not in FAKEDDIT_2WAY_TO_INTERNAL:
        raise LabelMappingError(
            f"알 수 없는 2_way_label: {raw_label!r} (허용: {sorted(FAKEDDIT_2WAY_TO_INTERNAL)})"
        )
    return FAKEDDIT_2WAY_TO_INTERNAL[key]


def map_fakeddit_2way_series(raw_labels: Iterable[int | str | float]) -> pd.Series:
    """Series/iterable 단위 변환. 하나라도 미지의 값이면 즉시 실패한다."""
    return pd.Series([map_fakeddit_2way_label(value) for value in raw_labels], dtype="int64")


def validate_internal_labels(labels: Iterable[int]) -> None:
    """manifest 라벨이 내부 규약(0/1)만 담고 있는지 검증한다."""
    unknown = sorted({int(value) for value in labels} - {LABEL_REAL, LABEL_FAKE})
    if unknown:
        raise LabelMappingError(f"내부 라벨은 0(REAL)/1(FAKE)만 허용됩니다. 발견: {unknown}")


def label_distribution(labels: Iterable[int]) -> dict[str, int]:
    """``{"REAL": n, "FAKE": n}`` 분포를 돌려준다 (리포트/로그용)."""
    counts = {name: 0 for name in LABEL_NAMES.values()}
    for value in labels:
        counts[LABEL_NAMES[int(value)]] += 1
    return counts
