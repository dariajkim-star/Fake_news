"""라벨 규약 고정 테스트 (Story 1.2 AC2).

라벨 반전은 전 실험을 무효화하는 치명적 버그이므로, 매핑을 여기서 못박는다.
"""

from __future__ import annotations

import pytest

from src.data.labels import (
    FAKEDDIT_2WAY_TO_INTERNAL,
    LABEL_FAKE,
    LABEL_REAL,
    LabelMappingError,
    label_distribution,
    map_fakeddit_2way_label,
    map_fakeddit_2way_series,
    validate_internal_labels,
)


def test_internal_convention_is_real0_fake1():
    """architecture.md NewsSample: 0=REAL, 1=FAKE."""
    assert (LABEL_REAL, LABEL_FAKE) == (0, 1)


def test_fakeddit_mapping_is_inverted_not_identity():
    """Fakeddit 2_way_label(1=true)과 내부 규약(1=FAKE)은 서로 반대다."""
    assert FAKEDDIT_2WAY_TO_INTERNAL == {1: 0, 0: 1}
    assert map_fakeddit_2way_label(1) == LABEL_REAL
    assert map_fakeddit_2way_label(0) == LABEL_FAKE
    # 항등 매핑이면 즉시 실패 — 회귀 방지
    assert map_fakeddit_2way_label(1) != 1


@pytest.mark.parametrize("raw", ["1", 1.0, True])
def test_mapping_accepts_stringy_and_numeric_inputs(raw):
    assert map_fakeddit_2way_label(raw) == LABEL_REAL


@pytest.mark.parametrize("raw", [2, -1, "fake", None])
def test_unknown_raw_label_raises(raw):
    with pytest.raises(LabelMappingError):
        map_fakeddit_2way_label(raw)


def test_map_series_preserves_order():
    mapped = map_fakeddit_2way_series([1, 0, 0, 1])
    assert mapped.tolist() == [0, 1, 1, 0]


def test_validate_internal_labels_rejects_raw_multiclass():
    validate_internal_labels([0, 1, 1, 0])
    with pytest.raises(LabelMappingError):
        validate_internal_labels([0, 1, 5])


def test_label_distribution_counts_by_name():
    assert label_distribution([0, 0, 1]) == {"REAL": 2, "FAKE": 1}


def test_synthetic_manifest_labels_are_inverted_from_raw(synthetic_raw, synthetic_manifest):
    """전처리를 거친 manifest 라벨이 원본과 반대인지 end-to-end로 확인."""
    raw_by_id = dict(zip(synthetic_raw["id"], synthetic_raw["2_way_label"]))
    for row in synthetic_manifest.to_dict(orient="records"):
        assert row["label"] == FAKEDDIT_2WAY_TO_INTERNAL[int(raw_by_id[row["sample_id"]])]
        assert row["label"] != int(raw_by_id[row["sample_id"]])
