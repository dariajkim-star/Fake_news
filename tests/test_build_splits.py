"""build_splits의 불변식 검증.

여기서 지키려는 것은 성능이 아니라 **결과를 무의미하게 만드는 오류**다:
family leakage, 평가 분포 훼손, 재현 불가능성.
"""
from __future__ import annotations

import random

import pytest

from src.preprocess.build_splits import (
    SPLIT_FRACTIONS,
    Family,
    assign_splits,
    build_families,
    cap_fakes,
    verify_no_leakage,
)


def make_meta(n_families: int = 60, fakes_per: int = 8) -> dict[str, dict]:
    """REAL 1개 + FAKE n개 구조의 합성 metadata."""
    meta: dict[str, dict] = {}
    for i in range(n_families):
        real = f"real_{i:03d}.mp4"
        meta[real] = {"label": "REAL", "split": "train", "_part": "p0"}
        for j in range(fakes_per):
            meta[f"fake_{i:03d}_{j}.mp4"] = {
                "label": "FAKE", "split": "train", "original": real, "_part": "p0",
            }
    return meta


def test_families_group_by_original():
    fams = build_families(make_meta(3, 4), present=None)
    assert len(fams) == 3
    for fam in fams.values():
        assert fam.real is not None
        assert len(fam.fakes) == 4
        assert fam.size == 5


def test_orphan_fake_keeps_own_family():
    """original이 배포분에 없는 FAKE도 자체 family로 남아야 한다 (REAL만 없음)."""
    meta = {"fake_x.mp4": {"label": "FAKE", "original": "missing.mp4", "_part": "p0"}}
    fams = build_families(meta, present=None)
    assert len(fams) == 1
    fam = fams["missing.mp4"]
    assert fam.real is None and fam.fakes == ["fake_x.mp4"]


def test_present_filter_excludes_undownloaded():
    meta = make_meta(2, 3)
    present = {"real_000.mp4", "fake_000_0.mp4"}
    fams = build_families(meta, present=present)
    assert sum(f.size for f in fams.values()) == 2


def test_no_family_crosses_splits():
    """가장 중요한 불변식 — family가 두 split에 걸치면 얼굴을 외운다."""
    meta = make_meta()
    fams = build_families(meta, present=None)
    assignment = assign_splits(fams, seed=42)
    rows = [
        {"video": v, "family": key, "split": assignment[key]}
        for key, fam in fams.items()
        for v in ([fam.real] if fam.real else []) + fam.fakes
    ]
    verify_no_leakage(rows)  # 실패 시 SystemExit


def test_verify_no_leakage_detects_violation():
    rows = [
        {"video": "a.mp4", "family": "f1", "split": "train"},
        {"video": "b.mp4", "family": "f1", "split": "test"},   # 같은 family가 갈림
    ]
    with pytest.raises(SystemExit, match="LEAKAGE"):
        verify_no_leakage(rows)


def test_split_fractions_apply_to_families_not_videos():
    """분할 비율의 분모는 family다 — 독립 표본이 family이기 때문."""
    fams = build_families(make_meta(200, 8), present=None)
    assignment = assign_splits(fams, seed=42)
    for name, frac in SPLIT_FRACTIONS.items():
        share = sum(1 for s in assignment.values() if s == name) / len(fams)
        assert abs(share - frac) < 0.03, f"{name}: {share:.3f} vs {frac}"


def test_large_families_not_hoarded_by_train():
    """크기 층화 회귀 방지 — 예전 구현은 큰 family를 전부 train에 몰아넣었다."""
    sizes = [2] * 100 + [30] * 100          # 극단적으로 이봉분포
    meta: dict[str, dict] = {}
    for i, n in enumerate(sizes):
        real = f"real_{i:03d}.mp4"
        meta[real] = {"label": "REAL", "_part": "p0"}
        for j in range(n - 1):
            meta[f"fake_{i:03d}_{j}.mp4"] = {"label": "FAKE", "original": real, "_part": "p0"}
    fams = build_families(meta, present=None)
    assignment = assign_splits(fams, seed=42)

    for name in SPLIT_FRACTIONS:
        members = [fams[k].size for k, s in assignment.items() if s == name]
        big = sum(1 for sz in members if sz > 10) / len(members)
        assert 0.3 < big < 0.7, f"{name}: 큰 family 비율 {big:.2f} — 층화 실패"


def test_fake_cap_limits_train_only():
    fam = Family(original="r.mp4", real="r.mp4", fakes=[f"f{i}.mp4" for i in range(30)])
    rng = random.Random(0)
    assert len(cap_fakes(fam, 5, rng)) == 5
    assert len(cap_fakes(fam, None, rng)) == 30   # val/test는 원본 분포 유지


def test_cap_is_deterministic_under_seed():
    fam = Family(original="r.mp4", real="r.mp4", fakes=[f"f{i}.mp4" for i in range(30)])
    a = cap_fakes(fam, 5, random.Random(42))
    b = cap_fakes(fam, 5, random.Random(42))
    assert a == b


def test_assignment_is_deterministic_under_seed():
    fams = build_families(make_meta(), present=None)
    assert assign_splits(fams, 42) == assign_splits(fams, 42)
    assert assign_splits(fams, 42) != assign_splits(fams, 7)
