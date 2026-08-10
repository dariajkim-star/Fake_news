"""train/val/test 분할 테스트 — Story 1.2 AC3, AC7."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data.labels import label_distribution
from src.data.splits import (
    DEFAULT_RATIOS,
    SPLIT_NAMES,
    SplitError,
    assert_disjoint,
    split_by_official,
    split_reports,
    stratified_split,
    write_splits,
)


def _big_manifest(n: int = 200, fake_ratio: float = 0.3) -> pd.DataFrame:
    rows = []
    for index in range(n):
        label = 1 if index < int(n * fake_ratio) else 0
        rows.append(
            {
                "sample_id": f"b{index:04d}",
                "image_path": f"images/b{index:04d}.jpg",
                "title": f"headline {index}",
                "body": "",
                "text": f"headline {index}",
                "label": label,
                "source": "fakeddit",
            }
        )
    return pd.DataFrame(rows)


def test_split_is_reproducible_with_same_seed():
    manifest = _big_manifest()
    first = stratified_split(manifest, seed=42)
    second = stratified_split(manifest, seed=42)
    for name in SPLIT_NAMES:
        assert first[name]["sample_id"].tolist() == second[name]["sample_id"].tolist()


def test_split_changes_with_different_seed():
    manifest = _big_manifest()
    assert (
        stratified_split(manifest, seed=42)["train"]["sample_id"].tolist()
        != stratified_split(manifest, seed=7)["train"]["sample_id"].tolist()
    )


def test_split_is_invariant_to_input_row_order():
    """입력 행 순서가 달라도 동일 분할이 나와야 재현성이 성립한다."""
    manifest = _big_manifest()
    shuffled = manifest.sample(frac=1.0, random_state=123).reset_index(drop=True)
    base = stratified_split(manifest, seed=42)
    other = stratified_split(shuffled, seed=42)
    for name in SPLIT_NAMES:
        assert base[name]["sample_id"].tolist() == other[name]["sample_id"].tolist()


def test_splits_cover_all_samples_without_overlap():
    manifest = _big_manifest()
    splits = stratified_split(manifest, seed=42)
    assert_disjoint(splits)
    total = sum(len(frame) for frame in splits.values())
    assert total == len(manifest)


def test_split_sizes_follow_ratios():
    manifest = _big_manifest(n=200)
    splits = stratified_split(manifest, DEFAULT_RATIOS, seed=42)
    assert len(splits["train"]) == pytest.approx(160, abs=2)
    assert len(splits["val"]) == pytest.approx(20, abs=2)
    assert len(splits["test"]) == pytest.approx(20, abs=2)


def test_label_distribution_is_preserved_per_split():
    manifest = _big_manifest(n=200, fake_ratio=0.3)
    splits = stratified_split(manifest, seed=42)
    for name, frame in splits.items():
        ratio = label_distribution(frame["label"])["FAKE"] / len(frame)
        assert ratio == pytest.approx(0.3, abs=0.05), name


def test_invalid_ratios_raise():
    manifest = _big_manifest(n=20)
    with pytest.raises(SplitError):
        stratified_split(manifest, {"train": 0.5, "val": 0.2, "test": 0.2})
    with pytest.raises(SplitError):
        stratified_split(manifest, {"train": 0.8, "val": 0.2})


def test_assert_disjoint_detects_leakage():
    manifest = _big_manifest(n=20)
    with pytest.raises(SplitError):
        assert_disjoint({"train": manifest, "test": manifest.head(1)})


def test_split_by_official_uses_given_ids():
    manifest = _big_manifest(n=30)
    ids = manifest["sample_id"].tolist()
    splits = split_by_official(
        manifest, {"train": ids[:20], "val": ids[20:25], "test": ids[25:]}
    )
    assert [len(splits[name]) for name in SPLIT_NAMES] == [20, 5, 5]
    assert_disjoint(splits)


def test_write_splits_creates_files_and_report(tmp_path):
    manifest = _big_manifest(n=50)
    splits = stratified_split(manifest, seed=1)
    paths = write_splits(splits, tmp_path / "processed")
    for name in SPLIT_NAMES:
        assert paths[name].is_file()
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert [entry["name"] for entry in report] == list(SPLIT_NAMES)
    assert sum(entry["n_samples"] for entry in report) == len(manifest)


def test_split_reports_shape():
    reports = split_reports(stratified_split(_big_manifest(n=40), seed=3))
    assert all("label_counts" in report for report in reports)
