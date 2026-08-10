"""전처리(클린 manifest 생성) 테스트 — Story 1.2 AC2, AC7."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.preprocess import (
    MANIFEST_COLUMNS,
    SOURCE_NAME,
    PreprocessError,
    build_manifest,
    clean_text,
    is_loadable_image,
    manifest_report,
    read_manifest,
    resolve_image_path,
    write_manifest,
)


def test_manifest_schema_matches_newssample(synthetic_manifest):
    assert tuple(synthetic_manifest.columns) == MANIFEST_COLUMNS
    assert set(synthetic_manifest["source"]) == {SOURCE_NAME}
    # Fakeddit은 clean_title만 제공 -> body는 빈 문자열
    assert set(synthetic_manifest["body"]) == {""}


def test_all_valid_samples_are_kept(synthetic_raw, synthetic_image_dir):
    manifest, stats = build_manifest(synthetic_raw, image_dir=synthetic_image_dir)
    assert stats["raw"] == len(synthetic_raw)
    assert stats["kept"] == len(synthetic_raw)
    assert len(manifest) == len(synthetic_raw)


def test_missing_image_samples_are_dropped(synthetic_raw, synthetic_image_dir):
    (synthetic_image_dir / "s003.jpg").unlink()
    manifest, stats = build_manifest(synthetic_raw, image_dir=synthetic_image_dir)
    assert stats["dropped_missing_image"] == 1
    assert "s003" not in set(manifest["sample_id"])


def test_corrupt_image_samples_are_dropped(synthetic_raw, synthetic_image_dir):
    """PIL로 열리지 않는 파일은 제외된다 (AC2 손상 파일 필터)."""
    corrupt = synthetic_image_dir / "s005.jpg"
    corrupt.write_bytes(b"not a real jpeg at all")
    manifest, stats = build_manifest(synthetic_raw, image_dir=synthetic_image_dir)
    assert stats["dropped_corrupt_image"] == 1
    assert "s005" not in set(manifest["sample_id"])
    assert is_loadable_image(corrupt) is False


def test_empty_text_samples_are_dropped(synthetic_raw, synthetic_image_dir):
    raw = synthetic_raw.copy()
    raw.loc[0, "clean_title"] = "   "
    raw.loc[1, "clean_title"] = None
    manifest, stats = build_manifest(raw, image_dir=synthetic_image_dir)
    assert stats["dropped_empty_text"] == 2
    assert len(manifest) == len(raw) - 2


def test_allow_missing_image_keeps_sample(synthetic_raw, tmp_path):
    manifest, stats = build_manifest(
        synthetic_raw, image_dir=tmp_path / "empty", require_image=False
    )
    assert stats["kept"] == len(synthetic_raw)
    assert set(manifest["image_path"]) == {""}


def test_relative_to_makes_portable_paths(synthetic_raw, synthetic_image_dir):
    root = synthetic_image_dir.parent
    manifest, _ = build_manifest(synthetic_raw, image_dir=synthetic_image_dir, relative_to=root)
    assert manifest.loc[0, "image_path"] == "images/s000.jpg"
    assert not Path(manifest.loc[0, "image_path"]).is_absolute()


def test_write_read_manifest_roundtrip(synthetic_manifest, tmp_path):
    path = write_manifest(synthetic_manifest, tmp_path / "processed" / "manifest.csv")
    loaded = read_manifest(path)
    pd.testing.assert_frame_equal(
        loaded.reset_index(drop=True), synthetic_manifest.reset_index(drop=True)
    )


def test_read_manifest_missing_file_raises(tmp_path):
    with pytest.raises(PreprocessError):
        read_manifest(tmp_path / "nope.csv")


def test_clean_text_normalizes_whitespace_and_nan():
    assert clean_text("  a   b \n c ") == "a b c"
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def test_resolve_image_path_tries_extensions(tmp_path, write_image):
    write_image(tmp_path / "x1.png")
    assert resolve_image_path(tmp_path, "x1").name == "x1.png"
    assert resolve_image_path(tmp_path, "missing") is None


def test_manifest_report_counts_and_ratios(synthetic_manifest):
    report = manifest_report(synthetic_manifest, name="full")
    assert report["n_samples"] == len(synthetic_manifest)
    assert sum(report["label_counts"].values()) == len(synthetic_manifest)
    assert pytest.approx(sum(report["label_ratio"].values()), abs=1e-6) == 1.0
