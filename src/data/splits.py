"""train/val/test 분할 생성 (Story 1.2 AC3).

분할은 **파일로 고정 저장**되며, Epic 2~5의 모든 ablation이 동일 파일을
참조해야 비교가 성립한다 (NFR3/NFR6).

두 가지 경로를 지원한다:
1. Fakeddit 공식 분할(별도 tsv)이 있으면 그것을 사용 — `split_by_official`
2. 없으면 seed 고정 stratified split — `stratified_split`
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.preprocess import manifest_report, write_manifest

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
DEFAULT_RATIOS: dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}


class SplitError(ValueError):
    """분할 설정이 유효하지 않을 때."""


def _validate_ratios(ratios: Mapping[str, float]) -> dict[str, float]:
    missing = [name for name in SPLIT_NAMES if name not in ratios]
    if missing:
        raise SplitError(f"분할 비율에 {missing}가 없습니다")
    total = sum(float(ratios[name]) for name in SPLIT_NAMES)
    if abs(total - 1.0) > 1e-6:
        raise SplitError(f"분할 비율의 합이 1이어야 합니다 (현재 {total})")
    return {name: float(ratios[name]) for name in SPLIT_NAMES}


def stratified_split(
    manifest: pd.DataFrame,
    ratios: Mapping[str, float] | None = None,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """라벨 분포를 유지한 채 seed 고정으로 train/val/test를 나눈다.

    동일 (manifest, ratios, seed)이면 항상 동일한 분할이 나온다 — 정렬된
    sample_id 순서에서 시작해 라벨 그룹별로 셔플하므로 입력 행 순서에도 불변.
    """
    ratios = _validate_ratios(ratios or DEFAULT_RATIOS)
    ordered = manifest.sort_values("sample_id", kind="mergesort").reset_index(drop=True)

    parts: dict[str, list[pd.DataFrame]] = {name: [] for name in SPLIT_NAMES}
    for label in sorted(ordered["label"].unique()):
        group = ordered[ordered["label"] == label]
        rng = np.random.default_rng(int(seed) + int(label))
        permuted = group.iloc[rng.permutation(len(group))]

        n = len(permuted)
        n_train = int(round(n * ratios["train"]))
        n_val = int(round(n * ratios["val"]))
        # 반올림 잔차는 test가 흡수 — 총합 보존
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        parts["train"].append(permuted.iloc[:n_train])
        parts["val"].append(permuted.iloc[n_train : n_train + n_val])
        parts["test"].append(permuted.iloc[n_train + n_val :])

    splits: dict[str, pd.DataFrame] = {}
    for name in SPLIT_NAMES:
        frames = [frame for frame in parts[name] if len(frame)]
        combined = (
            pd.concat(frames, ignore_index=True)
            if frames
            else manifest.iloc[0:0].reset_index(drop=True)
        )
        splits[name] = combined.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    return splits


def split_by_official(
    manifest: pd.DataFrame,
    official_ids: Mapping[str, Sequence[str]],
) -> dict[str, pd.DataFrame]:
    """Fakeddit 공식 분할(split별 sample_id 목록)로 manifest를 나눈다."""
    splits: dict[str, pd.DataFrame] = {}
    for name in SPLIT_NAMES:
        ids = {str(value) for value in official_ids.get(name, [])}
        subset = manifest[manifest["sample_id"].astype(str).isin(ids)]
        splits[name] = subset.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    return splits


def assert_disjoint(splits: Mapping[str, pd.DataFrame]) -> None:
    """분할 간 sample_id 누수(leakage)가 없는지 검증한다."""
    seen: dict[str, str] = {}
    for name, frame in splits.items():
        for sample_id in frame["sample_id"].astype(str):
            if sample_id in seen:
                raise SplitError(f"sample_id {sample_id!r}가 {seen[sample_id]}와 {name}에 중복됩니다")
            seen[sample_id] = name


def split_reports(splits: Mapping[str, pd.DataFrame]) -> list[dict[str, object]]:
    """분할별 샘플 수·라벨 분포 리포트 (AC3)."""
    return [manifest_report(splits[name], name=name) for name in SPLIT_NAMES if name in splits]


def write_splits(splits: Mapping[str, pd.DataFrame], out_dir: str | Path) -> dict[str, Path]:
    """분할 manifest와 통계 리포트(json)를 저장한다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: write_manifest(frame, out_dir / f"{name}.csv") for name, frame in splits.items()}

    report_path = out_dir / "split_report.json"
    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(split_reports(splits), fp, ensure_ascii=False, indent=2)
    paths["report"] = report_path
    return paths
