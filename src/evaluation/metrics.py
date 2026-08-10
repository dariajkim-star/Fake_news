"""분류 지표 계산 (NFR4: Accuracy / Precision / Recall / F1 / AUROC).

Epic 1 baseline부터 Epic 5 ablation table까지 모든 실험이 이 함수 하나를
재사용해야 수치 비교가 성립한다. 라벨 규약은 0=REAL, 1=FAKE이고
Precision/Recall/F1은 FAKE(양성=1) 기준이다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

METRIC_KEYS: tuple[str, ...] = ("accuracy", "precision", "recall", "f1", "auroc")

POSITIVE_LABEL = 1  # FAKE


def compute_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """정답 라벨과 FAKE 확률로 지표를 계산한다.

    Args:
        y_true: 정답 라벨 (0=REAL, 1=FAKE).
        y_prob: FAKE일 확률 [0, 1]. 이진 예측값(0/1)을 넘겨도 동작하지만
            그 경우 AUROC는 의미가 제한적이다.
        threshold: FAKE 판정 임계값.

    Returns:
        accuracy/precision/recall/f1/auroc 딕셔너리. 한 클래스만 존재해
        AUROC가 정의되지 않으면 `float("nan")`을 넣는다.
    """
    y_true_arr = np.asarray(y_true).astype(int).ravel()
    y_prob_arr = np.asarray(y_prob, dtype=float).ravel()
    if y_true_arr.shape != y_prob_arr.shape:
        raise ValueError(
            f"y_true와 y_prob의 길이가 다릅니다: {y_true_arr.shape} vs {y_prob_arr.shape}"
        )
    if y_true_arr.size == 0:
        raise ValueError("빈 입력으로는 metric을 계산할 수 없습니다")

    y_pred = (y_prob_arr >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred, average="binary", pos_label=POSITIVE_LABEL, zero_division=0
    )
    try:
        auroc = float(roc_auc_score(y_true_arr, y_prob_arr))
    except ValueError:
        # 정답에 클래스가 하나뿐인 경우 (예: 아주 작은 val split)
        auroc = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": auroc,
    }


def confusion(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, int]:
    """오류 분석용 confusion matrix (Story 5.2에서 사용)."""
    y_true_arr = np.asarray(y_true).astype(int).ravel()
    y_pred = (np.asarray(y_prob, dtype=float).ravel() >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def save_metrics(
    metrics: Mapping[str, Any],
    out_dir: str | Path,
    filename: str = "metrics.json",
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """지표를 `<out_dir>/metrics.json`으로 저장한다 (AC5)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {**dict(metrics)}
    if extra:
        payload.update(dict(extra))
    path = out_dir / filename
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
    return path
