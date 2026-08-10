"""지표 계산 테스트 (Story 1.1 AC5, AC7).

라벨 규약 0=REAL, 1=FAKE이며 Precision/Recall/F1은 FAKE 기준이다.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from sklearn.metrics import f1_score, roc_auc_score

from src.evaluation.metrics import METRIC_KEYS, compute_metrics, confusion, save_metrics


def test_perfect_prediction_scores_one():
    metrics = compute_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    for key in METRIC_KEYS:
        assert metrics[key] == pytest.approx(1.0)


def test_matches_sklearn_reference():
    y_true = [0, 1, 1, 0, 1, 0, 1, 0]
    y_prob = [0.2, 0.8, 0.4, 0.6, 0.9, 0.1, 0.55, 0.45]
    metrics = compute_metrics(y_true, y_prob)
    y_pred = (np.asarray(y_prob) >= 0.5).astype(int)
    assert metrics["f1"] == pytest.approx(f1_score(y_true, y_pred, pos_label=1))
    assert metrics["auroc"] == pytest.approx(roc_auc_score(y_true, y_prob))
    assert metrics["accuracy"] == pytest.approx((y_pred == np.asarray(y_true)).mean())


def test_positive_class_is_fake():
    # 전부 FAKE(1)로 예측 → FAKE recall은 1.0, precision은 실제 FAKE 비율
    metrics = compute_metrics([0, 0, 1, 1], [0.9, 0.9, 0.9, 0.9])
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(0.5)


def test_threshold_is_applied():
    y_true = [0, 1]
    y_prob = [0.4, 0.6]
    assert compute_metrics(y_true, y_prob, threshold=0.5)["accuracy"] == pytest.approx(1.0)
    assert compute_metrics(y_true, y_prob, threshold=0.7)["accuracy"] == pytest.approx(0.5)


def test_single_class_auroc_is_nan_not_crash():
    metrics = compute_metrics([1, 1, 1], [0.9, 0.8, 0.7])
    assert math.isnan(metrics["auroc"])
    assert metrics["recall"] == pytest.approx(1.0)


def test_length_mismatch_and_empty_input_raise():
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0.5])
    with pytest.raises(ValueError):
        compute_metrics([], [])


def test_confusion_counts():
    assert confusion([0, 0, 1, 1], [0.1, 0.9, 0.2, 0.8]) == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}


def test_save_metrics_writes_json(tmp_path):
    metrics = compute_metrics([0, 1], [0.1, 0.9])
    path = save_metrics(metrics, tmp_path, extra={"exp_name": "t", "split": "val"})
    assert path.name == "metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["exp_name"] == "t"
    assert payload["f1"] == pytest.approx(1.0)
