"""Tests du module d'évaluation : calcul des métriques de détection."""

import numpy as np

from src.evaluation.metrics import compute_metrics


def test_perfect_prediction():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1_score"] == 1.0
    assert m["false_positive_rate"] == 0.0


def test_all_wrong_prediction():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.0
    assert m["recall"] == 0.0
    assert m["false_positive_rate"] == 1.0


def test_metrics_confusion_matrix_counts():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1])
    m = compute_metrics(y_true, y_pred)
    assert m["true_negatives"] == 2
    assert m["false_positives"] == 1
    assert m["false_negatives"] == 1
    assert m["true_positives"] == 2


def test_metrics_no_positive_predictions_zero_division_safe():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])
    m = compute_metrics(y_true, y_pred)
    # precision indéfinie (0 prédiction positive) -> convention scikit-learn = 0
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["false_positive_rate"] == 0.0


def test_detection_rate_equals_recall():
    y_true = np.array([0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    m = compute_metrics(y_true, y_pred)
    assert m["detection_rate"] == m["recall"]
