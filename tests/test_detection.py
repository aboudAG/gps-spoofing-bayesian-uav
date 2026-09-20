"""Tests du module de détection : application du seuil de décision."""

import numpy as np

from src.detection.detector import classify, label_to_str


def test_classify_threshold_basic():
    probs = np.array([0.1, 0.4, 0.5, 0.6, 0.9])
    preds = classify(probs, threshold=0.5)
    np.testing.assert_array_equal(preds, [0, 0, 1, 1, 1])


def test_classify_all_below_threshold():
    probs = np.array([0.0, 0.1, 0.2])
    preds = classify(probs, threshold=0.5)
    assert np.all(preds == 0)


def test_classify_all_above_threshold():
    probs = np.array([0.6, 0.7, 0.99])
    preds = classify(probs, threshold=0.5)
    assert np.all(preds == 1)


def test_classify_threshold_is_configurable():
    probs = np.array([0.3, 0.3, 0.3])
    assert np.all(classify(probs, threshold=0.2) == 1)
    assert np.all(classify(probs, threshold=0.4) == 0)


def test_label_to_str():
    assert label_to_str(1) == "SPOOFING"
    assert label_to_str(0) == "NORMAL"
