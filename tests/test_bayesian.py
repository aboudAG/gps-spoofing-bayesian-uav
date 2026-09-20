"""Tests du réseau bayésien : construction de la structure, entraînement des
CPD, et inférence probabiliste."""

import numpy as np
import pytest

from config.config import DEFAULT_CONFIG
from src.bayesian.network import (FEATURE_NODES, TARGET_NODE,
                                   build_network_structure, fit_network,
                                   infer_spoofing_probabilities)
from src.pipeline import build_training_set, train_model
from src.bayesian.features import discretize_features, fit_discretization_edges


@pytest.fixture
def cfg():
    return DEFAULT_CONFIG


def test_build_network_structure_edges():
    model = build_network_structure()
    edges = set(model.edges())
    assert edges == {(TARGET_NODE, f) for f in FEATURE_NODES}


def test_fit_network_produces_valid_model(cfg):
    training_raw = build_training_set(cfg)
    edges = fit_discretization_edges(training_raw, cfg.features)
    discretized = discretize_features(training_raw, edges)
    discretized["spoofing"] = training_raw["spoofing"].values

    structure = build_network_structure()
    model = fit_network(structure, discretized, cfg.bayesian)

    # Le modèle doit être valide (CPD cohérentes, réseau acyclique, etc.)
    assert model.check_model()
    # Une CPD doit exister pour chaque nœud du réseau
    cpd_vars = {cpd.variable for cpd in model.get_cpds()}
    assert cpd_vars == {TARGET_NODE} | set(FEATURE_NODES)


def test_inference_returns_probabilities_in_range(cfg):
    model, edges = train_model(cfg)

    training_raw = build_training_set(cfg)
    discretized = discretize_features(training_raw, edges)

    probs, mean_time = infer_spoofing_probabilities(model, discretized)
    assert len(probs) == len(discretized)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert mean_time >= 0.0


def test_inference_probability_higher_for_attack_evidence(cfg):
    """Sur des données labellisées, la probabilité moyenne inférée doit être
    significativement plus élevée pour les échantillons réellement attaqués."""
    model, edges = train_model(cfg)

    training_raw = build_training_set(cfg)
    discretized = discretize_features(training_raw, edges)
    probs, _ = infer_spoofing_probabilities(model, discretized)

    label = training_raw["spoofing"].values.astype(bool)
    assert probs[label].mean() > probs[~label].mean()
