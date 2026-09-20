"""Tests de l'extraction des indicateurs de cohérence inter-capteurs (features)."""

import numpy as np
import pytest

from config.config import DEFAULT_CONFIG
from src.simulation.uav import simulate_true_trajectory
from src.simulation.sensors import simulate_sensors
from src.simulation.spoofing import apply_spoofing
from src.bayesian.features import (compute_raw_features, discretize_features,
                                    fit_discretization_edges)


@pytest.fixture
def cfg():
    return DEFAULT_CONFIG


@pytest.fixture
def normal_sensors(cfg):
    traj = simulate_true_trajectory(cfg.simulation)
    return simulate_sensors(traj, cfg.simulation, cfg.noise)


def test_compute_raw_features_columns(cfg, normal_sensors):
    raw = compute_raw_features(normal_sensors, cfg.simulation, cfg.features)
    assert set(raw.columns) == {"t", "vel_residual", "alt_residual", "pos_jump"}
    assert len(raw) == len(normal_sensors)


def test_raw_features_are_non_negative(cfg, normal_sensors):
    raw = compute_raw_features(normal_sensors, cfg.simulation, cfg.features)
    assert (raw["vel_residual"] >= 0).all()
    assert (raw["alt_residual"] >= 0).all()
    assert (raw["pos_jump"] >= 0).all()


def test_vel_residual_higher_under_spoofing(cfg):
    """L'indicateur vel_residual doit, en moyenne, être plus élevé pendant une
    attaque de spoofing progressif qu'en vol normal."""
    traj = simulate_true_trajectory(cfg.simulation)
    sensors = simulate_sensors(traj, cfg.simulation, cfg.noise)
    spoofed, label = apply_spoofing(sensors, cfg.simulation, cfg.spoofing, "progressive")

    raw = compute_raw_features(spoofed, cfg.simulation, cfg.features)
    mean_attack = raw["vel_residual"].values[label.astype(bool)].mean()
    mean_normal = raw["vel_residual"].values[~label.astype(bool)].mean()
    assert mean_attack > mean_normal


def test_discretization_produces_expected_number_of_bins(cfg, normal_sensors):
    raw = compute_raw_features(normal_sensors, cfg.simulation, cfg.features)
    edges = fit_discretization_edges(raw, cfg.features)
    discretized = discretize_features(raw, edges)

    for col in ["vel_residual", "alt_residual", "pos_jump"]:
        n_unique = discretized[col].nunique()
        assert n_unique <= cfg.features.n_bins_residual
        assert discretized[col].min() >= 0


def test_discretization_edges_keys(cfg, normal_sensors):
    raw = compute_raw_features(normal_sensors, cfg.simulation, cfg.features)
    edges = fit_discretization_edges(raw, cfg.features)
    assert set(edges.keys()) == {"vel_residual", "alt_residual", "pos_jump"}
