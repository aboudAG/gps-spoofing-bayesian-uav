"""Tests de la simulation : trajectoire réelle, capteurs bruités, scénarios de spoofing."""

import numpy as np
import pytest

from config.config import DEFAULT_CONFIG
from src.simulation.uav import simulate_true_trajectory
from src.simulation.sensors import simulate_sensors
from src.simulation.spoofing import apply_spoofing


@pytest.fixture
def cfg():
    return DEFAULT_CONFIG


def test_true_trajectory_shape(cfg):
    traj = simulate_true_trajectory(cfg.simulation)
    assert len(traj) == cfg.simulation.n_steps
    expected_cols = {"t", "x_true", "y_true", "z_true",
                      "vx_true", "vy_true", "vz_true",
                      "ax_true", "ay_true", "az_true"}
    assert expected_cols.issubset(set(traj.columns))


def test_true_trajectory_reproducibility(cfg):
    """Une même seed doit produire exactement la même trajectoire."""
    traj1 = simulate_true_trajectory(cfg.simulation)
    traj2 = simulate_true_trajectory(cfg.simulation)
    np.testing.assert_allclose(traj1["x_true"].values, traj2["x_true"].values)
    np.testing.assert_allclose(traj1["y_true"].values, traj2["y_true"].values)


def test_trajectory_altitude_reasonable(cfg):
    """L'altitude simulée doit rester proche de l'altitude de croisière configurée."""
    traj = simulate_true_trajectory(cfg.simulation)
    assert np.all(np.abs(traj["z_true"] - cfg.simulation.cruise_altitude)
                  <= cfg.simulation.altitude_variation + 1e-6)


def test_sensors_add_noise(cfg):
    """Les mesures capteurs ne doivent pas être identiques à la trajectoire réelle
    (le bruit doit avoir un effet mesurable)."""
    traj = simulate_true_trajectory(cfg.simulation)
    sensors = simulate_sensors(traj, cfg.simulation, cfg.noise)
    assert not np.allclose(sensors["gps_x"].values, traj["x_true"].values)
    assert sensors["gps_x"].std() > 0


def test_sensors_noise_scales_with_multiplier(cfg):
    """Un multiplicateur de bruit plus élevé doit produire un écart-type de résidu
    de mesure plus élevé (en moyenne, sur plusieurs tirages)."""
    traj = simulate_true_trajectory(cfg.simulation)
    low = simulate_sensors(traj, cfg.simulation, cfg.noise, noise_multiplier=0.5, seed_offset=1)
    high = simulate_sensors(traj, cfg.simulation, cfg.noise, noise_multiplier=3.0, seed_offset=1)

    resid_low = np.std(low["gps_x"].values - traj["x_true"].values)
    resid_high = np.std(high["gps_x"].values - traj["x_true"].values)
    assert resid_high > resid_low


def test_spoofing_none_leaves_gps_unchanged(cfg):
    traj = simulate_true_trajectory(cfg.simulation)
    sensors = simulate_sensors(traj, cfg.simulation, cfg.noise)
    spoofed, label = apply_spoofing(sensors, cfg.simulation, cfg.spoofing, "none")
    np.testing.assert_allclose(spoofed["gps_x"].values, sensors["gps_x"].values)
    assert np.all(label == 0)


@pytest.mark.parametrize("scenario", ["progressive", "brutal", "intermittent"])
def test_spoofing_scenarios_modify_gps_and_label_attacks(cfg, scenario):
    traj = simulate_true_trajectory(cfg.simulation)
    sensors = simulate_sensors(traj, cfg.simulation, cfg.noise)
    spoofed, label = apply_spoofing(sensors, cfg.simulation, cfg.spoofing, scenario)

    # Le label doit contenir au moins une période d'attaque et une période normale
    assert label.sum() > 0
    assert label.sum() < len(label)

    # La position GPS doit être modifiée pendant la période d'attaque
    diff = np.abs(spoofed["gps_x"].values - sensors["gps_x"].values)
    assert diff[label.astype(bool)].mean() > diff[~label.astype(bool)].mean()


def test_spoofing_unknown_scenario_raises(cfg):
    traj = simulate_true_trajectory(cfg.simulation)
    sensors = simulate_sensors(traj, cfg.simulation, cfg.noise)
    with pytest.raises(ValueError):
        apply_spoofing(sensors, cfg.simulation, cfg.spoofing, "scenario_inexistant")


def test_spoofing_intensity_override_increases_offset(cfg):
    traj = simulate_true_trajectory(cfg.simulation)
    sensors = simulate_sensors(traj, cfg.simulation, cfg.noise)
    low, label_low = apply_spoofing(sensors, cfg.simulation, cfg.spoofing,
                                     "progressive", intensity_override=5.0)
    high, label_high = apply_spoofing(sensors, cfg.simulation, cfg.spoofing,
                                       "progressive", intensity_override=50.0)

    diff_low = np.abs(low["gps_x"].values - sensors["gps_x"].values)[label_low.astype(bool)].mean()
    diff_high = np.abs(high["gps_x"].values - sensors["gps_x"].values)[label_high.astype(bool)].mean()
    assert diff_high > diff_low
