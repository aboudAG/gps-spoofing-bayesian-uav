

import numpy as np
import pandas as pd

from config.config import SimulationConfig


def simulate_true_trajectory(cfg: SimulationConfig) -> pd.DataFrame:
    """Génère la trajectoire réelle (vérité terrain) de l'UAV.

    Parameters
    ----------
    cfg : SimulationConfig
        Paramètres de simulation (nombre de pas, pas de temps, seed, etc.)

    Returns
    -------
    pd.DataFrame
        Colonnes : t, x_true, y_true, z_true, vx_true, vy_true, vz_true,
        ax_true, ay_true, az_true
    """
    rng = np.random.default_rng(cfg.seed)

    t = np.arange(cfg.n_steps) * cfg.dt
    total_time = t[-1] if len(t) > 1 else 1.0

    # Vitesse angulaire nécessaire pour parcourir la trajectoire circulaire
    # à la vitesse de croisière moyenne.
    omega = cfg.cruise_speed / cfg.trajectory_radius

    # Légère variation aléatoire (marche lente) de la vitesse angulaire pour
    # simuler des corrections de cap naturelles.
    speed_noise = rng.normal(0, cfg.speed_variation * 0.02, size=cfg.n_steps)
    omega_series = omega + np.cumsum(speed_noise) / cfg.n_steps

    theta = np.cumsum(omega_series * cfg.dt)

    x_true = cfg.trajectory_radius * np.cos(theta)
    y_true = cfg.trajectory_radius * np.sin(theta)

    # Altitude : oscillation lente autour de l'altitude de croisière.
    z_true = cfg.cruise_altitude + cfg.altitude_variation * np.sin(
        2 * np.pi * t / max(total_time, 1e-6) * 3
    )

    # Vitesses par différentiation numérique (schéma centré)
    vx_true = np.gradient(x_true, cfg.dt)
    vy_true = np.gradient(y_true, cfg.dt)
    vz_true = np.gradient(z_true, cfg.dt)

    # Accélérations par différentiation numérique
    ax_true = np.gradient(vx_true, cfg.dt)
    ay_true = np.gradient(vy_true, cfg.dt)
    az_true = np.gradient(vz_true, cfg.dt)

    return pd.DataFrame({
        "t": t,
        "x_true": x_true,
        "y_true": y_true,
        "z_true": z_true,
        "vx_true": vx_true,
        "vy_true": vy_true,
        "vz_true": vz_true,
        "ax_true": ax_true,
        "ay_true": ay_true,
        "az_true": az_true,
    })
