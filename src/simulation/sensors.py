

import numpy as np
import pandas as pd

from config.config import NoiseConfig, SimulationConfig


def simulate_sensors(true_traj: pd.DataFrame, sim_cfg: SimulationConfig,
                      noise_cfg: NoiseConfig, noise_multiplier: float = 1.0,
                      seed_offset: int = 1) -> pd.DataFrame:
    """Génère les mesures bruitées des capteurs à partir de la trajectoire réelle.

    Parameters
    ----------
    true_traj : pd.DataFrame
        Trajectoire réelle produite par `simulate_true_trajectory`.
    sim_cfg : SimulationConfig
    noise_cfg : NoiseConfig
    noise_multiplier : float
        Multiplicateur global appliqué à tous les bruits (utilisé pour
        l'expérience de sensibilité au bruit).
    seed_offset : int
        Décalage de seed pour obtenir un bruit indépendant de la trajectoire.

    Returns
    -------
    pd.DataFrame
        Mesures capteurs : gps_x, gps_y, gps_z, gps_vx, gps_vy,
        imu_ax, imu_ay, imu_az, alt_z
    """
    rng = np.random.default_rng(sim_cfg.seed + seed_offset)
    n = len(true_traj)
    m = noise_multiplier

    gps_x = true_traj["x_true"].values + rng.normal(0, noise_cfg.gps_position_std * m, n)
    gps_y = true_traj["y_true"].values + rng.normal(0, noise_cfg.gps_position_std * m, n)
    gps_z = true_traj["z_true"].values + rng.normal(0, noise_cfg.gps_position_std * m, n)

    gps_vx = true_traj["vx_true"].values + rng.normal(0, noise_cfg.gps_velocity_std * m, n)
    gps_vy = true_traj["vy_true"].values + rng.normal(0, noise_cfg.gps_velocity_std * m, n)

    imu_ax = true_traj["ax_true"].values + rng.normal(0, noise_cfg.imu_accel_std * m, n)
    imu_ay = true_traj["ay_true"].values + rng.normal(0, noise_cfg.imu_accel_std * m, n)
    imu_az = true_traj["az_true"].values + rng.normal(0, noise_cfg.imu_accel_std * m, n)

    alt_z = true_traj["z_true"].values + rng.normal(0, noise_cfg.altimeter_std * m, n)

    return pd.DataFrame({
        "t": true_traj["t"].values,
        "gps_x": gps_x, "gps_y": gps_y, "gps_z": gps_z,
        "gps_vx": gps_vx, "gps_vy": gps_vy,
        "imu_ax": imu_ax, "imu_ay": imu_ay, "imu_az": imu_az,
        "alt_z": alt_z,
    })
