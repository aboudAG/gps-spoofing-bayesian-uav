

import numpy as np
import pandas as pd

from config.config import SimulationConfig, SpoofingConfig


def apply_spoofing(sensors: pd.DataFrame, sim_cfg: SimulationConfig,
                    spf_cfg: SpoofingConfig, scenario: str,
                    intensity_override: float = None) -> tuple:
    """Applique un scénario de spoofing aux mesures GPS.

    Parameters
    ----------
    sensors : pd.DataFrame
        Mesures capteurs non-attaquées (sortie de `simulate_sensors`).
    scenario : str
        Un parmi {"none", "progressive", "brutal", "intermittent"}.
    intensity_override : float, optional
        Si fourni, remplace l'amplitude par défaut du scénario (utilisé pour
        l'expérience de sensibilité à l'intensité du spoofing).

    Returns
    -------
    (pd.DataFrame, np.ndarray)
        Le DataFrame des capteurs avec GPS potentiellement modifié, et le
        vecteur binaire label (1 = spoofing actif à ce pas de temps, 0 sinon).
    """
    out = sensors.copy()
    n = len(out)
    label = np.zeros(n, dtype=int)
    rng = np.random.default_rng(sim_cfg.seed + 999)

    def persistent_jitter(active_mask: np.ndarray, amplitude: float, dt: float) -> np.ndarray:
        """Perturbation persistante d'un spoofer imparfait pendant l'attaque
        (somme de deux sinusoïdes de fréquences incommensurables, à l'amplitude
        proportionnelle à l'intensité du décalage principal)."""
        t = np.arange(n) * dt
        jitter = amplitude * (0.6 * np.sin(2 * np.pi * 0.35 * t + 0.7)
                               + 0.4 * np.sin(2 * np.pi * 0.9 * t + 2.1))
        return jitter * active_mask

    if scenario == "none":
        return out, label

    if scenario == "progressive":
        start = int(spf_cfg.progressive_start_frac * n)
        end = int(spf_cfg.progressive_end_frac * n)
        max_offset = intensity_override if intensity_override is not None \
            else spf_cfg.progressive_max_offset

        offset = np.zeros(n)
        ramp_len = max(end - start, 1)
        offset[start:end] = np.linspace(0, max_offset, ramp_len)
        offset[end:] = max_offset

        active_mask = np.zeros(n)
        active_mask[start:] = 1.0
        offset = offset + persistent_jitter(active_mask, 0.15 * max_offset, sim_cfg.dt)

        # Le décalage est appliqué le long d'une direction fixe (diagonale)
        direction = np.array([1.0, 0.5]) / np.linalg.norm([1.0, 0.5])
        out["gps_x"] = out["gps_x"] + offset * direction[0]
        out["gps_y"] = out["gps_y"] + offset * direction[1]

        label[start:] = 1

    elif scenario == "brutal":
        start = int(spf_cfg.brutal_start_frac * n)
        offset_amp = intensity_override if intensity_override is not None \
            else spf_cfg.brutal_offset

        offset = np.zeros(n)
        offset[start:] = offset_amp

        active_mask = np.zeros(n)
        active_mask[start:] = 1.0
        offset = offset + persistent_jitter(active_mask, 0.2 * offset_amp, sim_cfg.dt)

        direction = np.array([0.3, 1.0]) / np.linalg.norm([0.3, 1.0])
        out["gps_x"] = out["gps_x"] + offset * direction[0]
        out["gps_y"] = out["gps_y"] + offset * direction[1]

        label[start:] = 1

    elif scenario == "intermittent":
        start = int(spf_cfg.intermittent_start_frac * n)
        end = int(spf_cfg.intermittent_end_frac * n)
        amp = intensity_override if intensity_override is not None \
            else spf_cfg.intermittent_offset
        period = spf_cfg.intermittent_period_steps

        offset = np.zeros(n)
        idx = np.arange(start, end)
        # Créneau carré : spoofing actif la moitié du temps, par blocs de `period`
        active = ((idx - start) // period) % 2 == 0
        offset[idx] = np.where(active, amp, 0.0)

        active_mask = np.zeros(n)
        active_mask[idx] = active.astype(float)
        offset = offset + persistent_jitter(active_mask, 0.2 * amp, sim_cfg.dt)

        direction = np.array([-0.5, 1.0]) / np.linalg.norm([-0.5, 1.0])
        out["gps_x"] = out["gps_x"] + offset * direction[0]
        out["gps_y"] = out["gps_y"] + offset * direction[1]

        label[idx] = active.astype(int)

    else:
        raise ValueError(f"Scénario de spoofing inconnu : {scenario}")

    return out, label
