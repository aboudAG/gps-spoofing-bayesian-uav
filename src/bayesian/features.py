

import numpy as np
import pandas as pd

from config.config import FeatureConfig, SimulationConfig


def _rolling_regression_slope(values: np.ndarray, window: int, dt: float) -> np.ndarray:
    """Estime, pour chaque instant t, la pente locale (vitesse) d'une série
    de positions par régression linéaire sur les `window` derniers points
    (fenêtre causale). Réduit fortement le bruit de mesure par rapport à une
    simple différence d'échantillons consécutifs, car l'incertitude d'une
    pente de régression décroît rapidement avec le nombre de points utilisés.
    """
    series = pd.Series(values)

    def slope(arr):
        idx = np.arange(len(arr))
        if len(arr) < 2 or np.all(arr == arr[0]):
            return 0.0
        return np.polyfit(idx, arr, 1)[0]

    raw_slope = series.rolling(window=window, min_periods=2).apply(slope, raw=True)
    raw_slope = raw_slope.fillna(0.0).values
    return raw_slope / dt


def compute_raw_features(sensors: pd.DataFrame, sim_cfg: SimulationConfig,
                          feature_cfg: FeatureConfig = None) -> pd.DataFrame:
    """Calcule les indicateurs continus de cohérence inter-capteurs.

    Parameters
    ----------
    sensors : pd.DataFrame
        Mesures capteurs (potentiellement spoofées sur le flux GPS).
    sim_cfg : SimulationConfig
    feature_cfg : FeatureConfig

    Returns
    -------
    pd.DataFrame avec colonnes t, vel_residual, alt_residual, pos_jump
    """
    dt = sim_cfg.dt
    window = feature_cfg.window_size if feature_cfg is not None else 15

    # --- vel_residual : vitesse "vue par la position" (régression glissante)
    # vs vitesse GPS Doppler rapportée ---
    vx_from_pos = _rolling_regression_slope(sensors["gps_x"].values, window, dt)
    vy_from_pos = _rolling_regression_slope(sensors["gps_y"].values, window, dt)

    vel_residual = np.sqrt(
        (vx_from_pos - sensors["gps_vx"].values) ** 2
        + (vy_from_pos - sensors["gps_vy"].values) ** 2
    )

    # --- alt_residual : GPS vs altimètre barométrique (capteur indépendant) ---
    alt_residual = np.abs(sensors["alt_z"].values - sensors["gps_z"].values)

    # --- pos_jump : écart instantané déplacement observé vs déplacement
    # attendu d'après la vitesse Doppler rapportée (non lissé, pour capter
    # les discontinuités brutales) ---
    dx = np.diff(sensors["gps_x"].values, prepend=sensors["gps_x"].values[0])
    dy = np.diff(sensors["gps_y"].values, prepend=sensors["gps_y"].values[0])
    observed_disp = np.sqrt(dx ** 2 + dy ** 2)
    expected_disp = np.sqrt(sensors["gps_vx"].values ** 2 + sensors["gps_vy"].values ** 2) * dt
    pos_jump = np.abs(observed_disp - expected_disp)

    return pd.DataFrame({
        "t": sensors["t"].values,
        "vel_residual": vel_residual,
        "alt_residual": alt_residual,
        "pos_jump": pos_jump,
    })


def _discretize_series(series: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Discrétise une série continue selon des bornes fixées (2 bornes -> 3 catégories)."""
    return np.digitize(series, edges)


def fit_discretization_edges(raw_features: pd.DataFrame, cfg: FeatureConfig) -> dict:
    """Détermine les bornes de discrétisation (quantiles) à partir de données
    d'entraînement (idéalement un mélange équilibré de normal/attaque).

    Returns
    -------
    dict : {feature_name: np.ndarray(edges)}
    """
    edges = {}
    for col, n_bins in [
        ("vel_residual", cfg.n_bins_residual),
        ("alt_residual", cfg.n_bins_residual),
        ("pos_jump", cfg.n_bins_jump),
    ]:
        quantiles = np.linspace(0, 100, n_bins + 1)[1:-1]
        edges[col] = np.percentile(raw_features[col].values, quantiles)
    return edges


def discretize_features(raw_features: pd.DataFrame, edges: dict) -> pd.DataFrame:
    """Applique la discrétisation à un jeu de features continues."""
    out = pd.DataFrame({"t": raw_features["t"].values})
    for col, e in edges.items():
        out[col] = _discretize_series(raw_features[col].values, e)
    return out
