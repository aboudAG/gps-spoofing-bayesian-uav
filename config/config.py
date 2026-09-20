

from dataclasses import dataclass, field


@dataclass
class SimulationConfig:
    """Paramètres de la simulation temporelle du vol UAV."""
    n_steps: int = 600            # nombre de points temporels
    dt: float = 0.1               # pas de temps (s) -> fréquence d'échantillonnage 10 Hz
    seed: int = 42                # graine pour la reproductibilité

    # Vitesse de croisière nominale du drone (m/s) et variations
    cruise_speed: float = 8.0
    speed_variation: float = 1.5

    # Altitude nominale (m) et variation
    cruise_altitude: float = 50.0
    altitude_variation: float = 3.0

    # Rayon de la trajectoire circulaire simulée (m)
    trajectory_radius: float = 120.0


@dataclass
class NoiseConfig:
    """Niveaux de bruit sensoriel (écarts-types des bruits gaussiens)."""
    gps_position_std: float = 1.5      # bruit GPS sur la position (m), ~ précision GPS civile
    gps_velocity_std: float = 0.3      # bruit GPS sur la vitesse (m/s)
    imu_accel_std: float = 0.15        # bruit accéléromètre IMU (m/s^2)
    imu_gyro_std: float = 0.02         # bruit gyroscope IMU (rad/s)
    altimeter_std: float = 0.5         # bruit altimètre barométrique (m)


@dataclass
class SpoofingConfig:
    """Paramètres des scénarios d'attaque GPS spoofing."""
    # Spoofing progressif : dérive linéaire de la position GPS rapportée
    progressive_start_frac: float = 0.4     # début du spoofing (fraction de la durée totale)
    progressive_end_frac: float = 0.9       # fin de la dérive
    progressive_max_offset: float = 40.0    # décalage maximal atteint (m)

    # Spoofing brutal : saut soudain de position GPS
    brutal_start_frac: float = 0.5
    brutal_offset: float = 35.0             # amplitude du saut (m)

    # Spoofing intermittent (3e scénario) : sauts répétés de plus faible intensité
    intermittent_start_frac: float = 0.3
    intermittent_end_frac: float = 0.95
    intermittent_period_steps: int = 60     # période des oscillations de spoofing
    intermittent_offset: float = 15.0


@dataclass
class FeatureConfig:
    """Paramètres d'extraction des indicateurs de cohérence inter-capteurs."""
    # Fenêtre glissante (nombre de pas) utilisée pour estimer, par régression
    # linéaire locale, la vitesse "vue par la position GPS" (robuste au bruit
    # de mesure), comparée à la vitesse GPS Doppler rapportée (vel_residual).
    window_size: int = 15

    # Nombre de catégories (bins) utilisées pour discrétiser chaque variable
    # continue avant de l'injecter dans le réseau bayésien.
    n_bins_residual: int = 3
    n_bins_jump: int = 3
    n_bins_consistency: int = 3


@dataclass
class BayesianConfig:
    """Paramètres du réseau bayésien et de la détection."""
    detection_threshold: float = 0.5   # seuil sur P(Spoofing | observations)
    laplace_smoothing: float = 1.0     # pseudo-comptage pour l'estimation des CPD


@dataclass
class ExperimentConfig:
    """Paramètres pour les expériences de sensibilité (bruit / intensité)."""
    noise_levels: tuple = (0.5, 1.0, 1.5, 2.0, 3.0)   # multiplicateurs du bruit GPS de base
    spoofing_intensities: tuple = (10.0, 20.0, 30.0, 40.0, 50.0)  # décalage max testé (m)


@dataclass
class Config:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    spoofing: SpoofingConfig = field(default_factory=SpoofingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    bayesian: BayesianConfig = field(default_factory=BayesianConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    # Chemins de sortie
    data_dir: str = "data/generated"
    models_dir: str = "models"
    figures_dir: str = "results/figures"
    metrics_dir: str = "results/metrics"


DEFAULT_CONFIG = Config()
