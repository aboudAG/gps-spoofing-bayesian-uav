

import json
import os

import numpy as np
import pandas as pd

from config.config import DEFAULT_CONFIG, Config
from src.simulation.uav import simulate_true_trajectory
from src.simulation.sensors import simulate_sensors
from src.simulation.spoofing import apply_spoofing
from src.bayesian.features import (compute_raw_features, discretize_features,
                                    fit_discretization_edges)
from src.bayesian.network import (build_network_structure, fit_network,
                                   infer_spoofing_probabilities, TARGET_NODE,
                                   FEATURE_NODES)
from src.detection.detector import classify
from src.evaluation.metrics import compute_metrics
from src.visualization import plots


SCENARIOS = ["none", "progressive", "brutal", "intermittent"]
SCENARIO_LABELS_FR = {
    "none": "vol_normal",
    "progressive": "spoofing_progressif",
    "brutal": "spoofing_brutal",
    "intermittent": "spoofing_intermittent",
}


def generate_scenario(cfg: Config, scenario: str, seed_offset: int,
                       noise_multiplier: float = 1.0,
                       intensity_override: float = None):
    """Génère une trajectoire réelle, des mesures capteurs et applique un
    scénario de spoofing donné. Retourne (true_traj, sensors, label)."""
    true_traj = simulate_true_trajectory(cfg.simulation)
    sensors = simulate_sensors(true_traj, cfg.simulation, cfg.noise,
                                noise_multiplier=noise_multiplier,
                                seed_offset=seed_offset)
    sensors, label = apply_spoofing(sensors, cfg.simulation, cfg.spoofing,
                                     scenario, intensity_override=intensity_override)
    return true_traj, sensors, label


def build_training_set(cfg: Config) -> pd.DataFrame:
    """Construit un jeu de données d'entraînement équilibré (normal + 3
    scénarios d'attaque) pour estimer les CPD du réseau bayésien."""
    frames = []
    for i, scenario in enumerate(SCENARIOS):
        _, sensors, label = generate_scenario(cfg, scenario, seed_offset=100 + i)
        raw = compute_raw_features(sensors, cfg.simulation, cfg.features)
        raw["spoofing"] = label
        frames.append(raw)
    return pd.concat(frames, ignore_index=True)


def train_model(cfg: Config):
    """Entraîne la discrétisation et le réseau bayésien. Retourne (model, edges)."""
    training_raw = build_training_set(cfg)
    edges = fit_discretization_edges(training_raw, cfg.features)

    discretized = discretize_features(training_raw, edges)
    discretized["spoofing"] = training_raw["spoofing"].values

    structure = build_network_structure()
    model = fit_network(structure, discretized, cfg.bayesian)
    return model, edges


def run_scenario_pipeline(cfg: Config, model, edges: dict, scenario: str,
                           seed_offset: int, out_dir_figs: str,
                           noise_multiplier: float = 1.0,
                           intensity_override: float = None,
                           generate_plots: bool = True) -> dict:
    """Exécute la détection complète sur un scénario de test et calcule les métriques."""
    true_traj, sensors, label = generate_scenario(
        cfg, scenario, seed_offset=seed_offset,
        noise_multiplier=noise_multiplier, intensity_override=intensity_override)

    raw_features = compute_raw_features(sensors, cfg.simulation, cfg.features)
    disc_features = discretize_features(raw_features, edges)

    probs, mean_inference_time = infer_spoofing_probabilities(model, disc_features)
    predictions = classify(probs, cfg.bayesian.detection_threshold)

    metrics = compute_metrics(label, predictions)
    metrics["mean_inference_time_sec"] = mean_inference_time
    metrics["scenario"] = scenario

    scenario_name = SCENARIO_LABELS_FR[scenario]

    if generate_plots:
        plots.plot_trajectory_comparison(true_traj, sensors, out_dir_figs, scenario_name)
        plots.plot_feature_evolution(raw_features["t"].values, raw_features, label,
                                      out_dir_figs, scenario_name)
        plots.plot_spoofing_probability(raw_features["t"].values, probs, label,
                                         cfg.bayesian.detection_threshold,
                                         out_dir_figs, scenario_name)
        plots.plot_detection_result(raw_features["t"].values, label, predictions,
                                     out_dir_figs, scenario_name)
        plots.plot_confusion_matrix(metrics["confusion_matrix"], out_dir_figs, scenario_name)
        plots.plot_metrics_bar(metrics, out_dir_figs, scenario_name)

    return metrics


def run_noise_sensitivity(cfg: Config, model, edges: dict, out_dir_figs: str) -> dict:
    """Expérience 4 : influence du bruit des capteurs sur les performances."""
    results = {"accuracy": [], "f1_score": [], "false_positive_rate": [], "recall": []}
    levels = cfg.experiment.noise_levels
    for level in levels:
        metrics = run_scenario_pipeline(cfg, model, edges, "progressive",
                                         seed_offset=500, out_dir_figs=out_dir_figs,
                                         noise_multiplier=level, generate_plots=False)
        for k in results:
            results[k].append(metrics[k])

    plots.plot_sensitivity(
        levels, {"Accuracy": results["accuracy"], "F1-score": results["f1_score"],
                 "Recall": results["recall"]},
        x_label="Multiplicateur de bruit capteur", y_label="Score",
        out_dir=out_dir_figs, name="sensibilite_bruit",
        title="Influence du bruit des capteurs sur la détection (scénario progressif)")

    return {"noise_levels": list(levels), **results}


def run_intensity_sensitivity(cfg: Config, model, edges: dict, out_dir_figs: str) -> dict:
    """Expérience 5 : influence de l'intensité du spoofing sur les performances."""
    results = {"accuracy": [], "f1_score": [], "recall": [], "false_positive_rate": []}
    intensities = cfg.experiment.spoofing_intensities
    for intensity in intensities:
        metrics = run_scenario_pipeline(cfg, model, edges, "progressive",
                                         seed_offset=700, out_dir_figs=out_dir_figs,
                                         intensity_override=intensity, generate_plots=False)
        for k in results:
            results[k].append(metrics[k])

    plots.plot_sensitivity(
        intensities, {"Accuracy": results["accuracy"], "F1-score": results["f1_score"],
                       "Recall": results["recall"]},
        x_label="Intensité du spoofing (m)", y_label="Score",
        out_dir=out_dir_figs, name="sensibilite_intensite",
        title="Influence de l'intensité du spoofing sur la détection")

    return {"intensities": list(intensities), **results}


def run_all(cfg: Config = DEFAULT_CONFIG) -> dict:
    """Exécute l'intégralité du pipeline (entraînement + 5 expériences) et
    sauvegarde les résultats sur disque."""
    os.makedirs(cfg.figures_dir, exist_ok=True)
    os.makedirs(cfg.metrics_dir, exist_ok=True)

    print("[1/7] Entraînement du réseau bayésien sur données simulées...")
    model, edges = train_model(cfg)

    all_results = {}

    print("[2/7] Expérience 1 : données normales...")
    all_results["exp1_normal"] = run_scenario_pipeline(
        cfg, model, edges, "none", seed_offset=200, out_dir_figs=cfg.figures_dir)

    print("[3/7] Expérience 2 : GPS spoofing progressif...")
    all_results["exp2_progressif"] = run_scenario_pipeline(
        cfg, model, edges, "progressive", seed_offset=201, out_dir_figs=cfg.figures_dir)

    print("[4/7] Expérience 3 : GPS spoofing brutal...")
    all_results["exp3_brutal"] = run_scenario_pipeline(
        cfg, model, edges, "brutal", seed_offset=202, out_dir_figs=cfg.figures_dir)

    print("[5/7] (Bonus) Scénario intermittent...")
    all_results["bonus_intermittent"] = run_scenario_pipeline(
        cfg, model, edges, "intermittent", seed_offset=203, out_dir_figs=cfg.figures_dir)

    print("[6/7] Expérience 4 : sensibilité au bruit des capteurs...")
    all_results["exp4_bruit"] = run_noise_sensitivity(cfg, model, edges, cfg.figures_dir)

    print("[7/7] Expérience 5 : sensibilité à l'intensité du spoofing...")
    all_results["exp5_intensite"] = run_intensity_sensitivity(cfg, model, edges, cfg.figures_dir)

    metrics_path = os.path.join(cfg.metrics_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nRésultats sauvegardés dans {metrics_path}")
    print(f"Figures sauvegardées dans {cfg.figures_dir}/")

    return all_results
