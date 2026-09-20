
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.config import DEFAULT_CONFIG, SimulationConfig
from src.pipeline import (
    build_training_set,
    generate_scenario,
)
from src.bayesian.features import (
    compute_raw_features,
    discretize_features,
    fit_discretization_edges,
)
from src.bayesian.network import (
    build_network_structure,
    fit_network,
    infer_spoofing_probabilities,
)
from src.detection.detector import classify
from src.evaluation.metrics import compute_metrics


SEEDS = [42, 43, 44, 45, 46]

SCENARIOS = [
    "progressive",
    "brutal",
    "intermittent",
]


def main():

    cfg = DEFAULT_CONFIG

    print("=" * 70)
    print("ANALYSE DE ROBUSTESSE MULTI-SEEDS")
    print("=" * 70)

    print("\nSeeds testés :")
    print(", ".join(map(str, SEEDS)))

    print("\nScénarios testés :")
    print(", ".join(SCENARIOS))

    results = []

    for seed in SEEDS:

        print("\n" + "-" * 70)
        print(f"SEED : {seed}")
        print("-" * 70)

        # -----------------------------------------------------
        # Configuration avec le seed courant
        # -----------------------------------------------------

        sim_cfg = SimulationConfig(
            n_steps=cfg.simulation.n_steps,
            dt=cfg.simulation.dt,
            seed=seed,
            cruise_speed=cfg.simulation.cruise_speed,
            speed_variation=cfg.simulation.speed_variation,
            cruise_altitude=cfg.simulation.cruise_altitude,
            altitude_variation=cfg.simulation.altitude_variation,
            trajectory_radius=cfg.simulation.trajectory_radius,
        )

        # Recréer la configuration globale
        cfg_seed = cfg.__class__(
            simulation=sim_cfg,
            noise=cfg.noise,
            spoofing=cfg.spoofing,
            features=cfg.features,
            bayesian=cfg.bayesian,
            experiment=cfg.experiment,
        )

        # -----------------------------------------------------
        # Entraînement
        # -----------------------------------------------------

        print("  Génération du jeu d'entraînement...")

        training_raw = build_training_set(cfg_seed)

        edges = fit_discretization_edges(
            training_raw,
            cfg_seed.features,
        )

        training_discretized = discretize_features(
            training_raw,
            edges,
        )

        training_discretized["spoofing"] = (
            training_raw["spoofing"].values
        )

        structure = build_network_structure()

        model = fit_network(
            structure,
            training_discretized,
            cfg_seed.bayesian,
        )

        # -----------------------------------------------------
        # Évaluation
        # -----------------------------------------------------

        for scenario_index, scenario in enumerate(SCENARIOS):

            if scenario not in SCENARIOS:
                continue

            _, sensors, labels = generate_scenario(
                cfg_seed,
                scenario,
                seed_offset=500 + scenario_index,
            )

            raw_features = compute_raw_features(
                sensors,
                cfg_seed.simulation,
                cfg_seed.features,
            )

            discretized_features = discretize_features(
                raw_features,
                edges,
            )

            probabilities, inference_time = (
                infer_spoofing_probabilities(
                    model,
                    discretized_features,
                )
            )

            predictions = classify(
                probabilities,
                cfg_seed.bayesian.detection_threshold,
            )

            metrics = compute_metrics(
                labels,
                predictions,
            )

            results.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                    "false_positive_rate": metrics[
                        "false_positive_rate"
                    ],
                    "mean_inference_time_sec": inference_time,
                }
            )

            print(
                f"  {scenario:12s} | "
                f"F1={metrics['f1_score']:.4f} | "
                f"Recall={metrics['recall']:.4f} | "
                f"FPR={metrics['false_positive_rate']:.4f}"
            )

    # ---------------------------------------------------------
    # Résultats
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    output_dir = os.path.join(
        PROJECT_ROOT,
        "results",
        "metrics",
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    output_path = os.path.join(
        output_dir,
        "multi_seed.csv",
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Moyennes et écarts-types
    # ---------------------------------------------------------

    summary = (
        results_df
        .groupby("scenario")[
            [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "false_positive_rate",
            ]
        ]
        .agg(["mean", "std"])
    )

    summary_path = os.path.join(
        output_dir,
        "multi_seed_summary.csv",
    )

    summary.to_csv(summary_path)

    print("\n" + "=" * 70)
    print("RÉSULTATS PAR SCÉNARIO")
    print("=" * 70)

    print(summary.to_string())

    # ---------------------------------------------------------
    # Résumé global
    # ---------------------------------------------------------

    global_mean = results_df["f1_score"].mean()
    global_std = results_df["f1_score"].std()

    print("\n" + "=" * 70)
    print("RÉSULTAT GLOBAL")
    print("=" * 70)

    print(
        f"F1 moyen : {global_mean:.4f}"
    )

    print(
        f"Écart-type du F1 : {global_std:.4f}"
    )

    print("\nRésultats sauvegardés dans :")
    print(output_path)

    print("\nRésumé sauvegardé dans :")
    print(summary_path)


if __name__ == "__main__":
    main()
