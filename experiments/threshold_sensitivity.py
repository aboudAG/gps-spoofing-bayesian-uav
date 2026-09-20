

import os
import sys

import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.config import DEFAULT_CONFIG
from src.pipeline import (
    SCENARIOS,
    build_training_set,
    generate_scenario,
)
from src.bayesian.features import (
    compute_raw_features,
    discretize_features,
    fit_discretization_edges,
)
from src.bayesian.network import build_network_structure, fit_network
from src.evaluation.metrics import compute_metrics


THRESHOLDS = np.arange(0.1, 1.0, 0.1)


def infer_probabilities(model, discretized_features):
    """
    Calcule P(spoofing=1 | observations)
    pour chaque observation.
    """

    inference = VariableElimination(model)

    probabilities = np.zeros(len(discretized_features))

    for i in range(len(discretized_features)):
        evidence = {
            "vel_residual": int(
                discretized_features.iloc[i]["vel_residual"]
            ),
            "alt_residual": int(
                discretized_features.iloc[i]["alt_residual"]
            ),
            "pos_jump": int(
                discretized_features.iloc[i]["pos_jump"]
            ),
        }

        result = inference.query(
            variables=["spoofing"],
            evidence=evidence,
            show_progress=False,
        )

        probabilities[i] = result.values[1]

    return probabilities


def main():
    cfg = DEFAULT_CONFIG

    print("=" * 70)
    print("ANALYSE DE SENSIBILITÉ DU SEUIL")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Jeu d'entraînement
    # ---------------------------------------------------------

    print("\n[1/3] Génération du jeu d'entraînement...")

    training_raw = build_training_set(cfg)

    edges = fit_discretization_edges(
        training_raw,
        cfg.features,
    )

    training_discretized = discretize_features(
        training_raw,
        edges,
    )

    training_discretized["spoofing"] = training_raw[
        "spoofing"
    ].values

    print(
        f"      {len(training_discretized)} observations"
    )

    # ---------------------------------------------------------
    # 2. Réseau bayésien
    # ---------------------------------------------------------

    print("\n[2/3] Entraînement du réseau bayésien...")
    structure = build_network_structure()
    model = fit_network(
	structure,
        training_discretized,
        cfg.bayesian,
    )

    print("      Réseau bayésien construit.")

    # ---------------------------------------------------------
    # 3. Probabilités + seuils
    # ---------------------------------------------------------

    print("\n[3/3] Évaluation des seuils...\n")

    results = []

    for i, scenario in enumerate(SCENARIOS):

        print("-" * 70)
        print(f"Scénario : {scenario}")

        _, sensors, labels = generate_scenario(
            cfg,
            scenario,
            seed_offset=300 + i,
        )

        raw_features = compute_raw_features(
            sensors,
            cfg.simulation,
            cfg.features,
        )

        discretized_features = discretize_features(
            raw_features,
            edges,
        )

        probabilities = infer_probabilities(
            model,
            discretized_features,
        )

        for threshold in THRESHOLDS:

            predictions = (
                probabilities >= threshold
            ).astype(int)

            metrics = compute_metrics(
                labels,
                predictions,
            )

            results.append(
                {
                    "scenario": scenario,
                    "threshold": round(
                        float(threshold),
                        1,
                    ),
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                    "false_positive_rate": metrics[
                        "false_positive_rate"
                    ],
                }
            )

            print(
                f"  seuil={threshold:.1f} | "
                f"F1={metrics['f1_score']:.4f} | "
                f"Recall={metrics['recall']:.4f} | "
                f"FPR={metrics['false_positive_rate']:.4f}"
            )

    # ---------------------------------------------------------
    # Sauvegarde
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
        "threshold_sensitivity.csv",
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Meilleur seuil selon F1 moyen
    # ---------------------------------------------------------

    summary = (
        results_df
        .groupby("threshold")[
            [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "false_positive_rate",
            ]
        ]
        .mean()
        .sort_values(
            "f1_score",
            ascending=False,
        )
    )

    summary_path = os.path.join(
        output_dir,
        "threshold_sensitivity_summary.csv",
    )

    summary.to_csv(summary_path)

    print("\n" + "=" * 70)
    print("MOYENNE PAR SEUIL")
    print("=" * 70)

    print(summary.to_string())

    best_threshold = summary.index[0]
    best_f1 = summary.iloc[0]["f1_score"]

    print("\n" + "=" * 70)
    print("MEILLEUR SEUIL SELON LE F1")
    print("=" * 70)

    print(
        f"Seuil : {best_threshold:.1f}"
    )

    print(
        f"F1 moyen : {best_f1:.4f}"
    )

    print("\nRésultats sauvegardés dans :")
    print(output_path)

    print("\nRésumé sauvegardé dans :")
    print(summary_path)


if __name__ == "__main__":
    main()
