

import os
import sys
import time

import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

# Permet d'importer src/ et config/ depuis le dossier racine
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
from src.detection.detector import classify
from src.evaluation.metrics import compute_metrics


FEATURES = [
    "vel_residual",
    "alt_residual",
    "pos_jump",
]

FEATURE_CONFIGURATIONS = {
    "vel_residual": ["vel_residual"],
    "alt_residual": ["alt_residual"],
    "pos_jump": ["pos_jump"],
    "vel+alt": ["vel_residual", "alt_residual"],
    "vel+pos": ["vel_residual", "pos_jump"],
    "alt+pos": ["alt_residual", "pos_jump"],
    "all": ["vel_residual", "alt_residual", "pos_jump"],
}


def build_subset_network(training_data, selected_features, cfg):
    """
    Construit et entraîne un réseau bayésien en utilisant uniquement
    les features sélectionnées.
    """

    target = "spoofing"

    # Structure : spoofing -> chaque feature sélectionnée
    edges = [(target, feature) for feature in selected_features]

    model = DiscreteBayesianNetwork(edges)

    # Définition explicite des états possibles
    state_names = {
        target: [0, 1],
    }

    for feature in selected_features:
        observed_states = sorted(
            training_data[feature].unique().tolist()
        )

        max_state = max(observed_states) if observed_states else 0

        state_names[feature] = list(range(max_state + 1))

    data = training_data[[target] + selected_features].copy()

    estimator = BayesianEstimator(
        model,
        data,
        state_names=state_names,
    )

    cpds = estimator.get_parameters(
        prior_type="BDeu",
        equivalent_sample_size=cfg.bayesian.laplace_smoothing * 10,
    )

    model.add_cpds(*cpds)
    model.check_model()

    return model


def infer(model, discretized_features, selected_features):
    """
    Effectue l'inférence bayésienne pour les features sélectionnées.
    """

    inference = VariableElimination(model)

    probabilities = np.zeros(len(discretized_features))

    start = time.perf_counter()

    for i in range(len(discretized_features)):

        evidence = {
            feature: int(discretized_features.iloc[i][feature])
            for feature in selected_features
        }

        result = inference.query(
            variables=["spoofing"],
            evidence=evidence,
            show_progress=False,
        )

        probabilities[i] = result.values[1]

    elapsed = time.perf_counter() - start

    mean_time = elapsed / max(len(discretized_features), 1)

    return probabilities, mean_time


def main():

    cfg = DEFAULT_CONFIG

    print("=" * 70)
    print("EXPÉRIENCE D'ABLATION DES FEATURES")
    print("=" * 70)

    print("\nFeatures disponibles :")
    for feature in FEATURES:
        print(f"  - {feature}")

    print("\nConfigurations testées :")
    for name, features in FEATURE_CONFIGURATIONS.items():
        print(f"  - {name}: {', '.join(features)}")

    # ---------------------------------------------------------
    # 1. Construction du dataset d'entraînement
    # ---------------------------------------------------------

    print("\n[1/3] Génération du jeu d'entraînement...")

    training_raw = build_training_set(cfg)

    # Les bornes de discrétisation sont apprises UNE SEULE FOIS
    # à partir du même dataset pour toutes les configurations.
    edges = fit_discretization_edges(
        training_raw,
        cfg.features,
    )

    training_discretized = discretize_features(
        training_raw,
        edges,
    )

    training_discretized["spoofing"] = training_raw["spoofing"].values

    print(
        f"      {len(training_discretized)} observations d'entraînement"
    )

    # ---------------------------------------------------------
    # 2. Génération des jeux de test
    # ---------------------------------------------------------

    print("\n[2/3] Génération des jeux de test...")

    test_sets = {}

    for i, scenario in enumerate(SCENARIOS):

        _, sensors, labels = generate_scenario(
            cfg,
            scenario,
            seed_offset=200 + i,
        )

        raw_features = compute_raw_features(
            sensors,
            cfg.simulation,
            cfg.features,
        )

        discretized = discretize_features(
            raw_features,
            edges,
        )

        test_sets[scenario] = {
            "features": discretized,
            "labels": labels,
        }

        print(
            f"      {scenario}: {len(labels)} observations"
        )

    # ---------------------------------------------------------
    # 3. Ablation
    # ---------------------------------------------------------

    print("\n[3/3] Évaluation des configurations...\n")

    results = []

    for config_name, selected_features in FEATURE_CONFIGURATIONS.items():

        print("-" * 70)
        print(
            f"Configuration : {config_name}"
            f" ({', '.join(selected_features)})"
        )

        # Entraînement du modèle avec uniquement les features sélectionnées
        model = build_subset_network(
            training_discretized,
            selected_features,
            cfg,
        )

        for scenario in SCENARIOS:

            discretized = test_sets[scenario]["features"]
            labels = test_sets[scenario]["labels"]

            probabilities, mean_inference_time = infer(
                model,
                discretized,
                selected_features,
            )

            predictions = classify(
                probabilities,
                cfg.bayesian.detection_threshold,
            )

            metrics = compute_metrics(
                labels,
                predictions,
            )

            row = {
                "configuration": config_name,
                "features": "+".join(selected_features),
                "scenario": scenario,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "false_positive_rate": metrics["false_positive_rate"],
                "mean_inference_time_sec": mean_inference_time,
            }

            results.append(row)

            print(
                f"  {scenario:12s} | "
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

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        "feature_ablation.csv",
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("RÉSULTATS FINAUX")
    print("=" * 70)

    print(
        results_df[
            [
                "configuration",
                "scenario",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "false_positive_rate",
            ]
        ].to_string(index=False)
    )

    print("\nRésultats sauvegardés dans :")
    print(output_path)

    # ---------------------------------------------------------
    # Résumé moyen par configuration
    # ---------------------------------------------------------

    summary = (
        results_df
        .groupby("configuration")
        [
            [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "false_positive_rate",
            ]
        ]
        .mean()
        .sort_values("f1_score", ascending=False)
    )

    summary_path = os.path.join(
        output_dir,
        "feature_ablation_summary.csv",
    )

    summary.to_csv(summary_path)

    print("\n" + "=" * 70)
    print("MOYENNE PAR CONFIGURATION")
    print("=" * 70)

    print(summary.to_string())

    print("\nRésumé sauvegardé dans :")
    print(summary_path)


if __name__ == "__main__":
    main()
