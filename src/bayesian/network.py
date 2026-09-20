
import time

import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from config.config import BayesianConfig

FEATURE_NODES = ["vel_residual", "alt_residual", "pos_jump"]
TARGET_NODE = "spoofing"


def build_network_structure() -> DiscreteBayesianNetwork:
    """Définit la structure (graphe) du réseau bayésien."""
    edges = [(TARGET_NODE, f) for f in FEATURE_NODES]
    return DiscreteBayesianNetwork(edges)


def fit_network(model: DiscreteBayesianNetwork, training_data: pd.DataFrame,
                 cfg: BayesianConfig) -> DiscreteBayesianNetwork:
    """Estime les CPD du réseau à partir de données d'entraînement labellisées.

    Parameters
    ----------
    training_data : pd.DataFrame
        Doit contenir les colonnes `spoofing` (0/1) et les 3 features
        discrétisées (accel_residual, alt_residual, pos_jump).
    """
    # On force explicitement l'ensemble des états possibles (0/1 pour la
    # cible, 0..n_bins-1 pour chaque feature discrétisée) afin que le modèle
    # sache gérer toute combinaison d'évidence rencontrée à l'inférence,
    # même si elle est rare ou absente des données d'entraînement.
    state_names = {TARGET_NODE: [0, 1]}
    for f in FEATURE_NODES:
        observed_states = sorted(training_data[f].unique().tolist())
        max_state = max(observed_states) if observed_states else 0
        state_names[f] = list(range(max_state + 1))

    data = training_data[[TARGET_NODE] + FEATURE_NODES]
    estimator = BayesianEstimator(model, data, state_names=state_names)
    cpds = estimator.get_parameters(
        prior_type="BDeu",
        equivalent_sample_size=cfg.laplace_smoothing * 10,
    )
    model.add_cpds(*cpds)
    model.check_model()
    return model


def infer_spoofing_probabilities(model: DiscreteBayesianNetwork,
                                  discretized_features: pd.DataFrame) -> tuple:
    """Effectue l'inférence bayésienne pour chaque pas de temps observé.

    Returns
    -------
    (np.ndarray, float)
        Probabilités P(Spoofing=1 | observations) pour chaque ligne, et le
        temps moyen d'inférence par échantillon (secondes).
    """
    infer = VariableElimination(model)
    n = len(discretized_features)
    probs = np.zeros(n)

    start = time.perf_counter()
    for i in range(n):
        evidence = {
            f: int(discretized_features.iloc[i][f]) for f in FEATURE_NODES
        }
        result = infer.query(variables=[TARGET_NODE], evidence=evidence, show_progress=False)
        # L'état '1' correspond à "spoofing actif"
        probs[i] = result.values[1] if len(result.values) > 1 else result.values[0]
    elapsed = time.perf_counter() - start
    mean_inference_time = elapsed / max(n, 1)

    return probs, mean_inference_time
