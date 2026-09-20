

import numpy as np


def classify(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Applique le seuil de détection.

    Parameters
    ----------
    probabilities : np.ndarray
        P(Spoofing=1 | observations) pour chaque pas de temps.
    threshold : float
        Seuil de décision (entre 0 et 1).

    Returns
    -------
    np.ndarray
        Vecteur binaire : 1 = SPOOFING détecté, 0 = NORMAL.
    """
    return (probabilities >= threshold).astype(int)


def label_to_str(label: int) -> str:
    return "SPOOFING" if label == 1 else "NORMAL"
