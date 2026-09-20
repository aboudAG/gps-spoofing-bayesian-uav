

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _savefig(fig, out_dir: str, name: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_trajectory_comparison(true_traj: pd.DataFrame, sensors: pd.DataFrame,
                                out_dir: str, scenario_name: str) -> str:
    """Trajectoire réelle vs trajectoire GPS (potentiellement spoofée)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(true_traj["x_true"], true_traj["y_true"], label="Trajectoire réelle", color="tab:blue")
    ax.plot(sensors["gps_x"], sensors["gps_y"], label="Trajectoire GPS observée",
            color="tab:red", linestyle="--", alpha=0.8)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Trajectoire réelle vs GPS — scénario : {scenario_name}")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    return _savefig(fig, out_dir, f"trajectory_{scenario_name}.png")


def plot_feature_evolution(t: np.ndarray, features: pd.DataFrame, label: np.ndarray,
                            out_dir: str, scenario_name: str) -> str:
    """Évolution temporelle des indicateurs de cohérence inter-capteurs."""
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    cols = ["vel_residual", "alt_residual", "pos_jump"]
    titles = ["Résidu vitesse position/Doppler", "Résidu altitude GPS/Altimètre",
              "Saut de position GPS"]
    for ax, col, title in zip(axes, cols, titles):
        ax.plot(t, features[col], color="tab:purple")
        ax.set_ylabel(title, fontsize=9)
        ax.fill_between(t, 0, ax.get_ylim()[1], where=label.astype(bool),
                         color="red", alpha=0.1, transform=ax.get_xaxis_transform())
    axes[-1].set_xlabel("Temps (s)")
    fig.suptitle(f"Évolution des indicateurs de cohérence — {scenario_name}")
    fig.tight_layout()
    return _savefig(fig, out_dir, f"features_{scenario_name}.png")


def plot_spoofing_probability(t: np.ndarray, probs: np.ndarray, label: np.ndarray,
                               threshold: float, out_dir: str, scenario_name: str) -> str:
    """Probabilité de spoofing inférée au cours du temps, avec seuil et vérité terrain."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, probs, label="P(Spoofing | observations)", color="tab:orange")
    ax.axhline(threshold, color="black", linestyle=":", label=f"Seuil = {threshold}")
    ax.fill_between(t, 0, 1, where=label.astype(bool), color="red", alpha=0.08,
                     label="Attaque réelle (vérité terrain)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Probabilité de spoofing")
    ax.set_title(f"Probabilité de spoofing inférée — {scenario_name}")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return _savefig(fig, out_dir, f"probability_{scenario_name}.png")


def plot_detection_result(t: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                           out_dir: str, scenario_name: str) -> str:
    """Comparaison classification NORMAL/SPOOFING prédite vs réelle."""
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.step(t, y_true, label="Réel", where="post", color="tab:blue", linewidth=2)
    ax.step(t, y_pred + 0.02, label="Prédit", where="post", color="tab:red",
            linewidth=1.5, linestyle="--")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["NORMAL", "SPOOFING"])
    ax.set_xlabel("Temps (s)")
    ax.set_title(f"Détection NORMAL / SPOOFING — {scenario_name}")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return _savefig(fig, out_dir, f"detection_{scenario_name}.png")


def plot_confusion_matrix(cm: list, out_dir: str, scenario_name: str) -> str:
    """Matrice de confusion."""
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["NORMAL", "SPOOFING"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["NORMAL", "SPOOFING"])
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Réel")
    ax.set_title(f"Matrice de confusion — {scenario_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _savefig(fig, out_dir, f"confusion_matrix_{scenario_name}.png")


def plot_metrics_bar(metrics: dict, out_dir: str, scenario_name: str) -> str:
    """Barres des métriques principales."""
    keys = ["accuracy", "precision", "recall", "f1_score", "false_positive_rate"]
    labels = ["Accuracy", "Precision", "Recall", "F1-score", "FPR"]
    values = [metrics[k] for k in keys]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=["tab:green"] * 4 + ["tab:red"])
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Métriques de détection — {scenario_name}")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    return _savefig(fig, out_dir, f"metrics_{scenario_name}.png")


def plot_sensitivity(x_values, y_values, x_label: str, y_label: str,
                      out_dir: str, name: str, title: str) -> str:
    """Courbe de sensibilité (bruit ou intensité de spoofing) vs métrique."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for series_name, ys in y_values.items():
        ax.plot(x_values, ys, marker="o", label=series_name)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _savefig(fig, out_dir, f"{name}.png")
