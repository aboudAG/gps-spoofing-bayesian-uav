# Détection de GPS Spoofing sur UAV avec un réseau bayésien

Projet personnel autour de la **cybersécurité des systèmes autonomes**, avec pour objectif de détecter une attaque de **GPS spoofing** sur un drone (UAV) à partir de la cohérence entre plusieurs capteurs.

Le projet repose sur une simulation complète : trajectoire du drone, mesures GPS/IMU/altimètre, injection de spoofing, extraction d'indicateurs de cohérence et détection à l'aide d'un réseau bayésien avec `pgmpy`.

---

## Présentation

Le GPS joue un rôle important dans la navigation autonome d'un UAV. Une attaque de GPS spoofing consiste à fournir au récepteur de fausses informations de position afin de faire croire au drone qu'il se trouve à un autre endroit.

L'idée de ce projet est de ne pas regarder uniquement la position GPS, mais de rechercher des **incohérences entre plusieurs mesures**.

Par exemple, si la position GPS indique un déplacement qui ne correspond pas à la vitesse GPS rapportée, une anomalie peut être détectée.

Le système simule donc :

- la trajectoire réelle d'un UAV ;
- un GPS avec du bruit de mesure ;
- une IMU ;
- un altimètre barométrique ;
- plusieurs scénarios de GPS spoofing ;
- des indicateurs de cohérence entre les mesures ;
- un réseau bayésien permettant d'estimer la probabilité d'une attaque.

---

## Objectif

L'objectif est de construire un détecteur capable d'estimer :

```text
P(Spoofing | observations)
```

à partir des mesures disponibles, puis de transformer cette probabilité en une décision :

```text
NORMAL
ou
SPOOFING
```

Le projet permet également d'étudier les limites du détecteur en faisant varier :

- les features utilisées ;
- le seuil de détection ;
- le niveau de bruit ;
- l'intensité du spoofing ;
- la seed de simulation.

---

## Architecture

```text
                 Simulation UAV
                       │
                       ▼
              Trajectoire réelle
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
         GPS           IMU       Altimètre
          │            │            │
          └────────────┼────────────┘
                       │
                       ▼
              GPS Spoofing
             (GPS position)
                       │
                       ▼
          Indicateurs de cohérence
                       │
                       ▼
              Discrétisation
                       │
                       ▼
             Réseau bayésien
                    pgmpy
                       │
                       ▼
        P(Spoofing | observations)
                       │
                       ▼
              Seuil de décision
                       │
              ┌────────┴────────┐
              ▼                 ▼
           NORMAL           SPOOFING
                       │
                       ▼
          Évaluation + visualisations
```

---

## Simulation de l'UAV

Le drone suit une trajectoire circulaire avec :

- une vitesse de croisière autour de 8 m/s ;
- une altitude autour de 50 m ;
- de petites variations de vitesse ;
- de petites variations d'altitude ;
- du bruit sur les mesures des capteurs.

La trajectoire réelle sert uniquement de référence pour générer les mesures simulées.

Le détecteur n'utilise pas directement la vérité terrain : il travaille sur les mesures des capteurs.

Les paramètres de simulation sont regroupés dans :

```text
config/config.py
```

La simulation est reproductible grâce aux seeds aléatoires.

---

## Capteurs simulés

| Capteur | Mesures | Bruit par défaut |
|---|---|---:|
| GPS | Position X/Y/Z + vitesse X/Y | 1.5 m / 0.3 m/s |
| IMU | Accélération X/Y/Z | 0.15 m/s² |
| Altimètre | Altitude | 0.5 m |

Le spoofing est appliqué uniquement à la **position GPS X/Y** dans la simulation actuelle.

---

## Scénarios de spoofing

### Progressive

Le décalage GPS augmente progressivement jusqu'à atteindre une valeur maximale.

```text
Position GPS
     │
     │          /
     │        /
     │      /
     │    /
     │___/____________ Temps
```

Paramètre par défaut :

```text
offset maximal : 40 m
```

### Brutal

Un décalage important est introduit soudainement dans la position GPS.

```text
Position GPS
     │
     │       ┌────────
     │       │
     │_______│________ Temps
```

Paramètre par défaut :

```text
offset : 35 m
```

### Intermittent

Le spoofing apparaît par périodes, avec des phases actives et inactives.

Paramètre par défaut :

```text
offset : 15 m
```

Ce scénario est plus difficile à détecter et permet de tester le comportement du système face à une attaque moins continue.

---

## Indicateurs utilisés

Trois indicateurs sont actuellement calculés.

### `vel_residual`

On compare la vitesse estimée à partir de l'évolution des positions GPS avec la vitesse GPS rapportée.

Une différence importante peut indiquer que la position GPS ne correspond plus correctement à la dynamique observée.

### `alt_residual`

On compare l'altitude GPS avec l'altitude donnée par l'altimètre barométrique.

L'altimètre étant indépendant du GPS, une différence peut constituer une information supplémentaire.

Dans la simulation actuelle, le spoofing ne modifie pas l'altitude GPS. Cette feature apporte donc peu d'information dans les expériences réalisées.

### `pos_jump`

On compare le déplacement GPS entre deux instants avec le déplacement attendu à partir de la vitesse GPS.

Cette feature est particulièrement intéressante pour rechercher des changements brusques de position.

---

## Réseau bayésien

Le réseau utilisé est un réseau bayésien discret construit avec **pgmpy**.

Sa structure actuelle est :

```text
                  Spoofing
                 /    |    \
                /     |     \
      vel_residual alt_residual pos_jump
```

La variable `spoofing` représente l'état du système :

```text
0 → normal
1 → spoofing
```

Les trois indicateurs sont discrétisés en catégories avant l'inférence.

Les tables de probabilités conditionnelles sont estimées à partir des données simulées labellisées avec un `BayesianEstimator` et un prior BDeu.

L'inférence est réalisée avec :

```text
VariableElimination
```

pour obtenir :

```text
P(Spoofing = 1 | observations)
```

---

## Détection

La probabilité calculée par le réseau est comparée à un seuil.

Le seuil utilisé par défaut est :

```text
0.5
```

La règle de décision est :

```text
P(Spoofing | observations) >= 0.5
        ↓
     SPOOFING

P(Spoofing | observations) < 0.5
        ↓
      NORMAL
```

Le seuil est configurable dans :

```text
config/config.py
```

---

## Méthode d'évaluation

Les données d'entraînement et les données de test sont générées séparément.

Les tests utilisent des simulations indépendantes afin de ne pas simplement réévaluer le réseau sur les mêmes observations utilisées pour estimer ses paramètres.

Les métriques utilisées sont :

- Accuracy ;
- Precision ;
- Recall ;
- F1-score ;
- False Positive Rate (FPR) ;
- matrice de confusion ;
- temps moyen d'inférence.

---

## Résultats principaux

Les résultats dépendent du scénario.

Sur l'expérience principale :

| Scénario | Accuracy | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|
| Normal | 92.67% | — | — | — | 7.33% |
| Progressive | 80.17% | 92.88% | 72.50% | 81.44% | 8.33% |
| Brutal | 83.50% | 87.08% | 78.67% | 82.66% | 11.67% |
| Intermittent | 71.00% | 58.82% | 57.14% | 57.97% | 21.54% |

Le scénario intermittent est le plus difficile à détecter dans cette configuration.

Les métriques détaillées sont disponibles dans :

```text
results/metrics/metrics.json
```

---

## Expériences complémentaires

### 1. Ablation des features

J'ai testé différentes combinaisons :

```text
vel_residual
alt_residual
pos_jump

vel_residual + alt_residual
vel_residual + pos_jump
alt_residual + pos_jump

vel_residual + alt_residual + pos_jump
```

Dans la simulation actuelle, `vel_residual` est la feature qui apporte l'essentiel de l'information pour la détection.

L'ajout de `alt_residual` et `pos_jump` ne modifie pas les performances dans cette configuration.

Résultats :

```text
results/metrics/feature_ablation.csv
results/metrics/feature_ablation_summary.csv
```

### 2. Sensibilité au seuil

J'ai testé des seuils compris entre :

```text
0.1 → 0.9
```

Le meilleur F1 moyen obtenu dans cette expérience correspond au seuil `0.3`.

Cependant, les seuils de `0.3` à `0.7` produisent les mêmes décisions dans les simulations testées.

Le seuil par défaut de `0.5` est donc conservé.

Résultats :

```text
results/metrics/threshold_sensitivity.csv
results/metrics/threshold_sensitivity_summary.csv
```

### 3. Robustesse multi-seeds

Le détecteur a été testé avec cinq seeds :

```text
42
43
44
45
46
```

Résultats moyens :

| Scénario | F1 moyen | Écart-type |
|---|---:|---:|
| Progressive | 81.73% | 0.29% |
| Brutal | 85.18% | 0.44% |
| Intermittent | 60.51% | 2.03% |

Les résultats restent relativement stables d'une simulation à l'autre, surtout pour les scénarios progressif et brutal.

Les résultats complets sont disponibles dans :

```text
results/metrics/multi_seed.csv
results/metrics/multi_seed_summary.csv
```

---

## Installation

Cloner le projet puis entrer dans son dossier :

```bash
git clone https://github.com/aboudAG/gps-spoofing-bayesian-uav.git
cd gps-spoofing-bayesian-uav
```

Créer un environnement virtuel :

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

---

## Utilisation

Lancer le pipeline principal :

```bash
python main.py
```

Cela permet de :

- générer les simulations ;
- entraîner le réseau bayésien ;
- effectuer la détection ;
- calculer les métriques ;
- générer les visualisations.

Lancer les tests :

```bash
pytest tests/ -v
```

Les tests actuels passent avec :

```text
30 passed
```

---

## Lancer les expériences

### Ablation des features

```bash
python experiments/feature_ablation.py
```

### Sensibilité du seuil

```bash
python experiments/threshold_sensitivity.py
```

### Robustesse multi-seeds

```bash
python experiments/multi_seed.py
```

Les résultats sont enregistrés dans :

```text
results/metrics/
```

---

## Structure du projet

```text
gps-spoofing-bayesian-uav/
│
├── config/
│   └── config.py
│
├── src/
│   ├── simulation/
│   │   ├── uav.py
│   │   ├── sensors.py
│   │   └── spoofing.py
│   │
│   ├── bayesian/
│   │   ├── features.py
│   │   └── network.py
│   │
│   ├── detection/
│   │   └── detector.py
│   │
│   ├── evaluation/
│   │   └── metrics.py
│   │
│   ├── visualization/
│   │   └── plots.py
│   │
│   └── pipeline.py
│
├── experiments/
│   ├── feature_ablation.py
│   ├── threshold_sensitivity.py
│   └── multi_seed.py
│
├── results/
│   ├── figures/
│   └── metrics/
│
├── tests/
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Limites actuelles

Le projet reste une simulation.

Les principales limites sont :

- la trajectoire de l'UAV est simplifiée ;
- les modèles de capteurs sont basés sur du bruit simulé ;
- le spoofing est appliqué uniquement à la position GPS X/Y ;
- les scénarios d'attaque ne représentent pas toutes les formes possibles de GPS spoofing ;
- le réseau bayésien utilise une structure définie à l'avance ;
- les paramètres sont appris à partir de données simulées ;
- les résultats ne permettent pas de conclure sur les performances d'un système embarqué réel.

Ces limites sont importantes à garder en tête avant toute transposition vers un véritable UAV.

---

## Pistes d'amélioration

Plusieurs évolutions sont possibles :

- intégrer davantage de variables issues de l'IMU ;
- améliorer le modèle dynamique du drone ;
- simuler davantage de stratégies de spoofing ;
- utiliser des données réelles ;
- tester d'autres structures de réseaux bayésiens ;
- utiliser un **Dynamic Bayesian Network (DBN)** pour prendre en compte l'évolution temporelle ;
- étudier une fusion multisensorielle plus avancée ;
- étudier une implémentation adaptée au temps réel.

---

## Technologies utilisées

- Python
- NumPy
- Pandas
- SciPy
- pgmpy
- Matplotlib
- Pytest

---

## Ce que ce projet m'a permis de travailler

À travers ce projet, j'ai travaillé sur :

- la simulation de systèmes autonomes ;
- la modélisation de capteurs ;
- le GPS spoofing ;
- l'analyse de cohérence entre capteurs ;
- les réseaux bayésiens ;
- l'inférence probabiliste ;
- l'évaluation d'un système de détection ;
- l'analyse de robustesse ;
- la mise en place d'expériences reproductibles en Python.

---

## Auteur

**Abdenour AGAG**
