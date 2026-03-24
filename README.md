# Prédiction des retards des transports en commun Île-de-France

Projet de data science visant à prédire le retard en secondes des trains, RER, trams et métros d'Île-de-France à partir des données temps réel de l'API PRIM IDFM.

---

## Objectif

Prédire `retard_sec` (différence entre l'heure d'arrivée estimée et l'heure prévue) à partir de features contextuelles : ligne, heure, période de la journée, météo, occupation de la station, jour férié.

---

## Architecture du projet

```
Projet_Data_Science/
│
├── python_file/
│   ├── prim_global_csv.py      # Collecte API + feature engineering + imputation
│   ├── build_dataset.py        # CLI : orchestre collecte et/ou correction du dataset
│   ├── data_preparation.py     # Chargement, filtrage strict, calcul retard_sec, encodage
│   └── prediction.py           # Comparaison de 8 modèles de régression (ML)
│
├── analyse/
│   ├── datacreation.py         # Création de datasets d'analyse
│   ├── datatransformation.py   # Jointures référentiels (arrêts, GTFS, occupation)
│   ├── analyse.py              # Analyses exploratoires
│   └── test.py                 # Tests de cohérence des données
│
├── dataset_predictions/        # CSVs produits par la collecte et le pipeline
│   ├── passages_tglobal.csv    # Données brutes collectées (collecte continue)
│   ├── passages_global.csv     # Dataset enrichi (après datatransformation.py)
│   ├── dataset_final.csv       # Dataset prêt pour l'entraînement (sortie build_dataset)
│   └── resultats_modeles.csv   # Tableau comparatif des modèles (sortie prediction.py)
│
└── dataset_other/              # Données de référence IDFM
    ├── arrets .csv             # Référentiel des arrêts (ArRId, ArRName, ZdAId…)
    ├── arrets_transporteur.csv # Arrêts par transporteur
    ├── occupation_horaire.csv  # Taux d'occupation par station, heure, type de jour
    └── validations-...csv      # Validations réseau ferré par profil horaire
```

---

## Pipeline de données

### 1. Collecte (`prim_global_csv.py` / `build_dataset.py --collecter`)

Interroge l'API SIRI Lite PRIM IDFM (`/estimated-timetable`) toutes les 2 minutes pour toutes les lignes (métro, RER, Transilien, tram). Chaque appel retourne les passages prévus et estimés pour chaque arrêt de chaque course. Les données sont ajoutées au CSV `passages_tglobal.csv`.

> **Note** : le métro n'est pas retourné par l'API RATP/SNCF → `retard_sec` sera NaN pour ces lignes. Seuls les RER, Transilien et trams disposent de données complètes.

### 2. Feature engineering (`build_features()` dans `prim_global_csv.py`)

| Feature | Source | Description |
|---|---|---|
| `retard_sec` | Calculé | `horaire_arrivee_estime - horaire_arrivee_prevu` en secondes |
| `nom_ligne` | `line_ref` → dictionnaire | Nom lisible (ex : "RER A", "Tram T9") |
| `mois` | `horaire_arrivee_prevu` | Mois de l'horaire (1–12) |
| `jour_ferie` | Liste statique 2025–2026 | Booléen |
| `meteo` | Open-Meteo API (gratuit) | Description météo Paris ce jour-là |
| `occupation` | `occupation_horaire.csv` | Entrées/heure à l'arrêt (jointure ArRId → nom) |

### 3. Imputation (`impute_missing()` dans `prim_global_csv.py`)

| Colonne | Stratégie |
|---|---|
| `meteo` | `"Inconnu"` si API indisponible ou date future |
| `occupation` | Moyenne groupée par `(nom_ligne, heure_tranche)` → `SimpleImputer(mean)` global |
| `retard_sec` | Non imputé (supprimé lors de l'entraînement) |

### 4. Construction du dataset final (`build_dataset.py --corriger`)

Enchaîne `build_features()` + `impute_missing()` et exporte `dataset_final.csv`.

---

## Utilisation

### Collecte des données

```bash
# Collecte continue (Ctrl+C pour arrêter)
python python_file/build_dataset.py --collecter

# Collecte pendant 2h puis arrêt
python python_file/build_dataset.py --collecter --duree 2.0
```

### Construction du dataset d'entraînement

```bash
# Feature engineering + imputation → dataset_final.csv
python python_file/build_dataset.py --corriger

# Sur un CSV spécifique
python python_file/build_dataset.py --corriger --csv dataset_predictions/passages_global.csv --output dataset_predictions/dataset_final.csv

# Tout en une fois (collecte 1h puis correction)
python python_file/build_dataset.py --collecter --corriger --duree 1.0
```

### Entraînement et comparaison des modèles

```bash
# Sur le dataset final (déjà feature-engineered)
python python_file/prediction.py --csv dataset_predictions/dataset_final.csv

# Sur le CSV brut (applique build_features en interne)
python python_file/prediction.py --csv dataset_predictions/passages_tglobal.csv --build-features
```

---

## Modèles comparés

8 algorithmes de régression sont évalués en parallèle sur deux versions du dataset :

| Modèle | Gestion NaN |
|---|---|
| LinearRegression | Via SimpleImputer (pipeline) |
| Ridge | Via SimpleImputer (pipeline) |
| DecisionTree | Via SimpleImputer (pipeline) |
| **RandomForest** | Via SimpleImputer (pipeline) |
| **ExtraTrees** | Via SimpleImputer (pipeline) |
| GradientBoosting | Via SimpleImputer (pipeline) |
| **HistGradientBoosting** | Natif (pas de pipeline) |
| KNeighbors | Via SimpleImputer (pipeline) |

**Expérience A** — Dataset brut : NaN dans les features → `SimpleImputer(mean)` encapsulé dans chaque pipeline (sauf HistGBM qui gère les NaN nativement).

**Expérience B** — Dataset corrigé : `impute_missing()` appliqué en amont, aucun NaN dans X.

Métriques reportées : **MAE**, **RMSE**, **R²**, **MAPE**.

### Premiers résultats (344 778 lignes, CSV brut)

| Modèle | R² | MAE |
|---|---|---|
| ExtraTrees | **0.848** | 53s |
| RandomForest | 0.835 | 54s |
| KNeighbors | 0.814 | 57s |
| DecisionTree | 0.803 | 63s |
| HistGBM | 0.756 | 66s |

> Les modèles linéaires (R² ≈ 0.08) confirment que la relation retard/features est non linéaire.

---

## Filtrage des données d'entraînement

Seules les lignes disposant **simultanément** de `horaire_arrivee_estime` et `horaire_arrivee_prevu` sont conservées. Les lignes sans ces deux valeurs (métro, données incomplètes) sont exclues de l'entraînement.

Sur le CSV de collecte global : **117 138 lignes supprimées** (horaires manquants) + **5 531** (parsing échoué) sur ~467 000 lignes brutes.

---

## Dépendances

```bash
pip install pandas numpy scikit-learn requests
```

L'API météo [Open-Meteo](https://open-meteo.com/) est utilisée pour les données historiques (gratuite, sans clé).
Les données IDFM proviennent de l'[API PRIM](https://prim.iledefrance-mobilites.fr/) (clé API requise).
