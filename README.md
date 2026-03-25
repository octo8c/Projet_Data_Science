# Prédiction des retards — Transports en commun Île-de-France

Projet de data science visant à prédire le **retard en secondes** des RER, Transilien et trams d'Île-de-France à partir des données temps réel de l'API PRIM IDFM, enrichies de données contextuelles (météo, occupation, calendrier).

---

## Architecture du projet

```
Projet_Data_Science/
│
├── python_file/
│   ├── build_dataset.py        # Collecte API + feature engineering + imputation
│   ├── data_preparation.py     # (legacy) Chargement, filtrage, encodage
│   ├── datatransformation.py   # Jointures référentiels (arrêts, GTFS, occupation)
│   ├── prediction.py           # Comparaison de 8 modèles de régression (async)
│   └── rapport.py              # Génération du rapport Markdown des résultats
│
├── analyse/
│   ├── analyse.py              # Analyses exploratoires et graphiques
│   ├── datacreation.py         # Création de datasets d'analyse
│   └── test.py                 # Tests de cohérence des données
│
├── dataset_predictions/        # CSVs produits par la collecte et le pipeline
│   ├── passages_tglobal.csv    # Données brutes collectées (collecte continue)
│   └── dataset_final.csv       # Dataset enrichi prêt pour l'entraînement
│
├── dataset_other/              # Données de référence IDFM
│   ├── arrets .csv             # Référentiel des arrêts (ArRId, ArRName…)
│   ├── arrets_transporteur.csv # Arrêts par transporteur
│   ├── occupation_horaire.csv  # Taux d'occupation par station, heure, type de jour
│   ├── data-rf-2024/           # Validations réseau ferré 2024 (S1, T3, T4)
│   └── validations-...csv      # Validations réseau ferré par profil horaire
│
├── resultats_modeles/          # Sorties de prediction.py (créé automatiquement)
│   ├── resultats_YYYYMMDD_HHMMSS.csv   # Tableau comparatif des modèles
│   ├── rapport_YYYYMMDD_HHMMSS.md      # Rapport Markdown lisible
│   └── best_params.json                # Cache des meilleurs hyperparamètres
│
└── images/                     # Graphiques générés par analyse.py
```

---

## Description des fichiers principaux

### `build_dataset.py` — Collecte et construction du dataset

Ce fichier est le point d'entrée principal pour **construire le dataset d'entraînement**.

Il se découpe en 3 parties :

**Partie 1 — Collecte API**
Interroge l'API SIRI Lite PRIM IDFM (`/estimated-timetable`) toutes les 2 minutes pour toutes les lignes configurées (métro, RER, Transilien, tram). Chaque réponse est parsée pour extraire, par passage :
- les horaires prévus et estimés d'arrivée/départ
- l'arrêt, la ligne, l'opérateur, le terminus
- l'enrichissement temporel : `jour_semaine`, `heure_tranche`, `periode_journee`

Les lignes sont ajoutées au CSV `passages_tglobal.csv` (mode append).

> **Limite** : le métro parisien n'est pas retourné par l'API → `retard_sec` sera NaN pour ces lignes. Seuls RER, Transilien et trams disposent de données complètes.

**Partie 2 — Feature engineering (`build_features`)**

| Feature | Source | Description |
|---|---|---|
| `retard_sec` | Calculé | `horaire_arrivee_estime − horaire_arrivee_prevu` en secondes |
| `nom_ligne` | `line_ref` → dictionnaire | Nom lisible (ex : "RER A", "Tram T9") |
| `mois` | `horaire_arrivee_prevu` | Mois de l'horaire (1–12) |
| `jour_ferie` | Liste statique 2025–2026 | Booléen (1 = jour férié ou dimanche) |
| `meteo` | API Open-Meteo (gratuite) | Description météo Paris ce jour-là |
| `occupation` | `occupation_horaire.csv` | Entrées/heure à l'arrêt (jointure ArRId → ArRName) |

**Partie 3 — Imputation des valeurs manquantes (`impute_missing`)**

| Colonne | Stratégie |
|---|---|
| `meteo` | `"Inconnu"` si API indisponible ou date future |
| `occupation` | Moyenne groupée par `(nom_ligne, heure_tranche)` → `SimpleImputer(mean)` global |
| `retard_sec` | Non imputé — les lignes sans cible calculable sont supprimées à l'entraînement |

---

### `prediction.py` — Comparaison des modèles de régression

Ce fichier **entraîne et compare 8 algorithmes de régression** sur le dataset enrichi.

**Fonctionnement général**

1. Chargement du CSV et calcul de `retard_sec` depuis les horaires
2. Suppression des lignes où la cible n'est pas calculable (`dropna`)
3. Encodage des variables catégorielles (`nom_ligne`, `jour_semaine`, `periode_journee`, `meteo`) via `pandas Categorical`
4. Imputation des NaN résiduels sur les features numériques par la moyenne (dataset corrigé)
5. Split train/test 80/20 fixe (`random_state=42`)
6. Lancement **en parallèle** de tous les modèles via `asyncio` + `ProcessPoolExecutor`

**Recherche d'hyperparamètres**

Pour chaque modèle disposant d'une grille (`PARAM_GRIDS`), un `GridSearchCV(cv=3, scoring='r2')` est exécuté dans un process séparé pour trouver les meilleurs hyperparamètres. Les résultats sont mis en cache dans `resultats_modeles/best_params.json` — au prochain run, le GridSearch est sauté et les params sont relus depuis le cache.

**Modèles comparés**

| Modèle | Hyperparamètres optimisés |
|---|---|
| LinearRegression | Aucun |
| Ridge | `alpha` |
| DecisionTree | `max_depth`, `min_samples_split`, `min_samples_leaf` |
| RandomForest | `n_estimators`, `max_depth`, `min_samples_leaf` |
| ExtraTrees | `n_estimators`, `max_depth`, `min_samples_leaf` |
| GradientBoosting | `n_estimators`, `learning_rate`, `max_depth` |
| HistGradientBoosting | `max_iter`, `learning_rate`, `max_leaf_nodes`, `l2_regularization` |
| KNeighbors | `n_neighbors`, `weights` |

> `max_depth=None` est exclu des grilles pour DecisionTree, RandomForest et ExtraTrees afin d'éviter l'overfitting. HistGBM se régularise via `max_leaf_nodes` et `l2_regularization`.

**Features utilisées pour l'entraînement**

| Type | Features |
|---|---|
| Catégorielles | `nom_ligne`, `jour_semaine`, `periode_journee`, `meteo` |
| Numériques | `heure_tranche`, `mois`, `jour_ferie`, `occupation` |

---

## Métriques d'évaluation

Chaque modèle est évalué sur le jeu de test (20%) avec 5 métriques :

| Métrique | Formule | Interprétation |
|---|---|---|
| **R²** | `1 − SS_res / SS_tot` | Part de variance expliquée. 1 = parfait, 0 = équivalent à prédire la moyenne. Peut être négatif si le modèle est pire que la moyenne. |
| **MAE** | `mean(|y − ŷ|)` | Erreur absolue moyenne en secondes. Robuste aux outliers, toutes les erreurs ont le même poids. |
| **RMSE** | `√mean((y − ŷ)²)` | Comme MAE mais les grandes erreurs sont amplifiées. Utile pour détecter les prédictions très éloignées de la réalité. |
| **MAPE** | `mean(|y − ŷ| / y) × 100` | Erreur relative en %. Instable si le retard réel est proche de 0 (division par ~0). |
| **Proche≤60s (%)** | `mean(|y − ŷ| ≤ 60) × 100` | % de prédictions à moins de 60 secondes de la valeur réelle. Métrique la plus fiable quand le dataset contient peu de retards importants. |

Les résultats sont affichés en 4 classements indépendants (un par métrique) et exportés en CSV + rapport Markdown horodatés dans `resultats_modeles/`.

---

## Exécution

### Prérequis

```bash
pip install pandas numpy scikit-learn requests
```

### 1. Collecte des données

```bash
# Collecte continue (Ctrl+C pour arrêter)
python python_file/build_dataset.py --collecter

# Collecte pendant 2h
python python_file/build_dataset.py --collecter --duree 2.0
```

### 2. Construction du dataset enrichi

```bash
# Feature engineering + imputation → dataset_final.csv
python python_file/build_dataset.py --corriger

# Sur un CSV spécifique avec sortie personnalisée
python python_file/build_dataset.py --corriger \
  --csv  dataset_predictions/passages_tglobal.csv \
  --output dataset_predictions/dataset_final.csv

# Collecte 1h puis enrichissement enchaîné
python python_file/build_dataset.py --collecter --corriger --duree 1.0
```

### 3. Entraînement et comparaison des modèles

```bash
# Sur le dataset enrichi (recommandé)
python python_file/prediction.py --csv dataset_predictions/dataset_final.csv

# Sur le CSV brut (retard_sec calculé à la volée, features manquantes tolérées)
python python_file/prediction.py --csv dataset_predictions/passages_tglobal.csv
```

Les résultats sont écrits dans `resultats_modeles/` :
- `resultats_YYYYMMDD_HHMMSS.csv` — tableau complet des métriques
- `rapport_YYYYMMDD_HHMMSS.md` — rapport lisible avec classements et glossaire
- `best_params.json` — cache des hyperparamètres (évite de relancer GridSearch)

Pour forcer une nouvelle recherche d'hyperparamètres sur un modèle, supprimer son entrée dans `best_params.json`.

---

## Sources des données

| Source | Usage |
|---|---|
| [API PRIM IDFM](https://prim.iledefrance-mobilites.fr/) | Horaires temps réel (clé API requise) |
| [Open-Meteo](https://open-meteo.com/) | Météo historique Paris (gratuit, sans clé) |
| Données IDFM (open data) | Référentiel arrêts, occupation horaire, validations |
