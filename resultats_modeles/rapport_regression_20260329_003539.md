# Rapport — Comparaison des modèles de régression IDFM

**Date :** 2026-03-29  
**Heure :** 003539  
**Source CSV :** `dataset_predictions/dataset_ml.csv`  
**Lignes utilisées :** 93,498  
**Évaluation :** KFold k=5 (shuffle, random_state=42)  
**Hyperparamètres :** GridSearchCV (cv=3, scoring=r²)  

---

## Features utilisées

  - **Identification ligne** : `nom_ligne`, `direction_ref`
  - **Temporelles** : `heure_tranche`, `mois`, `jour_semaine`, `periode_journee`, `jour_ferie`
  - **Destination & arrêt** : `terminus_encoded`, `station_encoded`
  - **Météo horaire** : `meteo_groupe`, `precipitation`, `snowfall`, `wind_speed`, `temperature`
  - **Alertes réseau PRIM** : `categorie_alerte`
  - **Fréquentation arrêt** : `occupation`

### Détail des features

| Feature | Type | Description |
|---------|------|-------------|
| `nom_ligne` | Catégorielle | Nom de la ligne (Métro 1, RER A…) — encodé entier fixe |
| `jour_semaine` | Catégorielle | Lundi … Dimanche — encodé entier fixe |
| `periode_journee` | Catégorielle | Pointe matin / Creuse / Méridienne / Pointe soir… — encodé entier fixe |
| `meteo_groupe` | Catégorielle | Groupe météo horaire (ensoleille / pluie / neige / orage…) — encodé entier fixe |
| `categorie_alerte` | Catégorielle | Type d'alerte PRIM (aucune / greve / incident / travaux…) — encodé entier fixe |
| `heure_tranche` | Numérique | Heure de passage (0–23) |
| `mois` | Numérique | Mois de l'année (1–12) |
| `jour_ferie` | Numérique | 1 si jour férié français, 0 sinon |
| `occupation` | Numérique | Nombre moyen d'entrées/heure à l'arrêt (données IDFM) |
| `direction_ref` | Numérique | Sens de la course : 1=aller, 2=retour, 0=inconnu |
| `terminus_encoded` | Numérique | Retard moyen historique du terminus (target encoding) |
| `station_encoded` | Numérique | Retard moyen historique de l'arrêt (target encoding) |
| `precipitation` | Numérique | Précipitations en mm/h |
| `snowfall` | Numérique | Chutes de neige en cm |
| `wind_speed` | Numérique | Vitesse du vent en km/h |
| `temperature` | Numérique | Température en °C |

---

## Classements par métrique

> Métriques calculées par KFold k=5. Format : **moyenne ± écart-type**.

### R² — variance expliquée (↑ mieux)

| Rang | Modèle | Dataset | Valeur (moy ± std) | Meilleurs params |
|:----:|--------|---------|-------------------:|------------------|
| 1 | ExtraTrees | ML-ready | 0.988 ± 0.002 | `{'max_depth': 10, 'min_samples_leaf': 5, 'n_estimators': 200}` |
| 2 | HistGradientBoosting | ML-ready | 0.987 ± 0.003 | `{'l2_regularization': 0.0, 'learning_rate': 0.2, 'max_iter': 200, 'max_leaf_nodes': 31}` |
| 3 | GradientBoosting | ML-ready | 0.980 ± 0.003 | `{'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 100}` |
| 4 | DecisionTree | ML-ready | 0.978 ± 0.008 | `{'max_depth': 12, 'min_samples_leaf': 5, 'min_samples_split': 2}` |
| 5 | RandomForest | ML-ready | 0.966 ± 0.007 | `{'max_depth': 20, 'min_samples_leaf': 20, 'n_estimators': 100}` |
| 6 | KNeighbors | ML-ready | 0.625 ± 0.024 | `{'n_neighbors': 10, 'weights': 'distance'}` |
| 7 | LinearRegression | ML-ready | 0.421 ± 0.004 | `—` |
| 8 | Ridge | ML-ready | 0.421 ± 0.004 | `{'alpha': 100.0}` |

### MAE (s) — erreur moyenne absolue (↓ mieux)

| Rang | Modèle | Dataset | Valeur (moy ± std) | Meilleurs params |
|:----:|--------|---------|-------------------:|------------------|
| 1 | DecisionTree | ML-ready | 74.5 ± 3.7 | `{'max_depth': 12, 'min_samples_leaf': 5, 'min_samples_split': 2}` |
| 2 | HistGradientBoosting | ML-ready | 79.5 ± 3.8 | `{'l2_regularization': 0.0, 'learning_rate': 0.2, 'max_iter': 200, 'max_leaf_nodes': 31}` |
| 3 | ExtraTrees | ML-ready | 81.8 ± 2.8 | `{'max_depth': 10, 'min_samples_leaf': 5, 'n_estimators': 200}` |
| 4 | RandomForest | ML-ready | 104.4 ± 5.5 | `{'max_depth': 20, 'min_samples_leaf': 20, 'n_estimators': 100}` |
| 5 | GradientBoosting | ML-ready | 113.5 ± 4.7 | `{'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 100}` |
| 6 | KNeighbors | ML-ready | 278.4 ± 10.3 | `{'n_neighbors': 10, 'weights': 'distance'}` |
| 7 | Ridge | ML-ready | 944.3 ± 14.3 | `{'alpha': 100.0}` |
| 8 | LinearRegression | ML-ready | 980.0 ± 15.2 | `—` |

### RMSE (s) — pénalise les grandes erreurs (↓ mieux)

| Rang | Modèle | Dataset | Valeur (moy ± std) | Meilleurs params |
|:----:|--------|---------|-------------------:|------------------|
| 1 | ExtraTrees | ML-ready | 339.4 ± 25.4 | `{'max_depth': 10, 'min_samples_leaf': 5, 'n_estimators': 200}` |
| 2 | HistGradientBoosting | ML-ready | 353.4 ± 39.2 | `{'l2_regularization': 0.0, 'learning_rate': 0.2, 'max_iter': 200, 'max_leaf_nodes': 31}` |
| 3 | GradientBoosting | ML-ready | 445.2 ± 21.9 | `{'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 100}` |
| 4 | DecisionTree | ML-ready | 457.1 ± 82.2 | `{'max_depth': 12, 'min_samples_leaf': 5, 'min_samples_split': 2}` |
| 5 | RandomForest | ML-ready | 583.6 ± 84.5 | `{'max_depth': 20, 'min_samples_leaf': 20, 'n_estimators': 100}` |
| 6 | KNeighbors | ML-ready | 1929.7 ± 74.8 | `{'n_neighbors': 10, 'weights': 'distance'}` |
| 7 | LinearRegression | ML-ready | 2399.7 ± 111.6 | `—` |
| 8 | Ridge | ML-ready | 2400.6 ± 111.2 | `{'alpha': 100.0}` |

### Proche≤60s — % prédictions proches (↑ mieux)

| Rang | Modèle | Dataset | Valeur (moy ± std) | Meilleurs params |
|:----:|--------|---------|-------------------:|------------------|
| 1 | KNeighbors | ML-ready | 79.365 ± 0.230 | `{'n_neighbors': 10, 'weights': 'distance'}` |
| 2 | DecisionTree | ML-ready | 79.325 ± 0.512 | `{'max_depth': 12, 'min_samples_leaf': 5, 'min_samples_split': 2}` |
| 3 | RandomForest | ML-ready | 78.420 ± 0.254 | `{'max_depth': 20, 'min_samples_leaf': 20, 'n_estimators': 100}` |
| 4 | HistGradientBoosting | ML-ready | 76.341 ± 0.464 | `{'l2_regularization': 0.0, 'learning_rate': 0.2, 'max_iter': 200, 'max_leaf_nodes': 31}` |
| 5 | ExtraTrees | ML-ready | 75.860 ± 0.309 | `{'max_depth': 10, 'min_samples_leaf': 5, 'n_estimators': 200}` |
| 6 | GradientBoosting | ML-ready | 72.535 ± 0.711 | `{'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 100}` |
| 7 | Ridge | ML-ready | 4.585 ± 0.472 | `{'alpha': 100.0}` |
| 8 | LinearRegression | ML-ready | 2.520 ± 0.664 | `—` |

---

## Meilleur modèle global (R² moyen)

**[ML-ready] ExtraTrees**

| Métrique | Moyenne | Écart-type |
|----------|--------:|----------:|
| R²       | `0.9882` | `0.0024` |
| MAE (s)  | `81.8` | `2.8` |
| RMSE (s) | `339.4` | `25.4` |
| MAPE (%) | `288.7` | `58.1` |
| Proche≤60s (%) | `75.9%` | `0.3%` |
| Params   | `{'max_depth': 10, 'min_samples_leaf': 5, 'n_estimators': 200}` | — |

---

## Glossaire

| Métrique | Description |
|----------|-------------|
| **R²** | Part de variance expliquée. 1 = parfait, 0 = équivalent à prédire la moyenne. |
| **MAE** | Erreur absolue moyenne en secondes. |
| **RMSE** | Comme MAE mais les grandes erreurs sont amplifiées. |
| **MAPE** | Erreur en % relatif à la valeur réelle. Instable si retard ≈ 0. |
| **Proche≤60s** | % de prédictions à moins de 60s de la réalité. |
| **KFold k=5** | Validation croisée sur 5 blocs. L'écart-type mesure la stabilité. |