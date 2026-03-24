"""
prediction.py — Comparaison de modèles de régression pour la prédiction du retard.

Deux expériences :
  A) Dataset BRUT    : features avec NaN → SimpleImputer(mean) dans le Pipeline
                       + HistGradientBoosting qui gère les NaN nativement
  B) Dataset CORRIGÉ : imputation mean des features NaN appliquée en amont

Usage :
    python prediction.py --csv dataset_predictions/passages_tglobal.csv
    python prediction.py --csv dataset_predictions/dataset_final.csv
"""

import argparse
import warnings
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

TARGET       = "retard_sec"
FEATURES_CAT = ["nom_ligne", "jour_semaine", "periode_journee", "meteo"]
FEATURES_NUM = ["heure_tranche", "mois", "jour_ferie", "occupation"]
FEATURES     = FEATURES_CAT + FEATURES_NUM

TEST_SIZE    = 0.20
RANDOM_STATE = 42

MODELES = {
    "LinearRegression":     LinearRegression(),
    "Ridge":                Ridge(alpha=1.0),
    "DecisionTree":         DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE),
    "RandomForest":         RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    "ExtraTrees":           ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    "GradientBoosting":     GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
    "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=100, random_state=RANDOM_STATE),
    "KNeighbors(k=10)":     KNeighborsRegressor(n_neighbors=10, n_jobs=-1),
}

# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────

def _encoder_categoriques(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode les colonnes catégorielles via pandas Categorical (sans sklearn).
    Les NaN et valeurs vides sont regroupés sous '_inconnu'.
    Retourne un DataFrame avec les colonnes catégorielles remplacées par des entiers.
    """
    df = df.copy()
    for col in FEATURES_CAT:
        if col not in df.columns:
            df[col] = 0
        else:
            serie = df[col].fillna("_inconnu").astype(str).str.strip()
            serie = serie.where(serie != "", other="_inconnu")
            df[col] = serie.astype("category").cat.codes
    return df


def charger(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge le CSV, calcule retard_sec et retourne (df_brut, df_corrige).

    - Filtre uniquement les lignes où la cible n'est pas calculable.
    - Encode les catégorielles via pandas (pas de sklearn).
    - df_brut    : features numériques telles quelles (NaN possibles)
    - df_corrige : NaN des features numériques remplacés par la moyenne
    """
    df = pd.read_csv(csv_path, low_memory=False)

    # ── Calcul de la cible ───────────────────────────────────
    arr_est  = pd.to_datetime(df["horaire_arrivee_estime"], utc=True, errors="coerce")
    arr_prev = pd.to_datetime(df["horaire_arrivee_prevu"],  utc=True, errors="coerce")
    df[TARGET] = (arr_est - arr_prev).dt.total_seconds().round()

    avant = len(df)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    print(f"  Lignes supprimées (cible non calculable) : {avant - len(df):,}")
    print(f"  Lignes conservées                        : {len(df):,}")

    # ── Typage des features numériques ──────────────────────
    for col in FEATURES_NUM:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    # ── Encodage des catégorielles ───────────────────────────
    df = _encoder_categoriques(df)

    # ── df_brut : NaN features conservés (expérience A) ─────
    df_brut = df.copy()

    # ── df_corrige : imputation mean features (expérience B) ─
    df_corrige = df.copy()
    for col in FEATURES_NUM:
        if df_corrige[col].isna().any():
            mean_val = df_corrige[col].mean()
            df_corrige[col] = df_corrige[col].fillna(
                mean_val if pd.notna(mean_val) else 0.0
            )

    return df_brut, df_corrige


# ─────────────────────────────────────────────
# MÉTRIQUES
# ─────────────────────────────────────────────

def metriques(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = (np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            if mask.sum() > 0 else float("nan"))
    return {"MAE (s)": mae, "RMSE (s)": rmse, "R²": r2, "MAPE (%)": mape}


# ─────────────────────────────────────────────
# ENTRAÎNEMENT & ÉVALUATION
# ─────────────────────────────────────────────

def evaluer(df: pd.DataFrame, label: str, pipeline_imputer: bool) -> pd.DataFrame:
    """
    Entraîne tous les modèles sur df, affiche et retourne les métriques.

    pipeline_imputer=True  → SimpleImputer(mean) dans le Pipeline (expérience A)
                             HistGradientBoosting gère les NaN nativement (pas de pipeline)
    pipeline_imputer=False → modèle direct, aucun NaN attendu (expérience B)
    """
    cols = [c for c in FEATURES if c in df.columns]
    X    = df[cols].values.astype(float)
    y: np.ndarray = df[TARGET].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    lignes = []
    for nom, modele in MODELES.items():
        modele_clone = clone(modele)

        if pipeline_imputer and nom != "HistGradientBoosting":
            estimateur = Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("modele",  modele_clone),
            ])
        else:
            estimateur = modele_clone

        estimateur.fit(X_train, y_train)
        y_pred: np.ndarray = np.asarray(estimateur.predict(X_test), dtype=float)

        m = metriques(y_test, y_pred)
        m["Modèle"]  = nom
        m["Dataset"] = label
        lignes.append(m)

        print(f"  [{label:<8}] {nom:<25}"
              f"  MAE={m['MAE (s)']:8.1f}s"
              f"  RMSE={m['RMSE (s)']:8.1f}s"
              f"  R²={m['R²']:6.3f}"
              f"  MAPE={m['MAPE (%)']:6.1f}%")

    return pd.DataFrame(lignes)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Comparaison modèles régression — retard IDFM")
    parser.add_argument("--csv", default="dataset_predictions/passages_tglobal.csv",
                        help="Chemin vers le CSV source (brut ou post build_features)")
    args = parser.parse_args()

    print("=" * 65)
    print("  CHARGEMENT & PRÉPARATION")
    print("=" * 65)
    df_brut, df_corrige = charger(args.csv)
    print(f"  Features : {FEATURES}")
    print(f"  Split    : {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}")

    print("\n" + "=" * 65)
    print("  EXPÉRIENCE A — Dataset BRUT  (NaN features → SimpleImputer / HistGBM natif)")
    print("=" * 65)
    res_a = evaluer(df_brut, "Brut", pipeline_imputer=True)

    print("\n" + "=" * 65)
    print("  EXPÉRIENCE B — Dataset CORRIGÉ  (imputation mean des features appliquée)")
    print("=" * 65)
    res_b = evaluer(df_corrige, "Corrigé", pipeline_imputer=False)

    # ── Score composite : R² (qualité globale) + RMSE inversé (pénalise les grandes erreurs)
    # Normalisation min-max sur l'ensemble des résultats puis moyenne pondérée 50/50.
    # RMSE est inversé car une valeur basse est meilleure.
    resultats = pd.concat([res_a, res_b], ignore_index=True)
    resultats = resultats[["Dataset", "Modèle", "MAE (s)", "RMSE (s)", "R²", "MAPE (%)"]]

    rmse_min, rmse_max = resultats["RMSE (s)"].min(), resultats["RMSE (s)"].max()
    r2_min,   r2_max   = resultats["R²"].min(),       resultats["R²"].max()
    rmse_norm = (resultats["RMSE (s)"] - rmse_min) / (rmse_max - rmse_min + 1e-9)
    r2_norm   = (resultats["R²"]       - r2_min)   / (r2_max   - r2_min   + 1e-9)
    resultats["Score"] = (0.5 * r2_norm + 0.5 * (1.0 - rmse_norm)).round(4)

    resultats = resultats.sort_values("Score", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 65)
    print("  RÉSULTATS COMPARATIFS (trié par Score = R²·50% + RMSE·50% inversé)")
    print("=" * 65)
    print(resultats.to_string(index=False, float_format=lambda x: f"{x:>9.3f}"))

    best = resultats.iloc[0]
    print(f"\n>>> Meilleur modèle : [{best['Dataset']}] {best['Modèle']}"
          f"  Score={best['Score']:.4f}  R²={best['R²']:.4f}"
          f"  MAE={best['MAE (s)']:.1f}s  RMSE={best['RMSE (s)']:.1f}s")

    out = os.path.join(os.path.dirname(args.csv), "resultats_modeles.csv")
    resultats.to_csv(out, index=False)
    print(f"\nRésultats exportés → {out}")


if __name__ == "__main__":
    main()
