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
import asyncio
import json
import warnings
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
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
    "Ridge":                Ridge(),
    "DecisionTree":         DecisionTreeRegressor(random_state=RANDOM_STATE),
    "RandomForest":         RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    "ExtraTrees":           ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    "GradientBoosting":     GradientBoostingRegressor(random_state=RANDOM_STATE),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    "KNeighbors":           KNeighborsRegressor(n_jobs=-1),
}

# Grilles d'hyperparamètres — GridSearchCV (cv=3, scoring=r2)
# LinearRegression n'a pas d'hyperparamètre → exclu
PARAM_GRIDS: dict[str, dict] = {
    "Ridge": {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
    # max_depth borné : None (sans limite) = overfitting assuré sur données réelles
    "DecisionTree": {
        "max_depth":         [3, 5, 8, 12, 18],
        "min_samples_split": [2, 10, 50],
        "min_samples_leaf":  [1, 5, 20],
    },
    "RandomForest": {
        "n_estimators":     [100, 200],
        "max_depth":        [5, 10, 15, 20],
        "min_samples_leaf": [1, 5, 20],
    },
    "ExtraTrees": {
        "n_estimators":     [100, 200],
        "max_depth":        [5, 10, 15, 20],
        "min_samples_leaf": [1, 5, 20],
    },
    "GradientBoosting": {
        "n_estimators":  [100, 200],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth":     [3, 5],
    },
    # HistGBM : max_depth=None autorisé car il régularise via max_leaf_nodes et l2
    "HistGradientBoosting": {
        "max_iter":        [100, 200],
        "learning_rate":   [0.05, 0.1, 0.2],
        "max_leaf_nodes":  [15, 31, 63],
        "l2_regularization": [0.0, 0.1, 1.0],
    },
    "KNeighbors": {
        "n_neighbors": [5, 10, 20, 50],
        "weights":     ["uniform", "distance"],
    },
}


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

# Seuils de tolérance pour le taux de prédictions "proches"
# ⚠ Si le dataset contient peu de retards réels (retard_sec ≈ 0 sur la majorité des
#   lignes), R² sera artificiellement gonflé (un modèle qui prédit toujours 0 semble bon)
#   et MAPE sera instable (division par ~0). Proche(%) est alors la métrique la plus
#   fiable car elle mesure l'erreur absolue indépendamment de la distribution de la cible.
SEUIL_PROCHE_S = 60   # prédiction considérée "proche" si |erreur| ≤ 60 secondes


def metriques(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    r2    = r2_score(y_true, y_pred)
    mask  = y_true != 0
    mape  = (np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
             if mask.sum() > 0 else float("nan"))
    # % de prédictions à moins de SEUIL_PROCHE_S secondes de la vraie valeur
    proche = float(np.mean(np.abs(y_true - y_pred) <= SEUIL_PROCHE_S) * 100)
    return {"MAE (s)": mae, "RMSE (s)": rmse, "R²": r2, "MAPE (%)": mape,
            f"Proche≤{SEUIL_PROCHE_S}s (%)": proche}


# ─────────────────────────────────────────────
# CACHE DES HYPERPARAMÈTRES
# ─────────────────────────────────────────────

_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "resultats_modeles", "best_params.json",
)


def _lire_cache() -> dict:
    """Charge le cache JSON des meilleurs hyperparamètres (dict nom → params)."""
    if os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _maj_cache(nom: str, params: dict) -> None:
    """Ajoute/met à jour les params d'un modèle dans le cache JSON."""
    cache = _lire_cache()
    cache[nom] = params
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────
# WORKER (top-level pour être sérialisable par ProcessPoolExecutor)
# ─────────────────────────────────────────────

def _train_eval_modele(
    nom: str,
    modele,
    param_grid: dict,
    cached_params: dict | None,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seuil_proche: int,
    label: str,
    pipeline_imputer: bool,
) -> dict:
    """
    Exécuté dans un process séparé.
    - Si cached_params fourni  → on réutilise directement (pas de GridSearch).
    - Si param_grid fourni     → GridSearchCV(n_jobs=1), résultats retournés
                                  pour mise en cache côté main process.
    - Sinon                    → fit direct (ex. LinearRegression).
    """
    import warnings
    warnings.filterwarnings("ignore")

    from sklearn.base import clone as sk_clone
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    modele_clone     = sk_clone(modele)
    meilleurs_params: dict = {}
    gs_run = False

    if cached_params:
        # ── Params connus → on saute le GridSearch ────────────
        modele_clone.set_params(**cached_params)
        modele_clone.fit(X_train, y_train)
        meilleurs_params = cached_params
    elif param_grid:
        # ── GridSearch ────────────────────────────────────────
        gs = GridSearchCV(
            sk_clone(modele_clone),
            param_grid,
            cv=3,
            scoring="r2",
            n_jobs=1,   # parallélisme géré par ProcessPoolExecutor en dehors
        )
        gs.fit(X_train, y_train)
        meilleurs_params = gs.best_params_
        modele_clone     = gs.best_estimator_
        gs_run           = True
    else:
        if pipeline_imputer:
            modele_clone = Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("modele",  modele_clone),
            ])
        modele_clone.fit(X_train, y_train)

    y_pred = np.asarray(modele_clone.predict(X_test), dtype=float)

    mae   = mean_absolute_error(y_test, y_pred)
    rmse  = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2    = r2_score(y_test, y_pred)
    mask  = y_test != 0
    mape  = (float(np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100)
             if mask.sum() > 0 else float("nan"))
    proche = float(np.mean(np.abs(y_test - y_pred) <= seuil_proche) * 100)

    return {
        "Modèle":            nom,
        "Dataset":           label,
        "MAE (s)":           mae,
        "RMSE (s)":          rmse,
        "R²":                r2,
        "MAPE (%)":          mape,
        f"Proche≤{seuil_proche}s (%)": proche,
        "Meilleurs params":  str(meilleurs_params) if meilleurs_params else "—",
        "_params_raw":       meilleurs_params,   # dict brut pour le cache
        "_gs_run":           gs_run,             # True = GridSearch effectué
    }


# ─────────────────────────────────────────────
# ENTRAÎNEMENT & ÉVALUATION (async)
# ─────────────────────────────────────────────

async def evaluer(df: pd.DataFrame, label: str, pipeline_imputer: bool) -> pd.DataFrame:
    """
    Lance tous les modèles en parallèle via ProcessPoolExecutor.
    - Charge le cache JSON → si params déjà connus, GridSearch sauté.
    - Affiche chaque modèle dès qu'il termine (asyncio.as_completed).
    - Met à jour le cache JSON après chaque GridSearch réussi.
    """
    cols = [c for c in FEATURES if c in df.columns]
    X    = df[cols].values.astype(float)
    y: np.ndarray = df[TARGET].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    cache = _lire_cache()
    loop  = asyncio.get_event_loop()

    avec_cache    = [nom for nom in MODELES if nom in cache]
    sans_cache    = [nom for nom in MODELES if nom in PARAM_GRIDS and nom not in cache]
    sans_grille   = [nom for nom in MODELES if nom not in PARAM_GRIDS]
    print(f"  Cache     : {avec_cache  or '—'}")
    print(f"  GridSearch: {sans_cache  or '—'}")
    print(f"  Direct    : {sans_grille or '—'}")
    print(f"  Lancement de {len(MODELES)} modèles en parallèle...\n")

    executor = ProcessPoolExecutor()
    # Associe chaque future au nom du modèle pour le message de progression
    futures = {
        loop.run_in_executor(
            executor,
            _train_eval_modele,
            nom, modele,
            PARAM_GRIDS.get(nom, {}),
            cache.get(nom),           # None si pas en cache
            X_train, X_test, y_train, y_test,
            SEUIL_PROCHE_S, label, pipeline_imputer,
        ): nom
        for nom, modele in MODELES.items()
    }

    # ── Traitement au fil de l'eau ────────────────────────────
    proche_key = f"Proche≤{SEUIL_PROCHE_S}s (%)"
    lignes: list[dict] = []

    for future in asyncio.as_completed(futures):
        m = await future
        lignes.append(m)

        # Mise en cache si GridSearch vient d'être exécuté
        if m["_gs_run"] and m["_params_raw"]:
            _maj_cache(m["Modèle"], m["_params_raw"])
            print(f"  [GridSearch OK] {m['Modèle']:<25} → params : {m['_params_raw']}")
            print(f"                  cache mis à jour → {_CACHE_PATH}")
        elif m["_params_raw"]:
            print(f"  [Cache utilisé] {m['Modèle']:<25} → params : {m['_params_raw']}")

        print(f"  [Terminé ✓]     {m['Modèle']:<25}"
              f"  MAE={m['MAE (s)']:8.1f}s"
              f"  RMSE={m['RMSE (s)']:8.1f}s"
              f"  R²={m['R²']:6.3f}"
              f"  Proche={m[proche_key]:5.1f}%\n")

    executor.shutdown(wait=False)

    # Retirer les clés internes avant de retourner le DataFrame
    for m in lignes:
        m.pop("_params_raw", None)
        m.pop("_gs_run",     None)

    return pd.DataFrame(lignes)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Comparaison modèles régression — retard IDFM")
    parser.add_argument("--csv", default="dataset_predictions/passages_tglobal.csv",
                        help="Chemin vers le CSV source (brut ou post build_features)")
    args = parser.parse_args()

    print("=" * 65)
    print("  CHARGEMENT & PRÉPARATION")
    print("=" * 65)
    # df_brut est conservé pour une future expérience A (NaN features → SimpleImputer)
    _df_brut, df_corrige = charger(args.csv)
    print(f"  Features : {FEATURES}")
    print(f"  Split    : {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}")

    # ── Expérience A désactivée (dataset brut, NaN dans les features) ────────
    # print("\n" + "=" * 65)
    # print("  EXPÉRIENCE A — Dataset BRUT  (NaN features → SimpleImputer / HistGBM natif)")
    # print("=" * 65)
    # res_a = evaluer(_df_brut, "Brut", pipeline_imputer=True)

    print("\n" + "=" * 65)
    print("  DATASET CORRIGÉ  (features NaN imputées par la moyenne)")
    print("=" * 65)
    res_b = await evaluer(df_corrige, "Corrigé", pipeline_imputer=False)

    proche_col = f"Proche≤{SEUIL_PROCHE_S}s (%)"
    # resultats = pd.concat([res_a, res_b], ...)  # réactiver quand expérience A relancée
    resultats  = res_b.copy()
    resultats  = resultats[["Dataset", "Modèle", "MAE (s)", "RMSE (s)", "R²",
                             "MAPE (%)", proche_col, "Meilleurs params"]]

    # ── Affichage par métrique ────────────────────────────────
    metriques_tri = [
        ("R²",         True,  "R² (plus élevé = mieux)"),
        ("MAE (s)",    False, "MAE — erreur moyenne absolue (plus bas = mieux)"),
        ("RMSE (s)",   False, "RMSE — pénalise les grandes erreurs (plus bas = mieux)"),
        (proche_col,   True,  f"Proche≤{SEUIL_PROCHE_S}s — % prédictions proches (plus élevé = mieux)"),
    ]
    for col, desc_asc, titre in metriques_tri:
        tri = resultats.sort_values(col, ascending=not desc_asc).reset_index(drop=True)
        tri.insert(0, "Rang", range(1, len(tri) + 1))
        print(f"\n{'=' * 70}")
        print(f"  {titre}")
        print("=" * 70)
        print(tri[["Rang", "Dataset", "Modèle", col]].to_string(index=False))

    from datetime import datetime
    from rapport import ecrire_rapport  # type: ignore

    dossier = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "resultats_modeles")
    os.makedirs(dossier, exist_ok=True)
    horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_csv = os.path.join(dossier, f"resultats_{horodatage}.csv")
    resultats.to_csv(out_csv, index=False)
    print(f"\nRésultats CSV    → {out_csv}")

    out_md = ecrire_rapport(
        resultats   = resultats,
        dossier     = dossier,
        horodatage  = horodatage,
        csv_source  = args.csv,
        n_lignes    = len(df_corrige),
        features    = FEATURES,
        test_size   = TEST_SIZE,
        seuil_proche= SEUIL_PROCHE_S,
    )
    print(f"Rapport Markdown → {out_md}")


if __name__ == "__main__":
    asyncio.run(main())
