"""
prediction.py — Comparaison de modèles de régression pour la prédiction du retard.

Le dataset d'entrée doit être ML-ready (produit par build_dataset.py --preparer ou
--snapshot) : toutes les colonnes sont numériques, les catégorielles sont déjà encodées
en entiers, les NaN sont imputés.

Évaluation :
  - Hyperparamètres : GridSearchCV (cv=3, scoring=r²) sur l'ensemble complet
  - Métriques finales : KFold k=5 (shuffle, random_state=42) → moyenne ± écart-type

Usage :
    python prediction.py --csv dataset_predictions/dataset_ml.csv
    python prediction.py          # snapshot API temps réel → ML-ready → entraînement
"""

import argparse
import asyncio
import json
import warnings
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset import (
    preparer_ml,
    collecter_snapshot_ml,
    ML_COLONNES,
)

import numpy as np
import pandas as pd
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
FEATURES_CAT = ["nom_ligne", "jour_semaine", "periode_journee", "meteo_groupe", "categorie_alerte"]
FEATURES_NUM = [
    "heure_tranche", "mois", "jour_ferie", "occupation",
    "direction_ref", "terminus_encoded",
    "precipitation", "snowfall", "wind_speed", "temperature",
]
FEATURES = FEATURES_CAT + FEATURES_NUM

N_FOLDS      = 5    # KFold k=5 pour l'évaluation finale
RANDOM_STATE = 42

# Seuil de tolérance : prédiction "proche" si |erreur| ≤ 60 secondes
SEUIL_PROCHE_S = 60

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

PARAM_GRIDS: dict[str, dict] = {
    "Ridge": {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
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
    "HistGradientBoosting": {
        "max_iter":          [100, 200],
        "learning_rate":     [0.05, 0.1, 0.2],
        "max_leaf_nodes":    [15, 31, 63],
        "l2_regularization": [0.0, 0.1, 1.0],
    },
    "KNeighbors": {
        "n_neighbors": [5, 10, 20, 50],
        "weights":     ["uniform", "distance"],
    },
}


# ─────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────

def charger(csv_path: str) -> pd.DataFrame:
    """
    Charge un CSV ML-ready produit par build_dataset.py.
    Si le CSV est brut (colonnes horaires présentes), appelle preparer_ml() à la volée.

    Retourne un DataFrame contenant TARGET + FEATURES, sans NaN, prêt pour sklearn.
    """
    import csv as _csv_mod

    with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as _f:
        header = next(_csv_mod.reader(_f))

    if set(ML_COLONNES).issubset(set(header)):
        df = pd.read_csv(csv_path, low_memory=False)
    else:
        print(f"  CSV brut détecté ({len(header)} cols) — application du pipeline ML…")
        df = preparer_ml(csv_path)

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    avant = len(df)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    print(f"  Lignes conservées (retard_sec calculable) : {len(df):,} / {avant:,}")

    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    return df


def _charger_api() -> pd.DataFrame:
    """Snapshot API temps réel → DataFrame ML-ready."""
    print("  Mode API temps réel (snapshot unique)")
    df = collecter_snapshot_ml(output_ml=None)

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    avant = len(df)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    print(f"  Lignes avec retard calculable : {len(df):,} / {avant:,}")

    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    return df


# ─────────────────────────────────────────────
# CACHE DES HYPERPARAMÈTRES
# ─────────────────────────────────────────────

_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "resultats_modeles", "best_params.json",
)


def _lire_cache() -> dict:
    if os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _maj_cache(nom: str, params: dict) -> None:
    cache = _lire_cache()
    cache[nom] = params
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────
# WORKER (top-level pour ProcessPoolExecutor)
# ─────────────────────────────────────────────

def _train_eval_modele(
    nom: str,
    modele,
    param_grid: dict,
    cached_params: dict | None,
    X: np.ndarray,
    y: np.ndarray,
    seuil_proche: int,
    label: str,
    n_folds: int,
) -> dict:
    """
    Exécuté dans un process séparé.

    Phase 1 — Hyperparamètres :
      - cached_params fourni → réutilisés directement (pas de GridSearch)
      - param_grid fourni    → GridSearchCV(cv=3, scoring=r²) sur l'ensemble complet
      - ni l'un ni l'autre   → fit direct (ex. LinearRegression)

    Phase 2 — Évaluation KFold k=n_folds :
      Pour chaque fold : entraîne avec les meilleurs params → prédit → calcule métriques.
      Retourne moyenne ± écart-type de chaque métrique sur les n_folds folds.
    """
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as _np
    from sklearn.base import clone as sk_clone
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    proche_key = f"Proche≤{seuil_proche}s (%)"

    # ── Phase 1 : recherche des meilleurs hyperparamètres ────────
    best_params: dict = {}
    gs_run = False

    if cached_params:
        best_params = cached_params
    elif param_grid:
        gs = GridSearchCV(sk_clone(modele), param_grid, cv=3, scoring="r2", n_jobs=1)
        gs.fit(X, y)
        best_params = gs.best_params_
        gs_run = True

    # ── Phase 2 : évaluation KFold k=n_folds ─────────────────────
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    maes, rmses, r2s, mapes, proches = [], [], [], [], []

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        m = sk_clone(modele)
        if best_params:
            m.set_params(**best_params)
        m.fit(X_tr, y_tr)
        y_pred = _np.asarray(m.predict(X_te), dtype=float)

        maes.append(mean_absolute_error(y_te, y_pred))
        rmses.append(float(_np.sqrt(mean_squared_error(y_te, y_pred))))
        r2s.append(r2_score(y_te, y_pred))

        mask = y_te != 0
        if mask.sum() > 0:
            mapes.append(float(_np.mean(_np.abs((y_te[mask] - y_pred[mask]) / y_te[mask])) * 100))

        proches.append(float(_np.mean(_np.abs(y_te - y_pred) <= seuil_proche) * 100))

    return {
        "Modèle":            nom,
        "Dataset":           label,
        "MAE (s)":           float(_np.mean(maes)),
        "MAE (s) ±":         float(_np.std(maes)),
        "RMSE (s)":          float(_np.mean(rmses)),
        "RMSE (s) ±":        float(_np.std(rmses)),
        "R²":                float(_np.mean(r2s)),
        "R² ±":              float(_np.std(r2s)),
        "MAPE (%)":          float(_np.mean(mapes))  if mapes  else float("nan"),
        "MAPE (%) ±":        float(_np.std(mapes))   if mapes  else float("nan"),
        proche_key:          float(_np.mean(proches)),
        proche_key + " ±":   float(_np.std(proches)),
        "Meilleurs params":  str(best_params) if best_params else "—",
        "_params_raw":       best_params,
        "_gs_run":           gs_run,
    }


# ─────────────────────────────────────────────
# ENTRAÎNEMENT & ÉVALUATION (async)
# ─────────────────────────────────────────────

async def evaluer(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Lance tous les modèles en parallèle via ProcessPoolExecutor.

    Pour chaque modèle :
      1. GridSearchCV (cv=3) pour trouver les meilleurs hyperparamètres (ou cache)
      2. KFold k=N_FOLDS pour obtenir des métriques robustes (moyenne ± std)

    Le dataset est supposé ML-ready : toutes les features numériques, sans NaN.
    """
    cols = [c for c in FEATURES if c in df.columns]
    X    = df[cols].values.astype(float)
    y    = df[TARGET].to_numpy(dtype=float)

    cache = _lire_cache()
    loop  = asyncio.get_event_loop()

    avec_cache  = [nom for nom in MODELES if nom in cache]
    sans_cache  = [nom for nom in MODELES if nom in PARAM_GRIDS and nom not in cache]
    sans_grille = [nom for nom in MODELES if nom not in PARAM_GRIDS]
    print(f"  Cache     : {avec_cache  or '—'}")
    print(f"  GridSearch: {sans_cache  or '—'}")
    print(f"  Direct    : {sans_grille or '—'}")
    print(f"  Évaluation : KFold k={N_FOLDS}  ({len(X):,} lignes)")
    print(f"  Lancement de {len(MODELES)} modèles en parallèle…\n")

    executor = ProcessPoolExecutor()
    futures = {
        loop.run_in_executor(
            executor,
            _train_eval_modele,
            nom, modele,
            PARAM_GRIDS.get(nom, {}),
            cache.get(nom),
            X, y,
            SEUIL_PROCHE_S, label, N_FOLDS,
        ): nom
        for nom, modele in MODELES.items()
    }

    proche_key = f"Proche≤{SEUIL_PROCHE_S}s (%)"
    lignes: list[dict] = []

    for future in asyncio.as_completed(futures):
        m = await future
        lignes.append(m)

        if m["_gs_run"] and m["_params_raw"]:
            _maj_cache(m["Modèle"], m["_params_raw"])
            print(f"  [GridSearch OK] {m['Modèle']:<25} → params : {m['_params_raw']}")
            print(f"                  cache mis à jour → {_CACHE_PATH}")
        elif m["_params_raw"]:
            print(f"  [Cache utilisé] {m['Modèle']:<25} → params : {m['_params_raw']}")

        print(
            f"  [Terminé ✓]     {m['Modèle']:<25}"
            f"  MAE={m['MAE (s)']:7.1f}±{m['MAE (s) ±']:5.1f}s"
            f"  R²={m['R²']:6.3f}±{m['R² ±']:.3f}"
            f"  Proche={m[proche_key]:5.1f}±{m[proche_key + ' ±']:4.1f}%\n"
        )

    executor.shutdown(wait=False)

    for m in lignes:
        m.pop("_params_raw", None)
        m.pop("_gs_run",     None)

    return pd.DataFrame(lignes)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

async def main():
    """
    Usage :
        python prediction.py                          # snapshot API temps réel
        python prediction.py --csv dataset_ml.csv    # CSV ML-ready déjà préparé
        python prediction.py --csv passages.csv       # CSV brut (préparation auto)
    """
    parser = argparse.ArgumentParser(description="Comparaison modèles régression — retard IDFM")
    parser.add_argument(
        "--csv", default=None,
        help="CSV source ML-ready ou brut. Si absent : snapshot API temps réel.",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  CHARGEMENT & PRÉPARATION")
    print("=" * 65)

    if args.csv:
        df = charger(args.csv)
    else:
        df = _charger_api()

    if len(df) < N_FOLDS * 2:
        print(f"  Pas assez de données ({len(df)} lignes). Abandon.")
        return

    print(f"  Features : {FEATURES}")
    print(f"  Méthode  : KFold k={N_FOLDS}, GridSearchCV (cv=3, scoring=r²)")

    print("\n" + "=" * 65)
    print("  ENTRAÎNEMENT & ÉVALUATION")
    print("=" * 65)
    resultats = await evaluer(df, "ML-ready")

    proche_col = f"Proche≤{SEUIL_PROCHE_S}s (%)"
    cols_affichage = [
        "Dataset", "Modèle",
        "MAE (s)", "MAE (s) ±",
        "RMSE (s)", "RMSE (s) ±",
        "R²", "R² ±",
        "MAPE (%)", "MAPE (%) ±",
        proche_col, proche_col + " ±",
        "Meilleurs params",
    ]
    resultats = resultats[[c for c in cols_affichage if c in resultats.columns]]

    # ── Classements par métrique (triés sur la moyenne) ──────────
    metriques_tri = [
        ("R²",         True,  "R² (plus élevé = mieux)"),
        ("MAE (s)",    False, "MAE — erreur moyenne absolue en secondes (plus bas = mieux)"),
        ("RMSE (s)",   False, "RMSE — pénalise les grandes erreurs (plus bas = mieux)"),
        (proche_col,   True,  f"Proche≤{SEUIL_PROCHE_S}s — % prédictions proches (plus élevé = mieux)"),
    ]
    for col, desc_asc, titre in metriques_tri:
        std_col = col + " ±"
        tri = resultats.sort_values(col, ascending=not desc_asc).reset_index(drop=True)
        tri.insert(0, "Rang", range(1, len(tri) + 1))
        # Colonne affichée : "moy ± std"
        tri[f"{col} (moy ± std)"] = tri.apply(
            lambda r: f"{r[col]:.3f} ± {r[std_col]:.3f}" if std_col in tri.columns else f"{r[col]:.3f}",
            axis=1,
        )
        print(f"\n{'=' * 70}")
        print(f"  {titre}")
        print("=" * 70)
        print(tri[["Rang", "Modèle", f"{col} (moy ± std)"]].to_string(index=False))

    from datetime import datetime
    from rapport import ecrire_rapport  # type: ignore

    dossier    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "resultats_modeles")
    os.makedirs(dossier, exist_ok=True)
    horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_csv = os.path.join(dossier, f"resultats_{horodatage}.csv")
    resultats.to_csv(out_csv, index=False)
    print(f"\nRésultats CSV    → {out_csv}")

    out_md = ecrire_rapport(
        resultats    = resultats,
        dossier      = dossier,
        horodatage   = horodatage,
        csv_source   = args.csv,
        n_lignes     = len(df),
        features     = FEATURES,
        n_folds      = N_FOLDS,
        seuil_proche = SEUIL_PROCHE_S,
    )
    print(f"Rapport Markdown → {out_md}")


if __name__ == "__main__":
    asyncio.run(main())
