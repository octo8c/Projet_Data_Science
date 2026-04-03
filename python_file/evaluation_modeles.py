"""
evaluation_modeles.py — Comparaison de modèles de régression ET de classification pour le retard IDFM.

Régression     : prédit le retard en secondes (retard_sec)
Classification : prédit si le train est en retard de plus de SEUIL_RETARD secondes (binaire 0/1)

Évaluation :
  - Hyperparamètres : GridSearchCV (cv=3) en parallèle via ProcessPoolExecutor
    → scoring r²  pour la régression
    → scoring roc_auc pour la classification
  - Métriques finales : KFold k=5 / StratifiedKFold k=5
"""


import argparse
import asyncio
import json
import warnings
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset import preparer_ml, collecter_snapshot_ml

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, HistGradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

import tools as tl

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DOSSIER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "resultats_modeles",
)
_CACHE_REGRESSION     = os.path.join(DOSSIER, "best_params_regression.json")
_CACHE_CLASSIFICATION = os.path.join(DOSSIER, "best_params_classification.json")

# ── Modèles ──────────────────────────────────
MODELES_REGRESSION = {
    "LinearRegression":     LinearRegression(),
    "Ridge":                Ridge(),
    "DecisionTree":         DecisionTreeRegressor(random_state=tl.RANDOM_STATE),
    "RandomForest":         RandomForestRegressor(random_state=tl.RANDOM_STATE, n_jobs=1),
    "ExtraTrees":           ExtraTreesRegressor(random_state=tl.RANDOM_STATE, n_jobs=1),
    "GradientBoosting":     GradientBoostingRegressor(random_state=tl.RANDOM_STATE),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=tl.RANDOM_STATE),
    "KNeighbors":           KNeighborsRegressor(n_jobs=1),
}

MODELES_CLASSIFICATION = {
    "LogisticRegression":   LogisticRegression(max_iter=1000),
    "SVC":                  SVC(probability=False, random_state=tl.RANDOM_STATE, max_iter= 2000),
    "DecisionTree":         DecisionTreeClassifier(random_state=tl.RANDOM_STATE),
    "RandomForest":         RandomForestClassifier(random_state=tl.RANDOM_STATE, n_jobs=1),
    "GradientBoosting":     GradientBoostingClassifier(random_state=tl.RANDOM_STATE),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=tl.RANDOM_STATE),
    "KNeighbors":           KNeighborsClassifier(n_jobs=1),
}


# ─────────────────────────────────────────────
# CACHE DES HYPERPARAMÈTRES
# ─────────────────────────────────────────────

def _lire_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _maj_cache(path: str, nom: str, params: dict) -> None:
    cache = _lire_cache(path)
    cache[nom] = params
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────
# WORKER RÉGRESSION
# ─────────────────────────────────────────────

def _train_eval_regression(
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
    import warnings; warnings.filterwarnings("ignore")
    import numpy as _np
    from sklearn.base import clone as sk_clone
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    proche_key = f"Proche≤{seuil_proche}s (%)"
    best_params: dict = {}
    gs_run = False

    if cached_params:
        best_params = cached_params
    elif param_grid:
        gs = GridSearchCV(sk_clone(modele), param_grid, cv=3, scoring="r2", n_jobs=1)
        gs.fit(X, y)
        best_params = gs.best_params_
        gs_run = True

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
        "Modèle":           nom,
        "Dataset":          label,
        "MAE (s)":          float(_np.mean(maes)),
        "MAE (s) ±":        float(_np.std(maes)),
        "RMSE (s)":         float(_np.mean(rmses)),
        "RMSE (s) ±":       float(_np.std(rmses)),
        "R²":               float(_np.mean(r2s)),
        "R² ±":             float(_np.std(r2s)),
        "MAPE (%)":         float(_np.mean(mapes))  if mapes else float("nan"),
        "MAPE (%) ±":       float(_np.std(mapes))   if mapes else float("nan"),
        proche_key:         float(_np.mean(proches)),
        proche_key + " ±":  float(_np.std(proches)),
        "Meilleurs params": str(best_params) if best_params else "—",
        "_params_raw":      best_params,
        "_gs_run":          gs_run,
    }


# ─────────────────────────────────────────────
# WORKER CLASSIFICATION
# ─────────────────────────────────────────────

def _train_eval_classification(
    nom: str,
    modele,
    param_grid: dict,
    cached_params: dict | None,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
) -> dict:
    import warnings; warnings.filterwarnings("ignore")
    import numpy as _np
    from sklearn.base import clone as sk_clone
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.metrics import roc_auc_score, roc_curve

    best_params: dict = {}
    gs_run = False

    if cached_params:
        best_params = cached_params
    elif param_grid:
        gs = GridSearchCV(sk_clone(modele), param_grid, cv=3, scoring="roc_auc", n_jobs=1)
        gs.fit(X, y)
        best_params = gs.best_params_
        gs_run = True

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    aucs = []

    # Pour la courbe ROC moyenne
    base_fpr = _np.linspace(0, 1, 100)
    tprs = []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        m = sk_clone(modele)
        if best_params:
            m.set_params(**best_params)
        m.fit(X_tr, y_tr)

        # Score pour AUC
        if hasattr(m, "predict_proba"):
            y_score = m.predict_proba(X_te)[:, 1]
        elif hasattr(m, "decision_function"):
            y_score = m.decision_function(X_te)
        else:
            y_score = m.predict(X_te).astype(float)

        try:
            aucs.append(roc_auc_score(y_te, y_score))
            fpr, tpr, _ = roc_curve(y_te, y_score)
            tprs.append(_np.interp(base_fpr, fpr, tpr))
        except Exception:
            pass

    mean_tpr = _np.mean(tprs, axis=0).tolist() if tprs else base_fpr.tolist()

    return {
        "Modèle":           nom,
        "AUC":              float(_np.mean(aucs)) if aucs else float("nan"),
        "AUC ±":            float(_np.std(aucs))  if aucs else float("nan"),
        "Meilleurs params": str(best_params) if best_params else "—",
        "_params_raw":      best_params,
        "_gs_run":          gs_run,
        "_fpr":             base_fpr.tolist(),
        "_tpr":             mean_tpr,
    }


# ─────────────────────────────────────────────
# ÉVALUATION RÉGRESSION (async)
# ─────────────────────────────────────────────

async def evaluer_regression(df: pd.DataFrame, label: str) -> pd.DataFrame:
    cols = [c for c in tl.FEATURES if c in df.columns]
    X    = df[cols].values.astype(float)
    y    = df[tl.TARGET].to_numpy(dtype=float)

    cache = _lire_cache(_CACHE_REGRESSION)
    loop  = asyncio.get_event_loop()

    avec_cache  = [n for n in MODELES_REGRESSION if n in cache]
    sans_cache  = [n for n in MODELES_REGRESSION if n in tl.PARAM_GRIDS_REGRESSION and n not in cache]
    sans_grille = [n for n in MODELES_REGRESSION if n not in tl.PARAM_GRIDS_REGRESSION]
    print(f"  Cache     : {avec_cache  or '—'}")
    print(f"  GridSearch: {sans_cache  or '—'}")
    print(f"  Direct    : {sans_grille or '—'}")
    print(f"  Évaluation : KFold k={tl.N_FOLDS}  ({len(X):,} lignes)")
    print(f"  Lancement de {len(MODELES_REGRESSION)} modèles en parallèle…\n")

    executor = ProcessPoolExecutor()
    futures = {
        loop.run_in_executor(
            executor,
            _train_eval_regression,
            nom, modele,
            tl.PARAM_GRIDS_REGRESSION.get(nom, {}),
            cache.get(nom),
            X, y,
            tl.SEUIL_PROCHE_S, label, tl.N_FOLDS,
        ): nom
        for nom, modele in MODELES_REGRESSION.items()
    }

    proche_key = f"Proche≤{tl.SEUIL_PROCHE_S}s (%)"
    lignes: list[dict] = []

    for future in asyncio.as_completed(futures):
        m = await future
        lignes.append(m)

        if m["_gs_run"] and m["_params_raw"]:
            _maj_cache(_CACHE_REGRESSION, m["Modèle"], m["_params_raw"])
            print(f"  [GridSearch OK] {m['Modèle']:<25} → {m['_params_raw']}")
        elif m["_params_raw"]:
            print(f"  [Cache utilisé] {m['Modèle']:<25} → {m['_params_raw']}")

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
# ÉVALUATION CLASSIFICATION (async)
# ─────────────────────────────────────────────

async def evaluer_classification(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, dict]:
    """
    Retourne (DataFrame résultats, dict {nom_modèle: (_fpr, _tpr)})
    pour les courbes ROC.
    """
    cols = [c for c in tl.FEATURES if c in df.columns]
    X    = df[cols].values.astype(float)
    y    = (df[tl.TARGET] >= tl.SEUIL_RETARD).to_numpy(dtype=int)

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    print(f"  Classe 0 (à l'heure) : {n_neg:,}  |  Classe 1 (retard >{tl.SEUIL_RETARD}s) : {n_pos:,}")
    print(f"  Évaluation : StratifiedKFold k={tl.N_FOLDS}  ({len(X):,} lignes)")
    print(f"  Lancement de {len(MODELES_CLASSIFICATION)} modèles en parallèle…\n")

    cache = _lire_cache(_CACHE_CLASSIFICATION)
    loop  = asyncio.get_event_loop()

    SVC_MAX_SAMPLES = 3_000
    if len(X) > SVC_MAX_SAMPLES:
        rng = np.random.default_rng(42)
        svc_idx = rng.choice(len(X), size=SVC_MAX_SAMPLES, replace=False)
        X_svc, y_svc = X[svc_idx], y[svc_idx]
        print(f"  SVC limité à {SVC_MAX_SAMPLES:,} lignes tirées au hasard (dataset trop grand)")
    else:
        X_svc, y_svc = X, y

    executor = ProcessPoolExecutor()
    futures = {
        loop.run_in_executor(
            executor,
            _train_eval_classification,
            nom, modele,
            tl.PARAM_GRIDS_CLASSIFICATION.get(nom, {}),
            cache.get(nom),
            X_svc if nom == "SVC" else X,
            y_svc if nom == "SVC" else y,
            tl.N_FOLDS,
        ): nom
        for nom, modele in MODELES_CLASSIFICATION.items()
    }

    lignes: list[dict] = []
    roc_data: dict[str, tuple] = {}

    for future in asyncio.as_completed(futures):
        m = await future
        lignes.append(m)

        if m["_gs_run"] and m["_params_raw"]:
            _maj_cache(_CACHE_CLASSIFICATION, m["Modèle"], m["_params_raw"])
            print(f"  [GridSearch OK] {m['Modèle']:<25} → {m['_params_raw']}")
        elif m["_params_raw"]:
            print(f"  [Cache utilisé] {m['Modèle']:<25} → {m['_params_raw']}")

        print(
            f"  [Terminé ✓]     {m['Modèle']:<25}"
            f"  AUC={m['AUC']:.3f}±{m['AUC ±']:.3f}\n"
        )

        roc_data[m["Modèle"]] = (m["_fpr"], m["_tpr"])

    executor.shutdown(wait=False)

    for m in lignes:
        m.pop("_params_raw", None)
        m.pop("_gs_run", None)
        m.pop("_fpr", None)
        m.pop("_tpr", None)

    return pd.DataFrame(lignes), roc_data
