"""
prediction.py — Comparaison de modèles de régression ET de classification pour le retard IDFM.

Régression     : prédit le retard en secondes (retard_sec)
Classification : prédit si le train est en retard de plus de SEUIL_RETARD secondes (binaire 0/1)

Évaluation :
  - Hyperparamètres : GridSearchCV (cv=3) en parallèle via ProcessPoolExecutor
    → scoring r²  pour la régression
    → scoring roc_auc pour la classification
  - Métriques finales : KFold k=5 / StratifiedKFold k=5

Usage :
    python prediction.py --csv dataset_predictions/dataset_ml.csv
    python prediction.py          # snapshot API temps réel
"""

import argparse
import asyncio
import json
import warnings
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset import preparer_ml, collecter_snapshot_ml, ML_COLONNES

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

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

TARGET       = "retard_sec"
FEATURES_CAT = ["nom_ligne", "jour_semaine", "periode_journee", "meteo_groupe", "categorie_alerte","stations"]
FEATURES_NUM = [
    "heure_tranche", "mois", "jour_ferie", "occupation",
    "direction_ref", "terminus_encoded", "station_encoded",
    "precipitation", "snowfall", "wind_speed", "temperature",
]
FEATURES = FEATURES_CAT + FEATURES_NUM

N_FOLDS        = 5
RANDOM_STATE   = 42
SEUIL_PROCHE_S = 60   # régression : prédiction "proche" si |erreur| ≤ 60 s
SEUIL_RETARD   = 300  # classification : retard si > 5 min

_DOSSIER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "resultats_modeles",
)
_CACHE_REGRESSION     = os.path.join(_DOSSIER, "best_params_regression.json")
_CACHE_CLASSIFICATION = os.path.join(_DOSSIER, "best_params_classification.json")

# ── Modèles ──────────────────────────────────
MODELES_REGRESSION = {
    "LinearRegression":     LinearRegression(),
    "Ridge":                Ridge(),
    "DecisionTree":         DecisionTreeRegressor(random_state=RANDOM_STATE),
    "RandomForest":         RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
    "ExtraTrees":           ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=1),
    "GradientBoosting":     GradientBoostingRegressor(random_state=RANDOM_STATE),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    "KNeighbors":           KNeighborsRegressor(n_jobs=1),
}

MODELES_CLASSIFICATION = {
    "LogisticRegression":   LogisticRegression(max_iter=1000),
    "SVC":                  SVC(probability=True, random_state=RANDOM_STATE),
    "DecisionTree":         DecisionTreeClassifier(random_state=RANDOM_STATE),
    "RandomForest":         RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
    "GradientBoosting":     GradientBoostingClassifier(random_state=RANDOM_STATE),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    "KNeighbors":           KNeighborsClassifier(n_jobs=1),
}

# ── Grilles de paramètres ─────────────────────
PARAM_GRIDS_REGRESSION: dict[str, dict] = {
    "Ridge": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
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

PARAM_GRIDS_CLASSIFICATION: dict[str, dict] = {
    "LogisticRegression": {"C": [0.01, 0.1, 1.0, 10.0]},
    "SVC":                {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]},
    "DecisionTree": {
        "max_depth":         [3, 5, 8, 12],
        "min_samples_split": [2, 10, 50],
    },
    "RandomForest": {
        "n_estimators":     [100, 200],
        "max_depth":        [5, 10, 15],
        "min_samples_leaf": [1, 5],
    },
    "GradientBoosting": {
        "n_estimators":  [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth":     [3, 5],
    },
    "HistGradientBoosting": {
        "max_iter":       [100, 200],
        "learning_rate":  [0.05, 0.1],
        "max_leaf_nodes": [15, 31],
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
    """Charge un CSV ML-ready. Si brut, appelle preparer_ml() à la volée."""
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
    df = collecter_snapshot_ml()

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
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, roc_curve, confusion_matrix,
    )

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
    accs, f1s, precs, recs, aucs = [], [], [], [], []

    # Pour la courbe ROC moyenne et la matrice de confusion agrégée
    base_fpr = _np.linspace(0, 1, 100)
    tprs = []
    all_y_true, all_y_pred = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        m = sk_clone(modele)
        if best_params:
            m.set_params(**best_params)
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)

        accs.append(accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred, zero_division=0))
        precs.append(precision_score(y_te, y_pred, zero_division=0))
        recs.append(recall_score(y_te, y_pred, zero_division=0))

        all_y_true.extend(y_te.tolist())
        all_y_pred.extend(y_pred.tolist())

        # Score pour AUC
        if hasattr(m, "predict_proba"):
            y_score = m.predict_proba(X_te)[:, 1]
        elif hasattr(m, "decision_function"):
            y_score = m.decision_function(X_te)
        else:
            y_score = y_pred.astype(float)

        try:
            aucs.append(roc_auc_score(y_te, y_score))
            fpr, tpr, _ = roc_curve(y_te, y_score)
            tprs.append(_np.interp(base_fpr, fpr, tpr))
        except Exception:
            pass

    # Matrice de confusion agrégée sur tous les folds
    cm = confusion_matrix(all_y_true, all_y_pred).tolist()
    mean_tpr = _np.mean(tprs, axis=0).tolist() if tprs else base_fpr.tolist()

    return {
        "Modèle":           nom,
        "Accuracy":         float(_np.mean(accs)),
        "Accuracy ±":       float(_np.std(accs)),
        "F1":               float(_np.mean(f1s)),
        "F1 ±":             float(_np.std(f1s)),
        "Précision":        float(_np.mean(precs)),
        "Précision ±":      float(_np.std(precs)),
        "Rappel":           float(_np.mean(recs)),
        "Rappel ±":         float(_np.std(recs)),
        "AUC":              float(_np.mean(aucs)) if aucs else float("nan"),
        "AUC ±":            float(_np.std(aucs))  if aucs else float("nan"),
        "Meilleurs params": str(best_params) if best_params else "—",
        "_params_raw":      best_params,
        "_gs_run":          gs_run,
        "_confusion_matrix": cm,
        "_fpr":             base_fpr.tolist(),
        "_tpr":             mean_tpr,
    }


# ─────────────────────────────────────────────
# ÉVALUATION RÉGRESSION (async)
# ─────────────────────────────────────────────

async def evaluer_regression(df: pd.DataFrame, label: str) -> pd.DataFrame:
    cols = [c for c in FEATURES if c in df.columns]
    X    = df[cols].values.astype(float)
    y    = df[TARGET].to_numpy(dtype=float)

    cache = _lire_cache(_CACHE_REGRESSION)
    loop  = asyncio.get_event_loop()

    avec_cache  = [n for n in MODELES_REGRESSION if n in cache]
    sans_cache  = [n for n in MODELES_REGRESSION if n in PARAM_GRIDS_REGRESSION and n not in cache]
    sans_grille = [n for n in MODELES_REGRESSION if n not in PARAM_GRIDS_REGRESSION]
    print(f"  Cache     : {avec_cache  or '—'}")
    print(f"  GridSearch: {sans_cache  or '—'}")
    print(f"  Direct    : {sans_grille or '—'}")
    print(f"  Évaluation : KFold k={N_FOLDS}  ({len(X):,} lignes)")
    print(f"  Lancement de {len(MODELES_REGRESSION)} modèles en parallèle…\n")

    executor = ProcessPoolExecutor()
    futures = {
        loop.run_in_executor(
            executor,
            _train_eval_regression,
            nom, modele,
            PARAM_GRIDS_REGRESSION.get(nom, {}),
            cache.get(nom),
            X, y,
            SEUIL_PROCHE_S, label, N_FOLDS,
        ): nom
        for nom, modele in MODELES_REGRESSION.items()
    }

    proche_key = f"Proche≤{SEUIL_PROCHE_S}s (%)"
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

async def evaluer_classification(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, dict, dict]:
    """
    Retourne (DataFrame résultats, dict {nom_modèle: (_fpr, _tpr)})
    pour les courbes ROC.
    """
    cols = [c for c in FEATURES if c in df.columns]
    X    = df[cols].values.astype(float)
    y    = (df[TARGET] >= SEUIL_RETARD).to_numpy(dtype=int)

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    print(f"  Classe 0 (à l'heure) : {n_neg:,}  |  Classe 1 (retard >{SEUIL_RETARD}s) : {n_pos:,}")
    print(f"  Évaluation : StratifiedKFold k={N_FOLDS}  ({len(X):,} lignes)")
    print(f"  Lancement de {len(MODELES_CLASSIFICATION)} modèles en parallèle…\n")

    cache = _lire_cache(_CACHE_CLASSIFICATION)
    loop  = asyncio.get_event_loop()

    SVC_MAX_SAMPLES = 10_000
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
            PARAM_GRIDS_CLASSIFICATION.get(nom, {}),
            cache.get(nom),
            X_svc if nom == "SVC" else X,
            y_svc if nom == "SVC" else y,
            N_FOLDS,
        ): nom
        for nom, modele in MODELES_CLASSIFICATION.items()
    }

    lignes: list[dict] = []
    roc_data: dict[str, tuple] = {}
    cm_data:  dict[str, list]  = {}

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
            f"  AUC={m['AUC']:.3f}±{m['AUC ±']:.3f}"
            f"  F1={m['F1']:.3f}±{m['F1 ±']:.3f}"
            f"  Acc={m['Accuracy']:.3f}±{m['Accuracy ±']:.3f}\n"
        )

        # Afficher la matrice de confusion dans le terminal
        cm = m["_confusion_matrix"]
        if len(cm) == 2:
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            print(f"    Matrice de confusion (agrégée) :")
            print(f"      TN={tn:,}  FP={fp:,}")
            print(f"      FN={fn:,}  TP={tp:,}\n")

        roc_data[m["Modèle"]] = (m["_fpr"], m["_tpr"])
        cm_data[m["Modèle"]]  = m["_confusion_matrix"]

    executor.shutdown(wait=False)

    for m in lignes:
        m.pop("_params_raw",      None)
        m.pop("_gs_run",          None)
        m.pop("_confusion_matrix",None)
        m.pop("_fpr",             None)
        m.pop("_tpr",             None)

    return pd.DataFrame(lignes), roc_data, cm_data


# ─────────────────────────────────────────────
# GRAPHIQUES ROC
# ─────────────────────────────────────────────

def tracer_matrices_confusion(cm_data: dict, horodatage: str) -> list[str]:
    """
    Trace une matrice de confusion par modèle, chacune dans son propre PNG.
    Retourne la liste des chemins créés.
    """
    import matplotlib.pyplot as plt

    os.makedirs(_DOSSIER, exist_ok=True)
    chemins = []

    for nom, cm_list in cm_data.items():
        cm  = np.array(cm_list)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, shrink=0.8)

        ax.set_title(
            f"{nom}\nMatrice de confusion — retard > {SEUIL_RETARD}s\n"
            f"(agrégée sur {N_FOLDS} folds StratifiedKFold)",
            fontsize=10, fontweight="bold",
        )
        ax.set_xlabel("Prédit", fontsize=9)
        ax.set_ylabel("Réel", fontsize=9)
        tick_labels = ["À l'heure (0)", "En retard (1)"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(tick_labels, fontsize=9, rotation=15)
        ax.set_yticklabels(tick_labels, fontsize=9)

        thresh = cm.max() / 2.0
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                ax.text(col, row, f"{cm[row, col]:,}",
                        ha="center", va="center", fontsize=12,
                        color="white" if cm[row, col] > thresh else "black")

        fig.tight_layout()
        nom_fichier = nom.replace(" ", "_").lower()
        chemin = os.path.join(_DOSSIER, f"confusion_{nom_fichier}_{horodatage}.png")
        fig.savefig(chemin, dpi=150, bbox_inches="tight")
        plt.close(fig)
        chemins.append(chemin)

    return chemins


def tracer_courbes_roc(roc_data: dict, horodatage: str) -> str:
    """Trace toutes les courbes ROC sur un même graphique. Retourne le chemin du PNG."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aléatoire (AUC = 0.50)")

    couleurs = [plt.cm.get_cmap("tab10")(i) for i in range(10)]
    for i, (nom, (fpr, tpr)) in enumerate(sorted(roc_data.items())):
        auc_approx = float(np.trapezoid(tpr, fpr))
        ax.plot(fpr, tpr, lw=2, color=couleurs[i % len(couleurs)],
                label=f"{nom}  (AUC ≈ {auc_approx:.3f})")

    ax.set_xlabel("Taux de faux positifs (FPR)", fontsize=12)
    ax.set_ylabel("Taux de vrais positifs (TPR)", fontsize=12)
    ax.set_title(f"Courbes ROC — Classification retard > {SEUIL_RETARD}s\n"
                 f"(moyenne sur {N_FOLDS} folds StratifiedKFold)", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    os.makedirs(_DOSSIER, exist_ok=True)
    chemin = os.path.join(_DOSSIER, f"roc_curves_{horodatage}.png")
    fig.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return chemin


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Comparaison modèles régression & classification — retard IDFM")
    parser.add_argument("--csv", default=None,
                        help="CSV ML-ready ou brut. Absent → snapshot API temps réel.")
    args = parser.parse_args()

    print("=" * 65)
    print("  CHARGEMENT & PRÉPARATION")
    print("=" * 65)

    df = charger(args.csv) if args.csv else _charger_api()

    if len(df) < N_FOLDS * 2:
        print(f"  Pas assez de données ({len(df)} lignes). Abandon.")
        return

    from datetime import datetime
    from rapport import ecrire_rapport_regression, ecrire_rapport_classification

    horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(_DOSSIER, exist_ok=True)

    # ══════════════════════════════════════════
    # RÉGRESSION
    # ══════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  RÉGRESSION  (cible : retard_sec en secondes)")
    print("=" * 65)

    res_reg = await evaluer_regression(df, "ML-ready")

    proche_col = f"Proche≤{SEUIL_PROCHE_S}s (%)"
    _afficher_classements_regression(res_reg, proche_col)

    out_csv_reg = os.path.join(_DOSSIER, f"resultats_regression_{horodatage}.csv")
    res_reg.to_csv(out_csv_reg, index=False)
    print(f"\nRésultats CSV régression    → {out_csv_reg}")

    out_md_reg = ecrire_rapport_regression(
        resultats    = res_reg,
        dossier      = _DOSSIER,
        horodatage   = horodatage,
        csv_source   = args.csv,
        n_lignes     = len(df),
        features     = FEATURES,
        n_folds      = N_FOLDS,
        seuil_proche = SEUIL_PROCHE_S,
    )
    print(f"Rapport Markdown régression → {out_md_reg}")

    # ══════════════════════════════════════════
    # CLASSIFICATION
    # ══════════════════════════════════════════
    print("\n" + "=" * 65)
    print(f"  CLASSIFICATION  (cible : retard > {SEUIL_RETARD}s)")
    print("=" * 65)

    res_clf, roc_data, cm_data = await evaluer_classification(df, "ML-ready")

    _afficher_classements_classification(res_clf)

    out_csv_clf = os.path.join(_DOSSIER, f"resultats_classification_{horodatage}.csv")
    res_clf.to_csv(out_csv_clf, index=False)
    print(f"\nRésultats CSV classification → {out_csv_clf}")

    out_roc = tracer_courbes_roc(roc_data, horodatage)
    print(f"Courbes ROC (PNG)            → {out_roc}")

    out_cms = tracer_matrices_confusion(cm_data, horodatage)
    for p in out_cms:
        print(f"Matrice de confusion (PNG)   → {p}")

    out_md_clf = ecrire_rapport_classification(
        resultats    = res_clf,
        dossier      = _DOSSIER,
        horodatage   = horodatage,
        csv_source   = args.csv,
        n_lignes     = len(df),
        features     = FEATURES,
        n_folds      = N_FOLDS,
        seuil_retard = SEUIL_RETARD,
        roc_png      = os.path.basename(out_roc),
        cm_pngs      = [os.path.basename(p) for p in out_cms],
    )
    print(f"Rapport Markdown classification → {out_md_clf}")


def _afficher_classements_regression(resultats: pd.DataFrame, proche_col: str) -> None:
    metriques = [
        ("R²",       True,  "R² (↑ mieux)"),
        ("MAE (s)",  False, "MAE en secondes (↓ mieux)"),
        ("RMSE (s)", False, "RMSE (↓ mieux)"),
        ("MAPE (%)", False, "MAPE en % (↓ mieux)"),
        (proche_col, True,  f"Proche≤{SEUIL_PROCHE_S}s % (↑ mieux)"),
    ]
    for col, desc_asc, titre in metriques:
        std_col = col + " ±"
        tri = resultats.sort_values(col, ascending=not desc_asc).reset_index(drop=True)
        tri.insert(0, "Rang", range(1, len(tri) + 1))
        tri[f"{col} (moy ± std)"] = tri.apply(
            lambda r: f"{r[col]:.3f} ± {r[std_col]:.3f}" if std_col in tri.columns else f"{r[col]:.3f}",
            axis=1,
        )
        print(f"\n{'=' * 65}")
        print(f"  {titre}")
        print("=" * 65)
        print(tri[["Rang", "Modèle", f"{col} (moy ± std)"]].to_string(index=False))


def _afficher_classements_classification(resultats: pd.DataFrame) -> None:
    metriques = [
        ("AUC",      True, "AUC-ROC (↑ mieux)"),
        ("F1",       True, "F1-score (↑ mieux)"),
        ("Accuracy", True, "Accuracy (↑ mieux)"),
        ("Précision",True, "Précision (↑ mieux)"),
        ("Rappel",   True, "Rappel (↑ mieux)"),
    ]
    for col, asc, titre in metriques:
        std_col = col + " ±"
        tri = resultats.sort_values(col, ascending=not asc).reset_index(drop=True)
        tri.insert(0, "Rang", range(1, len(tri) + 1))
        tri[f"{col} (moy ± std)"] = tri.apply(
            lambda r: f"{r[col]:.3f} ± {r[std_col]:.3f}" if std_col in tri.columns else f"{r[col]:.3f}",
            axis=1,
        )
        print(f"\n{'=' * 65}")
        print(f"  {titre}")
        print("=" * 65)
        print(tri[["Rang", "Modèle", f"{col} (moy ± std)"]].to_string(index=False))


if __name__ == "__main__":
    asyncio.run(main())
