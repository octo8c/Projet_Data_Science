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
from build_dataset import collecter_snapshot_ml

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, HistGradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

TARGET = "retard_sec"

LIGNES_RER_TRANSILIEN = {
    "RER A", "RER B", "RER C", "RER D", "RER E",
    "Ligne H", "Ligne J", "Ligne K", "Ligne L",
    "Ligne N", "Ligne P", "Ligne R", "Ligne U",
}

# Colonnes pour KFold target encoding (traitées dans evaluer_*(), pas ici)
FEATURES_CAT_TARGET = ["nom_ligne", "terminus", "stop_ref"]

# Features numériques + OHE disponibles directement après transformer_dataset()
FEATURES_NUM = [
    # Numériques brutes
    "heure_tranche", "direction_ref",
    "mois", "jour_ferie", "vacances", "est_vacances", "occupation",
    "terminus_encoded", "station_encoded", "alerte_active",
    "precipitation", "snowfall", "wind_speed", "temperature",
    # OHE periode_journee
    "periode_Nuit", "periode_Pointe matin", "periode_Creuse matin",
    "periode_Méridienne", "periode_Creuse après-midi", "periode_Pointe soir", "periode_Soirée",
    # OHE categorie_alerte
    "alerte_aucune", "alerte_greve", "alerte_incident", "alerte_travaux",
    "alerte_meteo", "alerte_retard", "alerte_voyageur", "alerte_autre",
    # OHE jour_semaine
    "jour_Lundi", "jour_Mardi", "jour_Mercredi", "jour_Jeudi",
    "jour_Vendredi", "jour_Samedi", "jour_Dimanche",
    # OHE meteo_groupe
    "meteo_ensoleille", "meteo_nuageux", "meteo_brouillard", "meteo_pluie",
    "meteo_neige", "meteo_averses", "meteo_orage", "meteo_autre", "meteo_inconnu",
]
FEATURES = FEATURES_NUM  # FEATURES_CAT_TARGET ajoutés via KFold target encoding dans evaluer_*

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
}

MODELES_CLASSIFICATION = {
    "LogisticRegression":   LogisticRegression(max_iter=1000),
    "DecisionTree":         DecisionTreeClassifier(random_state=RANDOM_STATE),
    "RandomForest":         RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
    "GradientBoosting":     GradientBoostingClassifier(random_state=RANDOM_STATE),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
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
}

PARAM_GRIDS_CLASSIFICATION: dict[str, dict] = {
    "LogisticRegression": {"C": [0.01, 0.1, 1.0, 10.0]},
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
}


# ─────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────

def charger(csv_path: str) -> pd.DataFrame:
    """Charge un CSV brut PRIM directement.

    transformer_dataset() appelé ensuite dans main() se charge de toutes
    les transformations (suppression data leakage, OHE, encodage cyclique…).
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    avant = len(df)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    print(f"  {csv_path} — {len(df):,} lignes avec retard_sec / {avant:,} total")
    return df


def _charger_api() -> pd.DataFrame:
    """Snapshot API temps réel → DataFrame brut (même format que CSV brut PRIM)."""
    from build_dataset import get_estimated_timetable, CSV_COLONNES
    print("  Mode API temps réel (snapshot unique)")
    rows = get_estimated_timetable()
    if not rows:
        raise ValueError("L'API PRIM n'a retourné aucune donnée.")
    df = pd.DataFrame(rows, columns=CSV_COLONNES)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    avant = len(df)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    print(f"  Lignes avec retard calculable : {len(df):,} / {avant:,}")
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
        
# ________________________________________________
# Preprocessing du dataset
# ________________________________________________

def transformer_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Transforme un DataFrame brut PRIM en features ML-ready avec sklearn.

    Après cette fonction, nom_ligne, terminus et direction_ref restent en string :
    ils seront encodés via KFold target encoding dans evaluer_*() pour éviter
    tout data leakage.

    Transformations :
      1. Suppression des colonnes de data leakage (horaires prévus/estimés, identifiants)
      2. alerte_active, jour_ferie  : booléen → 0/1 (LabelBinarizer)
      4. periode_journee  → OneHotEncoder sklearn (7 colonnes)
      5. categorie_alerte → OneHotEncoder sklearn (8 colonnes)
      6. jour_semaine     → OneHotEncoder sklearn (7 colonnes)
      7. meteo_groupe     → OneHotEncoder sklearn (9 colonnes)
      8. Colonnes numériques restantes → float (NaN → 0)
    """
    from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
    from sklearn.preprocessing import FunctionTransformer

    df = df.copy()

    # ── 1. Suppression data leakage + identifiants ────────────────────────────
    df = df.drop(columns=[c for c in [
        "horaire_arrivee_prevu", "horaire_depart_prevu",
        "horaire_arrivee_estime", "horaire_depart_estime",
        "arrivee_prevue_hhmm", "depart_prevu_hhmm", "depart_estime_hhmm",
        "date_capture", "line_ref", "operateur", "nom_arret",
    ] if c in df.columns])

    # ── 2. Booléens → 0/1 via LabelBinarizer ─────────────────────────────────
    for col in ("alerte_active", "jour_ferie", "vacances", "est_vacances"):
        if col in df.columns:
            lb = LabelBinarizer()
            vals = df[col].map({True: "1", False: "0", "True": "1", "False": "0",
                                1: "1", 0: "0"}).fillna("0")
            df[col] = np.asarray(lb.fit_transform(vals)).ravel().astype(int)

    # ── 3. direction_ref → 0/1  (Aller/inbound/A = 1, Retour/outbound/R = 0) ─
    if "direction_ref" in df.columns:
        aller  = {"Aller", "inbound", "A", "1", "aller"}
        retour = {"Retour", "outbound", "R", "2", "retour"}
        def _encode_dir(v):
            s = str(v).strip()
            if s in aller:  return 1
            if s in retour: return 0
            return np.nan
        df["direction_ref"] = df["direction_ref"].map(_encode_dir).fillna(0).astype(int)

    # ── Helper OHE sklearn ────────────────────────────────────────────────────
    def _ohe_sklearn(df: pd.DataFrame, col: str, prefix: str,
                     categories: list[str]) -> pd.DataFrame:
        enc = OneHotEncoder(
            categories=[categories],
            sparse_output=False,
            handle_unknown="ignore",
            dtype=np.int8,
        )
        col_data = df[[col]].fillna(categories[0]).astype(str)
        arr = enc.fit_transform(col_data)
        col_names = [f"{prefix}_{cat}" for cat in categories]
        dummies = pd.DataFrame(arr, columns=col_names, index=df.index)
        return pd.concat([df.drop(columns=[col]), dummies], axis=1)

    # ── 4. OHE periode_journee ────────────────────────────────────────────────
    if "periode_journee" in df.columns and df["periode_journee"].dtype == object:
        df = _ohe_sklearn(df, "periode_journee", "periode", [
            "Nuit", "Pointe matin", "Creuse matin", "Méridienne",
            "Creuse après-midi", "Pointe soir", "Soirée",
        ])

    # ── 5. OHE categorie_alerte ───────────────────────────────────────────────
    if "categorie_alerte" in df.columns and df["categorie_alerte"].dtype == object:
        df = _ohe_sklearn(df, "categorie_alerte", "alerte", [
            "aucune", "greve", "incident", "travaux",
            "meteo", "retard", "voyageur", "autre",
        ])

    # ── 6. OHE jour_semaine ───────────────────────────────────────────────────
    if "jour_semaine" in df.columns and df["jour_semaine"].dtype == object:
        df = _ohe_sklearn(df, "jour_semaine", "jour", [
            "Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche",
        ])

    # ── 7. OHE meteo_groupe ───────────────────────────────────────────────────
    if "meteo_groupe" in df.columns and df["meteo_groupe"].dtype == object:
        df = _ohe_sklearn(df, "meteo_groupe", "meteo", [
            "ensoleille", "nuageux", "brouillard", "pluie",
            "neige", "averses", "orage", "autre", "inconnu",
        ])

    # ── 8. Numériques → float ─────────────────────────────────────────────────
    for col in ("mois", "occupation", "terminus_encoded", "station_encoded",
                "precipitation", "snowfall", "wind_speed", "temperature"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # nom_ligne, terminus, direction_ref restent en string → KFold target encoding
    return df
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
        # Sous-échantillonnage pour accélérer le GridSearch (max 50 000 lignes)
        _GS_MAX = 50_000
        if len(X) > _GS_MAX:
            rng_gs = _np.random.default_rng(42)
            idx_gs = rng_gs.choice(len(X), size=_GS_MAX, replace=False)
            X_gs, y_gs = X[idx_gs], y[idx_gs]
        else:
            X_gs, y_gs = X, y
        gs = GridSearchCV(sk_clone(modele), param_grid, cv=3, scoring="r2", n_jobs=1)
        gs.fit(X_gs, y_gs)
        best_params = gs.best_params_
        gs_run = True

    kf = KFold(n_splits=n_folds, shuffle=False)
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
# SOUS-ÉCHANTILLONNAGE STRATIFIÉ PAR LIGNE
# ─────────────────────────────────────────────

def _sous_echantillonner(df: pd.DataFrame, group_col: str = "nom_ligne") -> pd.DataFrame:
    """Sous-échantillonne la classe majoritaire (à l'heure) en préservant
    les proportions de chaque ligne de transport.

    Pour chaque ligne : garde tous les passages en retard (classe 1) et
    tire aléatoirement le même nombre de passages à l'heure (classe 0).
    L'ordre chronologique des indices est conservé.
    """
    y = (df[TARGET] >= SEUIL_RETARD).astype(int)
    df_tmp = df.copy()
    df_tmp["_y"] = y

    kept: list[int] = []
    rng = np.random.default_rng(RANDOM_STATE)

    for _, grp in df_tmp.groupby(group_col, sort=False):
        pos_idx = grp[grp["_y"] == 1].index.tolist()
        neg_idx = grp[grp["_y"] == 0].index.tolist()
        n_pos = len(pos_idx)
        if n_pos == 0:
            continue
        n_sample = min(len(neg_idx), n_pos)
        neg_sampled = rng.choice(neg_idx, size=n_sample, replace=False).tolist()
        kept.extend(pos_idx + neg_sampled)

    kept.sort()  # préserve l'ordre chronologique
    return df.loc[kept].reset_index(drop=True)


# ─────────────────────────────────────────────
# ÉVALUATION CLASSIFICATION (async)
# ─────────────────────────────────────────────

async def evaluer_classification(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, dict, dict]:
    """
    Retourne (DataFrame résultats, dict vide pour roc_data, dict {nom: confusion_matrix})
    """
    from sklearn.impute import SimpleImputer

    # ── Affichage déséquilibre initial ────────────────────────────────────────
    y_init = (df[TARGET] >= SEUIL_RETARD).astype(int)
    n_pos_init, n_neg_init = y_init.sum(), (y_init == 0).sum()
    print(f"  Déséquilibre initial — Classe 0: {n_neg_init:,}  |  Classe 1: {n_pos_init:,}"
          f"  (ratio {n_pos_init / max(len(y_init), 1):.1%} retards)")

    # ── Sous-échantillonnage stratifié par ligne ──────────────────────────────
    if "nom_ligne" in df.columns and n_pos_init > 0:
        df = _sous_echantillonner(df, group_col="nom_ligne")
        y_after = (df[TARGET] >= SEUIL_RETARD).astype(int)
        print(f"  Après rééchantillonnage    — Classe 0: {(y_after==0).sum():,}  |  Classe 1: {y_after.sum():,}")

    cols_cat = [c for c in FEATURES_CAT_TARGET if c in df.columns]
    cols_num = [c for c in FEATURES_NUM        if c in df.columns]
    X = df[cols_cat + cols_num]
    y = (df[TARGET] >= SEUIL_RETARD).to_numpy(dtype=int)

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    print(f"  Évaluation : StratifiedKFold k={N_FOLDS}  ({len(X):,} lignes)\n")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=False)
    lignes: list[dict] = []
    cm_data: dict[str, list] = {}
    roc_data: dict[str, tuple] = {}

    for nom, modele in MODELES_CLASSIFICATION.items():
        pipeline = Pipeline([
            ("encoder", ColumnTransformer([
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="inconnu")),
                    ("te",      TargetEncoder(smooth="auto", target_type="binary", random_state=RANDOM_STATE)),
                ]), cols_cat),
                ("num", "passthrough", cols_num),
            ])),
            ("model", modele),
        ])

        auc_scores = cross_val_score(pipeline, X, y, cv=skf, scoring="roc_auc", n_jobs=1)
        y_pred     = cross_val_predict(pipeline, X, y, cv=skf, method="predict",       n_jobs=1)
        y_proba    = cross_val_predict(pipeline, X, y, cv=skf, method="predict_proba", n_jobs=1)[:, 1]

        from sklearn.metrics import roc_curve as _roc_curve
        fpr, tpr, _ = _roc_curve(y, y_proba)
        roc_data[nom] = (fpr.tolist(), tpr.tolist())

        cm = confusion_matrix(y, y_pred).tolist()
        cm_data[nom] = cm
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

        print(
            f"  [Terminé ✓] {nom:<25}"
            f"  AUC={auc_scores.mean():.3f}±{auc_scores.std():.3f}"
            f"  Acc={( tn + tp) / (tn + fp + fn + tp):.3f}"
        )
        print(f"    Matrice de confusion (agrégée) :")
        print(f"      TN={tn:,}  FP={fp:,}")
        print(f"      FN={fn:,}  TP={tp:,}\n")

        lignes.append({
            "Modèle":    nom,
            "AUC":       float(auc_scores.mean()),
            "AUC ±":     float(auc_scores.std()),
            "Accuracy":  float((tn + tp) / (tn + fp + fn + tp)),
            "F1":        float(f1_score(y, y_pred, zero_division=0)),
            "Précision": float(precision_score(y, y_pred, zero_division=0)),
            "Rappel":    float(recall_score(y, y_pred, zero_division=0)),
        })

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
        im = ax.imshow(cm, interpolation="nearest", cmap="YlOrRd")
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

        thresh = cm.max() * 0.6
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                ax.text(col, row, f"{cm[row, col]:,}",
                        ha="center", va="center", fontsize=12, fontweight="bold",
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

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1)

    couleurs = [plt.cm.get_cmap("tab10")(i) for i in range(10)]
    for i, (nom, (fpr, tpr)) in enumerate(sorted(roc_data.items())):
        auc_approx = float(np.trapezoid(tpr, fpr))
        ax.plot(fpr, tpr, lw=2, color=couleurs[i % len(couleurs)],
                label=f"{nom} (AUC={auc_approx:.2f})")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("Courbes ROC", fontsize=13)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

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

    # Filtre RER / Transilien uniquement
    if "nom_ligne" in df.columns:
        avant = len(df)
        df = df[df["nom_ligne"].isin(LIGNES_RER_TRANSILIEN)].reset_index(drop=True)
        print(f"  Filtre RER/Transilien : {avant:,} → {len(df):,} lignes"
              f"  ({avant - len(df):,} retirées)")

    # Trier chronologiquement UNE SEULE FOIS avant tout traitement
    if "date_capture" in df.columns:
        df["date_capture"] = pd.to_datetime(df["date_capture"], errors="coerce")
        df = df.sort_values("date_capture", na_position="first").reset_index(drop=True)
        valid = df["date_capture"].notna()
        print(f"  Données triées chronologiquement"
              f"  ({df.loc[valid, 'date_capture'].min()} → {df.loc[valid, 'date_capture'].max()})")

    df = transformer_dataset(df)

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
