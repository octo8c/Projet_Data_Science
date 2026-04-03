"""
evaluation.py — Évaluation des modèles avec Pipeline + KFold target encoding.

Target encoding via sklearn Pipeline :
  Pipeline([
      ('encoder', ColumnTransformer : TargetEncoder sur cols catégorielles + passthrough num),
      ('model', <estimateur>)
  ])
  → cross_val_score / cross_val_predict (pas de data leakage, refit à chaque fold)

Usage :
    python evaluation.py --csv dataset_predictions/dataset_ml.csv
    python evaluation.py --csv dataset_predictions/passages_tglobal.csv
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from tools import TARGET, SEUIL_RETARD, N_FOLDS, RANDOM_STATE

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

COLS_CAT = ["nom_ligne", "terminus"]

COLS_NUM = [
    "mois", "heure_tranche", "jour_ferie", "occupation",
    "direction_ref",
    "precipitation", "snowfall", "wind_speed", "temperature",
    "periode_Nuit", "periode_Pointe matin", "periode_Creuse matin",
    "periode_Méridienne", "periode_Creuse après-midi", "periode_Pointe soir", "periode_Soirée",
    "alerte_aucune", "alerte_greve", "alerte_incident", "alerte_travaux",
    "alerte_meteo", "alerte_retard", "alerte_voyageur", "alerte_autre",
    "jour_Lundi", "jour_Mardi", "jour_Mercredi", "jour_Jeudi",
    "jour_Vendredi", "jour_Samedi", "jour_Dimanche",
    "meteo_ensoleille", "meteo_nuageux", "meteo_brouillard", "meteo_pluie",
    "meteo_neige", "meteo_averses", "meteo_orage", "meteo_autre", "meteo_inconnu",
]

MODELES_REGRESSION = {
    "LinearRegression":     LinearRegression(),
    "Ridge":                Ridge(),
    "DecisionTree":         DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
    "RandomForest":         RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    "GradientBoosting":     GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    "KNeighbors":           KNeighborsRegressor(n_neighbors=10, n_jobs=-1),
}

MODELES_CLASSIFICATION = {
    "LogisticRegression":   LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "DecisionTree":         DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
    "RandomForest":         RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    "GradientBoosting":     GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    "KNeighbors":           KNeighborsClassifier(n_neighbors=10, n_jobs=-1),
}


# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────

def charger(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    avant = len(df)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    print(f"  {len(df):,} lignes avec retard_sec valide (/{avant:,} total)")
    return df


# ─────────────────────────────────────────────
# CONSTRUCTION DU PIPELINE
# ─────────────────────────────────────────────

def _build_pipeline(modele, task: str) -> Pipeline:
    """
    Pipeline([
        ('encoder', ColumnTransformer : TargetEncoder sur cols cat + passthrough num),
        ('model', modele)
    ])
    Le TargetEncoder est refit à chaque fold par cross_val_score/cross_val_predict
    → aucun data leakage.
    """
    target_type = "continuous" if task == "regression" else "binary"
    preprocessor = ColumnTransformer(
        transformers=[
            ("target_enc", TargetEncoder(
                smooth="auto",
                target_type=target_type,
                random_state=RANDOM_STATE,
            ), COLS_CAT),
            ("num", "passthrough", COLS_NUM),
        ]
    )
    return Pipeline([
        ("encoder", preprocessor),
        ("model", modele),
    ])


def _preparer_X(df: pd.DataFrame) -> pd.DataFrame:
    """Retourne un DataFrame avec les colonnes cat (str) + num (float) disponibles."""
    cols_cat_ok = [c for c in COLS_CAT if c in df.columns]
    cols_num_ok = [c for c in COLS_NUM if c in df.columns]

    X = df[cols_cat_ok + cols_num_ok].copy()
    for c in cols_cat_ok:
        X[c] = X[c].astype(str)
    for c in cols_num_ok:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    # Mettre à jour les listes globales selon ce qui est réellement présent
    global COLS_CAT, COLS_NUM
    COLS_CAT = cols_cat_ok
    COLS_NUM = cols_num_ok
    return X


# ─────────────────────────────────────────────
# ÉVALUATION RÉGRESSION
# ─────────────────────────────────────────────

def evaluer_regression(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("RÉGRESSION — prédiction de retard_sec (scoring : R²)")
    print("=" * 60)

    X = _preparer_X(df)
    y = df[TARGET].to_numpy(dtype=float)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    lignes = []
    for nom, modele in MODELES_REGRESSION.items():
        pipeline = _build_pipeline(modele, "regression")
        scores   = cross_val_score(pipeline, X, y, cv=kf, scoring="r2", n_jobs=1)
        ligne = {
            "Modèle": nom,
            "R² moyen": round(float(scores.mean()), 4),
            "R² ±":    round(float(scores.std()),  4),
            "Scores par fold": [round(s, 4) for s in scores.tolist()],
        }
        lignes.append(ligne)
        folds_str = "  ".join(f"fold{i+1}={s:+.4f}" for i, s in enumerate(scores))
        print(f"  {nom:<25}  R²={scores.mean():+.4f} ± {scores.std():.4f}   [{folds_str}]")

    return (
        pd.DataFrame(lignes)
        .sort_values("R² moyen", ascending=False)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────
# ÉVALUATION CLASSIFICATION + MATRICES DE CONFUSION
# ─────────────────────────────────────────────

def evaluer_classification(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print(f"CLASSIFICATION — retard > {SEUIL_RETARD}s  (scoring : AUC-ROC)")
    print("=" * 60)

    X = _preparer_X(df)
    y = (df[TARGET] > SEUIL_RETARD).astype(int).to_numpy()
    print(f"  Taux positifs (retard > {SEUIL_RETARD}s) : {y.mean():.1%}")

    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    lignes = []
    n_mod  = len(MODELES_CLASSIFICATION)

    fig, axes = plt.subplots(1, n_mod, figsize=(4 * n_mod, 4))
    if n_mod == 1:
        axes = [axes]

    for ax, (nom, modele) in zip(axes, MODELES_CLASSIFICATION.items()):
        pipeline = _build_pipeline(modele, "classification")

        # AUC via cross_val_score
        scores = cross_val_score(pipeline, X, y, cv=skf, scoring="roc_auc", n_jobs=1)

        # Prédictions agrégées sur tous les folds pour la matrice de confusion
        y_pred = cross_val_predict(pipeline, X, y, cv=skf, method="predict", n_jobs=1)
        cm     = confusion_matrix(y, y_pred)

        ligne = {
            "Modèle":    nom,
            "AUC moyen": round(float(scores.mean()), 4),
            "AUC ±":     round(float(scores.std()),  4),
            "Scores par fold": [round(s, 4) for s in scores.tolist()],
        }
        lignes.append(ligne)
        folds_str = "  ".join(f"fold{i+1}={s:.4f}" for i, s in enumerate(scores))
        print(f"  {nom:<25}  AUC={scores.mean():.4f} ± {scores.std():.4f}   [{folds_str}]")

        # Affichage matrice de confusion
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["À l'heure", "En retard"],
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{nom}\nAUC={scores.mean():.3f}", fontsize=9)

    fig.suptitle(
        f"Matrices de confusion — {N_FOLDS}-fold StratifiedKFold\n"
        f"(prédictions agrégées, seuil retard > {SEUIL_RETARD}s)",
        fontsize=11,
    )
    plt.tight_layout()

    dossier = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "resultats_modeles",
    )
    os.makedirs(dossier, exist_ok=True)
    path_fig = os.path.join(dossier, "confusion_matrices.png")
    plt.savefig(path_fig, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n  Matrices de confusion sauvegardées : {path_fig}")

    return (
        pd.DataFrame(lignes)
        .sort_values("AUC moyen", ascending=False)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────
# def sauvegarder(df_reg, df_clf):
#     """Sauvegarde les résultats en CSV — désactivé pour l'instant."""
#     dossier = os.path.join(
#         os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
#         "resultats_modeles",
#     )
#     os.makedirs(dossier, exist_ok=True)
#     df_reg.to_csv(os.path.join(dossier, "evaluation_regression.csv"), index=False, encoding="utf-8-sig")
#     df_clf.to_csv(os.path.join(dossier, "evaluation_classification.csv"), index=False, encoding="utf-8-sig")
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Chemin vers le CSV")
    args = parser.parse_args()

    print(f"\nChargement : {args.csv}")
    df = charger(args.csv)

    df_reg = evaluer_regression(df)
    df_clf = evaluer_classification(df)

    print("\n\n── Résultats régression ──")
    print(df_reg[["Modèle", "R² moyen", "R² ±"]].to_string(index=False))

    print("\n── Résultats classification ──")
    print(df_clf[["Modèle", "AUC moyen", "AUC ±"]].to_string(index=False))


if __name__ == "__main__":
    main()
