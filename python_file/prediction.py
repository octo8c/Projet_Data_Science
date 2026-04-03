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
from build_dataset import preparer_ml, collecter_snapshot_ml

import numpy as np
import pandas as pd

import tools as tl
from evaluation_modeles import (DOSSIER, evaluer_regression, evaluer_classification)

warnings.filterwarnings("ignore")

COULEURS_MODELES_REGRESSION = {
    "GradientBoosting": "tab:blue",
    "DecisionTree": "tab:orange",
    "ExtraTrees": "tab:green",
    "HistGradientBoosting": "tab:red",
    "RandomForest": "tab:purple",
    "KNeighbors": "tab:brown",
    "Ridge": "tab:pink",
    "LinearRegression": "tab:gray",
}

# ─────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────

def charger(csv_path: str) -> pd.DataFrame:
    """Charge un CSV ML-ready. Si brut, appelle preparer_ml() à la volée."""
    import csv as _csv_mod

    with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as _f:
        header = next(_csv_mod.reader(_f))

    if set(tl.ML_COLONNES).issubset(set(header)):
        df = pd.read_csv(csv_path, low_memory=False)
    else:
        print(f"  CSV brut détecté ({len(header)} cols) — application du pipeline ML…")
        df = preparer_ml(csv_path)

    df[tl.TARGET] = pd.to_numeric(df[tl.TARGET], errors="coerce")
    avant = len(df)
    df = df.dropna(subset=[tl.TARGET]).reset_index(drop=True)
    print(f"  Lignes conservées (retard_sec calculable) : {len(df):,} / {avant:,}")

    for col in tl.FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    return df


def _charger_api() -> pd.DataFrame:
    """Snapshot API temps réel → DataFrame ML-ready."""
    print("  Mode API temps réel (snapshot unique)")
    df = collecter_snapshot_ml()

    df[tl.TARGET] = pd.to_numeric(df[tl.TARGET], errors="coerce")
    avant = len(df)
    df = df.dropna(subset=[tl.TARGET]).reset_index(drop=True)
    print(f"  Lignes avec retard calculable : {len(df):,} / {avant:,}")

    for col in tl.FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    return df

# ─────────────────────────────────────────────
# GRAPHIQUES ROC
# ─────────────────────────────────────────────

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
    ax.set_title(f"Courbes ROC — Classification retard > {tl.SEUIL_RETARD}s\n"
                 f"(moyenne sur {tl.N_FOLDS} folds StratifiedKFold)", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    os.makedirs(DOSSIER, exist_ok=True)
    chemin = os.path.join(DOSSIER, f"roc_curves_{horodatage}.png")
    fig.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return chemin

def tracer_scores_regression(resultats: pd.DataFrame, horodatage: str) -> str:
    """
    Trace un bar chart du score global de régression par modèle.
    Retourne le chemin du PNG créé.
    """
    import matplotlib.pyplot as plt

    tri = resultats.sort_values("Score global", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    couleurs = [COULEURS_MODELES_REGRESSION.get(m, "tab:blue") for m in tri["Modèle"]]
    ax.bar(tri["Modèle"], tri["Score global"], color=couleurs)

    ax.set_title("Score global des modèles de régression", fontsize=13)
    ax.set_xlabel("Modèle", fontsize=11)
    ax.set_ylabel("Score global", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=25, ha="right")

    for i, v in enumerate(tri["Score global"]):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)

    fig.tight_layout()

    os.makedirs(DOSSIER, exist_ok=True)
    chemin = os.path.join(DOSSIER, f"regression_scores_{horodatage}.png")
    fig.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return chemin

def tracer_radar_regression(resultats: pd.DataFrame, proche_col: str, horodatage: str) -> str:
    """
    Trace un radar chart comparant les modèles de régression
    sur plusieurs métriques normalisées dans [0, 1].
    Plus la valeur est grande, meilleur est le modèle.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    df = resultats.copy()

    def normaliser_positif(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        s_min, s_max = s.min(), s.max()
        if pd.isna(s_min) or pd.isna(s_max):
            return pd.Series(0.0, index=s.index)
        if s_max == s_min:
            return pd.Series(1.0, index=s.index)
        return (s - s_min) / (s_max - s_min)

    def normaliser_negatif(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        s_min, s_max = s.min(), s.max()
        if pd.isna(s_min) or pd.isna(s_max):
            return pd.Series(0.0, index=s.index)
        if s_max == s_min:
            return pd.Series(1.0, index=s.index)
        return (s_max - s) / (s_max - s_min)

    radar_df = pd.DataFrame({
        "Modèle": df["Modèle"],
        "R²": normaliser_positif(df["R²"]),
        "MAE": normaliser_negatif(df["MAE (s)"]),
        "RMSE": normaliser_negatif(df["RMSE (s)"]),
        "MAPE": normaliser_negatif(df["MAPE (%)"]),
        "Proche": normaliser_positif(df[proche_col]),
    })

    categories = ["R²", "MAE", "RMSE", "MAPE", "Proche"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for _, row in radar_df.iterrows():
        values = [row[c] for c in categories]
        values += values[:1]
        couleur = COULEURS_MODELES_REGRESSION.get(row["Modèle"], "tab:blue")
        ax.plot(angles, values, linewidth=2, label=row["Modèle"], color=couleur)
        ax.fill(angles, values, alpha=0.08, color=couleur)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
    ax.set_ylim(0, 1)

    ax.set_title(
        "Radar chart — comparaison des modèles de régression\n(métriques normalisées, plus grand = mieux)",
        fontsize=13,
        pad=25,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    os.makedirs(DOSSIER, exist_ok=True)
    chemin = os.path.join(DOSSIER, f"regression_radar_{horodatage}.png")
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

    if len(df) < tl.N_FOLDS * 2:
        print(f"  Pas assez de données ({len(df)} lignes). Abandon.")
        return

    from datetime import datetime
    from rapport import ecrire_rapport_regression, ecrire_rapport_classification

    horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(DOSSIER, exist_ok=True)

    # ══════════════════════════════════════════
    # RÉGRESSION
    # ══════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  RÉGRESSION  (cible : retard_sec en secondes)")
    print("=" * 65)

    res_reg = await evaluer_regression(df, "ML-ready")

    proche_col = f"Proche≤{tl.SEUIL_PROCHE_S}s (%)"
    res_reg = tl.ajouter_score_global_regression(res_reg, proche_col)
    _afficher_classements_regression(res_reg, proche_col)
    _afficher_meilleur_modele_regression(res_reg, proche_col)

    out_reg_plot = tracer_scores_regression(res_reg, horodatage)
    print(f"Graphique score régression  → {out_reg_plot}")

    out_reg_radar = tracer_radar_regression(res_reg, proche_col, horodatage)
    print(f"Radar régression            → {out_reg_radar}")

    out_csv_reg = os.path.join(DOSSIER, f"resultats_regression_{horodatage}.csv")
    res_reg.to_csv(out_csv_reg, index=False)
    print(f"\nRésultats CSV régression    → {out_csv_reg}")

    out_md_reg = ecrire_rapport_regression(
        resultats    = res_reg,
        dossier      = DOSSIER,
        horodatage   = horodatage,
        csv_source   = args.csv,
        n_lignes     = len(df),
        features     = tl.FEATURES,
        n_folds      = tl.N_FOLDS,
        seuil_proche = tl.SEUIL_PROCHE_S,
        score_png    = os.path.basename(out_reg_plot),
        radar_png    = os.path.basename(out_reg_radar),
    )
    print(f"Rapport Markdown régression → {out_md_reg}")

    # ══════════════════════════════════════════
    # CLASSIFICATION
    # ══════════════════════════════════════════
    print("\n" + "=" * 65)
    print(f"  CLASSIFICATION  (cible : retard > {tl.SEUIL_RETARD}s)")
    print("=" * 65)

    res_clf, roc_data = await evaluer_classification(df, "ML-ready")

    _afficher_classements_classification(res_clf)

    out_csv_clf = os.path.join(DOSSIER, f"resultats_classification_{horodatage}.csv")
    res_clf.to_csv(out_csv_clf, index=False)
    print(f"\nRésultats CSV classification → {out_csv_clf}")

    out_roc = tracer_courbes_roc(roc_data, horodatage)
    print(f"Courbes ROC (PNG)            → {out_roc}")

    out_md_clf = ecrire_rapport_classification(
        resultats    = res_clf,
        dossier      = DOSSIER,
        horodatage   = horodatage,
        csv_source   = args.csv,
        n_lignes     = len(df),
        features     = tl.FEATURES,
        n_folds      = tl.N_FOLDS,
        seuil_retard = tl.SEUIL_RETARD,
        roc_png      = os.path.basename(out_roc),
        cm_pngs      = [],
    )
    print(f"Rapport Markdown classification → {out_md_clf}")

def _afficher_meilleur_modele_regression(resultats: pd.DataFrame, proche_col: str) -> None:
    best = tl.selectionner_meilleur_modele_regression(resultats)

    print(f"\n{'=' * 65}")
    print(f"  MEILLEUR MODÈLE RÉGRESSION ({tl.CRITERE_SELECTION_REGRESSION})")
    print("=" * 65)
    print(f"Modèle retenu : {best['Modèle']}")
    print(f"Dataset       : {best['Dataset']}")
    print(f"Score global  : {best['Score global']:.3f}")
    print(f"R²            : {best['R²']:.3f} ± {best.get('R² ±', float('nan')):.3f}")
    print(f"MAE (s)       : {best['MAE (s)']:.1f} ± {best.get('MAE (s) ±', float('nan')):.1f}")
    print(f"RMSE (s)      : {best['RMSE (s)']:.1f} ± {best.get('RMSE (s) ±', float('nan')):.1f}")
    print(f"MAPE (%)      : {best['MAPE (%)']:.1f} ± {best.get('MAPE (%) ±', float('nan')):.1f}")
    print(f"{proche_col:<14}: {best[proche_col]:.1f}% ± {best.get(proche_col + ' ±', float('nan')):.1f}%")
    print(f"Params        : {best['Meilleurs params']}\n")

def _afficher_classements_regression(resultats: pd.DataFrame, proche_col: str) -> None:
    metriques = [
        ("Score global", True, "Score global pondéré (↑ mieux)"),
        ("R²",       True,  "R² (↑ mieux)"),
        ("MAE (s)",  False, "MAE en secondes (↓ mieux)"),
        ("RMSE (s)", False, "RMSE (↓ mieux)"),
        ("MAPE (%)", False, "MAPE en % (↓ mieux)"),
        (proche_col, True,  f"Proche≤{tl.SEUIL_PROCHE_S}s % (↑ mieux)"),
    ]
    for col, desc_asc, titre in metriques:
        std_col = col + " ±"
        tri = resultats.sort_values(col, ascending=not desc_asc).reset_index(drop=True)
        tri.insert(0, "Rang", range(1, len(tri) + 1))
        tri[f"{col} (moy ± std)"] = tri.apply(
            lambda r: f"{r[col]:.3f} ± {r[std_col]:.3f}"
            if std_col in tri.columns and pd.notna(r.get(std_col, np.nan))
            else f"{r[col]:.3f}",
            axis=1,
        )
        print(f"\n{'=' * 65}")
        print(f"  {titre}")
        print("=" * 65)
        print(tri[["Rang", "Modèle", f"{col} (moy ± std)"]].to_string(index=False))


def _afficher_classements_classification(resultats: pd.DataFrame) -> None:
    col = "AUC"
    std_col = "AUC ±"

    tri = resultats.sort_values(col, ascending=False).reset_index(drop=True)
    tri.insert(0, "Rang", range(1, len(tri) + 1))
    tri[f"{col} (moy ± std)"] = tri.apply(
        lambda r: f"{r[col]:.3f} ± {r[std_col]:.3f}",
        axis=1,
    )

    print(f"\n{'=' * 65}")
    print("  AUC-ROC (↑ mieux)")
    print("=" * 65)
    print(tri[["Rang", "Modèle", f"{col} (moy ± std)"]].to_string(index=False))


if __name__ == "__main__":
    asyncio.run(main())
