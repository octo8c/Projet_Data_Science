"""
data_preparation.py — Chargement et préparation des données pour la prédiction.

Règle de filtrage stricte :
  On ne conserve que les lignes ayant SIMULTANÉMENT :
    - horaire_arrivee_estime  défini (non NaN, non vide)
    - horaire_arrivee_prevu   défini (non NaN, non vide)
  => retard_sec est ensuite calculé sur ces lignes uniquement.
"""

import sys
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────
# CONSTANTES PARTAGÉES (importées dans prediction.py)
# ─────────────────────────────────────────────

TARGET       = "retard_sec"
FEATURES_CAT = ["nom_ligne", "jour_semaine", "periode_journee", "meteo"]
FEATURES_NUM = ["heure_tranche", "mois", "jour_ferie", "occupation"]
FEATURES     = FEATURES_CAT + FEATURES_NUM


# ─────────────────────────────────────────────
# HELPERS INTERNES
# ─────────────────────────────────────────────

def _est_valide(serie: pd.Series) -> pd.Series:
    """True si chaque valeur est présente (non NaN et non chaîne vide)."""
    return serie.notna() & (serie.astype(str).str.strip() != "")


def _col_ou_vide(df: pd.DataFrame, col: str) -> pd.Series:
    """Retourne la colonne si elle existe, sinon une Series vide de même index."""
    return df[col] if col in df.columns else pd.Series("", index=df.index)


def _encoder_categoriques(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode les colonnes catégorielles (NaN → '_inconnu')."""
    df = df.copy()
    for col in FEATURES_CAT:
        if col not in df.columns:
            df[col] = 0
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].fillna("_inconnu").astype(str))
    for col in FEATURES_NUM:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ─────────────────────────────────────────────
# FONCTION PRINCIPALE
# ─────────────────────────────────────────────

def charger(csv_path: str, build_features_flag: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge le CSV et retourne (df_brut, df_corrige) prêts pour l'entraînement.

    Étapes :
      1. Chargement (+ build_features/impute_missing si demandé)
      2. Filtrage strict sur horaire_arrivee_estime et horaire_arrivee_prevu
      3. Calcul de retard_sec à partir des horaires filtrés
      4. Encodage des catégorielles
    """
    # ── 1. Chargement ────────────────────────────────────────
    if build_features_flag:
        from prim_global_csv import build_features, impute_missing  # type: ignore
        print("  Application de build_features()...")
        df_base         = build_features(csv_path)
        print("  Application de impute_missing()...")
        df_corrige_base = impute_missing(df_base.copy())
    else:
        df_base         = pd.read_csv(csv_path, low_memory=False)
        df_corrige_base = df_base.copy()
        # Imputation simple des features numériques pour l'expérience B
        for col in FEATURES_NUM:
            if col in df_corrige_base.columns:
                s        = pd.to_numeric(df_corrige_base[col], errors="coerce")
                mean_val = s.mean()
                df_corrige_base[col] = s.fillna(mean_val if pd.notna(mean_val) else 0.0)

    # ── 2. Filtrage strict sur les horaires d'arrivée ────────
    # Le masque est calculé sur df_base (mêmes colonnes brutes dans les deux cas)
    avant  = len(df_base)
    masque = (
        _est_valide(_col_ou_vide(df_base, "horaire_arrivee_estime")) &
        _est_valide(_col_ou_vide(df_base, "horaire_arrivee_prevu"))
    )
    df_base         = df_base[masque].reset_index(drop=True)
    df_corrige_base = df_corrige_base[masque].reset_index(drop=True)

    print(f"  Lignes supprimées (horaires d'arrivée manquants) : {avant - len(df_base):,}")
    print(f"  Lignes conservées                                 : {len(df_base):,}")

    # ── 3. Calcul de retard_sec ──────────────────────────────
    # On recalcule systématiquement depuis les horaires (garantit la cohérence)
    arr_est  = pd.to_datetime(df_base["horaire_arrivee_estime"], utc=True, errors="coerce")
    arr_prev = pd.to_datetime(df_base["horaire_arrivee_prevu"],  utc=True, errors="coerce")
    df_base[TARGET]         = (arr_est - arr_prev).dt.total_seconds().round().astype(float)
    df_corrige_base[TARGET] = df_base[TARGET].values  # même calcul, même base

    # Suppression des lignes où le parsing datetime a échoué (retard_sec encore NaN)
    masque_valide = df_base[TARGET].notna()
    n_invalides   = (~masque_valide).sum()
    if n_invalides > 0:
        df_base         = df_base[masque_valide].reset_index(drop=True)
        df_corrige_base = df_corrige_base[masque_valide].reset_index(drop=True)
        print(f"  Lignes supprimées (parsing datetime échoué)      : {n_invalides:,}")
        print(f"  Lignes finales                                    : {len(df_base):,}")

    # ── 4. Encodage des catégorielles ────────────────────────
    df_brut_enc    = _encoder_categoriques(df_base)
    df_corrige_enc = _encoder_categoriques(df_corrige_base)

    # Imputation finale des NaN résiduels dans les features du dataset corrigé
    # (colonnes absentes du CSV brut — ex: mois, occupation, jour_ferie)
    for col in FEATURES:
        if col in df_corrige_enc.columns and df_corrige_enc[col].isna().any():
            mean_val = df_corrige_enc[col].mean()
            df_corrige_enc[col] = df_corrige_enc[col].fillna(
                mean_val if pd.notna(mean_val) else 0.0
            )

    return df_brut_enc, df_corrige_enc
