"""
fix_horaires.py — Corrige les colonnes horaire_*_estime au format HH:MM
en les reconstruisant en timestamps ISO complets (UTC).

Usage :
    python fix_horaires.py
    python fix_horaires.py --input dataset_predictions/dataset_final.csv \
                           --output dataset_predictions/dataset_final.csv
"""

import argparse
import sys

import pandas as pd

# ──────────────────────────────────────────────────────────────
# Colonnes à corriger : (estime_a_corriger, prevu_reference)
# ──────────────────────────────────────────────────────────────
PAIRES = [
    ("horaire_arrivee_estime", "horaire_arrivee_prevu"),
    ("horaire_depart_estime",  "horaire_depart_prevu"),
]


def _reconstruire(df: pd.DataFrame, col_estime: str, col_prevu: str) -> pd.Series:
    """
    Pour chaque ligne où col_estime est en format HH:MM :
      1. Extrait la date depuis col_prevu (ISO complet).
      2. Combine date + HH:MM + ':00.000Z' pour former un timestamp ISO.
      3. Gère le passage de minuit (si estime < prevu_heure - 6h → lendemain).
    Retourne la colonne col_estime corrigée (Series de str).
    """
    s_est  = df[col_estime].copy()
    s_prev = df[col_prevu]

    # Masque des lignes en HH:MM
    mask_hhmm = s_est.str.match(r"^\d{2}:\d{2}$", na=False)
    n_hhmm = mask_hhmm.sum()
    if n_hhmm == 0:
        print(f"  {col_estime} : aucune valeur HH:MM, rien à corriger.")
        return s_est

    print(f"  {col_estime} : {n_hhmm:,} valeurs HH:MM à reconstruire...")

    # Parse les timestamps de référence (prevu) pour les lignes concernées
    prevu_dt = pd.to_datetime(s_prev[mask_hhmm], utc=True, errors="coerce")

    # Extrait heure et minute depuis HH:MM
    hhmm_str = s_est[mask_hhmm]
    est_h = hhmm_str.str[:2].astype(int)
    est_m = hhmm_str.str[3:5].astype(int)

    # Date de référence (depuis prevu_dt)
    date_ref = prevu_dt.dt.normalize()          # minuit du jour prevu (UTC)
    prevu_h  = prevu_dt.dt.hour

    # Passage de minuit : si prevu est tard (≥ 21h) et estime est tôt (< 6h)
    # → l'estime appartient au lendemain
    next_day = (prevu_h >= 21) & (est_h < 6)
    date_ref = date_ref + pd.to_timedelta(next_day.values.astype(int), unit="D")

    # Reconstruction du timestamp complet
    estime_dt = date_ref + pd.to_timedelta(est_h * 3600 + est_m * 60, unit="s")

    # Pour les lignes sans prevu_dt (NaT), on ne peut pas reconstruire → on garde NaN
    no_ref = prevu_dt.isna()
    estime_dt = estime_dt.where(~no_ref)
    if no_ref.sum():
        print(f"    {no_ref.sum():,} lignes sans référence prevu → laissées NaN")

    # Formater en ISO UTC (même format que le reste du CSV)
    reconstructed = estime_dt.dt.strftime("%Y-%m-%dT%H:%M:%S.000Z").where(estime_dt.notna())

    s_est[mask_hhmm] = reconstructed
    print(f"    Corrigées avec succès : {(~no_ref).sum():,}")
    return s_est


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="dataset_predictions/dataset_final.csv")
    parser.add_argument("--output", default="dataset_predictions/dataset_final.csv")
    args = parser.parse_args()

    print(f"Chargement : {args.input}")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"  {len(df):,} lignes chargées\n")

    # ── Correction des colonnes horaires ───────────────────────
    for col_estime, col_prevu in PAIRES:
        if col_estime not in df.columns:
            print(f"  {col_estime} absente, ignorée.")
            continue
        if col_prevu not in df.columns:
            print(f"  {col_prevu} absente, impossible de corriger {col_estime}.")
            continue
        df[col_estime] = _reconstruire(df, col_estime, col_prevu)

    # ── Recalcul retard_sec ────────────────────────────────────
    print("\nRecalcul retard_sec...")
    arr_est  = pd.to_datetime(df["horaire_arrivee_estime"], utc=True, errors="coerce")
    arr_prev = pd.to_datetime(df["horaire_arrivee_prevu"],  utc=True, errors="coerce")
    df["retard_sec"] = (arr_est - arr_prev).dt.total_seconds()

    n_total  = len(df)
    n_retard = df["retard_sec"].notna().sum()
    print(f"  retard_sec calculable : {n_retard:,} / {n_total:,} ({n_retard/n_total*100:.1f} %)")

    # ── Sauvegarde ─────────────────────────────────────────────
    print(f"\nSauvegarde : {args.output}")
    df.to_csv(args.output, index=False)
    print("Terminé.")


if __name__ == "__main__":
    main()
