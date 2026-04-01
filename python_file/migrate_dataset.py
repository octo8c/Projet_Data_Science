"""
Fusion de passages_tglobal.csv et passages_global_old.csv
vers un schéma commun (CSV_COLONNES de build_dataset.py).

Usage:
    python migrate_dataset.py [--output PATH]

Sortie par défaut : dataset_predictions/dataset_fusionne.csv
"""

import argparse
import os
import sys
import pandas as pd
from tools import LIGNES_PAR_REF, CSV_COLONNES


def _calc_retard(df: pd.DataFrame) -> pd.Series:
    """Calcule retard_sec = horaire_depart_estime - horaire_depart_prevu (en secondes)."""
    dep_est  = pd.to_datetime(df["horaire_depart_estime"], utc=True, errors="coerce")
    dep_prev = pd.to_datetime(df["horaire_depart_prevu"],  utc=True, errors="coerce")
    retard = (dep_est - dep_prev).dt.total_seconds().round()
    # Fallback sur arrivée si départ absent
    mask = retard.isna()
    if mask.any():
        arr_est  = pd.to_datetime(df.loc[mask, "horaire_arrivee_estime"], utc=True, errors="coerce")
        arr_prev = pd.to_datetime(df.loc[mask, "horaire_arrivee_prevu"],  utc=True, errors="coerce")
        retard[mask] = (arr_est - arr_prev).dt.total_seconds().round()
    return retard.astype("Int64")


def transformer_tglobal(path: str) -> pd.DataFrame:
    """Transforme passages_tglobal.csv vers le schéma cible."""
    print(f"  Lecture {path} ...")
    df = pd.read_csv(path, dtype=str, low_memory=False)

    # nom_ligne absent → dériver depuis line_ref
    df["nom_ligne"] = df["line_ref"].map(LIGNES_PAR_REF).fillna("")

    # alerte_active et categorie_alerte absents
    df["alerte_active"]    = "False"
    df["categorie_alerte"] = "aucune"

    # date_capture absent → vide (sera imputé plus tard)
    if "date_capture" not in df.columns:
        df["date_capture"] = ""

    # retard_sec : calculer depuis les timestamps
    df["retard_sec"] = _calc_retard(df)

    return df[CSV_COLONNES].copy()


def transformer_global_old(path: str) -> pd.DataFrame:
    """Transforme passages_global_old.csv vers le schéma cible."""
    print(f"  Lecture {path} ...")
    df = pd.read_csv(path, dtype=str, low_memory=False)

    # nom_ligne déjà présent mais on le recalcule pour cohérence
    df["nom_ligne"] = df["line_ref"].map(LIGNES_PAR_REF).fillna(df["nom_ligne"].fillna(""))

    # alerte_active / categorie_alerte absents
    df["alerte_active"]    = "False"
    df["categorie_alerte"] = "aucune"

    # date_capture : dériver depuis timestamp_collecte
    df["date_capture"] = pd.to_datetime(
        df["timestamp_collecte"], errors="coerce"
    ).dt.strftime("%Y-%m-%dT%H:%M:%S.000Z").fillna("")

    # retard_sec : utiliser retard_depart_sec si disponible, sinon calculer
    if "retard_depart_sec" in df.columns:
        df["retard_sec"] = pd.to_numeric(df["retard_depart_sec"], errors="coerce").astype("Int64")
        # Pour les lignes sans horaire théorique, retard_sec reste NaN (correct)
    else:
        df["retard_sec"] = _calc_retard(df)

    return df[CSV_COLONNES].copy()


def main():
    parser = argparse.ArgumentParser(description="Fusionne passages_tglobal et passages_global_old")
    parser.add_argument(
        "--tglobal", default="dataset_predictions/passages_tglobal.csv",
        help="Chemin vers passages_tglobal.csv"
    )
    parser.add_argument(
        "--old", default="dataset_predictions/passages_global_old.csv",
        help="Chemin vers passages_global_old.csv"
    )
    parser.add_argument(
        "--output", default="dataset_predictions/dataset_fusionne.csv",
        help="Chemin du fichier de sortie"
    )
    args = parser.parse_args()

    for p in [args.tglobal, args.old]:
        if not os.path.exists(p):
            print(f"Fichier introuvable : {p}", file=sys.stderr)
            sys.exit(1)

    print("=== Transformation passages_tglobal ===")
    df_tglobal = transformer_tglobal(args.tglobal)
    print(f"  -> {len(df_tglobal):,} lignes")

    print("\n=== Transformation passages_global_old ===")
    df_old = transformer_global_old(args.old)
    print(f"  -> {len(df_old):,} lignes")

    print("\n=== Fusion ===")
    df_final = pd.concat([df_tglobal, df_old], ignore_index=True)
    print(f"  -> Total : {len(df_final):,} lignes")

    # Dédoublonnage sur les colonnes clés (même passage capturé deux fois)
    cles_dedup = ["line_ref", "stop_ref", "horaire_depart_estime", "horaire_arrivee_estime"]
    avant = len(df_final)
    df_final = df_final.drop_duplicates(subset=cles_dedup, keep=False)
    print(f"  -> Après dédoublonnage : {len(df_final):,} lignes ({avant - len(df_final):,} lignes supprimées)")

    print(f"\n=== Écriture → {args.output} ===")
    df_final.to_csv(args.output, index=False, lineterminator="\n", encoding="utf-8-sig")
    print("  Done.")

    # Résumé rapide
    print("\n=== Résumé du dataset fusionné ===")
    print(f"  Colonnes      : {list(df_final.columns)}")
    print(f"  Lignes totales: {len(df_final):,}")
    print(f"  retard_sec NaN: {df_final['retard_sec'].isna().mean()*100:.1f}%")
    print(f"  nom_ligne vide: {(df_final['nom_ligne'] == '').mean()*100:.1f}%")


if __name__ == "__main__":
    main()
