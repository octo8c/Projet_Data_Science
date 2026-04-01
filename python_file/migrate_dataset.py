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

# ── Mapping line_ref → nom_ligne (extrait de build_dataset.py) ───────────────
LIGNES_PAR_REF = {
    "STIF:Line::C01371:": "Métro 1",  "STIF:Line::C01372:": "Métro 2",
    "STIF:Line::C01373:": "Métro 3",  "STIF:Line::C01386:": "Métro 3b",
    "STIF:Line::C01374:": "Métro 4",  "STIF:Line::C01375:": "Métro 5",
    "STIF:Line::C01376:": "Métro 6",  "STIF:Line::C01377:": "Métro 7",
    "STIF:Line::C01387:": "Métro 7b", "STIF:Line::C01378:": "Métro 8",
    "STIF:Line::C01379:": "Métro 9",  "STIF:Line::C01380:": "Métro 10",
    "STIF:Line::C01381:": "Métro 11", "STIF:Line::C01382:": "Métro 12",
    "STIF:Line::C01383:": "Métro 13", "STIF:Line::C01384:": "Métro 14",
    "STIF:Line::C01742:": "RER A",    "STIF:Line::C01743:": "RER B",
    "STIF:Line::C01727:": "RER C",    "STIF:Line::C01728:": "RER D",
    "STIF:Line::C01729:": "RER E",
    "STIF:Line::C01737:": "Ligne H",  "STIF:Line::C01738:": "Ligne J",
    "STIF:Line::C01739:": "Ligne K",  "STIF:Line::C01740:": "Ligne L",
    "STIF:Line::C01741:": "Ligne N",  "STIF:Line::C01744:": "Ligne P",
    "STIF:Line::C01745:": "Ligne R",  "STIF:Line::C01746:": "Ligne U",
    "STIF:Line::C01389:": "Tram T1",  "STIF:Line::C01390:": "Tram T2",
    "STIF:Line::C01391:": "Tram T3a", "STIF:Line::C01679:": "Tram T3b",
    "STIF:Line::C01392:": "Tram T4",  "STIF:Line::C01775:": "Tram T5",
    "STIF:Line::C01776:": "Tram T6",  "STIF:Line::C01777:": "Tram T7",
    "STIF:Line::C01778:": "Tram T8",  "STIF:Line::C02317:": "Tram T9",
    "STIF:Line::C02316:": "Tram T10", "STIF:Line::C02024:": "Tram T11",
    "STIF:Line::C02048:": "Tram T13",
}

# Schéma cible (ordre des colonnes dans le CSV final)
CSV_COLONNES = [
    "line_ref", "nom_ligne", "operateur", "direction_ref", "terminus",
    "stop_ref", "nom_arret",
    "horaire_arrivee_prevu", "horaire_depart_prevu",
    "horaire_arrivee_estime", "horaire_depart_estime",
    "arrivee_prevue_hhmm", "depart_prevu_hhmm", "depart_estime_hhmm",
    "jour_semaine", "heure_tranche", "periode_journee",
    "alerte_active", "categorie_alerte", "date_capture",
    "retard_sec",
]


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
