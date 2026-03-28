"""
migrate_dataset.py — Migration du CSV passages_tglobal de l'ancien schéma (18 cols)
vers le schéma courant (19 cols, build_dataset.py CSV_COLONNES).

Ancien schéma (18 cols) :
    nom_ligne, line_ref, operateur, direction_ref, terminus, date_course,
    stop_ref, nom_arret, horaire_*, arrivee_prevue_hhmm, depart_prevu_hhmm,
    depart_estime_hhmm, jour_semaine, heure_tranche, periode_journee

Nouveau schéma (19 cols, CSV_COLONNES) :
    line_ref, operateur, direction_ref, terminus, stop_ref, nom_arret,
    horaire_*, arrivee_prevue_hhmm, depart_prevu_hhmm, depart_estime_hhmm,
    jour_semaine, heure_tranche, periode_journee,
    alerte_active, categorie_alerte, date_capture

Traitements appliqués aux anciennes lignes :
  - Suppression de 'nom_ligne'   (recalculée dans build_features)
  - Suppression de 'date_course' (non utilisée dans le schéma courant)
  - alerte_active    = False   (non collectée à l'époque → "pas d'alerte connue")
  - categorie_alerte = 'aucune'
  - date_capture     = ''      (météo imputée par moyenne globale dans impute_missing)

Usage :
    python migrate_dataset.py
    python migrate_dataset.py --source dataset_predictions/passages_tglobal.csv \\
                              --output dataset_predictions/passages_tglobal_migre.csv
"""

import argparse
import csv
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset import CSV_COLONNES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# SCHÉMAS CONNUS
# ─────────────────────────────────────────────

# Schéma intermédiaire 18 cols (nom_ligne en tête, date_course, sans alerte/date_capture)
_OLD_HEADER_18 = [
    "nom_ligne", "line_ref", "operateur", "direction_ref", "terminus", "date_course",
    "stop_ref", "nom_arret",
    "horaire_arrivee_prevu", "horaire_depart_prevu",
    "horaire_arrivee_estime", "horaire_depart_estime",
    "arrivee_prevue_hhmm", "depart_prevu_hhmm", "depart_estime_hhmm",
    "jour_semaine", "heure_tranche", "periode_journee",
]

# Schéma précoce 17 cols : identique au schéma courant SANS alerte_active/categorie_alerte,
# date_capture en position 16. Détectable car vals[0] commence par "STIF:" (line_ref en tête).
_EARLY_HEADER_17 = list(CSV_COLONNES[:16]) + ["date_capture"]

_N_NEW    = len(CSV_COLONNES)      # 19
_N_OLD_18 = len(_OLD_HEADER_18)    # 18


# ─────────────────────────────────────────────
# CONVERSION
# ─────────────────────────────────────────────

def _mapper_schema_18(vals: list) -> dict:
    """Schéma intermédiaire 18 cols (nom_ligne en tête). Supprime nom_ligne et date_course."""
    row = dict(zip(_OLD_HEADER_18[:len(vals)], vals))
    return {
        "line_ref":               row.get("line_ref", ""),
        "operateur":              row.get("operateur", ""),
        "direction_ref":          row.get("direction_ref", ""),
        "terminus":               row.get("terminus", ""),
        "stop_ref":               row.get("stop_ref", ""),
        "nom_arret":              row.get("nom_arret", ""),
        "horaire_arrivee_prevu":  row.get("horaire_arrivee_prevu", ""),
        "horaire_depart_prevu":   row.get("horaire_depart_prevu", ""),
        "horaire_arrivee_estime": row.get("horaire_arrivee_estime", ""),
        "horaire_depart_estime":  row.get("horaire_depart_estime", ""),
        "arrivee_prevue_hhmm":    row.get("arrivee_prevue_hhmm", ""),
        "depart_prevu_hhmm":      row.get("depart_prevu_hhmm", ""),
        "depart_estime_hhmm":     row.get("depart_estime_hhmm", ""),
        "jour_semaine":           row.get("jour_semaine", ""),
        "heure_tranche":          row.get("heure_tranche", ""),
        "periode_journee":        row.get("periode_journee", ""),
        "alerte_active":          False,
        "categorie_alerte":       "aucune",
        "date_capture":           "",    # météo imputée par moyenne globale
    }


def _mapper_schema_precoce_17(vals: list) -> dict:
    """
    Schéma précoce 17 cols : line_ref en tête, date_capture en pos 16, sans champs alerte.
    Colonnes 0-15 = CSV_COLONNES[0:16] (même ordre). date_capture conservée.
    """
    row = dict(zip(_EARLY_HEADER_17[:len(vals)], vals))
    return {
        "line_ref":               row.get("line_ref", ""),
        "operateur":              row.get("operateur", ""),
        "direction_ref":          row.get("direction_ref", ""),
        "terminus":               row.get("terminus", ""),
        "stop_ref":               row.get("stop_ref", ""),
        "nom_arret":              row.get("nom_arret", ""),
        "horaire_arrivee_prevu":  row.get("horaire_arrivee_prevu", ""),
        "horaire_depart_prevu":   row.get("horaire_depart_prevu", ""),
        "horaire_arrivee_estime": row.get("horaire_arrivee_estime", ""),
        "horaire_depart_estime":  row.get("horaire_depart_estime", ""),
        "arrivee_prevue_hhmm":    row.get("arrivee_prevue_hhmm", ""),
        "depart_prevu_hhmm":      row.get("depart_prevu_hhmm", ""),
        "depart_estime_hhmm":     row.get("depart_estime_hhmm", ""),
        "jour_semaine":           row.get("jour_semaine", ""),
        "heure_tranche":          row.get("heure_tranche", ""),
        "periode_journee":        row.get("periode_journee", ""),
        "alerte_active":          False,
        "categorie_alerte":       "aucune",
        "date_capture":           row.get("date_capture", ""),   # conservée
    }


def migrer(source: str, destination: str) -> None:
    """Lit source ligne par ligne et écrit destination au schéma courant."""
    n_courant = n_migre = n_ignore = 0

    with (
        open(source,      "r", encoding="utf-8-sig", errors="replace") as fin,
        open(destination, "w", encoding="utf-8-sig", newline="")        as fout,
    ):
        reader = csv.reader(fin)
        writer = csv.DictWriter(
            fout, fieldnames=CSV_COLONNES,
            extrasaction="ignore", lineterminator="\n",
        )
        writer.writeheader()

        file_header = next(reader, None)
        if file_header is None:
            log.error("Fichier source vide.")
            return

        if file_header == list(CSV_COLONNES):
            log.info("Le fichier source est déjà au schéma courant. Copie directe.")
            for vals in reader:
                writer.writerow(dict(zip(CSV_COLONNES, vals)))
                n_courant += 1
        else:
            log.info(
                f"Schéma source : {len(file_header)} cols  "
                f"→ migration vers {_N_NEW} cols"
            )
            for i, vals in enumerate(reader, start=2):
                n = len(vals)
                if n == _N_NEW:
                    # Ligne schéma courant (19 cols) : conservation directe
                    writer.writerow(dict(zip(CSV_COLONNES, vals)))
                    n_courant += 1
                elif n == 17 and vals and vals[0].startswith("STIF:"):
                    # Schéma précoce 17 cols (line_ref en tête, date_capture en pos 16)
                    writer.writerow(_mapper_schema_precoce_17(vals))
                    n_migre += 1
                elif n <= _N_OLD_18:
                    # Schéma intermédiaire 18 cols (nom_ligne en tête) ou fragment
                    writer.writerow(_mapper_schema_18(vals))
                    n_migre += 1
                else:
                    log.warning(f"Ligne {i} ignorée : {n} colonnes (inattendu)")
                    n_ignore += 1

                if (n_courant + n_migre) % 500_000 == 0 and (n_courant + n_migre) > 0:
                    log.info(f"  … {n_courant + n_migre:,} lignes traitées")

    log.info(
        f"Migration terminée :\n"
        f"  {n_courant:,} lignes schéma courant conservées\n"
        f"  {n_migre:,}   lignes anciennes converties\n"
        f"  {n_ignore}    lignes ignorées\n"
        f"  Total → {n_courant + n_migre:,} lignes dans {destination}"
    )


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Migration CSV PRIM : ancien schéma → schéma courant")
    parser.add_argument(
        "--source", default="dataset_predictions/passages_tglobal.csv",
        metavar="PATH", help="CSV source (ancien schéma)",
    )
    parser.add_argument(
        "--output", default="dataset_predictions/passages_tglobal_migre.csv",
        metavar="PATH", help="CSV de sortie (schéma courant)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.source):
        log.error(f"Fichier source introuvable : {args.source}")
        sys.exit(1)

    if os.path.abspath(args.source) == os.path.abspath(args.output):
        log.error("source et output ne peuvent pas être le même fichier.")
        sys.exit(1)

    log.info(f"Source  : {args.source}")
    log.info(f"Sortie  : {args.output}")
    migrer(args.source, args.output)


if __name__ == "__main__":
    main()
