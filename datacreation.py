"""
datacreation.py — Construction du dataset complet
Fichiers source (locaux) :
  data-rf-2024/
    2024_S1_NB_FER.txt      → validations brutes   Jan-Jun
    2024_S1_PROFIL_FER.txt  → profils horaires %   Jan-Jun
    2024_T3_NB_FER.txt      → validations brutes   Jul-Sep
    2024_T3_PROFIL_FER.txt  → profils horaires %   Jul-Sep
    2024_T4_NB_FER.txt      → validations brutes   Oct-Dec
    2024_T4_PROFIL_FER.txt  → profils horaires %   Oct-Dec
  stop_times.txt            → GTFS horaires théoriques (métro RATP)
  passages_global.csv       → collecte PRIM temps réel (prim_global_csv.py)
"""

import pandas as pd
import os
import matplotlib.pyplot as ply
import numpy as np
from pathlib import Path
print("####Nouvelle execution ####")
# ─────────────────────────────────────────────
# CONFIGURATION CHEMINS
# ─────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent          # dossier du script
DATA_DIR   = BASE_DIR / "data-rf-2024"
GTFS_FILE  = BASE_DIR / "stop_times.txt"
PRIM_FILE  = BASE_DIR / "passages_global.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Séparateur des fichiers IDFM (point-virgule)
SEP = ";"

# ─────────────────────────────────────────────
# ÉTAPE 1 — CHARGEMENT ET FUSION DES FICHIERS
# ─────────────────────────────────────────────

def charger_fichiers(pattern: str) -> pd.DataFrame:
    """
    Charge et concatène tous les fichiers correspondant au pattern.
    Ajoute une colonne 'periode' extraite du nom de fichier (S1, T3, T4).
    Détecte automatiquement l'encodage (utf-8 ou latin-1).
    """
    fichiers = sorted(DATA_DIR.glob(pattern))
    if not fichiers:
        raise FileNotFoundError(
            f"Aucun fichier '{pattern}' trouvé dans {DATA_DIR}\n"
            f"Fichiers présents : {[f.name for f in DATA_DIR.iterdir()]}"
        )

    dfs = []
    for f in fichiers:
        # Détection du séparateur : .txt → tabulation, .csv → point-virgule
        sep = "\t" if f.suffix.lower() == ".txt" else SEP

        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                df = pd.read_csv(f, sep=sep, encoding=enc, low_memory=False)

                # Nettoyage noms de colonnes : espaces, BOM, casse
                df.columns = (
                    df.columns
                      .str.strip()
                      .str.replace("\ufeff", "", regex=False)
                      .str.upper()
                )

                # Vérification que le fichier a bien été parsé (> 1 colonne)
                if len(df.columns) == 1:
                    raise ValueError(
                        f"Séparateur incorrect — 1 seule colonne détectée. "
                        f"Colonne : {df.columns[0][:60]}"
                    )

                df["PERIODE"] = f.stem.split("_")[1]
                df["FICHIER"] = f.name
                dfs.append(df)
                print(f"  [OK] {f.name} — {len(df):,} lignes ({enc}, sep={repr(sep)})")
                print(f"       Colonnes : {df.columns.tolist()}")
                break
            except UnicodeDecodeError:
                continue
            except ValueError as e:
                print(f"  [!] {f.name} ({enc}) : {e}")
                break

    result = pd.concat(dfs, ignore_index=True)
    result.columns = result.columns.str.strip()
    print('Les columns : ',result.columns)    
    return result


def nettoyer_nb_vald(series: pd.Series) -> pd.Series:
    """
    Gère la valeur 'Moins de 5' (fichiers < 2022) et les chaînes avec espaces.
    Remplace par 5 (convention IDFM) puis convertit en numérique.
    """
    return (
        series.astype(str)
              .str.strip()
              .str.replace(" ", "", regex=False)   # "1 820" → "1820"
              .replace({"Moinside5": "5", "Moinsde5": "5",
                        "moins de 5": "5", "Moins de 5": "5"})
              .pipe(pd.to_numeric, errors="coerce")
    )


# ── Chargement profils (%) ──
print("\n[1/4] Chargement des profils horaires...")
df_profils = charger_fichiers("*PROFIL_FER*")
print(f"  Total profils : {len(df_profils):,} lignes")
print(f"  Colonnes      : {df_profils.columns.tolist()}")
print(df_profils)

# ── Chargement NB validations ──
print("\n[2/4] Chargement des validations (NB)...")
df_nb = charger_fichiers("*NB_FER*")
df_nb["NB_VALD"] = nettoyer_nb_vald(df_nb["NB_VALD"])
print(f"  Total NB      : {len(df_nb):,} lignes")
print(f"  Colonnes      : {df_nb.columns.tolist()}")

# ─────────────────────────────────────────────
# ÉTAPE 2 — CALCUL DU VOLUME HORAIRE ABSOLU
# ─────────────────────────────────────────────
print("\n[3/4] Calcul du volume horaire absolu...")

# Colonnes clés (noms IDFM standards — ajuster si différent après affichage)
COL_ARRET   = "CODE_STIF_ARRET"
COL_LIBELLE = "LIBELLE_ARRET"
COL_RESEAU  = "CODE_STIF_RES"
COL_TRANS   = "CODE_STIF_TRNS"
COL_CATJOUR = "CAT_JOUR"
COL_TRANCHE = "TRNC_HORR_60"
COL_POURC   = "POURC_VALIDATIONS"
COL_NBVALD  = "NB_VALD"

# Vérification des colonnes critiques
for col in [COL_ARRET, COL_LIBELLE, COL_CATJOUR, COL_TRANCHE, COL_POURC]:
    if col not in df_profils.columns:
        print(f"  [!] Colonne manquante dans profils : '{col}'")
        print(f"      Colonnes disponibles : {df_profils.columns.tolist()}")

for col in [COL_ARRET, COL_LIBELLE, COL_NBVALD]:
    if col not in df_nb.columns:
        print(f"  [!] Colonne manquante dans NB : '{col}'")
        print(f"      Colonnes disponibles : {df_nb.columns.tolist()}")

# Structure réelle du fichier NB_FER :
# JOUR | CODE_STIF_TRNS | CODE_STIF_RES | CODE_STIF_ARRET | LIBELLE_ARRET
# ID_ZDC | CATEGORIE_TITRE | NB_VALD
#
# → Pas de CAT_JOUR dans NB → on aggrège toutes les catégories de titres
# → JOUR est une date précise → on calcule la moyenne journalière par arrêt
#   puis on somme sur toutes les CATEGORIE_TITRE pour avoir le total/jour

# Structure NB_FER : 1 ligne par (JOUR × ARRET × CATEGORIE_TITRE)
# Ex: même arrêt le même jour → N lignes (Navigo, Ticket+, Imagine R, etc.)
#
# Étape A : somme par (arrêt, jour, catégorie de titre)
# → total validations par catégorie de titre par jour
nb_par_jour = (
    df_nb.groupby([COL_ARRET, COL_LIBELLE, "JOUR", "CATEGORIE_TITRE"])[COL_NBVALD]
         .sum()
         .reset_index()
)

# Étape B : somme de toutes les catégories de titre pour avoir le total/arrêt/jour
nb_par_jour_total = (
    nb_par_jour.groupby([COL_ARRET, COL_LIBELLE, "JOUR"])[COL_NBVALD]
               .sum()
               .reset_index()
)

# Étape C : moyenne journalière sur toute l'année
# On considère que la moyenne 2024 est représentative des autres années
nb_moyen = (
    nb_par_jour_total.groupby([COL_ARRET, COL_LIBELLE])[COL_NBVALD]
                     .mean()
                     .reset_index()
                     .rename(columns={COL_NBVALD: "NB_VALD_JOUR_MOY"})
)

print(f"  nb_moyen : {len(nb_moyen):,} arrêts uniques")
print(f"  Exemple :\n{nb_moyen.head(5).to_string(index=False)}")

# Jointure profils × nb_moyen sur l'identifiant arrêt
# Les profils ont CAT_JOUR (JOHV/SAVS/...) → on garde toutes les catégories
# Le NB_VALD_JOUR_MOY est identique pour toutes les CAT_JOUR d'un même arrêt
df = df_profils.merge(nb_moyen, on=[COL_ARRET, COL_LIBELLE], how="left")

# Volume horaire = % × total journalier
# ── Unification des noms de colonnes selon la période ──
# S1  → colonne "POURC_VALIDATIONS"       (virgule décimale : "1,85")
# T3  → colonne "POURCENTAGE_VALIDATIONS" (virgule décimale : "1,76")
# T4  → colonne "POURCENTAGE_VALIDATIONS" (point décimal   :  0.92)
# Après concat les deux colonnes coexistent avec des NaN croisés → on les fusionne
if "POURCENTAGE_VALIDATIONS" in df.columns:
    df[COL_POURC] = df[COL_POURC].fillna(df["POURCENTAGE_VALIDATIONS"])

# Conversion : remplace la virgule décimale française par un point avant to_numeric
# (T4 est déjà float, .replace(",", ".") est sans effet)
df[COL_POURC] = (
    df[COL_POURC]
    .astype(str)
    .str.replace(",", ".", regex=False)
    .pipe(pd.to_numeric, errors="coerce")
)
df["NB_ENTREES_HEURE"] = (
    (df[COL_POURC] / 100) * df["NB_VALD_JOUR_MOY"]
).round(0).astype("Int64")

# Catégorisation de l'occupation
q33 = df["NB_ENTREES_HEURE"].quantile(0.33)
q66 = df["NB_ENTREES_HEURE"].quantile(0.66)

def categoriser(n):
    if pd.isna(n): return "Inconnu"
    if n < q33:    return "Faible"
    if n < q66:    return "Moyen"
    return "Élevé"

df["CATEGORIE_OCCUPATION"] = df["NB_ENTREES_HEURE"].apply(categoriser)

# Exemple de résultat
print("\n  Exemple — NATION, jours ouvrés hors vacances (JOHV) :")
mask = (df[COL_LIBELLE].str.upper() == "NATION") & (df[COL_CATJOUR] == "JOHV")
result = (
    df[mask]
    [[COL_TRANCHE, COL_POURC, "NB_ENTREES_HEURE", "CATEGORIE_OCCUPATION"]]
    .sort_values(COL_TRANCHE)
)
print(result.to_string(index=False))

# Sauvegarde
out_occ = OUTPUT_DIR / "occupation_horaire.csv"
df.to_csv(out_occ, index=False, sep=SEP, encoding="utf-8-sig")
print(f"\n  [OK] Sauvegardé → {out_occ} ({len(df):,} lignes)")

# ─────────────────────────────────────────────
# ÉTAPE 3 — JOINTURE GTFS (retard métro RATP)
# ─────────────────────────────────────────────
print("\n[4/4] Jointure GTFS stop_times...")

if not GTFS_FILE.exists():
    print(f"  [!] {GTFS_FILE} introuvable — étape GTFS ignorée")
elif not PRIM_FILE.exists():
    print(f"  [!] {PRIM_FILE} introuvable — lance d'abord prim_global_csv.py")
else:
    # Chargement GTFS stop_times
    # Colonnes : trip_id, arrival_time, departure_time, stop_id, stop_sequence
    gtfs = pd.read_csv(
        GTFS_FILE,
        dtype=str,
        usecols=["trip_id", "stop_id", "departure_time", "arrival_time", "stop_sequence"],
    )
    print(f"  GTFS chargé : {len(gtfs):,} lignes")

    # Chargement données PRIM
    df_prim = pd.read_csv(PRIM_FILE, low_memory=False)
    print(f"  PRIM chargé : {len(df_prim):,} lignes")

    # Nettoyage des identifiants pour la jointure
    # vehicle_journey_ref PRIM  ←→  trip_id GTFS
    # Le trip_id GTFS peut être préfixé différemment selon l'opérateur
    # ex: "RATP:vehicle_journey:XXXX" vs "XXXX" dans PRIM
    # On extrait la partie numérique/finale pour matcher
    def extraire_id(s: str) -> str:
        if pd.isna(s): return ""
        s = str(s)
        # Garde uniquement la dernière partie après ":" ou "/"
        return s.rsplit(":", 1)[-1].rsplit("/", 1)[-1].strip()

    df_prim["vj_clean"]   = df_prim["vehicle_journey_ref"].apply(extraire_id)
    df_prim["stop_clean"] = df_prim["stop_ref"].apply(extraire_id)
    gtfs["trip_clean"]    = gtfs["trip_id"].apply(extraire_id)
    gtfs["stop_clean"]    = gtfs["stop_id"].apply(extraire_id)

    # Jointure : df_prim a "vj_clean", gtfs a "trip_clean" → left_on / right_on
    df_merged = df_prim.merge(
        gtfs[["trip_clean", "stop_clean", "departure_time", "stop_sequence"]]
            .rename(columns={"departure_time": "aimed_dep_gtfs",
                             "stop_sequence":  "ordre_gtfs"}),
        left_on=["vj_clean", "stop_clean"],
        right_on=["trip_clean", "stop_clean"],
        how="left",
    )

    # Calcul du retard exact avec l'Aimed GTFS — version vectorisée
    # GTFS autorise des heures > 23 (ex: "25:30:00" = lendemain 01:30)

    # 1. Décomposer "HH:MM:SS" en colonnes numériques
    # reindex garantit les 3 colonnes même si aimed_dep_gtfs est majoritairement NaN
    gtfs_parts = (
        df_merged["aimed_dep_gtfs"]
        .str.split(":", expand=True)
        .reindex(columns=[0, 1, 2])
    )
    gtfs_h = pd.to_numeric(gtfs_parts[0], errors="coerce")
    gtfs_m = pd.to_numeric(gtfs_parts[1], errors="coerce")
    gtfs_s = pd.to_numeric(gtfs_parts[2], errors="coerce")

    # 2. Date de référence de la course (fallback = aujourd'hui)
    base_date = pd.to_datetime(df_merged["date_course"], errors="coerce")
    base_date = base_date.fillna(pd.Timestamp("today").normalize())

    # 3. Convertir HH:MM:SS en Timedelta (gère h >= 24)
    total_sec = gtfs_h * 3600 + gtfs_m * 60 + gtfs_s
    df_merged["aimed_ts"] = base_date + pd.to_timedelta(total_sec, unit="s")

    # Remettre NaT là où aimed_dep_gtfs était absent
    df_merged.loc[df_merged["aimed_dep_gtfs"].isna(), "aimed_ts"] = pd.NaT

    # 4. Timestamp estimé (PRIM) — supprime le timezone pour homogénéiser
    df_merged["expected_ts"] = pd.to_datetime(
        df_merged["horaire_depart_estime"], errors="coerce", utc=True
    ).dt.tz_convert(None)

    df_merged["retard_sec_gtfs"] = (
        df_merged["expected_ts"] - df_merged["aimed_ts"]
    ).dt.total_seconds()

    df_merged["retard_min_gtfs"] = df_merged["retard_sec_gtfs"].div(60).round(1)

    def statut_gtfs(s):
        if pd.isna(s):  return "N/A (pas de match GTFS)"
        if s < -60:     return "En avance"
        if s <= 60:     return "A l'heure"
        if s <= 180:    return "Léger retard"
        if s <= 360:    return "Retard"
        return "Retard important"

    df_merged["statut_retard_gtfs"] = df_merged["retard_sec_gtfs"].apply(statut_gtfs)

    # Stats de jointure
    total      = len(df_merged)
    matchés    = df_merged["aimed_dep_gtfs"].notna().sum()
    retard_ok  = df_merged["retard_sec_gtfs"].notna().sum()
    print(f"\n  Lignes totales    : {total:,}")
    print(f"  Matchés GTFS      : {matchés:,}  ({100*matchés/total:.1f}%)")
    print(f"  Retard calculable : {retard_ok:,}  ({100*retard_ok/total:.1f}%)")

    if retard_ok > 0:
        print(f"\n  Répartition retards :")
        print(df_merged["statut_retard_gtfs"].value_counts().to_string())

    out_prim = OUTPUT_DIR / "passages_avec_retard.csv"
    df_merged.to_csv(out_prim, index=False, sep=SEP, encoding="utf-8-sig")
    print(f"\n  [OK] Sauvegardé → {out_prim}")

print("\n[DONE] Tous les fichiers sont dans le dossier output/")