"""
datacreation.py — Construction du dataset de prédiction des retards
Fichiers source :
  data-rf-2024/*NB_FER*       → validations brutes 2024
  data-rf-2024/*PROFIL_FER*   → profils horaires % 2024
  passages_global.csv         → collecte PRIM temps réel
Sortie :
  output/occupation_horaire.csv   → volume horaire par arrêt
  output/dataset_prediction.csv   → dataset ML final (cible = retard_depart_sec)
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("#### Nouvelle exécution ####")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data-rf-2024"
PRIM_FILE  = BASE_DIR / "passages_global.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
SEP = ";"

# ─────────────────────────────────────────────
# ÉTAPE 1 — CHARGEMENT DES FICHIERS IDFM
# ─────────────────────────────────────────────

def charger_fichiers(pattern: str) -> pd.DataFrame:
    fichiers = sorted(DATA_DIR.glob(pattern))
    if not fichiers:
        raise FileNotFoundError(
            f"Aucun fichier '{pattern}' dans {DATA_DIR}\n"
            f"Fichiers présents : {[f.name for f in DATA_DIR.iterdir()]}"
        )
    dfs = []
    for f in fichiers:
        charge = False
        # Essaie tab puis point-virgule pour les .txt (certains fichiers IDFM
        # utilisent le point-virgule même avec l'extension .txt)
        separateurs = ["\t", ";"] if f.suffix.lower() == ".txt" else [SEP]
        for sep in separateurs:
            if charge:
                break
            for enc in ("utf-8", "latin-1", "cp1252"):
                try:
                    df = pd.read_csv(f, sep=sep, encoding=enc, low_memory=False)
                    df.columns = (
                        df.columns.str.strip()
                        .str.replace("\ufeff", "", regex=False)
                        .str.upper()
                    )
                    if len(df.columns) <= 1:
                        # Mauvais séparateur → essaie le suivant
                        raise ValueError("1 colonne détectée")
                    df["PERIODE"] = f.stem.split("_")[1]
                    dfs.append(df)
                    print(f"  [OK] {f.name} — {len(df):,} lignes (sep={repr(sep)}, {enc})")
                    charge = True
                    break
                except UnicodeDecodeError:
                    continue
                except ValueError:
                    break   # mauvais séparateur, passe au suivant
        if not charge:
            print(f"  [!!] Impossible de charger {f.name} — fichier ignoré")

    if not dfs:
        raise RuntimeError(
            f"Aucun fichier chargé pour le pattern '{pattern}'.\n"
            f"Fichiers trouvés : {[f.name for f in fichiers]}\n"
            f"Vérifie le séparateur et l'encodage."
        )

    result = pd.concat(dfs, ignore_index=True)
    result.columns = result.columns.str.strip()
    return result


def nettoyer_nb_vald(series: pd.Series) -> pd.Series:
    return (
        series.astype(str).str.strip()
        .str.replace(" ", "", regex=False)
        .replace({
            "Moinside5": "5", "Moinsde5": "5",
            "moins de 5": "5", "Moins de 5": "5",
            "ND": np.nan, "nd": np.nan, "N/A": np.nan, "nan": np.nan, "": np.nan,
        })
        .pipe(pd.to_numeric, errors="coerce")
    )


print("\n[1/4] Chargement des profils horaires...")
df_profils = charger_fichiers("*PROFIL_FER*")
print(f"  Total : {len(df_profils):,} lignes | Colonnes : {df_profils.columns.tolist()}")

print("\n[2/4] Chargement des validations (NB)...")
df_nb = charger_fichiers("*NB_FER*")
df_nb["NB_VALD"] = nettoyer_nb_vald(df_nb["NB_VALD"])
print(f"  Total : {len(df_nb):,} lignes | Colonnes : {df_nb.columns.tolist()}")

# ─────────────────────────────────────────────
# ÉTAPE 2 — CALCUL VOLUME HORAIRE
# ─────────────────────────────────────────────
print("\n[3/4] Calcul de l'occupation horaire...")

COL_ARRET   = "CODE_STIF_ARRET"
COL_LIBELLE = "LIBELLE_ARRET"
COL_CATJOUR = "CAT_JOUR"
COL_TRANCHE = "TRNC_HORR_60"
COL_POURC   = "POURC_VALIDATIONS"
COL_NBVALD  = "NB_VALD"

# Nettoyage CODE_STIF_ARRET AVANT tout groupby/merge
# (les valeurs 'ND' deviennent NaN de type float64 dans les deux DataFrames)
df_profils[COL_ARRET] = pd.to_numeric(df_profils[COL_ARRET], errors="coerce")
df_nb[COL_ARRET]      = pd.to_numeric(df_nb[COL_ARRET],      errors="coerce")

# Moyenne journalière par arrêt (toutes catégories de titre confondues)
nb_moyen = (
    df_nb.groupby([COL_ARRET, COL_LIBELLE, "JOUR"])[COL_NBVALD]
    .sum().reset_index()
    .groupby([COL_ARRET, COL_LIBELLE])[COL_NBVALD]
    .mean().reset_index()
    .rename(columns={COL_NBVALD: "NB_VALD_JOUR_MOY"})
)
print(f"  {len(nb_moyen):,} arrêts uniques")

# Jointure profils × nb_moyen
df_occ = df_profils.merge(nb_moyen, on=[COL_ARRET, COL_LIBELLE], how="left")

# Unification des colonnes % (S1 vs T3/T4)
if "POURCENTAGE_VALIDATIONS" in df_occ.columns:
    df_occ[COL_POURC] = df_occ[COL_POURC].fillna(df_occ["POURCENTAGE_VALIDATIONS"])

df_occ[COL_POURC] = (
    df_occ[COL_POURC].astype(str)
    .str.replace(",", ".", regex=False)
    .pipe(pd.to_numeric, errors="coerce")
)

df_occ["NB_ENTREES_HEURE"] = (
    pd.to_numeric(
        (df_occ[COL_POURC] / 100) * df_occ["NB_VALD_JOUR_MOY"],
        errors="coerce"
    ).round(0).astype("Int64")
)

# Tranche horaire → entier pour la jointure ultérieure
# TRNC_HORR_60 peut être "8", "08", "8H", "8h00", etc.
df_occ["HEURE"] = (
    df_occ[COL_TRANCHE].astype(str)
    .str.extract(r"(\d+)", expand=False)
    .pipe(pd.to_numeric, errors="coerce")
    .astype("Int64")
)

# Catégorisation de l'occupation
q33 = df_occ["NB_ENTREES_HEURE"].quantile(0.33)
q66 = df_occ["NB_ENTREES_HEURE"].quantile(0.66)

def categoriser(n):
    if pd.isna(n): return "Inconnu"
    if n < q33:    return "Faible"
    if n < q66:    return "Moyen"
    return "Élevé"

df_occ["CATEGORIE_OCCUPATION"] = df_occ["NB_ENTREES_HEURE"].apply(categoriser)

out_occ = OUTPUT_DIR / "occupation_horaire.csv"
df_occ.to_csv(out_occ, index=False, sep=SEP, encoding="utf-8-sig")
print(f"  [OK] → {out_occ} ({len(df_occ):,} lignes)")

# ─────────────────────────────────────────────
# ÉTAPE 3 — CONSTRUCTION DU DATASET ML
# ─────────────────────────────────────────────
print("\n[4/4] Construction du dataset de prédiction...")

if not PRIM_FILE.exists():
    print(f"  [!] {PRIM_FILE} introuvable — lance d'abord prim_global_csv.py")
else:
    df_prim = pd.read_csv(PRIM_FILE, low_memory=False)
    print(f"  PRIM chargé : {len(df_prim):,} lignes")

    # ── Cible : retard à l'ARRIVÉE ───────────────────────────────────────────
    # L'utilisateur veut savoir QUAND le train arrive à sa station,
    # pas quand il repart. On calcule le retard arrivée depuis les deux
    # horodatages PRIM : horaire_arrivee_prevu et horaire_arrivee_estime.
    _arr_prevu  = pd.to_datetime(df_prim["horaire_arrivee_prevu"],  errors="coerce", utc=True)
    _arr_estime = pd.to_datetime(df_prim["horaire_arrivee_estime"], errors="coerce", utc=True)
    df_prim["retard_arrivee_sec"] = (_arr_estime - _arr_prevu).dt.total_seconds()

    # Filtrage outliers (train > 1h de retard ou > 10min en avance → erreur capteur)
    df_prim = df_prim[
        df_prim["retard_arrivee_sec"].isna() |
        df_prim["retard_arrivee_sec"].between(-600, 3600)
    ].copy()

    # ── Feature engineering ──────────────────────────────────────────────────

    # Date réelle de la course
    df_prim["date_course"] = pd.to_datetime(df_prim["date_course"], errors="coerce")

    # Heure théorique d'ARRIVÉE (c'est l'heure que voit le voyageur sur le quai)
    _arr_prevu_local = _arr_prevu.dt.tz_convert("Europe/Paris")
    df_prim["heure_arrivee_prevue"] = _arr_prevu_local.dt.strftime("%H:%M")  # lisible
    df_prim["heure"]    = _arr_prevu_local.dt.hour
    df_prim["minute"]   = _arr_prevu_local.dt.minute
    df_prim["jour_semaine"] = df_prim["date_course"].dt.dayofweek  # 0=Lundi … 6=Dim
    df_prim["mois"]         = df_prim["date_course"].dt.month

    # Catégorie de jour : semaine → JOHV, weekend → SAVS
    df_prim["cat_jour"] = np.where(df_prim["jour_semaine"] < 5, "JOHV", "SAVS")

    # Heure d'arrivée estimée finale (ce que le modèle permettra de calculer)
    # = heure théorique + retard prédit
    # On la stocke déjà dans le dataset pour pouvoir valider le modèle
    df_prim["heure_arrivee_estimee"] = (
        _arr_prevu_local + pd.to_timedelta(
            df_prim["retard_arrivee_sec"].fillna(0), unit="s"
        )
    ).dt.strftime("%H:%M")

    # Statut du retard (cible catégorielle pour classification)
    def statut(s):
        if pd.isna(s):  return "Inconnu"
        if s < -60:     return "En avance"
        if s <= 60:     return "A l'heure"
        if s <= 180:    return "Léger retard"
        if s <= 360:    return "Retard"
        return "Retard important"

    df_prim["statut_retard"] = df_prim["retard_arrivee_sec"].apply(statut)

    # CODE_STIF_ARRET depuis stop_ref ("STIF:StopPoint:Q:22084:" → "22084")
    df_prim["code_arret"] = (
        df_prim["stop_ref"].astype(str)
        .str.extract(r":Q:(\d+):", expand=False)
        .pipe(pd.to_numeric, errors="coerce")
        .astype("Int64")
    )

    # ── Jointure occupation ──────────────────────────────────────────────────
    # On fait la moyenne occupation par (arret, heure, cat_jour) pour la jointure
    occ_join = (
        df_occ.groupby([COL_ARRET, "HEURE", COL_CATJOUR])[["NB_ENTREES_HEURE"]]
        .mean()
        .reset_index()
        .rename(columns={
            COL_ARRET:  "code_arret",
            "HEURE":    "heure",
            COL_CATJOUR:"cat_jour",
        })
    )
    occ_join["code_arret"] = pd.to_numeric(occ_join["code_arret"], errors="coerce").astype("Int64")

    df_ml = df_prim.merge(
        occ_join,
        on=["code_arret", "heure", "cat_jour"],
        how="left",
    )

    # Catégorisation de l'occupation sur le dataset final
    df_ml["CATEGORIE_OCCUPATION"] = df_ml["NB_ENTREES_HEURE"].apply(categoriser)

    # ── Estimation de l'occupation de la rame ────────────────────────────────
    # Capacité maximale par type de ligne (passagers debout + assis)
    CAPACITE_RAME = {
        "Métro":      700,   # MF01/MF67/MP05 (5 voitures)
        "RER":       2500,   # MI09/Z2N (8-10 voitures)
        "Transilien": 1000,
        "Tramway":    300,
        "Bus":        100,
        "Noctilien":  100,
    }

    def get_capacite(nom_ligne: str) -> int:
        for k, v in CAPACITE_RAME.items():
            if k in str(nom_ligne):
                return v
        return 700  # défaut

    df_ml["capacite_rame"] = df_ml["nom_ligne"].apply(get_capacite)

    # Modèle d'occupation cumulative par voyage :
    # Pour chaque trip (vehicle_journey_ref + date_course), trié par ordre_arret :
    #   entrees_cumulees[i] = somme des NB_ENTREES_HEURE aux arrêts 0..i
    #   position_relative[i] = rang_i / n_stops   (entre 0 et 1)
    #   passagers_rame[i] = entrees_cumulees[i] × (1 − position_relative[i])
    #     → les passagers sortent progressivement (modèle de sortie uniforme)
    #     → au terminus (position=1) : tout le monde est sorti → 0 passager
    if "ordre_arret" in df_ml.columns and "vehicle_journey_ref" in df_ml.columns:
        df_ml["ordre_arret"] = pd.to_numeric(df_ml["ordre_arret"], errors="coerce")
        df_ml = df_ml.sort_values(
            ["vehicle_journey_ref", "date_course", "ordre_arret"]
        ).reset_index(drop=True)

        grp = df_ml.groupby(["vehicle_journey_ref", "date_course"], sort=False)

        # Somme cumulative des entrées (NaN traité comme 0 pour ne pas casser la somme)
        df_ml["entrees_cumulees"] = (
            grp["NB_ENTREES_HEURE"]
            .transform(lambda x: x.fillna(0).cumsum())
            .astype("Int64")
        )

        # Rang de l'arrêt dans le voyage (1-based) et nombre total d'arrêts
        df_ml["rang_stop"]    = grp.cumcount() + 1
        df_ml["n_stops_trip"] = grp["ordre_arret"].transform("count")

        # Position relative dans le voyage [0 → départ, 1 → terminus]
        df_ml["position_relative"] = (
            df_ml["rang_stop"] / df_ml["n_stops_trip"]
        ).clip(0, 1).round(3)

        # Passagers estimés dans la rame à cet arrêt
        df_ml["passagers_rame"] = (
            df_ml["entrees_cumulees"] * (1 - df_ml["position_relative"])
        ).round(0).astype("Int64")

        # Taux d'occupation [0 → vide, 1 → plein, >1 → surcharge]
        df_ml["taux_occupation"] = (
            df_ml["passagers_rame"] / df_ml["capacite_rame"]
        ).clip(0, 2).round(3)

        print(f"\n  Taux d'occupation moyen : {df_ml['taux_occupation'].mean():.2f}")
        print(f"  Passagers rame médian   : {df_ml['passagers_rame'].median():.0f}")
    else:
        # Fallback si ordre_arret absent : proxy direct par NB_ENTREES_HEURE
        df_ml["passagers_rame"]   = df_ml["NB_ENTREES_HEURE"]
        df_ml["taux_occupation"]  = (
            df_ml["NB_ENTREES_HEURE"] / df_ml["capacite_rame"]
        ).clip(0, 2).round(3)
        df_ml["position_relative"] = np.nan
        print("  [!] ordre_arret absent — taux_occupation calculé depuis NB_ENTREES_HEURE")

    # ── Sélection des colonnes finales ───────────────────────────────────────
    colonnes_finales = [
        # Identifiants
        "date_course", "nom_ligne", "nom_arret", "stop_ref", "code_arret",
        "terminus", "direction_ref",
        # Features temporelles
        "heure", "minute", "jour_semaine", "mois", "cat_jour",
        # Feature occupation arrêt (entrées à CET arrêt)
        "NB_ENTREES_HEURE", "CATEGORIE_OCCUPATION",
        # Feature occupation rame (passagers DANS le train)
        "passagers_rame", "taux_occupation", "position_relative", "capacite_rame",
        # Cible principale : retard à l'arrivée (secondes) → régression
        "retard_arrivee_sec",
        # Cible catégorielle → classification (A l'heure / Léger retard / etc.)
        "statut_retard",
        # Heure d'arrivée théorique et estimée (pour valider / afficher le résultat)
        "heure_arrivee_prevue", "heure_arrivee_estimee",
    ]
    # Garder uniquement les colonnes qui existent réellement
    colonnes_finales = [c for c in colonnes_finales if c in df_ml.columns]
    df_ml = df_ml[colonnes_finales].copy()

    # ── Stats finales ────────────────────────────────────────────────────────
    total      = len(df_ml)
    avec_cible = df_ml["retard_arrivee_sec"].notna().sum()
    avec_occ   = df_ml["NB_ENTREES_HEURE"].notna().sum()

    print(f"\n  Lignes totales              : {total:,}")
    print(f"  Avec retard arrivée (cible) : {avec_cible:,}  ({100*avec_cible/total:.1f}%)")
    print(f"  Avec occupation (feature)   : {avec_occ:,}  ({100*avec_occ/total:.1f}%)")
    print(f"\n  Répartition des statuts :")
    print(df_ml["statut_retard"].value_counts().to_string())
    print(f"\n  Retard arrivée moyen  : {df_ml['retard_arrivee_sec'].mean():.1f}s")
    print(f"  Retard arrivée médian : {df_ml['retard_arrivee_sec'].median():.1f}s")
    print(f"\n  Aperçu du dataset :")
    print(df_ml.head(3).to_string())

    out_ml = OUTPUT_DIR / "dataset_prediction.csv"
    df_ml.to_csv(out_ml, index=False, sep=SEP, encoding="utf-8-sig")
    print(f"\n  [OK] Dataset ML → {out_ml} ({len(df_ml):,} lignes, {len(df_ml.columns)} colonnes)")
    print(f"  Colonnes : {df_ml.columns.tolist()}")

print("\n[DONE] Fichiers dans output/")
