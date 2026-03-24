"""
PRIM IDFM — Collecteur global : toutes les lignes → CSV
API : GET /estimated-timetable (SIRI Lite)

Structure :
  1. Traitement de la requête API
  2. Feature engineering 
"""

import requests
import csv
import os
import time
import logging
from datetime import datetime, timezone

import pandas as pd

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

API_KEYS = [
    "bh8enPs1rJpVrXrpxUrXGAWq3f3BbCLu",
    "UxZ2oEXNyvd3ym0zbx2AGgW0w8AYgHSj",
]
_current_key_idx = 0

BASE_URL         = "https://prim.iledefrance-mobilites.fr/marketplace"
CSV_FILE         = "dataset_predictions/passages_tglobal.csv"
INTERVALLE_CYCLE = 120  # secondes entre deux cycles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# RÉFÉRENTIEL DES LIGNES
# ─────────────────────────────────────────────

LIGNES = {
    # Métro
    "Métro 1":  "STIF:Line::C01371:", "Métro 2":  "STIF:Line::C01372:",
    "Métro 3":  "STIF:Line::C01373:", "Métro 3b": "STIF:Line::C01386:",
    "Métro 4":  "STIF:Line::C01374:", "Métro 5":  "STIF:Line::C01375:",
    "Métro 6":  "STIF:Line::C01376:", "Métro 7":  "STIF:Line::C01377:",
    "Métro 7b": "STIF:Line::C01387:", "Métro 8":  "STIF:Line::C01378:",
    "Métro 9":  "STIF:Line::C01379:", "Métro 10": "STIF:Line::C01380:",
    "Métro 11": "STIF:Line::C01381:", "Métro 12": "STIF:Line::C01382:",
    "Métro 13": "STIF:Line::C01383:", "Métro 14": "STIF:Line::C01384:",
    # RER
    "RER A": "STIF:Line::C01742:", "RER B": "STIF:Line::C01743:",
    "RER C": "STIF:Line::C01727:", "RER D": "STIF:Line::C01728:",
    "RER E": "STIF:Line::C01729:",
    # Transilien
    "Ligne H": "STIF:Line::C01737:", "Ligne J": "STIF:Line::C01738:",
    "Ligne K": "STIF:Line::C01739:", "Ligne L": "STIF:Line::C01740:",
    "Ligne N": "STIF:Line::C01741:", "Ligne P": "STIF:Line::C01744:",
    "Ligne R": "STIF:Line::C01745:", "Ligne U": "STIF:Line::C01746:",
    # Tramway
    "Tram T1":  "STIF:Line::C01389:", "Tram T2":  "STIF:Line::C01390:",
    "Tram T3a": "STIF:Line::C01391:", "Tram T3b": "STIF:Line::C01679:",
    "Tram T4":  "STIF:Line::C01392:", "Tram T5":  "STIF:Line::C01775:",
    "Tram T6":  "STIF:Line::C01776:", "Tram T7":  "STIF:Line::C01777:",
    "Tram T8":  "STIF:Line::C01778:", "Tram T9":  "STIF:Line::C02317:",
    "Tram T10": "STIF:Line::C02316:", "Tram T11": "STIF:Line::C02024:",
    "Tram T13": "STIF:Line::C02048:",
}

LIGNES_PAR_REF = {ref: nom for nom, ref in LIGNES.items()}

# ─────────────────────────────────────────────
# COLONNES CSV
# ─────────────────────────────────────────────

CSV_COLONNES = [
    # Identification de la course
    "line_ref",
    "operateur",
    "direction_ref",
    "terminus",
    # Arrêt
    "stop_ref",
    "nom_arret",
    # Horaires
    "horaire_arrivee_prevu",
    "horaire_depart_prevu",
    "horaire_arrivee_estime",
    "horaire_depart_estime",
    "arrivee_prevue_hhmm",
    "depart_prevu_hhmm",
    "depart_estime_hhmm",
    # Enrichissement temporel
    "jour_semaine",
    "heure_tranche",
    "periode_journee",
    # Métadonnée de collecte
    "date_capture",
]


# ══════════════════════════════════════════════════════════════
# PARTIE 1 — TRAITEMENT DE LA REQUÊTE API
# ══════════════════════════════════════════════════════════════

JOURS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]


def _val(field) -> str:
    """Dépaquète les formats SIRI hybrides (str, dict, list)."""
    if field is None:
        return ""
    if isinstance(field, str):
        return field.strip()
    if isinstance(field, dict):
        return str(field.get("value", "")).strip()
    if isinstance(field, list):
        parts = [
            str(item.get("value", "")).strip() if isinstance(item, dict) else str(item).strip()
            for item in field
        ]
        return " / ".join(p for p in parts if p)
    return str(field).strip()


def fmt_hhmm(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime("%H:%M")
    except Exception:
        return ""


def _periode(h: int) -> str:
    if h < 6:   return "Nuit"
    if h < 9:   return "Pointe matin"
    if h < 11:  return "Creuse matin"
    if h < 14:  return "Méridienne"
    if h < 16:  return "Creuse après-midi"
    if h < 20:  return "Pointe soir"
    return "Soirée"


def _enrichissement_temporel(heure_ref: str) -> tuple:
    """Retourne (heure_tranche, periode_journee) depuis une chaîne ISO."""
    if not heure_ref:
        return "", ""
    try:
        h = datetime.fromisoformat(heure_ref.replace("Z", "+00:00")).hour
        return h, _periode(h)
    except Exception:
        return "", ""


def _parse_calls(journey: dict) -> list:
    """Extrait tous les appels (RecordedCall + EstimatedCall) d'une course."""
    def normalise(bloc, key):
        data = bloc.get(key, [])
        if isinstance(data, dict):
            data = [data]
        return data or []

    recorded  = normalise(journey.get("RecordedCalls", {}), "RecordedCall")
    estimated = normalise(journey.get("EstimatedCalls", {}), "EstimatedCall")
    return [(c, True) for c in recorded] + [(c, False) for c in estimated]


def _parse_journey(journey: dict, now_local: datetime, now_capture: str) -> list[dict]:
    """Transforme une EstimatedVehicleJourney en liste de lignes CSV."""
    line_ref    = _val(journey.get("LineRef"))
    #nom_ligne   = LIGNES_PAR_REF.get(line_ref, line_ref)
    operateur   = _val(journey.get("OperatorRef"))
    direction   = _val(journey.get("DirectionRef"))
    terminus    = _val(journey.get("DestinationName"))

    rows = []
    for call, _ in _parse_calls(journey):
        aimed_arr = _val(call.get("AimedArrivalTime"))
        aimed_dep = _val(call.get("AimedDepartureTime"))
        exp_arr   = _val(call.get("ExpectedArrivalTime"))
        exp_dep   = _val(call.get("ExpectedDepartureTime"))

        heure_ref            = aimed_dep or aimed_arr or exp_dep or exp_arr
        h_tranche, h_periode = _enrichissement_temporel(heure_ref)

        rows.append({
            #"nom_ligne":              nom_ligne,
            "line_ref":               line_ref,
            "operateur":              operateur,
            "direction_ref":          direction,
            "terminus":               terminus,
            "stop_ref":               _val(call.get("StopPointRef")),
            "nom_arret":              _val(call.get("StopPointName")),
            "horaire_arrivee_prevu":  aimed_arr,
            "horaire_depart_prevu":   aimed_dep,
            "horaire_arrivee_estime": exp_arr,
            "horaire_depart_estime":  exp_dep,
            "arrivee_prevue_hhmm":    fmt_hhmm(aimed_arr),
            "depart_prevu_hhmm":      fmt_hhmm(aimed_dep),
            "depart_estime_hhmm":     fmt_hhmm(exp_dep),
            "jour_semaine":           JOURS[now_local.weekday()],
            "heure_tranche":          h_tranche,
            "periode_journee":        h_periode,
            "date_capture":           now_capture,
        })
    return rows


def get_estimated_timetable() -> list[dict]:
    """Appelle /estimated-timetable?LineRef=ALL et retourne toutes les lignes parsées."""
    global _current_key_idx

    url = f"{BASE_URL}/estimated-timetable"
    for _ in range(len(API_KEYS)):
        headers = {"apiKey": API_KEYS[_current_key_idx]}
        resp    = requests.get(url, headers=headers, params={"LineRef": "ALL"}, timeout=15)
        if resp.status_code == 429:
            old              = _current_key_idx
            _current_key_idx = (_current_key_idx + 1) % len(API_KEYS)
            log.warning(f"429 — clé [{old}] épuisée, passage à [{_current_key_idx}]")
            continue
        resp.raise_for_status()
        break
    else:
        raise requests.HTTPError("Toutes les clés API sont en rate-limit (429)")

    data        = resp.json()
    now_local   = datetime.now()
    _now_utc    = datetime.now(timezone.utc)
    now_capture = _now_utc.strftime("%Y-%m-%dT%H:%M:%S.") + f"{_now_utc.microsecond // 1000:03d}Z"
    rows        = []

    try:
        deliveries = data["Siri"]["ServiceDelivery"]["EstimatedTimetableDelivery"]
        if isinstance(deliveries, dict):
            deliveries = [deliveries]

        for delivery in deliveries:
            frames = delivery.get("EstimatedJourneyVersionFrame", [])
            if isinstance(frames, dict):
                frames = [frames]

            for frame in frames:
                journeys = frame.get("EstimatedVehicleJourney", [])
                if isinstance(journeys, dict):
                    journeys = [journeys]

                for journey in journeys:
                    rows.extend(_parse_journey(journey, now_local, now_capture))

    except (KeyError, TypeError) as e:
        log.warning(f"Parsing échoué : {e}")

    return rows


def ecrire_csv(rows: list[dict]):
    nouveau = not os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLONNES, extrasaction="ignore")
        if nouveau:
            writer.writeheader()
        writer.writerows(rows)


# ══════════════════════════════════════════════════════════════
# PARTIE 2 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

# TODO : ajouter ici les features calculées sur les données brutes
# Exemples :
#   - retard (diff entre aimed et expected)
#   - features météo, jours fériés, vacances scolaires…
#   - encodages catégoriels des lignes / opérateurs

# Jours fériés français 2025-2026
_JOURS_FERIES = {
    "2025-01-01","2025-04-21","2025-05-01","2025-05-08","2025-05-29",
    "2025-06-09","2025-07-14","2025-08-15","2025-11-01","2025-11-11","2025-12-25",
    "2026-01-01","2026-04-06","2026-05-01","2026-05-08","2026-05-14",
    "2026-05-25","2026-07-14","2026-08-15","2026-11-01","2026-11-11","2026-12-25",
}

_WMO_CODE = {
    0:"Dégagé", 1:"Peu nuageux", 2:"Partiellement nuageux", 3:"Couvert",
    45:"Brouillard", 48:"Brouillard givrant",
    51:"Bruine légère", 53:"Bruine modérée", 55:"Bruine forte",
    61:"Pluie légère", 63:"Pluie modérée", 65:"Pluie forte",
    71:"Neige légère", 73:"Neige modérée", 75:"Neige forte",
    80:"Averses légères", 81:"Averses modérées", 82:"Averses fortes",
    95:"Orage",
}


def _cat_jour(date_str: str, weekday: int) -> str:
    """Retourne la catégorie jour IDFM selon la date (DIJFP / SAHV / JOHV)."""
    if date_str[:10] in _JOURS_FERIES or weekday == 6:
        return "DIJFP"
    if weekday == 5:
        return "SAHV"
    return "JOHV"


def _fetch_meteo_paris(dates: "list[str]") -> "dict[str, str]":
    """Récupère la météo historique Paris (Open-Meteo, sans clé API)."""
    unique = sorted({d[:10] for d in dates if d and len(d) >= 10})
    if not unique:
        return {}
    try:
        resp = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": 48.8566, "longitude": 2.3522,
                "start_date": unique[0], "end_date": unique[-1],
                "daily": "weathercode", "timezone": "Europe/Paris",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            d: _WMO_CODE.get(c, f"Code {c}")
            for d, c in zip(data["daily"]["time"], data["daily"]["weathercode"])
        }
    except Exception as e:
        log.warning(f"Météo non récupérée : {e}")
        return {}


def build_features(csv_path: str = CSV_FILE) -> pd.DataFrame:
    """Charge le CSV brut et calcule toutes les features."""
    df = pd.read_csv(csv_path, low_memory=False)

    # 1. Nom de la ligne (ref → nom)
    df['nom_ligne'] = df['line_ref'].map(LIGNES_PAR_REF)

    # 2. Retard arrivée en secondes (vectorisé)
    arr_est  = pd.to_datetime(df['horaire_arrivee_estime'], utc=True, errors='coerce')
    arr_prev = pd.to_datetime(df['horaire_arrivee_prevu'],  utc=True, errors='coerce')
    df['retard_sec'] = (arr_est - arr_prev).dt.total_seconds().round().astype('Int64')

    # 3. Mois (depuis l'horaire prévu, ou départ si arrivée absente)
    ref_dt   = pd.to_datetime(
        df['horaire_arrivee_prevu'].fillna(df['horaire_depart_prevu']),
        utc=True, errors='coerce',
    )
    df['mois'] = ref_dt.dt.month

    # 4. Jour férié (booléen)
    date_str = df['date_course'].astype(str)
    df['jour_ferie'] = date_str.str[:10].isin(_JOURS_FERIES)

    # 5. Météo historique Paris (une valeur par date)
    meteo_map = _fetch_meteo_paris(df['date_course'].dropna().astype(str).tolist())
    df['meteo'] = date_str.str[:10].map(meteo_map)

    # 6. Occupation — NB_ENTREES_HEURE (float, entrées/heure)
    #    Jointure : stop_ref → ArRId → ArRName (normalisé) → LIBELLE_ARRET
    #    Les SP (RER/Transilien) ne matcheront pas → NaN → imputation en partie 3
    df_occ = pd.read_csv(
        "dataset_other/occupation_horaire.csv", sep=';', low_memory=False,
        usecols=['LIBELLE_ARRET', 'CAT_JOUR', 'HEURE', 'NB_ENTREES_HEURE'],
    ).dropna(subset=['LIBELLE_ARRET', 'HEURE'])
    df_occ['_nom']  = df_occ['LIBELLE_ARRET'].str.upper().str.strip()
    df_occ['HEURE'] = df_occ['HEURE'].astype(int)

    df_ar = pd.read_csv("dataset_other/arrets .csv", sep=';', low_memory=False,
                        usecols=['ArRId', 'ArRName'])
    df_ar['_ArRId'] = df_ar['ArRId'].astype('Int64')
    df_ar['_nom']   = df_ar['ArRName'].str.upper().str.strip()

    df['_ArRId']   = df['stop_ref'].astype(str).str.extract(r':(?:Q|BP):(\d+):').astype('Int64')
    df['_heure']   = ref_dt.dt.hour.astype(int)
    df['_cat_jour'] = df.apply(
        lambda r: _cat_jour(str(r.get('date_course', '')),
                            pd.Timestamp(r['date_course']).weekday()
                            if pd.notna(r.get('date_course')) else 0),
        axis=1,
    )

    # Étape 1 : ArRId → nom normalisé
    df = df.merge(df_ar[['_ArRId', '_nom']], on='_ArRId', how='left')

    # Étape 2 : (nom, cat_jour, heure) → NB_ENTREES_HEURE
    df = df.merge(
        df_occ[['_nom', 'CAT_JOUR', 'HEURE', 'NB_ENTREES_HEURE']].rename(
            columns={'CAT_JOUR': '_cat_jour', 'HEURE': '_heure'}
        ),
        on=['_nom', '_cat_jour', '_heure'],
        how='left',
    )
    df.rename(columns={'NB_ENTREES_HEURE': 'occupation'}, inplace=True)
    df.drop(columns=['_ArRId', '_nom', '_heure', '_cat_jour'], inplace=True, errors='ignore')

    return df



# ══════════════════════════════════════════════════════════════
# PARTIE 3 — GESTION VALEUR/MANQUANTE
# ══════════════════════════════════════════════════════════════

# Gestion valeur manquante/Nan


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute les valeurs manquantes :
      C — meteo      : 'Inconnu' (date future ou API indisponible)
      C — occupation : moyenne par (nom_ligne, heure_tranche) → SimpleImputer global
      B — retard_sec : IterativeImputer (RandomForest) sur features numériques
    """
    from sklearn.impute import SimpleImputer, IterativeImputer
    from sklearn.ensemble import RandomForestRegressor

    # ── C : meteo ────────────────────────────────────────────
    df['meteo'] = df['meteo'].fillna('Inconnu')

    # ── C : occupation ───────────────────────────────────────
    # 1re passe : moyenne groupée (même ligne, même tranche horaire)
    df['occupation'] = df['occupation'].fillna(
        df.groupby(['nom_ligne', 'heure_tranche'])['occupation'].transform('mean')
    )
    # 2e passe : SimpleImputer global (cas résiduels : ligne inconnue, heure manquante)
    if df['occupation'].isna().any():
        imp_occ = SimpleImputer(strategy='mean')
        df[['occupation']] = imp_occ.fit_transform(df[['occupation']])

    # ── B : retard_sec ───────────────────────────────────────
    # TODO : le métro n'est pas retourné par l'API → retard_sec souvent NaN pour ces lignes.
    # Gerer ce cas la 
    # if df['retard_sec'].isna().any():
    #     X = pd.DataFrame({
    #         'heure_tranche': pd.to_numeric(df['heure_tranche'], errors='coerce'),
    #         'mois':          df['mois'],
    #         'jour_ferie':    df['jour_ferie'].astype(float),
    #         'ligne_code':    df['line_ref'].astype('category').cat.codes.astype(float),
    #         'occupation':    df['occupation'],
    #         'retard_sec':    df['retard_sec'].astype(float),
    #     })
    #     imp_ret = IterativeImputer(
    #         estimator=RandomForestRegressor(n_estimators=20, random_state=42),
    #         max_iter=5,
    #     )
    #     X_imputed = imp_ret.fit_transform(X)
    #     df['retard_sec'] = X_imputed[:, 5].round().astype('Int64')

    return df


# ─────────────────────────────────────────────
# COLLECTE CONTINUE
# ─────────────────────────────────────────────

def collecter_en_continu(duree_heures: float | None = None):
    debut  = datetime.now()
    cycle  = 0
    limite = duree_heures * 3600 if duree_heures else None

    log.info("=" * 55)
    log.info(f"  Collecte continue — {len(LIGNES)} lignes configurées")
    log.info(f"  Fichier CSV : {os.path.abspath(CSV_FILE)}")
    log.info(f"  Intervalle  : {INTERVALLE_CYCLE}s  |  Ctrl+C pour arrêter")
    log.info("=" * 55)

    try:
        while True:
            cycle += 1
            log.info(f"\n[Cycle {cycle}] {datetime.now().strftime('%H:%M:%S')}")
            try:
                rows = get_estimated_timetable()
                ecrire_csv(rows)
                log.info(f"  {len(rows)} lignes écrites")
            except requests.HTTPError as e:
                log.error(f"  HTTP {e.response.status_code} — {e.response.text[:120]}")
            except Exception as e:
                log.error(f"  Erreur : {e}")

            if limite and (datetime.now() - debut).total_seconds() >= limite:
                log.info(f"Durée atteinte ({duree_heures}h). Fin.")
                break

            time.sleep(INTERVALLE_CYCLE)

    except KeyboardInterrupt:
        log.info(f"\nArrêt manuel après {cycle} cycle(s).")
        log.info(f"CSV disponible : {os.path.abspath(CSV_FILE)}")


# ─────────────────────────────────────────────
# PROGRAMME PRINCIPAL (CLI)
# ─────────────────────────────────────────────

def main():
    """
    Usage :
        python build_dataset.py --collecter [--duree 2.0]
        python build_dataset.py --corriger [--csv PATH] [--output PATH]
        python build_dataset.py --collecter --corriger --duree 1.0

    --collecter   Lance la collecte API en continu
    --corriger    Applique feature engineering + imputation des NaN
    --csv         CSV source  (défaut : dataset_predictions/passages_tglobal.csv)
    --output      CSV de sortie (défaut : dataset_predictions/dataset_final.csv)
    --duree       Durée de collecte en heures (défaut : infini)
    """
    import argparse
    parser = argparse.ArgumentParser(description="Construction du dataset PRIM IDFM")
    parser.add_argument("--collecter", action="store_true",
                        help="Lance la collecte API en continu")
    parser.add_argument("--corriger",  action="store_true",
                        help="Applique feature engineering + imputation")
    parser.add_argument("--csv",    default=CSV_FILE,
                        metavar="PATH", help="Fichier CSV source")
    parser.add_argument("--output", default="dataset_predictions/dataset_final.csv",
                        metavar="PATH", help="Fichier CSV de sortie")
    parser.add_argument("--duree",  type=float, default=None,
                        metavar="HEURES", help="Durée de collecte en heures (défaut : infinie)")
    args = parser.parse_args()

    if not args.collecter and not args.corriger:
        parser.print_help()
        return

    if args.collecter:
        msg = (f"Démarrage collecte — durée : {args.duree}h"
               if args.duree else "Démarrage collecte — durée : infinie (Ctrl+C pour arrêter)")
        log.info(msg)
        collecter_en_continu(duree_heures=args.duree)

    if args.corriger:
        log.info(f"Feature engineering sur : {args.csv}")
        df = build_features(args.csv)
        log.info(f"  {len(df):,} lignes chargées")
        log.info("  Imputation des valeurs manquantes...")
        df = impute_missing(df)
        log.info(f"  NaN résiduels : {df.isnull().sum().sum()}")
        df.to_csv(args.output, index=False, encoding="utf-8-sig")
        log.info(f"Dataset final exporté → {args.output}  ({len(df):,} lignes, {len(df.columns)} colonnes)")


if __name__ == "__main__":
    main()
