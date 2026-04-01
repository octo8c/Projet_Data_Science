"""
PRIM IDFM — Construction du dataset pour la prédiction des retards.
API : GET /estimated-timetable (SIRI Lite)

Structure :
  1. Collecte API        → CSV brut  (passages_tglobal.csv) — sans transformation
  2. Feature engineering → calcul retard_sec, météo, jours fériés, occupation…
  3. Imputation          → NaN comblés (météo, occupation, terminus_encoded)
  4. Encodage & ML-ready → catégorielles → entiers fixes (dataset_ml.csv) — avec transformation

Le script produit toujours 2 fichiers distincts :
  • Fichier 1 — sans transformation : dataset_predictions/passages_tglobal.csv
  • Fichier 2 — avec transformation : dataset_predictions/dataset_ml.csv

CLI :
  python build_dataset.py --collecter [--duree 2.0]
      Collecte API continue → fichier 1 (CSV brut, append)

  python build_dataset.py --construire [--csv PATH]
      → Sans --csv : appelle l'API → écrit fichier 1 + fichier 2
      → Avec --csv  : lit le CSV fourni (fichier 1) → écrit fichier 2
"""

import csv
import os
import tempfile
import time
import logging
from datetime import datetime, timezone

import pandas as pd
import requests
from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import train_test_split

import tools as tl

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

API_KEYS = [
    "bh8enPs1rJpVrXrpxUrXGAWq3f3BbCLu",
    "UxZ2oEXNyvd3ym0zbx2AGgW0w8AYgHSj",
]
_current_key_idx = 0

BASE_URL              = "https://prim.iledefrance-mobilites.fr/marketplace"
CSV_FILE              = "dataset_predictions/passages_tglobal.csv"
CSV_FILE_ML           = "dataset_predictions/dataset_ml.csv"
STOP_POINTS_CACHE     = "dataset_other/stop_points.csv"
INTERVALLE_CYCLE      = 120  # secondes entre deux cycles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# MAPPING ÉTENDU DEPUIS GTFS (couvre tous les bus, trams, etc.)
# ─────────────────────────────────────────────
def _build_gtfs_mapping() -> "dict[str, str]":
    _prefixe = {"0": "Tram", "1": "Métro", "2": "Ligne", "3": "Bus", "6": "Câble", "7": "Funiculaire"}
    _gtfs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "IDFM-gtfs(1)", "routes.txt")
    if not os.path.exists(_gtfs):
        return {}
    _routes = pd.read_csv(_gtfs, dtype=str, usecols=["route_id", "route_short_name", "route_type"])
    _out: dict[str, str] = {}
    for _, r in _routes.iterrows():
        _ref = "STIF:Line::" + str(r["route_id"]).replace("IDFM:", "") + ":"
        if _ref not in tl.LIGNES_PAR_REF:  # ne pas écraser les noms explicites
            _p = _prefixe.get(str(r["route_type"]), "Ligne")
            _n = str(r["route_short_name"]).strip()
            _out[_ref] = f"{_p} {_n}" if _n else _p
    return _out

_GTFS_LIGNES = _build_gtfs_mapping()

def resoudre_nom_ligne(line_ref: str) -> str:
    """Nom lisible d'une ligne : LIGNES_PAR_REF en priorité, puis GTFS."""
    return tl.LIGNES_PAR_REF.get(line_ref) or _GTFS_LIGNES.get(line_ref, "")

# ══════════════════════════════════════════════════════════════
# PARTIE 1 — COLLECTE API
# ══════════════════════════════════════════════════════════════

def _parse_journey(
    journey: dict,
    now_local: datetime,
    now_capture: str,
    alertes_map: "dict[str, dict]",
    ref_arrets: "dict[str, str] | None" = None,
) -> list[dict]:
    """Transforme une EstimatedVehicleJourney en liste de lignes CSV."""
    line_ref    = tl._val(journey.get("LineRef"))
    operateur   = tl._val(journey.get("OperatorRef"))
    direction   = tl._val(journey.get("DirectionRef"))
    terminus    = tl._val(journey.get("DestinationName"))

    alerte_info      = alertes_map.get(line_ref, {})
    alerte_active    = alerte_info.get("alerte_active", False)
    categorie_alerte = alerte_info.get("categorie_alerte", "aucune")

    # nom_ligne : LIGNES_PAR_REF en priorité, puis GTFS complet
    nom_ligne = resoudre_nom_ligne(line_ref)

    rows = []
    for call, _ in tl._parse_calls(journey):
        aimed_arr = tl._val(call.get("AimedArrivalTime"))
        aimed_dep = tl._val(call.get("AimedDepartureTime"))
        exp_arr   = tl._val(call.get("ExpectedArrivalTime"))
        exp_dep   = tl._val(call.get("ExpectedDepartureTime"))

        heure_ref            = aimed_dep or aimed_arr or exp_dep or exp_arr
        h_tranche, h_periode = tl._enrichissement_temporel(heure_ref)

        # nom_arret : priorité à StopPointName, fallback sur référentiel (même logique que build_features)
        stop_ref  = tl._val(call.get("StopPointRef"))
        nom_arret = tl._val(call.get("StopPointName"))
        if not nom_arret and ref_arrets:
            import re as _re
            m = _re.search(r":(?:Q|BP):(\d+):", stop_ref)
            if m:
                nom_arret = ref_arrets.get(m.group(1), "")

        rows.append({
            "line_ref":               line_ref,
            "nom_ligne":              nom_ligne,
            "operateur":              operateur,
            "direction_ref":          direction,
            "terminus":               terminus,
            "stop_ref":               stop_ref,
            "nom_arret":              nom_arret,
            "horaire_arrivee_prevu":  aimed_arr,
            "horaire_depart_prevu":   aimed_dep,
            "horaire_arrivee_estime": exp_arr,
            "horaire_depart_estime":  exp_dep,
            "arrivee_prevue_hhmm":    tl.fmt_hhmm(aimed_arr),
            "depart_prevu_hhmm":      tl.fmt_hhmm(aimed_dep),
            "depart_estime_hhmm":     tl.fmt_hhmm(exp_dep),
            "jour_semaine":           tl.JOURS[now_local.weekday()],
            "heure_tranche":          h_tranche,
            "periode_journee":        h_periode,
            "alerte_active":          alerte_active,
            "categorie_alerte":       categorie_alerte,
            "date_capture":           now_capture,
            "retard_sec":             tl._calc_retard_sec(aimed_arr, exp_arr),
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

    alertes_map = _fetch_alertes_prim()
    log.info(f"  Alertes actives : {len(alertes_map)} ligne(s) concernée(s)")

    ref_arrets = _charger_referentiel_arrets()
    log.info(f"  Référentiel arrêts chargé : {len(ref_arrets):,} entrées")

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
                    rows.extend(_parse_journey(journey, now_local, now_capture, alertes_map, ref_arrets))

    except (KeyError, TypeError) as e:
        log.warning(f"Parsing échoué : {e}")

    return rows


def ecrire_csv(rows: list[dict], csv_path: str = CSV_FILE):
    """Écrit les lignes dans le CSV brut (création ou append). Vérifie la compatibilité du schéma."""
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as _f:
            try:
                _existing_header = next(csv.reader(_f))
            except StopIteration:
                _existing_header = []
        if _existing_header and _existing_header != tl.CSV_COLONNES:
            raise ValueError(
                f"Incompatibilité de schéma CSV !\n"
                f"  Header existant ({len(_existing_header)} cols) ≠ schéma courant ({len(tl.CSV_COLONNES)} cols).\n"
                f"  Fichier : {os.path.abspath(csv_path)}\n"
                f"  Action  : renommer ou supprimer le fichier puis relancer la collecte."
            )

    nouveau = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=tl.CSV_COLONNES, extrasaction="ignore",
                                lineterminator="\n")
        if nouveau:
            writer.writeheader()
        writer.writerows(rows)


def _fetch_alertes_prim() -> "dict[str, dict]":
    """
    Appelle GET /marketplace/disruptions_bulk/disruptions/v2 (API PRIM — Disruptions Bulk v2)
    et retourne un dict {line_ref_stif: {alerte_active, categorie_alerte}}.

    Structure de la réponse JSON :
      lines[].id                              → "line:IDFM:C01371"
      lines[].mode                            → "Metro" | "Bus" | "RapidTransit" | "Tramway" | …
      lines[].impactedObjects[type="line"]
             .disruptionIds[]                 → IDs des perturbations actives
      disruptions[].id                         → ID de la perturbation
      disruptions[].cause                      → "TRAVAUX" | "PERTURBATION" | "INFORMATION"
      disruptions[].severity                   → "BLOQUANTE" | "PERTURBEE" | "INFORMATION"
      disruptions[].title                      → texte court (fallback mots-clés)

    Conversion : "line:IDFM:C01371" → "STIF:Line::C01371:"
    Seules les lignes métro/RER/Transilien/tramway sont conservées (pas les bus).
    En cas d'erreur retourne un dict vide (pas de crash de la collecte principale).
    """
    from collections import Counter
    global _current_key_idx

    _MODES_SURVEILLES = {"Metro", "RapidTransit", "Tramway", "LocalTrain", "Train"}
    _CAUSE_MAP    = {"TRAVAUX": "travaux"}
    _SEVERITY_MAP = {"BLOQUANTE": "incident", "PERTURBEE": "retard", "INFORMATION": "autre"}

    url     = f"{BASE_URL}/disruptions_bulk/disruptions/v2"
    headers = {"apiKey": API_KEYS[_current_key_idx], "accept": "application/json"}
    alertes: "dict[str, list[str]]" = {}

    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        disr_by_id = {d["id"]: d for d in data.get("disruptions", [])}

        for line in data.get("lines", []):
            if line.get("mode", "") not in _MODES_SURVEILLES:
                continue
            lid  = line.get("id", "")
            code = lid.split(":")[-1]
            if not code:
                continue
            stif_ref = f"STIF:Line::{code}:"

            disrids: list[str] = []
            for obj in line.get("impactedObjects", []):
                if obj.get("type") == "line":
                    disrids += obj.get("disruptionIds", [])

            for did in disrids:
                dis = disr_by_id.get(did)
                if not dis:
                    continue
                cause    = (dis.get("cause") or "").upper()
                severity = (dis.get("severity") or "").upper()
                cat = (
                    _CAUSE_MAP.get(cause)
                    or _SEVERITY_MAP.get(severity)
                    or _classifier_alerte(dis.get("title", ""))
                )
                alertes.setdefault(stif_ref, []).append(cat)

        return {
            ref: {
                "alerte_active":    True,
                "categorie_alerte": Counter(cats).most_common(1)[0][0],
            }
            for ref, cats in alertes.items()
        }

    except Exception as e:
        log.warning(f"Alertes PRIM non récupérées : {e}")
        return {}


def collecter_en_continu(duree_heures: float | None = None,csv_file = CSV_FILE):
    debut  = datetime.now()
    cycle  = 0
    limite = duree_heures * 3600 if duree_heures else None

    log.info("=" * 55)
    log.info(f"  Collecte continue — {len(tl.LIGNES)} lignes configurées")
    log.info(f"  Fichier CSV : {os.path.abspath(csv_file)}")
    log.info(f"  Intervalle  : {INTERVALLE_CYCLE}s  |  Ctrl+C pour arrêter")
    log.info("=" * 55)

    try:
        while True:
            cycle += 1
            log.info(f"\n[Cycle {cycle}] {datetime.now().strftime('%H:%M:%S')}")
            try:
                rows = get_estimated_timetable()
                ecrire_csv(rows,csv_file)
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
        log.info(f"CSV disponible : {os.path.abspath(csv_file)}")


# ══════════════════════════════════════════════════════════════
# PARTIE 2 — FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════


def _classifier_alerte(texte: str) -> str:
    """Classifie le texte libre d'une alerte en catégorie interprétable."""
    if not texte:
        return "autre"
    t = texte.lower()
    for cat, mots in tl._KEYWORDS_ALERTE.items():
        if any(m in t for m in mots):
            return cat
    return "autre"


def _cat_jour(date_str, weekday: int) -> str:
    """Retourne la catégorie jour IDFM (DIJFP / SAHV / JOHV)."""
    if isinstance(date_str, str) and date_str[:10] in tl._JOURS_FERIES or weekday == 6:
        return "DIJFP"
    if weekday == 5:
        return "SAHV"
    return "JOHV"


def _fetch_meteo_paris_horaire(dates: "list[str]") -> "dict[tuple, dict]":
    """
    Récupère la météo Paris par heure (Open-Meteo, sans clé API).
    Retourne un dict {(date_str '%Y-%m-%d', heure_int): {weathercode, precipitation, ...}}

    Deux APIs selon l'ancienneté des dates :
      - Archive  (> 5 jours) : archive-api.open-meteo.com/v1/archive
      - Forecast (≤ 5 jours) : api.open-meteo.com/v1/forecast  (past_days + forecast_days)
    """
    from datetime import date as _date, timedelta as _td

    unique = sorted({d[:10] for d in dates if d and len(d) >= 10})
    if not unique:
        return {}

    today          = _date.today()
    archive_limite = (today - _td(days=5)).isoformat()
    archive_dates  = [d for d in unique if d <= archive_limite]
    forecast_dates = [d for d in unique if d > archive_limite]

    _PARAMS_BASE = {
        "latitude":  48.8566,
        "longitude": 2.3522,
        "hourly":    "weathercode,precipitation,snowfall,wind_speed_10m,temperature_2m",
        "timezone":  "Europe/Paris",
    }
    result: dict = {}

    def _parser(hourly: dict) -> None:
        for ts, wc, prec, snow, wind, temp in zip(
            hourly["time"], hourly["weathercode"], hourly["precipitation"],
            hourly["snowfall"], hourly["wind_speed_10m"], hourly["temperature_2m"],
        ):
            dt = datetime.fromisoformat(ts)
            result[(dt.strftime("%Y-%m-%d"), dt.hour)] = {
                "weathercode":   wc,
                "precipitation": prec,
                "snowfall":      snow,
                "wind_speed":    wind,
                "temperature":   temp,
            }

    if archive_dates:
        try:
            resp = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params={**_PARAMS_BASE,
                        "start_date": archive_dates[0],
                        "end_date":   archive_dates[-1]},
                timeout=15,
            )
            resp.raise_for_status()
            _parser(resp.json()["hourly"])
        except Exception as e:
            log.warning(f"Météo archive non récupérée : {e}")

    if forecast_dates:
        try:
            past_days = (today - _date.fromisoformat(forecast_dates[0])).days + 1
            resp = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={**_PARAMS_BASE,
                        "past_days":     min(past_days, 92),
                        "forecast_days": 1},
                timeout=15,
            )
            resp.raise_for_status()
            _parser(resp.json()["hourly"])
        except Exception as e:
            log.warning(f"Météo récente non récupérée : {e}")

    return result

def _charger_referentiel_arrets() -> "dict[str, str]":
    """
    Retourne un dict {id_numerique: nom_arret} pour tous les arrêts IDFM.

    Source : GET /v2/navitia/stop_points (36 000+ arrêts, paginé par 1 000)
    Cache  : STOP_POINTS_CACHE (dataset_other/stop_points.csv)
             Régénéré uniquement si le fichier est absent.

    Mapping : "STIF:StopPoint:BP:5886:" → extrait "5886" → clé du dict → "Pissaloup"
              "STIF:StopPoint:Q:22083:" → extrait "22083" → "Gare de Lyon"
    """
    global _current_key_idx

    if os.path.exists(STOP_POINTS_CACHE):
        df_cache = pd.read_csv(STOP_POINTS_CACHE, dtype={"id": str, "nom": str})
        return dict(zip(df_cache["id"], df_cache["nom"]))

    log.info("  Référentiel arrêts absent — chargement depuis l'API PRIM…")
    url     = f"{BASE_URL}/v2/navitia/stop_points"
    headers = {"apiKey": API_KEYS[_current_key_idx], "accept": "application/json"}
    stop_dict: dict[str, str] = {}
    page = 0

    while True:
        resp = requests.get(url, headers=headers,
                            params={"count": 1000, "start_page": page}, timeout=30)
        resp.raise_for_status()
        data  = resp.json()
        stops = data.get("stop_points", [])
        for sp in stops:
            num_id = sp["id"].split(":")[-1]   # "stop_point:IDFM:5886" → "5886"
            stop_dict[num_id] = sp["name"]
        if len(stops) < 1000:
            break
        page += 1
        log.info(f"  Page {page} — {len(stop_dict):,} arrêts chargés…")

    os.makedirs(os.path.dirname(STOP_POINTS_CACHE), exist_ok=True)
    pd.DataFrame(list(stop_dict.items()), columns=["id", "nom"]).to_csv(
        STOP_POINTS_CACHE, index=False, encoding="utf-8-sig"
    )
    log.info(f"  {len(stop_dict):,} arrêts mis en cache → {STOP_POINTS_CACHE}")
    return stop_dict


def build_features(csv_path: str = CSV_FILE) -> pd.DataFrame:
    """
    Charge un CSV brut (schéma ancien ou courant) et calcule toutes les features.
    Retourne un DataFrame avec les NaN encore présents (à imputer ensuite).
    """
    with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as _f:
        _file_header = next(csv.reader(_f))

    if _file_header != tl.CSV_COLONNES:
        raise ValueError(
            f"Schéma CSV incompatible !\n"
            f"  Header fichier ({len(_file_header)} cols) ≠ schéma courant ({len(tl.CSV_COLONNES)} cols).\n"
            f"  Fichier : {os.path.abspath(csv_path)}\n"
            f"  Action  : supprimer ou remplacer le fichier, puis relancer la collecte."
        )

    df = pd.read_csv(csv_path, low_memory=False)

    # 0. nom_ligne : remplir les valeurs vides/absentes via LIGNES_PAR_REF + GTFS
    if "nom_ligne" not in df.columns:
        df["nom_ligne"] = df["line_ref"].map(resoudre_nom_ligne)
    else:
        mask_vide = df["nom_ligne"].isna() | (df["nom_ligne"].astype(str).str.strip() == "")
        df.loc[mask_vide, "nom_ligne"] = df.loc[mask_vide, "line_ref"].map(resoudre_nom_ligne)

    # 1. Retard arrivée en secondes
    arr_est  = pd.to_datetime(df["horaire_arrivee_estime"], utc=True, errors="coerce")
    arr_prev = pd.to_datetime(df["horaire_arrivee_prevu"],  utc=True, errors="coerce")
    df["retard_sec"] = (arr_est - arr_prev).dt.total_seconds().round().astype("Int64")

    # 3. Référence temporelle locale Paris
    ref_dt = pd.to_datetime(
        df["horaire_arrivee_prevu"].fillna(df["horaire_depart_prevu"]),
        utc=True, errors="coerce",
    ).dt.tz_convert("Europe/Paris")
    df["mois"]       = ref_dt.dt.month
    df["_date_str"]  = ref_dt.dt.strftime("%Y-%m-%d")
    df["_heure_int"] = ref_dt.dt.hour

    # 4. Jour férié
    df["jour_ferie"] = df["_date_str"].isin(tl._JOURS_FERIES)

    # 5. Météo horaire Paris
    meteo_map = _fetch_meteo_paris_horaire(df["_date_str"].dropna().tolist())
    df["weathercode"]   = [meteo_map.get((d, h), {}).get("weathercode")   for d, h in zip(df["_date_str"], df["_heure_int"])]
    df["precipitation"] = [meteo_map.get((d, h), {}).get("precipitation") for d, h in zip(df["_date_str"], df["_heure_int"])]
    df["snowfall"]      = [meteo_map.get((d, h), {}).get("snowfall")      for d, h in zip(df["_date_str"], df["_heure_int"])]
    df["wind_speed"]    = [meteo_map.get((d, h), {}).get("wind_speed")    for d, h in zip(df["_date_str"], df["_heure_int"])]
    df["temperature"]   = [meteo_map.get((d, h), {}).get("temperature")   for d, h in zip(df["_date_str"], df["_heure_int"])]
    df["meteo_groupe"]  = df["weathercode"].apply(tl._wmo_groupe)

    # 6. Direction numérique (1 = aller, 2 = retour, 0 = inconnu)
    #    L'API PRIM renvoie du texte ("Aller"/"Retour"/"outbound"/"inbound"/"A"/"R")
    #    → mapping explicite vers des entiers avant la conversion numérique.
    _DIR_MAP = {
        "Aller": 1, "aller": 1, "outbound": 1, "OUTBOUND": 1, "A": 1, "1": 1,
        "Retour": 2, "retour": 2, "inbound": 2,  "INBOUND": 2, "R": 2, "2": 2,
    }
    df["direction_ref"] = (
        df["direction_ref"].astype(str).str.strip().map(_DIR_MAP).fillna(0).astype(int)
    )

    # 7. heure_tranche numérique (peut être string vide dans anciens CSV)
    df["heure_tranche"] = pd.to_numeric(df["heure_tranche"], errors="coerce").fillna(0).astype(int)

    # 8. Target encoding du terminus et de la station (arrêt)
    df = tl._target_encode_terminus(df)
    df = tl._target_encode_station(df)

    # 9. Alertes réseau
    if "alerte_active" not in df.columns:
        df["alerte_active"]    = False
        df["categorie_alerte"] = "aucune"
    else:
        df["alerte_active"]    = df["alerte_active"].fillna(False)
        df["categorie_alerte"] = df["categorie_alerte"].fillna("aucune")

    # 10. Résolution du nom de l'arrêt via le référentiel PRIM (API / cache)
    #     Extrait l'ID numérique de stop_ref (gère :BP: et :Q:)
    #     puis effectue le lookup dans le référentiel PRIM complet (36 000+ arrêts).
    df["_ArRId"] = df["stop_ref"].astype(str).str.extract(r":(?:Q|BP):(\d+):").astype("Int64")
    ref_arrets   = _charger_referentiel_arrets()
    df["nom_arret"] = (
        df["_ArRId"].astype(str)
        .map(ref_arrets)
        .fillna(df["nom_arret"].replace("", pd.NA))   # fallback sur la valeur déjà présente
    )
    taux_rempli = df["nom_arret"].notna().mean()
    log.info(f"  nom_arret rempli : {taux_rempli:.1%} des lignes")

    # 11. Occupation horaire (entrées/heure, jointure datasets IDFM)
    df_occ = pd.read_csv(
        "dataset_other/occupation_horaire.csv", sep=";", low_memory=False,
        usecols=["LIBELLE_ARRET", "CAT_JOUR", "HEURE", "NB_ENTREES_HEURE"],
    ).dropna(subset=["LIBELLE_ARRET", "HEURE"])
    df_occ["_nom"]  = df_occ["LIBELLE_ARRET"].str.upper().str.strip()
    df_occ["HEURE"] = df_occ["HEURE"].astype("Int64")

    df_ar = pd.read_csv("dataset_other/arrets .csv", sep=";", low_memory=False,
                        usecols=["ArRId", "ArRName"])
    df_ar["_ArRId"] = df_ar["ArRId"].astype("Int64")
    df_ar["_nom"]   = df_ar["ArRName"].str.upper().str.strip()

    df["_cat_jour"] = [
        _cat_jour(d, datetime.strptime(d, "%Y-%m-%d").weekday() if isinstance(d, str) and len(d) == 10 else 0)
        for d in df["_date_str"]
    ]

    df = df.merge(df_ar[["_ArRId", "_nom"]], on="_ArRId", how="left")
    df = df.merge(
        df_occ[["_nom", "CAT_JOUR", "HEURE", "NB_ENTREES_HEURE"]].rename(
            columns={"CAT_JOUR": "_cat_jour", "HEURE": "_heure_int"}
        ),
        on=["_nom", "_cat_jour", "_heure_int"],
        how="left",
    )
    df.rename(columns={"NB_ENTREES_HEURE": "occupation"}, inplace=True)
    df.drop(columns=["_ArRId", "_nom", "_date_str", "_heure_int", "_cat_jour"],
            inplace=True, errors="ignore")

    return df


# ══════════════════════════════════════════════════════════════
# PARTIE 3 — IMPUTATION DES VALEURS MANQUANTES
# ══════════════════════════════════════════════════════════════

#Fonction impute_missings dans tools.py

# ══════════════════════════════════════════════════════════════
# PARTIE 4 — ENCODAGE & PRÉPARATION ML
# ══════════════════════════════════════════════════════════════


def encoder_categoriques(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode les colonnes catégorielles avec les mappings fixes de CAT_ENCODINGS.
    Valeurs inconnues/vides → -1.
    Booléens (jour_ferie, alerte_active) → 0 / 1.
    """
    df = df.copy()
    for col, mapping in tl.CAT_ENCODINGS.items():
        if col not in df.columns:
            continue
        serie = df[col].fillna("_inconnu").astype(str).str.strip()
        serie = serie.where(serie != "", other="_inconnu")
        df[col] = serie.map(mapping).fillna(-1).astype(int)

    for col in ("jour_ferie", "alerte_active"):
        if col in df.columns:
            df[col] = (
                df[col]
                .map({True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0})
                .fillna(0)
                .astype(int)
            )
    return df


def preparer_ml(
    csv_path: str = CSV_FILE,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Pipeline complet CSV brut → DataFrame ML-ready.

    Étapes :
      1. build_features()    : calcule retard_sec, météo, occupation, terminus_encoded…
      2. impute_missing()    : comble les NaN
      3. encoder_categoriques() : catégorielles → entiers fixes
      4. Sélection de ML_COLONNES uniquement
      5. Conversion numérique stricte de toutes les colonnes
      6. Sauvegarde si output_path fourni

    Gère les CSV ancienne et nouvelle génération (schéma 18 ou 19 colonnes).
    Les lignes sans retard_sec calculable (horaires manquants) sont conservées
    pour permettre l'inférence ; prediction.py les filtre pour l'entraînement.

    Retourne le DataFrame ML-ready.
    """
    log.info(f"  Feature engineering sur : {csv_path}")
    df = build_features(csv_path)
    log.info(f"  {len(df):,} lignes chargées — imputation NaN…")
    df = tl.impute_missing(df)
    log.info("  Encodage des variables catégorielles…")
    df = encoder_categoriques(df)

    # Sélectionner uniquement les colonnes ML (dans l'ordre défini)
    cols_presentes = [c for c in tl.ML_COLONNES if c in df.columns]
    cols_manquantes = [c for c in tl.ML_COLONNES if c not in df.columns]
    if cols_manquantes:
        log.warning(f"  Colonnes absentes (mises à 0) : {cols_manquantes}")
        for c in cols_manquantes:
            df[c] = 0
    df = df[tl.ML_COLONNES].copy()

    # Conversion numérique stricte : tout devient float (NaN résiduels → 0 sauf retard_sec)
    for col in tl.ML_COLONNES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in [c for c in tl.ML_COLONNES if c != "retard_sec"]:
        df[col] = df[col].fillna(0.0)

    nan_retard = df["retard_sec"].isna().sum()
    if nan_retard:
        log.info(f"  retard_sec NaN : {nan_retard:,} lignes (conservées pour l'inférence)")

    log.info(f"  Dataset ML-ready : {len(df):,} lignes × {len(df.columns)} colonnes")

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        log.info(f"  CSV ML-ready exporté → {output_path}")

    return df


def collecter_snapshot_ml() -> pd.DataFrame:
    """
    Appel API unique → produit les 2 fichiers du projet :
      Fichier 1 (sans transformation) → CSV_FILE  (passages_tglobal.csv)
      Fichier 2 (avec transformation) → CSV_FILE_ML (dataset_ml.csv)

    Retourne le DataFrame ML-ready (fichier 2).
    Utilisé par prediction.py pour l'inférence temps réel.
    """
    log.info(f"  Collecte snapshot API PRIM — {BASE_URL}/estimated-timetable")
    rows = get_estimated_timetable()
    if not rows:
        raise ValueError("L'API PRIM n'a retourné aucune donnée.")
    log.info(f"  {len(rows):,} passages récupérés")

    # Fichier 1 — sans transformation
    ecrire_csv(rows, CSV_FILE)
    log.info(f"  Fichier 1 (sans transformation) → {os.path.abspath(CSV_FILE)}")

    # Fichier 2 — avec transformation (via fichier temporaire pour éviter de relire CSV_FILE)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8-sig", newline=""
    ) as _f:
        _tmp = _f.name
        writer = csv.DictWriter(_f, fieldnames=tl.CSV_COLONNES,
                                extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    try:
        df = preparer_ml(_tmp, output_path=CSV_FILE_ML)
    finally:
        os.unlink(_tmp)

    log.info(f"  Fichier 2 (avec transformation)  → {os.path.abspath(CSV_FILE_ML)}")
    return df


# ─────────────────────────────────────────────
# PROGRAMME PRINCIPAL (CLI)
# ─────────────────────────────────────────────

def main():
    """
    Deux modes, deux fichiers produits :

      --collecter [--duree H]
          Collecte API continue (append) → fichier 1 (sans transformation)
          Fichier : dataset_predictions/passages_tglobal.csv

      --construire [--csv PATH]
          Produit toujours les 2 fichiers :
            Fichier 1 (sans transformation) → dataset_predictions/passages_tglobal.csv
            Fichier 2 (avec transformation) → dataset_predictions/dataset_ml.csv
          Sans --csv : appelle l'API pour générer le fichier 1, puis le transforme.
          Avec --csv : lit le CSV fourni comme fichier 1 et génère le fichier 2.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Construction du dataset PRIM IDFM")
    parser.add_argument("--collecter",  action="store_true",
                        help="Collecte API continue → fichier 1 (CSV brut, sans transformation)")
    parser.add_argument("--construire", action="store_true",
                        help="Produit les 2 fichiers : brut + ML-ready. "
                             "Sans --csv : appelle l'API d'abord.")
    parser.add_argument("--csv",   default=None, metavar="PATH",
                        help="CSV brut source pour --construire (défaut : appel API)")
    parser.add_argument("--duree", type=float, default=None, metavar="HEURES",
                        help="Durée de collecte en heures pour --collecter (défaut : infinie)")
    args = parser.parse_args()

    if not args.collecter and not args.construire:
        parser.print_help()
        return

    if args.collecter:
        csv_collecte = CSV_FILE
        if args.csv:
            csv_collecte = os.path.abspath(args.csv)
        msg = (f"Démarrage collecte — durée : {args.duree}h"
               if args.duree else "Démarrage collecte — durée : infinie (Ctrl+C pour arrêter)")
        log.info(msg)
        collecter_en_continu(duree_heures=args.duree,csv_file = csv_collecte)

    if args.construire:
        log.info("=" * 55)
        log.info("  CONSTRUCTION DES 2 FICHIERS")
        log.info("=" * 55)

        if args.csv:
            # Fichier 1 fourni → on génère uniquement le fichier 2
            csv_brut = args.csv
            CSV_DISPONIBLE = csv_brut
            log.info(f"  Fichier 1 (sans transformation) → {os.path.abspath(csv_brut)}")
            df_ml = preparer_ml(csv_brut, output_path=CSV_FILE_ML)
            log.info(f"  Fichier 2 (avec transformation)  → {os.path.abspath(CSV_FILE_ML)}")
        else:
            # Pas de CSV → appel API → fichier 1 + fichier 2
            log.info("  Appel API PRIM — collecte des passages…")
            rows = get_estimated_timetable()
            if not rows:
                log.error("  Aucune donnée retournée par l'API.")
                return
            ecrire_csv(rows, CSV_FILE)
            log.info(f"  Fichier 1 (sans transformation) → {os.path.abspath(CSV_FILE)}")
            df_ml = preparer_ml(CSV_FILE, output_path=CSV_FILE_ML)
            log.info(f"  Fichier 2 (avec transformation)  → {os.path.abspath(CSV_FILE_ML)}")

        log.info(f"  {len(df_ml):,} lignes × {len(df_ml.columns)} colonnes dans le fichier 2")


if __name__ == "__main__":
    main()
