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
        if _ref not in LIGNES_PAR_REF:  # ne pas écraser les noms explicites
            _p = _prefixe.get(str(r["route_type"]), "Ligne")
            _n = str(r["route_short_name"]).strip()
            _out[_ref] = f"{_p} {_n}" if _n else _p
    return _out

_GTFS_LIGNES = _build_gtfs_mapping()

def resoudre_nom_ligne(line_ref: str) -> str:
    """Nom lisible d'une ligne : LIGNES_PAR_REF en priorité, puis GTFS."""
    return LIGNES_PAR_REF.get(line_ref) or _GTFS_LIGNES.get(line_ref, "")

# ─────────────────────────────────────────────
# COLONNES CSV BRUT (collecte)
# ─────────────────────────────────────────────

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


# ══════════════════════════════════════════════════════════════
# PARTIE 1 — COLLECTE API
# ══════════════════════════════════════════════════════════════

JOURS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]


def _val(field) -> str:
    """Dépaquète les formats SIRI hybrides (str, dict, list)."""
    if field is None:
        return ""
    if isinstance(field, str):
        return field.strip().replace('\r', ' ').replace('\n', ' ')
    if isinstance(field, dict):
        return str(field.get("value", "")).strip().replace('\r', ' ').replace('\n', ' ')
    if isinstance(field, list):
        parts = [
            str(item.get("value", "")).strip() if isinstance(item, dict) else str(item).strip()
            for item in field
        ]
        return " / ".join(p for p in parts if p).replace('\r', ' ').replace('\n', ' ')
    return str(field).strip().replace('\r', ' ').replace('\n', ' ')


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


def _calc_retard_sec(prevu: str, estime: str) -> int | None:
    """Retourne (estime - prevu) en secondes entiers, ou None si l'un des deux est absent."""
    if not prevu or not estime:
        return None
    try:
        t_prev = datetime.fromisoformat(prevu.replace("Z", "+00:00"))
        t_est  = datetime.fromisoformat(estime.replace("Z", "+00:00"))
        return int(round((t_est - t_prev).total_seconds()))
    except Exception:
        return None


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


def _parse_journey(
    journey: dict,
    now_local: datetime,
    now_capture: str,
    alertes_map: "dict[str, dict]",
    ref_arrets: "dict[str, str] | None" = None,
) -> list[dict]:
    """Transforme une EstimatedVehicleJourney en liste de lignes CSV."""
    line_ref    = _val(journey.get("LineRef"))
    operateur   = _val(journey.get("OperatorRef"))
    direction   = _val(journey.get("DirectionRef"))
    terminus    = _val(journey.get("DestinationName"))

    alerte_info      = alertes_map.get(line_ref, {})
    alerte_active    = alerte_info.get("alerte_active", False)
    categorie_alerte = alerte_info.get("categorie_alerte", "aucune")

    # nom_ligne : LIGNES_PAR_REF en priorité, puis GTFS complet
    nom_ligne = resoudre_nom_ligne(line_ref)

    rows = []
    for call, _ in _parse_calls(journey):
        aimed_arr = _val(call.get("AimedArrivalTime"))
        aimed_dep = _val(call.get("AimedDepartureTime"))
        exp_arr   = _val(call.get("ExpectedArrivalTime"))
        exp_dep   = _val(call.get("ExpectedDepartureTime"))

        heure_ref            = aimed_dep or aimed_arr or exp_dep or exp_arr
        h_tranche, h_periode = _enrichissement_temporel(heure_ref)

        # nom_arret : priorité à StopPointName, fallback sur référentiel (même logique que build_features)
        stop_ref  = _val(call.get("StopPointRef"))
        nom_arret = _val(call.get("StopPointName"))
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
            "arrivee_prevue_hhmm":    fmt_hhmm(aimed_arr),
            "depart_prevu_hhmm":      fmt_hhmm(aimed_dep),
            "depart_estime_hhmm":     fmt_hhmm(exp_dep),
            "jour_semaine":           JOURS[now_local.weekday()],
            "heure_tranche":          h_tranche,
            "periode_journee":        h_periode,
            "alerte_active":          alerte_active,
            "categorie_alerte":       categorie_alerte,
            "date_capture":           now_capture,
            "retard_sec":             _calc_retard_sec(aimed_dep, exp_dep),
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


def filter_line(rows: list[dict]) -> list[dict]:
    """Filtre les entrées pour ne garder que les lignes à surveiller (définies dans LIGNES)."""
    lignes_surveillees = set(LIGNES.keys())
    avant = len(rows)
    rows = [r for r in rows if r.get("nom_ligne") in lignes_surveillees]
    log.info(f"  filter_line : {avant} → {len(rows)} entrées (lignes non surveillées retirées)")
    return rows


def ecrire_csv(rows: list[dict], csv_path: str = CSV_FILE):
    """Écrit les lignes dans le CSV brut (création ou append). Vérifie la compatibilité du schéma."""
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as _f:
            try:
                _existing_header = next(csv.reader(_f))
            except StopIteration:
                _existing_header = []
        if _existing_header and _existing_header != CSV_COLONNES:
            raise ValueError(
                f"Incompatibilité de schéma CSV !\n"
                f"  Header existant ({len(_existing_header)} cols) ≠ schéma courant ({len(CSV_COLONNES)} cols).\n"
                f"  Fichier : {os.path.abspath(csv_path)}\n"
                f"  Action  : renommer ou supprimer le fichier puis relancer la collecte."
            )

    nouveau = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLONNES, extrasaction="ignore",
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
    log.info(f"  Collecte continue — {len(LIGNES)} lignes configurées")
    log.info(f"  Fichier CSV : {os.path.abspath(csv_file)}")
    log.info(f"  Intervalle  : {INTERVALLE_CYCLE}s  |  Ctrl+C pour arrêter")
    log.info("=" * 55)

    try:
        while True:
            cycle += 1
            log.info(f"\n[Cycle {cycle}] {datetime.now().strftime('%H:%M:%S')}")
            try:
                rows = get_estimated_timetable()
                rows = filter_line(rows)
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
# ══════════════════════════════════════════════════════════════

# Jours fériés français 2024-2026
_JOURS_FERIES = {
    "2024-01-01", "2024-04-01", "2024-05-01", "2024-05-08", "2024-05-09",
    "2024-05-20", "2024-07-14", "2024-08-15", "2024-11-01", "2024-11-11", "2024-12-25",
    "2025-01-01", "2025-04-21", "2025-05-01", "2025-05-08", "2025-05-29",
    "2025-06-09", "2025-07-14", "2025-08-15", "2025-11-01", "2025-11-11", "2025-12-25",
    "2026-01-01", "2026-04-06", "2026-05-01", "2026-05-08", "2026-05-14",
    "2026-05-25", "2026-07-14", "2026-08-15", "2026-11-01", "2026-11-11", "2026-12-25",
}

# Vacances scolaires IDF (zone C) 2024-2026
_VACANCES_IDF = [
    ("2024-10-19", "2024-11-04"),  # Toussaint 2024
    ("2024-12-21", "2025-01-06"),  # Noël 2024
    ("2025-02-22", "2025-03-10"),  # Hiver 2025
    ("2025-04-19", "2025-05-05"),  # Printemps 2025
    ("2025-07-05", "2025-09-01"),  # Été 2025
    ("2025-10-18", "2025-11-03"),  # Toussaint 2025
    ("2025-12-20", "2026-01-05"),  # Noël 2025
    ("2026-02-14", "2026-03-02"),  # Hiver 2026
    ("2026-04-11", "2026-04-27"),  # Printemps 2026
    ("2026-07-04", "2026-08-31"),  # Été 2026
]


def est_jour_ferie(date_str: str) -> bool:
    """Retourne True si date_str (YYYY-MM-DD) est un jour férié français."""
    return isinstance(date_str, str) and date_str[:10] in _JOURS_FERIES


def est_vacances(date_str: str) -> bool:
    """Retourne True si date_str (YYYY-MM-DD) tombe dans les vacances scolaires IDF (zone C)."""
    if not isinstance(date_str, str) or len(date_str) < 10:
        return False
    d = date_str[:10]
    return any(start <= d <= end for start, end in _VACANCES_IDF)


def _wmo_groupe(code) -> str:
    """Réduit les codes WMO (0-99) en 7 groupes interprétables pour le ML."""
    if code is None or (isinstance(code, float) and pd.isna(code)):
        return "inconnu"
    c = int(code)
    if c == 0:              return "ensoleille"
    if c <= 3:              return "nuageux"
    if c in (45, 48):       return "brouillard"
    if 51 <= c <= 67:       return "pluie"
    if 71 <= c <= 77:       return "neige"
    if 80 <= c <= 82:       return "averses"
    if 95 <= c <= 99:       return "orage"
    return "autre"


_KEYWORDS_ALERTE = {
    "greve":    ["grève", "greve", "préavis", "mouvement social"],
    "incident": ["incident", "accident", "avarie", "panne", "défaillance", "défaut"],
    "travaux":  ["travaux", "chantier", "fermeture", "coupure", "interruption"],
    "meteo":    ["météo", "neige", "verglas", "vent", "inondation", "chaleur", "canicule"],
    "voyageur": ["malaise voyageur", "bagage", "colis", "urgence médicale"],
    "retard":   ["retard", "perturbation", "ralentissement", "trafic perturbé", "allongement"],
}


def _classifier_alerte(texte: str) -> str:
    """Classifie le texte libre d'une alerte en catégorie interprétable."""
    if not texte:
        return "autre"
    t = texte.lower()
    for cat, mots in _KEYWORDS_ALERTE.items():
        if any(m in t for m in mots):
            return cat
    return "autre"


def _cat_jour(date_str, weekday: int) -> str:
    """Retourne la catégorie jour IDFM (DIJFP / SAHV / JOHV)."""
    if est_jour_ferie(date_str) or weekday == 6:
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
    Charge un CSV brut (schéma courant) et calcule toutes les features.
    Retourne un DataFrame avec les NaN encore présents (à imputer ensuite).
    """
    df = pd.read_csv(csv_path, low_memory=False)

    # 0. nom_ligne : remplir les valeurs vides/absentes via LIGNES_PAR_REF + GTFS
    if "nom_ligne" not in df.columns:
        df["nom_ligne"] = df["line_ref"].map(resoudre_nom_ligne)
    else:
        mask_vide = df["nom_ligne"].isna() | (df["nom_ligne"].astype(str).str.strip() == "")
        df.loc[mask_vide, "nom_ligne"] = df.loc[mask_vide, "line_ref"].map(resoudre_nom_ligne)

    # 0b. Filtrer les lignes non surveillées (même logique que filter_line à la collecte)
    lignes_surveillees = set(LIGNES.keys())
    avant = len(df)
    df = df[df["nom_ligne"].isin(lignes_surveillees)].reset_index(drop=True)
    log.info(f"  build_features filter : {avant} → {len(df)} lignes (non surveillées retirées)")

    # 1. Retard arrivée en secondes
    arr_est  = pd.to_datetime(df["horaire_depart_estime"], utc=True, errors="coerce")
    arr_prev = pd.to_datetime(df["horaire_depart_prevu"],  utc=True, errors="coerce")
    df["retard_sec"] = (arr_est - arr_prev).dt.total_seconds().round().astype("Int64")

    # 3. Référence temporelle locale Paris
    ref_dt = pd.to_datetime(
        df["horaire_arrivee_prevu"].fillna(df["horaire_depart_prevu"]),
        utc=True, errors="coerce",
    ).dt.tz_convert("Europe/Paris")
    df["mois"]       = ref_dt.dt.month
    df["_date_str"]  = ref_dt.dt.strftime("%Y-%m-%d")
    df["_heure_int"] = ref_dt.dt.hour

    # 4. Jour férié et vacances scolaires IDF
    df["jour_ferie"] = df["_date_str"].apply(est_jour_ferie)
    df["vacances"]   = df["_date_str"].apply(est_vacances)

    # 5. Météo horaire Paris
    meteo_map = _fetch_meteo_paris_horaire(df["_date_str"].dropna().tolist())
    df["weathercode"]   = [meteo_map.get((d, h), {}).get("weathercode")   for d, h in zip(df["_date_str"], df["_heure_int"])]
    df["precipitation"] = [meteo_map.get((d, h), {}).get("precipitation") for d, h in zip(df["_date_str"], df["_heure_int"])]
    df["snowfall"]      = [meteo_map.get((d, h), {}).get("snowfall")      for d, h in zip(df["_date_str"], df["_heure_int"])]
    df["wind_speed"]    = [meteo_map.get((d, h), {}).get("wind_speed")    for d, h in zip(df["_date_str"], df["_heure_int"])]
    df["temperature"]   = [meteo_map.get((d, h), {}).get("temperature")   for d, h in zip(df["_date_str"], df["_heure_int"])]
    df["meteo_groupe"]  = df["weathercode"].apply(_wmo_groupe)

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

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute les valeurs manquantes :
      C — meteo_groupe     : 'inconnu'
      C — categorie_alerte : 'aucune'
      N — météo continues, terminus_encoded : mean global
      N — occupation : moyenne groupée (ligne × heure) puis SimpleImputer global
    """
    from sklearn.impute import SimpleImputer

    df["meteo_groupe"]     = df["meteo_groupe"].fillna("inconnu")
    df["categorie_alerte"] = df["categorie_alerte"].fillna("aucune")

    for col in ["precipitation", "snowfall", "wind_speed", "temperature", "terminus_encoded", "station_encoded"]:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())

    # Occupation : 1re passe groupée, 2e passe globale
    df["occupation"] = df["occupation"].fillna(
        df.groupby(["nom_ligne", "heure_tranche"])["occupation"].transform("mean")
    )
    if df["occupation"].isna().any():
        df[["occupation"]] = SimpleImputer(strategy="mean").fit_transform(df[["occupation"]])

    return df

def collecter_snapshot_ml():
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
        writer = csv.DictWriter(_f, fieldnames=CSV_COLONNES,
                                extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    # Fichier 2 — feature engineering sur le fichier temporaire
    try:
        df = build_features(_tmp)
        df = impute_missing(df)
        df.to_csv(CSV_FILE_ML, index=False, encoding="utf-8-sig")
        log.info(f"  Fichier 2 (ML-ready) → {os.path.abspath(CSV_FILE_ML)}  ({len(df):,} lignes)")
    finally:
        os.unlink(_tmp)

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
                        help="Feature engineering → fichier 2 (dataset_ml.csv). "
                             "Avec --csv : lit le CSV fourni. Sans --csv : appelle l'API.")
    parser.add_argument("--csv",   default=None, metavar="PATH",
                        help="CSV brut source pour --construire (défaut : appel API)")
    parser.add_argument("--output", default=None, metavar="PATH",
                        help="Chemin de sortie pour --construire (défaut : dataset_predictions/dataset_ml.csv)")
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
        collecter_en_continu(duree_heures=args.duree, csv_file=csv_collecte)

    if args.construire:
        csv_src = args.csv
        if not csv_src:
            log.info("Aucun --csv fourni : collecte API snapshot → fichier 1")
            rows = get_estimated_timetable()
            if not rows:
                raise ValueError("L'API PRIM n'a retourné aucune donnée.")
            ecrire_csv(rows, CSV_FILE)
            csv_src = CSV_FILE
            log.info(f"  Fichier 1 → {os.path.abspath(CSV_FILE)}")

        log.info(f"Feature engineering sur : {csv_src}")
        df = build_features(csv_src)
        df = impute_missing(df)

        out = args.output or CSV_FILE_ML
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        df.to_csv(out, index=False, encoding="utf-8-sig")
        log.info(f"  Fichier 2 (ML-ready) → {os.path.abspath(out)}  ({len(df):,} lignes)")


if __name__ == "__main__":
    main()
