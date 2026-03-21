"""
PRIM IDFM — Requête globale : toutes les lignes, tous les arrêts → CSV
API : GET /estimated-timetable  (SIRI Lite)
Doc : https://prim.iledefrance-mobilites.fr/fr/apis/idfm-ivtr-requete_globale

Différence clé vs stop-monitoring :
  - stop-monitoring  → 1 arrêt  → ses N prochains passages
  - estimated-timetable → 1 ligne → TOUTES les courses en cours
                          avec TOUS leurs arrêts et horaires

Ce script :
  1. Charge le référentiel des lignes depuis IDFM Open Data (CSV)
  2. Pour chaque ligne, appelle /estimated-timetable
  3. Déroule chaque course → chaque arrêt → calcule le retard
  4. Écrit dans un CSV incrémental (mode append)

Quotas /estimated-timetable : 10 000 requêtes/jour (à vérifier dans votre compte)
"""

import requests
import csv
import os
import time
import logging
from datetime import datetime, timezone

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

API_KEYS        = ["bh8enPs1rJpVrXrpxUrXGAWq3f3BbCLu","UxZ2oEXNyvd3ym0zbx2AGgW0w8AYgHSj"]
_current_key_idx = 0   # index de la clé active
BASE_URL  = "https://prim.iledefrance-mobilites.fr/marketplace"
CSV_FILE  = "dataset_predictions/passages_global.csv"

# Pause entre chaque requête ligne (secondes) — évite le rate-limiting
PAUSE_ENTRE_LIGNES = 1.0

# Pause entre deux cycles complets (secondes)
INTERVALLE_CYCLE = 120

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# LIGNES À SURVEILLER
# Format PRIM obligatoire : "STIF:Line::CXXXXX:"  (double :: et : final)
# Exemple curl validé :
#   curl 'https://prim.iledefrance-mobilites.fr/marketplace/estimated-timetable
#         ?LineRef=STIF%3ALine%3A%3AC01742%3A'
# ─────────────────────────────────────────────

# Dictionnaire inversé : line_ref → nom lisible (pour remplir nom_ligne depuis la réponse ALL)
LIGNES_PAR_REF: dict[str, str] = {}  # rempli après LIGNES

LIGNES = {
    # ── Métro ──────────────────────────────
    "Métro 1":   "STIF:Line::C01371:",
    "Métro 2":   "STIF:Line::C01372:",
    "Métro 3":   "STIF:Line::C01373:",
    "Métro 3b":  "STIF:Line::C01386:",
    "Métro 4":   "STIF:Line::C01374:",
    "Métro 5":   "STIF:Line::C01375:",
    "Métro 6":   "STIF:Line::C01376:",
    "Métro 7":   "STIF:Line::C01377:",
    "Métro 7b":  "STIF:Line::C01387:",
    "Métro 8":   "STIF:Line::C01378:",
    "Métro 9":   "STIF:Line::C01379:",
    "Métro 10":  "STIF:Line::C01380:",
    "Métro 11":  "STIF:Line::C01381:",
    "Métro 12":  "STIF:Line::C01382:",
    "Métro 13":  "STIF:Line::C01383:",
    "Métro 14":  "STIF:Line::C01384:",
    # ── RER ────────────────────────────────
    "RER A":     "STIF:Line::C01742:",
    "RER B":     "STIF:Line::C01743:",
    "RER C":     "STIF:Line::C01727:",
    "RER D":     "STIF:Line::C01728:",
    "RER E":     "STIF:Line::C01729:",
    # ── Transilien ─────────────────────────
    "Ligne H":   "STIF:Line::C01737:",
    "Ligne J":   "STIF:Line::C01738:",
    "Ligne K":   "STIF:Line::C01739:",
    "Ligne L":   "STIF:Line::C01740:",
    "Ligne N":   "STIF:Line::C01741:",
    "Ligne P":   "STIF:Line::C01744:",
    "Ligne R":   "STIF:Line::C01745:",
    "Ligne U":   "STIF:Line::C01746:",
    # ── Tramway ────────────────────────────
    "Tram T1":   "STIF:Line::C01389:",
    "Tram T2":   "STIF:Line::C01390:",
    "Tram T3a":  "STIF:Line::C01391:",
    "Tram T3b":  "STIF:Line::C01679:",
    "Tram T4":   "STIF:Line::C01392:",
    "Tram T5":   "STIF:Line::C01775:",
    "Tram T6":   "STIF:Line::C01776:",
    "Tram T7":   "STIF:Line::C01777:",
    "Tram T8":   "STIF:Line::C01778:",
    "Tram T9":   "STIF:Line::C02317:",
    "Tram T10":  "STIF:Line::C02316:",
    "Tram T11":  "STIF:Line::C02024:",
    "Tram T13":  "STIF:Line::C02048:",
}

# Remplissage du dict inversé
LIGNES_PAR_REF = {ref: nom for nom, ref in LIGNES.items()}

# ─────────────────────────────────────────────
# COLONNES CSV
# ─────────────────────────────────────────────

CSV_COLONNES = [
    # — Contexte de collecte —
    "timestamp_collecte",       # Quand on a fait la requête
    "timestamp_utc",            # Idem en UTC
    # — Identification de la course —
    "nom_ligne",                # Nom lisible (ex: "RER B")
    "line_ref",                 # ID IDFM de la ligne
    "operateur",                # RATP / SNCF Transilien…
    "direction_ref",            # Code direction (Aller / Retour)
    "terminus",                 # Terminus de la course
    "vehicle_journey_ref",      # Identifiant unique de la course
    "date_course",              # Jour de la course (YYYY-MM-DD)
    # — Arrêt concerné —
    "ordre_arret",              # Position dans la course (1, 2, 3…)
    "stop_ref",                 # Identifiant de l'arrêt IDFM
    "nom_arret",                # Nom de l'arrêt
    # — Horaires —
    "horaire_arrivee_prevu",    # AimedArrivalTime (ISO 8601)
    "horaire_depart_prevu",     # AimedDepartureTime (ISO 8601)
    "horaire_arrivee_estime",   # ExpectedArrivalTime (temps réel)
    "horaire_depart_estime",    # ExpectedDepartureTime (temps réel)
    "arrivee_prevue_hhmm",      # HH:MM lisible
    "depart_prevu_hhmm",        # HH:MM lisible
    "depart_estime_hhmm",       # HH:MM lisible
    # — Retard calculé —
    "retard_arrivee_sec",       # En secondes (négatif = en avance)
    "retard_depart_sec",
    "retard_depart_min",        # En minutes, arrondi à 1 décimale
    "statut_retard",            # Catégorie textuelle
    # — Type de passage —
    "est_enregistre",           # True si RecordedCall (déjà passé)
    "quai",                     # Numéro de quai si fourni
    # — Enrichissement temporel —
    "jour_semaine",             # Lundi…Dimanche
    "heure_tranche",            # 0–23 (heure entière du départ prévu)
    "periode_journee",          # Nuit / Pointe matin / Creuse…
]

# ─────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────

JOURS = ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"]

def fmt_hhmm(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        return datetime.fromisoformat(iso.replace("Z","+00:00")).strftime("%H:%M")
    except Exception:
        return ""

def _val(field) -> str:
    """
    Dépaquète les formats SIRI hybrides retournés par PRIM :
      - Chaîne simple           "RATP"                  → "RATP"
      - Dict valeur             {'value': 'RATP'}        → "RATP"
      - Liste de dicts          [{'value': 'Nation'}]    → "Nation"
      - Liste de chaînes        ['Nation']               → "Nation"
      - None / vide             None                     → ""
    Pour les listes avec plusieurs éléments on joint par " / ".
    """
    if field is None:
        return ""
    if isinstance(field, str):
        return field.strip()
    if isinstance(field, dict):
        return str(field.get("value", "")).strip()
    if isinstance(field, list):
        parts = []
        for item in field:
            if isinstance(item, dict):
                parts.append(str(item.get("value", "")).strip())
            else:
                parts.append(str(item).strip())
        return " / ".join(p for p in parts if p)
    return str(field).strip()


def diff_sec(iso_aimed: str | None, iso_expected: str | None) -> int | None:
    if not iso_aimed or not iso_expected:
        return None
    try:
        t1 = datetime.fromisoformat(iso_aimed.replace("Z", "+00:00"))
        t2 = datetime.fromisoformat(iso_expected.replace("Z", "+00:00"))
        return round((t2 - t1).total_seconds())
    except Exception:
        return None


def classer_retard(sec: int | None) -> str:
    if sec is None: return "Inconnu"
    if sec < -60:   return "En avance"
    if sec <= 60:   return "A l'heure"
    if sec <= 180:  return "Léger retard"
    if sec <= 360:  return "Retard"
    return "Retard important"


def periode(h: int) -> str:
    if h < 6:   return "Nuit"
    if h < 9:   return "Pointe matin"
    if h < 11:  return "Creuse matin"
    if h < 14:  return "Méridienne"
    if h < 16:  return "Creuse après-midi"
    if h < 20:  return "Pointe soir"
    return "Soirée"


def _iso(field) -> str:
    """Extrait une chaîne ISO 8601 depuis un champ potentiellement wrappé."""
    v = _val(field)
    return v if v else ""


# ─────────────────────────────────────────────
# APPEL API — /estimated-timetable
# ─────────────────────────────────────────────

def get_estimated_timetable() -> list[dict]:
    """
    Appelle GET /estimated-timetable avec LineRef=ALL (toutes les lignes en une requête).
    Extrait line_ref depuis chaque course SIRI et déduit nom_ligne via LIGNES_PAR_REF.
    """
    global _current_key_idx
    url    = f"{BASE_URL}/estimated-timetable"
    params = {"LineRef": "ALL"}

    # Rotation de clé en cas de 429 — essaie chaque clé une fois
    for _ in range(len(API_KEYS)):
        headers = {"apiKey": API_KEYS[_current_key_idx]}
        resp    = requests.get(url, headers=headers, params=params, timeout=15)

        if resp.status_code == 429:
            ancienne = _current_key_idx
            _current_key_idx = (_current_key_idx + 1) % len(API_KEYS)
            log.warning(
                f"  429 Rate limit — clé [{ancienne}] épuisée, "
                f"passage à la clé [{_current_key_idx}]"
            )
            continue   # réessaie avec la nouvelle clé

        resp.raise_for_status()
        break
    else:
        raise requests.HTTPError(f"Toutes les clés API sont en rate-limit (429)")

    data = resp.json()

    now_local = datetime.now()
    now_utc   = datetime.now(timezone.utc)
    rows      = []

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
                    # — Métadonnées de la course — dépaquetage systématique
                    line_ref  = _val(journey.get("LineRef"))
                    nom_ligne = LIGNES_PAR_REF.get(line_ref, line_ref)
                    vj_ref = _val(
                        journey.get("FramedVehicleJourneyRef", {})
                               .get("DatedVehicleJourneyRef")
                    )
                    date_course = _val(
                        journey.get("FramedVehicleJourneyRef", {})
                               .get("DataFrameRef")
                    )
                    terminus  = _val(journey.get("DestinationName"))
                    direction = _val(journey.get("DirectionRef"))
                    operateur = _val(journey.get("OperatorRef"))

                    # — Arrêts déjà passés (RecordedCall) —
                    recorded = journey.get("RecordedCalls", {})
                    if isinstance(recorded, dict):
                        recorded = recorded.get("RecordedCall", [])
                    if isinstance(recorded, dict):
                        recorded = [recorded]

                    # — Arrêts à venir (EstimatedCall) —
                    calls = journey.get("EstimatedCalls", {})
                    if isinstance(calls, dict):
                        calls = calls.get("EstimatedCall", [])
                    if isinstance(calls, dict):
                        calls = [calls]

                    all_calls = (
                        [(c, True)  for c in (recorded or [])] +
                        [(c, False) for c in (calls    or [])]
                    )

                    for call, is_recorded in all_calls:
                        # Dépaqueter tous les champs potentiellement wrappés
                        aimed_arr  = _iso(call.get("AimedArrivalTime"))
                        aimed_dep  = _iso(call.get("AimedDepartureTime"))
                        exp_arr    = _iso(call.get("ExpectedArrivalTime"))
                        exp_dep    = _iso(call.get("ExpectedDepartureTime"))
                        stop_ref   = _val(call.get("StopPointRef"))
                        nom_arret  = _val(call.get("StopPointName"))
                        ordre      = _val(call.get("Order"))
                        quai       = _val(call.get("ArrivalPlatformName")
                                         or call.get("DeparturePlatformName"))

                        # Calcul retard — None si Aimed absent (métro RATP)
                        ret_arr_sec = diff_sec(aimed_arr, exp_arr) if aimed_arr else None
                        ret_dep_sec = diff_sec(aimed_dep, exp_dep) if aimed_dep else None
                        ret_dep_min = round(ret_dep_sec / 60, 1) if ret_dep_sec is not None else ""

                        # Statut retard enrichi : distingue "pas de données" vs "à l'heure"
                        if ret_dep_sec is None and not aimed_dep:
                            statut = "N/A (pas d'horaire théorique)"
                        else:
                            statut = classer_retard(ret_dep_sec)

                        # Tranche horaire : priorité Aimed, fallback Expected
                        heure_ref = aimed_dep or aimed_arr or exp_dep or exp_arr
                        h_tranche, h_periode = "", ""
                        if heure_ref:
                            try:
                                h = datetime.fromisoformat(
                                    heure_ref.replace("Z", "+00:00")
                                ).hour
                                h_tranche = h
                                h_periode = periode(h)
                            except Exception:
                                pass

                        rows.append({
                            "timestamp_collecte":     now_local.isoformat(timespec="seconds"),
                            "timestamp_utc":          now_utc.isoformat(timespec="seconds"),
                            "nom_ligne":              nom_ligne,
                            "line_ref":               line_ref,
                            "operateur":              operateur,
                            "direction_ref":          direction,
                            "terminus":               terminus,
                            "vehicle_journey_ref":    vj_ref,
                            "date_course":            date_course,
                            "ordre_arret":            ordre,
                            "stop_ref":               stop_ref,
                            "nom_arret":              nom_arret,
                            "horaire_arrivee_prevu":  aimed_arr,
                            "horaire_depart_prevu":   aimed_dep,
                            "horaire_arrivee_estime": exp_arr,
                            "horaire_depart_estime":  exp_dep,
                            "arrivee_prevue_hhmm":    fmt_hhmm(aimed_arr),
                            "depart_prevu_hhmm":      fmt_hhmm(aimed_dep),
                            "depart_estime_hhmm":     fmt_hhmm(exp_dep),
                            "retard_arrivee_sec":     ret_arr_sec if ret_arr_sec is not None else "",
                            "retard_depart_sec":      ret_dep_sec if ret_dep_sec is not None else "",
                            "retard_depart_min":      ret_dep_min,
                            "statut_retard":          statut,
                            "est_enregistre":         is_recorded,
                            "quai":                   quai,
                            "jour_semaine":           JOURS[now_local.weekday()],
                            "heure_tranche":          h_tranche,
                            "periode_journee":        h_periode,
                        })

    except (KeyError, TypeError) as e:
        log.warning(f"Parsing échoué : {e}")

    return rows

# ─────────────────────────────────────────────
# ÉCRITURE CSV
# ─────────────────────────────────────────────

def ecrire_csv(rows: list[dict]):
    nouveau = not os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLONNES, extrasaction="ignore")
        if nouveau:
            writer.writeheader()
        writer.writerows(rows)

# ─────────────────────────────────────────────
# COLLECTE
# ─────────────────────────────────────────────

def collecter_toutes_les_lignes() -> int:
    """Un cycle : interroge toutes les lignes, retourne le nombre de lignes écrites."""
    total = 0
    #for nom, ref in LIGNES.items():
    #    log.info(f"  → {nom} ({ref})")
    #    try:
    #        rows = get_estimated_timetable(ref)
    #        for r in rows:
    #            r["nom_ligne"] = nom
    #        ecrire_csv(rows)
    #        total += len(rows)
    #        log.info(f"     {len(rows)} arrêts écrits")
    #    except requests.HTTPError as e:
    #        log.error(f"     HTTP {e.response.status_code} — {e.response.text[:120]}")
    #    except Exception as e:
    #        log.error(f"     Erreur : {e}")
    #    time.sleep(PAUSE_ENTRE_LIGNES)
    try:
        rows = get_estimated_timetable()
        ecrire_csv(rows)
        total = len(rows)
        log.info(f"  {total} arrêts écrits (requête ALL)")
    except requests.HTTPError as e:
        log.error(f"  HTTP {e.response.status_code} — {e.response.text[:120]}")
    except Exception as e:
        log.error(f"  Erreur : {e}")
    return total


def collecter_en_continu(duree_heures: float | None = None):
    debut    = datetime.now()
    cycle    = 0
    limite   = duree_heures * 3600 if duree_heures else None

    log.info("=" * 55)
    log.info(f"  Collecte continue — {len(LIGNES)} lignes configurées")
    log.info(f"  Fichier CSV : {os.path.abspath(CSV_FILE)}")
    log.info(f"  Intervalle  : {INTERVALLE_CYCLE}s  |  Ctrl+C pour arrêter")
    log.info("=" * 55)

    try:
        while True:
            cycle += 1
            log.info(f"\n[Cycle {cycle}] {datetime.now().strftime('%H:%M:%S')}")
            total = collecter_toutes_les_lignes()
            log.info(f"  Total cycle {cycle} : {total} lignes CSV")

            if limite and (datetime.now() - debut).total_seconds() >= limite:
                log.info(f"Durée atteinte ({duree_heures}h). Fin.")
                break

            time.sleep(INTERVALLE_CYCLE)

    except KeyboardInterrupt:
        log.info(f"\nArrêt manuel après {cycle} cycle(s).")
        log.info(f"CSV disponible : {os.path.abspath(CSV_FILE)}")

# ─────────────────────────────────────────────
# PROGRAMME PRINCIPAL
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── MODE A : snapshot unique (pour tester) ──────────────────
    # Décommenter pour une seule collecte
    #
    # total = collecter_toutes_les_lignes()
    # log.info(f"Terminé : {total} lignes écrites dans {CSV_FILE}")

    # ── MODE B : collecte continue toutes les 2 minutes ─────────
    # Durée None = tourne indéfiniment
    # Chaque cycle produit ~3 000 – 8 000 lignes selon l'heure
    #
    collecter_en_continu(duree_heures=None)

# ─────────────────────────────────────────────
# ANALYSE RAPIDE DU CSV PRODUIT
# ─────────────────────────────────────────────
# import pandas as pd
#
# df = pd.read_csv("passages_global.csv")
#
# # Retard moyen par ligne
# df.groupby("nom_ligne")["retard_depart_min"].mean().sort_values(ascending=False)
#
# # Distribution des statuts de retard
# df["statut_retard"].value_counts(normalize=True) * 100
#
# # Top 10 des arrêts les plus retardés
# df.groupby("nom_arret")["retard_depart_sec"].mean().nlargest(10)
#
# # Retard moyen par tranche horaire (toutes lignes)
# df.groupby("heure_tranche")["retard_depart_min"].mean()
#
# # Courses avec retard > 5 min
# gros = df[df["retard_depart_min"] > 5][
#     ["timestamp_collecte","nom_ligne","vehicle_journey_ref",
#      "nom_arret","depart_prevu_hhmm","depart_estime_hhmm","retard_depart_min"]
# ]
