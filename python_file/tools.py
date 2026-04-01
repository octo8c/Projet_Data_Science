from datetime import datetime
from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

# Mapping line_ref → nom_ligne (fallback si prim_global_csv ne l'a pas rempli)
LIGNES_PAR_REF = {
    "STIF:Line::C01371:": "Métro 1",   "STIF:Line::C01372:": "Métro 2",
    "STIF:Line::C01373:": "Métro 3",   "STIF:Line::C01386:": "Métro 3b",
    "STIF:Line::C01374:": "Métro 4",   "STIF:Line::C01375:": "Métro 5",
    "STIF:Line::C01376:": "Métro 6",   "STIF:Line::C01377:": "Métro 7",
    "STIF:Line::C01387:": "Métro 7b",  "STIF:Line::C01378:": "Métro 8",
    "STIF:Line::C01379:": "Métro 9",   "STIF:Line::C01380:": "Métro 10",
    "STIF:Line::C01381:": "Métro 11",  "STIF:Line::C01382:": "Métro 12",
    "STIF:Line::C01383:": "Métro 13",  "STIF:Line::C01384:": "Métro 14",
    "STIF:Line::C01742:": "RER A",     "STIF:Line::C01743:": "RER B",
    "STIF:Line::C01727:": "RER C",     "STIF:Line::C01728:": "RER D",
    "STIF:Line::C01729:": "RER E",
    "STIF:Line::C01737:": "Ligne H",   "STIF:Line::C01738:": "Ligne J",
    "STIF:Line::C01739:": "Ligne K",   "STIF:Line::C01740:": "Ligne L",
    "STIF:Line::C01741:": "Ligne N",   "STIF:Line::C01744:": "Ligne P",
    "STIF:Line::C01745:": "Ligne R",   "STIF:Line::C01746:": "Ligne U",
    "STIF:Line::C01389:": "Tram T1",   "STIF:Line::C01390:": "Tram T2",
    "STIF:Line::C01391:": "Tram T3a",  "STIF:Line::C01679:": "Tram T3b",
    "STIF:Line::C01392:": "Tram T4",   "STIF:Line::C01775:": "Tram T5",
    "STIF:Line::C01776:": "Tram T6",   "STIF:Line::C01777:": "Tram T7",
    "STIF:Line::C01778:": "Tram T8",   "STIF:Line::C02317:": "Tram T9",
    "STIF:Line::C02316:": "Tram T10",  "STIF:Line::C02024:": "Tram T11",
    "STIF:Line::C02048:": "Tram T13",
}

# ─────────────────────────────────────────────
# LIGNES À SURVEILLER
# Format PRIM obligatoire : "STIF:Line::CXXXXX:"  (double :: et : final)
# Exemple curl validé :
#   curl 'https://prim.iledefrance-mobilites.fr/marketplace/estimated-timetable
#         ?LineRef=STIF%3ALine%3A%3AC01742%3A'
# ─────────────────────────────────────────────
LIGNES = {ref: nom for nom, ref in LIGNES_PAR_REF.items()}

_KEYWORDS_ALERTE = {
    "greve":    ["grève", "greve", "préavis", "mouvement social"],
    "incident": ["incident", "accident", "avarie", "panne", "défaillance", "défaut"],
    "travaux":  ["travaux", "chantier", "fermeture", "coupure", "interruption"],
    "meteo":    ["météo", "neige", "verglas", "vent", "inondation", "chaleur", "canicule"],
    "voyageur": ["malaise voyageur", "bagage", "colis", "urgence médicale"],
    "retard":   ["retard", "perturbation", "ralentissement", "trafic perturbé", "allongement"],
}

JOURS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

# Jours fériés français 2025-2026
_JOURS_FERIES = {
    "2025-01-01", "2025-04-21", "2025-05-01", "2025-05-08", "2025-05-29",
    "2025-06-09", "2025-07-14", "2025-08-15", "2025-11-01", "2025-11-11", "2025-12-25",
    "2026-01-01", "2026-04-06", "2026-05-01", "2026-05-08", "2026-05-14",
    "2026-05-25", "2026-07-14", "2026-08-15", "2026-11-01", "2026-11-11", "2026-12-25",
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


# ─────────────────────────────────────────────
# COLONNES CSV
# ─────────────────────────────────────────────
CSV_COLONNES_COLLECTE = [
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

# Colonnes du CSV ML-ready (dans l'ordre final)
ML_COLONNES = [
    # Target
    "retard_sec",
    # Numériques
    "direction_ref", "terminus_encoded", "station_encoded", "heure_tranche", "mois",
    "jour_ferie", "precipitation", "snowfall", "wind_speed", "temperature", "occupation",
    # Catégorielles encodées (int)
    "nom_ligne", "jour_semaine", "periode_journee", "meteo_groupe", "categorie_alerte",
]

# Mappings entiers fixes pour chaque colonne catégorielle.
# Déterministes : le même code sera assigné indépendamment du dataset traité.
# Valeur inconnue → -1 (jamais vu à l'entraînement mais géré par les modèles).
CAT_ENCODINGS: dict[str, dict[str, int]] = {
    "nom_ligne": {
        nom: i for i, nom in enumerate(["_inconnu"] + sorted(LIGNES.keys()))
    },
    "jour_semaine": {j: i for i, j in enumerate(JOURS)},
    "periode_journee": {
        "Nuit": 0, "Pointe matin": 1, "Creuse matin": 2, "Méridienne": 3,
        "Creuse après-midi": 4, "Pointe soir": 5, "Soirée": 6,
    },
    "meteo_groupe": {
        "ensoleille": 0, "nuageux": 1, "brouillard": 2, "pluie": 3,
        "neige": 4, "averses": 5, "orage": 6, "autre": 7, "inconnu": 8,
    },
    "categorie_alerte": {
        "aucune": 0, "greve": 1, "incident": 2, "travaux": 3,
        "meteo": 4, "retard": 5, "voyageur": 6, "autre": 7,
    },
}

#-----


TARGET       = "retard_sec"
FEATURES_CAT = ["nom_ligne", "jour_semaine", "periode_journee", "meteo_groupe", "categorie_alerte","stations"]
FEATURES_NUM = [
    "heure_tranche", "mois", "jour_ferie", "occupation",
    "direction_ref", "terminus_encoded", "station_encoded",
    "precipitation", "snowfall", "wind_speed", "temperature",
]
FEATURES = FEATURES_CAT + FEATURES_NUM

N_FOLDS        = 5
RANDOM_STATE   = 42
SEUIL_PROCHE_S = 60   # régression : prédiction "proche" si |erreur| ≤ 60 s
SEUIL_RETARD   = 300  # classification : retard si > 5 min


# ── Grilles de paramètres ─────────────────────
PARAM_GRIDS_REGRESSION: dict[str, dict] = {
    "Ridge": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "DecisionTree": {
        "max_depth":         [3, 5, 8, 12, 18],
        "min_samples_split": [2, 10, 50],
        "min_samples_leaf":  [1, 5, 20],
    },
    "RandomForest": {
        "n_estimators":     [100, 200],
        "max_depth":        [5, 10, 15, 20],
        "min_samples_leaf": [1, 5, 20],
    },
    "ExtraTrees": {
        "n_estimators":     [100, 200],
        "max_depth":        [5, 10, 15, 20],
        "min_samples_leaf": [1, 5, 20],
    },
    "GradientBoosting": {
        "n_estimators":  [100, 200],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth":     [3, 5],
    },
    "HistGradientBoosting": {
        "max_iter":          [100, 200],
        "learning_rate":     [0.05, 0.1, 0.2],
        "max_leaf_nodes":    [15, 31, 63],
        "l2_regularization": [0.0, 0.1, 1.0],
    },
    "KNeighbors": {
        "n_neighbors": [5, 10, 20, 50],
        "weights":     ["uniform", "distance"],
    },
}

PARAM_GRIDS_CLASSIFICATION: dict[str, dict] = {
    "LogisticRegression": {"C": [0.01, 0.1, 1.0, 10.0]},
    "SVC":                {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]},
    "DecisionTree": {
        "max_depth":         [3, 5, 8, 12],
        "min_samples_split": [2, 10, 50],
    },
    "RandomForest": {
        "n_estimators":     [100, 200],
        "max_depth":        [5, 10, 15],
        "min_samples_leaf": [1, 5],
    },
    "GradientBoosting": {
        "n_estimators":  [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth":     [3, 5],
    },
    "HistGradientBoosting": {
        "max_iter":       [100, 200],
        "learning_rate":  [0.05, 0.1],
        "max_leaf_nodes": [15, 31],
    },
    "KNeighbors": {
        "n_neighbors": [5, 10, 20, 50],
        "weights":     ["uniform", "distance"],
    },
}

#-----

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

def _target_encode_terminus(df: pd.DataFrame) -> pd.DataFrame:
    """Encode la colonne 'terminus' par le retard moyen observé (target encoding)."""
    global_mean = df["retard_sec"].mean(skipna=True)
    if pd.isna(global_mean):
        global_mean = 0.0
    terminus_clean = df["terminus"].fillna("_inconnu").str.strip().where(
        df["terminus"].notna() & (df["terminus"].str.strip() != ""), other="_inconnu"
    )
    means = df.groupby(terminus_clean)["retard_sec"].mean()
    df["terminus_encoded"] = terminus_clean.map(means).fillna(global_mean)
    return df


def _target_encode_station(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode stop_ref par le retard moyen observé à cet arrêt (target encoding).
    Capture l'effet propre à chaque station indépendamment de la ligne.
    Les arrêts sans retard observé reçoivent le retard moyen global.
    """
    global_mean = df["retard_sec"].mean(skipna=True)
    if pd.isna(global_mean):
        global_mean = 0.0
    stop_clean = df["stop_ref"].fillna("_inconnu").astype(str).str.strip()
    stop_clean = stop_clean.where(stop_clean != "", other="_inconnu")
    means = df.groupby(stop_clean)["retard_sec"].mean()
    df["station_encoded"] = stop_clean.map(means).fillna(global_mean)
    
    TargetEncoder()
    return df


def target_encode(df:pd.DataFrame,categorical_features:list,target:str) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        categorical_features (list): _description_

    Returns:
        pd.DataFrame: _description_
    """
    te = TargetEncoder(categories='auto',target_type='continuous',cv=5,smooth='auto',random_state=42)
    y = df[target]
    X = df[categorical_features]
    X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=.80)
    te.fit(X_train,y_train)
    te.transform(X_train)
    te.transform(X_test)
    return df

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