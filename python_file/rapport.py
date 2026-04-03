"""
rapport.py — Génère les rapports Markdown à partir des résultats de prediction.py.
"""

import os
import pandas as pd
import tools as tl


_FEAT_GROUPS = {
    "Identification ligne": ["nom_ligne", "direction_ref"],
    "Temporelles":          ["heure_tranche", "mois", "jour_semaine", "periode_journee", "jour_ferie"],
    "Destination & arrêt":  ["terminus_encoded", "station_encoded"],
    "Météo horaire":        ["meteo_groupe", "precipitation", "snowfall", "wind_speed", "temperature"],
    "Alertes réseau PRIM":  ["categorie_alerte"],
    "Fréquentation arrêt":  ["occupation"],
}

_FEAT_DESC = {
    "nom_ligne":        ("Catégorielle", "Nom de la ligne (Métro 1, RER A…) — encodé entier fixe"),
    "direction_ref":    ("Numérique",    "Sens de la course : 1=aller, 2=retour, 0=inconnu"),
    "terminus_encoded": ("Numérique",    "Retard moyen historique du terminus (target encoding)"),
    "station_encoded":  ("Numérique",    "Retard moyen historique de l'arrêt (target encoding)"),
    "heure_tranche":    ("Numérique",    "Heure de passage (0–23)"),
    "mois":             ("Numérique",    "Mois de l'année (1–12)"),
    "jour_semaine":     ("Catégorielle", "Lundi … Dimanche — encodé entier fixe"),
    "periode_journee":  ("Catégorielle", "Pointe matin / Creuse / Méridienne / Pointe soir… — encodé entier fixe"),
    "jour_ferie":       ("Numérique",    "1 si jour férié français, 0 sinon"),
    "meteo_groupe":     ("Catégorielle", "Groupe météo horaire (ensoleille / pluie / neige / orage…) — encodé entier fixe"),
    "precipitation":    ("Numérique",    "Précipitations en mm/h"),
    "snowfall":         ("Numérique",    "Chutes de neige en cm"),
    "wind_speed":       ("Numérique",    "Vitesse du vent en km/h"),
    "temperature":      ("Numérique",    "Température en °C"),
    "categorie_alerte": ("Catégorielle", "Type d'alerte PRIM (aucune / greve / incident / travaux…) — encodé entier fixe"),
    "occupation":       ("Numérique",    "Nombre moyen d'entrées/heure à l'arrêt (données IDFM)"),
}


def _feat_block(features: list[str]) -> str:
    feat_set = set(features)
    lines = []
    for groupe, cols in _FEAT_GROUPS.items():
        actives = [c for c in cols if c in feat_set]
        if actives:
            lines.append(f"  - **{groupe}** : {', '.join(f'`{c}`' for c in actives)}")
    return "\n".join(lines) if lines else "  " + ", ".join(features)


def _feat_table(features: list[str]) -> list[str]:
    feat_set = set(features)
    rows = ["| Feature | Type | Description |", "|---------|------|-------------|"]
    for f in features:
        if f in feat_set and f in _FEAT_DESC:
            typ, desc = _FEAT_DESC[f]
            rows.append(f"| `{f}` | {typ} | {desc} |")
    return rows


def _fmt(row, col: str) -> str:
    std_col = col + " ±"
    val = row[col]
    std = row.get(std_col, None)
    if std is not None and pd.notna(std):
        if "%" in col or col in ("AUC", "F1", "Accuracy", "Précision", "Rappel"):
            return f"{val:.3f} ± {std:.3f}"
        if "(s)" in col:
            return f"{val:.1f} ± {std:.1f}"
        return f"{val:.3f} ± {std:.3f}"
    return f"{val:.3f}"


def _entete(titre: str, horodatage: str, csv_source: str, n_lignes: int,
            n_folds: int, cv_methode: str, scoring: str) -> list[str]:
    date_fmt  = horodatage[:8]
    h         = horodatage[9:]
    heure_fmt = f"{h[:2]}:{h[2:4]}:{h[4:]}"
    return [
        f"# {titre}\n",
        f"**Date :** {date_fmt[:4]}-{date_fmt[4:6]}-{date_fmt[6:]}  ",
        f"**Heure :** {heure_fmt}  ",
        f"**Source CSV :** `{csv_source}`  ",
        f"**Lignes utilisées :** {n_lignes:,}  ",
        f"**Évaluation :** {cv_methode} k={n_folds} (shuffle, random_state=42)  ",
        f"**Hyperparamètres :** GridSearchCV (cv=3, scoring={scoring})  \n",
        "---\n",
    ]


# ─────────────────────────────────────────────
# RAPPORT RÉGRESSION
# ─────────────────────────────────────────────

def ecrire_rapport_regression(
    resultats: pd.DataFrame,
    dossier: str,
    horodatage: str,
    csv_source: str,
    n_lignes: int,
    features: list[str],
    n_folds: int,
    seuil_proche: int,
    score_png: str,
    radar_png: str,
) -> str:
    """Génère rapport_regression_<horodatage>.md. Retourne le chemin."""
    proche_col = f"Proche≤{seuil_proche}s (%)"
    best = resultats.sort_values(tl.CRITERE_SELECTION_REGRESSION, ascending=False).iloc[0]

    lignes = _entete(
        "Rapport — Comparaison des modèles de régression IDFM",
        horodatage, csv_source, n_lignes, n_folds,
        cv_methode="KFold", scoring="r²",
    )

    lignes += [
        "## Features utilisées\n",
        _feat_block(features) + "\n",
        "### Détail des features\n",
    ]
    lignes += _feat_table(features)
    lignes += [
        "",
        "---\n",
        "## Visualisation du score global\n",
        f"![Score global des modèles de régression]({score_png})\n",
        "> Plus le score global est élevé, meilleur est le compromis global entre R², RMSE, MAE, MAPE et pourcentage de prédictions proches.\n",
    ]
    lignes += [
        "## Visualisation radar des modèles\n",
        f"![Radar des modèles de régression]({radar_png})\n",
        "> Chaque axe correspond à une métrique normalisée entre 0 et 1. Pour toutes les dimensions, plus la valeur est élevée, meilleur est le modèle.\n",
    ]
    lignes += [
        "",
        "---\n",
        "## Classements par métrique\n",
        f"> Métriques calculées par KFold k={n_folds}. Format : **moyenne ± écart-type**.\n",
        f"> Le **score global** est un score pondéré défini comme suit : "
        f"`0.35×R² + 0.25×RMSE_norm_inverse + 0.20×MAE_norm_inverse + "
        f"0.15×Proche≤{seuil_proche}s_norm + 0.05×MAPE_norm_inverse`.\n",
    ]

    configs = [
        ("Score global", False, "Score global pondéré", "↑ mieux"),
        ("R²",           False, "R² — variance expliquée", "↑ mieux"),
        ("MAE (s)",      True,  "MAE (s) — erreur moyenne absolue", "↓ mieux"),
        ("RMSE (s)",     True,  "RMSE (s) — pénalise les grandes erreurs", "↓ mieux"),
        ("MAPE (%)",     True,  "MAPE (%) — erreur relative", "↓ mieux"),
        (proche_col,     False, f"Proche≤{seuil_proche}s — % prédictions proches", "↑ mieux"),
    ]

    for col, ascending, titre, sens in configs:
        tri = resultats.sort_values(col, ascending=ascending).reset_index(drop=True)
        lignes.append(f"### {titre} ({sens})\n")
        lignes.append("| Rang | Modèle | Dataset | Valeur (moy ± std) | Meilleurs params |")
        lignes.append("|:----:|--------|---------|-------------------:|------------------|")
        for rang, (_, row) in enumerate(tri.iterrows(), start=1):
            lignes.append(
                f"| {rang} | {row['Modèle']} | {row['Dataset']} "
                f"| {_fmt(row, col)} | `{row['Meilleurs params']}` |"
            )
        lignes.append("")

    lignes += [
        "---\n",
        f"## Meilleur modèle global ({tl.CRITERE_SELECTION_REGRESSION})\n",
        f"**[{best['Dataset']}] {best['Modèle']}**\n",
        "| Métrique | Moyenne | Écart-type |",
        "|----------|--------:|----------:|",
        f"| Score global | `{best['Score global']:.4f}` | — |",
        f"| R²       | `{best['R²']:.4f}` | `{best.get('R² ±', float('nan')):.4f}` |",
        f"| MAE (s)  | `{best['MAE (s)']:.1f}` | `{best.get('MAE (s) ±', float('nan')):.1f}` |",
        f"| RMSE (s) | `{best['RMSE (s)']:.1f}` | `{best.get('RMSE (s) ±', float('nan')):.1f}` |",
        f"| MAPE (%) | `{best['MAPE (%)']:.1f}` | `{best.get('MAPE (%) ±', float('nan')):.1f}` |",
        f"| {proche_col} | `{best[proche_col]:.1f}%` | `{best.get(proche_col + ' ±', float('nan')):.1f}%` |",
        f"| Params   | `{best['Meilleurs params']}` | — |\n",
        "---\n",
        "## Glossaire\n",
        "| Métrique | Description |",
        "|----------|-------------|",
        "| **Score global** | Score pondéré entre 0 et 1 combinant R², RMSE, MAE, MAPE et % de prédictions proches. Plus il est élevé, meilleur est le compromis global. |",
        "| **R²** | Part de variance expliquée. 1 = parfait, 0 = équivalent à prédire la moyenne. |",
        "| **MAE** | Erreur absolue moyenne en secondes. |",
        "| **RMSE** | Comme MAE mais les grandes erreurs sont amplifiées. |",
        "| **MAPE** | Erreur en % relatif à la valeur réelle. Instable si retard ≈ 0. |",
        f"| **Proche≤{seuil_proche}s** | % de prédictions à moins de {seuil_proche}s de la réalité. |",
        f"| **KFold k={n_folds}** | Validation croisée sur {n_folds} blocs. L'écart-type mesure la stabilité. |",
    ]

    chemin = os.path.join(dossier, f"rapport_regression_{horodatage}.md")
    with open(chemin, "w", encoding="utf-8") as f:
        f.write("\n".join(lignes))
    return chemin


# ─────────────────────────────────────────────
# RAPPORT CLASSIFICATION
# ─────────────────────────────────────────────

def ecrire_rapport_classification(
    resultats: pd.DataFrame,
    dossier: str,
    horodatage: str,
    csv_source: str,
    n_lignes: int,
    features: list[str],
    n_folds: int,
    seuil_retard: int,
    roc_png: str,
    cm_pngs: list[str],
) -> str:
    """Génère rapport_classification_<horodatage>.md. Retourne le chemin."""
    best = resultats.sort_values("AUC", ascending=False).iloc[0]

    lignes = _entete(
        f"Rapport — Comparaison des modèles de classification IDFM (retard > {seuil_retard}s)",
        horodatage, csv_source, n_lignes, n_folds,
        cv_methode="StratifiedKFold", scoring="roc_auc",
    )

    lignes += [
        f"**Cible :** `retard_sec > {seuil_retard}` s → classe 1 (train en retard), classe 0 (à l'heure)  \n",
        "---\n",
        "## Features utilisées\n",
        _feat_block(features) + "\n",
        "### Détail des features\n",
    ]
    lignes += _feat_table(features)
    lignes += [
        "", "---\n",
        "## Courbes ROC\n",
        f"![Courbes ROC]({roc_png})\n",
        f"> Chaque courbe est la moyenne des {n_folds} folds (StratifiedKFold).\n",
        "---\n",
        "## Matrices de confusion\n",
        f"> Matrices agrégées sur les {n_folds} folds StratifiedKFold (somme de toutes les prédictions).\n",
    ]
    for png in cm_pngs:
        nom = png.replace(f"confusion_", "").replace(f"_{horodatage}.png", "").replace("_", " ").title()
        lignes.append(f"\n### {nom}\n")
        lignes.append(f"![Matrice de confusion {nom}]({png})\n")
    lignes += [
        "---\n",
        "## Classements par métrique\n",
        f"> Métriques calculées par StratifiedKFold k={n_folds}. Format : **moyenne ± écart-type**.\n",
    ]

    configs = [
        ("AUC",       False, "AUC-ROC",                    "↑ mieux"),
        ("F1",        False, "F1-score (seuil 0.5)",       "↑ mieux"),
        ("Accuracy",  False, "Accuracy",                   "↑ mieux"),
        ("Précision", False, "Précision",                  "↑ mieux"),
        ("Rappel",    False, "Rappel",                     "↑ mieux"),
    ]
    for col, ascending, titre, sens in configs:
        tri = resultats.sort_values(col, ascending=ascending).reset_index(drop=True)
        lignes.append(f"### {titre} ({sens})\n")
        lignes.append("| Rang | Modèle | Valeur (moy ± std) | Meilleurs params |")
        lignes.append("|:----:|--------|-------------------:|------------------|")
        for rang, (_, row) in enumerate(tri.iterrows(), start=1):
            lignes.append(
                f"| {rang} | {row['Modèle']} "
                f"| {_fmt(row, col)} | `{row['Meilleurs params']}` |"
            )
        lignes.append("")

    lignes += [
        "---\n",
        "## Meilleur modèle global (AUC moyen)\n",
        f"**{best['Modèle']}**\n",
        "| Métrique | Moyenne | Écart-type |",
        "|----------|--------:|----------:|",
        f"| AUC      | `{best['AUC']:.4f}` | `{best.get('AUC ±', float('nan')):.4f}` |",
        f"| F1       | `{best['F1']:.4f}` | `{best.get('F1 ±', float('nan')):.4f}` |",
        f"| Accuracy | `{best['Accuracy']:.4f}` | `{best.get('Accuracy ±', float('nan')):.4f}` |",
        f"| Précision | `{best['Précision']:.4f}` | `{best.get('Précision ±', float('nan')):.4f}` |",
        f"| Rappel   | `{best['Rappel']:.4f}` | `{best.get('Rappel ±', float('nan')):.4f}` |",
        f"| Params   | `{best['Meilleurs params']}` | — |\n",
        "---\n",
        "## Glossaire\n",
        "| Métrique | Description |",
        "|----------|-------------|",
        "| **AUC-ROC** | Aire sous la courbe ROC. 1 = parfait, 0.5 = aléatoire. |",
        "| **F1** | Moyenne harmonique précision/rappel. Utile si les classes sont déséquilibrées. |",
        "| **Accuracy** | % de prédictions correctes toutes classes confondues. |",
        "| **Précision** | Parmi les trains prédits en retard, combien le sont vraiment. |",
        "| **Rappel** | Parmi les trains réellement en retard, combien sont détectés. |",
        f"| **StratifiedKFold k={n_folds}** | Validation croisée stratifiée : conserve la proportion de chaque classe dans chaque fold. |",
        f"| **Seuil retard** | Un train est considéré en retard si `retard_sec > {seuil_retard}` (>{seuil_retard // 60} min). |",
    ]

    chemin = os.path.join(dossier, f"rapport_classification_{horodatage}.md")
    with open(chemin, "w", encoding="utf-8") as f:
        f.write("\n".join(lignes))
    return chemin
