"""
rapport.py — Génère un rapport Markdown à partir des résultats de prediction.py.
"""

import os
import pandas as pd


def ecrire_rapport(
    resultats: pd.DataFrame,
    dossier: str,
    horodatage: str,
    csv_source: str,
    n_lignes: int,
    features: list[str],
    n_folds: int,
    seuil_proche: int,
) -> str:
    """
    Génère un fichier rapport_<horodatage>.md dans dossier.
    Retourne le chemin du fichier créé.

    Le DataFrame resultats doit contenir pour chaque métrique deux colonnes :
      "MAE (s)"  (moyenne sur les k folds) et "MAE (s) ±"  (écart-type).
    """
    proche_col = f"Proche≤{seuil_proche}s (%)"
    best       = resultats.sort_values("R²", ascending=False).iloc[0]
    date_fmt   = horodatage[:8]
    heure_fmt  = horodatage[9:].replace("-", ":")

    def _fmt(row, col) -> str:
        """Formate 'moy ± std' pour une métrique donnée."""
        std_col = col + " ±"
        if std_col in row.index and pd.notna(row[std_col]):
            if col == proche_col or "%" in col:
                return f"{row[col]:.1f}% ± {row[std_col]:.1f}%"
            return f"{row[col]:.3f} ± {row[std_col]:.3f}"
        if col == proche_col or "%" in col:
            return f"{row[col]:.1f}%"
        return f"{row[col]:.3f}"

    # Groupes de features pour affichage lisible
    _FEAT_GROUPS = {
        "Identification ligne":  ["nom_ligne", "direction_ref"],
        "Temporelles":           ["heure_tranche", "mois", "jour_semaine", "periode_journee", "jour_ferie"],
        "Destination":           ["terminus_encoded"],
        "Météo horaire":         ["meteo_groupe", "precipitation", "snowfall", "wind_speed", "temperature"],
        "Alertes réseau PRIM":   ["categorie_alerte"],
        "Fréquentation arrêt":   ["occupation"],
    }
    feat_set   = set(features)
    feat_lines = []
    for groupe, cols in _FEAT_GROUPS.items():
        actives = [c for c in cols if c in feat_set]
        if actives:
            feat_lines.append(f"  - **{groupe}** : {', '.join(f'`{c}`' for c in actives)}")
    feat_block = "\n".join(feat_lines) if feat_lines else f"  {', '.join(features)}"

    lignes: list[str] = [
        "# Rapport — Comparaison des modèles de régression IDFM\n",
        f"**Date :** {date_fmt[:4]}-{date_fmt[4:6]}-{date_fmt[6:]}  ",
        f"**Heure :** {heure_fmt}  ",
        f"**Source CSV :** `{csv_source}`  ",
        f"**Lignes utilisées :** {n_lignes:,}  ",
        f"**Évaluation :** KFold k={n_folds} (shuffle, random_state=42)  ",
        f"**Hyperparamètres :** GridSearchCV (cv=3, scoring=r²)  \n",
        "---\n",
        "## Features utilisées\n",
        feat_block + "\n",
        "### Détail des features\n",
        "| Feature | Type | Description |",
        "|---------|------|-------------|",
        "| `nom_ligne` | Catégorielle | Nom de la ligne (Métro 1, RER A…) — encodé entier fixe |",
        "| `direction_ref` | Numérique | Sens de la course (1 = aller, 2 = retour) |",
        "| `terminus_encoded` | Numérique | Retard moyen historique du terminus (target encoding) |",
        "| `heure_tranche` | Numérique | Heure de passage (0–23) |",
        "| `mois` | Numérique | Mois de l'année (1–12) |",
        "| `jour_semaine` | Catégorielle | Lundi … Dimanche — encodé entier fixe |",
        "| `periode_journee` | Catégorielle | Pointe matin / Creuse / Méridienne / Pointe soir… — encodé entier fixe |",
        "| `jour_ferie` | Numérique | 1 si jour férié français, 0 sinon |",
        "| `meteo_groupe` | Catégorielle | Groupe météo horaire (ensoleille / pluie / neige / orage…) — encodé entier fixe |",
        "| `precipitation` | Numérique | Précipitations en mm à l'heure du passage |",
        "| `snowfall` | Numérique | Chutes de neige en cm à l'heure du passage |",
        "| `wind_speed` | Numérique | Vitesse du vent (km/h) à l'heure du passage |",
        "| `temperature` | Numérique | Température (°C) à l'heure du passage |",
        "| `categorie_alerte` | Catégorielle | Type d'alerte PRIM active (aucune / greve / incident / travaux…) — encodé entier fixe |",
        "| `occupation` | Numérique | Nombre moyen d'entrées/heure à l'arrêt (données IDFM) |\n",
        "---\n",
        "## Classements par métrique\n",
        f"> Métriques calculées par validation croisée KFold k={n_folds}.",
        "> Les valeurs sont exprimées sous la forme **moyenne ± écart-type** sur les folds.\n",
    ]

    configs = [
        ("R²",       False, "R² — variance expliquée",                          "↑ mieux"),
        ("MAE (s)",  True,  "MAE (s) — erreur moyenne absolue",                 "↓ mieux"),
        ("RMSE (s)", True,  "RMSE (s) — pénalise les grandes erreurs",          "↓ mieux"),
        (proche_col, False, f"Proche≤{seuil_proche}s — % prédictions proches",  "↑ mieux"),
    ]

    for col, ascending, titre, sens in configs:
        tri = resultats.sort_values(col, ascending=ascending).reset_index(drop=True)
        lignes.append(f"### {titre} ({sens})\n")
        lignes.append("| Rang | Modèle | Dataset | Valeur (moy ± std) | Meilleurs params |")
        lignes.append("|:----:|--------|---------|-------------------:|-----------------|")
        for rang, row in tri.iterrows():
            lignes.append(
                f"| {rang + 1} | {row['Modèle']} | {row['Dataset']} "
                f"| {_fmt(row, col)} | `{row['Meilleurs params']}` |"
            )
        lignes.append("")

    lignes += [
        "---\n",
        "## Meilleur modèle global (R² moyen)\n",
        f"**[{best['Dataset']}] {best['Modèle']}**\n",
        "| Métrique | Moyenne | Écart-type |",
        "|----------|--------:|----------:|",
        f"| R²       | `{best['R²']:.4f}` | `{best.get('R² ±', float('nan')):.4f}` |",
        f"| MAE (s)  | `{best['MAE (s)']:.1f}` | `{best.get('MAE (s) ±', float('nan')):.1f}` |",
        f"| RMSE (s) | `{best['RMSE (s)']:.1f}` | `{best.get('RMSE (s) ±', float('nan')):.1f}` |",
        f"| MAPE (%) | `{best['MAPE (%)']:.1f}` | `{best.get('MAPE (%) ±', float('nan')):.1f}` |",
        f"| {proche_col} | `{best[proche_col]:.1f}%` | `{best.get(proche_col + ' ±', float('nan')):.1f}%` |",
        f"| Params   | `{best['Meilleurs params']}` | — |\n",
        "---\n",
        "## Glossaire des métriques\n",
        "| Métrique | Description |",
        "|----------|-------------|",
        "| **R²** | Part de variance expliquée. 1 = parfait, 0 = équivalent à prédire la moyenne. |",
        "| **MAE** | Erreur absolue moyenne en secondes. Toutes les erreurs ont le même poids. |",
        "| **RMSE** | Comme MAE mais les grandes erreurs sont amplifiées (mise au carré). |",
        "| **MAPE** | Erreur en % relatif à la valeur réelle. Instable si retard ≈ 0. |",
        f"| **Proche≤{seuil_proche}s** | % de prédictions à moins de {seuil_proche}s de la réalité. |",
        f"| **KFold k={n_folds}** | Validation croisée : données découpées en {n_folds} blocs, chaque bloc sert de test une fois. L'écart-type mesure la stabilité du modèle. |",
    ]

    chemin = os.path.join(dossier, f"rapport_{horodatage}.md")
    with open(chemin, "w", encoding="utf-8") as f:
        f.write("\n".join(lignes))
    return chemin
