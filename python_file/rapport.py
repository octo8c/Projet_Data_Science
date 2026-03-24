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
    test_size: float,
    seuil_proche: int,
) -> str:
    """
    Génère un fichier rapport_<horodatage>.md dans dossier.
    Retourne le chemin du fichier créé.
    """
    proche_col = f"Proche≤{seuil_proche}s (%)"
    best       = resultats.sort_values("R²", ascending=False).iloc[0]
    date_fmt   = horodatage[:8]  # YYYYMMDD
    heure_fmt  = horodatage[9:].replace("-", ":")  # HH:MM:SS

    lignes: list[str] = [
        "# Rapport — Comparaison des modèles de régression IDFM\n",
        f"**Date :** {date_fmt[:4]}-{date_fmt[4:6]}-{date_fmt[6:]}  ",
        f"**Heure :** {heure_fmt}  ",
        f"**Source CSV :** `{csv_source}`  ",
        f"**Lignes utilisées :** {n_lignes:,}  ",
        f"**Features :** {', '.join(features)}  ",
        f"**Split train/test :** {int((1 - test_size) * 100)}/{int(test_size * 100)}  ",
        f"**Hyperparamètres :** GridSearchCV (cv=3, scoring=r²)  \n",
        "---\n",
        "## Classements par métrique\n",
    ]

    configs = [
        ("R²",       False, "R² — variance expliquée",                    "↑ mieux"),
        ("MAE (s)",  True,  "MAE (s) — erreur moyenne absolue",           "↓ mieux"),
        ("RMSE (s)", True,  "RMSE (s) — pénalise les grandes erreurs",    "↓ mieux"),
        (proche_col, False, f"Proche≤{seuil_proche}s — % prédictions proches", "↑ mieux"),
    ]

    for col, ascending, titre, sens in configs:
        tri = resultats.sort_values(col, ascending=ascending).reset_index(drop=True)
        lignes.append(f"### {titre} ({sens})\n")
        lignes.append("| Rang | Modèle | Dataset | Valeur | Meilleurs params |")
        lignes.append("|:----:|--------|---------|-------:|-----------------|")
        for rang, row in tri.iterrows():
            val = f"{row[col]:.1f}%" if col == proche_col else f"{row[col]:.3f}"
            lignes.append(
                f"| {rang + 1} | {row['Modèle']} | {row['Dataset']} | {val} "
                f"| `{row['Meilleurs params']}` |"
            )
        lignes.append("")

    lignes += [
        "---\n",
        "## Meilleur modèle global (R²)\n",
        f"**[{best['Dataset']}] {best['Modèle']}**\n",
        f"| Métrique | Valeur |",
        f"|----------|--------|",
        f"| R²       | `{best['R²']:.4f}` |",
        f"| MAE      | `{best['MAE (s)']:.1f}s` |",
        f"| RMSE     | `{best['RMSE (s)']:.1f}s` |",
        f"| MAPE     | `{best['MAPE (%)']:.1f}%` |",
        f"| Proche   | `{best[proche_col]:.1f}%` |",
        f"| Params   | `{best['Meilleurs params']}` |\n",
        "---\n",
        "## Glossaire des métriques\n",
        "| Métrique | Description |",
        "|----------|-------------|",
        "| **R²** | Part de variance expliquée. 1 = parfait, 0 = équivalent à prédire la moyenne. |",
        "| **MAE** | Erreur absolue moyenne en secondes. Toutes les erreurs ont le même poids. |",
        "| **RMSE** | Comme MAE mais les grandes erreurs sont amplifiées (mise au carré). |",
        "| **MAPE** | Erreur en % relatif à la valeur réelle. Instable si retard ≈ 0. |",
        f"| **Proche≤{seuil_proche}s** | % de prédictions à moins de {seuil_proche}s de la réalité. |",
    ]

    chemin = os.path.join(dossier, f"rapport_{horodatage}.md")
    with open(chemin, "w", encoding="utf-8") as f:
        f.write("\n".join(lignes))
    return chemin
