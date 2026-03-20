import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df_passage = pd.read_csv('passages_global.csv')

# ── 1. Nettoyage ──────────────────────────────────────────────────────────────

df_passage['retard_depart_sec'] = pd.to_numeric(
    df_passage['retard_depart_sec'], errors='coerce'
)
# Les valeurs négatives sont conservées (train en avance = valeur réelle)

# ── 2. Agrégation par ligne ───────────────────────────────────────────────────

stats_ligne = (
    df_passage
    .groupby('nom_ligne')['retard_depart_sec']
    .mean()
    .reset_index()
    .rename(columns={'retard_depart_sec': 'retard_moyen'})
    .sort_values('retard_moyen', ascending=False)
)

print("\nRetard moyen par ligne (secondes) :")
print(stats_ligne.to_string(index=False))

# ── Helper : étiquette + position de texte ────────────────────────────────────

def etiqueter_barres(ax, barres, valeurs):
    for bar, val in zip(barres, valeurs):
        if pd.isna(val):
            label = "N/A"
        else:
            label = f"{val:+.1f}s"          # + devant les positifs, - devant les négatifs
        offset = 1 if val >= 0 else -3
        va     = 'bottom' if val >= 0 else 'top'
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            label,
            ha='center', va=va, fontsize=8, fontweight='bold'
        )

def couleur_barre(val):
    """Vert si en avance, rouge si en retard, gris si NaN."""
    if pd.isna(val):
        return '#aaaaaa'
    return '#e74c3c' if val > 0 else '#2ecc71'

# ── 3. Graphique 1 : lignes RER uniquement (1 couleur par ligne) ──────────────

rer = stats_ligne[stats_ligne['nom_ligne'].str.startswith('RER')].copy()

if rer.empty:
    print("\nAucune ligne RER trouvée.")
else:
    # Palette distincte : 1 couleur par ligne RER
    palette = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
               '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
    couleurs_rer = [palette[i % len(palette)] for i in range(len(rer))]

    fig, ax = plt.subplots(figsize=(10, 6))
    barres = ax.bar(rer['nom_ligne'], rer['retard_moyen'],
                    color=couleurs_rer, edgecolor='white', width=0.6)

    # Ligne de référence à 0
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    etiqueter_barres(ax, barres, rer['retard_moyen'].values)

    ax.set_title("Retard moyen au départ — Lignes RER\n(négatif = en avance sur l'horaire)")
    ax.set_xlabel("Ligne")
    ax.set_ylabel("Retard moyen (secondes)")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig("retard_rer.png", dpi=150)
    plt.show()
    print("Graphique sauvegardé : retard_rer.png")

# ── 4. Graphique 2 : toutes les lignes du dataset ────────────────────────────

toutes = stats_ligne.copy()
couleurs_toutes = [couleur_barre(v) for v in toutes['retard_moyen']]

fig, ax = plt.subplots(figsize=(max(14, len(toutes) * 0.6), 7))
barres = ax.bar(toutes['nom_ligne'], toutes['retard_moyen'],
                color=couleurs_toutes, edgecolor='white', width=0.7)

ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

etiqueter_barres(ax, barres, toutes['retard_moyen'].values)

# Légende manuelle
from matplotlib.patches import Patch
legende = [
    Patch(facecolor='#e74c3c', label='En retard'),
    Patch(facecolor='#2ecc71', label='En avance'),
    Patch(facecolor='#aaaaaa', label='Données manquantes'),
]
ax.legend(handles=legende, loc='upper right')

ax.set_title("Retard moyen au départ — Toutes les lignes\n(négatif = en avance, ex: RER A ≈ −7s)")
ax.set_xlabel("Ligne")
ax.set_ylabel("Retard moyen (secondes)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("retard_toutes_lignes.png", dpi=150)
plt.show()
print("Graphique sauvegardé : retard_toutes_lignes.png")
