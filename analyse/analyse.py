import matplotlib
matplotlib.use('TkAgg')  # Backend interactif sans Qt/Wayland
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# ── Agrégation par ligne ──────────────────────────────────────────────────────
def agregation_ligne(df_passage:pd.DataFrame):
    stats_ligne = (
        df_passage
        .groupby('nom_ligne')['retard_depart_sec']
        .mean()
        .reset_index()
        .rename(columns={'retard_depart_sec': 'retard_moyen'})
        .sort_values('retard_moyen', ascending=False)
    )

    # Lignes avec un nom lisible (exclure codes STIF bruts)
    stats_lisibles = stats_ligne[~stats_ligne['nom_ligne'].str.startswith('STIF')]

    print("\nRetard moyen par ligne nommée (secondes) :")
    print(stats_lisibles.to_string(index=False))

# ── Helpers ───────────────────────────────────────────────────────────────────

PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def etiqueter_barres(ax, barres, valeurs):
    for bar, val in zip(barres, valeurs):
        h = bar.get_height()
        if pd.isna(val) or not np.isfinite(h):
            continue
        offset = 1 if val >= 0 else -3
        va = 'bottom' if val >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                f"{val:+.1f}s", ha='center', va=va, fontsize=8, fontweight='bold')


def graphique_lignes(df_stats, prefixe, titre, fichier):
    """Graphique barres pour toutes les lignes commençant par `prefixe`."""
    sous = df_stats[df_stats['nom_ligne'].str.startswith(prefixe)].copy()
    if sous.empty:
        print(f"\nAucune ligne '{prefixe}' trouvée.")
        return
    couleurs = [PALETTE[i % len(PALETTE)] for i in range(len(sous))]
    fig, ax = plt.subplots(figsize=(max(8, len(sous) * 1.4), 6))
    barres = ax.bar(sous['nom_ligne'], sous['retard_moyen'],
                    color=couleurs, edgecolor='white', width=0.6)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    etiqueter_barres(ax, barres, sous['retard_moyen'].values)
    ax.set_title(titre)
    ax.set_xlabel("Ligne")
    ax.set_ylabel("Retard moyen (secondes)")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(fichier, dpi=150)
    print(f"Graphique sauvegardé : {fichier}")
    plt.show()


def pourcentage_retard(df:pd.DataFrame,seuil_retard):
    lignes = ['RER A', 'RER B', 'RER C', 'RER D', 'RER E']
    pourcentages = []
    for ligne in lignes:
        serie = df[df['nom_ligne'] == ligne]['retard_sec'].dropna()
        if len(serie) == 0:
            pourcentages.append(0.0)
        else:
            pourcentages.append((serie > seuil_retard).sum() / len(serie) * 100)

    # Graphique 1 : global (toutes lignes confondues)
    serie_globale = df['retard_sec'].dropna()
    pct_global = (serie_globale > seuil_retard).sum() / len(serie_globale) * 100 if len(serie_globale) > 0 else 0.0

    _, ax1 = plt.subplots()
    ax1.bar(['Toutes lignes'], [pct_global], color='#6a4c93')
    ax1.set_ylabel('% de trains en retard')
    ax1.set_title(f'% de trains en retard — toutes lignes (seuil : {seuil_retard}s)')
    ax1.set_ylim(0, 100)
    ax1.text(0, pct_global + 1, f'{pct_global:.1f}%', ha='center', fontsize=11)
    plt.tight_layout()
    plt.show()

    # Graphique 2 : par RER côte à côte
    _, ax2 = plt.subplots()
    ax2.bar(lignes, pourcentages, color=['#e63946', '#457b9d', '#2a9d8f', '#e9c46a', '#f4a261'])
    ax2.set_ylabel('% de trains en retard')
    ax2.set_title(f'% de trains en retard par RER (seuil : {seuil_retard}s)')
    ax2.set_ylim(0, 100)
    for i, v in enumerate(pourcentages):
        ax2.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9)
    plt.tight_layout()
    plt.show()
    
def main():
    import argparse,os
    parser = argparse.ArgumentParser(description="Prends un fichier et réalise une analyse dessus")
    parser.add_argument("--csv", type=str, help="Chemin vers le fichier CSV à analyser")
    args = parser.parse_args()
    if not args.csv:
        parser.print_help()
        return
    csv = os.path.abspath(args.csv)
    df_passage = pd.read_csv(csv)
    df_passage['date_course'] = pd.to_datetime(df_passage['date_course'], errors='coerce').dt.date
    df_passage['retard_depart_sec'] = pd.to_numeric(df_passage['retard_depart_sec'], errors='coerce')
    # ── Graphique 1 : Lignes RER ──────────────────────────────────────────────────
    
    # ── Agrégation par ligne ──────────────────────────────────────────────────────
    stats_ligne = (
        df_passage
        .groupby('nom_ligne')['retard_depart_sec']
        .mean()
        .reset_index()
        .rename(columns={'retard_depart_sec': 'retard_moyen'})
        .sort_values('retard_moyen', ascending=False)
    )

    # Lignes avec un nom lisible (exclure codes STIF bruts)
    stats_lisibles = stats_ligne[~stats_ligne['nom_ligne'].str.startswith('STIF')]

    print("\nRetard moyen par ligne nommée (secondes) :")
    print(stats_lisibles.to_string(index=False))
    graphique_lignes(stats_ligne, 'RER',
                 "Retard moyen au départ — Lignes RER\n(négatif = en avance sur l'horaire)",
                 "retard_rer.png")
    graphique_lignes(stats_ligne, 'RER',
                 "Retard moyen au départ — Lignes RER\n(négatif = en avance sur l'horaire)",
                 "retard_rer.png")

# ── Graphique 2 : Lignes Tram ─────────────────────────────────────────────────

    graphique_lignes(stats_ligne, 'Tram',
                     "Retard moyen au départ — Lignes Tram\n(négatif = en avance sur l'horaire)",
                     "retard_tram.png")

# ── Graphique 3 : Top 20 stations générant le plus de retard ──────────────────

    TOP_N = 20
    stats_arret = (
        df_passage[df_passage['retard_depart_sec'] > 0]
        .groupby('nom_arret')['retard_depart_sec']
        .mean()
        .reset_index()
        .rename(columns={'retard_depart_sec': 'retard_moyen'})
        .sort_values('retard_moyen', ascending=False)
        .head(TOP_N)
    )
    print(f"\nTop {TOP_N} stations par retard moyen :")
    print(stats_arret.to_string(index=False))

    fig, ax = plt.subplots(figsize=(14, 7))
    barres = ax.bar(stats_arret['nom_arret'], stats_arret['retard_moyen'],
                    color='#e74c3c', edgecolor='white', width=0.7)
    etiqueter_barres(ax, barres, stats_arret['retard_moyen'].values)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(f"Top {TOP_N} stations générant le plus de retard moyen au départ")
    ax.set_xlabel("Station")
    ax.set_ylabel("Retard moyen (secondes)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("retard_top_stations.png", dpi=150)
    print("Graphique sauvegardé : retard_top_stations.png")
    plt.show()

# ── Graphique 4 : Top 5 stations par ligne (subplots) ────────────────────────

    TOP_LIGNES = 6
    TOP_ARRETS = 5

    top_lignes_names = (
        stats_lisibles[stats_lisibles['retard_moyen'] > 0]
        .head(TOP_LIGNES)['nom_ligne']
        .tolist()
    )

    rows = []
    for ligne in top_lignes_names:
        sub = (
            df_passage[
                (df_passage['nom_ligne'] == ligne) &
                (df_passage['retard_depart_sec'] > 0)
            ]
            .groupby('nom_arret')['retard_depart_sec']
            .mean()
            .nlargest(TOP_ARRETS)
            .reset_index()
            .rename(columns={'retard_depart_sec': 'retard_moyen', 'nom_arret': 'station'})
        )
        sub['ligne'] = ligne
        rows.append(sub)

    if rows:
        df_top = pd.concat(rows, ignore_index=True)
        n = len(top_lignes_names)
        fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 6), sharey=False)
        if n == 1:
            axes = [axes]
        for ax, ligne in zip(axes, top_lignes_names):
            data = df_top[df_top['ligne'] == ligne].sort_values('retard_moyen')
            ax.barh(data['station'], data['retard_moyen'], color='#e74c3c', edgecolor='white')
            xmax = data['retard_moyen'].max()
            for i, (_, row) in enumerate(data.iterrows()):
                ax.text(row['retard_moyen'] + xmax * 0.02, i,
                        f"{row['retard_moyen']:.0f}s", va='center', fontsize=8)
            ax.set_title(ligne, fontsize=10, fontweight='bold')
            ax.set_xlabel("Retard moyen (s)")
            ax.tick_params(axis='y', labelsize=8)
        fig.suptitle(f"Top {TOP_ARRETS} stations par retard — {TOP_LIGNES} lignes les plus en retard",
                     fontsize=12)
        plt.tight_layout()
        plt.savefig("retard_stations_par_ligne.png", dpi=150, bbox_inches='tight')
        print("Graphique sauvegardé : retard_stations_par_ligne.png")
        plt.show()

# ── Graphique 5 : Évolution du retard par date ───────────────────────────────

    TOP_LIGNES_DATE = 5
    top_lignes_date = (
        stats_lisibles[stats_lisibles['retard_moyen'] > 0]
        .head(TOP_LIGNES_DATE)['nom_ligne']
        .tolist()
    )

    stats_date = (
        df_passage[df_passage['nom_ligne'].isin(top_lignes_date)]
        .groupby(['date_course', 'nom_ligne'])['retard_depart_sec']
        .mean()
        .reset_index()
        .rename(columns={'retard_depart_sec': 'retard_moyen'})
    )

    couleur_map = {l: PALETTE[i % len(PALETTE)] for i, l in enumerate(sorted(top_lignes_date))}

    fig, ax = plt.subplots(figsize=(14, 6))
    for ligne in sorted(top_lignes_date):
        data = stats_date[stats_date['nom_ligne'] == ligne].sort_values('date_course')
        if data.empty:
            continue
        ax.plot(data['date_course'].astype(str), data['retard_moyen'],
                marker='o', markersize=4, label=ligne, color=couleur_map[ligne])

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(f"Évolution du retard moyen par date — Top {TOP_LIGNES_DATE} lignes les plus en retard")
    ax.set_xlabel("Date")
    ax.set_ylabel("Retard moyen (secondes)")
    plt.xticks(rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig("retard_par_date.png", dpi=150)
    print("Graphique sauvegardé : retard_par_date.png")
    plt.show()


if __name__ == "__main__":
    main()
