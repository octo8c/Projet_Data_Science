import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "dataset_predictions/new_data.csv"

# Plages horaires à afficher (modifiez cette liste)
HORAIRES = [6, 7, 8, 9, 10, 17, 18, 19, 20]

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig", low_memory=False)

# Filtrer sur les horaires définis
df_filtre = df[df["heure_tranche"].isin(HORAIRES)]

# Compter le nombre de trajets par heure
trajets_par_heure = df_filtre.groupby("heure_tranche").size().reindex(HORAIRES, fill_value=0)

# Histogramme
plt.figure(figsize=(10, 5))
plt.bar(trajets_par_heure.index.astype(str), trajets_par_heure.values, color="steelblue", edgecolor="black")
plt.xlabel("Heure de passage")
plt.ylabel("Nombre de trajets")
plt.title("Nombre de trajets par tranche horaire")
plt.tight_layout()
plt.savefig("trajets_par_horaire.png", dpi=150)
plt.show()
print("Graphique sauvegardé : trajets_par_horaire.png")
