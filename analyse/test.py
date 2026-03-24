import pandas as pd

df = pd.read_csv('dataset_predictions/passages_global.csv')
print("Colonnes :", df.columns.tolist())

col = df[df['nom_ligne'] == 'Tram T9']
print(f"\nNombre de passages Tram T9 : {len(col)}")

retard = col['retard_arrivee_sec']
print("Moyenne             :", retard.mean())
print("Pourcentage remplis :", f"{retard.notna().sum() / len(col) * 100:.1f}%")

valeur_max = retard.max()
ligne_max = col[retard == valeur_max][['nom_arret', 'date_course','timestamp_collecte','horaire_arrivee_estime','horaire_arrivee_prevu', 'retard_arrivee_sec']]
print("Valeur max          :", valeur_max)
print("Arrêt(s) avec max   :\n", ligne_max.to_string(index=False))


df_t9 = pd.read_csv('dataset_predictions/passages_t9.csv')
print("Moyenne collecte spécifique",df_t9['retard_arrivee_sec'].mean())
retard_max = df_t9[df_t9['retard_arrivee_sec'] == df_t9['retard_arrivee_sec'].max()]
print(retard_max[['nom_arret', 'date_course', 'timestamp_collecte', 'horaire_arrivee_estime', 'horaire_arrivee_prevu', 'retard_arrivee_sec']].to_string(index=False))

df_tglobal = pd.read_csv('dataset_predictions/passages_tglobal.csv')
print(df_tglobal[df_tglobal['nom_ligne']=='RER A'][['nom_arret', 'date_course','timestamp_collecte','horaire_arrivee_estime','horaire_arrivee_prevu', 'retard_arrivee_sec']])
