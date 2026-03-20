import pandas as pd

DIR_PREDICT = 'dataset_predictions/'
DIR_OTHER = 'dataset_other/'

# Chargement
df = pd.read_csv(DIR_PREDICT + 'passages_global.csv', low_memory=False)
df_arrets = pd.read_csv(DIR_OTHER + 'arrets .csv', sep=';', low_memory=False)
df_arrets['ArRId'] = df_arrets['ArRId'].astype('Int64')
df_arrets['ZdAId'] = df_arrets['ZdAId'].astype('Int64')

# === MERGE 1 : StopPoint:Q → ArRId (quais individuels) ===
df['_ArRId'] = df['stop_ref'].str.extract(r':Q:(\d+):').astype('Int64')
df = df.merge(df_arrets[['ArRId', 'ArRName']].rename(columns={'ArRId': '_ArRId'}),
              on='_ArRId', how='left')
df['nom_arret'] = df['nom_arret'].fillna(df['ArRName'])
df.drop(columns=['_ArRId', 'ArRName'], inplace=True)

# === MERGE 2 : StopArea:SP → ZdAId (zones d'arrêts, lignes ferroviaires) ===
df['_ZdAId'] = df['stop_ref'].str.extract(r':SP:(\d+):').astype('Int64')
zdaid_to_name = (
    df_arrets.dropna(subset=['ZdAId'])
             .groupby('ZdAId')['ArRName']
             .first()
             .reset_index()
             .rename(columns={'ZdAId': '_ZdAId'})
)
df = df.merge(zdaid_to_name, on='_ZdAId', how='left')
df['nom_arret'] = df['nom_arret'].fillna(df['ArRName'])
df.drop(columns=['_ZdAId', 'ArRName'], inplace=True)

# Sauvegarde
df.to_csv(DIR_PREDICT + 'passages_global.csv', index=False)

total = len(df)
remplis = df['nom_arret'].notna().sum()
print(f"Merge terminé : {remplis} / {total} arrêts remplis ({remplis/total*100:.1f}%)")
print(f"Encore vides  : {total - remplis}")
