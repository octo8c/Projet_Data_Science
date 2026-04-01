import pandas as pd
from datetime import datetime

DIR_PREDICT = 'dataset_predictions/'
DIR_OTHER = 'dataset_other/'

# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────

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

df = pd.read_csv(DIR_PREDICT + 'passages_global.csv', low_memory=False)
df_arrets = pd.read_csv(DIR_OTHER + 'arrets .csv', sep=';', low_memory=False)
df_arrets['ArRId'] = df_arrets['ArRId'].astype('Int64')
df_arrets['ZdAId'] = df_arrets['ZdAId'].astype('Int64')

# ─────────────────────────────────────────────
# 1. NOM_ARRET — merge depuis référentiel arrêts
# ─────────────────────────────────────────────

# Merge 1 : StopPoint:Q et StopPoint:BP → ArRId
df['_ArRId'] = df['stop_ref'].str.extract(r':(?:Q|BP):(\d+):').astype('Int64')
df = df.merge(df_arrets[['ArRId', 'ArRName']].rename(columns={'ArRId': '_ArRId'}),
              on='_ArRId', how='left')
df['nom_arret'] = df['nom_arret'].fillna(df['ArRName'])
df.drop(columns=['_ArRId', 'ArRName'], inplace=True)

# Merge 2 : StopArea:SP → ZdAId (RER / Transilien)
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

# Fallback nom_ligne depuis line_ref si vide
df['nom_ligne'] = df['nom_ligne'].fillna(df['line_ref'].map(LIGNES_PAR_REF))

nom_arret_remplis = df['nom_arret'].notna().sum()

# ─────────────────────────────────────────────
# 2. DATE_COURSE — depuis timestamp_collecte
# ─────────────────────────────────────────────

df['date_course'] = df['date_course'].fillna(
    pd.to_datetime(df['timestamp_collecte']).dt.date.astype(str)
)

date_remplis = df['date_course'].notna().sum()

# ─────────────────────────────────────────────
# 3. ORDRE_ARRET — depuis GTFS stop_times.txt
# ─────────────────────────────────────────────

st = pd.read_csv(DIR_OTHER + 'stop_times.txt', low_memory=False,
                 usecols=['trip_id', 'stop_id', 'stop_sequence'])

# Extraire line_code (ex: C01371) et stop_num depuis le GTFS
st['line_code'] = st['trip_id'].str.extract(r'-(C\d+)-')
st['stop_num']  = st['stop_id'].str.extract(r'IDFM:(\d+)').astype('Int64')
st = st.dropna(subset=['line_code', 'stop_num'])

# Table de référence : ordre modal par (ligne, arrêt)
ref_ordre = (
    st.groupby(['line_code', 'stop_num'])['stop_sequence']
      .agg(lambda x: x.mode().iloc[0])
      .reset_index()
      .rename(columns={'stop_sequence': 'ordre_gtfs'})
)

# Clés de jointure dans passages
df['line_code'] = df['line_ref'].str.extract(r'::(C\d+):')
df['stop_num']  = df['stop_ref'].str.extract(r':(?:Q|BP):(\d+):').astype('Int64')

df = df.merge(ref_ordre, on=['line_code', 'stop_num'], how='left')
df['ordre_arret'] = df['ordre_arret'].fillna(df['ordre_gtfs'])
df.drop(columns=['line_code', 'stop_num', 'ordre_gtfs'], inplace=True)

ordre_remplis = df['ordre_arret'].notna().sum()

# ─────────────────────────────────────────────
# SAUVEGARDE avec date dans le nom de fichier
# ─────────────────────────────────────────────

date_str = datetime.today().strftime('%Y-%m-%d')
output_file = DIR_PREDICT + f'passages_global_{date_str}.csv'
df.to_csv(output_file, index=False)

total = len(df)
print(f"Fichier sauvegardé : {output_file}")
print(f"  nom_arret           : {nom_arret_remplis} / {total} ({nom_arret_remplis/total*100:.1f}%)")
print(f"  date_course         : {date_remplis} / {total} ({date_remplis/total*100:.1f}%)")
print(f"  ordre_arret         : {ordre_remplis} / {total} ({ordre_remplis/total*100:.1f}%)")
print(f"  vehicle_journey_ref : non disponible (non retourné par l'API RATP/SNCF)")
