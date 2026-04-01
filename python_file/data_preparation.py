import pandas as pd 

df = pd.read_csv('dataset_predictions/dataset_fusionne.csv')
print(df[df['nom_ligne']=='Metro 14']['retard_sec'])
