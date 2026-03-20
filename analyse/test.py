import pandas as pd
df = pd.read_csv('output/dataset_prediction.csv',sep=';')
print(df.columns)
print(df)

df_passages = pd.read_csv('passages_global.csv')
