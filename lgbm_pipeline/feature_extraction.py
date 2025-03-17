import feature_load as loader
import pandas as pd
import pprint
import numpy as np

def cleanData(df: pd.DataFrame) -> pd.DataFrame:
  clean_df = df.copy()
  clean_df.ffill(inplace=True)
  clean_df.fillna(value=1, inplace=True)

  return clean_df



df_raw = loader.loadTrainingData(path_pattern='../training_setA/*.psv', max_files=1000)
df_clean = cleanData(df_raw)
print(df_raw.head(10))
print(df_clean.head(10))
print("DataFrame columns:", df_clean.columns)