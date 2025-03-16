import feature_load as loader
import pandas as pd

def cleanData(df: pd.DataFrame,
               ):

  return 0



df_raw = loader.loadTrainingData(path_pattern='../training_setA/*.psv', max_files=1000)
print("DataFrame columns:", df_raw.columns)