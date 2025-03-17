import feature_load as loader
import pandas as pd
import pprint
import numpy as np

def cleanData(patient_dict: pd.DataFrame) -> pd.DataFrame:    
  cleaned_dict = {}
  for filename, df in patient_dict.items():
      clean_df = df.copy()
      clean_df.ffill(inplace=True)       # forward fill
      clean_df.fillna(value=1, inplace=True)
      cleaned_dict[filename] = clean_df
  return cleaned_dict


# patient_dict = loader.loadTrainingData(path_pattern='../training_setA/*.psv', max_files=1000)
# cleaned_dict = cleanData(patient_dict)
# some_filename = list(cleaned_dict.keys())[0]
# print(cleaned_dict[some_filename].head(100))