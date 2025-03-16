from pprint import pprint
import pandas as pd
import os
import glob
from tqdm import tqdm


def loadFile(fileName: str) -> pd.DataFrame:
  """Reads in a single .psv file, returns its DataFrame."""
  df = pd.read_csv(fileName, sep='|')
  return df


def loadTrainingData(path_pattern='../training_setA/*.psv', max_files=None):
  """
  Loads .psv files, removes the 6 columns before the sepsis label,
  and concatenates into a single DataFrame.
  """
      
  psv_files = glob.glob(path_pattern)
    
  if max_files is not None:
      psv_files = psv_files[:max_files]

  training_df = pd.DataFrame()

  for file in tqdm(psv_files, desc='Loading PSV Files'):
    patient_record = loadFile(file)
    patient_record = patient_record.drop(patient_record.columns[-7:-1], axis=1)
    training_df = pd.concat([training_df, patient_record], ignore_index=True)

  return training_df
