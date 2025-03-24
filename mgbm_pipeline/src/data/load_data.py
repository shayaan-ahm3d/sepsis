from pprint import pprint
import pandas as pd
import os
import glob
from tqdm import tqdm


def loadFile(fileName: str) -> pd.DataFrame:
  """Reads in a single .psv file, returns its DataFrame."""
  df = pd.read_csv(fileName, sep='|')
  return df


def loadTrainingData(path_pattern='../training_setA/*.psv', max_files=None, ignore_columns=['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']):
  """
  Loads .psv files, removes the 6 columns before the sepsis label,
  and concatenates into a single DataFrame.
  """
      
  psv_files = glob.glob(path_pattern)
    
  if max_files is not None:
      psv_files = psv_files[:max_files]

  patient_dict = {}

  for file_path in tqdm(psv_files, desc='Loading PSV Files'):
      filename = os.path.basename(file_path)  # e.g. "p000001.psv"
      patient_record = loadFile(file_path)

      patient_record = patient_record.drop(ignore_columns, axis=1)

      patient_dict[filename] = patient_record

  return patient_dict
