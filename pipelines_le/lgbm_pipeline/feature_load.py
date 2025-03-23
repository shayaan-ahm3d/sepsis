import glob

import pandas as pd
from tqdm import tqdm

def load_training_data(path_pattern='../training_setA/*.psv', max_files=None):
  """
  Loads .psv files, removes the 6 columns before the sepsis label,
  and concatenates into a single DataFrame.
  """
      
  psv_files = glob.glob(path_pattern)
    
  if max_files is not None:
      psv_files = psv_files[:max_files]

  patients = []

  for file_path in tqdm(psv_files, desc='Loading PSV Files'):
      patient_record = pd.read_csv(file_path, sep='|')

    # Remove Unit1, Unit2, HospAdmTime, ICULOS columns
      patient_record = patient_record.drop(patient_record.columns[-5:-1], axis=1)

      patients.append(patient_record)

  return patients
