import glob

import pandas as pd
from tqdm import tqdm


def load_training_data(path_pattern='../../training_set?/*.psv', max_files=None) -> list[pd.DataFrame]:
  """
  Loads .psv files. Removes `Unit1`, `Unit2`, `HospAdmTime` & `ICULOS` columns before the `SepsisLabel`
  """
      
  psv_files: list[str] = glob.glob(path_pattern)
    
  if max_files is not None:
      psv_files = psv_files[:max_files]

  patients: list[pd.DataFrame] = []

  for file_path in tqdm(psv_files, desc='Loading PSV Files'):
      patient_record: pd.DataFrame = pd.read_csv(file_path, sep='|')

    # Remove Unit1, Unit2, HospAdmTime, ICULOS columns
      patient_record = patient_record.drop(patient_record.columns[-5:-1], axis=1)

      patients.append(patient_record)

  return patients
