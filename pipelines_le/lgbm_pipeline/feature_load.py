import glob
import os
import fireducks.pandas as pd
from tqdm import tqdm


def load_training_data(path_pattern='../../training_set?/*.psv', max_files=None) -> list[pd.DataFrame]:
	"""
    Loads .psv files. Removes `Unit1`, `Unit2`, `HospAdmTime` & `ICULOS` columns.
    """

	psv_files: list[str] = glob.glob(path_pattern)

	if max_files is not None:
		psv_files = psv_files[:max_files]

	patients: list[pd.DataFrame] = []

	for file_path in tqdm(psv_files, desc='Loading PSV Files'):
		parquet_path = f"{file_path}.parquet"

		if os.path.exists(parquet_path):
			# If parquet file exists, load it directly
			patient = pd.read_parquet(parquet_path)
		else:
			# If not, read the PSV and save as parquet
			patient = pd.read_csv(file_path, sep='|')
			# Remove Unit1, Unit2, HospAdmTime, ICULOS columns
			patient.drop(columns=patient.columns[-5:-1], inplace=True)
			patient.to_parquet(parquet_path)

		patients.append(patient)

	return patients