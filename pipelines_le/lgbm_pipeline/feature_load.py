import glob
import os

import numpy as np
import fireducks.pandas as pd
from tqdm import tqdm


def load_data(path_pattern='../../training_set?/*.psv', max_files=None) -> list[pd.DataFrame]:
	psv_files: list[str] = glob.glob(path_pattern)

	if max_files is not None:
		psv_files = psv_files[:max_files]

	patients: list[pd.DataFrame] = []

	for file_path in tqdm(psv_files, desc='Loading files'):
		parquet_path = f"{os.path.splitext(file_path)[0]}.parquet"
		patient_id = os.path.splitext(os.path.basename(file_path))[0]

		if os.path.exists(parquet_path):
			# If parquet file exists, load it directly
			patient = pd.read_parquet(parquet_path, engine="fastparquet")
		else:
			# If not, read the PSV and save as parquet
			patient = pd.read_csv(file_path, sep='|')
			patient.to_parquet(parquet_path)

		multi = pd.MultiIndex.from_product([[patient_id], pd.to_timedelta(patient.index, 'h')], names=['PatientId', 'Time'])
		patient.set_index(multi, inplace=True)

		patients.append(patient)

	return patients