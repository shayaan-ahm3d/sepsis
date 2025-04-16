import glob
import os

import polars as pl
from tqdm import tqdm

schema = {
	'HR'              : pl.Float32,
	'O2Sat'           : pl.Float32,
	'Temp'            : pl.Float32,
	'SBP'             : pl.Float32,
	'MAP'             : pl.Float32,
	'DBP'             : pl.Float32,
	'Resp'            : pl.Float32,
	'EtCO2'           : pl.Float32,
	'BaseExcess'      : pl.Float32,
	'HCO3'            : pl.Float32,
	'FiO2'            : pl.Float32,
	'pH'              : pl.Float32,
	'PaCO2'           : pl.Float32,
	'SaO2'            : pl.Float32,
	'AST'             : pl.Float32,
	'BUN'             : pl.Float32,
	'Alkalinephos'    : pl.Float32,
	'Calcium'         : pl.Float32,
	'Chloride'        : pl.Float32,
	'Creatinine'      : pl.Float32,
	'Bilirubin_direct': pl.Float32,
	'Glucose'         : pl.Float32,
	'Lactate'         : pl.Float32,
	'Magnesium'       : pl.Float32,
	'Phosphate'       : pl.Float32,
	'Potassium'       : pl.Float32,
	'Bilirubin_total' : pl.Float32,
	'TroponinI'       : pl.Float32,
	'Hct'             : pl.Float32,
	'Hgb'             : pl.Float32,
	'PTT'             : pl.Float32,
	'WBC'             : pl.Float32,
	'Fibrinogen'      : pl.Float32,
	'Platelets'       : pl.Float32,
	'Age'             : pl.Float32,
	'Gender'          : pl.UInt8,
	'Unit1'           : pl.Float32,
	'Unit2'           : pl.Float32,
	'HospAdmTime'     : pl.Float32,
	'ICULOS'          : pl.Float32,
	'SepsisLabel'     : pl.UInt8
}


def load_data(path_pattern='../../training_set?/*.psv', max_files=None) -> list[pl.DataFrame]:
	psv_files: list[str] = glob.glob(path_pattern)

	if max_files is not None:
		psv_files = psv_files[:max_files]

	patients: list[pl.DataFrame] = []

	for file_path in tqdm(psv_files, desc='Loading files'):
		patient_id = os.path.splitext(os.path.basename(file_path))[0]
		try:
			patient = pl.read_csv(file_path, separator='|', schema=schema, null_values='NaN', has_header=True)
			patient = patient.drop(['Unit1', 'Unit2']) # drop hospital designator
			patient = patient.cast({'SepsisLabel': pl.Boolean, 'Gender': pl.Boolean})
			patients.append(patient)
		except TypeError as err:
			raise TypeError(f"{err} occurred parsing {patient_id}")

	return patients