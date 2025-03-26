from enum import Enum, auto

from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class FillMethod(Enum):
	FORWARD = auto()
	BACKWARD = auto()
	LINEAR = auto()

def fill(patients: list[pd.DataFrame], method=FillMethod.FORWARD) -> list[pd.DataFrame]:
	filled_data: list[pd.DataFrame] = []
	for patient in tqdm(patients, "Filling gaps in patient data"):
		#patient.dropna(axis="columns", how="all", inplace=True)
		if method == FillMethod.FORWARD:
			filled_df: pd.DataFrame = patient.ffill(inplace=False)
			filled_df.bfill(inplace=True)
			filled_data.append(filled_df)
		elif method == FillMethod.BACKWARD:
			filled_df = patient.bfill(inplace=False)
			filled_df.ffill(inplace=True)
			filled_data.append(filled_df)
		elif method == FillMethod.LINEAR:
			filled_df = patient.interpolate(inplace=False)
			filled_df.ffill(inplace=True)
			filled_df.bfill(inplace=True)
			filled_data.append(filled_df)

	return filled_data

def correlation(correlation_matrices: dict[FillMethod, pd.DataFrame], fill_method: FillMethod, feature: str) -> float:
	return correlation_matrices[fill_method][feature]["SepsisLabel"]

def select_best_fill_methods(patients: dict[FillMethod, list[pd.DataFrame]]) -> dict[str, FillMethod]:
	# Compute patient-by-patient correlation matrices
	correlation_matrices: dict[FillMethod, pd.DataFrame] = {}

	for method in tqdm(FillMethod, "Computing correlation matrices"):
		corr_matrix_sum: pd.DataFrame = pd.DataFrame()
		for patient in patients[method]:
			corr_matrix_sum = corr_matrix_sum.add(patient.corr(), fill_value=0)

		correlation_matrices[method] = corr_matrix_sum.div(len(patients[method]), fill_value=1)
		correlation_matrices[method].fillna(0, inplace=True)

	# Determine the best fill method for each feature
	features_to_fill_methods: dict[str, FillMethod] = {}

	for feature in tqdm(patients[FillMethod.FORWARD][0].columns, "Finding optimal fill methods"):
		max_corr: float = 0
		best_method: FillMethod = FillMethod.FORWARD

		for method in tqdm(FillMethod):
			corr: float = correlation(correlation_matrices, method, feature)
			if abs(corr) > max_corr:
				max_corr = corr
				best_method = method

		# Fill that feature with the method that gave the highest correlation with SepsisLabel
		features_to_fill_methods[feature] = best_method

	return features_to_fill_methods

def mixed_fill(patients: list[pd.DataFrame],
				  patients_forward: list[pd.DataFrame],
				  patients_backward: list[pd.DataFrame],
				  patients_linear: list[pd.DataFrame],
				  fill_methods_to_use: dict[str, FillMethod]) -> list[pd.DataFrame]:
	patients_mixed: list[pd.DataFrame] = []

	for i in tqdm(range(len(patients)), "Doing mixed fill"):
		mixed_fill_df = pd.DataFrame(columns=patients[0].columns)

		for feature in fill_methods_to_use:
			if fill_methods_to_use[feature] == FillMethod.FORWARD:
				mixed_fill_df[feature] = patients_forward[i][feature]
			elif fill_methods_to_use[feature] == FillMethod.BACKWARD:
				mixed_fill_df[feature] = patients_backward[i][feature]
			elif fill_methods_to_use[feature] == FillMethod.LINEAR:
				mixed_fill_df[feature] = patients_linear[i][feature]

		patients_mixed.append(mixed_fill_df)
	return patients_mixed