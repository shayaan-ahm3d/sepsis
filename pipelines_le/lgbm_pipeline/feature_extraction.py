from enum import Enum, auto

from tqdm import tqdm
import polars as pl


class FillMethod(Enum):
	FORWARD = auto()
	BACKWARD = auto()
	LINEAR = auto()


def fill(patients: list[pl.DataFrame], method=FillMethod.FORWARD) -> list[pl.DataFrame]:
	filled_data: list[pl.DataFrame] = []
	for patient in tqdm(patients, "Filling gaps in patient data"):
		if method == FillMethod.FORWARD:
			forward: pl.DataFrame = patient.fill_null(strategy="forward")
			filled_data.append(forward.fill_null(strategy="backward"))
		elif method == FillMethod.BACKWARD:
			backward = patient.fill_null(strategy="backward")
			filled_data.append(backward.fill_null(strategy="forward"))
		elif method == FillMethod.LINEAR:
			interpolated = patient.interpolate()
			forward = interpolated.fill_null(strategy="forward")
			filled_data.append(forward.fill_null(strategy="backward"))

	return filled_data


def best_fill_method_for_feature(correlation_matrices, features: list[str]) -> dict[
	str, FillMethod]:
	# Determine the best fill method for each feature
	features_to_fill_methods: dict[str, FillMethod] = {}

	for feature in tqdm(features, "Finding optimal fill methods"):
		max_corr: float = 0
		best_method: FillMethod = FillMethod.FORWARD

		for method in tqdm(FillMethod):
			print(f"Series: {correlation_matrices[method][feature]} = {correlation_matrices[method][feature][-1]}")
			corr: float = correlation_matrices[method][feature][-1]
			if abs(corr) > max_corr:
				max_corr = corr
				best_method = method

		# Fill that feature with the method that gave the highest correlation with SepsisLabel
		features_to_fill_methods[feature] = best_method

	return features_to_fill_methods


def mixed_fill(patients: list[pl.DataFrame],
               fill_methods_to_patients: dict[FillMethod, list[pl.DataFrame]],
               fill_method_for_features: dict[str, FillMethod]) -> list[pl.DataFrame]:
	patients_mixed: list[pl.DataFrame] = []

	for i in tqdm(range(len(patients)), "Performing mixed fill"):
		mixed_fill_df = pl.DataFrame()

		for feature in fill_method_for_features:
			# Select the optimally filled column for the current feature and add it to the DataFrame
			filled_column = fill_methods_to_patients[fill_method_for_features[feature]][i].select(feature)
			mixed_fill_df = mixed_fill_df.with_columns(filled_column)

		patients_mixed.append(mixed_fill_df)

	return patients_mixed


def extract_pre_sepsis_window(patients: list[pl.DataFrame], window_hours: int = 6) -> list[pl.DataFrame]:
	"""
	For patients who develop sepsis, extracts data from the specified hours before SepsisLabel turns to 1.
	For patients who never develop sepsis, returns the original data unchanged.

	Args:
		patients: List of patient DataFrames
		window_hours: Number of hours before sepsis onset to include

	Returns:
		List of DataFrames with either pre-sepsis window data or original data
	"""
	result = []

	for patient in tqdm(patients, "Processing pre-sepsis windows"):
		# Check if patient ever develops sepsis
		if not patient["SepsisLabel"].any():
			# Do nothing for non-sepsis patients
			result.append(patient)
			continue

		# Find the first time SepsisLabel turns to 1
		sepsis_indices = patient[patient["SepsisLabel"] == 1].index
		sepsis_onset_time = sepsis_indices[0]

		# Define the window start time (window_hours before sepsis onset)
		window_start = sepsis_onset_time - pl.Timedelta(hours=window_hours)

		# If window starts before patient data, adjust to start of patient data
		if window_start < patient.index[0]:
			window_start = patient.index[0]

		# Extract the window data (up to but not including the sepsis onset point)
		window_data = patient.loc[window_start:sepsis_onset_time]

		result.append(window_data)

	return result