from enum import Enum, auto

from tqdm import tqdm
import fireducks.pandas as pd

class FillMethod(Enum):
	FORWARD = auto()
	BACKWARD = auto()
	LINEAR = auto()

def fill(patients: list[pd.DataFrame], method=FillMethod.FORWARD) -> list[pd.DataFrame]:
	filled_data: list[pd.DataFrame] = []
	for patient in tqdm(patients, "Filling gaps in patient data"):
		if method == FillMethod.FORWARD:
			forward: pd.DataFrame = patient.ffill(inplace=False)
			filled_data.append(forward.bfill(inplace=False))
		elif method == FillMethod.BACKWARD:
			backward = patient.bfill(inplace=False)
			filled_data.append(backward.ffill(inplace=False))
		elif method == FillMethod.LINEAR:
			interpolated = patient.interpolate(inplace=False)
			forward = interpolated.ffill(inplace=False)
			filled_data.append(forward.bfill(inplace=False))

	return filled_data

def best_fill_method_for_feature(correlation_matrices: dict[FillMethod, pd.DataFrame], features: list[str]) -> dict[str, FillMethod]:
	# Determine the best fill method for each feature
	features_to_fill_methods: dict[str, FillMethod] = {}

	for feature in tqdm(features, "Finding optimal fill methods"):
		max_corr: float = 0
		best_method: FillMethod = FillMethod.FORWARD

		for method in tqdm(FillMethod):
			corr: float = correlation_matrices[method][feature]["SepsisLabel"]
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

	for i in tqdm(range(len(patients)), "Performing mixed fill"):
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

def extract_pre_sepsis_window(patients: list[pd.DataFrame], window_hours: int = 6) -> list[pd.DataFrame]:
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
        window_start = sepsis_onset_time - pd.Timedelta(hours=window_hours)

        # If window starts before patient data, adjust to start of patient data
        if window_start < patient.index[0]:
            window_start = patient.index[0]

        # Extract the window data (up to but not including the sepsis onset point)
        window_data = patient.loc[window_start:sepsis_onset_time]

        result.append(window_data)

    return result