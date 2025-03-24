from enum import Enum, auto
import pandas as pd

class FillMethod(Enum):
	FORWARD = auto()
	BACKWARD = auto()
	LINEAR = auto()

def fill(patients: list[pd.DataFrame], method=FillMethod.FORWARD) -> list[pd.DataFrame]:
	filled_data = []
	for patient in patients:
		#patient.dropna(axis="columns", how="all", inplace=True)
		if method == FillMethod.FORWARD:
			filled_df = patient.ffill(inplace=False)
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

def mixed_fill(patients: dict[FillMethod, list[pd.DataFrame]]) -> pd.DataFrame:
	all_patients: dict[FillMethod, pd.DataFrame] = {FillMethod.FORWARD: pd.concat(patients[FillMethod.FORWARD]),
	                FillMethod.BACKWARD: pd.concat(patients[FillMethod.BACKWARD]),
	                FillMethod.LINEAR: pd.concat(patients[FillMethod.LINEAR])}

	all_patients_mixed: pd.DataFrame = pd.DataFrame(columns=all_patients[FillMethod.FORWARD].columns)

	correlation_matrices: dict[FillMethod, pd.DataFrame] = {FillMethod.FORWARD: all_patients[FillMethod.FORWARD].corr(),
	                        FillMethod.BACKWARD: all_patients[FillMethod.BACKWARD].corr(),
	                        FillMethod.LINEAR: all_patients[FillMethod.LINEAR].corr()}

	for feature in all_patients_mixed.columns:
		max_corr: float = 0
		best_method: FillMethod = FillMethod.FORWARD

		for method in FillMethod:
			corr: float = correlation(correlation_matrices, method, feature)
			if abs(corr) > max_corr:
				max_corr = corr
				best_method = method

		# fill that feature with the method that gave the highest correlation with SepsisLabel
		all_patients_mixed[feature] = all_patients[best_method][feature]

	return all_patients_mixed