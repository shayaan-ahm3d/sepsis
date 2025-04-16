from enum import Enum, auto

from pandas.errors import InvalidColumnName
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
			corr: float = correlation_matrices[method][feature]["SepsisLabel"]
			if abs(corr) > max_corr:
				max_corr = corr
				best_method = method

		# Fill that feature with the method that gave the highest correlation with SepsisLabel
		features_to_fill_methods[feature] = best_method

	return features_to_fill_methods


def mixed_fill(fill_methods_to_patients: dict[FillMethod, list[pl.DataFrame]],
               fill_method_for_features: dict[str, FillMethod]) -> list[pl.DataFrame]:
	patients_mixed: list[pl.DataFrame] = []

	for i in tqdm(range(len(fill_methods_to_patients[FillMethod.FORWARD])), "Performing mixed fill"):
		mixed_fill_df = pl.DataFrame()

		for feature in fill_method_for_features:
			# Select the optimally filled column for the current feature and add it to the DataFrame
			filled_column = fill_methods_to_patients[fill_method_for_features[feature]][i].select(feature)
			mixed_fill_df = mixed_fill_df.with_columns(filled_column)

		# Add the SepsisLabel column from the original patient DataFrame
		mixed_fill_df = mixed_fill_df.with_columns(
			fill_methods_to_patients[FillMethod.FORWARD][i].select("SepsisLabel"))
		patients_mixed.append(mixed_fill_df)

	return patients_mixed


def compute_expanding_min_max(df: pl.DataFrame, columns: list[str] = None) -> pl.DataFrame:
	VITALS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
	LABS = [
		'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
		'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
		'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total',
		'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets'
	]
	DEMOGRAPHICS = ['Age', 'Gender']
	DROPS = ['Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']
	OUTCOME = 'SepsisLabel'

	FEATURES = VITALS + LABS + DEMOGRAPHICS

	if columns is None:
		columns = VITALS + LABS

	for col in columns:
		if col not in df.columns:
			raise IndexError(f"{col} not in DataFrame")

	min_expressions = [df.select(col).to_series().cum_min().alias(f"{col}_min") for col in columns]
	max_expressions = [df.select(col).to_series().cum_max().alias(f"{col}_max") for col in columns]

	# Add all expressions at once
	if min_expressions or max_expressions:
		return df.with_columns(min_expressions + max_expressions)

	return df