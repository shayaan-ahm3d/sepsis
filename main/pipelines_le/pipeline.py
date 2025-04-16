from pathlib import Path

import lgbm_pipeline.feature_load as loader
import lgbm_pipeline.feature_extraction as extractor
from lgbm_pipeline.feature_extraction import VITALS, LABS, DEMOGRAPHICS, DROPS, OUTCOME, FEATURES
import mgbm_pipeline.src.features.derive_features as derive

from tqdm import tqdm
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, make_scorer, classification_report, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.utils import shuffle
import xgboost as xgb


def find_project_root(marker=".gitignore") -> Path:
	current = Path.cwd()
	for parent in [current] + list(current.parents):
		if (parent / marker).exists():
			return parent.resolve()
	raise FileNotFoundError(f"Project root marker '{marker}' not found starting from {current}")


root = find_project_root()

patients: list[pl.DataFrame] = loader.load_data(f"{root}/training_set?/*.psv", max_files=None)

# # Train/test split
# Ensure enough sepsis patient representation in train and test sets

sepsis_patients: list[pl.DataFrame] = []
non_sepsis_patients: list[pl.DataFrame] = []

for patient in tqdm(patients, "Splitting sepsis/non-sepsis patients"):
	if patient.select(pl.any("SepsisLabel")).item():
		sepsis_patients.append(patient)
	else:
		non_sepsis_patients.append(patient)

train_sepsis_patients, test_sepsis_patients = train_test_split(sepsis_patients, random_state=42)
train_non_sepsis_patients, test_non_sepsis_patients = train_test_split(non_sepsis_patients, random_state=42)

ratio: float = len(train_non_sepsis_patients) / len(train_sepsis_patients)
print(f"Ratio: {ratio}")

train_patients: list[pl.DataFrame] = train_sepsis_patients + train_non_sepsis_patients
test_patients: list[pl.DataFrame] = test_sepsis_patients + test_non_sepsis_patients

print(f"Number of sepsis patients in training set: {len(train_sepsis_patients)}")
print(f"Number of non-sepsis patients in training set: {len(train_non_sepsis_patients)}")
print(f"Number of patients in training set: {len(train_patients)}\n")
print(f"Number of sepsis patients in testing set: {len(test_sepsis_patients)}")
print(f"Number of non-sepsis patients in testing set: {len(test_non_sepsis_patients)}")
print(f"Number of patients in testing set: {len(test_patients)}")

# # Data imputation

train_patients_forward = extractor.fill(train_patients, extractor.FillMethod.FORWARD)
train_patients_backward = extractor.fill(train_patients, extractor.FillMethod.BACKWARD)
train_patients_linear = extractor.fill(train_patients, extractor.FillMethod.LINEAR)

fill_to_list: dict[extractor.FillMethod, list[pl.DataFrame]] = {
	extractor.FillMethod.FORWARD : train_patients_forward,
	extractor.FillMethod.BACKWARD: train_patients_backward,
	extractor.FillMethod.LINEAR  : train_patients_linear,
}

fill_to_concat: dict[extractor.FillMethod, pl.DataFrame] = {
	extractor.FillMethod.FORWARD : pl.concat(train_patients_forward, how="vertical"),
	extractor.FillMethod.BACKWARD: pl.concat(train_patients_backward, how="vertical"),
	extractor.FillMethod.LINEAR  : pl.concat(train_patients_linear, how="vertical"),
}

fill_to_corr = {
	extractor.FillMethod.FORWARD : fill_to_concat[extractor.FillMethod.FORWARD].to_pandas().corr(),
	extractor.FillMethod.BACKWARD: fill_to_concat[extractor.FillMethod.BACKWARD].to_pandas().corr(),
	extractor.FillMethod.LINEAR  : fill_to_concat[extractor.FillMethod.LINEAR].to_pandas().corr(),
}

fill_methods_to_use: dict[str, extractor.FillMethod] = extractor.best_fill_method_for_feature(fill_to_corr, FEATURES)
train_patients_mixed: list[pl.DataFrame] = extractor.mixed_fill(fill_to_list, fill_methods_to_use)

test_patients_forward: list[pl.DataFrame] = extractor.fill(test_patients, extractor.FillMethod.FORWARD)
test_patients_backward: list[pl.DataFrame] = extractor.fill(test_patients, extractor.FillMethod.BACKWARD)
test_patients_linear: list[pl.DataFrame] = extractor.fill(test_patients, extractor.FillMethod.LINEAR)

fill_method_to_test_patients: dict[extractor.FillMethod, list[pl.DataFrame]] = {
	extractor.FillMethod.FORWARD : test_patients_forward,
	extractor.FillMethod.BACKWARD: test_patients_backward,
	extractor.FillMethod.LINEAR  : test_patients_linear,
}

test_patients_mixed: list[pl.DataFrame] = extractor.mixed_fill(fill_method_to_test_patients, fill_methods_to_use)

# # Down sample non-sepsis patient data

# mixed_sepsis = []
# mixed_non_sepsis = []
#
# for patient in tqdm(train_patients_mixed, "Splitting sepsis/non-sepsis patients"):
# 	if patient.select(pl.any("SepsisLabel")).item():
# 		mixed_sepsis.append(patient)
# 	else:
# 		mixed_non_sepsis.append(patient)
#
# mixed_non_sepsis = shuffle(mixed_non_sepsis, random_state=42, n_samples=2*len(mixed_sepsis))
# final_train = mixed_non_sepsis + mixed_sepsis

train_mixed = derive.compute_derived_features_polars(pl.concat(train_patients_mixed, how="vertical"))
train_mixed = extractor.compute_expanding_min_max(train_mixed)
train_mixed = extractor.compute_sliding_stats(train_mixed)
test_mixed = derive.compute_derived_features_polars(pl.concat(test_patients_mixed, how="vertical"))
test_mixed = extractor.compute_expanding_min_max(test_mixed)
test_mixed = extractor.compute_sliding_stats(test_mixed)

X_train = train_mixed.drop("SepsisLabel")
y_train = train_mixed.select("SepsisLabel").to_series()
X_test = test_mixed.drop("SepsisLabel")
y_test = test_mixed.select("SepsisLabel").to_series()

f = make_scorer(fbeta_score, beta=5.5)

clf = xgb.XGBClassifier(objective="binary:logistic", eval_metric=f, scale_pos_weight=ratio)
bst = clf.fit(X_train, y_train)

y_pred = bst.predict(X_test)

print(classification_report(y_test, y_pred))

for method in tqdm(extractor.FillMethod, "Training on different fills"):
	if method == extractor.FillMethod.FORWARD:
		print("Forward")
		train = derive.compute_derived_features_polars(pl.concat(train_patients_forward, how="vertical"))
		test = derive.compute_derived_features_polars(pl.concat(test_patients_forward, how="vertical"))
	elif method == extractor.FillMethod.BACKWARD:
		print("Backward")
		train = derive.compute_derived_features_polars(pl.concat(train_patients_backward, how="vertical"))
		test = derive.compute_derived_features_polars(pl.concat(test_patients_backward, how="vertical"))
	elif method == extractor.FillMethod.LINEAR:
		print("Linear")
		train = derive.compute_derived_features_polars(pl.concat(train_patients_linear, how="vertical"))
		test = derive.compute_derived_features_polars(pl.concat(test_patients_linear, how="vertical"))

	train = extractor.compute_expanding_min_max(train)
	train = extractor.compute_sliding_stats(train)
	test = extractor.compute_expanding_min_max(test)
	test = extractor.compute_sliding_stats(test)

	X_train = train.drop("SepsisLabel")
	y_train = train.select("SepsisLabel").to_series()
	X_test = test.drop("SepsisLabel")
	y_test = test.select("SepsisLabel").to_series()

	bst = clf.fit(X_train, y_train)

	y_pred = bst.predict(X_test)

	print(classification_report(y_test, y_pred))