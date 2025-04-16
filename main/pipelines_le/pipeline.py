import lgbm_pipeline.feature_load as loader
import lgbm_pipeline.feature_extraction as extractor
from lgbm_pipeline.feature_extraction import VITALS, LABS, DEMOGRAPHICS, DROPS, OUTCOME, FEATURES
import mgbm_pipeline.src.features.derive_features as derive

from pathlib import Path

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import polars as pl
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, make_scorer, classification_report, \
	ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
import xgboost as xgb


def find_project_root(marker=".gitignore") -> Path:
	current = Path.cwd()
	for parent in [current] + list(current.parents):
		if (parent / marker).exists():
			return parent.resolve()
	raise FileNotFoundError(f"Project root marker '{marker}' not found starting from {current}")


root = find_project_root()

patients: list[pl.DataFrame] = loader.load_data(f"{root}/training_set?/*.psv", max_files=None)

n_folds = 3
skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

# Create patient IDs and corresponding labels for stratification
patient_ids = np.arange(len(patients))
patient_labels = np.array([1 if patient.select(pl.any("SepsisLabel")).item() else 0 for patient in patients])

# Store results from each fold
fold_results = []

# Perform cross-validation
for fold, (train_idx, test_idx) in enumerate(skf.split(patient_ids, patient_labels)):
	print(f"\n--- Fold {fold + 1}/{n_folds} ---")

	# Split patients by fold
	train_patients = [patients[i] for i in train_idx]
	test_patients = [patients[i] for i in test_idx]

	# Count sepsis/non-sepsis in this fold
	train_sepsis_count = sum(1 for p in train_patients if p.select(pl.any("SepsisLabel")).item())
	train_non_sepsis_count = len(train_patients) - train_sepsis_count
	test_sepsis_count = sum(1 for p in test_patients if p.select(pl.any("SepsisLabel")).item())
	test_non_sepsis_count = len(test_patients) - test_sepsis_count

	print(f"Train: {train_sepsis_count} sepsis, {train_non_sepsis_count} non-sepsis")
	print(f"Test: {test_sepsis_count} sepsis, {test_non_sepsis_count} non-sepsis")

	ratio = train_non_sepsis_count / max(train_sepsis_count, 1)
	print(f"Ratio: {ratio}")

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

	# Down sample non-sepsis patient data

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

	f = make_scorer(fbeta_score, beta=4)
	clf = xgb.XGBClassifier(objective="binary:logistic", eval_metric=f, scale_pos_weight=ratio)

	bst = clf.fit(X_train, y_train)

	# Get predicted probabilities that the patient gets sepsis at that time
	y_pred_proba = bst.predict_proba(X_test)[:, 1]

	# Define threshold range to evaluate
	thresholds = np.linspace(0.1, 0.9, 50)
	beta = 4

	# Store metrics for each threshold
	results = []
	for threshold in thresholds:
		y_pred_binary = (y_pred_proba >= threshold).astype(int)

		results.append({
			'threshold': threshold,
			'precision': precision_score(y_test, y_pred_binary),
			'recall'   : recall_score(y_test, y_pred_binary),
			'f1'       : f1_score(y_test, y_pred_binary),
			'fbeta'    : fbeta_score(y_test, y_pred_binary, beta=beta),
		})

	results_df = pd.DataFrame(results)

	# Find threshold that maximizes a certain metric
	best_idx = results_df['fbeta'].idxmax()
	best = results_df.iloc[best_idx]
	# Save fold results
	fold_results.append({
		'fold'     : fold + 1,
		'threshold': best['threshold'],
		'precision': best['precision'],
		'recall'   : best['recall'],
		'f1'       : best['f1'],
		'fbeta'    : best['fbeta'],
	})
	# Print fold results
	print(f"Best threshold: {best['threshold']:.4f}")
	print(f"F-{beta} score: {best['fbeta']:.4f}")

	# Create optimal predictions
	y_pred_optimal = (y_pred_proba >= best['threshold']).astype(int)
	print(classification_report(y_test, y_pred_optimal))

# Calculate average metrics across folds
fold_results_df = pd.DataFrame(fold_results)
print("\n--- Cross-Validation Summary ---")
print(f"Average F-beta: {fold_results_df['fbeta'].mean():.4f} ± {fold_results_df['fbeta'].std():.4f}")
print(f"Average threshold: {fold_results_df['threshold'].mean():.4f} ± {fold_results_df['threshold'].std():.4f}")
print(f"Average precision: {fold_results_df['precision'].mean():.4f} ± {fold_results_df['precision'].std():.4f}")
print(f"Average recall: {fold_results_df['recall'].mean():.4f} ± {fold_results_df['recall'].std():.4f}")
"""
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

	# Get predicted probabilities that the patient gets sepsis at that time
	y_pred_proba = bst.predict_proba(X_test)[:, 1]

	# Define threshold range to evaluate
	thresholds = np.linspace(0.1, 0.9, 50)
	beta = 4

	# Store metrics for each threshold
	results = []
	for threshold in thresholds:
		y_pred_binary = (y_pred_proba >= threshold).astype(int)

		results.append({
			'threshold': threshold,
			'precision': precision_score(y_test, y_pred_binary),
			'recall'   : recall_score(y_test, y_pred_binary),
			'f1'       : f1_score(y_test, y_pred_binary),
			'fbeta'    : fbeta_score(y_test, y_pred_binary, beta=beta),
		})

	results_df = pd.DataFrame(results)

	# Find threshold that maximizes a certain metric
	best_idx = results_df['fbeta'].idxmax()
	best = results_df.iloc[best_idx]
	print(best)

	y_pred_optimal = (y_pred_proba >= best['threshold']).astype(int)

	print(f"Best threshold: {best['threshold']:.4f}")
	print(f"F-{beta} score at optimal threshold: {best['fbeta']:.4f}")
	print(classification_report(y_test, y_pred_optimal))
"""