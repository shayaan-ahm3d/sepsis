import os
import pickle

import lgbm_pipeline.feature_load as loader
import lgbm_pipeline.feature_extraction as extractor

from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, make_scorer, RocCurveDisplay, ConfusionMatrixDisplay, classification_report
import lightgbm as lgbm
import xgboost as xgb



patients: list[pd.DataFrame] = loader.load_training_data(f"../training_set?/*.psv", max_files=None)


sepsis_patients: list[pd.DataFrame] = []
non_sepsis_patients: list[pd.DataFrame] = []

for patient in tqdm(patients, "Converting indices to timedeltas"):
    patient.index = pd.to_timedelta(patient.index, 'h')
    if patient["SepsisLabel"].any():
        sepsis_patients.append(patient)
    else:
        non_sepsis_patients.append(patient)

train_sepsis_patients, test_sepsis_patients = train_test_split(sepsis_patients)
train_non_sepsis_patients, test_non_sepsis_patients = train_test_split(non_sepsis_patients)

ratio: float = len(train_non_sepsis_patients)/len(train_sepsis_patients)
print(f"Ratio: {ratio}")

train_patients: list[pd.DataFrame] = train_sepsis_patients + train_non_sepsis_patients
test_patients: list[pd.DataFrame] = test_sepsis_patients + test_non_sepsis_patients

print(f"Number of sepsis patients in training set: {len(train_sepsis_patients)}")
print(f"Number of non-sepsis patients in training set: {len(train_non_sepsis_patients)}")
print(f"Number of patients in training set: {len(train_patients)}\n")
print(f"Number of sepsis patients in testing set: {len(test_sepsis_patients)}")
print(f"Number of non-sepsis patients in testing set: {len(test_non_sepsis_patients)}")
print(f"Number of patients in testing set: {len(test_patients)}")

train_patients_forward: list[pd.DataFrame] = extractor.fill(train_patients, extractor.FillMethod.FORWARD)
train_patients_backward: list[pd.DataFrame] = extractor.fill(train_patients, extractor.FillMethod.BACKWARD)
train_patients_linear: list[pd.DataFrame] = extractor.fill(train_patients, extractor.FillMethod.LINEAR)

fill_method_to_train_patients: dict[extractor.FillMethod, list[pd.DataFrame]] = {extractor.FillMethod.FORWARD: train_patients_forward,
                              extractor.FillMethod.BACKWARD: train_patients_backward,
							  extractor.FillMethod.LINEAR: train_patients_linear}



fill_methods_to_use = extractor.best_fill_method_for_feature(fill_method_to_train_patients, cor)

test_patients_forward: list[pd.DataFrame] = extractor.fill(test_patients, extractor.FillMethod.FORWARD)
test_patients_backward: list[pd.DataFrame] = extractor.fill(test_patients, extractor.FillMethod.BACKWARD)
test_patients_linear: list[pd.DataFrame] = extractor.fill(test_patients, extractor.FillMethod.LINEAR)

train_patients_mixed = extractor.mixed_fill(train_patients, train_patients_forward, train_patients_backward, train_patients_linear, fill_methods_to_use)
test_patients_mixed = extractor.mixed_fill(test_patients, test_patients_forward, test_patients_backward, test_patients_linear, fill_methods_to_use)

# Find the maximum length of the DataFrames in train_patients_mixed
#max_length = max(len(df) for df in train_patients_mixed)

# Adjust the length of each DataFrame in X_train to match the maximum length and forward-fill missing values
# X_train = []
# y_train = []

# for j in tqdm(range(len(train_patients_mixed)), "Extending indices and splitting into (X_train, y_train)"):
#     train_df = train_patients_mixed[j]
#     # Generate a new index that extends to the maximum length
#     new_index = pd.timedelta_range(start=train_df.index[0], periods=max_length, freq='h')
#     train_df = train_df.reindex(new_index).ffill()  # Reindex to the new index and forward-fill
#     X_train.append(train_df.drop(columns="SepsisLabel", inplace=False))
#     y_train.append(train_df["SepsisLabel"])

# # Adjust the length of each DataFrame in X_test similarly
# X_test = []
# y_test = []

# for k in tqdm(range(len(test_patients_mixed)), "Extending indices and splitting into (X_test, y_test)"):
#     test_df = test_patients_mixed[k]
#     new_index = pd.timedelta_range(start=test_df.index[0], periods=max_length, freq='h')
#     test_df = test_df.reindex(new_index).ffill()
#     X_test.append(test_df.drop(columns="SepsisLabel", inplace=False))
#     y_test.append(test_df["SepsisLabel"])

train = pd.concat(train_patients_mixed)
test = pd.concat(test_patients_mixed)

X_train = train.drop(columns="SepsisLabel", inplace=False)
y_train = train["SepsisLabel"]
X_test = test.drop(columns="SepsisLabel", inplace=False)
y_test = test["SepsisLabel"]

f = make_scorer(fbeta_score, beta=1)

clf = xgb.XGBClassifier(objective="binary:logistic", eval_metric=f, scale_pos_weight=ratio)
bst = clf.fit(X_train, y_train)
y_pred = bst.predict(X_test)

RocCurveDisplay.from_predictions(y_test, y_pred)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
print(classification_report(y_test, y_pred))