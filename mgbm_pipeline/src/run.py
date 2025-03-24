import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb

from data.load_data import loadTrainingData
from data.helper_data import concat_dict_of_dataframes
from plots.feature_plots import plot_missingness,plot_roc_auc,plot_confusion_matrix
from data.clean_data import forwardFillMAP,forwardFillDBP,forwardFillSBP,forwardFillHasselbalch
from tqdm import tqdm
from features.create_feature_vectors import extract_features_from_patient_dict,extract_features_with_expanding_window

"""
TODO:
•⁠  ⁠upsample tests ><
•⁠  ⁠⁠blue crystal run
•⁠  ⁠⁠plot output matrix ><
•⁠  ⁠⁠custom auc max function
•⁠  ⁠⁠debug vector on 1 patient
•⁠  ⁠⁠convert to Jupyter  notebook
"""


directories = ['../../training_setA/', '../../training_setB/']
max_files = 100  # Change this to a number (e.g., 1000) if you want to limit the number of files

patient_dict = {}

for directory in directories:
    # Build the path pattern for .psv files in the directory.
    pattern = os.path.join(directory, "*.psv")
    print(f"\nLoading data from: {pattern} with max_files={max_files}")
    patient_data = loadTrainingData(pattern, max_files, ignore_columns=['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime'])
    patient_dict.update(patient_data)
    


# for patient_id, df in tqdm(patient_dict.items(), desc="Filling Confirmed Values:"):
#     temp_df = forwardFillMAP(df)
#     temp_df = forwardFillDBP(temp_df)
#     temp_df = forwardFillSBP(temp_df)
#     temp_df = forwardFillHasselbalch(temp_df)

#     patient_dict[patient_id] = temp_df
    
# plot_missingness(concat_dict_of_dataframes(patient_dict))


feature_df = extract_features_with_expanding_window(patient_dict)

# feature_df = extract_features_from_patient_dict(patient_dict)
print(feature_df.head(100))


# Drop non-feature columns if present
if "patient_id" in feature_df.columns:
    feature_df.drop(columns=["patient_id"], inplace=True, errors="ignore")
if "window_size" in feature_df.columns:
    feature_df.drop(columns=["window_size"], inplace=True, errors="ignore")


X = feature_df.drop(columns=["SepsisLabel"], errors="ignore")  # features
y = feature_df["SepsisLabel"]                                  # target

neg_samples, pos_samples = y.value_counts()
print(neg_samples,pos_samples)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss",  # internal XGB metric
    scale_pos_weight = neg_samples / pos_samples
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # probability for positive class

# Calculate ROC AUC
# auc = roc_auc_score(y_test, y_proba)
# print(f"Test AUC: {auc:.4f}")
plot_roc_auc(model,X_test,y_test)
plot_confusion_matrix(y_test, y_pred, labels=("No Sepsis", "Sepsis"))

# Print classification metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))