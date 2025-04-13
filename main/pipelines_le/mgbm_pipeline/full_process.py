import sys
import os
import pandas as pd
import tqdm as tqdm
import xgboost as xgb
import numpy as np
from pathlib import Path
from src.data.load_data import loadTrainingData
from sklearn.model_selection import train_test_split
from src.features.create_feature_vectors import extract_features_with_expanding_window
from src.plots.feature_plots import plot_roc_auc, plot_confusion_matrix
from sklearn.metrics import classification_report, fbeta_score, confusion_matrix
from sklearn import set_config

# Import your custom modules. Adjust the module paths as needed.

set_config(display="text")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_seq_items', None)
pd.set_option('display.max_rows', None)

# Find Project Root
def find_project_root(marker=".gitignore"):
  current = Path.cwd()
  for parent in [current] + list(current.parents):
    if (parent / marker).exists():
      return parent.resolve()
  raise FileNotFoundError(
    f"Project root marker '{marker}' not found starting from {current}")

root = find_project_root()

# Load Data
directories = [f'{root}/training_setA/', f'{root}/training_setB/']
max_files = 1000  # Adjust as needed

patient_dict = {}

for directory in directories:
  pattern = os.path.join(directory, "*.psv")
  print(f"\nLoading data from: {pattern} with max_files={max_files}")
  patient_data = loadTrainingData(
    pattern,
    max_files,
    ignore_columns=['Unit2', 'Unit1', 'ICULOS', 'HospAdmTime']
  )
  patient_dict.update(patient_data)

# Create Feature Vectors
feature_df = extract_features_with_expanding_window(patient_dict)

# Add/Remove Features
print(feature_df.columns)
print(feature_df.shape)

# Split Sets into Test and Train on Patient ID of Dictionary
patient_groups = {patient_id: group 
          for patient_id, group in feature_df.groupby("patient_id")}

labeled_patients = {}
for patient_id, df in patient_groups.items():
  sepsis_label = "1" if df["SepsisLabel"].any() else "0"
  new_key = f"{patient_id}_{sepsis_label}"
  labeled_patients[new_key] = df

sepsis_count = sum(1 for key in labeled_patients if key.endswith('_1'))
nonsepsis_count = sum(1 for key in labeled_patients if key.endswith('_0'))
print(f"Number of SEPSIS patients: {sepsis_count}")
print(f"Number of NON-SEPSIS patients: {nonsepsis_count}")

keys = list(labeled_patients.keys())
labels = [1 if key.endswith('_sepsis') else 0 for key in keys]

train_keys, test_keys, _, _ = train_test_split(
  keys, labels, test_size=0.2, random_state=42, stratify=labels
)

train_data_dict = {key: labeled_patients[key] for key in train_keys}
test_data_dict = {key: labeled_patients[key] for key in test_keys}

train_sepsis = sum(1 for key in train_data_dict if key.endswith('_1'))
test_sepsis = sum(1 for key in test_data_dict if key.endswith('_1'))
print(f"Train SEPSIS: {train_sepsis}, NON-SEPSIS: {len(train_data_dict) - train_sepsis}")
print(f"Test SEPSIS: {test_sepsis}, NON-SEPSIS: {len(test_data_dict) - test_sepsis}")

train_df = pd.concat(train_data_dict.values(), ignore_index=True)
test_df = pd.concat(test_data_dict.values(), ignore_index=True)

train_df = train_df.drop(columns=['patient_id'])
test_df = test_df.drop(columns=['patient_id'])

# Train Model
X_train = train_df.drop(columns=["SepsisLabel"], errors="ignore")
y_train = train_df["SepsisLabel"]

X_test = test_df.drop(columns=["SepsisLabel"], errors="ignore")
y_test = test_df["SepsisLabel"]

neg_samples, pos_samples = y_train.value_counts()
neg_samples_test, pos_samples_test = y_test.value_counts()
print(f"Negative samples of Train: {neg_samples}, Positive samples of Train: {pos_samples}")
print(f"Negative samples of Test: {neg_samples_test}, Positive samples of Test: {pos_samples_test}")

model = xgb.XGBClassifier(
  random_state=42,
  objective='binary:logistic',
  eval_metric="auc",
  scale_pos_weight=neg_samples / pos_samples
)
model.fit(X_train, y_train, 
      eval_set=[(X_test, y_test)],
      verbose=1)

# Evaluate Model
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Maximise Threshold
thresholds = np.arange(0.0, 1.0, 0.01)

f_beta_scores = []
beta_val = 4.5

for t in thresholds:
  y_pred_threshold = (y_proba >= t).astype(int)
  fb = fbeta_score(y_test, y_pred_threshold, beta=beta_val)
  f_beta_scores.append(fb)

optimal_threshold = thresholds[np.argmax(f_beta_scores)]
print(f"Optimal threshold: {optimal_threshold}, F Beta {beta_val} Score: {max(f_beta_scores)}")

# Re-evaluate Model
y_pred_custom = (y_proba >= optimal_threshold).astype(int)

plot_roc_auc(model, X_test, y_test, optimal_threshold)

print(classification_report(y_test, y_pred_custom))
print(confusion_matrix(y_test, y_pred_custom))

# Feature Importance
feature_importances = model.feature_importances_
features = X_test.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df.sort_values(by='Importance', ascending=False, inplace=True)
print(importance_df.head(100))