# Cell 1: Imports
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb

from data.load_data import loadTrainingData
from plots.feature_plots import plot_roc_auc, plot_confusion_matrix
from features.create_feature_vectors import extract_features_with_expanding_window

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Cell 2: Load data
directories = ['../../../training_setA/', '../../../training_setB/']
max_files = 10000  # Adjust as needed

patient_dict = {}

for directory in directories:
    pattern = os.path.join(directory, "*.psv")
    print(f"\nLoading data from: {pattern} with max_files={max_files}")
    patient_data = loadTrainingData(
        pattern,
        max_files,
        ignore_columns=['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime']
    )
    patient_dict.update(patient_data)

# Cell 3: Create features
feature_df = extract_features_with_expanding_window(patient_dict)
feature_df.head(10)  # Adjust as needed for a quick glance

# Drop non-feature columns if present
for col in ["patient_id", "window_size"]:
    if col in feature_df.columns:
        feature_df.drop(columns=[col], inplace=True, errors="ignore")

# Separate features and target
X = feature_df.drop(columns=["SepsisLabel"], errors="ignore")
y = feature_df["SepsisLabel"]


# Cell 4: Train/test split
neg_samples, pos_samples = y.value_counts()
print(f"Negative samples: {neg_samples}, Positive samples: {pos_samples}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Cell 5: Train model
model = xgb.XGBClassifier(
    random_state=42,
    objective='binary:logistic',
    eval_metric="auc",
    scale_pos_weight=neg_samples / pos_samples
)
model.fit(X_train, y_train, 
          eval_set=[(X_test, y_test)],
          verbose=1)

# Cell 6: Evaluate model
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Plot ROC curve
plot_roc_auc(model, X_test, y_test)

# Plot confusion matrix
plot_confusion_matrix(y_test, y_pred, labels=("No Sepsis", "Sepsis"))

# Print classification metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))