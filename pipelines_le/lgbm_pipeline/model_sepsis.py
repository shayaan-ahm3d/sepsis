import os
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import xgboost as xgb
import feature_load as loader
import feature_extraction as cleaner
import feature_plots as plotter
from src.features.feature_engineer import partial_sofa

# List of vitals columns to include in the feature vectors
VITALS_COLUMNS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
LABS_COLUMNS = ['PaCO2']

def create_window_feature_vector(patient_df: pd.DataFrame, window=6):
    """
    Given a single patient's cleaned DataFrame, create rolling window features
    for each row, but only using the specified columns.
    Returns (X_rolled, y_aligned) for that patient.
    """
    y = patient_df["SepsisLabel"]
    
    # Here, we keep all columns except SepsisLabel, then add partialSOFA as a feature
    X_vitals = patient_df.drop(columns=["SepsisLabel"], errors='ignore').copy()
    X_vitals['PartialSOFA'] = partial_sofa(patient_df)

    # Compute rolling stats
    df_mean = X_vitals.rolling(window).mean().add_suffix("_mean")
    df_min  = X_vitals.rolling(window).min().add_suffix("_min")
    df_max  = X_vitals.rolling(window).max().add_suffix("_max")
    df_std  = X_vitals.rolling(window).std().add_suffix("_std")

    # Concatenate all rolling features
    X_rolled = pd.concat([df_mean, df_min, df_max, df_std], axis=1)

    # Drop the initial rows that do not have a full window
    X_rolled = X_rolled.iloc[window-1:].reset_index(drop=True)
    y_aligned = y.iloc[window-1:].reset_index(drop=True)
    
    return X_rolled,y_aligned

# ------------------------------------------------------------------------------
# 1) Load & clean data
patient_dict = loader.loadTrainingData(
    path_pattern='../training_setA/*.psv', 
    max_files=100000
)
cleaned_dict = cleaner.cleanData(patient_dict)

# ------------------------------------------------------------------------------
# 2) Build training set from all patients, then train
all_X = []
all_y = []
for filename, df in cleaned_dict.items():
    X_rolled, y_aligned = create_window_feature_vector(df, window=6)
    if (X_rolled is not None) and (len(X_rolled) > 0):
        all_X.append(X_rolled)
        all_y.append(y_aligned)

X_all = pd.concat(all_X, ignore_index=True)
y_all = pd.concat(all_y, ignore_index=True)

print("Shape of X_all:", X_all.shape)
print("Shape of y_all:", y_all.shape)

# Handle class imbalance
neg_samples, pos_samples = y_all.value_counts()

# Split train/test for local evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, 
    test_size=0.2, 
    random_state=42, 
    shuffle=True
)

# ------------------------------------------------------------------------------
# 3) Train model
model = xgb.XGBClassifier(
    scale_pos_weight=neg_samples / pos_samples,
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=1
)

# ------------------------------------------------------------------------------
# 4) Evaluate locally
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

plotter.plot_roc_auc(model, X_test, y_test)
plotter.plot_most_important_features(model, feature_names=X_train.columns, top_n=10)

# ------------------------------------------------------------------------------
# 5) Create output files for official evaluation
#    Each patient's file will contain two columns: [PredictedProbability, PredictedLabel]
#    The file name matches the patient's original .psv (e.g. p000001.psv).
#    The delimiter should be '|', and no header row is included.
# ------------------------------------------------------------------------------

output_dir = 'src/utility_score/predictions'
os.makedirs(output_dir, exist_ok=True)

for filename, df in cleaned_dict.items():
    # Rebuild the rolled features for this patient
    X_rolled, y_aligned = create_window_feature_vector(df, window=6)

    # If there's no data (or we can't build features), skip
    if X_rolled is None or len(X_rolled) == 0:
        # Optionally, you could create an empty file or pad 
        # to match row count with 0-prob predictions.
        continue

    # Predict probabilities
    y_prob = model.predict_proba(X_rolled)[:, 1]

    # Binarize predictions at threshold=0.5
    y_bin = (y_prob >= 0.5).astype(int)

    # Create a DataFrame for the 2 required columns
    # IMPORTANT: The official code typically expects exactly as many rows 
    # as the label file, which includes the first (window-1) rows we dropped.
    # If you want to match exactly, you must add those missing rows back.
    prediction_df = pd.DataFrame({
        'PredictedProbability': y_prob,
        'PredictedLabel': y_bin
    })

    missing = (6 - 1)
    padding = pd.DataFrame({
        'PredictedProbability': [0.0]*missing,
        'PredictedLabel': [0]*missing
    })
    prediction_df = pd.concat([padding, prediction_df], ignore_index=True)
    # Save to pipe-delimited .psv without headers or index
    out_path = os.path.join(output_dir, f"p{filename}.psv")  # e.g. "predictions/p000001.psv"
    prediction_df.to_csv(out_path, sep='|', index=False, header=True)

print(f"Saved prediction files to: {output_dir}")
