import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import xgboost as xgb
import feature_load as loader
import feature_extraction as cleaner
import feature_plots as plotter

# sliding window to create feature vector of means min max of continuous features
# Upsample data imbalance sepsis / not-sepsis

def create_window_feature_vector(patient_df: pd.DataFrame, window=6):
  """
  Given a single patient's cleaned DataFrame, create rolling window features
  for each row. Returns (X_rolled, y_aligned) for that patient.
  """

  X = patient_df.drop(columns=["SepsisLabel"])
  y = patient_df["SepsisLabel"]

  df_mean = X.rolling(window).mean().add_suffix("_mean")
  df_min  = X.rolling(window).min().add_suffix("_min")
  df_max  = X.rolling(window).max().add_suffix("_max")
  df_std  = X.rolling(window).std().add_suffix("_std")

  X_rolled = pd.concat([df_mean, df_min, df_max, df_std], axis=1)
  X_rolled = X_rolled.iloc[window-1:].reset_index(drop=True)
  y_aligned = y.iloc[window-1:].reset_index(drop=True)
    
  # Return the rolling features + label for this patient
  return X_rolled, y_aligned

patient_dict = loader.loadTrainingData(path_pattern='../training_setA/*.psv', max_files=100000)
cleaned_dict = cleaner.cleanData(patient_dict)

all_X = []
all_y = []

for filename, df in cleaned_dict.items():
    X_rolled, y_aligned = create_window_feature_vector(df, window=6)
    # Skip if empty or None
    if X_rolled is not None and len(X_rolled) > 0:
        all_X.append(X_rolled)
        all_y.append(y_aligned)

X_all = pd.concat(all_X, ignore_index=True)
y_all = pd.concat(all_y, ignore_index=True)

print("Shape of X_rolled:", X_all.shape)
print("Shape of y_aligned:", y_all.shape)

neg_samples, pos_samples = y_all.value_counts()

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, 
    test_size=0.2, 
    random_state=42, 
    shuffle=True
) 

model = xgb.XGBClassifier(
    scale_pos_weight = neg_samples/pos_samples,
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=1)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

plotter.plot_roc_auc(model, X_test, y_test)