import numpy as np
import pandas as pd
from itertools import chain
from joblib import Parallel, delayed
from tqdm import tqdm

from src.features.derive_features import compute_derived_features

# Feature definitions
DEMOGRAPHIC_A = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']
VITALS_A = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
VITALS_B = [
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
    'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
    'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total',
    'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets'
]

def impute_demographics(df):
    existing_cols = [col for col in DEMOGRAPHIC_A if col in df.columns]
    df[existing_cols] = df[existing_cols].ffill().bfill()
    return df

def impute_features(df: pd.DataFrame, columns):
  df[columns] = df[columns].ffill().bfill()
  return df

def sliding_window_features(values: np.ndarray, window_sizes: list = [5, 11]) -> dict:
  """
  Compute statistical features over sliding windows for the given values.
  Returns a dictionary of features for each specified window size.
  """
  feat_dict = {}
  for w in window_sizes:
    if len(values) < w:
      padded = np.concatenate([np.repeat(values[0], w - len(values)), values])
    else:
      padded = values[-w:]  # last w points

    feat_dict[f"{w}_mean"] = np.mean(padded)
    feat_dict[f"{w}_min"] = np.min(padded)
    feat_dict[f"{w}_max"] = np.max(padded)
    feat_dict[f"{w}_median"] = np.median(padded)
    feat_dict[f"{w}_var"] = np.var(padded)
    feat_dict[f"{w}_q95"] = np.quantile(padded, 0.95)
    feat_dict[f"{w}_q05"] = np.quantile(padded, 0.05)
    feat_dict[f'{w}_energy'] = np.sum(np.square(padded))  
    
    if len(padded) > 1:
      x = np.arange(len(padded))
      slope, _ = np.polyfit(x, padded, 1)
      feat_dict[f'{w}_slope'] = slope
    else:
      feat_dict[f'{w}_slope'] = 0.0

  return feat_dict

def extract_features_for_patient(df: pd.DataFrame) -> dict:
  features = {}
  df_copy = df.copy()
  
  for col in VITALS_A:
    sw_feats = sliding_window_features(df_copy[col].values, window_sizes=[5])
    for k, v in sw_feats.items():
      features[f"{col}_{k}"] = v

  return features
  

def extract_features_for_patient_with_windows(patient_id: int, df: pd.DataFrame) -> list:
  num_rows = len(df)
  
  # Impute A_FEATURES before processing.
  df_imputed = impute_features(df.copy(), VITALS_A + VITALS_B)
  df_imputed = impute_demographics(df_imputed)
  expanded_features = []

  for i in range(1, num_rows + 1):
    # Create the expanding window up to row i
    partial_df = df_imputed.iloc[:i]
    # Extract window-based features
    features = extract_features_for_patient(partial_df)
    # Get the last row of the current window (including SepsisLabel and raw values)
    last_row = df_imputed.iloc[i - 1].to_dict()
    derived_features = compute_derived_features(last_row)
    # Combine the extracted features with the last row values
    combined_features = {**features, **last_row, **derived_features}
    combined_features["patient_id"] = patient_id

    expanded_features.append(combined_features)

  return expanded_features


def extract_features_with_expanding_window(patient_dict: dict) -> pd.DataFrame:
  # Directly create the feature DataFrame from the parallel job's output
  feature_df = pd.DataFrame(chain.from_iterable(
      Parallel(n_jobs=-1, verbose=5)(
          delayed(extract_features_for_patient_with_windows)(pid, df)
          for pid, df in tqdm(patient_dict.items(), desc="Extracting features with expanding window")
      )
  ))
  
  # all_features = []
  # for pid, df in tqdm(patient_dict.items(), desc="Extracting features with expanding window"):
  #   patient_features = extract_features_for_patient_with_windows(pid, df)
  #   all_features.extend(patient_features)

  # feature_df = pd.DataFrame(all_features)
  
  print("Final shape of expanded feature DataFrame:", feature_df.shape)
  return feature_df
