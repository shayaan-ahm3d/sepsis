import pandas as pd
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
from itertools import chain



A_FEATURES = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
B_FEATURES = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
              'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
              'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
              'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
              'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
              'Fibrinogen', 'Platelets']

def impute_A_features(df, cols, global_means):
  if len(df) < 3:
    for col in cols:
      if col in df.columns:
        df[col] = df[col].fillna(global_means[col])
    return df
  
  for col in cols:
    if col not in df.columns:
      continue
        
    if df[col].isna().all():
      df[col] = df[col].fillna(global_means[col])
        
    elif df[col].notna().sum() > 3:
      df[col] = df[col].interpolate(method='linear', limit_direction='both')
    

    df[col] = df[col].ffill()
    df[col] = df[col].bfill()
    
  return df

def sliding_window_features(values, window_sizes=[5, 11]):
    feat_dict = {}
    for w in window_sizes:
        if len(values) < w:
            padded = np.concatenate([np.repeat(values[0], w - len(values)), values])
        else:
            padded = values[-w:]  # last w points

        feat_dict[f"mean_{w}"] = np.mean(padded)
        feat_dict[f"min_{w}"]  = np.min(padded)
        feat_dict[f"max_{w}"]  = np.max(padded)
        feat_dict[f"median_{w}"] = np.median(padded)
        feat_dict[f"var_{w}"] = np.var(padded)
        feat_dict[f"q95_{w}"] = np.quantile(padded, 0.95)
        feat_dict[f"q99_{w}"] = np.quantile(padded, 0.99)
        feat_dict[f"q05_{w}"] = np.quantile(padded, 0.05)
        feat_dict[f"q01_{w}"] = np.quantile(padded, 0.01)

    return feat_dict

def missingness_sequences_features(series):
  """
  Given a Series (possibly containing NaNs),
  returns a dictionary with:
    - mean and variance of all sequence lengths (LC)
    - sum and variance of valid-only sequence lengths (LCV)
    - mean and variance of sequences in last 5 hours (LO)
  """
  data = series.isna().to_numpy().astype(int)  # 1 if NaN, 0 if not
  if len(data) == 0:
    return {
        "LC_mean": 0, "LC_var": 0,
        "LCV_sum": 0, "LCV_var": 0,
    }

  seq_lengths = []
  seq_lengths_valid = []
  current_val = data[0]
  current_len = 1

  for i in range(1, len(data)):
    if data[i] == current_val:
      current_len += 1
    else:
      # store the finished sequence
      seq_lengths.append(current_len)
      if current_val == 0:
        seq_lengths_valid.append(current_len)
      # start a new sequence
      current_val = data[i]
      current_len = 1
      
      
  # append last segment
  seq_lengths.append(current_len)
  if current_val == 0:
    seq_lengths_valid.append(current_len)

  # LC: mean/var of all sequence lengths
  lc_mean = np.mean(seq_lengths)
  lc_var = np.var(seq_lengths)

  # LCV: sum/var of only valid-value sequences
  if len(seq_lengths_valid) == 0:
    lcv_sum = 0
    lcv_var = 0
  else:
    lcv_sum = np.sum(seq_lengths_valid)
    lcv_var = np.var(seq_lengths_valid)

  return {
    "LC_mean": lc_mean, "LC_var": lc_var,
    "LCV_sum": lcv_sum, "LCV_var": lcv_var,
  }
  
def missingness_last_5_rows(rows: pd.DataFrame):
  all_seq_lengths = []

  for idx, row_data in rows.iterrows():
    row_isna = row_data.isna().astype(int).to_numpy()
    
    if len(row_isna) == 0:
      continue
    
    seq_lengths = []
    current_val = row_isna[0]
    current_len = 1
    
    for i in range(1, len(row_isna)):
      if row_isna[i] == current_val:
          current_len += 1
      else:
        seq_lengths.append(current_len)
        current_val = row_isna[i]
        current_len = 1
    
    seq_lengths.append(current_len)
    
    all_seq_lengths.extend(seq_lengths)
  
  if len(all_seq_lengths) == 0:
    return {"L0_sum": 0, "L0_var": 0}

  # Compute sum and variance of all sequence lengths across these rows
  l0_sum = np.mean(all_seq_lengths)
  l0_var = np.var(all_seq_lengths)
    
  return {"L0_sum": l0_sum, "L0_var": l0_var}

def extract_features_for_patient(df):
    df_copy = df.copy()
    feats = {}

    for col in A_FEATURES:
      if col not in df_copy.columns:
        continue

      arr = df_copy[col].values
    
      feats[f"{col}_ns_NaN"] = np.isnan(df[col].values).sum()

      sw_feats = sliding_window_features(arr, window_sizes=[5, 11])
      for k, v in sw_feats.items():
        feats[f"{col}_sw_{k}"] = v

      feats[f"{col}_last"] = arr[-1] if len(arr) > 0 else np.nan

    df_raw = df
    for col in B_FEATURES:
        if col not in df_raw.columns:
            continue

        seq_feats = missingness_sequences_features(df_raw[col])
        for k, v in seq_feats.items():
            feats[f"{col}_miss_{k}"] = v

    last_5_rows = df[B_FEATURES].tail(5)
    l0_feats = missingness_last_5_rows(last_5_rows)
    
    for k, v in l0_feats.items():
      feats[f"last_rows_miss_{k}"] = v

    return feats

# def extract_features_from_patient_dict(patient_dict: dict) -> pd.DataFrame:
#   # Get Split of columns with missingness 
#   # ~Set to AB features for now
  
#   all_training_features = []
  
#   all_data = pd.concat(patient_dict.values(), axis=0)
#   global_means = all_data.mean(numeric_only=True) 
#   print(all_data.shape)

  
#   for patient_id, df in tqdm(patient_dict.items(),desc="extracting features"):
#     feat_dict = extract_features_for_patient(df, global_means)
#     feat_dict["patient_id"] = patient_id  # keep track of which patient
#     all_training_features.append(feat_dict)
    
  
#   feature_df = pd.DataFrame(all_training_features)
#   feature_df.set_index("patient_id", inplace=True)

#   print(feature_df.shape)
#   feature_df.head()
  
#   return feature_df

def extract_features_for_patient_with_windows(patient_id, df, global_means):
    """This function does what your current loop does for a single patient."""
    num_rows = len(df)
    expanded_features = []
    df_imputed = df.copy()

    df_imputed = impute_A_features(df_imputed, A_FEATURES, global_means)
    
    for i in range(1, num_rows + 1):
        partial_df = df_imputed.iloc[:i]
        feat_dict = extract_features_for_patient(partial_df)
        
        sepsis_label_i = df["SepsisLabel"].iloc[i - 1]
        feat_dict["SepsisLabel"] = sepsis_label_i
        feat_dict["patient_id"] = patient_id
        feat_dict["window_size"] = i

        expanded_features.append(feat_dict)

    del df_imputed
    return expanded_features

def extract_features_with_expanding_window(patient_dict: dict) -> pd.DataFrame:  
    # combine all for global means
    all_data = pd.concat(patient_dict.values(), axis=0)
    global_means = all_data.mean(numeric_only=True)
    print(all_data.shape)

    del all_data
    
    # Use joblib to parallelize each patient
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(extract_features_for_patient_with_windows)(pid, df, global_means)
        for pid, df in tqdm(patient_dict.items(), desc="extracting features with expanding window")
    )

    del patient_dict
    # 'results' is a list of lists (one list of dicts per patient). Flatten it:
    all_expanded_features = chain.from_iterable(results)
    
    feature_df = pd.DataFrame(all_expanded_features)
    del all_expanded_features
    print("Final shape of expanded feature DataFrame:", feature_df.shape)
    return feature_df