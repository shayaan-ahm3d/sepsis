import numpy as np
import pandas as pd
from itertools import chain
from joblib import Parallel, delayed
from tqdm import tqdm

# Feature definitions
A_FEATURES = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
B_FEATURES = [
    'Temp', 'DBP', 'EtCO2', 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2',
    'SaO2', 'AST', 'BUN', 'Calcium', 'Chloride', 'Creatinine', 'Glucose',
    'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Hct', 'Hgb', 'PTT',
    'WBC', 'Platelets'
]


def impute_A_features(df: pd.DataFrame, cols: list, global_means: pd.Series) -> pd.DataFrame:
    """
    Impute missing values in the specified columns of the dataframe.
    For small dataframes (< 3 rows), use the global mean;
    otherwise, attempt linear interpolation, then forward/backward fill.
    """
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

        df[col] = df[col].ffill().bfill()

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

        feat_dict[f"mean_{w}"] = np.mean(padded)
        feat_dict[f"min_{w}"] = np.min(padded)
        feat_dict[f"max_{w}"] = np.max(padded)
        feat_dict[f"median_{w}"] = np.median(padded)
        feat_dict[f"var_{w}"] = np.var(padded)
        feat_dict[f"q95_{w}"] = np.quantile(padded, 0.95)
        feat_dict[f"q99_{w}"] = np.quantile(padded, 0.99)
        feat_dict[f"q05_{w}"] = np.quantile(padded, 0.05)
        feat_dict[f"q01_{w}"] = np.quantile(padded, 0.01)

    return feat_dict


def missingness_sequences_features(series: pd.Series) -> dict:
    """
    Given a Series (possibly containing NaNs), returns a dictionary with:
      - LC_mean and LC_var: mean and variance of all sequence lengths
      - LCV_sum and LCV_var: sum and variance of valid (non-NaN) sequence lengths
    """
    data = series.isna().to_numpy().astype(int)  # 1 if NaN, 0 if not

    if len(data) == 0:
        return {"LC_mean": 0, "LC_var": 0, "LCV_sum": 0, "LCV_var": 0}

    seq_lengths = []
    seq_lengths_valid = []
    current_val = data[0]
    current_len = 1

    for i in range(1, len(data)):
        if data[i] == current_val:
            current_len += 1
        else:
            seq_lengths.append(current_len)
            if current_val == 0:
                seq_lengths_valid.append(current_len)
            current_val = data[i]
            current_len = 1

    # Append the final sequence
    seq_lengths.append(current_len)
    if current_val == 0:
        seq_lengths_valid.append(current_len)

    lc_mean = np.mean(seq_lengths)
    lc_var = np.var(seq_lengths)
    if len(seq_lengths_valid) == 0:
        lcv_sum = 0
        lcv_var = 0
    else:
        lcv_sum = np.sum(seq_lengths_valid)
        lcv_var = np.var(seq_lengths_valid)

    return {"LC_mean": lc_mean, "LC_var": lc_var, "LCV_sum": lcv_sum, "LCV_var": lcv_var}


def missingness_last_5_rows(rows: pd.DataFrame) -> dict:
    """
    For the last 5 rows of a dataframe, compute the mean and variance
    of the sequence lengths of consecutive missing values.
    """
    all_seq_lengths = []

    for _, row_data in rows.iterrows():
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

    l0_sum = np.mean(all_seq_lengths)
    l0_var = np.var(all_seq_lengths)
    return {"L0_sum": l0_sum, "L0_var": l0_var}


def extract_features_for_patient(df: pd.DataFrame) -> dict:
    """
    Extract features for a single patient from the given dataframe.
    Computes sliding window features for A_FEATURES and missingness
    features for B_FEATURES.
    """
    df_copy = df.copy()
    feats = {}

    # Process A_FEATURES
    for col in A_FEATURES:
        arr = df_copy[col].values
        feats[f"{col}_ns_NaN"] = np.isnan(arr).sum()

        sw_feats = sliding_window_features(arr, window_sizes=[5, 11])
        for k, v in sw_feats.items():
            feats[f"{col}_sw_{k}"] = v

        feats[f"{col}_last"] = arr[-1] if len(arr) > 0 else np.nan

    # Process B_FEATURES for missingness sequences
    for col in B_FEATURES:
        seq_feats = missingness_sequences_features(df[col])
        for k, v in seq_feats.items():
            feats[f"{col}_miss_{k}"] = v

    # Process missingness over the last 5 rows
    last_5_rows = df[B_FEATURES].tail(5)
    l0_feats = missingness_last_5_rows(last_5_rows)
    
    for k, v in l0_feats.items():
        feats[f"last_rows_miss_{k}"] = v

    return feats


def extract_features_for_patient_with_windows(patient_id: int, df: pd.DataFrame, global_means: pd.Series) -> list:
    """
    For a single patient, extract features using an expanding window.
    For each row in the patient dataframe, features are extracted from the data
    up to that point.
    """
    num_rows = len(df)
    df_imputed = impute_A_features(df.copy(), A_FEATURES, global_means)
    expanded_features = []

    for i in range(1, num_rows + 1):
        partial_df = df_imputed.iloc[:i]
        
        feat_vector = partial_df.iloc[-1:]
        
        feat_vector.append(extract_features_for_patient(partial_df))
        
        feat_vector['SepsisLabel'] = df["SepsisLabel"].iloc[i - 1]
        feat_vector["patient_id"] = patient_id

        expanded_features.append(feat_vector)

    return expanded_features


def extract_features_with_expanding_window(patient_dict: dict) -> pd.DataFrame:
    """
    Given a dictionary mapping patient IDs to their dataframes,
    computes global means and then extracts features for each patient
    using an expanding window approach. The result is returned as a DataFrame.
    """
    # Combine all patient data for global mean computation
    all_data = pd.concat(patient_dict.values(), axis=0)
    global_means = all_data.mean(numeric_only=True)
    print("Combined data shape:", all_data.shape)

    # Directly create the feature DataFrame from the parallel job's output
    feature_df = pd.DataFrame(chain.from_iterable(
        Parallel(n_jobs=-1, verbose=5)(
            delayed(extract_features_for_patient_with_windows)(pid, df, global_means)
            for pid, df in tqdm(patient_dict.items(), desc="Extracting features with expanding window")
        )
    ))
    print("Final shape of expanded feature DataFrame:", feature_df.shape)
    return feature_df