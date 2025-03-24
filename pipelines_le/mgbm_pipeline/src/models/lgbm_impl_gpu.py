import numpy as np
import pandas as pd
import cupy as cp  # GPU-accelerated NumPy
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from plots.feature_plots import plot_roc_auc
from tqdm import tqdm
from joblib import Parallel, delayed  # For parallel execution

def create_window_feature_vector(patient_df: pd.DataFrame, window=6):
    """
    Optimized rolling window feature computation using CuPy (GPU).
    """
    y = patient_df["SepsisLabel"]
    X_vitals = patient_df.drop(columns=["SepsisLabel", "SepsisLabel_delta"], errors='ignore').copy()

    # Convert to CuPy array for GPU acceleration
    X_cp = cp.array(X_vitals.values)

    # Compute rolling window statistics using CuPy
    df_mean = cp.mean(X_cp.reshape(-1, window, X_cp.shape[1]), axis=1)
    df_min  = cp.min(X_cp.reshape(-1, window, X_cp.shape[1]), axis=1)
    df_max  = cp.max(X_cp.reshape(-1, window, X_cp.shape[1]), axis=1)
    df_std  = cp.std(X_cp.reshape(-1, window, X_cp.shape[1]), axis=1)

    # Convert back to Pandas DataFrame
    X_rolled = pd.DataFrame(cp.asnumpy(np.hstack([df_mean, df_min, df_max, df_std])))

    # Align labels
    y_aligned = y.iloc[window-1:].reset_index(drop=True)

    return X_rolled, y_aligned

def process_patient(patient_id, df, window):
    """
    Wrapper function for parallel processing of each patient.
    """
    return create_window_feature_vector(df, window)

def train_and_evaluate_lgbm(patient_dict: dict, window=6):
    """
    Train an LGBM classifier with GPU acceleration and parallelized data processing.
    """
    # Parallelize feature extraction
    results = Parallel(n_jobs=-1)(delayed(process_patient)(pid, df, window) for pid, df in tqdm(patient_dict.items(), desc="Processing Patients"))

    # Concatenate all results
    X_all, y_all = zip(*results)
    X_all = pd.concat(X_all, ignore_index=True)
    y_all = pd.concat(y_all, ignore_index=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    # Train LGBM classifier with GPU acceleration
    lgbm = LGBMClassifier(device="gpu", gpu_platform_id=0, gpu_device_id=0)
    lgbm.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = lgbm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", report)
    plot_roc_auc(lgbm, X_test, y_test)

    return {"accuracy": accuracy, "classification_report": report}
