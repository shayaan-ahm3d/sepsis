import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from plots.feature_plots import plot_roc_auc
from tqdm import tqdm

def create_window_feature_vector(patient_df: pd.DataFrame, window=6):
    """
    Given a single patient's cleaned DataFrame, create rolling window features
    for each row.
    
    Returns:
    - X_rolled: Feature DataFrame with rolling statistics.
    - y_aligned: Corresponding SepsisLabel (0 or 1).
    """
    # Use SepsisLabel (0 or 1) as the target variable
    y = patient_df["SepsisLabel"]
    
    # Drop SepsisLabel and SepsisLabel_delta from features
    X_vitals = patient_df.drop(columns=["SepsisLabel", "SepsisLabel_delta"], errors='ignore').copy()

    # Compute rolling statistics
    df_mean = X_vitals.rolling(window).mean().add_suffix("_mean")
    df_min  = X_vitals.rolling(window).min().add_suffix("_min")
    df_max  = X_vitals.rolling(window).max().add_suffix("_max")
    df_std  = X_vitals.rolling(window).std().add_suffix("_std")

    # Concatenate rolling features
    X_rolled = pd.concat([df_mean, df_min, df_max, df_std], axis=1)

    # Drop initial rows that do not have a full window
    X_rolled = X_rolled.iloc[window-1:].reset_index(drop=True)
    y_aligned = y.iloc[window-1:].reset_index(drop=True)
    
    return X_rolled, y_aligned

def train_and_evaluate_lgbm(patient_dict: dict, window=6):
    """
    Train an LGBM classifier to predict SepsisLabel and evaluate its performance.

    Parameters:
    - patient_dict (dict): Dictionary containing patient DataFrames.
    - window (int): Size of the rolling window for feature extraction.

    Returns:
    - dict: Dictionary containing accuracy and classification report.
    """
    X_all = pd.DataFrame()
    y_all = pd.Series(dtype="int")  # Empty Series to avoid deprecation warning

    # Extract features and labels for each patient
    for patient_id, df in tqdm(patient_dict.items(),desc="Creating Windows"):
        X_patient, y_patient = create_window_feature_vector(df, window=window)
        X_all = pd.concat([X_all, X_patient], ignore_index=True)
        y_all = pd.concat([y_all, y_patient], ignore_index=True)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    # Train LGBM classifier
    lgbm = LGBMClassifier()
    lgbm.fit(X_train, y_train)

    # Predict SepsisLabel (0 or 1) for test set
    y_pred = lgbm.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", report)
    plot_roc_auc(lgbm, X_test, y_test)

    return {
        "accuracy": accuracy,
        "classification_report": report
    }
