import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report

def create_window_feature_vector(patient_df: pd.DataFrame, window=6):
    """
    Given a single patient's cleaned DataFrame, create rolling window features
    for each row, but only using the specified columns.
    Returns (X_rolled, y_aligned) for that patient.
    """
    y = patient_df["SepsisLabel_delta"]
    
    # Here, we keep all columns except SepsisLabel and SepsisLabel_delta
    X_vitals = patient_df.drop(columns=["SepsisLabel", "SepsisLabel_delta"], errors='ignore').copy()
    
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
    
    return X_rolled, y_aligned
  
def train_and_evaluate_lgbm(patient_dict: dict, window=6):
    """
    Train an LGBM classifier to predict SepsisLabel_delta and evaluate its performance.
    
    Parameters:
    - patient_dict (dict): Dictionary containing patient dataframes.
    - window (int): Size of the rolling window for feature extraction.

    Returns:
    - dict: Dictionary containing accuracy report.
    """
    X_all = pd.DataFrame()
    y_all = pd.Series()

    # Create feature vectors for each patient and concatenate them
    for patient_id, df in patient_dict.items():
        X_patient, y_patient = create_window_feature_vector(df, window=window)
        X_all = pd.concat([X_all, X_patient], ignore_index=True)
        y_all = pd.concat([y_all, y_patient], ignore_index=True)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    # Train LGBM classifier
    lgbm = LGBMClassifier()
    lgbm.fit(X_train, y_train)

    # Predict SepsisLabel_delta for test set
    y_pred_delta = lgbm.predict(X_test)

    # Convert SepsisLabel_delta prediction to SepsisLabel prediction
    y_pred_label = np.where((y_pred_delta > -6), 0.5 + np.minimum(0.5, 7 + y_pred_delta / 16), 0)
    
    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred_label)
    report = classification_report(y_test, y_pred_label)

    # Print and return results
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", report)

    return {
        "accuracy": accuracy,
        "classification_report": report
    }
