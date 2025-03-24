import pandas as pd

def save_feature_data(feature_df, file_path="feature_data.pkl"):
    """
    Saves the feature DataFrame to disk as a pickle file.
    
    Parameters:
    - feature_df: pd.DataFrame, the features DataFrame to be saved.
    - file_path: str, the path (including filename) where the DataFrame will be saved.
    """
    feature_df.to_pickle(file_path)
    print(f"Feature data saved to {file_path}")

def load_feature_data(file_path="feature_data.pkl"):
    """
    Loads the feature DataFrame from disk.
    
    Parameters:
    - file_path: str, the path to the saved feature DataFrame.
    
    Returns:
    - feature_df: pd.DataFrame, the loaded features.
    """
    feature_df = pd.read_pickle(file_path)
    print(f"Feature data loaded from {file_path}")
    return feature_df
