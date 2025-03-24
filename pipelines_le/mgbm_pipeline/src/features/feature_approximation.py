import pandas as pd
import numpy as np

def estimate_alveolar_oxygen(df: pd.DataFrame, P_atm: float, R: float = 0.8) -> int:
    """
    For rows in the patient DataFrame where 'Temp', 'FiO2', and 'PaCO2' exist,
    this function estimates the water vapor pressure (PH2O) based on the patient's temperature
    using a quadratic fit to reference points, then calculates the alveolar oxygen partial pressure (PAO2)
    using the alveolar gas equation:
    
      PAO2 = FiO2 * (P_atm - PH2O) - (PaCO2 / R)
    
    Parameters:
      df (pd.DataFrame): Patient's DataFrame with at least 'Temp', 'FiO2', and 'PaCO2' columns.
      P_atm (float): Atmospheric pressure (mmHg).
      R (float): Respiratory quotient (default is 0.8).
    
    Returns:
      int: Number of rows with an estimated PAO2 value.
    """
    # Ensure necessary columns are present.
    required_cols = ['Temp', 'FiO2', 'PaCO2']
    if not set(required_cols).issubset(df.columns):
        print("DataFrame is missing one or more required columns: Temp, FiO2, PaCO2.")
        return 0
    
    # Filter rows with complete data.
    subset = df[df['Temp'].notnull() & df['FiO2'].notnull() & df['PaCO2'].notnull()].copy()
    if subset.empty:
        print("No rows with complete data for Temp, FiO2, and PaCO2.")
        return 0
    
    # Fit a quadratic curve to the reference points:
    # (35°C, 42 mmHg), (37°C, 47 mmHg), (40°C, 55 mmHg)
    ref_temps = np.array([35, 37, 40])
    ref_ph2o = np.array([42, 47, 55])
    coeffs = np.polyfit(ref_temps, ref_ph2o, 2)
    
    def estimate_PH2O(temp):
        # Use the quadratic polynomial to estimate PH2O.
        return np.polyval(coeffs, temp)
    
    # Estimate PH2O for each row based on temperature.
    subset['PH2O'] = subset['Temp'].apply(estimate_PH2O)
    
    # Calculate alveolar oxygen partial pressure (PAO2).
    subset['PAO2'] = subset['FiO2'] * (P_atm - subset['PH2O']) - (subset['PaCO2'] / R)
    
    count_estimated = subset.shape[0]
    print(f"Estimated PAO2 for {count_estimated} rows.")
    
    # Optionally, you could merge 'PAO2' back into the original dataframe.
    # For now, we just return the count.
    return count_estimated