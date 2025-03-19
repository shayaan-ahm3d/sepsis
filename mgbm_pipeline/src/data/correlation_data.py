# https://pharmaceutical-journal.com/article/ld/how-to-interpret-arterial-blood-gas-results-2
import pandas as pd
import numpy as np


def test_pH_equation_accuracy(df: pd.DataFrame) -> float:
    """
    Tests how accurately the acid-base equilibrium equation predicts pH from PaCO2 and HCO3.
    
    The equation used is:
         pH = 6.1 + log10([HCO3] / (0.03 * PaCO2))
         
    For each row with complete data (pH, PaCO2, HCO3), the function calculates the predicted pH,
    computes the absolute error compared to the true pH, and then returns the mean absolute error.
    
    Parameters:
      df: pd.DataFrame - DataFrame containing columns 'pH', 'PaCO2', and 'HCO3'.
    
    Returns:
      float: Mean absolute error of the predictions.
    """
    # Filter rows with complete data for pH, PaCO2, and HCO3.
    complete = df[['pH', 'PaCO2', 'HCO3']].dropna()
    if complete.empty:
        print("No complete rows available for testing acid-base equation accuracy.")
        return None
    
    true_pH_values = []
    predicted_pH_values = []
    errors = []
    
    # Loop over each row with complete acid-base data.
    for idx, row in complete.iterrows():
        true_pH = row['pH']
        PaCO2 = row['PaCO2']
        HCO3 = row['HCO3']
        
        # Calculate predicted pH using the equation.
        predicted_pH = 6.1 + np.log10(HCO3 / (0.03 * PaCO2))
        
        true_pH_values.append(true_pH)
        predicted_pH_values.append(predicted_pH)
        errors.append(abs(true_pH - predicted_pH))
    
    # Convert lists to pandas Series.
    true_pH_series = pd.Series(true_pH_values, name='True pH')
    predicted_pH_series = pd.Series(predicted_pH_values, name='Predicted pH')
    error_series = pd.Series(errors, name='Absolute Error')
    

    # Convert lists to pandas Series.
    true_pH_series = pd.Series(true_pH_values, name='True pH')
    predicted_pH_series = pd.Series(predicted_pH_values, name='Predicted pH')
    error_series = pd.Series(errors, name='Absolute Error')
    
    # Calculate descriptive statistics.
    print("Descriptive statistics for True pH values:")
    print(true_pH_series.describe())
    print("\nDescriptive statistics for Predicted pH values:")
    print(predicted_pH_series.describe())
    print("\nDescriptive statistics for Absolute Error:")
    print(error_series.describe())

    mean_abs_error = np.mean(errors)
    print(f"\nMean Absolute Error of pH prediction: {mean_abs_error:.3f}")
    return mean_abs_error