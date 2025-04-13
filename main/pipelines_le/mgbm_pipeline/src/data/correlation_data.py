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


def evaluate_bilirubin_ratio(patient_dict: dict) -> None:
    """
    For each patient in the dictionary, where each value is a DataFrame with medical data,
    this function:
      1. Filters rows where both 'Bilirubin_total' and 'Bilirubin_direct' are available.
      2. Uses the first row (training row) to compute the ratio:
           ratio = Bilirubin_direct / Bilirubin_total.
      3. For the remaining rows (test rows), predicts Bilirubin_direct using:
           predicted = ratio * Bilirubin_total
         and computes a relative error.
      4. Calculates accuracy for each test row as:
           accuracy = 100 * (1 - relative error)
         (i.e. if predicted equals actual, accuracy is 100%).
      5. Finally, prints:
         - The mean accuracy across all test rows (across all patients)
         - The total number of training rows used (number of patients with at least one complete row)
         - The total number of test rows used.
    
    Parameters:
      patient_dict: dict
          Dictionary where each key is a patient id and each value is a DataFrame of their medical results.
    """
    all_accuracies = []
    total_train = 0
    total_test = 0
    
    # Loop over each patient.
    for patient_id, df in patient_dict.items():
        # Filter rows that have both values.
        complete_rows = df[['Bilirubin_total', 'Bilirubin_direct']].dropna()
        if complete_rows.shape[0] < 2:
            # Skip patients that don't have at least one training row and one test row.
            continue
        
        # Use the first row to compute the ratio.
        first_row = complete_rows.iloc[0]
        if first_row['Bilirubin_total'] == 0:
            continue  # Avoid division by zero.
        ratio = first_row['Bilirubin_direct'] / first_row['Bilirubin_total']
        total_train += 1  # one training row per patient
        
        # Use the remaining rows as test rows.
        test_rows = complete_rows.iloc[1:]
        total_test += test_rows.shape[0]
        
        # Compute predicted Bilirubin_direct and the relative error for each test row.
        # Relative error = abs(predicted - actual) / actual.
        test_rows = test_rows.copy()
        test_rows['Predicted'] = ratio * test_rows['Bilirubin_total']
        # Avoid division by zero: if actual bilirubin_direct is zero, skip.
        test_rows = test_rows[test_rows['Bilirubin_direct'] != 0]
        if test_rows.empty:
            continue
        test_rows['RelError'] = abs(test_rows['Predicted'] - test_rows['Bilirubin_direct']) / test_rows['Bilirubin_direct']
        # Accuracy as percentage.
        test_rows['Accuracy'] = 100 * (1 - test_rows['RelError'])
        all_accuracies.extend(test_rows['Accuracy'].tolist())
    
    if len(all_accuracies) > 0:
        overall_accuracy = np.mean(all_accuracies)
        print(f"Mean accuracy across all test rows: {overall_accuracy:.2f}%")
    else:
        print("No test rows available for evaluation.")
    print(f"Number of rows used to create the ratio (training rows): {total_train}")
    print(f"Number of test rows: {total_test}")

def evaluate_creatinine_BUN_ratio(patient_dict: dict) -> None:
    """
    For each patient (DataFrame) in the patient_dict:
      1. Filter rows where both 'Creatinine' and 'BUN' are present.
      2. Use the first complete row to calculate the ratio:
           ratio = BUN / Creatinine.
      3. For subsequent rows (test rows), predict BUN as:
           Predicted BUN = ratio * Creatinine.
         Then, compute the relative error:
           Relative Error = abs(Predicted BUN - Actual BUN) / Actual BUN.
         Define accuracy as:
           Accuracy = 100 * (1 - Relative Error).
      4. Finally, print the overall mean accuracy across all test rows, 
         the total number of training rows used, and the total number of test rows.
    
    Parameters:
      patient_dict: dict
          Dictionary where each key is a patient id and each value is a DataFrame of their medical data.
    """
    all_accuracies = []
    total_train = 0
    total_test = 0

    # Loop over each patient.
    for patient_id, df in patient_dict.items():
        # Filter rows with both Creatinine and BUN.
        complete_rows = df[['Creatinine', 'BUN']].dropna()
        # Need at least one row to compute ratio and one to test.
        if complete_rows.shape[0] < 2:
            continue

        # Use the first row to compute the ratio.
        first_row = complete_rows.iloc[0]
        # Avoid division by zero.
        if first_row['Creatinine'] == 0:
            continue
        ratio = first_row['BUN'] / first_row['Creatinine']
        total_train += 1  # one training row per patient

        # Use the remaining rows as test rows.
        test_rows = complete_rows.iloc[1:]
        total_test += test_rows.shape[0]

        # Predict BUN for test rows using the ratio.
        test_rows = test_rows.copy()
        test_rows['Predicted_BUN'] = ratio * test_rows['Creatinine']
        # Skip rows where actual BUN is zero to avoid division by zero in error calculation.
        test_rows = test_rows[test_rows['BUN'] != 0]
        if test_rows.empty:
            continue
        test_rows['RelError'] = abs(test_rows['Predicted_BUN'] - test_rows['BUN']) / test_rows['BUN']
        test_rows['Accuracy'] = 100 * (1 - test_rows['RelError'])
        all_accuracies.extend(test_rows['Accuracy'].tolist())

    if len(all_accuracies) > 0:
        overall_accuracy = np.mean(all_accuracies)
        print(f"Mean accuracy across all test rows: {overall_accuracy:.2f}%")
    else:
        print("No test rows available for evaluation.")
    print(f"Number of rows used to create the ratio (training rows): {total_train}")
    print(f"Number of test rows: {total_test}")

def evaluate_creatinine_BUN_ratio_mean(patient_dict: dict) -> None:
    all_accuracies = []
    total_rows_for_ratio = 0
    total_test_rows = 0
    
    for patient_id, df in patient_dict.items():
        # Filter rows with both Creatinine and BUN.
        complete_rows = df[['Creatinine', 'BUN']].dropna()
        if complete_rows.shape[0] < 1:
            continue  # Skip patient if no complete data
        
        # Compute the ratio for each row and then the mean ratio.
        ratios = complete_rows['BUN'] / complete_rows['Creatinine']
        mean_ratio = ratios.mean()
        total_rows_for_ratio += complete_rows.shape[0]
        
        # For each row, predict BUN using the mean ratio.
        complete_rows = complete_rows.copy()
        complete_rows['Predicted_BUN'] = mean_ratio * complete_rows['Creatinine']
        
        # Avoid division by zero when actual BUN is zero.
        complete_rows = complete_rows[complete_rows['BUN'] != 0]
        if complete_rows.empty:
            continue
        
        complete_rows['RelError'] = abs(complete_rows['Predicted_BUN'] - complete_rows['BUN']) / complete_rows['BUN']
        complete_rows['Accuracy'] = 100 * (1 - complete_rows['RelError'])
        
        all_accuracies.extend(complete_rows['Accuracy'].tolist())
        total_test_rows += complete_rows.shape[0]
    
    if len(all_accuracies) > 0:
        overall_accuracy = np.mean(all_accuracies)
        print(f"Mean accuracy across all test rows: {overall_accuracy:.2f}%")
    else:
        print("No test rows available for evaluation.")
    
    print(f"Total rows used for computing mean ratio (training rows): {total_rows_for_ratio}")
    print(f"Total test rows evaluated: {total_test_rows}")
    
    
def evaluate_bilirubin_ratio_mean(patient_dict: dict) -> None:
    all_accuracies = []
    total_train = 0  # Total rows used to compute the mean ratio.
    total_test = 0   # Total rows evaluated for accuracy.
    
    # Loop over each patient.
    for patient_id, df in patient_dict.items():
        # Filter rows that have both Bilirubin_total and Bilirubin_direct.
        complete_rows = df[['Bilirubin_total', 'Bilirubin_direct']].dropna()
        if complete_rows.shape[0] < 1:
            continue  # Skip patients with no complete rows.
        
        # Compute the ratio for each row and then compute the mean ratio.
        ratios = complete_rows['Bilirubin_direct'] / complete_rows['Bilirubin_total']
        mean_ratio = ratios.mean()
        total_train += complete_rows.shape[0]
        
        # Use the mean ratio to predict Bilirubin_direct for each row.
        complete_rows = complete_rows.copy()
        complete_rows['Predicted'] = mean_ratio * complete_rows['Bilirubin_total']
        
        # Skip rows where the actual Bilirubin_direct is zero to avoid division by zero.
        complete_rows = complete_rows[complete_rows['Bilirubin_direct'] != 0]
        if complete_rows.empty:
            continue
        
        # Calculate the relative error and corresponding accuracy.
        complete_rows['RelError'] = abs(complete_rows['Predicted'] - complete_rows['Bilirubin_direct']) / complete_rows['Bilirubin_direct']
        complete_rows['Accuracy'] = 100 * (1 - complete_rows['RelError'])
        
        # Aggregate accuracies and count test rows.
        all_accuracies.extend(complete_rows['Accuracy'].tolist())
        total_test += complete_rows.shape[0]
    
    if len(all_accuracies) > 0:
        overall_accuracy = np.mean(all_accuracies)
        print(f"Mean accuracy across all test rows: {overall_accuracy:.2f}%")
    else:
        print("No test rows available for evaluation.")
    
    print(f"Number of rows used to compute the mean ratio (training rows): {total_train}")
    print(f"Number of test rows evaluated: {total_test}")