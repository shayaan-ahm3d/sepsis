import numpy as np
import pandas as pd

def count_sepsis_labels(patient_dict: dict) -> None:
    """
    For each patient in the patient_dict (where each value is a DataFrame),
    if the patient has at least one septic label (i.e. 'SepsisLabel' sum > 0),
    record the total number of rows (time steps) and the number of septic labels (1s).
    
    Then, compute and print:
      - The average number of rows for septic patients.
      - The average number of septic labels per septic patient.
      - The minimum and maximum number of septic labels among septic patients.
    
    Parameters:
      patient_dict (dict): Dictionary where keys are patient IDs and values are DataFrames.
    """
    rows_list = []       # Total number of rows for each septic patient.
    sepsis_counts = []   # Count of septic labels for each septic patient.
    patient_ids = []     # IDs corresponding to each septic patient.
    
    for patient_id, df in patient_dict.items():
        # Check if patient has at least one septic label.
        if df['SepsisLabel'].sum() > 0:
            rows_list.append(len(df))
            count = df['SepsisLabel'].sum()
            sepsis_counts.append(count)
            patient_ids.append(patient_id)
    
    if sepsis_counts:
        avg_rows = np.mean(rows_list)
        avg_sepsis = np.mean(sepsis_counts)
        min_sepsis = np.min(sepsis_counts)
        max_sepsis = np.max(sepsis_counts)
        
        # Get the corresponding patient IDs for min and max.
        min_index = sepsis_counts.index(min_sepsis)
        max_index = sepsis_counts.index(max_sepsis)
        patient_id_min = patient_ids[min_index]
        patient_id_max = patient_ids[max_index]
        
        print(f"Average number of rows for septic patients: {avg_rows:.2f}")
        print(f"Average number of septic labels (1's) per septic patient: {avg_sepsis:.2f}")
        print(f"Minimum number of septic labels: {min_sepsis} (Patient ID: {patient_id_min})")
        print(f"Maximum number of septic labels: {max_sepsis} (Patient ID: {patient_id_max})")
    else:
        print("No septic patients found in the dictionary.")

def concat_dict_of_dataframes(df_dict: dict) -> pd.DataFrame:
    concatenated_df = pd.concat(
        [df.assign(patient_id=patient_id) for patient_id, df in df_dict.items()],
        ignore_index=True
    )
    return concatenated_df