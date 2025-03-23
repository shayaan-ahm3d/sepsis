import sys
import os
import pandas as pd
from pprint import pprint

# Import your custom modules. Adjust the module paths as needed.
from data.load_data import loadTrainingData
from plots.feature_plots import plot_missingness
from models.lgbm_impl import train_and_evaluate_lgbm
from data.clean_data import forwardFillMAP, forwardFillDBP, forwardFillSBP, forwardFillHasselbalch,dropColumns
from tqdm import tqdm
from models.xgb_impl import train_and_evaluate_xgb


# Define directories and max_files manually.
directories = ['../../training_setA/', '../../training_setB/']
max_files = None  # Change this to a number (e.g., 1000) if you want to limit the number of files
ignore_columns = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime']

dfs = []
patient_dict = {}

for directory in directories:
    # Build the path pattern for .psv files in the directory.
    pattern = os.path.join(directory, "*.psv")
    print(f"\nLoading data from: {pattern} with max_files={max_files}")
    patient_data = loadTrainingData(pattern, max_files)
    
    patient_dict.update(patient_data)
    

for patient_id, df in tqdm(patient_dict.items(), desc="Filling Confirmed Values:"):
    temp_df = forwardFillMAP(df)
    temp_df = forwardFillDBP(temp_df)
    temp_df = forwardFillSBP(temp_df)
    temp_df = forwardFillHasselbalch(temp_df)
    
#     temp_df = dropColumns(temp_df, ['','',])

    patient_dict[patient_id] = temp_df

print("Starting")
result = train_and_evaluate_xgb(patient_dict, window=6)
