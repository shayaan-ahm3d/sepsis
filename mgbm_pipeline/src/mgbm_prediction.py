import sys
import os
import pandas as pd
from pprint import pprint
from data.load_data import loadTrainingData
from plots.feature_plots import plot_missingness

def prepare_data(training_df:pd.DataFrame) -> pd.DataFrame:
  # Graph missingness on each training set - Create a graph function that takes a single training df
  plot_missingness(training_df, title=f"Missing Data")
  # Call a function to shift SepsisLabel 6 places back and pad the start with whatever the first label is
  # Call a function to clean
  return None

def train_model(hospitals_df:pd.DataFrame):
  cleaned_df = prepare_data(hospitals_df)

  return 0
    

def main():
  args = sys.argv[1:]
  
  # If no arguments are provided, use default directories and None for max_files.
  if not args:
    directories = ['../training_setA/', '../training_setB/']
    max_files = None
  else:
    try:
      # Attempt to interpret the last argument as an integer (max_files)
      max_files = int(args[-1])
      directories = args[:-1]
      if not directories:
        raise ValueError("At least one directory must be provided.")
    except ValueError:
      print("Error: The last argument must be an integer representing max_files and at least one directory must be provided.")
      print("Usage: python train.py <directory1> <directory2> ... <max_files>")
      sys.exit(1)

  dfs = []
  for directory in directories:
    # Build the path pattern for .psv files in the directory.
    pattern = os.path.join(directory, "*.psv")
    print(f"\nLoading data from: {pattern} with max_files={max_files}")
    patient_data = loadTrainingData(pattern, max_files)
    
    # Concatenate all DataFrames from the current directory.
    combined_df = pd.concat(patient_data.values(), ignore_index=True)
    dfs.append(combined_df)

  # Concatenate the combined DataFrames from all directories.
  all_data = pd.concat(dfs, ignore_index=True)
  print(f"\nCombined training set shape: {all_data.shape}")

  train_model(all_data)

if __name__ == '__main__':
    main()
