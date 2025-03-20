import pandas as pd
import numpy as np

def forwardFillData(patient_df: pd.DataFrame) -> pd.DataFrame:    
  clean_df = patient_df.copy()

  clean_df.ffill(inplace=True)       # forward fill
  clean_df.fillna(value=1, inplace=True)

  return clean_df

def forwardFillMAP(all_df: pd.DataFrame) -> pd.DataFrame:
  map_df = all_df.copy()
  
  mask = map_df['MAP'].isnull() & map_df['SBP'].notnull() & map_df['DBP'].notnull()    
  map_df.loc[mask, 'MAP'] = map_df.loc[mask, 'DBP'] + (map_df.loc[mask, 'SBP'] - map_df.loc[mask, 'DBP']) / 3

  return map_df

def forwardFillDBP(all_df: pd.DataFrame) -> pd.DataFrame:
  dbp_df = all_df.copy()
  
  mask_dbp = dbp_df['DBP'].isnull() & dbp_df['SBP'].notnull() & dbp_df['MAP'].notnull()
  dbp_df.loc[mask_dbp, 'DBP'] = (3 * dbp_df.loc[mask_dbp, 'MAP'] - dbp_df.loc[mask_dbp, 'SBP']) / 2

  return dbp_df


def forwardFillSBP(all_df: pd.DataFrame) -> pd.DataFrame:
  sbp_df = all_df.copy()
  
  mask_sbp = sbp_df['SBP'].isnull() & sbp_df['MAP'].notnull() & sbp_df['DBP'].notnull()
  sbp_df.loc[mask_sbp, 'SBP'] = 3 * sbp_df.loc[mask_sbp, 'MAP'] - 2 * sbp_df.loc[mask_sbp, 'DBP']

  return sbp_df

def forwardFillHasselbalch(all_df: pd.DataFrame) -> pd.DataFrame:
  df = all_df.copy()
  
  # Fill missing pH when PaCO2 and HCO3 are available.
  mask_ph = df['pH'].isnull() & df['PaCO2'].notnull() & df['HCO3'].notnull()
  df.loc[mask_ph, 'pH'] = 6.1 + np.log10(df.loc[mask_ph, 'HCO3'] / (0.03 * df.loc[mask_ph, 'PaCO2']))
  
  # Fill missing PaCO2 when pH and HCO3 are available.
  mask_paco2 = df['PaCO2'].isnull() & df['pH'].notnull() & df['HCO3'].notnull()
  df.loc[mask_paco2, 'PaCO2'] = df.loc[mask_paco2, 'HCO3'] / (0.03 * (10 ** (df.loc[mask_paco2, 'pH'] - 6.1)))
  
  # Fill missing HCO3 when pH and PaCO2 are available.
  mask_hco3 = df['HCO3'].isnull() & df['pH'].notnull() & df['PaCO2'].notnull()
  df.loc[mask_hco3, 'HCO3'] = 0.03 * df.loc[mask_hco3, 'PaCO2'] * (10 ** (df.loc[mask_hco3, 'pH'] - 6.1))

  return df

def backShiftSepsisLabel(patient_df: pd.DataFrame) -> pd.DataFrame:
    shifted_df = patient_df.copy()
    
    shifted_df['SepsisLabel'] = shifted_df['SepsisLabel'].shift(6)
    first_val = patient_df['SepsisLabel'].iloc[0]
    shifted_df['SepsisLabel'].fillna(first_val, inplace=True)
    
    return shifted_df
