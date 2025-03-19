import pandas as pd

def forwardFillData(patient_dict: pd.DataFrame) -> pd.DataFrame:    
  cleaned_dict = {}
  for filename, df in patient_dict.items():
      clean_df = df.copy()
      clean_df.ffill(inplace=True)       # forward fill
      clean_df.fillna(value=1, inplace=True)
      cleaned_dict[filename] = clean_df
  return cleaned_dict

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