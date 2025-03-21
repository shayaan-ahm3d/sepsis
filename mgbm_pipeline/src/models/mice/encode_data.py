import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer

DEMOGRAPHIC_COLS = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']

def encode_dict_deltas(patient_dict: dict) -> dict:
  encoded_patient_dict = {}
  
  for patient_id, df in tqdm(patient_dict.items(), desc='Encoding patients'):
    df_new = df.copy().reset_index(drop=True)
    
    df_new['patient_id'] = patient_id
    
    if df_new['SepsisLabel'].sum() == 0:
      df_new['SepsisLabel_delta'] = -250
    else:
      first_index = df_new.index[df_new['SepsisLabel'] == 1][0]
      df_new['SepsisLabel_delta'] = df_new.index - first_index
      
    feature_cols = [col for col in df_new.columns 
                    if col not in DEMOGRAPHIC_COLS + ['SepsisLabel', 'SepsisLabel_delta', 'patient_id']]
      
    for col in feature_cols:
      delta1 = []
      delta2 = []
      for i in range(len(df_new)):
        if i == 0:
          delta1.append(0)
        else:
          if pd.notnull(df_new.loc[i, col]) and pd.notnull(df_new.loc[i-1, col]):
            delta1.append(df_new.loc[i, col] - df_new.loc[i-1, col])
          else:
            delta1.append(np.nan)
        if i < 2:
          delta2.append(0)
        else:
          if pd.notnull(df_new.loc[i, col]) and pd.notnull(df_new.loc[i-2, col]):
            delta2.append(df_new.loc[i, col] - df_new.loc[i-2, col])
          else:
            delta2.append(np.nan)
            
      df_new[f'{col}_delta1'] = delta1
      df_new[f'{col}_delta2'] = delta2

      encoded_patient_dict[patient_id] = df_new

  return encoded_patient_dict

def merge_patient_dict(patient_dict: dict, sort_column: str) -> pd.DataFrame:
    merged_df = pd.concat(list(patient_dict.values()), ignore_index=True)
    merged_df.sort_values(by=sort_column, inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    
    return merged_df
  
def impute_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    delta_cols = [col for col in df.columns if col.endswith('_delta1') or col.endswith('_delta2')]
    if not delta_cols:
        return df
    
    imputer = IterativeImputer(random_state=42, max_iter=10, initial_strategy='mean', verbose=2)
    
    imputed_values = imputer.fit_transform(df[delta_cols])
    
    df_imputed = df.copy()
    df_imputed[delta_cols] = imputed_values
    return df_imputed
  
def split_and_restructure(df: pd.DataFrame) -> dict:
  patient_dict = {}

  for patient_id, group in tqdm(df.groupby('patient_id'), desc='Rebuild Dictionary'):
      if 'ICULOS' in group.columns:
          group = group.sort_values(by='ICULOS').reset_index(drop=True)
          
          if group['ICULOS'].max() != len(group):
              print(f"Warning: For patient {patient_id}, max(ICULOS) ({group['ICULOS'].max()}) != number of rows ({len(group)}).")
      else:
          group = group.reset_index(drop=True)
      group = group.drop(columns=['patient_id'])
      patient_dict[patient_id] = group
      
  return patient_dict
