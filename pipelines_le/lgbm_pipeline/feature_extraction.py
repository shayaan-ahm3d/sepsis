import pandas as pd

def clean_data(patients: list[pd.DataFrame]) -> list[pd.DataFrame]:
  cleaned = []
  for patient in patients:
      clean_df = patient.ffill()
      clean_df.fillna(value=1, inplace=True) # doesn't make sense for pH
      cleaned.append(clean_df)
  return cleaned