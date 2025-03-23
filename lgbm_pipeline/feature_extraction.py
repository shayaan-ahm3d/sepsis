from enum import Enum
import pandas as pd

class FillMethod(Enum):
	FORWARD = 1
	BACKWARD = 2
	LINEAR = 3
	MIXED = 4

def clean_data(patients: list[pd.DataFrame], method=FillMethod.FORWARD) -> list[pd.DataFrame]:
	cleaned = []
	for patient in patients:
		if method == FillMethod.FORWARD:
			clean_df = patient.ffill(inplace=False)
			clean_df.bfill(inplace=True)
		elif method == FillMethod.BACKWARD:
			clean_df = patient.bfill(inplace=False)
			clean_df.ffill(inplace=True)
		elif method == FillMethod.LINEAR:
			clean_df = patient.interpolate(inplace=False)
			clean_df.ffill(inplace=True)
			clean_df.bfill(inplace=True)
		clean_df.fillna(value=1, inplace=True) # doesn't make sense for pH
		cleaned.append(clean_df)
	return cleaned