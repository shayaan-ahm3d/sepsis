import glob
import os

import numpy as np
import fireducks.pandas as pd
from tqdm import tqdm


def load_data(path_pattern='../../training_set?/*.psv', max_files=None) -> pd.DataFrame:
	psv_files: list[str] = glob.glob(path_pattern)

	if max_files is not None:
		psv_files = psv_files[:max_files]

	patients: list[pd.DataFrame] = []

	for file_path in tqdm(psv_files, desc='Loading files'):
		parquet_path = f"{os.path.splitext(file_path)[0]}.parquet"
		patient_id = os.path.splitext(os.path.basename(file_path))[0]

		if os.path.exists(parquet_path):
			# If parquet file exists, load it directly
			patient = pd.read_parquet(parquet_path)
		else:
			# If not, read the PSV and save as parquet
			patient = pd.read_csv(file_path, sep='|')
			patient.to_parquet(parquet_path)

		multi = pd.MultiIndex.from_product([[patient_id], pd.to_timedelta(patient.index, 'h')], names=['PatientId', 'Time'])
		patient.set_index(multi, inplace=True)

		patients.append(patient)

	return pd.concat(patients)


def load_data2(data_dir) -> pd.DataFrame:
	"""
	Loads time series data from Parquet files in a directory, or reads from
	a PSV file with the same name if the Parquet file is not available,
	and structures it into a Pandas DataFrame with a MultiIndex ('Patient ID', 'Time').
	The time interval between samples is assumed to be 1 hour.

	Args:
		data_dir (str): The directory containing the Parquet and PSV files.
						The filename (without extension) is used as the Patient ID.

	Returns:
		pandas.DataFrame: A Pandas DataFrame with a MultiIndex
						  ('Patient ID', 'Time') and columns representing the
						  features. Returns an empty DataFrame if no Parquet
						  files are found.

	Raises:
		FileNotFoundError: If the specified directory does not exist.
		Exception: If there is an error reading the Parquet or PSV file.
	"""
	if not os.path.exists(data_dir):
		raise FileNotFoundError(f"Directory {data_dir} not found")

	parquet_files = []
	psv_files = []

	for file in os.scandir(data_dir):
		if file.name.endswith(".parquet"):
			parquet_files.append(file.path)
		elif file.name.endswith(".psv"):
			psv_files.append(file.path)

	if not parquet_files and not psv_files:
		raise FileNotFoundError(f"No Parquet or PSV files found in {data_dir}")

	data_frames: list[pd.DataFrame] = []
	index_tuples = []
	time_interval = '1h'  # Set the time interval to 1 hour

	for parquet_file in parquet_files:
		try:
			# Extract Patient ID from the filename (without the extension)
			patient_id = os.path.splitext(parquet_file)[0]
			table = pq.read_table(parquet_file)
			df = table.to_pandas()

			if df.empty:
				print(f"Skipping empty file for patient: {patient_id}")
				continue

			# Create time index as Timedelta
			patient_time_index = pd.to_timedelta(np.arange(len(df)) * time_interval)

			# Create index tuples for this patient's data
			for t in patient_time_index:
				index_tuples.append((patient_id, t))

			data_frames.append(df)  # Append the raw dataframe

		except Exception as e:
			print(f"Error reading file for patient {patient_id}: {e}")
			continue

	for parquet_file in parquet_files:
		try:
			# Extract Patient ID from the filename (without the extension)
			patient_id = os.path.splitext(parquet_file)[0]
			psv_path = os.path.join(data_dir, f"{patient_id}.psv")

			# Read the Parquet file into a Pandas DataFrame
			if os.path.exists(parquet_path):
				table = pq.read_table(parquet_path)
				df = table.to_pandas()
			elif os.path.exists(psv_path):
				df = pd.read_csv(psv_path, sep='|')
			else:
				print(f"Neither Parquet nor PSV file found for patient: {patient_id}")
				continue

			# Check if the dataframe is empty
			if df.empty:
				print(f"Skipping empty file for patient: {patient_id}")
				continue

			# Create time index as Timedelta
			patient_time_index = pd.to_timedelta(np.arange(len(df)) * time_interval)

			# Create index tuples for this patient's data
			for t in patient_time_index:
				index_tuples.append((patient_id, t))

			data_frames.append(df)  # Append the raw dataframe

		except Exception as e:
			print(f"Error reading file for patient {patient_id}: {e}")
			continue

	if not data_frames:
		return pd.DataFrame()  # Return an empty Pandas DataFrame

	# Create the MultiIndex
	index = pd.MultiIndex.from_tuples(index_tuples, names=['PatientID', 'Time'])

	# Concatenate all patient DataFrames using Pandas
	combined_df = pd.concat(data_frames)

	# Assign the MultiIndex using Pandas.
	combined_df = combined_df.set_index(index)

	return combined_df