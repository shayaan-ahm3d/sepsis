import pandas as pd
import os
from scipy.stats import chi2_contingency


def count_sepsis_labels(directory, folder_name):
    sepsis_counts = {0: 0, 1: 0}
    file_path = os.path.join(directory, folder_name)
    all_files = os.listdir(file_path)

    for file in all_files:
        df = pd.read_csv(os.path.join(file_path, file), sep="|", usecols=["SepsisLabel"])
        counts = df["SepsisLabel"].value_counts().to_dict()
        sepsis_counts[0] += counts.get(0, 0)
        sepsis_counts[1] += counts.get(1, 0)

    return sepsis_counts



dataset_paths = {
    "SetA": "../",  # Update this path
    "SetB": "../"  # Update this path
}

sepsis_counts_A = count_sepsis_labels(dataset_paths["SetA"], "training_setA")
sepsis_counts_B = count_sepsis_labels(dataset_paths["SetB"], "training_setB")

contingency_table = [
    [sepsis_counts_A[0], sepsis_counts_A[1]],  # Set A
    [sepsis_counts_B[0], sepsis_counts_B[1]]  # Set B
]

chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-square Statistic: {chi2_stat}")
print(f"p-value: {p_value}")

if p_value < 0.05:
    print("The label distributions in Set A and Set B are significantly different.")
else:
    print("No significant difference in label distributions between Set A and Set B.")
