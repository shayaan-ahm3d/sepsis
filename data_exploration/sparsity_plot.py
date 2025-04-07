#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 14:42:56 2025

@author: ray
"""

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def load_file(filename):
    
    df = pd.read_csv(filename, sep="|")
    return df
        

# Get a list of file names
files_A = os.listdir("../training_setA")
files_B = os.listdir("../training_setB")

dataframes = []
HospA_patients = []
HospB_patients = [] 

# Load files into dataframe dictionary 
for file in files_A: 
    patient_data = load_file("../training_setA/"+file)
    dataframes.append(patient_data)
    HospA_patients.append(patient_data)
    
for file in files_B: 
    patient_data = load_file("../training_setB/"+file)
    dataframes.append(patient_data)
    HospB_patients.append(patient_data)

# Create one dataframe of all data    
combined_data = pd.concat(dataframes)
HospA_data = pd.concat(HospA_patients)
HospB_data = pd.concat(HospB_patients)

# Calculate total NaN percentage
total_nan = combined_data.isna().sum().sum()
total_size = combined_data.size
percentage_nan = total_nan*100/total_size

print(f'{percentage_nan:.2f}% NaN values across all data.')

# Calculate hospital specific feature-wise NaN percentages
HospA_nan = HospA_data.isna().sum()
HospA_len = HospA_data.shape[0]
A_percent_NaN = HospA_nan*100/HospA_len

HospB_nan = HospB_data.isna().sum()
HospB_len = HospB_data.shape[0]
B_percent_NaN = HospB_nan*100/HospB_len

# Find all features over 80% NaN
all_percent = pd.DataFrame(A_percent_NaN).rename(columns={0: "A"})
all_percent["B"] = B_percent_NaN
NaN_80 = all_percent[(all_percent["A"] >= 80) & (all_percent["B"] >= 80)].shape[0]

print(f'{NaN_80} features missing more than 80% of values')

# # Plot Hospital specific feature NaN percentages
# sns.barplot(A_percent_NaN, color="tab:blue")
# plt.xticks(rotation = 90)
# plt.title("NaN in Hospital A Data")
# plt.xlabel("Features")
# plt.ylabel("% NaN")
# plt.savefig("../plots/NaN_percentage_HospA.png", dpi=300, bbox_inches='tight')        
# plt.show()

# sns.barplot(B_percent_NaN, color="tab:pink")
# plt.xticks(rotation = 90)
# plt.title("NaN in Hospital B Data")
# plt.xlabel("Features")
# plt.ylabel("% NaN")
# plt.savefig("../plots/NaN_percentage_HospB.png", dpi=300, bbox_inches='tight')        
# plt.show()