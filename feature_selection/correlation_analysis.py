#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 13:16:54 2025

@author: ray
"""

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# # Use LaTeX for all text in the plot
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.size": 12,
#     })

def load_file(filename):
    
    df = pd.read_csv(filename, sep="|")
    return df
        


# Get a list of file names
files_A = os.listdir("../training_setA")
files_B = os.listdir("../training_setB")

dataframes = {}  
 
for file in files_A: 
    patient_data = load_file("../training_setA/"+file)
    dataframes[str(file[:-4])] = patient_data
    
for file in files_B: 
    patient_data = load_file("../training_setB/"+file)
    dataframes[str(file[:-4])] = patient_data
    
combined_data = pd.concat(dataframes.values())

# Separate out features and labels
x = combined_data.drop(columns = "SepsisLabel")
x = x.drop(columns = "pH")
y = combined_data["SepsisLabel"]

"""
Display correlation pairs above a certain threshold (likely have same impact on
                                                     model)
"""

corr_matrix = x.corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, xticklabels=True, yticklabels=True)
plt.title("Correlation Plot of Features")
plt.savefig("../plots/feature_correlation.pdf", dpi=300, bbox_inches='tight')
plt.show()

# Find values above 0.9
rows, cols = np.where(abs(corr_matrix) > 0.9)

# Create a list of (index, column) pairs
high_values = [(corr_matrix.index[r], corr_matrix.columns[c]) for r, c in zip(rows, cols)
               if r != c]

for pair in high_values:
    print(f'{pair}: {corr_matrix.loc[pair[0], pair[1]]}')
    
    corr_a = y.corr(x.loc[:, pair[0]])
    corr_b = y.corr(x.loc[:, pair[1]])
    
    print(f'{pair[0]} to Sepsis: {corr_a}')
    print(f'{pair[1]} with Sepsis: {corr_b} \n')
    