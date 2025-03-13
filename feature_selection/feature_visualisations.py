#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 15:25:24 2025

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
    
    df = pd.read_csv(filename, sep="|", index_col="Unnamed: 0")
    return df
        


# Get a list of file names
files_A = os.listdir("../feature_reduced_data/reduced_training_setA")
files_B = os.listdir("../feature_reduced_data/reduced_training_setB")

dataframes = {}  

# Load files into dataframe dictionary 
for file in files_A: 
    patient_data = load_file("../feature_reduced_data/reduced_training_setA/"+file)
    dataframes[str(file[:-4])] = patient_data
    
for file in files_B: 
    patient_data = load_file("../feature_reduced_data/reduced_training_setB/"+file)
    dataframes[str(file[:-4])] = patient_data

# Create one dataframe of all data    
combined_data = pd.concat(dataframes.values())

# Separate out features and labels
x_data = combined_data.drop(columns = "SepsisLabel")
y = combined_data["SepsisLabel"]

# Plot 'box plots' of all features (basically just shows outliers)
plt.figure(figsize=(8,6))
sns.boxplot(data = x_data, orient="h")
plt.title("Outliers and Distribution of Values by Feature")
plt.savefig("../plots/all_feature_boxplot.pdf", dpi=300, bbox_inches='tight')
plt.show()


