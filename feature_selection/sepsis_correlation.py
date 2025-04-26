#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 16:37:41 2025

@author: ray
"""

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
from matplotlib.patches import Patch

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
files_A = os.listdir(f"{os.getcwd()}/../training_setA")
files_B = os.listdir(f"{os.getcwd()}/../training_setB")

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

# Graph setup
corr = []
features = []

for feature in x.columns:
    df = pd.DataFrame({'x': x[feature], 'y': y})
    df = df.dropna()
    corr.append(pointbiserialr(df['x'], df['y'])[0])
    features.append(feature)

# More graph colour setup    
corr_colour = ['tab:blue' if val > 0 else 'tab:red' for val in corr]
legend_elements = [Patch(facecolor='tab:blue', label='Positive Correlation'),
                   Patch(facecolor='tab:red', label='Negative Correlation')]
    
plt.figure(figsize=(6,8))
plt.xlim([-0.2, 0.2])
sns.barplot(x=corr, y=features, palette=corr_colour)
plt.title("Feature Correlation with Sepsis Label")
plt.ylabel("Feature")
plt.xlabel("Correlation Coefficient (Zoomed to [-0.2, 0.2])")
plt.legend(handles=legend_elements)
# plt.savefig("../plots/sepsis_correlation.png", dpi=300, bbox_inches='tight')
plt.show()
