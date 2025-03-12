#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 14:47:29 2025

@author: ray
"""

import pandas as pd
import os
from sklearn.feature_selection import VarianceThreshold


def load_file(filename):
    
    df = pd.read_csv(filename, sep="|")
    return df

def file_to_dict(patient, data):
    
    rolling_window = {}
    # Rolling window setup: 6 hour window
    for i in range(0, len(data.axes[0])-6):
        rolling_window[patient+"_"+str(i)] = data[i:i+6]
        
    return rolling_window
        


# Get a list of file names
files_A = os.listdir("../training_setA")
files_B = os.listdir("../training_setB")

dataframes = {}  
 
for file in files_A: 
    patient_data = load_file("../training_setA/"+file)
    dataframes |= file_to_dict(file[:-4], patient_data)
    
for file in files_B: 
    patient_data = load_file("../training_setB/"+file)
    dataframes |= file_to_dict(file[:-4], patient_data)
    
combined_data = pd.concat(dataframes.values())

# Separate out features and labels
x = combined_data.drop(columns = "SepsisLabel")
y = combined_data["SepsisLabel"]

# Variance Threshold setup
selector = VarianceThreshold(threshold=0.1)

selector.fit(x)
high_variance_indices = selector.get_support(indices=True)
high_variance = x.iloc[:, high_variance_indices]


