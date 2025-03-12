#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 13:16:54 2025

@author: ray
"""

import pandas as pd
import os
import seaborn as sns


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
    dataframes.update(file_to_dict(file[:-4], patient_data))
    
for file in files_B: 
    patient_data = load_file("../training_setB/"+file)
    dataframes.update(file_to_dict(file[:-4], patient_data))
    
combined_data = pd.concat(dataframes.values())

# Separate out features and labels
x = combined_data.drop(columns = "SepsisLabel")
y = combined_data["SepsisLabel"]

"""
Create correlation matrix
Display correlation pairs above a certain threshold (likely have same impact on
                                                     model)
Create heatmap
"""
