#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 14:47:29 2025

@author: ray
"""

import pandas as pd
import os


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
files_A = os.listdir("training_setA")
files_B = os.listdir("training_setB")

dataframes = {}  
 
for file in files_A: 
    patient_data = load_file("training_setA/"+file)
    dataframes |= file_to_dict(file[:-4], patient_data)
    
for file in files_B: 
    patient_data = load_file("training_setB/"+file)
    dataframes |= file_to_dict(file[:-4], patient_data)
    
combined_data = pd.concat(dataframes.values())

print(combined_data.var())
x = combined_data.drop(columns = "SepsisLabel")
y = combined_data["SepsisLabel"]

"""
Remove columns with low variance
- What is counted as low variance?

Do not remove Age column as is categorical and will therefore have low variance
"""
