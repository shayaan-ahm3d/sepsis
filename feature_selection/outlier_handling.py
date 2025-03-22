#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 12:05:11 2025

@author: ray
"""

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


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
x_data = combined_data.drop(columns = ["SepsisLabel", "Gender", "Unit1", 
                                       "HospAdmTime", "ICULOS"])
y = combined_data["SepsisLabel"]

# Clear outlier_removal_tracking file to be rewritten
open("outlier_removal_tracking.txt",'w').close()

feature_bounds = {}
# Create dictionary of outlier boundaries
for feature in x_data.columns:

    data = x_data[feature].dropna()
    mean, sigma = data.mean(), data.std()
    # Calculate outlier boundaries (twice what counts as an outlier)
    lower_bound, upper_bound = mean - 6*sigma, mean + 6*sigma 
    feature_bounds[feature] = (lower_bound, upper_bound)

for patient in dataframes.keys():
    data = dataframes[patient]
    patient_output = f'Patient {patient}:\n'
    any_outliers_removed = False # Reset per patient
    
    for feature in feature_bounds.keys():
        # Count outliers
        lower_outliers = (data[feature] < feature_bounds[feature][0]).sum()
        upper_outliers = (data[feature] > feature_bounds[feature][1]).sum()
        
        data.loc[data[feature] < feature_bounds[feature][0], feature] = None # Lower bound
        data.loc[data[feature] > feature_bounds[feature][1], feature] = None # Upper bound
        
        if lower_outliers != 0 or upper_outliers != 0:
            any_outliers_removed = True # Indicate outliers removed
            patient_output += f'{feature}: {upper_outliers} large outliers and {lower_outliers} small outliers removed\n'
    
    if any_outliers_removed:
        print(patient_output)
        print('\n')
        
        with open('outlier_removal_tracking.txt', 'a') as f:
            f.write(patient_output + '\n')
            
# Save data with outliers removed
for file in files_A: 
    patient_data = dataframes[file[:-4]]
    patient_data.to_csv("../feature_reduced_data/reduced_noOutliers_A/"+file, 
                        sep="|", index=False)
    
for file in files_B: 
    patient_data = dataframes[file[:-4]]
    patient_data.to_csv("../feature_reduced_data/reduced_noOutliers_B/"+file, 
                        sep="|", index=False)
    
        
"""
Identify upper and lower bounds for outlier removal per feature (yay)
For each patient, change values outside this range
Print/save to text file - how many upper and lower outliers have been removed 
for each patient

PATIENT NUMBER:
    x upper and y lower removed from feature
    
Save patient data as new individual files
"""