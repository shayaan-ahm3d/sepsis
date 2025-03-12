#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 15:44:27 2025

@author: ray
"""

import pandas as pd
import os
from sklearn.feature_selection import VarianceThreshold
from collections import Counter


def load_file(filename):
    
    df = pd.read_csv(filename, sep="|")
    return df

def file_to_dict(patient, data):
    # Rolling window setup: 6 hour window
    rolling_window = {
        f"{patient}_{i}": data.iloc[i:i+6] for i in range(len(data) - 5)
        }
        
    return rolling_window
        
def process(data):
    x = data.drop(columns = "SepsisLabel")
    x = x.dropna(axis="columns", thresh=2)
    
    # Variance Threshold setup
    selector = VarianceThreshold(threshold=0.1)

    selector.fit(x)
    high_variance = x.iloc[:, selector.get_support(indices=True)]

    dropped_cols = list(set(x.columns) - set(high_variance.columns))

    return dropped_cols

# Adapted from ChatGPT code
def find_frequent_entries(lists, threshold=0.8):
    # Number of lists
    num_lists = len(lists)
    # 80% threshold occurance
    min_count = int(threshold * num_lists)

    # Create Counter object to count occurances of each feature
    entry_counts = Counter(entry for sublist in lists for entry in set(sublist))

    # Find entries that appear in at least min_count lists
    frequent_entries = {entry for entry, count in entry_counts.items() if count >= min_count}

    return frequent_entries



# Get a list of file names
files_A = os.listdir("training_setA")
files_B = os.listdir("training_setB")

dataframes = {}  
 
for file in files_A: 
    patient_data = load_file("training_setA/"+file)
    dataframes |= file_to_dict(file[:-4], patient_data)
    
for file in files_B: 
    patient_data = load_file("training_setB/"+file)
    dataframes.update(file_to_dict(file[:-4], patient_data))

# Finds dropped columns in each dataset   
removed_cols = [process(frame) for frame in dataframes.values()]


frequent_drops = find_frequent_entries(removed_cols, 0.8)
print(frequent_drops)

