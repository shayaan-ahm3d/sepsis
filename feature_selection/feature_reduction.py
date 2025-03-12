#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 15:44:27 2025

@author: ray
"""

import pandas as pd
import os



def load_file(filename):
    
    df = pd.read_csv(filename, sep="|")
    df = df.drop(columns = ["pH", "Unit2", "Bilirubin_total", "Hct"])
    
    return df

# Get a list of file names
files_A = os.listdir("../training_setA")
files_B = os.listdir("../training_setB")

# Save reduced data
for file in files_A: 
    patient_data = load_file("../training_setA/"+file)
    patient_data.to_csv("../feature_reduced_data/reduced_training_setA/"+file, 
                        sep="|")
    
for file in files_B: 
    patient_data = load_file("../training_setB/"+file)
    patient_data.to_csv("../feature_reduced_data/reduced_training_setB/"+file, 
                        sep="|")