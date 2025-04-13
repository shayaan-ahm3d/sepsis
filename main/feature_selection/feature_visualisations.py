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
x_data = combined_data.drop(columns = ["SepsisLabel", "Gender", "Unit1", 
                                       "HospAdmTime", "ICULOS"])
y = combined_data["SepsisLabel"]


# Plot histograms - highlighting outliers
for feature in x_data.columns:

    data = x_data[feature].dropna()
    mean, sigma = data.mean(), data.std()
    # Calculate outlier boundaries
    lower_bound, upper_bound = mean - 3*sigma, mean + 3*sigma 

    # Define bin edges
    bins = np.histogram_bin_edges(data, bins=50)

    # Separate values inside and outside the 3-sigma range
    inlier_counts, _ = np.histogram(data[(data >= lower_bound) & (data <= upper_bound)], bins=bins)
    outlier_counts, _ = np.histogram(data[(data < lower_bound) | (data > upper_bound)], bins=bins)

    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.bar(bins[:-1], inlier_counts, width=np.diff(bins), color="blue", alpha=0.6, label="Within 3σ")
    plt.bar(bins[:-1], outlier_counts, width=np.diff(bins), color="red", alpha=0.6, label="Outliers")

    # Overlay vertical lines for mean and thresholds
    plt.axvline(mean, color="black", linestyle="--", label="Mean")
    plt.axvline(lower_bound, color="purple", linestyle="--", label="3 Std Dev")
    plt.axvline(upper_bound, color="purple", linestyle="--")

    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"../plots/{feature}_histogram.png", dpi=300, bbox_inches='tight')
    plt.show()


