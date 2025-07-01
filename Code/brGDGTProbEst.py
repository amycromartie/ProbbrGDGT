# -*- coding: utf-8 -*-
"""
Created on Thu May  1 17:31:08 2025

@author: amy.cromartie
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# -------------------
# Load data
# -------------------
# Set your directory for the database
GDGT_SMOTE = pd.read_csv('C:/Users/amy.cromartie/Documents/Python Scripts/SMOTENov242022.csv', delimiter=',')

#Put your core file here and the directory
cores = pd.read_csv('C:/Users/amy.cromartie/Documents/Python Scripts/ProbCores2025.csv', delimiter=';')

# -------------------
# Setup
# -------------------
FA = ["Ia", "Ib", "Ic", "IIa_5me", "IIa_6me", "IIb_5me", "IIb_6me", "IIc_5me", "IIc_6me",
      "IIIa_5me", "IIIa_6me", "IIIb_5me", "IIIb_6me", "IIIc_5me", "IIIc_6me"]

X = GDGT_SMOTE[FA]
y = GDGT_SMOTE["SampleTypeNum"]

core_dates = pd.to_numeric(cores["Core Dates"], errors='coerce')
core_features = cores[FA]

# -------------------
# Parameters
# -------------------
n_bootstraps = 10
class_labels = {0: "Lake", 1: "Soil", 2: "Peat"}
colors = {"Lake": "blue", "Soil": "saddlebrown", "Peat": "green"}

n_classes = len(class_labels)
random_state = 42

# -------------------
# Storage for bootstrap predictions
# -------------------
all_probs = np.zeros((n_bootstraps, core_features.shape[0], n_classes))

# -------------------
# Bootstrap loop
# -------------------
np.random.seed(random_state)

for i in range(n_bootstraps):
    X_resample, y_resample = resample(X, y, stratify=y, random_state=i)
    X_train, X_val, y_train, y_val = train_test_split(X_resample, y_resample, test_size=0.2, stratify=y_resample, random_state=i)

    rf = RandomForestClassifier(n_estimators=1000, random_state=i)
    rf.fit(X_train, y_train)

    cal_rf = CalibratedClassifierCV(rf, method='sigmoid', cv='prefit')
    cal_rf.fit(X_val, y_val)

    probs = cal_rf.predict_proba(core_features)
    all_probs[i] = probs

# -------------------
# Compute stats
# -------------------
mean_probs = np.mean(all_probs, axis=0)
lower_bounds = np.percentile(all_probs, 2.5, axis=0)
upper_bounds = np.percentile(all_probs, 97.5, axis=0)

# -------------------
# Output to CSV
# -------------------
output_df = pd.DataFrame()
output_df['BP'] = core_dates

for cls in range(n_classes):
    label = class_labels[cls]
    output_df[f'{label}_Mean'] = mean_probs[:, cls]
    output_df[f'{label}_Lower95CI'] = lower_bounds[:, cls]
    output_df[f'{label}_Upper95CI'] = upper_bounds[:, cls]

# Set your directory to output files here
output_df.to_csv('C:/Users/amy.cromartie/Documents/Ggdtarticle/RF_Probabilities_With_CI.csv', index=False)

# -------------------
# Plot: All classes
# -------------------
sorted_idx = np.argsort(core_dates)
BP_sorted = core_dates.iloc[sorted_idx]

plt.figure(figsize=(12, 6))
for cls in range(n_classes):
    label = class_labels[cls]
    plt.plot(BP_sorted, mean_probs[sorted_idx, cls], label=label, color=colors[label])
    plt.fill_between(BP_sorted,
                     lower_bounds[sorted_idx, cls],
                     upper_bounds[sorted_idx, cls],
                     color=colors[label],
                     alpha=0.3)

plt.xlabel("Years Before Present (BP)")
plt.ylabel("Predicted Probability")
plt.title("Random Forest Calibrated Class Probabilities with 95% CI")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.gca().invert_xaxis()  # 10,000 BP on the right
plt.savefig('C:/Users/amy.cromartie/Documents/Ggdtarticle/RF_Probabilities_Comparison.pdf', format='pdf')  # Save as PDF
plt.show()

# -------------------
# Separate Plots for Each Class
# -------------------
for cls in range(n_classes):
    label = class_labels[cls]
    color = colors[label]
    
    plt.figure(figsize=(10, 4))
    plt.plot(BP_sorted, mean_probs[sorted_idx, cls], label=f'{label}', color=color)
    plt.fill_between(BP_sorted,
                     lower_bounds[sorted_idx, cls],
                     upper_bounds[sorted_idx, cls],
                     color=color, alpha=0.3)
    plt.xlabel("Years Before Present (BP)")
    plt.ylabel("Predicted Probability")
    plt.title(f"{label} Probability Over Time (with 95% CI)")
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'C:/Users/amy.cromartie/Documents/Ggdtarticle/{label}_Probability_Over_Time.pdf', format='pdf')  # Save as PDF for each class
    plt.show()
