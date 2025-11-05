# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 17:21:59 2025

@author: amy.cromartie
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May  1 17:31:08 2025
Modified on Oct 24 2025
Author: amy.cromartie

Purpose:
    Random Forest classification with calibrated probabilities and bootstrap confidence intervals.
.
"""

# -------------------
# Imports
# -------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import csv
import os

# -------------------
# USER INPUTS: File & Directory Settings
# -------------------
# ⚙️ Set this base directory where your input/output files are stored
# For mac this can be easily modified with / 
BASE_DIR = r"C:\Users\Username\Documents\ProbGDGTs"

# ⚙️ Input file names (must exist in BASE_DIR)
GDGT_FILE = "ProbGDGT_SMOTE_DB.csv"
CORES_FILE = "Roblesetal2022_ExampleCore_Vanevan.csv"

# ⚙️ Output directory for results (can be same or different)
OUTPUT_DIR = r"C:\Users\Username\Documents\ProbGDGTs\Output"


# ⚙️ Output filenames
OUTPUT_CSV = "RF_Probabilities_With_CI.csv"
OUTPUT_PDF_ALL = "RF_Probabilities_Comparison.pdf"
OUTPUT_PDF_CLASS = "{}_Probability_Over_Time.pdf"  # {} will be replaced by class name

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------
# Load Data
# -------------------

def smart_read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(2048)
        try:
            dialect = csv.Sniffer().sniff(sample)
            sep = dialect.delimiter
        except csv.Error:
            sep = ","  # fallback if detection fails
    return pd.read_csv(path, sep=sep, engine="python")

cores = smart_read_csv(os.path.join(BASE_DIR, CORES_FILE))
GDGT_SMOTE = smart_read_csv(os.path.join(BASE_DIR, GDGT_FILE))


# -------------------
# Setup
# -------------------
FA = [
    "Ia", "Ib", "Ic", "IIa_5me", "IIa_6me", "IIb_5me", "IIb_6me", "IIc_5me", "IIc_6me",
    "IIIa_5me", "IIIa_6me", "IIIb_5me", "IIIb_6me", "IIIc_5me", "IIIc_6me"
]

X = GDGT_SMOTE[FA]
y = GDGT_SMOTE["SampleTypeNum"]

core_dates = pd.to_numeric(cores["Core Dates"], errors='coerce')
core_features = cores[FA]

# -------------------
# Parameters
# -------------------
# Keep the bootstrap low when testing to save cpu cycles. Increase once ready
n_bootstraps = 10
class_labels = {0: "Lake", 1: "Soil", 2: "Peat"}
colors = {"Lake": "blue", "Soil": "saddlebrown", "Peat": "green"}
n_classes = len(class_labels)
random_state = 42

# -------------------
# Storage for bootstrap predictions
# -------------------
all_probs = np.zeros((n_bootstraps, core_features.shape[0], n_classes))
np.random.seed(random_state)

# -------------------
# Bootstrap loop
# -------------------
for i in range(n_bootstraps):
    X_resample, y_resample = resample(X, y, stratify=y, random_state=i)
    X_train, X_val, y_train, y_val = train_test_split(
        X_resample, y_resample, test_size=0.2, stratify=y_resample, random_state=i
    )

    rf = RandomForestClassifier(n_estimators=1000, random_state=i)
    rf.fit(X_train, y_train)

    cal_rf = CalibratedClassifierCV(rf, method='sigmoid', cv='prefit')
    cal_rf.fit(X_val, y_val)

    probs = cal_rf.predict_proba(core_features)
    all_probs[i] = probs

# -------------------
# Compute statistics
# -------------------
mean_probs = np.mean(all_probs, axis=0)
lower_bounds = np.percentile(all_probs, 2.5, axis=0)
upper_bounds = np.percentile(all_probs, 97.5, axis=0)

# -------------------
# Output to CSV
# -------------------
output_df = pd.DataFrame({'BP': core_dates})
for cls in range(n_classes):
    label = class_labels[cls]
    output_df[f'{label}_Mean'] = mean_probs[:, cls]
    output_df[f'{label}_Lower95CI'] = lower_bounds[:, cls]
    output_df[f'{label}_Upper95CI'] = upper_bounds[:, cls]

output_path_csv = os.path.join(OUTPUT_DIR, OUTPUT_CSV)
output_df.to_csv(output_path_csv, index=False)

# -------------------
# Plot: All classes
# -------------------
sorted_idx = np.argsort(core_dates)
BP_sorted = core_dates.iloc[sorted_idx]

plt.figure(figsize=(12, 6))
for cls in range(n_classes):
    label = class_labels[cls]
    plt.plot(BP_sorted, mean_probs[sorted_idx, cls], label=label, color=colors[label])
    plt.fill_between(
        BP_sorted,
        lower_bounds[sorted_idx, cls],
        upper_bounds[sorted_idx, cls],
        color=colors[label],
        alpha=0.3
    )

plt.xlabel("Years Before Present (BP)")
plt.ylabel("Predicted Probability")
plt.title("Random Forest Calibrated Class Probabilities with 95% CI")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.gca().invert_xaxis()

output_path_allpdf = os.path.join(OUTPUT_DIR, OUTPUT_PDF_ALL)
plt.savefig(output_path_allpdf, format='pdf')
plt.show()

# -------------------
# Separate plots for each class
# -------------------
for cls in range(n_classes):
    label = class_labels[cls]
    color = colors[label]

    plt.figure(figsize=(10, 4))
    plt.plot(BP_sorted, mean_probs[sorted_idx, cls], label=f'{label}', color=color)
    plt.fill_between(
        BP_sorted,
        lower_bounds[sorted_idx, cls],
        upper_bounds[sorted_idx, cls],
        color=color,
        alpha=0.3
    )
    plt.xlabel("Years Before Present (BP)")
    plt.ylabel("Predicted Probability")
    plt.title(f"{label} Probability Over Time (with 95% CI)")
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.tight_layout()

    output_path_classpdf = os.path.join(OUTPUT_DIR, OUTPUT_PDF_CLASS.format(label))
    plt.savefig(output_path_classpdf, format='pdf')
    plt.show()
