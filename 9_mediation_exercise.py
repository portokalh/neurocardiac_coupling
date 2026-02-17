#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exercise → Heart Rate → Brain Coupling Mediation (ROI-based)

ROI defined as:
    ICA-level Exercise FDR mask (zmap_FDR_ICA.nii.gz)

Paths:
    a: Exercise → HR
    b: HR → Brain (controlling Exercise)
    c: Exercise → Brain (total)
    c': Exercise → Brain (direct)
    indirect: a*b

Bootstrap (nonparametric) CI for indirect effect.

Author: alex
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import trange
from sklearn.linear_model import LinearRegression
from nilearn import image
from nilearn.masking import apply_mask


# =====================================================
# Arguments
# =====================================================

parser = argparse.ArgumentParser()

parser.add_argument("--metric", type=str, default="Heart_Rate")
parser.add_argument("--ica", type=int, default=19)
parser.add_argument("--n_boot", type=int, default=5000)

args = parser.parse_args()

METRIC = args.metric
ICA = args.ica
N_BOOT = args.n_boot


# =====================================================
# USER: SET PATH TO YOUR ICA EXERCISE FDR MASK
# =====================================================

MASK_PATH = "/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/myICA_separate100_clean_0p1/level2/20260217_PURECOUPLING_ICA19_AGE-cont_smooth0p3_BHq0p05_FDRwithinICA_cluster10/Heart_Rate/ica19/Exercise_ec/zmap_FDR_ICA.nii.gz"

LEVEL1_DIR = f"/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/myICA_separate100_clean_0p1/level1/level1_ica{ICA:02d}"

METADATA_FILE = "/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/metadata/cardiac_design_updated3.csv"

print("\nUsing ROI mask:")
print(MASK_PATH)


# =====================================================
# Load data
# =====================================================

z_img = image.load_img(MASK_PATH)
z_data = z_img.get_fdata()

# Create binary mask from nonzero voxels
binary_mask_data = (z_data != 0).astype(int)

if binary_mask_data.sum() == 0:
    raise ValueError("FDR mask has zero surviving voxels.")

roi_mask = image.new_img_like(z_img, binary_mask_data)

print(f"ROI voxels: {int(binary_mask_data.sum())}")

metadata = pd.read_csv(METADATA_FILE)

# Exclude KO
metadata = metadata[metadata["KO_Genotype"] == 0].copy()

brain_vals = []
rows_used = []

for _, row in metadata.iterrows():
    sid = str(row["Arunno"])
    img_path = os.path.join(LEVEL1_DIR, f"seed_connectivity_map_{sid}.nii.gz")

    if not os.path.exists(img_path):
        continue

    img = image.load_img(img_path)
    roi_data = apply_mask(img, roi_mask)

    if roi_data.size == 0:
        continue

    brain_vals.append(np.mean(roi_data))
    rows_used.append(row)

df = pd.DataFrame(rows_used).reset_index(drop=True)
df["Brain_ROI"] = brain_vals

print(f"\nSubjects included in mediation: {len(df)}")


# =====================================================
# Prepare variables
# =====================================================

df["Exercise"] = pd.to_numeric(df["Exercise_Yes"], errors="coerce")
df["HR"] = pd.to_numeric(df[METRIC], errors="coerce")

covars = ["Age", "Sex_Male", "Diet_HFD", "HN", "E2_Genotype", "E4_Genotype"]

df = df.dropna(subset=["Exercise", "HR", "Brain_ROI"] + covars)

print(f"Subjects after NA drop: {len(df)}")

# Standardize continuous variables
df["HR"] = (df["HR"] - df["HR"].mean()) / df["HR"].std()
df["Brain_ROI"] = (df["Brain_ROI"] - df["Brain_ROI"].mean()) / df["Brain_ROI"].std()


# =====================================================
# Helper regression
# =====================================================

def regress(y, X):
    model = LinearRegression().fit(X, y)
    return model.coef_, model


# =====================================================
# Mediation paths
# =====================================================

# a path: Exercise → HR
Xa = df[["Exercise"] + covars]
ya = df["HR"]
coef_a, model_a = regress(ya, Xa)
a = coef_a[0]

# b + c' paths
Xb = df[["HR", "Exercise"] + covars]
yb = df["Brain_ROI"]
coef_b, model_b = regress(yb, Xb)
b = coef_b[0]        # HR
c_prime = coef_b[1]  # Exercise

# c total effect
Xc = df[["Exercise"] + covars]
yc = df["Brain_ROI"]
coef_c, model_c = regress(yc, Xc)
c_total = coef_c[0]

indirect = a * b


print("\n================ MEDIATION RESULTS ================")
print(f"a  (Exercise → HR):        {a:.4f}")
print(f"b  (HR → Brain):           {b:.4f}")
print(f"c  (total Exercise → Brain): {c_total:.4f}")
print(f"c' (direct Exercise → Brain): {c_prime:.4f}")
print(f"Indirect (a*b):            {indirect:.4f}")


# =====================================================
# Bootstrap indirect effect
# =====================================================

boot_vals = []
n = len(df)

for _ in trange(N_BOOT):
    samp = df.sample(n, replace=True)

    Xa_b = samp[["Exercise"] + covars]
    ya_b = samp["HR"]
    a_b = LinearRegression().fit(Xa_b, ya_b).coef_[0]

    Xb_b = samp[["HR", "Exercise"] + covars]
    yb_b = samp["Brain_ROI"]
    b_b = LinearRegression().fit(Xb_b, yb_b).coef_[0]

    boot_vals.append(a_b * b_b)

boot_vals = np.array(boot_vals)

ci_lower = np.percentile(boot_vals, 2.5)
ci_upper = np.percentile(boot_vals, 97.5)

print("\nBootstrap 95% CI for indirect effect:")
print(f"[{ci_lower:.4f}, {ci_upper:.4f}]")

if ci_lower > 0 or ci_upper < 0:
    print("Indirect effect is statistically significant (CI does not cross 0).")
else:
    print("Indirect effect NOT significant (CI crosses 0).")


# =====================================================
# Save results
# =====================================================

out_df = pd.DataFrame({
    "a": [a],
    "b": [b],
    "c_total": [c_total],
    "c_prime": [c_prime],
    "indirect_ab": [indirect],
    "CI_lower": [ci_lower],
    "CI_upper": [ci_upper],
    "n_subjects": [len(df)]
})

out_path = f"/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/myICA_separate100_clean_0p1/mediation_ICA{ICA:02d}_{METRIC}.tsv"

out_df.to_csv(out_path, sep="\t", index=False)

print("\nSaved mediation results to:")
print(out_path)
print("====================================================")
