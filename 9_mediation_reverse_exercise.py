#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reverse Mediation:
Exercise → Brain Coupling → Heart Rate

ROI defined as:
    ICA-level Exercise FDR mask (zmap_FDR_ICA.nii.gz)

Paths:
    a: Exercise → Brain
    b: Brain → HR (controlling Exercise)
    c: Exercise → HR (total)
    c': Exercise → HR (direct)
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
# Paths
# =====================================================

BASE_DIR = "/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/myICA_separate100_clean_0p1"

MASK_PATH = f"{BASE_DIR}/level2/20260217_PURECOUPLING_ICA{ICA:02d}_AGE-cont_smooth0p3_BHq0p05_FDRwithinICA_cluster10/{METRIC}/ica{ICA:02d}/Exercise_ec/zmap_FDR_ICA.nii.gz"

LEVEL1_DIR = f"{BASE_DIR}/level1/level1_ica{ICA:02d}"

METADATA_FILE = "/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/metadata/cardiac_design_updated3.csv"

print("\nUsing ROI mask:")
print(MASK_PATH)


# =====================================================
# Load ROI mask
# =====================================================

z_img = image.load_img(MASK_PATH)
z_data = z_img.get_fdata()

binary_mask_data = (z_data != 0).astype(int)

if binary_mask_data.sum() == 0:
    raise ValueError("FDR mask has zero surviving voxels.")

roi_mask = image.new_img_like(z_img, binary_mask_data)

print(f"ROI voxels: {int(binary_mask_data.sum())}")


# =====================================================
# Load metadata
# =====================================================

metadata = pd.read_csv(METADATA_FILE)
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
    return model.coef_


# =====================================================
# Mediation paths (REVERSE)
# =====================================================

# a path: Exercise → Brain
Xa = df[["Exercise"] + covars]
ya = df["Brain_ROI"]
a = regress(ya, Xa)[0]

# b + c'
Xb = df[["Brain_ROI", "Exercise"] + covars]
yb = df["HR"]
coefs = regress(yb, Xb)
b = coefs[0]
c_prime = coefs[1]

# c total
Xc = df[["Exercise"] + covars]
yc = df["HR"]
c_total = regress(yc, Xc)[0]

indirect = a * b


print("\n================ REVERSE MEDIATION RESULTS ================")
print(f"a  (Exercise → Brain):       {a:.4f}")
print(f"b  (Brain → HR):             {b:.4f}")
print(f"c  (total Exercise → HR):    {c_total:.4f}")
print(f"c' (direct Exercise → HR):   {c_prime:.4f}")
print(f"Indirect (a*b):              {indirect:.4f}")


# =====================================================
# Bootstrap
# =====================================================

boot_vals = []
n = len(df)

for _ in trange(N_BOOT):
    samp = df.sample(n, replace=True)

    a_b = LinearRegression().fit(
        samp[["Exercise"] + covars],
        samp["Brain_ROI"]
    ).coef_[0]

    b_b = LinearRegression().fit(
        samp[["Brain_ROI", "Exercise"] + covars],
        samp["HR"]
    ).coef_[0]

    boot_vals.append(a_b * b_b)

boot_vals = np.array(boot_vals)

ci_lower = np.percentile(boot_vals, 2.5)
ci_upper = np.percentile(boot_vals, 97.5)

print("\nBootstrap 95% CI for indirect effect:")
print(f"[{ci_lower:.4f}, {ci_upper:.4f}]")

if ci_lower > 0 or ci_upper < 0:
    print("Reverse indirect effect statistically significant.")
else:
    print("Reverse indirect effect NOT significant.")


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

out_path = f"{BASE_DIR}/reverse_mediation_ICA{ICA:02d}_{METRIC}.tsv"
out_df.to_csv(out_path, sep="\t", index=False)

print("\nSaved reverse mediation results to:")
print(out_path)
print("====================================================")
