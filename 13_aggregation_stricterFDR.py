#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 17:32:33 2026

@author: alex
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from statsmodels.stats.multitest import multipletests

# =====================================================
# USER SETTINGS
# =====================================================

LEVEL2_ROOT = "/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/myICA_separate100_clean_0p1/level2"

TARGET_SMOOTH = "smooth0p3"
TARGET_CLUSTER = "cluster10"
TARGET_Q = "BHq0p05"
TARGET_MODEL = "HierBH_SimesAcrossICA_FDRwithinICA"

ALPHA = 0.05

print("\n=================================================")
print("GLOBAL AGGREGATION")
print("=================================================\n")

# =====================================================
# STRICT ICA FOLDER DETECTION
# =====================================================

pattern = re.compile(
    r"^\d{8}_PURECOUPLING_ICA(\d+)_AGE-cont_"
    + TARGET_SMOOTH + "_"
    + TARGET_Q + "_"
    + TARGET_MODEL + "_"
    + TARGET_CLUSTER
    + r"$"
)

analysis_dirs = []

for d in os.listdir(LEVEL2_ROOT):
    match = pattern.match(d)
    if match:
        ica_num = int(match.group(1))
        if 1 <= ica_num <= 20:
            analysis_dirs.append(d)

analysis_dirs = sorted(analysis_dirs)

print("Detected ICA folders:")
for d in analysis_dirs:
    print("  ", d)

if len(analysis_dirs) != 20:
    print(f"\nWARNING: Expected 20 ICA folders, found {len(analysis_dirs)}\n")

if len(analysis_dirs) == 0:
    raise RuntimeError("No ICA folders found.")

# =====================================================
# LOAD ALL METRIC × ICA SUMMARIES
# =====================================================

all_rows = []

for d in analysis_dirs:

    folder_path = os.path.join(LEVEL2_ROOT, d)
    ica_match = re.search(r"ICA(\d+)", d)
    ica_number = int(ica_match.group(1))

    metric_dirs = glob.glob(os.path.join(folder_path, "*"))

    for metric_dir in metric_dirs:

        summary_file = os.path.join(metric_dir, "ICA_level_summary_with_BH.tsv")
        if not os.path.exists(summary_file):
            continue

        metric_name = os.path.basename(metric_dir)

        df = pd.read_csv(summary_file, sep="\t")
        df["metric"] = metric_name
        df["ica"] = ica_number

        all_rows.append(df)

if len(all_rows) == 0:
    raise RuntimeError("No summary files found.")

global_df = pd.concat(all_rows, ignore_index=True)

print("\nMetrics:", sorted(global_df["metric"].unique()))
print("ICAs:", sorted(global_df["ica"].unique()))
print("Effects:", sorted(global_df["effect"].unique()))
print()

# =====================================================
# RECOMPUTE GLOBAL BH ACROSS ICAs (CORRECT WAY)
# =====================================================

global_df["global_bh_sig"] = 0
global_df["global_bh_p"] = np.nan

for (metric, effect), sub in global_df.groupby(["metric", "effect"]):

    pvals = sub["ica_p_simes"].values

    reject, p_corr, _, _ = multipletests(pvals, alpha=ALPHA, method="fdr_bh")

    global_df.loc[sub.index, "global_bh_sig"] = reject.astype(int)
    global_df.loc[sub.index, "global_bh_p"] = p_corr

print("Global BH across ICAs recomputed.\n")

# =====================================================
# 1️⃣ ICA VULNERABILITY SCORE
# =====================================================

ica_vulnerability = (
    global_df
    .groupby("ica", as_index=False)
    .agg(
        n_metrics_tested=("metric", "nunique"),
        total_sig_across_metrics=("global_bh_sig", "sum"),
        total_fdr_vox=("n_fdr_vox", "sum"),
        median_pr2_max=("pr2_max", "median"),
    )
)

ica_vulnerability["vulnerability_fraction"] = (
    ica_vulnerability["total_sig_across_metrics"] /
    ica_vulnerability["n_metrics_tested"]
)

ica_vulnerability = ica_vulnerability.sort_values(
    "total_sig_across_metrics",
    ascending=False
)

# =====================================================
# 2️⃣ NEUROCARDIAC BURDEN SCORE
# =====================================================

metric_burden = (
    global_df
    .groupby("metric", as_index=False)
    .agg(
        n_icas=("ica", "nunique"),
        total_fdr_vox=("n_fdr_vox", "sum"),
        total_voxels_tested=("n_voxels", "sum"),
        median_peak_z=("peak_abs_z", "median"),
        median_pr2_max=("pr2_max", "median"),
    )
)

metric_burden["burden_fraction"] = (
    metric_burden["total_fdr_vox"] /
    metric_burden["total_voxels_tested"]
)

metric_burden = metric_burden.sort_values(
    "burden_fraction",
    ascending=False
)

# =====================================================
# 3️⃣ CROSS-METRIC STABILITY INDEX
# =====================================================

effect_stability = (
    global_df
    .groupby("effect", as_index=False)
    .agg(
        n_metrics_tested=("metric", "nunique"),
        n_sig_instances=("global_bh_sig", "sum"),
        total_fdr_vox=("n_fdr_vox", "sum"),
        median_pr2_max=("pr2_max", "median"),
        median_peak_z=("peak_abs_z", "median"),
    )
)

n_total_tests = (
    global_df["metric"].nunique() *
    global_df["ica"].nunique()
)

effect_stability["stability_fraction"] = (
    effect_stability["n_sig_instances"] / n_total_tests
)

effect_stability = effect_stability.sort_values(
    "n_sig_instances",
    ascending=False
)

# =====================================================
# EFFECT × METRIC MATRIX
# =====================================================

effect_metric_matrix = (
    global_df
    .groupby(["effect", "metric"], as_index=False)
    .agg(
        n_ica_sig=("global_bh_sig", "sum"),
        total_fdr_vox=("n_fdr_vox", "sum"),
        median_pr2=("pr2_max", "median"),
    )
)

effect_metric_pivot_sig = effect_metric_matrix.pivot(
    index="effect",
    columns="metric",
    values="n_ica_sig"
).fillna(0)

effect_metric_pivot_burden = effect_metric_matrix.pivot(
    index="effect",
    columns="metric",
    values="total_fdr_vox"
).fillna(0)

# =====================================================
# EFFECT × ICA MATRIX
# =====================================================

effect_ica_matrix = (
    global_df
    .groupby(["effect", "ica"], as_index=False)
    .agg(
        n_metrics_sig=("global_bh_sig", "sum"),
        total_fdr_vox=("n_fdr_vox", "sum"),
        median_pr2=("pr2_max", "median"),
    )
)

effect_ica_pivot = effect_ica_matrix.pivot(
    index="effect",
    columns="ica",
    values="n_metrics_sig"
).fillna(0)

# =====================================================
# WRITE OUTPUT
# =====================================================

output_root = os.path.join(LEVEL2_ROOT, "GLOBAL_AGGREGATION")
os.makedirs(output_root, exist_ok=True)

global_df.to_csv(os.path.join(output_root, "GLOBAL_all_metrics_all_ICAs.tsv"), sep="\t", index=False)
ica_vulnerability.to_csv(os.path.join(output_root, "GLOBAL_ICA_vulnerability.tsv"), sep="\t", index=False)
metric_burden.to_csv(os.path.join(output_root, "GLOBAL_metric_burden.tsv"), sep="\t", index=False)
effect_stability.to_csv(os.path.join(output_root, "GLOBAL_effect_stability.tsv"), sep="\t", index=False)

effect_metric_matrix.to_csv(os.path.join(output_root, "GLOBAL_effect_metric_matrix_long.tsv"), sep="\t", index=False)
effect_metric_pivot_sig.to_csv(os.path.join(output_root, "GLOBAL_effect_metric_sig_matrix.tsv"), sep="\t")
effect_metric_pivot_burden.to_csv(os.path.join(output_root, "GLOBAL_effect_metric_burden_matrix.tsv"), sep="\t")

effect_ica_matrix.to_csv(os.path.join(output_root, "GLOBAL_effect_ica_matrix_long.tsv"), sep="\t", index=False)
effect_ica_pivot.to_csv(os.path.join(output_root, "GLOBAL_effect_ica_sig_matrix.tsv"), sep="\t")

# Excel summary
excel_path = os.path.join(
    output_root,
    f"Neurocardiac_Global_Summary_{datetime.today().strftime('%Y%m%d')}.xlsx"
)

with pd.ExcelWriter(excel_path) as writer:
    global_df.to_excel(writer, sheet_name="GLOBAL_overview", index=False)
    ica_vulnerability.to_excel(writer, sheet_name="ICA_vulnerability", index=False)
    metric_burden.to_excel(writer, sheet_name="Metric_burden", index=False)
    effect_stability.to_excel(writer, sheet_name="Effect_stability", index=False)
    effect_metric_pivot_sig.to_excel(writer, sheet_name="Effect_x_Metric_sig")
    effect_ica_pivot.to_excel(writer, sheet_name="Effect_x_ICA_sig")

print("\n=================================================")
print("GLOBAL AGGREGATION COMPLETE")
print("Saved to:", output_root)
print("=================================================\n")
