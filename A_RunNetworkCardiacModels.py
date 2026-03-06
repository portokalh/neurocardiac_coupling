#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:27:02 2026

@author: alex
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# ======================================================
# PATHS
# ======================================================

NETWORK_FILE = "/mnt/newStor/paros/paros_WORK/aashika/results/network12_from_errts/Network12_amplitudes.tsv"
META_FILE    = "/mnt/newStor/paros/paros_WORK/aashika/data/metadata/cardiac_design_updated3.csv"
OUTDIR       = "/mnt/newStor/paros/paros_WORK/aashika/results/network_cardiac_stats_allmetrics"
os.makedirs(OUTDIR, exist_ok=True)

# ======================================================
# CARDIAC METRICS (EXACT COLUMN NAMES)
# ======================================================

CARDIAC_METRICS = [
    "Diastolic_LV_Volume",
    "Systolic_LV_Volume",
    "Heart_Rate",
    "Stroke_Volume",
    "Ejection_Fraction",
    "Cardiac_Output",
    "Diastolic_RV",
    "Systolic_RV",
    "Diastolic_LA",
    "Systolic_LA",
    "Diastolic_RA",
    "Systolic_RA",
    "Diastolic_Myo",
    "Systolic_Myo",
]

# Optional: covariates if present in metadata (auto-detected)
POSSIBLE_COVARS = ["Age", "Sex", "Genotype", "APOE", "APOE4", "Diet", "Exercise"]

POSSIBLE_COVARS = ["Age", "Sex_Male", "Exercise_Yes", "Diet_HFD"]

MIN_N = 8  # minimum sample size per regression

# ======================================================
# LOAD DATA
# ======================================================

print("Loading network amplitudes:", NETWORK_FILE)
nets = pd.read_csv(NETWORK_FILE, sep="\t")

print("Loading metadata:", META_FILE)
meta = pd.read_csv(META_FILE)

# Clean column names
nets.columns = nets.columns.str.strip()
meta.columns = meta.columns.str.strip()

# Ensure ID is string
if "Arunno" not in meta.columns:
    raise ValueError(f"'Arunno' not found in metadata columns: {meta.columns.tolist()}")
if "Arunno" not in nets.columns:
    raise ValueError(f"'Arunno' not found in network file columns: {nets.columns.tolist()}")

meta["Arunno"] = meta["Arunno"].astype(str).str.strip()
nets["Arunno"] = nets["Arunno"].astype(str).str.strip()

# Ensure cardiac metrics exist and are numeric
missing = [c for c in CARDIAC_METRICS if c not in meta.columns]
if missing:
    raise ValueError(f"Missing cardiac metrics in metadata: {missing}")

for c in CARDIAC_METRICS:
    meta[c] = pd.to_numeric(meta[c], errors="coerce")

# ======================================================
# MERGE
# ======================================================

df = pd.merge(meta, nets, on="Arunno", how="inner")
print("Subjects after merge:", len(df))

# Identify network amplitude columns
network_cols = [c for c in df.columns if c.startswith("Amp_")]
if len(network_cols) == 0:
    raise ValueError("No network columns found (expected columns starting with 'Amp_').")

print("Networks found:", len(network_cols))
print("Example network cols:", network_cols[:5])

# Identify covariates that exist
covars = [c for c in POSSIBLE_COVARS if c in df.columns]
print("Covariates used:", covars)

####standardization of metrics ###

scaler = StandardScaler()

# z-score cardiac metrics
df[CARDIAC_METRICS] = scaler.fit_transform(df[CARDIAC_METRICS])

# z-score network amplitudes
df[network_cols] = scaler.fit_transform(df[network_cols])

# ======================================================
# RUN MODELS
# ======================================================

results = []

for cardiac in CARDIAC_METRICS:
    print("\nTesting cardiac metric:", cardiac)

    for net in network_cols:
        cols = ["Arunno", cardiac, net] + covars
        subdf = df[cols].dropna()
        n = len(subdf)

        if n < MIN_N:
            continue

        rhs = " + ".join([net] + covars) if covars else net
        formula = f"{cardiac} ~ {rhs}"

        model = smf.ols(formula, data=subdf).fit()

        results.append({
            "CardiacMetric": cardiac,
            "Network": net,
            "N": n,
            "Beta": model.params.get(net, np.nan),
            "SE": model.bse.get(net, np.nan),
            "T": model.tvalues.get(net, np.nan),
            "P": model.pvalues.get(net, np.nan),
            "R2": model.rsquared,
            "Adj_R2": model.rsquared_adj
        })

res = pd.DataFrame(results).dropna(subset=["P"])

# ======================================================
# MULTIPLE COMPARISON CORRECTION (GLOBAL)
# ======================================================

res["FDR_global"] = multipletests(res["P"].values, method="fdr_bh")[1]
res = res.sort_values(["FDR_global", "P"])

# ======================================================
# SAVE TABLE
# ======================================================

table_out = os.path.join(OUTDIR, "network_cardiac_results_allmetrics.tsv")
res.to_csv(table_out, sep="\t", index=False)
print("\nSaved results:", table_out)

# ======================================================
# VOLCANO PLOTS (PER METRIC)
# ======================================================

for cardiac in CARDIAC_METRICS:
    sub = res[res["CardiacMetric"] == cardiac].copy()
    if len(sub) == 0:
        continue

    plt.figure(figsize=(6, 5))
    plt.scatter(sub["Beta"], -np.log10(sub["P"]), s=40)

    plt.axhline(-np.log10(0.05), linestyle="--")
    plt.xlabel("Effect size (beta for network amplitude)")
    plt.ylabel("-log10(p)")
    plt.title(cardiac)

    # Annotate FDR significant hits
    sig = sub[sub["FDR_global"] < 0.05]
    for _, r in sig.iterrows():
        plt.text(r["Beta"], -np.log10(r["P"]), r["Network"], fontsize=7)

    fig_out = os.path.join(OUTDIR, f"volcano_{cardiac}.png")
    plt.tight_layout()
    plt.savefig(fig_out, dpi=300)
    plt.close()

print("Volcano plots saved to:", OUTDIR)