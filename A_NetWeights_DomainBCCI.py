#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Brain–Cardiac Coupling Analysis Pipeline

Outputs:
- Network weights
- Cross-validated BCCI scores
- Network statistics with FDR correction
- Model performance statistics
- Brain network → cardiac domain heatmap
- Clustered network coupling heatmap
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from statsmodels.stats.multitest import multipletests


# ======================================================
# PATHS
# ======================================================

NETWORK_FILE = "/mnt/newStor/paros/paros_WORK/aashika/results/network12_from_errts/Network12_amplitudes.tsv"
META_FILE    = "/mnt/newStor/paros/paros_WORK/aashika/data/metadata/cardiac_design_updated3.csv"
OUTDIR       = "/mnt/newStor/paros/paros_WORK/aashika/results/bcci_domains"

os.makedirs(OUTDIR, exist_ok=True)


# ======================================================
# CARDIAC METRICS
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
    "Systolic_Myo"
]

POSSIBLE_COVARS = ["Age", "Sex_Male", "Exercise_Yes", "Diet_HFD"]


# ======================================================
# LOAD DATA
# ======================================================

print("Loading data")

nets = pd.read_csv(NETWORK_FILE, sep="\t")
meta = pd.read_csv(META_FILE)

nets.columns = nets.columns.str.strip()
meta.columns = meta.columns.str.strip()

nets["Arunno"] = nets["Arunno"].astype(str).str.strip()
meta["Arunno"] = meta["Arunno"].astype(str).str.strip()

df = pd.merge(meta, nets, on="Arunno", how="inner")

print("Subjects after merge:", len(df))


# ======================================================
# IDENTIFY NETWORK COLUMNS
# ======================================================

network_cols = [c for c in df.columns if c.startswith("Amp_")]
print("Networks detected:", len(network_cols))

covars = [c for c in POSSIBLE_COVARS if c in df.columns]
print("Covariates used:", covars)


# ======================================================
# STANDARDIZATION
# ======================================================

print("Standardizing variables")

scaler = StandardScaler()

df[CARDIAC_METRICS] = scaler.fit_transform(df[CARDIAC_METRICS])
df[network_cols]    = scaler.fit_transform(df[network_cols])


# ======================================================
# DEFINE CARDIAC DOMAINS
# ======================================================

print("Constructing cardiac domains")

df["rate_control_z"] = df["Heart_Rate"]

df["systolic_function_z"] = df[
    ["Stroke_Volume", "Ejection_Fraction", "Cardiac_Output"]
].mean(axis=1)

df["diastolic_function_z"] = df[
    ["Diastolic_LV_Volume", "Diastolic_RV", "Diastolic_LA", "Diastolic_RA"]
].mean(axis=1)

DOMAINS = [
    "rate_control_z",
    "systolic_function_z",
    "diastolic_function_z"
]


# ======================================================
# MULTIVARIATE NETWORK MODELS
# ======================================================

print("\nEstimating network weights")

weights = {}
models  = {}

for domain in DOMAINS:

    cols = [domain] + network_cols + covars
    subdf = df[cols].dropna()

    y = subdf[domain]
    X = subdf[network_cols + covars]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    models[domain] = model
    weights[domain] = model.params[network_cols]

    weights[domain].to_csv(
        os.path.join(OUTDIR, f"weights_{domain}.tsv"),
        sep="\t"
    )


# ======================================================
# CROSS-VALIDATED BCCI
# ======================================================

print("\nComputing cross-validated BCCI")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for domain in DOMAINS:

    df[f"BCCI_{domain}_cv"] = np.nan

    cols = [domain] + network_cols + covars
    subdf = df[cols].dropna()

    for train_idx, test_idx in kf.split(subdf):

        train = subdf.iloc[train_idx]
        test  = subdf.iloc[test_idx]

        y_train = train[domain]
        X_train = sm.add_constant(train[network_cols + covars])

        model = sm.OLS(y_train, X_train).fit()

        beta = model.params[network_cols]

        scores = test[network_cols].dot(beta)

        df.loc[test.index, f"BCCI_{domain}_cv"] = scores

    col = f"BCCI_{domain}_cv"

    df[col + "_z"] = (df[col] - df[col].mean()) / df[col].std()


# ======================================================
# SAVE SUBJECT SCORES
# ======================================================

df.to_csv(
    os.path.join(OUTDIR, "BCCI_subject_scores.tsv"),
    sep="\t",
    index=False
)

print("Saved BCCI scores")


# ======================================================
# NETWORK STATISTICS
# ======================================================

print("\nComputing network statistics")

stats_rows = []

for domain in DOMAINS:

    model = models[domain]

    for net in network_cols:

        stats_rows.append({
            "Domain": domain,
            "Network": net.replace("Amp_", ""),
            "Beta": model.params.get(net),
            "SE": model.bse.get(net),
            "T": model.tvalues.get(net),
            "P": model.pvalues.get(net)
        })

stats_df = pd.DataFrame(stats_rows)

stats_df["FDR"] = multipletests(
    stats_df["P"], method="fdr_bh"
)[1]

stats_df.to_csv(
    os.path.join(OUTDIR, "network_weight_statistics.tsv"),
    sep="\t",
    index=False
)

print("Saved network statistics")


# ======================================================
# MODEL PERFORMANCE
# ======================================================

model_stats = []

for domain in DOMAINS:

    model = models[domain]

    model_stats.append({
        "Domain": domain,
        "N": model.nobs,
        "R2": model.rsquared,
        "Adj_R2": model.rsquared_adj,
        "F": model.fvalue,
        "P": model.f_pvalue
    })

model_stats = pd.DataFrame(model_stats)

model_stats.to_csv(
    os.path.join(OUTDIR, "domain_model_statistics.tsv"),
    sep="\t",
    index=False
)

print("Saved model statistics")


# ======================================================
# HEATMAP: NETWORK → CARDIAC DOMAIN
# ======================================================

print("\nCreating heatmap")

weight_matrix = pd.DataFrame(weights)
weight_matrix.index = weight_matrix.index.str.replace("Amp_", "", regex=False)

plt.figure(figsize=(7,9))

sns.heatmap(
    weight_matrix,
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f",
    linewidths=0.5
)

plt.title("Brain Network → Cardiac Domain Coupling")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTDIR, "brain_cardiac_heatmap.png"),
    dpi=300
)

plt.close()


# ======================================================
# CLUSTERED HEATMAP
# ======================================================

print("Creating clustered heatmap")

sns.clustermap(
    weight_matrix,
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    figsize=(8,10)
)

plt.savefig(
    os.path.join(OUTDIR, "brain_cardiac_clustermap.png"),
    dpi=300
)

plt.close()

print("\nPipeline completed successfully")
