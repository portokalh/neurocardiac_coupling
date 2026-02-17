#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 18:02:11 2026

@author: alex
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from datetime import datetime

# =====================================================
# SETTINGS
# =====================================================

GLOBAL_ROOT = "/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/myICA_separate100_clean_0p1/level2/GLOBAL_AGGREGATION"
CLUSTER_DIR = os.path.join(GLOBAL_ROOT, "NEUROCARDIAC_CLUSTERED_PANELS")
output_root = GLOBAL_ROOT
PLOTS_DIR = os.path.join(GLOBAL_ROOT, "FINAL_NEUROCARDIAC_PLOTS")
os.makedirs(PLOTS_DIR, exist_ok=True)

LOG_EPS = 1
DPI = 300

# =====================================================
# LOAD DATA
# =====================================================

burden = pd.read_csv(
    os.path.join(GLOBAL_ROOT, "GLOBAL_effect_metric_burden_matrix.tsv"),
    sep="\t", index_col=0
)

sig = pd.read_csv(
    os.path.join(GLOBAL_ROOT, "GLOBAL_effect_metric_sig_matrix.tsv"),
    sep="\t", index_col=0
)

# Load full long-format global table
global_df = pd.read_csv(
    os.path.join(GLOBAL_ROOT, "GLOBAL_all_metrics_all_ICAs.tsv"),
    sep="\t"
)

# =====================================================
# CLEAN MATRIX FOR CLUSTERING
# =====================================================

burden = burden.fillna(0)
burden = burden.replace([np.inf, -np.inf], 0)

log_burden = np.log10(burden + LOG_EPS)

# Drop zero-variance rows (cannot cluster)
row_var = log_burden.var(axis=1)
log_burden = log_burden.loc[row_var > 0]

# =====================================================
# 1️⃣ HIERARCHICAL CLUSTER MAP (SAFE)
# =====================================================

sns.clustermap(
    log_burden,
    cmap="mako",
    method="average",
    metric="euclidean",
    figsize=(12, 8)
)

plt.savefig(
    os.path.join(PLOTS_DIR, "Clustered_logFDR_voxels.png"),
    dpi=DPI
)
plt.close()

# =====================================================
# 2️⃣ CLUSTERED SIGNIFICANT ICA COUNTS
# =====================================================

sig = sig.fillna(0)

sns.clustermap(
    sig,
    cmap="viridis",
    method="average",
    metric="euclidean",
    figsize=(12, 8)
)

plt.savefig(
    os.path.join(PLOTS_DIR, "Clustered_sigICA_counts.png"),
    dpi=DPI
)
plt.close()

# =====================================================
# 3️⃣ SEPARATE PANELS
# =====================================================

intervention = ["Exercise_ec"]
physiology = ["metric_z"]
risk = ["Age_cont","APOE_E4","APOE_E2","Diet_ec","Sex_ec","HN_ec"]

intervention = [x for x in intervention if x in burden.index]
physiology = [x for x in physiology if x in burden.index]
risk = [x for x in risk if x in burden.index]

# ---- Exercise vs Risk Panel ----

panel = burden.loc[intervention + risk]
panel_log = np.log10(panel + LOG_EPS)

plt.figure(figsize=(14,6))
sns.heatmap(panel_log, cmap="mako", linewidths=0.5)
plt.title("Exercise vs Risk Factors (log10 FDR voxels)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "Panel_Exercise_vs_Risk.png"), dpi=DPI)
plt.close()

# ---- Physiology Panel ----

if len(physiology) > 0:
    phys = burden.loc[physiology]
    phys_log = np.log10(phys + LOG_EPS)

    plt.figure(figsize=(14,3))
    sns.heatmap(phys_log, cmap="rocket", linewidths=0.5)
    plt.title("Cardiac Physiology Coupling (metric_z)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "Panel_metricZ.png"), dpi=DPI)
    plt.close()

# =====================================================
# 4️⃣ INTERPRETABLE ORDERED CARPET
# =====================================================

ordered_effects = physiology + intervention + risk
ordered_effects = [e for e in ordered_effects if e in burden.index]

ordered = burden.loc[ordered_effects]
ordered_log = np.log10(ordered + LOG_EPS)

plt.figure(figsize=(14,6))
sns.heatmap(ordered_log, cmap="mako", linewidths=0.5)
plt.title("Ordered Neurocardiac Impact Carpet")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "Ordered_Carpet.png"), dpi=DPI)
plt.close()


# =====================================================
# =====================================================
# 🧠 NEW SECTION: NEUROCARDIAC COMPOSITE PANELS
# =====================================================
# =====================================================

print("\n============================================")
print("Building Neurocardiac Composite Panels...")
print("============================================\n")

COMPOSITE_DIR = os.path.join(GLOBAL_ROOT, "NEUROCARDIAC_COMPOSITE")
os.makedirs(COMPOSITE_DIR, exist_ok=True)

# -----------------------------------------------------
# 1️⃣ Compute % ICA Coverage
# -----------------------------------------------------

global_df["coverage_fraction"] = (
    global_df["n_fdr_vox"] / global_df["n_voxels"]
)

# -----------------------------------------------------
# 2️⃣ Compute Neurocardiac Impact Score
# -----------------------------------------------------

global_df["neurocardiac_impact"] = (
    global_df["coverage_fraction"] * global_df["pr2_mean"]
)

# -----------------------------------------------------
# Aggregate across ICAs
# -----------------------------------------------------

composite_df = (
    global_df
    .groupby(["effect", "metric"], as_index=False)
    .agg(
        mean_coverage=("coverage_fraction", "mean"),
        median_pr2=("pr2_mean", "median"),
        impact_sum=("neurocardiac_impact", "sum"),
    )
)

# -----------------------------------------------------
# Pivot matrices for panels
# -----------------------------------------------------

coverage_mat = composite_df.pivot(
    index="effect",
    columns="metric",
    values="mean_coverage"
).fillna(0)

pr2_mat = composite_df.pivot(
    index="effect",
    columns="metric",
    values="median_pr2"
).fillna(0)

impact_mat = composite_df.pivot(
    index="effect",
    columns="metric",
    values="impact_sum"
).fillna(0)

# =====================================================
# PANEL 1 — % ICA Coverage
# =====================================================

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(14,6))
sns.heatmap(coverage_mat, cmap="mako", linewidths=0.5)
plt.title("Panel 1 — Mean % ICA Coverage")
plt.tight_layout()
plt.savefig(os.path.join(COMPOSITE_DIR, "Panel1_percent_ICA_coverage.png"), dpi=300)
plt.close()

# =====================================================
# PANEL 2 — Median Partial R²
# =====================================================

plt.figure(figsize=(14,6))
sns.heatmap(pr2_mat, cmap="rocket", linewidths=0.5)
plt.title("Panel 2 — Median Partial R²")
plt.tight_layout()
plt.savefig(os.path.join(COMPOSITE_DIR, "Panel2_median_partialR2.png"), dpi=300)
plt.close()

# =====================================================
# PANEL 3 — Neurocardiac Impact Score
# =====================================================

plt.figure(figsize=(14,6))
sns.heatmap(impact_mat, cmap="viridis", linewidths=0.5)
plt.title("Panel 3 — Neurocardiac Impact Score (Coverage × Magnitude)")
plt.tight_layout()
plt.savefig(os.path.join(COMPOSITE_DIR, "Panel3_neurocardiac_impact.png"), dpi=300)
plt.close()

# -----------------------------------------------------
# Save composite table
# -----------------------------------------------------

composite_df.to_csv(
    os.path.join(COMPOSITE_DIR, "Neurocardiac_Composite_Table.tsv"),
    sep="\t",
    index=False
)

print("\n============================================")
print("NEUROCARDIAC COMPOSITE PANELS COMPLETE")
print("Saved to:", COMPOSITE_DIR)
print("============================================\n")


print("\n=================================================")
print("FINAL NEUROCARDIAC VISUALIZATION COMPLETE")
print("Saved to:", PLOTS_DIR)
print("=================================================\n")
# =====================================================
# =====================================================
# 🔥 NEW SECTION: CLUSTERED PANELS + DOMINANCE + RESILIENCE
# =====================================================
# =====================================================

print("\n============================================")
print("Building clustered panels + dominance + resilience indices...")
print("============================================\n")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

CLUSTER_DIR = os.path.join(output_root, "NEUROCARDIAC_CLUSTERED_PANELS")
os.makedirs(CLUSTER_DIR, exist_ok=True)

EPS = 1e-12

# -----------------------------------------------------
# Define effect groups (edit if you change your model)
# -----------------------------------------------------
EFFECT_EXERCISE = ["Exercise_ec"]

# "Risk factors" here = all non-exercise effects EXCLUDING the coupling term metric_z (unless you want it included).
# You can choose to include metric_z in risk, but conceptually it's the cardiac-metric coupling term, not a "risk factor".
EFFECT_RISK = ["APOE_E2", "APOE_E4", "Age_cont", "Sex_ec", "Diet_ec", "HN_ec"]

# If you want metric_z treated as "coupling", keep it separate:
EFFECT_COUPLING = ["metric_z"]

# -----------------------------------------------------
# Helper: safe matrix for clustering
# - replaces NaN/Inf with 0
# - optional log1p for heavy-tailed counts / impact measures
# -----------------------------------------------------
def _finite_matrix(mat: pd.DataFrame) -> pd.DataFrame:
    m = mat.copy()
    m = m.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return m

def _cluster_order(mat: pd.DataFrame, axis: str = "rows", metric: str = "euclidean", method: str = "average"):
    """
    Returns ordering index list for rows or columns using scipy linkage.
    Works even if seaborn clustermap fails due to non-finite values.
    """
    m = _finite_matrix(mat)

    if axis == "rows":
        X = m.values
        labels = list(m.index)
    elif axis == "cols":
        X = m.values.T
        labels = list(m.columns)
    else:
        raise ValueError("axis must be 'rows' or 'cols'")

    if X.shape[0] <= 1:
        return labels

    # pdist requires finite values; we ensured this above
    d = pdist(X, metric=metric)
    Z = linkage(d, method=method)
    order = leaves_list(Z)
    return [labels[i] for i in order]

def _save_heatmap(mat: pd.DataFrame, title: str, out_png: str, log1p: bool = False):
    m = _finite_matrix(mat)
    if log1p:
        m = np.log1p(m)

    plt.figure(figsize=(14, 6))
    sns.heatmap(m, linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

# -----------------------------------------------------
# We will use the composite matrices you created earlier:
# coverage_mat, pr2_mat, impact_mat
# If those live under different variable names, map them here.
# -----------------------------------------------------

# Safety: ensure they exist
for _name in ["coverage_mat", "pr2_mat", "impact_mat"]:
    if _name not in globals():
        raise RuntimeError(f"Expected '{_name}' to exist. Make sure you ran the COMPOSITE section first.")

coverage_mat2 = _finite_matrix(coverage_mat)
pr2_mat2 = _finite_matrix(pr2_mat)
impact_mat2 = _finite_matrix(impact_mat)

# -----------------------------------------------------
# 1️⃣ HIERARCHICAL CLUSTERING ORDER (use impact as the driver)
# -----------------------------------------------------
row_order = _cluster_order(impact_mat2, axis="rows")
col_order = _cluster_order(impact_mat2, axis="cols")

coverage_ord = coverage_mat2.loc[row_order, col_order]
pr2_ord      = pr2_mat2.loc[row_order, col_order]
impact_ord   = impact_mat2.loc[row_order, col_order]

# Save clustered versions (full)
coverage_ord.to_csv(os.path.join(CLUSTER_DIR, "CLUSTERED_coverage_matrix.tsv"), sep="\t")
pr2_ord.to_csv(os.path.join(CLUSTER_DIR, "CLUSTERED_pr2_matrix.tsv"), sep="\t")
impact_ord.to_csv(os.path.join(CLUSTER_DIR, "CLUSTERED_impact_matrix.tsv"), sep="\t")

_save_heatmap(
    coverage_ord,
    "Clustered Panel A — Mean % ICA Coverage (ordered by Impact clustering)",
    os.path.join(CLUSTER_DIR, "Clustered_A_coverage.png"),
    log1p=False
)

_save_heatmap(
    pr2_ord,
    "Clustered Panel B — Median partial R² (ordered by Impact clustering)",
    os.path.join(CLUSTER_DIR, "Clustered_B_partialR2.png"),
    log1p=False
)

# impact can be heavy-tailed → log1p is usually nicer
_save_heatmap(
    impact_ord,
    "Clustered Panel C — Neurocardiac Impact Score (log1p; ordered by Impact clustering)",
    os.path.join(CLUSTER_DIR, "Clustered_C_impact_log1p.png"),
    log1p=True
)

# -----------------------------------------------------
# 2️⃣ SPLIT: Exercise vs Risk factors (separate panels)
# -----------------------------------------------------
def _subset_effects(mat: pd.DataFrame, effects: list, label: str):
    existing = [e for e in effects if e in mat.index]
    if len(existing) == 0:
        print(f"WARNING: none of {label} effects found in matrix index.")
        return None
    return mat.loc[existing, :]

impact_ex = _subset_effects(impact_ord, EFFECT_EXERCISE, "Exercise")
impact_rk = _subset_effects(impact_ord, EFFECT_RISK, "Risk")

if impact_ex is not None:
    _save_heatmap(
        impact_ex,
        "Exercise-only Panel — Impact (log1p)",
        os.path.join(CLUSTER_DIR, "Exercise_only_impact_log1p.png"),
        log1p=True
    )

if impact_rk is not None:
    _save_heatmap(
        impact_rk,
        "Risk-factors Panel — Impact (log1p)",
        os.path.join(CLUSTER_DIR, "Risk_only_impact_log1p.png"),
        log1p=True
    )

# Optional: coupling panel (metric_z) separate
impact_cp = _subset_effects(impact_ord, EFFECT_COUPLING, "Coupling(metric_z)")
if impact_cp is not None:
    _save_heatmap(
        impact_cp,
        "Coupling Panel — metric_z Impact (log1p)",
        os.path.join(CLUSTER_DIR, "Coupling_metric_z_impact_log1p.png"),
        log1p=True
    )

# -----------------------------------------------------
# 3️⃣ DIASTOLIC vs SYSTOLIC DOMINANCE INDEX
#    (computed per effect, using IMPACT across metrics)
# -----------------------------------------------------
metrics_all = list(impact_ord.columns)

diastolic_metrics = [m for m in metrics_all if m.startswith("Diastolic_")]
systolic_metrics  = [m for m in metrics_all if m.startswith("Systolic_")]

# If you want to include "Myo" as diastolic-like or separate, it already falls under Diastolic_Myo.
# Non-syst/diast metrics (Heart_Rate, Stroke_Volume, Cardiac_Output, Ejection_Fraction) are ignored for this dominance index.

dominance_rows = []
for eff in impact_ord.index:
    d_sum = float(impact_ord.loc[eff, diastolic_metrics].sum()) if len(diastolic_metrics) else 0.0
    s_sum = float(impact_ord.loc[eff, systolic_metrics].sum())  if len(systolic_metrics)  else 0.0

    # dominance in [-1, +1]
    dom = (d_sum - s_sum) / (d_sum + s_sum + EPS)

    dominance_rows.append({
        "effect": eff,
        "impact_diastolic_sum": d_sum,
        "impact_systolic_sum": s_sum,
        "diastolic_vs_systolic_dominance": dom
    })

dominance_df = pd.DataFrame(dominance_rows).sort_values(
    "diastolic_vs_systolic_dominance", ascending=False
)
dominance_df.to_csv(os.path.join(CLUSTER_DIR, "Diastolic_vs_Systolic_Dominance_byEffect.tsv"), sep="\t", index=False)

# Quick barplot
plt.figure(figsize=(10, 4))
sns.barplot(
    data=dominance_df,
    x="diastolic_vs_systolic_dominance",
    y="effect"
)
plt.title("Diastolic vs Systolic Dominance (Impact-weighted; + = diastolic-dominant)")
plt.tight_layout()
plt.savefig(os.path.join(CLUSTER_DIR, "Diastolic_vs_Systolic_Dominance_byEffect.png"), dpi=300)
plt.close()

# -----------------------------------------------------
# 4️⃣ RESILIENCE INDEX = Exercise ÷ Risk burden
#    We'll compute it per metric and per ICA in TWO WAYS:
#    A) Using the impact matrix collapsed across ICAs (effect×metric impact_sum)
#    B) Using the raw global_df to compute per-ICA resilience
# -----------------------------------------------------

# ---- A) per metric resilience (collapsed across ICAs)
# Exercise impact per metric:
if "Exercise_ec" in impact_mat2.index:
    ex_by_metric = impact_mat2.loc["Exercise_ec", :].copy()
else:
    ex_by_metric = pd.Series(0.0, index=impact_mat2.columns)

risk_present = [e for e in EFFECT_RISK if e in impact_mat2.index]
if len(risk_present) > 0:
    risk_by_metric = impact_mat2.loc[risk_present, :].sum(axis=0)
else:
    risk_by_metric = pd.Series(0.0, index=impact_mat2.columns)

resilience_by_metric = (ex_by_metric + EPS) / (risk_by_metric + EPS)

resilience_metric_df = pd.DataFrame({
    "metric": resilience_by_metric.index,
    "exercise_impact": ex_by_metric.values,
    "risk_impact_sum": risk_by_metric.values,
    "resilience_index_ex_over_risk": resilience_by_metric.values
}).sort_values("resilience_index_ex_over_risk", ascending=False)

resilience_metric_df.to_csv(os.path.join(CLUSTER_DIR, "Resilience_Index_byMetric.tsv"), sep="\t", index=False)

plt.figure(figsize=(12, 4))
sns.barplot(
    data=resilience_metric_df,
    x="resilience_index_ex_over_risk",
    y="metric"
)
plt.title("Resilience Index by Metric = Exercise Impact / Risk Impact")
plt.tight_layout()
plt.savefig(os.path.join(CLUSTER_DIR, "Resilience_Index_byMetric.png"), dpi=300)
plt.close()

# ---- B) per ICA resilience using raw global_df (more granular)
# define "impact" per row in global_df if not already present
if "coverage_fraction" not in global_df.columns:
    global_df["coverage_fraction"] = global_df["n_fdr_vox"] / global_df["n_voxels"]
if "neurocardiac_impact" not in global_df.columns:
    global_df["neurocardiac_impact"] = global_df["coverage_fraction"] * global_df["pr2_mean"]

# aggregate per ICA:
ica_ex = (
    global_df[global_df["effect"].isin(EFFECT_EXERCISE)]
    .groupby("ica", as_index=False)["neurocardiac_impact"].sum()
    .rename(columns={"neurocardiac_impact": "exercise_impact_sum"})
)

ica_rk = (
    global_df[global_df["effect"].isin(EFFECT_RISK)]
    .groupby("ica", as_index=False)["neurocardiac_impact"].sum()
    .rename(columns={"neurocardiac_impact": "risk_impact_sum"})
)

ica_res = pd.merge(ica_ex, ica_rk, on="ica", how="outer").fillna(0.0)
ica_res["resilience_index_ex_over_risk"] = (ica_res["exercise_impact_sum"] + EPS) / (ica_res["risk_impact_sum"] + EPS)
ica_res = ica_res.sort_values("resilience_index_ex_over_risk", ascending=False)

ica_res.to_csv(os.path.join(CLUSTER_DIR, "Resilience_Index_byICA.tsv"), sep="\t", index=False)

plt.figure(figsize=(10, 4))
sns.barplot(
    data=ica_res,
    x="resilience_index_ex_over_risk",
    y="ica"
)
plt.title("Resilience Index by ICA = Exercise Impact / Risk Impact")
plt.tight_layout()
plt.savefig(os.path.join(CLUSTER_DIR, "Resilience_Index_byICA.png"), dpi=300)
plt.close()

print("\n============================================")
print("CLUSTERED PANELS + DOMINANCE + RESILIENCE COMPLETE")
print("Saved to:", CLUSTER_DIR)
print("============================================\n")


# =====================================================
# 🔬 DIASTOLIC vs SYSTOLIC STATISTICAL TEST
# =====================================================

from scipy.stats import ttest_rel, wilcoxon
import numpy as np

print("\n============================================")
print("Testing Diastolic vs Systolic Dominance (paired across ICAs)")
print("============================================\n")

EPS = 1e-12

# Define metric groups
diastolic_metrics = [m for m in global_df["metric"].unique() if m.startswith("Diastolic_")]
systolic_metrics  = [m for m in global_df["metric"].unique() if m.startswith("Systolic_")]

effects = global_df["effect"].unique()

dominance_results = []

for eff in effects:

    df_eff = global_df[global_df["effect"] == eff]

    # Aggregate impact per ICA
    ica_summary = (
        df_eff
        .assign(impact=lambda x: (x["n_fdr_vox"] / x["n_voxels"]) * x["pr2_mean"])
        .groupby(["ica", "metric"], as_index=False)["impact"]
        .sum()
    )

    # Pivot to ICA × metric
    pivot = ica_summary.pivot(index="ica", columns="metric", values="impact").fillna(0)

    # Compute per-ICA diastolic and systolic sums
    d_vals = pivot[diastolic_metrics].sum(axis=1)
    s_vals = pivot[systolic_metrics].sum(axis=1)

    # Paired tests
    t_stat, t_p = ttest_rel(d_vals, s_vals)
    try:
        w_stat, w_p = wilcoxon(d_vals - s_vals)
    except:
        w_p = np.nan

    dominance_results.append({
        "effect": eff,
        "mean_diastolic_impact": d_vals.mean(),
        "mean_systolic_impact": s_vals.mean(),
        "mean_difference": (d_vals - s_vals).mean(),
        "t_paired_pvalue": t_p,
        "wilcoxon_pvalue": w_p
    })

dominance_test_df = pd.DataFrame(dominance_results).sort_values("t_paired_pvalue")

dominance_test_df.to_csv(
    os.path.join(CLUSTER_DIR, "Diastolic_vs_Systolic_Significance.tsv"),
    sep="\t",
    index=False
)

print(dominance_test_df)
