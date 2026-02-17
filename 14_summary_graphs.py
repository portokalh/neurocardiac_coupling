#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 15:11:37 2026

@author: alex
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu

# -----------------------------
# USER SETTINGS
# -----------------------------
OUTPUT_DIR = "/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/myICA_separate100_clean_0p1/level2/GLOBAL_AGGREGATION"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load your master table produced by aggregation
MASTER_TSV = os.path.join(OUTPUT_DIR, "GLOBAL_all_metrics_all_ICAs.tsv")
df = pd.read_csv(MASTER_TSV, sep="\t")

# -----------------------------
# Define diastolic/systolic groups
# -----------------------------
DIASTOLIC = {
    "Diastolic_LV_Volume", "Diastolic_RV", "Diastolic_LA", "Diastolic_RA", "Diastolic_Myo"
}
SYSTOLIC = {
    "Systolic_LV_Volume", "Systolic_RV", "Systolic_LA", "Systolic_RA"
}

def zscore_series(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return x * 0.0
    return (x - mu) / sd

# ============================================================
# 1) METRIC-LEVEL BURDEN TABLE + DIASTOLIC VS SYSTOLIC TEST
# ============================================================
metric_burden = (
    df.groupby("metric", as_index=False)
      .agg(
          n_icas=("ica", "nunique"),
          n_effects=("effect", "nunique"),
          total_fdr_vox=("n_fdr_vox", "sum"),
          total_voxels=("n_voxels", "sum"),
          median_pr2_max=("pr2_max", "median"),
          median_peak_abs_z=("peak_abs_z", "median"),
          # fraction of tests significant at ICA-level BH across ICAs
          hit_rate=("ica_significant_bh_acrossICAs", "mean"),
      )
)

metric_burden["burden_fraction"] = metric_burden["total_fdr_vox"] / metric_burden["total_voxels"]

def phase_label(m):
    if m in DIASTOLIC:
        return "Diastolic"
    if m in SYSTOLIC:
        return "Systolic"
    return "Other"

metric_burden["phase"] = metric_burden["metric"].map(phase_label)

metric_burden_path = os.path.join(OUTPUT_DIR, "SUMMARY_metric_burden.tsv")
metric_burden.to_csv(metric_burden_path, sep="\t", index=False)

# --- dominance tests only on diastolic vs systolic ---
dia_vals = metric_burden.loc[metric_burden["phase"] == "Diastolic", "burden_fraction"].values
sys_vals = metric_burden.loc[metric_burden["phase"] == "Systolic", "burden_fraction"].values

print("\n=== DIASTOLIC vs SYSTOLIC DOMINANCE TEST ===")
print("Using metric-level burden_fraction = total_fdr_vox / total_voxels")
print(f"Diastolic metrics (n={len(dia_vals)}): {sorted(DIASTOLIC)}")
print(f"Systolic metrics  (n={len(sys_vals)}): {sorted(SYSTOLIC)}")

if len(dia_vals) >= 2 and len(sys_vals) >= 2:
    # Mann-Whitney U
    u_stat, p_mwu = mannwhitneyu(dia_vals, sys_vals, alternative="two-sided")
    print(f"Mann–Whitney U: U={u_stat:.3f}, p={p_mwu:.4g}")
else:
    print("Not enough metrics in each group for MWU test.")

# Permutation test (robust, recommended)
rng = np.random.default_rng(123)
def perm_test(a, b, n_perm=20000):
    a = np.asarray(a); b = np.asarray(b)
    obs = np.mean(a) - np.mean(b)
    pool = np.concatenate([a, b])
    na = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        aa = pool[:na]
        bb = pool[na:]
        stat = np.mean(aa) - np.mean(bb)
        if abs(stat) >= abs(obs):
            count += 1
    p = (count + 1) / (n_perm + 1)
    return obs, p

if len(dia_vals) >= 2 and len(sys_vals) >= 2:
    obs_diff, p_perm = perm_test(dia_vals, sys_vals, n_perm=20000)
    print(f"Permutation test (mean diff): diff={obs_diff:.6g}, p={p_perm:.4g}")

# ============================================================
# 2) COMPOSITE NEUROCARDIAC IMPACT SCORE (metric-level)
# ============================================================
# Components: burden_fraction, hit_rate, median_pr2_max, median_peak_abs_z
score_df = metric_burden.copy()

score_df["z_burden"] = zscore_series(score_df["burden_fraction"])
score_df["z_hit"] = zscore_series(score_df["hit_rate"])
score_df["z_pr2"] = zscore_series(score_df["median_pr2_max"])
score_df["z_peakz"] = zscore_series(score_df["median_peak_abs_z"])

# Equal weights (simple + defensible)
score_df["impact_score"] = score_df[["z_burden", "z_hit", "z_pr2", "z_peakz"]].mean(axis=1)

score_df = score_df.sort_values("impact_score", ascending=False)

score_path = os.path.join(OUTPUT_DIR, "SUMMARY_metric_impact_score.tsv")
score_df.to_csv(score_path, sep="\t", index=False)

print("\n=== Composite Neurocardiac Impact Score (metric-level) ===")
print(f"Wrote: {score_path}")
print("Top metrics by impact_score:")
print(score_df[["metric", "phase", "impact_score", "burden_fraction", "hit_rate", "median_pr2_max", "median_peak_abs_z"]].head(10))

# ============================================================
# 3) HEATMAPS
# ============================================================

def plot_heatmap(mat: pd.DataFrame, title: str, out_png: str, xlabel: str, ylabel: str,
                 annotate: bool = False, fmt: str = ".0f"):
    """
    Matplotlib heatmap with readable labels.
    """
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * mat.shape[1]), max(6, 0.5 * mat.shape[0])))
    im = ax.imshow(mat.values, aspect="auto", interpolation="nearest")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns.tolist(), rotation=45, ha="right")

    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index.tolist())

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=10)

    if annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, format(mat.values[i, j], fmt),
                        ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

# -----------------------------
# Heatmap 1: Effect × Metric (# ICA significant)
# -----------------------------
effect_metric = (
    df.groupby(["effect", "metric"], as_index=False)
      .agg(n_ica_sig=("ica_significant_bh_acrossICAs", "sum"),
           total_fdr_vox=("n_fdr_vox", "sum"),
           median_pr2=("pr2_max", "median"))
)

mat_sig = effect_metric.pivot(index="effect", columns="metric", values="n_ica_sig").fillna(0)

sig_png = os.path.join(OUTPUT_DIR, "HEATMAP_effect_by_metric_nICASig.png")
plot_heatmap(mat_sig, "Effect × Metric: # ICAs significant (ICA-level BH)", sig_png,
             xlabel="Cardiac metric", ylabel="Effect", annotate=False)

# -----------------------------
# Heatmap 2: Effect × Metric (total FDR voxels)
# -----------------------------
mat_burden = effect_metric.pivot(index="effect", columns="metric", values="total_fdr_vox").fillna(0)

burden_png = os.path.join(OUTPUT_DIR, "HEATMAP_effect_by_metric_totalFDRvox.png")
plot_heatmap(mat_burden, "Effect × Metric: Total FDR voxels (summed across ICAs)", burden_png,
             xlabel="Cardiac metric", ylabel="Effect", annotate=False)

# -----------------------------
# Heatmap 3 (optional, very useful): Metric × ICA (# effects significant)
# -----------------------------
metric_ica = (
    df.groupby(["metric", "ica"], as_index=False)
      .agg(n_effects_sig=("ica_significant_bh_acrossICAs", "sum"),
           total_fdr_vox=("n_fdr_vox", "sum"))
)

mat_metric_ica = metric_ica.pivot(index="metric", columns="ica", values="n_effects_sig").fillna(0)
mica_png = os.path.join(OUTPUT_DIR, "HEATMAP_metric_by_ICA_nEffectsSig.png")
plot_heatmap(mat_metric_ica, "Metric × ICA: # effects significant", mica_png,
             xlabel="ICA", ylabel="Metric", annotate=False)

print("\n=== Heatmaps written ===")
print(sig_png)
print(burden_png)
print(mica_png)

print("\n=== Outputs ===")
print(metric_burden_path)
print(score_path)
print("Done.")
