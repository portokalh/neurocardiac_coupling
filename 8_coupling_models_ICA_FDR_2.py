#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Second-level voxelwise GLM for PURE neurocardiac coupling (publication-ready inference)

Implements (in this order):
1) Per-ICA voxelwise BH-FDR (q=ALPHA_FDR) across voxels (NOT pooled across ICA×voxels)
2) Partial R² per voxel for each effect (saved as NIfTI)
3) Network-level FDR across ICAs (BH across ICA-level p-values per effect) + TSV summary
4) VIF diagnostics printed (and saved) per ICA to flag collinearity

Model (pure coupling):
    ICA_voxel ~ metric_z + Age_cont + Sex_ec + Diet_ec + HN_ec + Exercise_ec + APOE_E2 + APOE_E4 + Intercept

Notes:
- Age is continuous, centered
- Binary covariates are effect-coded (-0.5/+0.5)
- Metric regressor is z-scored within ICA subset
- APOE contrasts built from genotype dummies
- Per-ICA voxelwise BH-FDR is the PRIMARY spatial inference
- ICA-level p-values are conservative screening stats: max-|Z| + Bonferroni over voxels within ICA.
  For gold-standard ICA-level inference, use permutation / maxT.

Default run (no args): Heart_Rate, ICA19

@author: alex
"""

import os
import argparse
import time
import socket
import psutil
from datetime import datetime

import numpy as np
import pandas as pd

from nilearn.glm.second_level import SecondLevelModel
from nilearn import image
from nilearn.image import new_img_like, threshold_img
from scipy.stats import norm, pearsonr
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor


# =====================================================
# Runtime Monitoring
# =====================================================
start_time = time.time()
process = psutil.Process(os.getpid())


# =====================================================
# Arguments (safe defaults)
# =====================================================
parser = argparse.ArgumentParser()

parser.add_argument(
    "--metric",
    type=str,
    default="Heart_Rate",
    help="Cardiac metric column name (default: Heart_Rate)"
)

parser.add_argument(
    "--ica",
    type=str,
    default="19",
    help="ICA index to run (e.g., 19) OR 'all' to run 1..N_ICA (default: 19)"
)

parser.add_argument(
    "--n_ica_total",
    type=int,
    default=20,
    help="Total number of ICAs available (used when --ica all). Default: 20"
)

parser.add_argument(
    "--alpha_fdr",
    type=float,
    default=0.05,
    help="BH FDR q (default: 0.05)"
)

parser.add_argument(
    "--cluster_extent",
    type=int,
    default=10,
    help="Cluster extent (voxels) after FDR; 0 disables (default: 10)"
)

parser.add_argument(
    "--smoothing_fwhm",
    type=float,
    default=0.3,
    help="SecondLevelModel smoothing_fwhm in mm; 0 disables (default: 0.3)"
)

args = parser.parse_args()

METRIC = args.metric
ALPHA_FDR = float(args.alpha_fdr)
CLUSTER_THRESHOLD = int(args.cluster_extent)
USE_CLUSTER_EXTENT = CLUSTER_THRESHOLD > 0

SMOOTHING_FWHM = None if (args.smoothing_fwhm is None or float(args.smoothing_fwhm) <= 0) else float(args.smoothing_fwhm)

ICA_ARG = args.ica.strip().lower()
N_ICA_TOTAL = int(args.n_ica_total)

print("==============================================")
print(f"Metric: {METRIC}")
print(f"ICA arg: {args.ica}")
print(f"BH q (voxelwise): {ALPHA_FDR}")
print(f"Cluster extent: {CLUSTER_THRESHOLD if USE_CLUSTER_EXTENT else 0}")
print(f"Smoothing_fwhm: {SMOOTHING_FWHM}")
print(f"Node: {socket.gethostname()}")
print(f"PID: {os.getpid()}")
print("==============================================")


# =====================================================
# Paths
# =====================================================
BASE_DIR = "/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/myICA_separate100_clean_0p1"
LEVEL1_DIR = os.path.join(BASE_DIR, "level1")

MASK_IMG_PATH = "/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/atlas/reference_maps/chass_atlas_mask_0p1.nii.gz"
METADATA_FILE = "/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/metadata/cardiac_design_updated3.csv"

TODAY = datetime.today().strftime("%Y%m%d")
smooth_tag = "nosmooth" if SMOOTHING_FWHM is None else f"smooth{str(SMOOTHING_FWHM).replace('.', 'p')}"
ica_tag = "ICAall" if ICA_ARG == "all" else f"ICA{int(ICA_ARG):02d}"

ANALYSIS_LABEL = (
    f"{TODAY}_PURECOUPLING_"
    f"{ica_tag}_"
    f"AGE-cont_"
    f"{smooth_tag}_"
    f"BHq{str(ALPHA_FDR).replace('.', 'p')}_"
    f"FDRwithinICA_"
    f"cluster{CLUSTER_THRESHOLD if USE_CLUSTER_EXTENT else 0}"
)

LEVEL2_ROOT = os.path.join(BASE_DIR, "level2", ANALYSIS_LABEL)
os.makedirs(LEVEL2_ROOT, exist_ok=True)

print(f"\nResults will be saved in:\n{LEVEL2_ROOT}\n")


# =====================================================
# Helpers
# =====================================================
def effect_code_01(x: pd.Series) -> pd.Series:
    """Convert 0/1 -> -0.5/+0.5 effect coding."""
    s = pd.to_numeric(x, errors="coerce")
    if s.isna().any():
        raise ValueError(f"Binary column has NaNs after coercion: {x.name}")
    vals = set(s.unique().tolist())
    if not vals.issubset({0, 1}):
        raise ValueError(f"Binary column {x.name} must be 0/1; got: {sorted(vals)}")
    return s * 0.5 + (1 - s) * (-0.5)

def zscore(x: pd.Series) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce")
    sd = s.std(ddof=0)
    if np.isnan(sd) or sd == 0:
        raise ValueError(f"Cannot z-score {x.name}: std={sd}")
    return (s - s.mean()) / sd

def two_sided_p_from_z(z: np.ndarray) -> np.ndarray:
    return 2.0 * (1.0 - norm.cdf(np.abs(z)))

def bh_fdr_reject_mask_from_p(pvals: np.ndarray, alpha: float) -> np.ndarray:
    pflat = pvals.ravel()
    reject, _, _, _ = multipletests(pflat, alpha=alpha, method="fdr_bh")
    return reject.reshape(pvals.shape)

def ica_level_p_maxabs_bonf(z: np.ndarray) -> float:
    """Conservative ICA-level p-value from max-|Z| with Bonferroni over voxels within ICA."""
    m = int(np.prod(z.shape))
    peak = float(np.nanmax(np.abs(z)))
    p_peak = float(2.0 * (1.0 - norm.cdf(peak)))
    return min(1.0, p_peak * m)


# =====================================================
# Load data
# =====================================================
mask_img = image.load_img(MASK_IMG_PATH)
metadata = pd.read_csv(METADATA_FILE)

if METRIC not in metadata.columns:
    raise ValueError(f"Metric '{METRIC}' not found in metadata columns.")

# genotype dummies should exist; keep robust coercion
needed_geno_cols = {"E2_Genotype", "E3_Genotype", "E4_Genotype", "KO_Genotype"}
for c in needed_geno_cols:
    if c in metadata.columns:
        metadata[c] = pd.to_numeric(metadata[c], errors="coerce")

# HN
if "HN" in metadata.columns:
    metadata["HN"] = pd.to_numeric(metadata["HN"], errors="coerce")

# Age continuous centered
if "Age" not in metadata.columns:
    raise ValueError("Expected continuous age column 'Age' not found.")
metadata["Age_cont"] = pd.to_numeric(metadata["Age"], errors="coerce")
metadata["Age_cont"] = metadata["Age_cont"] - metadata["Age_cont"].mean()
AGE_TERM = "Age_cont"

required_cols = [
    "Arunno",
    METRIC,
    AGE_TERM,
    "Sex_Male",
    "Diet_HFD",
    "Exercise_Yes",
    "HN",
    "E2_Genotype",
    "E3_Genotype",
    "E4_Genotype",
    "KO_Genotype",
]
missing = [c for c in required_cols if c not in metadata.columns]
if missing:
    raise ValueError(f"Missing required columns in metadata: {missing}")

# Coerce numeric for all required modeling columns
for c in [METRIC, AGE_TERM, "Sex_Male", "Diet_HFD", "Exercise_Yes", "HN",
          "E2_Genotype", "E3_Genotype", "E4_Genotype", "KO_Genotype"]:
    metadata[c] = pd.to_numeric(metadata[c], errors="coerce")

# Exclude KO
metadata = metadata[metadata["KO_Genotype"] == 0].copy()

# Sanity: E2+E3+E4 == 1
geno_sum = metadata[["E2_Genotype", "E3_Genotype", "E4_Genotype"]].sum(axis=1)
metadata = metadata.loc[geno_sum == 1].copy()

# APOE contrasts
metadata["APOE_E2"] = metadata["E2_Genotype"] * 1.0 + metadata["E3_Genotype"] * 0.0 + metadata["E4_Genotype"] * (-1.0)
metadata["APOE_E4"] = metadata["E2_Genotype"] * 0.0 + metadata["E3_Genotype"] * (-1.0) + metadata["E4_Genotype"] * 1.0

effects_to_test = [
    "metric_z",
    AGE_TERM,
    "Sex_ec",
    "Diet_ec",
    "HN_ec",
    "APOE_E2",
    "APOE_E4",
    "Exercise_ec",
]

# Which ICAs to run
if ICA_ARG == "all":
    icas_to_run = list(range(1, N_ICA_TOTAL + 1))
else:
    icas_to_run = [int(ICA_ARG)]


# =====================================================
# Main loop over ICAs (per-ICA voxelwise FDR + partial R²)
# =====================================================
ica_summary_rows = []

for j in icas_to_run:
    print(f"\n====================")
    print(f"Processing ICA {j:02d}")
    print(f"====================")

    level1_dir = os.path.join(LEVEL1_DIR, f"level1_ica{j:02d}")
    if not os.path.isdir(level1_dir):
        print(f"  Missing dir, skipping: {level1_dir}")
        continue

    level1_files = sorted([
        f for f in os.listdir(level1_dir)
        if (f.endswith(".nii.gz") or f.endswith(".nii"))
        and not f.startswith("._")
        and os.path.getsize(os.path.join(level1_dir, f)) > 1000
    ])
    if not level1_files:
        print("  No NIfTI files found, skipping.")
        continue

    matched_files = []
    matched_rows = []

    for _, row in metadata.iterrows():
        sid = str(row["Arunno"])
        hits = [f for f in level1_files if f.endswith(f"_{sid}.nii.gz") or f.endswith(f"_{sid}.nii")]
        if hits:
            matched_files.append(os.path.join(level1_dir, hits[0]))
            matched_rows.append(row)

    if len(matched_rows) < 5:
        print("  Too few matched subjects, skipping.")
        continue

    md = pd.DataFrame(matched_rows).copy()

    # Diagnostic: correlation metric vs Exercise
    try:
        mvals = pd.to_numeric(md[METRIC], errors="coerce")
        evals = pd.to_numeric(md["Exercise_Yes"], errors="coerce")
        valid = ~(mvals.isna() | evals.isna())
        if valid.sum() > 5:
            r, p_corr = pearsonr(mvals.loc[valid], evals.loc[valid])
            print(f"Correlation {METRIC} vs Exercise: r={r:.4f}, p={p_corr:.3e}")
    except Exception as e:
        print(f"Correlation diagnostic skipped: {e}")

    # Build raw design
    X = md[[METRIC, AGE_TERM, "Sex_Male", "Diet_HFD", "HN", "Exercise_Yes", "APOE_E2", "APOE_E4"]].apply(
        pd.to_numeric, errors="coerce"
    )

    keep = ~X.isna().any(axis=1)
    if keep.sum() < 5:
        print("  Too few complete cases after NA drop, skipping.")
        continue

    X = X.loc[keep].reset_index(drop=True)
    keep_idx = np.where(keep.values)[0]
    imgs = [image.load_img(matched_files[i]) for i in keep_idx]

    # coding
    X["metric_z"] = zscore(X[METRIC])
    X[AGE_TERM] = X[AGE_TERM] - X[AGE_TERM].mean()

    X["Sex_ec"] = effect_code_01(X["Sex_Male"])
    X["Diet_ec"] = effect_code_01(X["Diet_HFD"])
    X["HN_ec"] = effect_code_01((X["HN"] > 0).astype(int))
    X["Exercise_ec"] = effect_code_01(X["Exercise_Yes"])

    X["APOE_E2"] = X["APOE_E2"] - X["APOE_E2"].mean()
    X["APOE_E4"] = X["APOE_E4"] - X["APOE_E4"].mean()

    X["Intercept"] = 1.0

    design_cols = ["Intercept", "metric_z", AGE_TERM, "Sex_ec", "Diet_ec", "HN_ec", "Exercise_ec", "APOE_E2", "APOE_E4"]
    X_final = X[design_cols].copy()

    # =====================================================
    # VIF diagnostics (multicollinearity check)
    # =====================================================
    print("\n--- VIF diagnostics ---")
    X_vif = X_final.drop(columns=["Intercept"]).copy()
    vif_df = pd.DataFrame({
        "variable": X_vif.columns,
        "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    })
    print(vif_df)
    print("------------------------\n")

    # Save VIF
    vif_path = os.path.join(LEVEL2_ROOT, METRIC, f"ica{j:02d}", "VIF.tsv")
    os.makedirs(os.path.dirname(vif_path), exist_ok=True)
    vif_df.to_csv(vif_path, sep="\t", index=False)

    n_subj = int(X_final.shape[0])
    p_params = int(X_final.shape[1])
    df_resid = n_subj - p_params
    if df_resid <= 0:
        raise ValueError(f"Non-positive df_resid={df_resid} (n={n_subj}, p={p_params})")

    print(f"Subjects used: n={n_subj}, p={p_params}, df_resid={df_resid}")

    # Fit model
    model = SecondLevelModel(
        smoothing_fwhm=SMOOTHING_FWHM,
        mask_img=mask_img
    ).fit(imgs, design_matrix=X_final)

    cols = list(X_final.columns)

    # Per-effect outputs
    for effect in effects_to_test:
        if effect not in cols:
            continue

        contrast = np.zeros(len(cols), dtype=float)
        contrast[cols.index(effect)] = 1.0

        # Z and T maps
        z_map = model.compute_contrast(contrast, output_type="z_score")
        z_data = z_map.get_fdata()

        t_map = model.compute_contrast(contrast, output_type="stat")
        t_data = t_map.get_fdata()

        # Partial R² per voxel
        pr2 = (t_data ** 2) / ((t_data ** 2) + df_resid)
        pr2_img = new_img_like(t_map, pr2)

        # Uncorrected diagnostics
        peak_z = float(np.nanmax(np.abs(z_data)))
        p_unc = two_sided_p_from_z(z_data)
        n_unc_005 = int(np.sum(p_unc < 0.05))

        # --- Per-ICA voxelwise BH-FDR ---
        rej = bh_fdr_reject_mask_from_p(p_unc, alpha=ALPHA_FDR)
        n_fdr = int(rej.sum())

        z_thr = np.zeros_like(z_data)
        z_thr[rej] = z_data[rej]
        z_thr_img = new_img_like(z_map, z_thr)

        if USE_CLUSTER_EXTENT:
            z_thr_img = threshold_img(
                z_thr_img,
                threshold=0,
                cluster_threshold=CLUSTER_THRESHOLD,
                copy_header=True,
            )

        out_dir = os.path.join(LEVEL2_ROOT, METRIC, f"ica{j:02d}", effect)
        os.makedirs(out_dir, exist_ok=True)

        z_map.to_filename(os.path.join(out_dir, "zmap_uncorrected.nii.gz"))
        z_thr_img.to_filename(os.path.join(out_dir, "zmap_FDR_ICA.nii.gz"))
        t_map.to_filename(os.path.join(out_dir, "tmap_uncorrected.nii.gz"))
        pr2_img.to_filename(os.path.join(out_dir, "partialR2.nii.gz"))

        # ICA-level (conservative) p-value for across-ICA BH
        p_ica = ica_level_p_maxabs_bonf(z_data)

        ica_summary_rows.append({
            "metric": METRIC,
            "ica": int(j),
            "effect": effect,
            "n_subj": n_subj,
            "df_resid": df_resid,
            "n_voxels": int(z_data.size),
            "peak_abs_z": peak_z,
            "n_unc_p005": n_unc_005,
            "n_fdr_vox": n_fdr,
            "p_ica_maxabs_bonf": float(p_ica),
            "pr2_max": float(np.nanmax(pr2)),
            "pr2_mean": float(np.nanmean(pr2)),
        })

        print(f"\nEffect: {effect}")
        print(f"  Peak |Z|: {peak_z:.3f}")
        print(f"  Vox p<0.05 unc: {n_unc_005}")
        print(f"  Vox BH-FDR within ICA (q={ALPHA_FDR}): {n_fdr}")
        print(f"  ICA-level p (max-|Z| Bonf): {p_ica:.3e}")
        print(f"  partialR2: max={float(np.nanmax(pr2)):.4f}, mean={float(np.nanmean(pr2)):.6f}")


# =====================================================
# Network-level FDR across ICAs (BH across ICA-level p-values per effect)
# =====================================================
if len(ica_summary_rows) == 0:
    raise RuntimeError("No ICA/effect results were produced. Check matching Arunno↔filename and missingness.")

summary_df = pd.DataFrame(ica_summary_rows)

metric_dir = os.path.join(LEVEL2_ROOT, METRIC)
os.makedirs(metric_dir, exist_ok=True)

summary_path = os.path.join(metric_dir, "ICA_level_summary.tsv")
summary_df.to_csv(summary_path, sep="\t", index=False)
print(f"\nSaved ICA-level summary: {summary_path}")

# BH across ICAs per effect
net_df_list = []
for eff in sorted(summary_df["effect"].unique()):
    sub = summary_df[summary_df["effect"] == eff].copy()
    pvals = sub["p_ica_maxabs_bonf"].values
    reject, p_corr, _, _ = multipletests(pvals, alpha=ALPHA_FDR, method="fdr_bh")
    sub["p_ica_bh"] = p_corr
    sub["ica_significant_bh"] = reject.astype(int)
    net_df_list.append(sub)

net_df = pd.concat(net_df_list, ignore_index=True)
net_path = os.path.join(metric_dir, "ICA_level_summary_with_BH.tsv")
net_df.to_csv(net_path, sep="\t", index=False)
print(f"Saved ICA-level BH summary: {net_path}")

# Quick printout for primary effects
for eff in ["metric_z", "Exercise_ec"]:
    if eff in net_df["effect"].unique():
        hits = net_df[(net_df["effect"] == eff) & (net_df["ica_significant_bh"] == 1)]
        if len(hits) == 0:
            print(f"\nNo ICAs survive BH across ICAs for {eff} (using conservative max-|Z| Bonf p).")
        else:
            ica_list = hits["ica"].tolist()
            print(f"\nICAs surviving BH across ICAs for {eff}: {ica_list}")


# =====================================================
# Runtime report
# =====================================================
elapsed = time.time() - start_time
mem_mb = process.memory_info().rss / 1024**2

print("\n==============================================")
print(f"Finished metric: {METRIC}")
print(f"Execution time: {elapsed:.2f} sec ({elapsed/60:.2f} min)")
print(f"Resident memory usage: {mem_mb:.2f} MB")
print(f"Results saved under:\n{LEVEL2_ROOT}/{METRIC}/")
print("==============================================")




#LEVEL2_ROOT = "/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/myICA_separate100_clean_0p1/level2"
#ANALYSIS_LABEL = "20260217_PURECOUPLING_ICA19_AGE-cont_smooth0p3_BHq0p05_FDRwithinICA_cluster10"
#METRIC = "Heart_Rate"

summary_path = net_path
df = pd.read_csv(summary_path, sep="\t")

# Per ICA x effect table (already basically done)
table_ica = df.sort_values(["ica", "effect"])[
    ["metric", "ica", "effect", "n_subj", "peak_abs_z", "n_fdr_vox", "pr2_max", "pr2_mean", "p_ica_maxabs_bonf"]
]

# Optional: collapse across ICAs for each effect (median is robust)
table_effect = (
    df.groupby(["metric", "effect"], as_index=False)
      .agg(
          n_icas=("ica", "nunique"),
          median_pr2_max=("pr2_max", "median"),
          median_pr2_mean=("pr2_mean", "median"),
          max_pr2_max=("pr2_max", "max"),
          mean_pr2_mean=("pr2_mean", "mean"),
          total_fdr_vox=("n_fdr_vox", "sum")
      )
      .sort_values("median_pr2_max", ascending=False)
)

out_dir = os.path.dirname(summary_path)
table_ica.to_csv(os.path.join(out_dir, "Table_effectsizes_by_ICA.tsv"), sep="\t", index=False)
table_effect.to_csv(os.path.join(out_dir, "Table_effectsizes_collapsed.tsv"), sep="\t", index=False)

print("Wrote:")
print(os.path.join(out_dir, "Table_effectsizes_by_ICA.tsv"))
print(os.path.join(out_dir, "Table_effectsizes_collapsed.tsv"))
