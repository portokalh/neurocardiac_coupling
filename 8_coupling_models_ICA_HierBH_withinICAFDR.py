#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Second-level voxelwise GLM for PURE neurocardiac coupling (hierarchical BH)

Inference hierarchy (per effect):
A) ICA-level selection across ICAs (BH-FDR across ICAs), where each ICA gets an ICA-level p-value
   computed by Simes' method over voxelwise *uncorrected* p-values within that ICA.

B) Within-ICA spatial inference (primary maps you look at): voxelwise BH-FDR within each ICA.
   (Optionally enforce the hierarchy by zeroing maps for ICAs that fail ICA-level BH.)

Model (pure coupling):
    ICA_voxel ~ metric_z + Age_cont + Sex_ec + Diet_ec + HN_ec + Exercise_ec + APOE_E2 + APOE_E4 + Intercept

Notes:
- Primary spatial inference remains voxelwise BH-FDR within ICA (+ optional cluster extent).
- ICA-level p is Simes(p_unc over voxels). This is BH-flavored (less conservative than Bonf; more defensible than minP).
- Then BH across ICAs per effect controls FDR over networks.
- VIF printed + saved per ICA.

Default run (no args): Heart_Rate, ICA19, smoothing=0.3


How to run

Single ICA:

python 8_coupling_models_ICA_HierBH.py --metric Heart_Rate --ica 19 --smoothing_fwhm 0.3


All ICAs:

python 8_coupling_models_ICA_HierBH.py --metric Heart_Rate --ica all --n_ica_total 20 --smoothing_fwhm 0.3


If you want the extra “hierarchy-enforced” map:

python 8_coupling_models_ICA_HierBH.py --metric Heart_Rate --ica all --enforce_hierarchy
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
# Paths
# =====================================================
BASE_DIR = "/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/myICA_separate100_clean_0p1"
LEVEL1_DIR = os.path.join(BASE_DIR, "level1")

MASK_IMG_PATH = "/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/atlas/reference_maps/chass_atlas_mask_0p1.nii.gz"
METADATA_FILE = "/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/metadata/cardiac_design_updated3.csv"

TODAY = datetime.today().strftime("%Y%m%d")


ICA_MASK_DIR_DEFAULT = (
    "/mnt/newStor/paros/paros_WORK/aashika/"
    "resample_mouse_fmri/myICA_separate100/ICA_masks"
)


# =====================================================
# Arguments
# =====================================================
parser = argparse.ArgumentParser()

parser.add_argument("--metric", type=str, default="Heart_Rate",
                    help="Cardiac metric column name (default: Heart_Rate)")
parser.add_argument("--ica", type=str, default="19",
                    help="ICA index to run (e.g., 19) OR 'all' to run 1..N_ICA (default: 19)")
parser.add_argument("--n_ica_total", type=int, default=20,
                    help="Total number of ICAs available (used when --ica all). Default: 20")
parser.add_argument("--alpha_fdr", type=float, default=0.05,
                    help="BH FDR q (default: 0.05)")
parser.add_argument("--cluster_extent", type=int, default=10,
                    help="Cluster extent (voxels) after voxelwise FDR; 0 disables (default: 10)")
parser.add_argument("--smoothing_fwhm", type=float, default=0.3,
                    help="SecondLevelModel smoothing_fwhm in mm; 0 disables (default: 0.3)")
parser.add_argument("--enforce_hierarchy", action="store_true",
                    help="If set: write an additional map zmap_FDR_ICA_hier.nii.gz that is zeroed unless ICA passes ICA-level BH.")
#parser.add_argument("--ica_mask_dir", type=str, default=None,
#                    help="Directory containing ICA masks named like ica01_mask.nii.gz ... If provided, voxelwise BH is restricted to ICA mask.")
parser.add_argument(
    "--ica_mask_dir",
    type=str,
    default=ICA_MASK_DIR_DEFAULT,
    help="Directory containing ICA masks (default: project ICA masks)"
)


args = parser.parse_args()

METRIC = args.metric
ALPHA_FDR = float(args.alpha_fdr)
CLUSTER_THRESHOLD = int(args.cluster_extent)
USE_CLUSTER_EXTENT = CLUSTER_THRESHOLD > 0
ENFORCE_HIER = bool(args.enforce_hierarchy)

SMOOTHING_FWHM = None if (args.smoothing_fwhm is None or float(args.smoothing_fwhm) <= 0) else float(args.smoothing_fwhm)

ICA_ARG = args.ica.strip().lower()
N_ICA_TOTAL = int(args.n_ica_total)

# -------------------------------------------------
# Validate ICA mask directory
# -------------------------------------------------
if args.ica_mask_dir is not None:
    if not os.path.isdir(args.ica_mask_dir):
        raise RuntimeError(f"ICA mask directory does not exist: {args.ica_mask_dir}")
    else:
        print(f"ICA mask directory verified: {args.ica_mask_dir}")
else:
    print("WARNING: No ICA mask directory provided — full brain BH will be used.")


print("==============================================")
print(f"Metric: {METRIC}")
print(f"ICA arg: {args.ica}")
print(f"BH q (voxelwise within ICA): {ALPHA_FDR}")
print(f"BH q (across ICAs):          {ALPHA_FDR}")
print(f"Cluster extent: {CLUSTER_THRESHOLD if USE_CLUSTER_EXTENT else 0}")
print(f"Smoothing_fwhm: {SMOOTHING_FWHM}")
print(f"Enforce hierarchy (extra map): {ENFORCE_HIER}")
print(f"Node: {socket.gethostname()}")
print(f"PID: {os.getpid()}")
print("==============================================")


smooth_tag = "nosmooth" if SMOOTHING_FWHM is None else f"smooth{str(SMOOTHING_FWHM).replace('.', 'p')}"
ica_tag = "ICAall" if ICA_ARG == "all" else f"ICA{int(ICA_ARG):02d}"

ANALYSIS_LABEL = (
    f"{TODAY}_PURECOUPLING_"
    f"{ica_tag}_"
    f"AGE-cont_"
    f"{smooth_tag}_"
    f"BHq{str(ALPHA_FDR).replace('.', 'p')}_"
    f"HierBH_SimesAcrossICA_"
    f"FDRwithinICA_"
    f"cluster{CLUSTER_THRESHOLD if USE_CLUSTER_EXTENT else 0}"
)

LEVEL2_ROOT = os.path.join(BASE_DIR, "level2", ANALYSIS_LABEL)
os.makedirs(LEVEL2_ROOT, exist_ok=True)

print(f"\nResults will be saved in:\n{LEVEL2_ROOT}\n")

print(f"ICA mask dir: {args.ica_mask_dir}")

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

def simes_pvalue(pvals: np.ndarray) -> float:
    """
    Simes p-value for a family of tests.
    p_family = min_i (m/i * p_(i)), with p_(i) sorted ascending.
    """
    p = pvals.ravel()
    p = p[np.isfinite(p)]
    if p.size == 0:
        return 1.0
    p_sorted = np.sort(p)
    m = p_sorted.size
    idx = np.arange(1, m + 1, dtype=float)
    simes = np.min((m / idx) * p_sorted)
    return float(min(1.0, max(0.0, simes)))

def load_ica_mask(ica_mask_dir, ica_idx, ref_img, save_copy_dir=None):
    """
    Load and verify ICA mask.
    Optionally saves a copy of the mask used into the results directory.
    """
    if ica_mask_dir is None:
        return None

    mask_path = os.path.join(ica_mask_dir, f"ica{ica_idx:02d}_mask.nii.gz")

    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"ICA mask not found: {mask_path}")

    mask_img = image.load_img(mask_path)
    mask_data = mask_img.get_fdata() > 0

    ref_shape = ref_img.shape[:3]
    mask_shape = mask_data.shape[:3]

    if mask_shape != ref_shape:
        raise ValueError(
            f"Mask shape {mask_shape} does not match ref shape {ref_shape}"
        )

    n_mask_vox = int(np.sum(mask_data))
    total_vox = int(np.prod(ref_shape))

    print(f"  ICA {ica_idx:02d} mask loaded")
    print(f"  Mask voxels: {n_mask_vox}")
    print(f"  Total voxels: {total_vox}")
    print(f"  Mask coverage: {n_mask_vox / total_vox:.4f}")

    if n_mask_vox == 0:
        raise RuntimeError(f"ICA {ica_idx:02d} mask is EMPTY.")

    # Optionally save mask copy into results directory
    if save_copy_dir is not None:
        os.makedirs(save_copy_dir, exist_ok=True)
        mask_img.to_filename(os.path.join(save_copy_dir, "ICA_mask_used.nii.gz"))

    return mask_data




# =====================================================
# Load data
# =====================================================
mask_img = image.load_img(MASK_IMG_PATH)
metadata = pd.read_csv(METADATA_FILE)

if METRIC not in metadata.columns:
    raise ValueError(f"Metric '{METRIC}' not found in metadata columns.")

# robust coercion
for c in ["E2_Genotype", "E3_Genotype", "E4_Genotype", "KO_Genotype", "HN"]:
    if c in metadata.columns:
        metadata[c] = pd.to_numeric(metadata[c], errors="coerce")

# Age continuous centered
if "Age" not in metadata.columns:
    raise ValueError("Expected continuous age column 'Age' not found.")
metadata["Age_cont"] = pd.to_numeric(metadata["Age"], errors="coerce")
metadata["Age_cont"] = metadata["Age_cont"] - metadata["Age_cont"].mean()
AGE_TERM = "Age_cont"

required_cols = [
    "Arunno", METRIC, AGE_TERM,
    "Sex_Male", "Diet_HFD", "Exercise_Yes", "HN",
    "E2_Genotype", "E3_Genotype", "E4_Genotype", "KO_Genotype",
]
missing = [c for c in required_cols if c not in metadata.columns]
if missing:
    raise ValueError(f"Missing required columns in metadata: {missing}")

for c in [METRIC, AGE_TERM, "Sex_Male", "Diet_HFD", "Exercise_Yes", "HN",
          "E2_Genotype", "E3_Genotype", "E4_Genotype", "KO_Genotype"]:
    metadata[c] = pd.to_numeric(metadata[c], errors="coerce")

# Exclude KO
metadata = metadata[metadata["KO_Genotype"] == 0].copy()

# genotype sanity: E2+E3+E4 == 1
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

icas_to_run = list(range(1, N_ICA_TOTAL + 1)) if ICA_ARG == "all" else [int(ICA_ARG)]


# =====================================================
# Main loop over ICAs
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

    matched_files, matched_rows = [], []
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

    # Raw design
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

    # VIF
    print("\n--- VIF diagnostics ---")
    X_vif = X_final.drop(columns=["Intercept"]).copy()
    vif_df = pd.DataFrame({
        "variable": X_vif.columns,
        "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    })
    print(vif_df)
    print("------------------------\n")

    vif_path = os.path.join(LEVEL2_ROOT, METRIC, f"ica{j:02d}", "VIF.tsv")
    os.makedirs(os.path.dirname(vif_path), exist_ok=True)
    vif_df.to_csv(vif_path, sep="\t", index=False)

    n_subj = int(X_final.shape[0])
    p_params = int(X_final.shape[1])
    df_resid = n_subj - p_params
    if df_resid <= 0:
        raise ValueError(f"Non-positive df_resid={df_resid} (n={n_subj}, p={p_params})")

    print(f"Subjects used: n={n_subj}, p={p_params}, df_resid={df_resid}")

    # Fit
    model = SecondLevelModel(
        n_jobs=4,
        smoothing_fwhm=SMOOTHING_FWHM,
        mask_img=mask_img
    ).fit(imgs, design_matrix=X_final)
    
    
    # -------------------------------------------------
    # Load ICA mask once per ICA
    # -------------------------------------------------
    mask_copy_dir = os.path.join(LEVEL2_ROOT, METRIC, f"ica{j:02d}")
    
    ica_mask = load_ica_mask(
        args.ica_mask_dir,
        j,
        mask_img,  # reference space
        save_copy_dir=mask_copy_dir
    )


    cols = list(X_final.columns)

    # Per-effect outputs
    for effect in effects_to_test:
        if effect not in cols:
            continue

        contrast = np.zeros(len(cols), dtype=float)
        contrast[cols.index(effect)] = 1.0

        z_map = model.compute_contrast(contrast, output_type="z_score")
        z_data = z_map.get_fdata()

        t_map = model.compute_contrast(contrast, output_type="stat")
        t_data = t_map.get_fdata()

        pr2 = (t_data ** 2) / ((t_data ** 2) + df_resid)
        pr2_img = new_img_like(t_map, pr2)

        peak_z = float(np.nanmax(np.abs(z_data)))
        p_unc = two_sided_p_from_z(z_data)
        
       


        # diagnostics
        #n_unc_005 = int(np.sum(p_unc < 0.05))
        if ica_mask is None:
            n_unc_005 = int(np.sum(p_unc < 0.05))
            n_vox_tested = int(p_unc.size)
        else:
                n_unc_005 = int(np.sum(p_unc[ica_mask] < 0.05))
                n_vox_tested = int(p_unc[ica_mask].size)

        print(f"  Voxels tested: {n_vox_tested}")
        print(f"  Vox p<0.05 unc (within tested space): {n_unc_005}")


        # voxelwise BH within ICA
        #rej = bh_fdr_reject_mask_from_p(p_unc, alpha=ALPHA_FDR)
        #n_fdr = int(rej.sum())
        
        # -------------------------------------------------
        # Voxelwise BH within ICA (restricted to ICA mask if provided)
        # -------------------------------------------------
        #ica_mask = load_ica_mask(args.ica_mask_dir, j, z_map)
        
        mask_copy_dir = os.path.join(LEVEL2_ROOT, METRIC, f"ica{j:02d}")
        ica_mask = load_ica_mask(
            args.ica_mask_dir,
            j,
            z_map,
            save_copy_dir=mask_copy_dir
            )

        
        if ica_mask is None:
            # fallback: full-brain BH (current behavior)
            rej = bh_fdr_reject_mask_from_p(p_unc, alpha=ALPHA_FDR)
        else:
            # restrict BH to ICA spatial support
            p_in = p_unc[ica_mask]
            reject_in, _, _, _ = multipletests(p_in, alpha=ALPHA_FDR, method="fdr_bh")
        
            rej = np.zeros_like(p_unc, dtype=bool)
            rej[ica_mask] = reject_in
        
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

        # ICA-level p for hierarchical BH across ICAs (Simes over voxel p_unc)
        #p_ica_simes = simes_pvalue(p_unc)
        if ica_mask is None:
            p_ica_simes = simes_pvalue(p_unc)
        else:
            p_ica_simes = simes_pvalue(p_unc[ica_mask])

        if ica_mask is None:
            n_vox_tested = int(np.isfinite(p_unc).sum())
        else:
            n_vox_tested = int(ica_mask.sum())




        ica_summary_rows.append({
            "metric": METRIC,
            "ica": int(j),
            "effect": effect,
            "n_subj": n_subj,
            "df_resid": df_resid,
            "n_voxels": int(z_data.size),
            "n_voxels_tested": n_vox_tested,
            "peak_abs_z": peak_z,
            "ica_p_simes": float(p_ica_simes),
            "n_unc_p005": n_unc_005,
            "n_fdr_vox": n_fdr,
            "pr2_max": float(np.nanmax(pr2)),
            "pr2_mean": float(np.nanmean(pr2)),
        })

        print(f"\nEffect: {effect}")
        print(f"  Peak |Z|: {peak_z:.3f}")
        print(f"  Vox p<0.05 unc: {n_unc_005}")
        print(f"  Vox BH-FDR within ICA (q={ALPHA_FDR}): {n_fdr}")
        print(f"  ICA-level p (Simes over voxels): {p_ica_simes:.3e}")
        print(f"  partialR2: max={float(np.nanmax(pr2)):.4f}, mean={float(np.nanmean(pr2)):.6f}")


# =====================================================
# Hierarchical BH across ICAs per effect (using ICA-level Simes p-values)
# =====================================================
if len(ica_summary_rows) == 0:
    raise RuntimeError("No ICA/effect results were produced. Check matching Arunno↔filename and missingness.")

summary_df = pd.DataFrame(ica_summary_rows)

metric_dir = os.path.join(LEVEL2_ROOT, METRIC)
os.makedirs(metric_dir, exist_ok=True)

summary_path = os.path.join(metric_dir, "ICA_level_summary.tsv")
summary_df.to_csv(summary_path, sep="\t", index=False)
print(f"\nSaved ICA-level summary: {summary_path}")

net_df_list = []
for eff in sorted(summary_df["effect"].unique()):
    sub = summary_df[summary_df["effect"] == eff].copy()
    pvals = sub["ica_p_simes"].values

    reject, p_corr, _, _ = multipletests(pvals, alpha=ALPHA_FDR, method="fdr_bh")
    sub["p_ica_bh_acrossICAs"] = p_corr
    sub["ica_significant_bh_acrossICAs"] = reject.astype(int)

    net_df_list.append(sub)

net_df = pd.concat(net_df_list, ignore_index=True)

net_path = os.path.join(metric_dir, "ICA_level_summary_with_BH.tsv")
net_df.to_csv(net_path, sep="\t", index=False)
print(f"Saved ICA-level BH summary: {net_path}")

for eff in ["metric_z", "Exercise_ec"]:
    if eff in net_df["effect"].unique():
        hits = net_df[(net_df["effect"] == eff) & (net_df["ica_significant_bh_acrossICAs"] == 1)]
        if len(hits) == 0:
            print(f"\nNo ICAs survive BH across ICAs for {eff} (Simes -> BH across ICAs).")
        else:
            ica_list = hits["ica"].tolist()
            print(f"\nICAs surviving BH across ICAs for {eff}: {ica_list}")


# =====================================================
# Optional: enforce hierarchy by writing additional thresholded maps
# (only if requested; uses existing zmap_FDR_ICA and zeros it if ICA fails BH across ICAs)
# =====================================================
if ENFORCE_HIER:
    print("\nEnforcing hierarchy: writing zmap_FDR_ICA_hier.nii.gz (zeroed unless ICA survives ICA-level BH across ICAs).")

    # Build lookup: (ica, effect) -> pass/fail
    pass_lookup = {
        (int(r["ica"]), str(r["effect"])): int(r["ica_significant_bh_acrossICAs"])
        for _, r in net_df.iterrows()
    }

    for _, r in net_df.iterrows():
        ica = int(r["ica"])
        eff = str(r["effect"])
        ok = pass_lookup.get((ica, eff), 0)

        out_dir = os.path.join(LEVEL2_ROOT, METRIC, f"ica{ica:02d}", eff)
        f_in = os.path.join(out_dir, "zmap_FDR_ICA.nii.gz")
        f_out = os.path.join(out_dir, "zmap_FDR_ICA_hier.nii.gz")

        if not os.path.exists(f_in):
            continue

        img_in = image.load_img(f_in)
        dat = img_in.get_fdata()

        if ok == 0:
            dat = np.zeros_like(dat)

        image.new_img_like(img_in, dat).to_filename(f_out)


# =====================================================
# Effect-size tables
# =====================================================
df = pd.read_csv(net_path, sep="\t")

table_ica = df.sort_values(["ica", "effect"])[
    ["metric", "ica", "effect", "n_subj", "peak_abs_z", "n_fdr_vox", "pr2_max", "pr2_mean",
     "ica_p_simes", "p_ica_bh_acrossICAs", "ica_significant_bh_acrossICAs"]
]

table_effect = (
    df.groupby(["metric", "effect"], as_index=False)
      .agg(
          n_icas=("ica", "nunique"),
          median_pr2_max=("pr2_max", "median"),
          median_pr2_mean=("pr2_mean", "median"),
          max_pr2_max=("pr2_max", "max"),
          mean_pr2_mean=("pr2_mean", "mean"),
          total_fdr_vox=("n_fdr_vox", "sum"),
          n_ica_bh_sig=("ica_significant_bh_acrossICAs", "sum")
      )
      .sort_values("median_pr2_max", ascending=False)
)

out_dir = os.path.dirname(net_path)
table_ica.to_csv(os.path.join(out_dir, "Table_effectsizes_by_ICA.tsv"), sep="\t", index=False)
table_effect.to_csv(os.path.join(out_dir, "Table_effectsizes_collapsed.tsv"), sep="\t", index=False)

print("\nWrote:")
print(os.path.join(out_dir, "Table_effectsizes_by_ICA.tsv"))
print(os.path.join(out_dir, "Table_effectsizes_collapsed.tsv"))

#"n_voxels_tested": n_vox_tested,

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

print("\n==============================================")
print(f"Finished metric: {METRIC}")
print(f"Execution time: {elapsed:.2f} sec ({elapsed/60:.2f} min)")
print(f"Resident memory usage: {mem_mb:.2f} MB")
print("\nRESULTS DIRECTORY STRUCTURE:")
print(f"Root analysis directory:\n{LEVEL2_ROOT}")
print(f"\nMetric-specific directory:\n{os.path.join(LEVEL2_ROOT, METRIC)}")
print("\nKey files:")
print(f"- ICA summary (raw): {summary_path}")
print(f"- ICA summary (with BH across ICAs): {net_path}")
#print(f"- Table_effectsizes_by_ICA.tsv")
#print(f"- Table_effectsizes_collapsed.tsv")
print("==============================================\n")
