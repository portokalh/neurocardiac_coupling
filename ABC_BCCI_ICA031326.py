#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Brain–Cardiac Coupling Pipeline
Author: Alex

Major updates in this refactor
-----------------------------
1. Diet terms removed from downstream modeling where requested; Mass / Mass_z used instead.
2. Thresholded circuit maps are now truly thresholded:
   - percentile-thresholded signed weighted maps
   - absolute mask map saved separately
   - threshold value saved to disk
3. PLS now uses ALL cardiac domains jointly, not Heart_Rate alone.
4. Several formula / variable consistency issues fixed.
5. Added compact helper exports to make results easier to inspect.

Main outputs
------------
- network_cardiac_results_allmetrics.tsv
- network_cardiac_results_hierFDR.tsv
- network_weight_statistics_domains.tsv
- domain_model_statistics.tsv
- BCCI_subject_scores.tsv
- BCCI_domain_validation.tsv
- BCCI_vs_all_cardiac_metrics.tsv
- BCCI_moderation_results.tsv
- targeted_coupling_models.tsv
- BCCI_APOE_coefficients_noKO.tsv
- BCCI_APOE_interactions_noKO.tsv
- BCCI_APOE_model_stats_noKO.tsv
- Mass_APOE_coefficients_noKO.tsv
- Mass_APOE_interactions_noKO.tsv
- Mass_APOE_model_stats_noKO.tsv
- BCCI_all_models_full_results.csv
- BCCI_models_summary.csv
- BCCI_significant_predictors.csv
- PLS_all_domains_scores.tsv
- multiple heatmaps / PNGs / NIfTI circuit maps
"""

# ======================================================
# 1. IMPORTS
# ======================================================

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

from statsmodels.stats.multitest import multipletests
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import rankdata, norm
from scipy.ndimage import gaussian_filter

import statsmodels.api as sm
import statsmodels.formula.api as smf
import nibabel as nib

warnings.filterwarnings("ignore", category=FutureWarning)


# ======================================================
# 2. PATHS / SETTINGS
# ======================================================

BASE = "/mnt/newStor/paros/paros_WORK/aashika"

NETWORK_FILE = os.path.join(
    BASE,
    "results/network12_from_errts/Network12_amplitudes.tsv"
)

META_FILE = os.path.join(
    BASE,
    "data/metadata/cardiac_design_updated3.csv"
)

ICA_4D_FILE = os.path.join(
    BASE,
    "data/ICA/4DNetwork/Networks_12_4D.nii.gz"
)

OUTDIR = os.path.join(
    BASE,
    "results/network_cardiac_optimized_031326_refactored"
)

os.makedirs(OUTDIR, exist_ok=True)

BRAINMAPS_ENABLED = True
THR_PERCENTILE = 80
DISPLAY_SMOOTH_SIGMA = 0.75
USE_SQRT_LOGP_WEIGHT = True
MIN_N_UNIVARIATE = 8
MIN_N_MODERATION = 20
N_SPLITS_CV = 5
CV_RANDOM_STATE = 42
MIN_CLUSTER_SIZE = 25

POSSIBLE_COVARS = ["Age", "Sex_Male", "Exercise_Yes", "Mass"]
BCCI_RESIDUAL_COVARS = ["Age", "Sex_Male", "Mass"]

with open(os.path.join(OUTDIR, "run_config.json"), "w") as f:
    json.dump(
        {
            "NETWORK_FILE": NETWORK_FILE,
            "META_FILE": META_FILE,
            "ICA_4D_FILE": ICA_4D_FILE,
            "OUTDIR": OUTDIR,
            "THR_PERCENTILE": THR_PERCENTILE,
            "DISPLAY_SMOOTH_SIGMA": DISPLAY_SMOOTH_SIGMA,
            "USE_SQRT_LOGP_WEIGHT": USE_SQRT_LOGP_WEIGHT,
            "MIN_N_UNIVARIATE": MIN_N_UNIVARIATE,
            "MIN_N_MODERATION": MIN_N_MODERATION,
            "N_SPLITS_CV": N_SPLITS_CV,
            "CV_RANDOM_STATE": CV_RANDOM_STATE,
            "MIN_CLUSTER_SIZE": MIN_CLUSTER_SIZE,
        },
        f,
        indent=2,
    )


# ======================================================
# 3. CARDIAC METRICS / DOMAINS
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

CARDIAC_GROUPS = {
    "rate_control": [
        "Heart_Rate"
    ],
    "systolic_function": [
        "Stroke_Volume",
        "Cardiac_Output",
        "Ejection_Fraction",
        "Systolic_LV_Volume",
        "Systolic_RV",
        "Systolic_LA",
        "Systolic_RA",
        "Systolic_Myo"
    ],
    "diastolic_function": [
        "Diastolic_LV_Volume",
        "Diastolic_RV",
        "Diastolic_LA",
        "Diastolic_RA",
        "Diastolic_Myo"
    ]
}

metric_to_group = {m: g for g, ms in CARDIAC_GROUPS.items() for m in ms}
DOMAIN_ORDER = [f"{d}_z" for d in CARDIAC_GROUPS.keys()]


# ======================================================
# 4. HELPERS
# ======================================================

def safe_z(x):
    x = pd.to_numeric(x, errors="coerce")
    sd = np.nanstd(x)
    if np.isnan(sd) or sd == 0:
        return pd.Series(np.nan, index=getattr(x, "index", None))
    return (x - np.nanmean(x)) / sd


def simes_p(pvals):
    p = np.sort(np.asarray(pvals))
    m = len(p)
    if m == 0:
        return np.nan
    return np.min((m * p) / (np.arange(1, m + 1)))


def rank_inverse_normal(x):
    x = pd.to_numeric(x, errors="coerce")
    out = pd.Series(np.nan, index=x.index)
    valid = x.notna()
    xv = x.loc[valid]
    r = rankdata(xv, method="average")
    r = (r - 0.5) / len(r)
    out.loc[valid] = norm.ppf(r)
    return out


def safe_filename(s):
    s = str(s)
    for ch in [" ", "/", "\\", ":", ";", ",", "(", ")", "[", "]", "{", "}", "'", '"']:
        s = s.replace(ch, "_")
    return s


def robust_clip(arr, pct=99):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr, -1, 1
    vmax = np.percentile(np.abs(finite), pct)
    if vmax == 0 or np.isnan(vmax):
        return arr, -1, 1
    arr2 = np.clip(arr, -vmax, vmax)
    return arr2, -vmax, vmax


def save_heatmap(mat, title, filename, cmap="coolwarm", center=0,
                 figsize=(12, 7), annot=False, fmt=".2f",
                 vmin=None, vmax=None):
    plt.figure(figsize=figsize)
    sns.heatmap(
        mat.fillna(0),
        cmap=cmap,
        center=center,
        annot=annot,
        fmt=fmt,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.4,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, filename), dpi=300)
    plt.close()


def scale_matrix(mat):
    scaled = mat.copy()
    for col in scaled.columns:
        sd = scaled[col].std()
        if np.isfinite(sd) and sd > 0:
            scaled[col] = (scaled[col] - scaled[col].mean()) / sd
    return scaled


def build_weighted_map(weights, ica_data):
    return np.tensordot(ica_data, weights, axes=([3], [0]))


def save_display_png(weighted_map, out_png, smooth_sigma=0.75,
                     clip_pct=99, title=None):
    disp = gaussian_filter(weighted_map, sigma=smooth_sigma)
    disp, _, _ = robust_clip(disp, pct=clip_pct)
    proj = np.max(np.abs(disp), axis=0)

    plt.figure(figsize=(7, 5))
    plt.imshow(proj.T, origin="lower", cmap="magma", aspect="auto")
    plt.colorbar(shrink=0.8)
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def save_nifti(arr, affine, out_file):
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine), out_file)


def save_nifti_pair(weighted_map, affine, out_prefix, thr_percentile=80):
    """
    Saves:
      - raw signed weighted map
      - thresholded signed weighted map
      - binary absolute threshold mask
      - metadata .txt with threshold value and voxel count
    """
    save_nifti(weighted_map, affine, out_prefix + ".nii.gz")

    finite = weighted_map[np.isfinite(weighted_map)]
    if finite.size == 0:
        thr = np.nan
        thr_map = np.zeros_like(weighted_map)
        mask = np.zeros_like(weighted_map, dtype=np.uint8)
    else:
        thr = np.percentile(np.abs(finite), thr_percentile)
        mask = (np.abs(weighted_map) >= thr).astype(np.uint8)
        thr_map = np.where(mask == 1, weighted_map, 0)

    save_nifti(thr_map, affine, out_prefix + f"_thr{thr_percentile}.nii.gz")
    save_nifti(mask, affine, out_prefix + f"_thr{thr_percentile}_mask.nii.gz")

    with open(out_prefix + f"_thr{thr_percentile}_info.txt", "w") as f:
        f.write(f"threshold_percentile\t{thr_percentile}\n")
        f.write(f"absolute_threshold\t{thr}\n")
        f.write(f"n_nonzero_raw\t{int(np.sum(np.abs(weighted_map) > 0))}\n")
        f.write(f"n_nonzero_thresholded\t{int(np.sum(mask))}\n")

    return thr_map, mask, thr


def results_to_weights(results_df, network_cols, mode="beta_p_sqrt"):
    weights = np.zeros(len(network_cols), dtype=float)

    for _, r in results_df.iterrows():
        net = r["Network"]
        if net not in network_cols:
            continue

        idx = network_cols.index(net)
        beta = float(r["Beta"])

        if mode == "beta":
            w = beta
        else:
            p = max(float(r["P"]), 1e-12)
            if mode == "beta_p":
                w = beta * (-np.log10(p))
            elif mode == "beta_p_sqrt":
                w = beta * np.sqrt(-np.log10(p))
            else:
                raise ValueError(f"Unknown weighting mode: {mode}")

        weights[idx] += w

    return weights


def save_circuit_maps(results_df, network_cols, ica_data, affine, out_prefix,
                      weight_mode="beta_p_sqrt", thr_percentile=80,
                      save_png=True, png_title=None):
    weights = results_to_weights(results_df, network_cols, mode=weight_mode)

    if np.max(np.abs(weights)) == 0:
        return None, None, None

    weights = weights / np.max(np.abs(weights))
    weighted_map = build_weighted_map(weights, ica_data)
    thr_map, mask, thr = save_nifti_pair(
        weighted_map,
        affine,
        out_prefix,
        thr_percentile=thr_percentile
    )

    if save_png:
        save_display_png(
            weighted_map,
            out_prefix + ".png",
            smooth_sigma=DISPLAY_SMOOTH_SIGMA,
            clip_pct=99,
            title=png_title
        )
        save_display_png(
            thr_map,
            out_prefix + f"_thr{thr_percentile}.png",
            smooth_sigma=DISPLAY_SMOOTH_SIGMA,
            clip_pct=99,
            title=(png_title + f" thr{thr_percentile}" if png_title else None)
        )

    return weighted_map, thr_map, mask


def fit_multivariate_domain_model(subdf, domain, network_cols, covars):
    y = pd.to_numeric(subdf[domain], errors="coerce")
    X = subdf[network_cols + covars].apply(pd.to_numeric, errors="coerce")
    X = sm.add_constant(X)
    model = sm.OLS(y, X, missing="drop").fit()
    return model


def compute_domain_scores_pca(df, cardiac_groups):
    """
    Creates rate_control_z, systolic_function_z, diastolic_function_z
    using PCA within domain when >1 metric exists.
    """
    out = df.copy()

    for domain, metrics in cardiac_groups.items():
        existing = [m for m in metrics if m in out.columns]

        if len(existing) == 0:
            print(domain, ": no existing metrics")
            continue

        tmp = out[existing].apply(pd.to_numeric, errors="coerce")
        tmp_z = tmp.apply(safe_z)
        valid = tmp_z.dropna(how="all")

        if len(existing) == 1:
            out[domain + "_z"] = safe_z(tmp_z.iloc[:, 0])
            print(domain, ": single metric domain")
            continue

        if len(valid) < 3:
            print(domain, ": not enough data for PCA")
            continue

        valid_imp = valid.copy()
        for c in valid_imp.columns:
            valid_imp[c] = valid_imp[c].fillna(valid_imp[c].mean())

        pca = PCA(n_components=1)
        comp = pca.fit_transform(valid_imp)

        out.loc[valid_imp.index, domain + "_z"] = comp[:, 0]
        out[domain + "_z"] = safe_z(out[domain + "_z"])

        if domain == "systolic_function" and "Ejection_Fraction" in existing:
            corr = np.corrcoef(
                out.loc[valid_imp.index, domain + "_z"],
                out.loc[valid_imp.index, "Ejection_Fraction"]
            )[0, 1]
            if np.isfinite(corr) and corr < 0:
                out[domain + "_z"] = -out[domain + "_z"]

        loadings = pd.Series(pca.components_[0], index=existing)
        loadings.to_csv(
            os.path.join(OUTDIR, f"{domain}_pca_loadings.tsv"),
            sep="\t",
            header=False
        )

        print(
            f"{domain}: PCA variance explained =",
            round(pca.explained_variance_ratio_[0], 3)
        )

    return out


def make_scatter_with_fit(x, y, xlabel, ylabel, title, outfile):
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(tmp) < 5:
        return

    model = smf.ols("y ~ x", data=tmp).fit()
    xfit = np.linspace(tmp["x"].min(), tmp["x"].max(), 200)
    pred = model.get_prediction(pd.DataFrame({"x": xfit})).summary_frame()

    plt.figure(figsize=(6, 5))
    plt.scatter(tmp["x"], tmp["y"], s=70)
    plt.plot(xfit, pred["mean"], linewidth=2)
    plt.fill_between(
        xfit,
        pred["mean_ci_lower"],
        pred["mean_ci_upper"],
        alpha=0.2
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.text(
        0.05, 0.95,
        f"R² = {model.rsquared:.2f}\np = {model.pvalues['x']:.3g}",
        transform=plt.gca().transAxes,
        verticalalignment="top"
    )
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


def fit_formula_if_possible(formula, data, required_n=10):
    try:
        model = smf.ols(formula, data=data).fit()
        if int(model.nobs) < required_n:
            return None
        return model
    except Exception as e:
        print(f"Skipping formula due to error: {formula}\n{e}")
        return None


# ======================================================
# 5. FILE CHECKS
# ======================================================

for f in [NETWORK_FILE, META_FILE, ICA_4D_FILE]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing file: {f}")


# ======================================================
# 6. LOAD DATA
# ======================================================

print("Loading metadata")
meta = pd.read_csv(META_FILE)

print("Loading network amplitudes")
nets = pd.read_csv(NETWORK_FILE, sep="\t")

meta.columns = meta.columns.str.strip()
nets.columns = nets.columns.str.strip()

if "Arunno" not in meta.columns:
    raise ValueError("'Arunno' missing from metadata")
if "Arunno" not in nets.columns:
    raise ValueError("'Arunno' missing from network file")

meta["Arunno"] = meta["Arunno"].astype(str).str.strip()
nets["Arunno"] = nets["Arunno"].astype(str).str.strip()

missing_metrics = [c for c in CARDIAC_METRICS if c not in meta.columns]
if missing_metrics:
    raise ValueError(f"Missing cardiac metrics in metadata: {missing_metrics}")

for c in CARDIAC_METRICS:
    meta[c] = pd.to_numeric(meta[c], errors="coerce")

df = pd.merge(meta, nets, on="Arunno", how="inner")

if "Age" in df.columns:
    df["Age_z"] = safe_z(df["Age"])
if "Mass" in df.columns:
    df["Mass_z"] = safe_z(df["Mass"])

print("Subjects after merge:", len(df))

network_cols = [c for c in df.columns if c.startswith("Amp_")]
if len(network_cols) == 0:
    raise ValueError("No network columns found starting with 'Amp_'")

covars = [c for c in POSSIBLE_COVARS if c in df.columns]
print("Networks detected:", len(network_cols))
print("Covariates used:", covars)

# z-score copies for modeling
# Only transform the core cardiac metrics and networks here.
df_z = df.copy()
scaler = StandardScaler()
df_z[CARDIAC_METRICS] = scaler.fit_transform(df_z[CARDIAC_METRICS])
df_z[network_cols] = scaler.fit_transform(df_z[network_cols])

# Maintain Age_z / Mass_z explicitly from raw columns
if "Age" in df_z.columns:
    df_z["Age_z"] = safe_z(df_z["Age"])
if "Mass" in df_z.columns:
    df_z["Mass_z"] = safe_z(df_z["Mass"])

# Domain scores
print("Computing cardiac domain scores")
df_z = compute_domain_scores_pca(df_z, CARDIAC_GROUPS)
domain_vars = [d for d in DOMAIN_ORDER if d in df_z.columns]
print("Domain variables:", domain_vars)

geno_map = {
    "E22": "E2",
    "E2HN": "E2",
    "E33": "E3",
    "E3HN": "E3",
    "E44": "E4",
    "E4HN": "E4",
    "KO": "KO"
}

for d in [df, df_z]:
    if "Genotype" in d.columns:
        d["Genotype_group"] = d["Genotype"].map(geno_map)
        d["Genotype_group"] = d["Genotype_group"].astype("category")

if all(c in df_z.columns for c in ["Genotype", "Genotype_group"]):
    print(df_z[["Genotype", "Genotype_group"]].head(20))


# ======================================================
# 7. STAGE A: UNIVARIATE NETWORK × METRIC REGRESSION TABLE
# ======================================================

print("\nRunning univariate network × cardiac metric regressions")

results = []

for cardiac in CARDIAC_METRICS:
    for net in network_cols:
        cols = ["Arunno", cardiac, net] + covars
        subdf = df_z[cols].dropna()
        n = len(subdf)

        if n < MIN_N_UNIVARIATE:
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
res["FDR_global"] = multipletests(res["P"].values, method="fdr_bh")[1]
res = res.sort_values(["FDR_global", "P"])
res.to_csv(
    os.path.join(OUTDIR, "network_cardiac_results_allmetrics.tsv"),
    sep="\t",
    index=False
)

print("Saved network_cardiac_results_allmetrics.tsv")


# ======================================================
# 8. HIERARCHICAL FDR
# ======================================================

print("Running hierarchical FDR")

res["Group"] = res["CardiacMetric"].map(metric_to_group)
if res["Group"].isna().any():
    missing_group_metrics = res.loc[res["Group"].isna(), "CardiacMetric"].unique().tolist()
    raise ValueError(f"Some cardiac metrics not mapped to groups: {missing_group_metrics}")

group_p = (
    res.groupby("Group")["P"]
    .apply(lambda s: simes_p(s.values))
    .reset_index()
    .rename(columns={"P": "Group_P"})
)
group_p["Group_FDR"] = multipletests(group_p["Group_P"], method="fdr_bh")[1]

res = res.merge(group_p, on="Group", how="left")
res["FDR_within_group"] = np.nan

for g in res["Group"].unique():
    idx = res["Group"] == g
    res.loc[idx, "FDR_within_group"] = multipletests(
        res.loc[idx, "P"], method="fdr_bh"
    )[1]

res["Sig_hier"] = (
    (res["Group_FDR"] < 0.05) &
    (res["FDR_within_group"] < 0.05)
)

res.to_csv(
    os.path.join(OUTDIR, "network_cardiac_results_hierFDR.tsv"),
    sep="\t",
    index=False
)

print("Saved network_cardiac_results_hierFDR.tsv")


# ======================================================
# 9. METRIC-LEVEL HEATMAPS / CLUSTERING
# ======================================================

print("Creating metric-level heatmaps and clustering")

mlogp = res.pivot_table(index="Network", columns="CardiacMetric", values="P")
mlogp = -np.log10(mlogp)

beta_metric = res.pivot_table(index="Network", columns="CardiacMetric", values="Beta")

save_heatmap(
    mlogp,
    "Network–Cardiac (-log10 p)",
    "heatmap_logp.png",
    cmap="viridis",
    center=None,
    figsize=(12, 7),
)

save_heatmap(
    beta_metric,
    "Network–Cardiac beta",
    "heatmap_beta.png",
    cmap="coolwarm",
    center=0,
    figsize=(12, 7),
)

X_cluster = StandardScaler().fit_transform(beta_metric.fillna(0))
Z = linkage(X_cluster, method="ward")

plt.figure(figsize=(10, 6))
dendrogram(Z, labels=beta_metric.index.tolist(), leaf_rotation=90)
plt.title("Network clustering by cardiac profile")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "network_dendrogram.png"), dpi=300)
plt.close()


# ======================================================
# 10. STAGE B: DOMAIN-SPECIFIC MULTIVARIATE MODELS
# ======================================================

print("\nEstimating multivariate domain models")

domain_models = {}
domain_weights = {}
domain_weight_rows = []
domain_model_rows = []

for domain_var in domain_vars:
    cols = [domain_var] + network_cols + covars
    subdf = df_z[cols].dropna()

    if len(subdf) < max(15, len(network_cols) + len(covars) + 1):
        print("Skipping", domain_var, ": insufficient data for multivariate model")
        continue

    model = fit_multivariate_domain_model(subdf, domain_var, network_cols, covars)
    domain_models[domain_var] = model

    beta = model.params[network_cols]
    domain_weights[domain_var] = beta

    beta.to_csv(
        os.path.join(OUTDIR, f"weights_{domain_var}.tsv"),
        sep="\t"
    )

    domain_model_rows.append({
        "Domain": domain_var,
        "N": int(model.nobs),
        "R2": model.rsquared,
        "Adj_R2": model.rsquared_adj,
        "F": model.fvalue,
        "P": model.f_pvalue
    })

    for net in network_cols:
        domain_weight_rows.append({
            "Domain": domain_var,
            "Network": net.replace("Amp_", ""),
            "Beta": model.params.get(net, np.nan),
            "SE": model.bse.get(net, np.nan),
            "T": model.tvalues.get(net, np.nan),
            "P": model.pvalues.get(net, np.nan)
        })

domain_weight_stats = pd.DataFrame(domain_weight_rows)
if len(domain_weight_stats) > 0:
    domain_weight_stats["FDR"] = multipletests(domain_weight_stats["P"], method="fdr_bh")[1]
    domain_weight_stats.to_csv(
        os.path.join(OUTDIR, "network_weight_statistics_domains.tsv"),
        sep="\t",
        index=False
    )

domain_model_stats = pd.DataFrame(domain_model_rows)
domain_model_stats.to_csv(
    os.path.join(OUTDIR, "domain_model_statistics.tsv"),
    sep="\t",
    index=False
)

print("Saved domain-level weight and model statistics")


# ======================================================
# 11. DOMAIN-SPECIFIC BCCI (FULL DATA + CV)
# ======================================================

print("\nComputing domain-specific BCCI scores")

for domain_var, beta in domain_weights.items():
    bcci_name = "BCCI_" + domain_var.replace("_z", "")
    df_z[bcci_name] = df_z[network_cols].dot(beta)
    df_z[bcci_name + "_z"] = safe_z(df_z[bcci_name])

print("Computing cross-validated domain-specific BCCI")

kf = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=CV_RANDOM_STATE)

for domain_var in domain_weights.keys():
    bcci_cv_name = "BCCI_" + domain_var.replace("_z", "") + "_cv"
    df_z[bcci_cv_name] = np.nan

    cols = [domain_var] + network_cols + covars
    subdf = df_z[cols].dropna().copy()

    if len(subdf) < N_SPLITS_CV:
        continue

    for train_idx, test_idx in kf.split(subdf):
        train = subdf.iloc[train_idx]
        test = subdf.iloc[test_idx]

        model = fit_multivariate_domain_model(train, domain_var, network_cols, covars)
        beta = model.params[network_cols]

        scores = test[network_cols].dot(beta)
        df_z.loc[test.index, bcci_cv_name] = scores

    df_z[bcci_cv_name + "_z"] = safe_z(df_z[bcci_cv_name])

# Residualize domain-specific BCCI for optional downstream use
covars_for_bcci = [c for c in BCCI_RESIDUAL_COVARS if c in df_z.columns]

for domain_var in domain_weights.keys():
    bcci_z = "BCCI_" + domain_var.replace("_z", "") + "_z"
    resid_name = bcci_z.replace("_z", "_resid")
    resid_z_name = resid_name + "_z"

    if len(covars_for_bcci) > 0 and bcci_z in df_z.columns:
        tmp = df_z[[bcci_z] + covars_for_bcci].dropna().copy()
        X_cov = sm.add_constant(tmp[covars_for_bcci].apply(pd.to_numeric, errors="coerce"))
        model_cov = sm.OLS(tmp[bcci_z], X_cov).fit()
        df_z.loc[tmp.index, resid_name] = model_cov.resid
        df_z[resid_z_name] = safe_z(df_z[resid_name])
    elif bcci_z in df_z.columns:
        df_z[resid_z_name] = df_z[bcci_z]

df_z.to_csv(
    os.path.join(OUTDIR, "BCCI_subject_scores.tsv"),
    sep="\t",
    index=False
)

print("Saved BCCI_subject_scores.tsv")


# ======================================================
# 12. DOMAIN WEIGHT HEATMAPS
# ======================================================

print("Creating domain-level heatmaps")

if len(domain_weights) > 0:
    weight_matrix = pd.DataFrame(domain_weights)
    weight_matrix.index = [i.replace("Amp_", "") for i in weight_matrix.index]

    save_heatmap(
        weight_matrix,
        "Brain Network → Cardiac Domain Coupling",
        "brain_cardiac_heatmap.png",
        cmap="coolwarm",
        center=0,
        figsize=(8, 8),
        annot=True,
        fmt=".2f"
    )

    g = sns.clustermap(
        weight_matrix,
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        figsize=(8, 10)
    )
    g.savefig(os.path.join(OUTDIR, "brain_cardiac_clustermap.png"), dpi=300)
    plt.close("all")

metric_matrix = beta_metric.reindex(network_cols)
domain_matrix = pd.DataFrame(index=network_cols)

for domain_name, metrics in CARDIAC_GROUPS.items():
    sub = res[res["CardiacMetric"].isin(metrics)]
    net_beta = sub.groupby("Network")["Beta"].mean()
    domain_matrix[domain_name] = [net_beta.get(n, 0) for n in network_cols]

metric_scaled = scale_matrix(metric_matrix.fillna(0))
domain_scaled = scale_matrix(domain_matrix.fillna(0))

v_metric = np.percentile(np.abs(metric_scaled.values), 95)
v_domain = np.percentile(np.abs(domain_scaled.values), 95)

save_heatmap(
    metric_scaled,
    "Brain–Heart Coupling by Cardiac Metric",
    "network_importance_by_metric.png",
    cmap="coolwarm",
    center=0,
    figsize=(10, 7),
    vmin=-v_metric,
    vmax=v_metric,
)

save_heatmap(
    domain_scaled,
    "Brain–Heart Coupling by Cardiac Domain",
    "network_importance_by_domain.png",
    cmap="coolwarm",
    center=0,
    figsize=(8, 6),
    annot=True,
    fmt=".2f",
    vmin=-v_domain,
    vmax=v_domain,
)


# ======================================================
# 13. LOAD ICA ONCE
# ======================================================

print("Loading ICA file")
img = nib.load(ICA_4D_FILE)
ica_data = img.get_fdata()
affine = img.affine
n_components = ica_data.shape[3]

if len(network_cols) != n_components:
    print(f"Warning: network columns ({len(network_cols)}) != ICA components ({n_components})")


# ======================================================
# 14. METRIC-SPECIFIC CIRCUIT MAPS
# ======================================================

if BRAINMAPS_ENABLED:
    print("Creating individual metric brain maps")

    for metric in res["CardiacMetric"].unique():
        sub = res[res["CardiacMetric"] == metric].copy()
        if len(sub) == 0:
            continue

        out_prefix = os.path.join(OUTDIR, f"cardiac_map_{safe_filename(metric)}")
        weight_mode = "beta_p_sqrt" if USE_SQRT_LOGP_WEIGHT else "beta_p"

        save_circuit_maps(
            sub,
            network_cols,
            ica_data,
            affine,
            out_prefix,
            weight_mode=weight_mode,
            thr_percentile=THR_PERCENTILE,
            save_png=True,
            png_title=metric
        )


# ======================================================
# 15. DOMAIN-SPECIFIC CIRCUIT MAPS
# ======================================================

if BRAINMAPS_ENABLED:
    print("Creating domain-specific brain maps from univariate results")

    for domain_name, metrics in CARDIAC_GROUPS.items():
        sub = res[res["CardiacMetric"].isin(metrics)].copy()
        if len(sub) == 0:
            continue

        out_prefix = os.path.join(OUTDIR, f"cardiac_domain_map_{safe_filename(domain_name)}")
        weight_mode = "beta_p_sqrt" if USE_SQRT_LOGP_WEIGHT else "beta_p"

        save_circuit_maps(
            sub,
            network_cols,
            ica_data,
            affine,
            out_prefix,
            weight_mode=weight_mode,
            thr_percentile=THR_PERCENTILE,
            save_png=True,
            png_title=domain_name
        )

if BRAINMAPS_ENABLED and len(domain_weights) > 0:
    print("Creating domain-specific brain maps from multivariate weights")

    for domain_var, beta in domain_weights.items():
        tmp_df = pd.DataFrame({
            "Network": network_cols,
            "Beta": beta.reindex(network_cols).values,
            "P": np.repeat(0.01, len(network_cols))
        })

        out_prefix = os.path.join(
            OUTDIR,
            f"cardiac_domain_map_multivar_{safe_filename(domain_var.replace('_z', ''))}"
        )

        save_circuit_maps(
            tmp_df,
            network_cols,
            ica_data,
            affine,
            out_prefix,
            weight_mode="beta",
            thr_percentile=THR_PERCENTILE,
            save_png=True,
            png_title=f"{domain_var} multivariate"
        )


# ======================================================
# 16. PLS LATENT BRAIN–HEART MODEL USING ALL DOMAINS
# ======================================================

print("Running PLS model using all cardiac domains")

pls_domain_vars = [d for d in DOMAIN_ORDER if d in df_z.columns]

if len(pls_domain_vars) > 0:
    X_pls = df_z[network_cols].apply(pd.to_numeric, errors="coerce")
    Y_pls = df_z[pls_domain_vars].apply(pd.to_numeric, errors="coerce")

    valid_idx = X_pls.notna().all(axis=1) & Y_pls.notna().all(axis=1)
    X_pls_valid = X_pls.loc[valid_idx].values
    Y_pls_valid = Y_pls.loc[valid_idx].values

    n_comp = min(2, len(pls_domain_vars), len(network_cols))
    pls = PLSRegression(n_components=n_comp)
    pls.fit(X_pls_valid, Y_pls_valid)
    pred_pls = pls.predict(X_pls_valid)

    pls_r2_overall = r2_score(Y_pls_valid, pred_pls, multioutput="variance_weighted")
    print("PLS overall weighted R2:", round(pls_r2_overall, 3))

    kf_pls = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=1)
    pred_cv = np.zeros_like(Y_pls_valid, dtype=float)

    for train, test in kf_pls.split(X_pls_valid):
        pls_cv = PLSRegression(n_components=n_comp)
        pls_cv.fit(X_pls_valid[train], Y_pls_valid[train])
        pred_cv[test] = pls_cv.predict(X_pls_valid[test])

    pls_r2_cv = r2_score(Y_pls_valid, pred_cv, multioutput="variance_weighted")
    print("Cross-validated PLS overall weighted R2:", round(pls_r2_cv, 3))

    # Subject scores from first latent variable
    x_scores = pls.x_scores_[:, 0]
    y_scores = pls.y_scores_[:, 0]
    pls_scores_df = pd.DataFrame(index=X_pls.loc[valid_idx].index)
    pls_scores_df["PLS_brain_score"] = x_scores
    pls_scores_df["PLS_heart_score"] = y_scores
    pls_scores_df["PLS_brain_score_z"] = safe_z(pls_scores_df["PLS_brain_score"])
    pls_scores_df["PLS_heart_score_z"] = safe_z(pls_scores_df["PLS_heart_score"])
    pls_scores_df.to_csv(
        os.path.join(OUTDIR, "PLS_all_domains_scores.tsv"),
        sep="\t"
    )

    # Save weight tables
    pls_x_weights = pd.DataFrame({
        "Network": network_cols,
        "PLS1_x_weight": pls.x_weights_[:, 0]
    })
    pls_x_weights.to_csv(
        os.path.join(OUTDIR, "PLS_all_domains_xweights.tsv"),
        sep="\t",
        index=False
    )

    pls_y_weights = pd.DataFrame({
        "Domain": pls_domain_vars,
        "PLS1_y_weight": pls.y_weights_[:, 0]
    })
    pls_y_weights.to_csv(
        os.path.join(OUTDIR, "PLS_all_domains_yweights.tsv"),
        sep="\t",
        index=False
    )

    if BRAINMAPS_ENABLED:
        pls_weights = pls.x_weights_[:, 0].flatten()
        pls_df = pd.DataFrame({
            "Network": network_cols,
            "Beta": pls_weights,
            "P": np.repeat(0.01, len(pls_weights))
        })

        out_prefix = os.path.join(OUTDIR, "brain_heart_circuit_map_PLS_all_domains")
        save_circuit_maps(
            pls_df,
            network_cols,
            ica_data,
            affine,
            out_prefix,
            weight_mode="beta",
            thr_percentile=THR_PERCENTILE,
            save_png=True,
            png_title="PLS brain-heart circuit (all domains)"
        )


# ======================================================
# 17. BCCI VALIDATION AGAINST DOMAINS / METRICS
# ======================================================

print("Validating domain-specific BCCI")

validation_rows = []

for domain_var in domain_vars:
    bcci_z = "BCCI_" + domain_var.replace("_z", "") + "_cv_z"
    if bcci_z not in df_z.columns:
        continue

    tmp = df_z[[bcci_z, domain_var]].dropna().copy()
    if len(tmp) < 10:
        continue

    model = smf.ols(f"{domain_var} ~ {bcci_z}", data=tmp).fit()

    validation_rows.append({
        "Outcome": domain_var,
        "Predictor": bcci_z,
        "N": len(tmp),
        "Beta": model.params.get(bcci_z, np.nan),
        "P": model.pvalues.get(bcci_z, np.nan),
        "R2": model.rsquared
    })

validation_df = pd.DataFrame(validation_rows)
validation_df.to_csv(
    os.path.join(OUTDIR, "BCCI_domain_validation.tsv"),
    sep="\t",
    index=False
)

bcci_metric_results = []

for metric in CARDIAC_METRICS:
    dom = metric_to_group.get(metric)
    if dom is None:
        continue

    predictor = f"BCCI_{dom}_cv_z"
    if predictor not in df_z.columns:
        continue

    tmp = df_z[[predictor, metric]].copy()
    tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
    tmp = tmp.dropna()

    if len(tmp) < 15:
        continue

    model = smf.ols(f"{metric} ~ {predictor}", data=tmp).fit()

    bcci_metric_results.append({
        "Metric": metric,
        "Domain": dom,
        "Predictor": predictor,
        "N": len(tmp),
        "Beta_BCCI": model.params.get(predictor, np.nan),
        "P_BCCI": model.pvalues.get(predictor, np.nan),
        "R2": model.rsquared
    })

bcci_metric_results = pd.DataFrame(bcci_metric_results)
if len(bcci_metric_results) > 0:
    bcci_metric_results["FDR_BCCI"] = multipletests(
        bcci_metric_results["P_BCCI"], method="fdr_bh"
    )[1]
    bcci_metric_results = bcci_metric_results.sort_values("P_BCCI")
    bcci_metric_results.to_csv(
        os.path.join(OUTDIR, "BCCI_vs_all_cardiac_metrics.tsv"),
        sep="\t",
        index=False
    )


# ======================================================
# 18. PREP MODERATION VARIABLES
# ======================================================

print("Preparing moderation variables")

mods = ["Sex_Male", "Age", "Exercise_Yes", "Mass", "E4_Genotype", "HN"]
mods = [m for m in mods if m in df_z.columns]

for col in mods:
    df_z[col] = pd.to_numeric(df_z[col], errors="coerce")

if "Age" in df_z.columns:
    df_z["Age_z"] = safe_z(df_z["Age"])
if "Mass" in df_z.columns:
    df_z["Mass_z"] = safe_z(df_z["Mass"])

for m in CARDIAC_METRICS:
    if m in df_z.columns:
        df_z[m + "_z"] = safe_z(df_z[m])
        df_z[m + "_rin"] = rank_inverse_normal(df_z[m])


# ======================================================
# 19. MODERATION MODELS
# ======================================================

print("Running moderation models")

moderation_results = []
mods_to_test = []

if "Age_z" in df_z.columns:
    mods_to_test.append("Age_z")
for c in ["Sex_Male", "Exercise_Yes", "E4_Genotype"]:
    if c in df_z.columns:
        mods_to_test.append(c)
if "Mass_z" in df_z.columns:
    mods_to_test.append("Mass_z")
for c in ["HN"]:
    if c in df_z.columns:
        mods_to_test.append(c)

for metric in CARDIAC_METRICS:
    metric_z = metric + "_z"
    if metric_z not in df_z.columns:
        continue

    dom = metric_to_group.get(metric)
    predictor = f"BCCI_{dom}_cv_z"
    if predictor not in df_z.columns:
        continue

    for mod in mods_to_test:
        cols = [predictor, metric_z, mod]
        tmp = df_z[cols].dropna().copy()

        if len(tmp) < MIN_N_MODERATION:
            continue

        formula = f"{metric_z} ~ {predictor} * {mod}"
        model = smf.ols(formula, data=tmp).fit()

        interaction_term = f"{predictor}:{mod}"
        if interaction_term not in model.params.index:
            alt_term = f"{mod}:{predictor}"
            interaction_term = alt_term if alt_term in model.params.index else interaction_term

        moderation_results.append({
            "Metric": metric,
            "Domain": dom,
            "Moderator": mod,
            "Predictor": predictor,
            "N": len(tmp),
            "Beta_BCCI": model.params.get(predictor, np.nan),
            "P_BCCI": model.pvalues.get(predictor, np.nan),
            "Beta_Mod": model.params.get(mod, np.nan),
            "P_Mod": model.pvalues.get(mod, np.nan),
            "Beta_Interaction": model.params.get(interaction_term, np.nan),
            "P_Interaction": model.pvalues.get(interaction_term, np.nan),
            "R2": model.rsquared
        })

moderation_results = pd.DataFrame(moderation_results)

if len(moderation_results) > 0:
    moderation_results["FDR_Interaction"] = multipletests(
        moderation_results["P_Interaction"].fillna(1.0), method="fdr_bh"
    )[1]
    moderation_results = moderation_results.sort_values("P_Interaction")
    moderation_results.to_csv(
        os.path.join(OUTDIR, "BCCI_moderation_results.tsv"),
        sep="\t",
        index=False
    )


# ======================================================
# 20. PLOT SIGNIFICANT INTERACTIONS
# ======================================================

print("Plotting significant interactions")

sig_int = pd.DataFrame()
if len(moderation_results) > 0:
    sig_int = moderation_results[moderation_results["FDR_Interaction"] < 0.10].copy()

for _, row in sig_int.iterrows():
    metric = row["Metric"]
    mod = row["Moderator"]
    predictor = row["Predictor"]
    metric_z = metric + "_z"

    cols = [predictor, metric_z, mod]
    tmp = df_z[cols].dropna().copy()

    if len(tmp) < MIN_N_MODERATION:
        continue

    plt.figure(figsize=(6, 5))
    unique_vals = sorted(tmp[mod].dropna().unique())

    if len(unique_vals) <= 2:
        for val in unique_vals:
            sub = tmp[tmp[mod] == val]
            if len(sub) < 3:
                continue

            plt.scatter(
                sub[predictor], sub[metric_z],
                s=70, alpha=0.8, label=f"{mod}={val}"
            )

            fit = smf.ols(f"{metric_z} ~ {predictor}", data=sub).fit()
            xfit = np.linspace(sub[predictor].min(), sub[predictor].max(), 100)
            yfit = fit.params["Intercept"] + fit.params[predictor] * xfit
            plt.plot(xfit, yfit, linewidth=2)

        plt.legend()
    else:
        median_val = tmp[mod].median()
        tmp["Group"] = np.where(tmp[mod] <= median_val, "Low", "High")

        for grp in ["Low", "High"]:
            sub = tmp[tmp["Group"] == grp]
            if len(sub) < 3:
                continue

            plt.scatter(
                sub[predictor], sub[metric_z],
                s=70, alpha=0.8, label=f"{mod} {grp}"
            )

            fit = smf.ols(f"{metric_z} ~ {predictor}", data=sub).fit()
            xfit = np.linspace(sub[predictor].min(), sub[predictor].max(), 100)
            yfit = fit.params["Intercept"] + fit.params[predictor] * xfit
            plt.plot(xfit, yfit, linewidth=2)

        plt.legend()

    plt.xlabel(predictor)
    plt.ylabel(f"{metric} (z)")
    plt.title(f"{metric}: {predictor} × {mod}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTDIR, f"interaction_{safe_filename(metric)}_by_{safe_filename(mod)}.png"),
        dpi=300
    )
    plt.close()


# ======================================================
# 21. TARGETED DOMAIN INTERACTION MODELS
# ======================================================

print("Running targeted interaction models")

target_models = [
    ("systolic_function_z", "BCCI_systolic_function_cv_z", "Mass_z"),
    ("diastolic_function_z", "BCCI_diastolic_function_cv_z", "Sex_Male"),
    ("rate_control_z", "BCCI_rate_control_cv_z", "Exercise_Yes")
]

target_results = []

for metric, predictor, mod in target_models:
    if metric not in df_z.columns or predictor not in df_z.columns or mod not in df_z.columns:
        continue

    tmp = df_z[[predictor, metric, mod]].dropna().copy()
    if len(tmp) < MIN_N_MODERATION:
        continue

    model = smf.ols(f"{metric} ~ {predictor} * {mod}", data=tmp).fit()

    interaction_term = f"{predictor}:{mod}"
    if interaction_term not in model.params.index:
        alt_term = f"{mod}:{predictor}"
        interaction_term = alt_term if alt_term in model.params.index else interaction_term

    target_results.append({
        "Metric": metric,
        "Moderator": mod,
        "Predictor": predictor,
        "N": len(tmp),
        "Beta_BCCI": model.params.get(predictor, np.nan),
        "P_BCCI": model.pvalues.get(predictor, np.nan),
        "Beta_Interaction": model.params.get(interaction_term, np.nan),
        "P_Interaction": model.pvalues.get(interaction_term, np.nan),
        "R2": model.rsquared
    })

target_results = pd.DataFrame(target_results)
target_results.to_csv(
    os.path.join(OUTDIR, "targeted_coupling_models.tsv"),
    sep="\t",
    index=False
)


# ======================================================
# 22. GENOTYPE-SPECIFIC CIRCUIT MAPS
# ======================================================

if BRAINMAPS_ENABLED and "E4_Genotype" in df_z.columns:
    print("Creating genotype-specific domain circuit maps")

    genotype_groups = {
        "NonE4": df_z[df_z["E4_Genotype"] == 0],
        "E4": df_z[df_z["E4_Genotype"] == 1]
    }

    maps = {}

    for gname, gdf in genotype_groups.items():
        results_rows = []

        for net in network_cols:
            if "Heart_Rate" not in gdf.columns:
                continue

            tmp = gdf[[net, "Heart_Rate"]].dropna().copy()
            if len(tmp) < 10:
                continue

            model = sm.OLS(tmp["Heart_Rate"], sm.add_constant(tmp[[net]])).fit()

            results_rows.append({
                "Network": net,
                "Beta": model.params.iloc[1],
                "P": max(model.pvalues.iloc[1], 1e-10)
            })

        if len(results_rows) == 0:
            continue

        g_res = pd.DataFrame(results_rows)
        out_prefix = os.path.join(OUTDIR, f"brain_heart_map_{gname}")
        weight_mode = "beta_p_sqrt" if USE_SQRT_LOGP_WEIGHT else "beta_p"

        weighted_map, _, _ = save_circuit_maps(
            g_res,
            network_cols,
            ica_data,
            affine,
            out_prefix,
            weight_mode=weight_mode,
            thr_percentile=THR_PERCENTILE,
            save_png=True,
            png_title=f"{gname} brain-heart"
        )

        maps[gname] = weighted_map

    if "E4" in maps and "NonE4" in maps:
        diff_map = maps["E4"] - maps["NonE4"]
        diff_prefix = os.path.join(OUTDIR, "brain_heart_map_E4_minus_NonE4")
        save_nifti_pair(diff_map, affine, diff_prefix, thr_percentile=THR_PERCENTILE)
        save_display_png(diff_map, diff_prefix + ".png", DISPLAY_SMOOTH_SIGMA, 99, "E4 - NonE4")


# ======================================================
# 23. FIGURE 4 STYLE SUMMARY
# ======================================================

print("Creating Figure 4 style panels")

df_fig4 = df_z.copy()

if "Genotype" in df_fig4.columns:
    df_fig4["Genotype_group"] = df_fig4["Genotype"].map(geno_map)
    df_fig4["Genotype_group"] = df_fig4["Genotype_group"].astype("category")

if "Sex" in df_fig4.columns:
    df_fig4["Sex"] = df_fig4["Sex"].astype("category")

if all(c in df_fig4.columns for c in ["BCCI_rate_control_cv_z", "Age", "Heart_Rate"]) and \
   "Genotype_group" in df_fig4.columns and "Sex" in df_fig4.columns:

    X = df_fig4[["BCCI_rate_control_cv_z", "Age", "Sex", "Genotype_group"]]
    X = pd.get_dummies(X, drop_first=True)
    X = sm.add_constant(X).astype(float)
    y = df_fig4["Heart_Rate"].astype(float)

    model = sm.OLS(y, X, missing="drop").fit()

    beta_bcci = model.params["BCCI_rate_control_cv_z"]
    pval = model.pvalues["BCCI_rate_control_cv_z"]
    r2 = model.rsquared
    ci_low, ci_high = model.conf_int().loc["BCCI_rate_control_cv_z"]

    xfit = np.linspace(df_fig4["BCCI_rate_control_cv_z"].min(), df_fig4["BCCI_rate_control_cv_z"].max(), 200)
    Xpred = pd.DataFrame(0, index=np.arange(len(xfit)), columns=X.columns)
    Xpred["const"] = 1
    Xpred["BCCI_rate_control_cv_z"] = xfit
    if "Age" in Xpred.columns:
        Xpred["Age"] = df_fig4["Age"].mean()
    Xpred = Xpred.astype(float)

    pred = model.get_prediction(Xpred).summary_frame()

    plt.figure(figsize=(6, 5))
    plt.scatter(df_fig4["BCCI_rate_control_cv_z"], y, s=60)
    plt.plot(xfit, pred["mean"], linewidth=2)
    plt.fill_between(xfit, pred["mean_ci_lower"], pred["mean_ci_upper"], alpha=0.25)
    plt.xlabel("Rate-control BCCI (cv z)")
    plt.ylabel("Heart Rate")
    plt.title("Brain–Heart Coupling")
    plt.text(
        0.05, 0.95,
        f"β = {beta_bcci:.2f}\n95% CI [{ci_low:.2f},{ci_high:.2f}]\np = {pval:.3g}\nR² = {r2:.2f}",
        transform=plt.gca().transAxes,
        verticalalignment="top"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "Figure4A_BCCI_vs_HeartRate.png"), dpi=300)
    plt.close()

if "Genotype_group" in df_fig4.columns and all(c in df_fig4.columns for c in ["BCCI_rate_control_cv_z", "Age", "Sex", "Heart_Rate"]):
    X1 = df_fig4[["BCCI_rate_control_cv_z", "Age", "Sex", "Genotype_group"]]
    X1 = pd.get_dummies(X1, drop_first=True)
    X1 = sm.add_constant(X1).astype(float)
    m1 = sm.OLS(df_fig4["Heart_Rate"], X1, missing="drop").fit()

    df_noKO = df_fig4[df_fig4["Genotype_group"] != "KO"].copy()

    X2 = df_noKO[["BCCI_rate_control_cv_z", "Age", "Sex", "Genotype_group"]]
    X2 = pd.get_dummies(X2, drop_first=True)
    X2 = sm.add_constant(X2).astype(float)
    m2 = sm.OLS(df_noKO["Heart_Rate"], X2, missing="drop").fit()

    coef_vals = [m1.params["BCCI_rate_control_cv_z"], m2.params["BCCI_rate_control_cv_z"]]
    p_vals = [m1.pvalues["BCCI_rate_control_cv_z"], m2.pvalues["BCCI_rate_control_cv_z"]]
    r2_vals = [m1.rsquared, m2.rsquared]
    ci1 = m1.conf_int().loc["BCCI_rate_control_cv_z"]
    ci2 = m2.conf_int().loc["BCCI_rate_control_cv_z"]
    errors = [coef_vals[0] - ci1[0], coef_vals[1] - ci2[0]]
    labels = ["All animals", "No KO"]

    plt.figure(figsize=(5, 4))
    bars = plt.bar(labels, coef_vals, yerr=errors, capsize=6)
    plt.ylabel("BCCI coefficient (Heart Rate)")
    plt.title("Model robustness")

    for i, b in enumerate(bars):
        plt.text(
            b.get_x() + b.get_width() / 2,
            coef_vals[i],
            f"β={coef_vals[i]:.2f}\np={p_vals[i]:.3g}\nR²={r2_vals[i]:.2f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "Figure4B_model_robustness.png"), dpi=300)
    plt.close()

domain_beta = {}
domain_p = {}
domain_r2 = {}
domain_ci = {}

panel_c_pairs = [
    ("rate_control_z", "BCCI_rate_control_cv_z"),
    ("systolic_function_z", "BCCI_systolic_function_cv_z"),
    ("diastolic_function_z", "BCCI_diastolic_function_cv_z"),
]

for domain_var, predictor in panel_c_pairs:
    if domain_var not in df_fig4.columns or predictor not in df_fig4.columns:
        continue
    if "Genotype_group" not in df_fig4.columns or "Sex" not in df_fig4.columns:
        continue

    X = df_fig4[[predictor, "Age", "Sex", "Genotype_group"]]
    X = pd.get_dummies(X, drop_first=True)
    X = sm.add_constant(X).astype(float)
    y = df_fig4[domain_var].astype(float)

    model = sm.OLS(y, X, missing="drop").fit()

    domain_beta[domain_var] = model.params[predictor]
    domain_p[domain_var] = model.pvalues[predictor]
    domain_r2[domain_var] = model.rsquared
    domain_ci[domain_var] = model.conf_int().loc[predictor]

if len(domain_beta) > 0:
    plt.figure(figsize=(7, 4))
    domains = list(domain_beta.keys())
    betas = [domain_beta[d] for d in domains]
    errors = [betas[i] - domain_ci[domains[i]][0] for i in range(len(domains))]
    bars = plt.bar(domains, betas, yerr=errors, capsize=6)
    plt.ylabel("Matched BCCI effect")
    plt.title("Brain–heart coupling across cardiac domains")

    for i, d in enumerate(domains):
        ci_low, ci_high = domain_ci[d]
        plt.text(
            bars[i].get_x() + bars[i].get_width() / 2,
            betas[i],
            f"β={betas[i]:.2f}\n95%CI[{ci_low:.2f},{ci_high:.2f}]\np={domain_p[d]:.3g}\nR²={domain_r2[d]:.2f}",
            ha="center",
            va="bottom"
        )

    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "Figure4C_domain_effects.png"), dpi=300)
    plt.close()


# ======================================================
# 24. QUICK DIAGNOSTIC PLOTS
# ======================================================

if "BCCI_rate_control_cv_z" in df_z.columns and "Heart_Rate" in df_z.columns:
    make_scatter_with_fit(
        df_z["BCCI_rate_control_cv_z"],
        df_z["Heart_Rate"],
        "Rate-control BCCI (cv z)",
        "Heart Rate",
        "BCCI vs Heart Rate",
        os.path.join(OUTDIR, "BCCI_vs_HeartRate.png")
    )

    tmp = df_z[["BCCI_rate_control_cv_z", "Heart_Rate"]].dropna().copy()
    plt.figure(figsize=(6, 5))
    plt.scatter(tmp["BCCI_rate_control_cv_z"].rank(), tmp["Heart_Rate"].rank(), s=70)
    plt.xlabel("Ranked rate-control BCCI")
    plt.ylabel("Ranked Heart Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "spearman_rank_plot.png"), dpi=300)
    plt.close()


# ======================================================
# 25. DOMAIN BCCI SCATTER PLOTS
# ======================================================

print("Creating domain BCCI scatter plots")

domain_pairs = [
    ("BCCI_rate_control_cv_z", "Heart_Rate", "Rate Control"),
    ("BCCI_systolic_function_cv_z", "systolic_function_z", "Systolic Function"),
    ("BCCI_diastolic_function_cv_z", "diastolic_function_z", "Diastolic Function")
]

for predictor, outcome, label in domain_pairs:
    if predictor not in df_z.columns or outcome not in df_z.columns:
        print("Missing predictor or outcome for:", label)
        continue

    cols = [predictor, outcome] + covars
    tmp = df_z[cols].dropna().copy()

    if len(tmp) < 10:
        print("Skipping", label, "- insufficient data")
        continue

    rhs = predictor + (" + " + " + ".join(covars) if len(covars) > 0 else "")
    formula = outcome + " ~ " + rhs
    model = smf.ols(formula, data=tmp).fit()

    print("\n====================================")
    print("Domain:", label)
    print(model.summary())

    plt.figure(figsize=(6, 5))
    plt.scatter(tmp[predictor], tmp[outcome], s=70, alpha=0.8)

    xfit = np.linspace(tmp[predictor].min(), tmp[predictor].max(), 200)
    yfit = model.params["Intercept"] + model.params[predictor] * xfit

    plt.plot(xfit, yfit, linewidth=2)
    plt.xlabel(predictor)
    plt.ylabel(outcome)
    plt.title(label + " Brain–Cardiac Coupling")

    text_label = (
        "beta = %.2f\n"
        "p = %.4f\n"
        "R2 = %.2f"
        % (
            model.params[predictor],
            model.pvalues[predictor],
            model.rsquared
        )
    )

    plt.text(
        0.05,
        0.95,
        text_label,
        transform=plt.gca().transAxes,
        verticalalignment="top"
    )

    plt.tight_layout()
    outfile = os.path.join(OUTDIR, "BCCI_" + label.replace(" ", "_") + "_scatter.png")
    plt.savefig(outfile, dpi=300)
    plt.close()

print("\nDomain BCCI plots completed.")


# ======================================================
# 26. SAVE REGRESSION RESULTS + CORRELATIONS
# ======================================================

all_results = []

for predictor, outcome, label in domain_pairs:
    if predictor not in df_z.columns or outcome not in df_z.columns:
        continue

    cols = [predictor, outcome] + covars
    tmp = df_z[cols].dropna().copy()
    if len(tmp) < 5:
        continue

    rhs = predictor + (" + " + " + ".join(covars) if len(covars) > 0 else "")
    formula = outcome + " ~ " + rhs
    model = smf.ols(formula, data=tmp).fit()

    coef_table = pd.DataFrame({
        "Predictor": model.params.index,
        "Coef": model.params.values,
        "StdErr": model.bse.values,
        "t": model.tvalues.values,
        "p": model.pvalues.values
    })

    coef_table["Domain"] = label
    coef_table["Outcome"] = outcome
    coef_table["N"] = int(model.nobs)
    coef_table["R2"] = model.rsquared
    coef_table["Adj_R2"] = model.rsquared_adj
    coef_table["F_stat"] = model.fvalue
    coef_table["Model_p"] = model.f_pvalue

    all_results.append(coef_table)

if len(all_results) > 0:
    regression_results = pd.concat(all_results, ignore_index=True)
    regression_results.to_csv(
        os.path.join(OUTDIR, "BCCI_domain_regression_results.csv"),
        index=False
    )
    print("Saved regression results to CSV")

corr_vars = [
    "BCCI_rate_control_cv_z",
    "BCCI_systolic_function_cv_z",
    "BCCI_diastolic_function_cv_z",
    "Age",
    "Exercise_Yes",
    "Mass"
]
corr_vars = [c for c in corr_vars if c in df_z.columns]
if len(corr_vars) > 1:
    corr_matrix = df_z[corr_vars].corr()
    corr_matrix.to_csv(os.path.join(OUTDIR, "BCCI_covariate_correlations.csv"))
    print("Saved correlation matrix to CSV")
    print("\nCorrelation matrix:")
    print(corr_matrix)


# ======================================================
# 27. APOE by BCCI
# ======================================================

df_apoe = df_z[df_z["Genotype_group"].isin(["E2", "E3", "E4"])].copy()
df_apoe["Genotype_group"] = pd.Categorical(
    df_apoe["Genotype_group"],
    categories=["E3", "E2", "E4"],
    ordered=False
)

print("APOE analysis dataset size:", len(df_apoe))
print(df_apoe["Genotype_group"].value_counts())

if "Age" in df_apoe.columns:
    df_apoe["Age_z"] = safe_z(df_apoe["Age"])
if "Mass" in df_apoe.columns:
    df_apoe["Mass_z"] = safe_z(df_apoe["Mass"])


domains = {
    "Rate_Control": ("Heart_Rate", "BCCI_rate_control_cv_z"),
    "Systolic_Function": ("systolic_function_z", "BCCI_systolic_function_cv_z"),
    "Diastolic_Function": ("diastolic_function_z", "BCCI_diastolic_function_cv_z")
}

coef_results = []
interaction_results = []
model_stats = []

for domain_name, (yvar, bcci) in domains.items():
    if yvar not in df_apoe.columns or bcci not in df_apoe.columns:
        continue

    formula = f"""
    {yvar} ~ {bcci}
    + C(Genotype_group, Treatment(reference='E3'))
    + {bcci}:C(Genotype_group, Treatment(reference='E3'))
    + Age
    + Sex_Male
    + Exercise_Yes
    + Mass
    """

    model = fit_formula_if_possible(formula, df_apoe, required_n=15)
    if model is None:
        continue

    model_stats.append({
        "Domain": domain_name,
        "N": int(model.nobs),
        "R2": model.rsquared,
        "Adj_R2": model.rsquared_adj,
        "F_stat": model.fvalue,
        "F_p": model.f_pvalue
    })

    for term in model.params.index:
        coef_results.append({
            "Domain": domain_name,
            "Term": term,
            "Beta": model.params[term],
            "SE": model.bse[term],
            "t": model.tvalues[term],
            "P": model.pvalues[term]
        })

        if ":" in term:
            interaction_results.append({
                "Domain": domain_name,
                "Interaction": term,
                "Beta": model.params[term],
                "P": model.pvalues[term]
            })

pd.DataFrame(coef_results).to_csv(
    os.path.join(OUTDIR, "BCCI_APOE_coefficients_noKO.tsv"),
    sep="\t",
    index=False
)
pd.DataFrame(interaction_results).to_csv(
    os.path.join(OUTDIR, "BCCI_APOE_interactions_noKO.tsv"),
    sep="\t",
    index=False
)
pd.DataFrame(model_stats).to_csv(
    os.path.join(OUTDIR, "BCCI_APOE_model_stats_noKO.tsv"),
    sep="\t",
    index=False
)


# ======================================================
# 28. MASS × APOE ANALYSIS (NO KO)
# ======================================================

domains_mass = {
    "Rate_Control": "Heart_Rate",
    "Systolic_Function": "systolic_function_z",
    "Diastolic_Function": "diastolic_function_z"
}

mass_results = []
mass_interactions = []
mass_model_stats = []

for domain_name, yvar in domains_mass.items():
    if yvar not in df_apoe.columns or "Mass_z" not in df_apoe.columns:
        continue

    formula = f"""
    {yvar} ~ Mass_z
    + C(Genotype_group, Treatment(reference='E3'))
    + Mass_z:C(Genotype_group, Treatment(reference='E3'))
    + Age_z
    + Sex_Male
    + Exercise_Yes
    """

    model = fit_formula_if_possible(formula, df_apoe, required_n=15)
    if model is None:
        continue

    mass_model_stats.append({
        "Domain": domain_name,
        "N": int(model.nobs),
        "R2": model.rsquared,
        "Adj_R2": model.rsquared_adj,
        "F_stat": model.fvalue,
        "F_p": model.f_pvalue
    })

    for term in model.params.index:
        mass_results.append({
            "Domain": domain_name,
            "Term": term,
            "Beta": model.params[term],
            "SE": model.bse[term],
            "t": model.tvalues[term],
            "P": model.pvalues[term]
        })

        if ":" in term:
            mass_interactions.append({
                "Domain": domain_name,
                "Interaction": term,
                "Beta": model.params[term],
                "P": model.pvalues[term]
            })

pd.DataFrame(mass_results).to_csv(
    os.path.join(OUTDIR, "Mass_APOE_coefficients_noKO.tsv"),
    sep="\t",
    index=False
)
pd.DataFrame(mass_interactions).to_csv(
    os.path.join(OUTDIR, "Mass_APOE_interactions_noKO.tsv"),
    sep="\t",
    index=False
)
pd.DataFrame(mass_model_stats).to_csv(
    os.path.join(OUTDIR, "Mass_APOE_model_stats_noKO.tsv"),
    sep="\t",
    index=False
)


# ======================================================
# 29. BCCI residual models with Mass instead of Diet
# ======================================================

bcci_vars = [
    "BCCI_rate_control_resid_z",
    "BCCI_systolic_function_resid_z",
    "BCCI_diastolic_function_resid_z"
]
bcci_vars = [v for v in bcci_vars if v in df_apoe.columns]

for var in bcci_vars:
    formula = f"""
    {var} ~ Age_z + Sex_Male + Mass_z + Exercise_Yes
    + C(Genotype_group, Treatment(reference='E3'))
    + Mass_z:C(Genotype_group, Treatment(reference='E3'))
    + Exercise_Yes:C(Genotype_group, Treatment(reference='E3'))
    """

    model = fit_formula_if_possible(formula, df_apoe, required_n=15)
    if model is None:
        continue

    results_df = pd.DataFrame({
        "Variable": model.params.index,
        "Beta": model.params.values,
        "SE": model.bse.values,
        "t": model.tvalues.values,
        "p": model.pvalues.values
    })

    results_df.to_csv(os.path.join(OUTDIR, f"{var}_model_results.csv"), index=False)
    print(f"Saved {var}_model_results.csv")


# ======================================================
# 30. BCCI MODELS + EXPORT + INTERACTION PLOTS
# ======================================================

bcci_vars = [
    "BCCI_rate_control_resid_z",
    "BCCI_systolic_function_resid_z",
    "BCCI_diastolic_function_resid_z"
]
bcci_vars = [v for v in bcci_vars if v in df_apoe.columns]

formula_template = """
{var} ~ Age_z + Mass_z + Sex_Male + Exercise_Yes
+ C(Genotype_group, Treatment(reference='E3'))
+ Mass_z:C(Genotype_group, Treatment(reference='E3'))
+ Exercise_Yes:C(Genotype_group, Treatment(reference='E3'))
"""

all_results = []

for var in bcci_vars:
    formula = formula_template.format(var=var)
    model = fit_formula_if_possible(formula, df_apoe, required_n=15)
    if model is None:
        continue

    res_tmp = pd.DataFrame({
        "Outcome": var,
        "Predictor": model.params.index,
        "Beta": model.params.values,
        "SE": model.bse.values,
        "t": model.tvalues.values,
        "p": model.pvalues.values,
        "CI_low": model.conf_int()[0].values,
        "CI_high": model.conf_int()[1].values
    })

    res_tmp["Significant"] = res_tmp["p"] < 0.05
    all_results.append(res_tmp)

    if "Mass" in df_apoe.columns:
        plt.figure(figsize=(6, 5))
        sns.scatterplot(data=df_apoe, x="Mass", y=var, hue="Genotype_group")
        plt.title(f"{var} : APOE × Mass")
        plt.ylabel("Brain–Cardiac Coupling (z)")
        plt.xlabel("Mass")
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTDIR, f"{var}_APOE_Mass_interaction.png"),
            dpi=300
        )
        plt.close()

if len(all_results) > 0:
    results_df = pd.concat(all_results)
    results_df.to_csv(
        os.path.join(OUTDIR, "BCCI_all_models_full_results.csv"),
        index=False
    )

    summary = results_df.copy()
    summary["Beta_CI"] = summary.apply(
        lambda r: f"{r.Beta:.2f} ({r.CI_low:.2f},{r.CI_high:.2f})",
        axis=1
    )
    summary["p_fmt"] = summary["p"].apply(
        lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
    )
    summary_table = summary[["Outcome", "Predictor", "Beta_CI", "p_fmt", "Significant"]]
    summary_table.to_csv(
        os.path.join(OUTDIR, "BCCI_models_summary.csv"),
        index=False
    )

    sig = results_df[results_df["Significant"]]
    sig.to_csv(
        os.path.join(OUTDIR, "BCCI_significant_predictors.csv"),
        index=False
    )

print("BCCI analysis complete")
print(f"Results saved in {OUTDIR}")
print("\nPipeline completed successfully.")
