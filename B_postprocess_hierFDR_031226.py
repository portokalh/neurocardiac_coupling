#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Network–Cardiac Coupling Analysis Pipeline
Refactored full version
Author: Alex
"""

# ======================================================
# 1. IMPORTS
# ======================================================

import os
import json
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


# ======================================================
# 2. PATHS
# ======================================================

BASE = "/mnt/newStor/paros/paros_WORK/aashika"

IN_RESULTS = os.path.join(
    BASE,
    "results/network_cardiac_stats/network_cardiac_results.tsv"
)

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
    "results/network_cardiac_postproc031226"
)

os.makedirs(OUTDIR, exist_ok=True)

BRAINMAPS_ENABLED = True
THR_PERCENTILE = 80
DISPLAY_SMOOTH_SIGMA = 0.75   # display only
USE_SQRT_LOGP_WEIGHT = True


# ======================================================
# 3. FILE CHECKS
# ======================================================

for f in [IN_RESULTS, NETWORK_FILE, META_FILE, ICA_4D_FILE]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing file: {f}")

with open(os.path.join(OUTDIR, "run_config.json"), "w") as f:
    json.dump(
        {
            "IN_RESULTS": IN_RESULTS,
            "NETWORK_FILE": NETWORK_FILE,
            "META_FILE": META_FILE,
            "ICA_4D_FILE": ICA_4D_FILE,
            "OUTDIR": OUTDIR,
            "THR_PERCENTILE": THR_PERCENTILE,
            "DISPLAY_SMOOTH_SIGMA": DISPLAY_SMOOTH_SIGMA,
            "USE_SQRT_LOGP_WEIGHT": USE_SQRT_LOGP_WEIGHT,
        },
        f,
        indent=2,
    )


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


def save_heatmap(mat, title, filename, cmap="coolwarm", center=0, figsize=(12, 7),
                 annot=False, fmt=".2f", vmin=None, vmax=None):
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


def robust_clip(arr, pct=99):
    vmax = np.percentile(np.abs(arr[np.isfinite(arr)]), pct)
    if vmax == 0 or np.isnan(vmax):
        return arr, -1, 1
    arr2 = np.clip(arr, -vmax, vmax)
    return arr2, -vmax, vmax


def build_weighted_map(weights, ica_data):
    return np.tensordot(ica_data, weights, axes=([3], [0]))


def save_nifti_pair(weighted_map, affine, out_prefix, thr_percentile=80):
    nib.save(
        nib.Nifti1Image(weighted_map, affine),
        out_prefix + ".nii.gz"
    )

    thr = np.percentile(np.abs(weighted_map), thr_percentile)
    thr_map = weighted_map.copy()
    thr_map[np.abs(thr_map) < thr] = 0

    nib.save(
        nib.Nifti1Image(thr_map, affine),
        out_prefix + f"_thr{thr_percentile}.nii.gz"
    )

    return thr_map


def save_display_png(weighted_map, out_png, smooth_sigma=0.75, clip_pct=99, title=None):
    # display-only processing
    disp = gaussian_filter(weighted_map, sigma=smooth_sigma)
    disp, vmin, vmax = robust_clip(disp, pct=clip_pct)

    # use max projection across slices for quick QC / figure draft
    # sagittal-like summary
    proj = np.max(np.abs(disp), axis=0)

    plt.figure(figsize=(7, 5))
    plt.imshow(proj.T, origin="lower", cmap="magma", aspect="auto")
    plt.colorbar(shrink=0.8)
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def results_to_weights(results_df, network_cols, mode="beta_p_sqrt"):
    weights = np.zeros(len(network_cols), dtype=float)

    for _, r in results_df.iterrows():
        net = r["Network"]
        if net not in network_cols:
            continue

        idx = network_cols.index(net)
        beta = r["Beta"]

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
        return None, None

    weights = weights / np.max(np.abs(weights))
    weighted_map = build_weighted_map(weights, ica_data)
    thr_map = save_nifti_pair(weighted_map, affine, out_prefix, thr_percentile=thr_percentile)

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

    return weighted_map, thr_map


def fit_simple_ols(x, y):
    X = sm.add_constant(x)
    model = sm.OLS(y, X, missing="drop").fit()
    return model


def safe_filename(s):
    s = str(s)
    for ch in [" ", "/", "\\", ":", ";", ",", "(", ")", "[", "]", "{", "}"]:
        s = s.replace(ch, "_")
    return s


# ======================================================
# 5. LOAD DATA
# ======================================================

print("Loading regression results")
res = pd.read_csv(IN_RESULTS, sep="\t")

print("Loading metadata")
meta = pd.read_csv(META_FILE)

print("Loading network amplitudes")
nets = pd.read_csv(NETWORK_FILE, sep="\t")

meta["Arunno"] = meta["Arunno"].astype(str).str.strip()
nets["Arunno"] = nets["Arunno"].astype(str).str.strip()


# ======================================================
# 6. CARDIAC DOMAIN DEFINITIONS
# ======================================================

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


# ======================================================
# 7. ASSIGN METRIC GROUPS
# ======================================================

res["Group"] = res["CardiacMetric"].map(metric_to_group)

if res["Group"].isna().any():
    missing_metrics = res.loc[res["Group"].isna(), "CardiacMetric"].unique().tolist()
    raise ValueError(f"Some cardiac metrics not mapped: {missing_metrics}")


# ======================================================
# 8. HIERARCHICAL FDR
# ======================================================

print("Running hierarchical FDR")

group_p = (
    res.groupby("Group")["P"]
    .apply(lambda s: simes_p(s.values))
    .reset_index()
    .rename(columns={"P": "Group_P"})
)

group_p["Group_FDR"] = multipletests(group_p["Group_P"], method="fdr_bh")[1]

res = res.merge(group_p, on="Group")

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


# ======================================================
# 9. HEATMAPS
# ======================================================

mlogp = res.pivot_table(index="Network", columns="CardiacMetric", values="P")
mlogp = -np.log10(mlogp)

beta = res.pivot_table(index="Network", columns="CardiacMetric", values="Beta")

save_heatmap(
    mlogp,
    "Network–Cardiac (-log10 p)",
    "heatmap_logp.png",
    cmap="viridis",
    center=None,
    figsize=(12, 7),
)

save_heatmap(
    beta,
    "Network–Cardiac beta",
    "heatmap_beta.png",
    cmap="coolwarm",
    center=0,
    figsize=(12, 7),
)


# ======================================================
# 10. NETWORK CLUSTERING
# ======================================================

X_cluster = StandardScaler().fit_transform(beta.fillna(0))

Z = linkage(X_cluster, method="ward")

plt.figure(figsize=(10, 6))
dendrogram(Z, labels=beta.index.tolist(), leaf_rotation=90)
plt.title("Network clustering by cardiac profile")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "network_dendrogram.png"), dpi=300)
plt.close()


# ======================================================
# 11. COMPUTE BCCI
# ======================================================

print("Computing BCCI")

network_cols = [c for c in nets.columns if c.startswith("Amp_")]

if len(network_cols) == 0:
    raise ValueError("No network columns found starting with 'Amp_'")

nets_z = nets.copy()
nets_z[network_cols] = StandardScaler().fit_transform(nets_z[network_cols])

weights = beta.abs().mean(axis=1)
weights = weights / weights.sum()

weights.index = [
    x if str(x).startswith("Amp_") else f"Amp_{x}"
    for x in weights.index
]

weight_vec = np.array([weights.get(c, 0) for c in network_cols])

if np.all(weight_vec == 0):
    raise ValueError("All BCCI weights are zero. Check network naming consistency.")

nets_z["BCCI"] = np.dot(nets_z[network_cols].values, weight_vec)

nets_z[["Arunno", "BCCI"]].to_csv(
    os.path.join(OUTDIR, "brain_cardiac_index.tsv"),
    sep="\t",
    index=False
)


# ======================================================
# 12. MERGE METADATA
# ======================================================

df = pd.merge(meta, nets_z[["Arunno", "BCCI"]], on="Arunno")
print("Subjects:", len(df))

df["BCCI_z"] = safe_z(df["BCCI"])


# ======================================================
# 13. OPTIONAL BCCI ORTHOGONALIZATION
# ======================================================

covars_for_bcci = ["Age", "Sex_Male", "Mass"]
covars_for_bcci = [c for c in covars_for_bcci if c in df.columns]

if len(covars_for_bcci) > 0:
    X_cov = df[covars_for_bcci].apply(pd.to_numeric, errors="coerce")
    X_cov = sm.add_constant(X_cov)
    model_bcci_cov = sm.OLS(df["BCCI_z"], X_cov, missing="drop").fit()
    df["BCCI_resid"] = model_bcci_cov.resid
    df["BCCI_resid_z"] = safe_z(df["BCCI_resid"])
    print("BCCI orthogonalized from:", covars_for_bcci)
else:
    df["BCCI_resid_z"] = df["BCCI_z"]


# ======================================================
# 14. PCA CARDIAC DOMAINS
# ======================================================

print("Creating PCA cardiac domains")

for domain, metrics in CARDIAC_GROUPS.items():
    existing = [m for m in metrics if m in df.columns]

    if len(existing) == 0:
        print(domain, ": no existing metrics")
        continue

    tmp = df[existing].apply(pd.to_numeric, errors="coerce")
    tmp_z = tmp.apply(safe_z)

    valid = tmp_z.dropna(how="all")

    if len(existing) == 1:
        df[domain + "_z"] = safe_z(tmp_z.iloc[:, 0])
        print(domain, ": single metric domain")
        continue

    if len(valid) < 3:
        print(domain, ": not enough data for PCA")
        continue

    # impute within-domain by column mean after z-scoring
    valid_imp = valid.copy()
    for c in valid_imp.columns:
        valid_imp[c] = valid_imp[c].fillna(valid_imp[c].mean())

    pca = PCA(n_components=1)
    comp = pca.fit_transform(valid_imp)

    df.loc[valid_imp.index, domain + "_z"] = comp[:, 0]
    df[domain + "_z"] = safe_z(df[domain + "_z"])

    # Make domain direction interpretable
    if domain == "systolic_function" and "Ejection_Fraction" in existing:
        corr = np.corrcoef(
            df.loc[valid_imp.index, domain + "_z"],
            df.loc[valid_imp.index, "Ejection_Fraction"]
        )[0, 1]
        if np.isfinite(corr) and corr < 0:
            df[domain + "_z"] = -df[domain + "_z"]

    print(
        f"{domain}: PCA variance explained =",
        round(pca.explained_variance_ratio_[0], 3)
    )

    loadings = pd.Series(pca.components_[0], index=existing)
    loadings.to_csv(
        os.path.join(OUTDIR, f"{domain}_pca_loadings.tsv"),
        sep="\t",
        header=False
    )


# ======================================================
# 15. DIAGNOSTIC CORRELATIONS
# ======================================================

for d in ["rate_control_z", "systolic_function_z", "diastolic_function_z"]:
    if d not in df.columns:
        print("Missing:", d)
        continue
    r = np.corrcoef(df["BCCI_z"], df[d])[0, 1]
    print("BCCI vs", d, "r =", round(r, 3))


# ======================================================
# 16. BCCI vs HEART RATE
# ======================================================

tmp = df[["BCCI_z", "Heart_Rate"]].copy()
tmp["Heart_Rate"] = pd.to_numeric(tmp["Heart_Rate"], errors="coerce")
tmp = tmp.dropna()

model_hr = smf.ols("Heart_Rate ~ BCCI_z", data=tmp).fit()

r2 = model_hr.rsquared
p = model_hr.pvalues["BCCI_z"]

xfit = np.linspace(tmp["BCCI_z"].min(), tmp["BCCI_z"].max(), 200)
pred = model_hr.get_prediction(
    pd.DataFrame({"BCCI_z": xfit})
).summary_frame()

plt.figure(figsize=(6,5))

plt.scatter(tmp["BCCI_z"], tmp["Heart_Rate"], s=70)

plt.plot(xfit, pred["mean"], linewidth=2)

plt.fill_between(
    xfit,
    pred["mean_ci_lower"],
    pred["mean_ci_upper"],
    alpha=0.2
)

plt.xlabel("BCCI (z)")
plt.ylabel("Heart Rate")

plt.text(
    0.05,
    0.95,
    f"R² = {r2:.2f}\np = {p:.3g}",
    transform=plt.gca().transAxes,
    verticalalignment="top"
)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTDIR, "BCCI_vs_HeartRate.png"),
    dpi=300
)

plt.close()


# ======================================================
# 17. SPEARMAN-STYLE RANK PLOT
# ======================================================

tmp = df[["BCCI_z","Heart_Rate"]].copy()
tmp["Heart_Rate"] = pd.to_numeric(tmp["Heart_Rate"], errors="coerce")
tmp = tmp.dropna()

plt.figure(figsize=(6,5))

plt.scatter(
    tmp["BCCI_z"].rank(),
    tmp["Heart_Rate"].rank(),
    s=70
)

plt.xlabel("Ranked BCCI")
plt.ylabel("Ranked Heart Rate")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTDIR,"spearman_rank_plot.png"),
    dpi=300
)

plt.close()


# ======================================================
# 18. PLS MODEL
# ======================================================

X_pls = nets_z[network_cols].values
y_pls = pd.to_numeric(df["Heart_Rate"], errors="coerce").values.reshape(-1, 1)

valid_pls = np.isfinite(y_pls).ravel()
X_pls = X_pls[valid_pls]
y_pls = y_pls[valid_pls]

pls = PLSRegression(n_components=1)
pls.fit(X_pls, y_pls)
pred_pls = pls.predict(X_pls)

print("PLS R2:", r2_score(y_pls, pred_pls))


# ======================================================
# 19. CROSS-VALIDATED PLS
# ======================================================

kf = KFold(n_splits=5, shuffle=True, random_state=1)

pred_cv = np.zeros(len(y_pls))

for train, test in kf.split(X_pls):
    pls_cv = PLSRegression(n_components=1)
    pls_cv.fit(X_pls[train], y_pls[train])
    pred_cv[test] = pls_cv.predict(X_pls[test]).ravel()

print("Cross-validated R2:", r2_score(y_pls, pred_cv))


# ======================================================
# 20. LOAD ICA ONCE
# ======================================================

img = nib.load(ICA_4D_FILE)
ica_data = img.get_fdata()
affine = img.affine
n_components = ica_data.shape[3]

if len(network_cols) != n_components:
    print(
        f"Warning: network columns ({len(network_cols)}) != ICA components ({n_components})"
    )


# ======================================================
# 21. INDIVIDUAL METRIC BRAIN MAPS
# ======================================================

if BRAINMAPS_ENABLED:
    print("Creating individual metric brain maps")

    for metric in res["CardiacMetric"].unique():
        sub = res[res["CardiacMetric"] == metric]

        if len(sub) == 0:
            continue

        out_prefix = os.path.join(
            OUTDIR,
            f"cardiac_map_{safe_filename(metric)}"
        )

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
            png_title=f"{metric}"
        )


# ======================================================
# 22. DOMAIN BRAIN MAPS
# ======================================================

if BRAINMAPS_ENABLED:
    print("Creating cardiac domain brain maps")

    for domain, metrics in CARDIAC_GROUPS.items():
        sub = res[res["CardiacMetric"].isin(metrics)]

        if len(sub) == 0:
            continue

        out_prefix = os.path.join(
            OUTDIR,
            f"cardiac_domain_map_{safe_filename(domain)}"
        )

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
            png_title=f"{domain}"
        )


# ======================================================
# 23. PLS BRAIN–HEART CIRCUIT MAP
# ======================================================

if BRAINMAPS_ENABLED:
    print("Creating PLS brain–heart circuit map")

    pls_weights = pls.x_weights_.flatten()

    if len(pls_weights) > n_components:
        pls_weights = pls_weights[:n_components]

    pls_df = pd.DataFrame({
        "Network": network_cols[:len(pls_weights)],
        "Beta": pls_weights,
        "P": np.repeat(0.01, len(pls_weights))  # placeholder; not used in beta mode
    })

    out_prefix = os.path.join(OUTDIR, "brain_heart_circuit_map_PLS")

    save_circuit_maps(
        pls_df,
        network_cols[:len(pls_weights)],
        ica_data[..., :len(pls_weights)] if ica_data.shape[3] != len(pls_weights) else ica_data,
        affine,
        out_prefix,
        weight_mode="beta",
        thr_percentile=THR_PERCENTILE,
        save_png=True,
        png_title="PLS brain-heart circuit"
    )


# ======================================================
# 24. NETWORK IMPORTANCE FIGURES
# ======================================================

print("Creating network importance figures")

metric_matrix = res.pivot_table(
    index="Network",
    columns="CardiacMetric",
    values="Beta"
)

metric_matrix = metric_matrix.reindex(network_cols)

domain_matrix = pd.DataFrame(index=network_cols)

for domain, metrics in CARDIAC_GROUPS.items():
    sub = res[res["CardiacMetric"].isin(metrics)]

    if len(sub) == 0:
        continue

    net_beta = sub.groupby("Network")["Beta"].mean()
    vals = [net_beta.get(n, 0) for n in network_cols]
    domain_matrix[domain] = vals

domain_matrix = domain_matrix.fillna(0)

def scale_matrix(mat):
    scaled = mat.copy()
    for col in scaled.columns:
        sd = scaled[col].std()
        if sd > 0:
            scaled[col] = (scaled[col] - scaled[col].mean()) / sd
    return scaled

metric_scaled = scale_matrix(metric_matrix)
domain_scaled = scale_matrix(domain_matrix)

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
# 25. CLEAN / STANDARDIZE VARIABLES FOR MODERATION MODELS
# ======================================================

print("Preparing variables for moderation models")

mods = [
    "Sex_Male",
    "Age",
    "Exercise_Yes",
    "Mass",
    "E4_Genotype",
    "HN"
]
mods = [m for m in mods if m in df.columns]
print("Moderators found:", mods)

for col in mods + ["BCCI"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

if "Age" in df.columns:
    df["Age_z"] = safe_z(df["Age"])

if "Mass" in df.columns:
    df["Mass_z"] = safe_z(df["Mass"])

all_metrics = [m for m in metric_to_group.keys() if m in df.columns]
print("Cardiac metrics found:", all_metrics)

for m in all_metrics:
    df[m + "_z"] = safe_z(df[m])
    df[m + "_rin"] = rank_inverse_normal(df[m])

# also keep domain z vars already created
domain_vars = [d + "_z" for d in CARDIAC_GROUPS.keys() if d + "_z" in df.columns]


# ======================================================
# 26. BCCI vs ALL CARDIAC METRICS
# ======================================================

print("Running BCCI vs cardiac metrics")

bcci_metric_results = []

predictor = "BCCI_resid_z" if "BCCI_resid_z" in df.columns else "BCCI_z"

for metric in all_metrics:
    tmp = df[[predictor, metric]].copy()
    tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
    tmp = tmp.dropna()

    if len(tmp) < 15:
        continue

    model = smf.ols(f"{metric} ~ {predictor}", data=tmp).fit()

    bcci_metric_results.append({
        "Metric": metric,
        "N": len(tmp),
        "Predictor": predictor,
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

print(bcci_metric_results)


# ======================================================
# 27. MODERATION MODELS
# ======================================================

print("Running moderation models")

moderation_results = []

mods_to_test = []
if "Age_z" in df.columns:
    mods_to_test.append("Age_z")
for c in ["Sex_Male", "Exercise_Yes", "E4_Genotype", "HN"]:
    if c in df.columns:
        mods_to_test.append(c)

for metric in all_metrics:
    metric_z = metric + "_z"
    if metric_z not in df.columns:
        continue

    for mod in mods_to_test:
        cols = [predictor, metric_z, mod]
        tmp = df[cols].dropna().copy()

        if len(tmp) < 20:
            continue

        formula = f"{metric_z} ~ {predictor} * {mod}"
        model = smf.ols(formula, data=tmp).fit()

        interaction_term = f"{predictor}:{mod}"
        if interaction_term not in model.params.index:
            alt_term = f"{mod}:{predictor}"
            interaction_term = alt_term if alt_term in model.params.index else interaction_term

        moderation_results.append({
            "Metric": metric,
            "Moderator": mod,
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

print(moderation_results.head(20))


# ======================================================
# 28. PLOT SIGNIFICANT INTERACTIONS
# ======================================================

print("Plotting significant interactions")

if len(moderation_results) > 0:
    sig_int = moderation_results[moderation_results["FDR_Interaction"] < 0.10].copy()
else:
    sig_int = pd.DataFrame()

print("Significant interactions:", len(sig_int))

for _, row in sig_int.iterrows():
    metric = row["Metric"]
    mod = row["Moderator"]
    metric_z = metric + "_z"

    cols = [predictor, metric_z, mod]
    tmp = df[cols].dropna().copy()

    if len(tmp) < 20:
        continue

    plt.figure(figsize=(6, 5))

    unique_vals = sorted(tmp[mod].dropna().unique())

    if len(unique_vals) <= 2:
        for val in unique_vals:
            sub = tmp[tmp[mod] == val]

            plt.scatter(
                sub[predictor],
                sub[metric_z],
                s=70,
                alpha=0.8,
                label=f"{mod}={val}"
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

            plt.scatter(
                sub[predictor],
                sub[metric_z],
                s=70,
                alpha=0.8,
                label=f"{mod} {grp}"
            )

            fit = smf.ols(f"{metric_z} ~ {predictor}", data=sub).fit()
            xfit = np.linspace(sub[predictor].min(), sub[predictor].max(), 100)
            yfit = fit.params["Intercept"] + fit.params[predictor] * xfit
            plt.plot(xfit, yfit, linewidth=2)
        plt.legend()

    plt.xlabel(f"{predictor}")
    plt.ylabel(f"{metric} (z)")
    plt.title(f"{metric}: {predictor} × {mod}")
    plt.tight_layout()

    outfile = os.path.join(
        OUTDIR,
        f"interaction_{safe_filename(metric)}_by_{safe_filename(mod)}.png"
    )
    plt.savefig(outfile, dpi=300)
    plt.close()


# ======================================================
# 29. TARGETED INTERACTION MODELS
# ======================================================

print("Running targeted interaction models")

target_models = [
    ("systolic_function_z", "E4_Genotype"),
    ("diastolic_function_z", "Sex_Male"),
    ("rate_control_z", "Exercise_Yes")
]

target_results = []

for metric, mod in target_models:
    if metric not in df.columns:
        print("Missing metric:", metric)
        continue
    if mod not in df.columns:
        print("Missing moderator:", mod)
        continue

    cols = [predictor, metric, mod]
    tmp = df[cols].dropna()

    if len(tmp) < 20:
        continue

    model = smf.ols(f"{metric} ~ {predictor} * {mod}", data=tmp).fit()

    interaction_term = f"{predictor}:{mod}"
    if interaction_term not in model.params.index:
        alt_term = f"{mod}:{predictor}"
        interaction_term = alt_term if alt_term in model.params.index else interaction_term

    target_results.append({
        "Metric": metric,
        "Moderator": mod,
        "N": len(tmp),
        "Beta_BCCI": model.params.get(predictor, np.nan),
        "P_BCCI": model.pvalues.get(predictor, np.nan),
        "Beta_Interaction": model.params.get(interaction_term, np.nan),
        "P_Interaction": model.pvalues.get(interaction_term, np.nan),
        "R2": model.rsquared
    })

    print("\n", metric, "~", predictor, "*", mod)
    print(model.summary())

target_results = pd.DataFrame(target_results)

target_results.to_csv(
    os.path.join(OUTDIR, "targeted_coupling_models.tsv"),
    sep="\t",
    index=False
)


# ======================================================
# 30. GENOTYPE-SPECIFIC BRAIN–HEART CIRCUITS
# ======================================================

if BRAINMAPS_ENABLED and "E4_Genotype" in df.columns:
    print("Creating genotype-specific brain–heart circuit maps")

    genotype_groups = {
        "NonE4": df[df["E4_Genotype"] == 0],
        "E4": df[df["E4_Genotype"] == 1]
    }

    maps = {}

    for gname, gdf in genotype_groups.items():
        print("\nProcessing genotype:", gname, "N =", len(gdf))

        results_rows = []

        for net in network_cols:
            tmp = pd.merge(
                gdf[["Arunno", "Heart_Rate"]],
                nets[["Arunno", net]],
                on="Arunno"
            ).dropna()

            if len(tmp) < 10:
                continue

            model = sm.OLS(tmp["Heart_Rate"], sm.add_constant(tmp[net])).fit()

            results_rows.append({
                "Network": net,
                "Beta": model.params.iloc[1],
                "P": max(model.pvalues.iloc[1], 1e-10)
            })

        if len(results_rows) == 0:
            continue

        g_res = pd.DataFrame(results_rows)

        out_prefix = os.path.join(
            OUTDIR,
            f"brain_heart_map_{gname}"
        )

        weight_mode = "beta_p_sqrt" if USE_SQRT_LOGP_WEIGHT else "beta_p"

        weighted_map, _ = save_circuit_maps(
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
        save_display_png(diff_map, diff_prefix + ".png", smooth_sigma=DISPLAY_SMOOTH_SIGMA,
                         clip_pct=99, title="E4 - NonE4")
        save_display_png(
            np.where(np.abs(diff_map) >= np.percentile(np.abs(diff_map), THR_PERCENTILE), diff_map, 0),
            diff_prefix + f"_thr{THR_PERCENTILE}.png",
            smooth_sigma=DISPLAY_SMOOTH_SIGMA,
            clip_pct=99,
            title=f"E4 - NonE4 thr{THR_PERCENTILE}"
        )


# ======================================================
# 31. GENOTYPE-SPECIFIC MAPS PER CARDIAC METRIC
# ======================================================

if BRAINMAPS_ENABLED and "E4_Genotype" in df.columns:
    print("Creating genotype-specific maps for each cardiac metric")

    genotype_groups = {
        "NonE4": df[df["E4_Genotype"] == 0],
        "E4": df[df["E4_Genotype"] == 1]
    }

    cardiac_metrics = res["CardiacMetric"].unique()

    for metric in cardiac_metrics:
        print("\nMetric:", metric)

        maps = {}

        for gname, gdf in genotype_groups.items():
            print("Processing genotype:", gname, "N =", len(gdf))

            results_rows = []

            for net in network_cols:
                tmp = pd.merge(
                    gdf[["Arunno", metric]],
                    nets[["Arunno", net]],
                    on="Arunno"
                ).dropna()

                if len(tmp) < 10:
                    continue

                model = sm.OLS(tmp[metric], sm.add_constant(tmp[net])).fit()

                results_rows.append({
                    "Network": net,
                    "Beta": model.params.iloc[1],
                    "P": max(model.pvalues.iloc[1], 1e-10)
                })

            if len(results_rows) == 0:
                continue

            g_res = pd.DataFrame(results_rows)

            out_prefix = os.path.join(
                OUTDIR,
                f"brain_heart_{safe_filename(metric)}_{gname}"
            )

            weight_mode = "beta_p_sqrt" if USE_SQRT_LOGP_WEIGHT else "beta_p"

            weighted_map, _ = save_circuit_maps(
                g_res,
                network_cols,
                ica_data,
                affine,
                out_prefix,
                weight_mode=weight_mode,
                thr_percentile=THR_PERCENTILE,
                save_png=True,
                png_title=f"{metric} {gname}"
            )

            maps[gname] = weighted_map

        if "E4" in maps and "NonE4" in maps:
            diff_map = maps["E4"] - maps["NonE4"]
            diff_prefix = os.path.join(
                OUTDIR,
                f"brain_heart_{safe_filename(metric)}_E4_minus_NonE4"
            )
            save_nifti_pair(diff_map, affine, diff_prefix, thr_percentile=THR_PERCENTILE)
            save_display_png(
                diff_map,
                diff_prefix + ".png",
                smooth_sigma=DISPLAY_SMOOTH_SIGMA,
                clip_pct=99,
                title=f"{metric} E4 - NonE4"
            )


# ======================================================
# 32. FIGURE 4 – BRAIN–CARDIAC COUPLING
# ======================================================

print("\n==============================")
print("Running Figure 4 analysis")
print("==============================")

df_fig4 = df.copy()

if "Genotype" in df_fig4.columns:
    geno_map = {
        "E22": "E2",
        "E2HN": "E2",
        "E33": "E3",
        "E3HN": "E3",
        "E44": "E4",
        "E4HN": "E4",
        "KO": "KO"
    }
    df_fig4["Genotype_group"] = df_fig4["Genotype"].map(geno_map)
    df_fig4["Genotype_group"] = df_fig4["Genotype_group"].astype("category")

if "Sex" in df_fig4.columns:
    df_fig4["Sex"] = df_fig4["Sex"].astype("category")

# Panel A
if all(c in df_fig4.columns for c in ["BCCI_z", "Age", "Heart_Rate"]) and "Genotype_group" in df_fig4.columns and "Sex" in df_fig4.columns:
    X = df_fig4[["BCCI_z", "Age", "Sex", "Genotype_group"]]
    X = pd.get_dummies(X, drop_first=True)
    X = sm.add_constant(X).astype(float)

    y = df_fig4["Heart_Rate"].astype(float)

    model = sm.OLS(y, X, missing="drop").fit()

    beta_bcci = model.params["BCCI_z"]
    pval = model.pvalues["BCCI_z"]
    r2 = model.rsquared
    ci_low, ci_high = model.conf_int().loc["BCCI_z"]

    xfit = np.linspace(df_fig4["BCCI_z"].min(), df_fig4["BCCI_z"].max(), 200)

    Xpred = pd.DataFrame(0, index=np.arange(len(xfit)), columns=X.columns)
    Xpred["const"] = 1
    Xpred["BCCI_z"] = xfit
    if "Age" in Xpred.columns:
        Xpred["Age"] = df_fig4["Age"].mean()
    Xpred = Xpred.astype(float)

    pred = model.get_prediction(Xpred).summary_frame()

    plt.figure(figsize=(6, 5))
    plt.scatter(df_fig4["BCCI_z"], y, s=60)
    plt.plot(xfit, pred["mean"], linewidth=2)
    plt.fill_between(
        xfit,
        pred["mean_ci_lower"],
        pred["mean_ci_upper"],
        alpha=0.25
    )
    plt.xlabel("Brain–Cardiac Coupling Index (z)")
    plt.ylabel("Heart Rate")
    plt.title("Brain–Heart Coupling")
    plt.text(
        0.05,
        0.95,
        f"β = {beta_bcci:.2f}\n95% CI [{ci_low:.2f},{ci_high:.2f}]\np = {pval:.3g}\nR² = {r2:.2f}",
        transform=plt.gca().transAxes,
        verticalalignment="top"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "Figure4A_BCCI_vs_HeartRate.png"), dpi=300)
    plt.close()

# Panel B
if "Genotype_group" in df_fig4.columns and all(c in df_fig4.columns for c in ["BCCI_z", "Age", "Sex", "Heart_Rate"]):
    X1 = df_fig4[["BCCI_z", "Age", "Sex", "Genotype_group"]]
    X1 = pd.get_dummies(X1, drop_first=True)
    X1 = sm.add_constant(X1).astype(float)
    m1 = sm.OLS(df_fig4["Heart_Rate"], X1, missing="drop").fit()

    df_noKO = df_fig4[df_fig4["Genotype_group"] != "KO"].copy()

    X2 = df_noKO[["BCCI_z", "Age", "Sex", "Genotype_group"]]
    X2 = pd.get_dummies(X2, drop_first=True)
    X2 = sm.add_constant(X2).astype(float)
    m2 = sm.OLS(df_noKO["Heart_Rate"], X2, missing="drop").fit()

    coef_vals = [m1.params["BCCI_z"], m2.params["BCCI_z"]]
    p_vals = [m1.pvalues["BCCI_z"], m2.pvalues["BCCI_z"]]
    r2_vals = [m1.rsquared, m2.rsquared]
    ci1 = m1.conf_int().loc["BCCI_z"]
    ci2 = m2.conf_int().loc["BCCI_z"]
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

# Panel C
if "Genotype_group" in df_fig4.columns and "Sex" in df_fig4.columns:
    domain_beta = {}
    domain_p = {}
    domain_r2 = {}
    domain_ci = {}

    for domain in CARDIAC_GROUPS.keys():
        dom_var = domain + "_z"
        if dom_var not in df_fig4.columns:
            continue

        X = df_fig4[["BCCI_z", "Age", "Sex", "Genotype_group"]]
        X = pd.get_dummies(X, drop_first=True)
        X = sm.add_constant(X).astype(float)

        y = df_fig4[dom_var].astype(float)

        model = sm.OLS(y, X, missing="drop").fit()

        domain_beta[domain] = model.params["BCCI_z"]
        domain_p[domain] = model.pvalues["BCCI_z"]
        domain_r2[domain] = model.rsquared
        domain_ci[domain] = model.conf_int().loc["BCCI_z"]

    if len(domain_beta) > 0:
        plt.figure(figsize=(6, 4))
        domains = list(domain_beta.keys())
        betas = [domain_beta[d] for d in domains]
        errors = [betas[i] - domain_ci[domains[i]][0] for i in range(len(domains))]

        bars = plt.bar(domains, betas, yerr=errors, capsize=6)
        plt.ylabel("BCCI effect")
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

        plt.xticks(rotation=25)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "Figure4C_domain_effects.png"), dpi=300)
        plt.close()

print("\nFigure 4 completed")
print("\nPipeline completed successfully.")