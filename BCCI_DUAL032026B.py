#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BCCI PIPELINE — DUAL (UNADJUSTED vs ADJUSTED)

Author: cleaned + modular
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr, norm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from matplotlib.lines import Line2D
# =============================================================================
# PATHS
# =============================================================================

BASE = "/mnt/newStor/paros/paros_WORK/aashika"

META_FILE = os.path.join(BASE, "data/metadata/cardiac_design_updated3.csv")
NETWORK_FILE = os.path.join(BASE, "results/march16_2026/network_amplitudes_matrix_COPY.tsv")

OUTDIR = os.path.join(BASE, "results/BCCI_DUAL")
os.makedirs(OUTDIR, exist_ok=True)

# =============================================================================
# SETTINGS
# =============================================================================

MIN_N = 20

COVARS_BASE = ["Age", "Sex_Male"]
COVARS_FULL = ["Age", "Sex_Male", "Mass", "Exercise_Yes"]


NETWORK_LABELS = {
    "Amp_Net01": "Midline Autonomic Axis",
    "Amp_Net02": "Anterior Medial (mPFC–ACC)",
    "Amp_Net03": "Posterior Association",
    "Amp_Net04": "Sensorimotor–Insular",
    "Amp_Net05": "Primary Somatomotor",
    "Amp_Net06": "Ventral Temporal",
    "Amp_Net07": "Olfactory–Basal",
    "Amp_Net08": "Temporal–Insular",
    "Amp_Net09": "Thalamo–Brainstem",
    "Amp_Net10": "Tecto–Cerebellar",
    "Amp_Net11": "Cerebellar Crus",
    "Amp_Net12": "Olfactory Bulb"
}
# =============================================================================
# CARDIAC DOMAINS
# =============================================================================

CARDIAC_GROUPS = {
    "rate_control": ["Heart_Rate"],
    "pumping": ["Stroke_Volume","Cardiac_Output","Ejection_Fraction"],
    "systolic_function": ["Systolic_LV_Volume","Systolic_RV","Systolic_LA","Systolic_RA","Systolic_Myo"],
    "diastolic_function": ["Diastolic_LV_Volume","Diastolic_RV","Diastolic_LA","Diastolic_RA","Diastolic_Myo"]
}


# =============================================================================
# DOMAINS (used across Figure 1 + Figure 4)
# =============================================================================

DOMAINS = [
    "rate_control",
    "pumping",
    "systolic_function",
    "diastolic_function"
]
# =============================================================================
# HELPERS
# =============================================================================

def clean_id(x):
    return str(x).replace(".0","").strip()

def safe_mean_z(df):
    return StandardScaler().fit_transform(df).mean(axis=1)

def residualize(df, col, covars):
    tmp = df[[col] + covars].dropna()
    if len(tmp) < MIN_N:
        return pd.Series(index=df.index), None
    model = sm.OLS(tmp[col], sm.add_constant(tmp[covars])).fit()
    resid = pd.Series(index=df.index)
    resid.loc[tmp.index] = model.resid
    return resid, model

def build_weights(X, y):
    return np.array([pearsonr(X[:,i], y)[0] for i in range(X.shape[1])])

# =============================================================================
# BUILD WEIGHT MATRIX
# =============================================================================

def build_weight_matrix(df_weights):

    mat = df_weights.pivot(
        index="network",
        columns="domain",
        values="weight"
    )

    mat = mat.loc[[n for n in network_cols if n in mat.index]]

    # 🔥 APPLY LABELS HERE
    mat.index = [NETWORK_LABELS.get(n, n) for n in mat.index]

    return mat

# =============================================================================
# ROBUSTNESS CHECK — OUTLIERS / LEVERAGE (ENHANCED)
# =============================================================================

def check_bcci_robustness(df_model, suffix, tag):

    print(f"\nRunning robustness checks ({tag})")

    rows = []

    for group in CARDIAC_GROUPS:

        x = f"{group}_BCCI{suffix}"
        y = f"{group}_target{suffix}"

        if x not in df_model or y not in df_model:
            continue

        tmp = df_model[[x, y]].dropna()

        if len(tmp) < MIN_N:
            continue

        rho_full, _ = spearmanr(tmp[x], tmp[y])

        # remove top 3
        tmp_hi = tmp.sort_values(x).iloc[:-3]
        rho_hi, _ = spearmanr(tmp_hi[x], tmp_hi[y])

        # remove bottom 3
        tmp_lo = tmp.sort_values(x).iloc[3:]
        rho_lo, _ = spearmanr(tmp_lo[x], tmp_lo[y])

        # changes
        delta_hi = rho_full - rho_hi
        delta_lo = rho_full - rho_lo

        pct_change_hi = delta_hi / rho_full if rho_full != 0 else np.nan
        pct_change_lo = delta_lo / rho_full if rho_full != 0 else np.nan

        unstable = (abs(pct_change_hi) > 0.3) or (abs(pct_change_lo) > 0.3)

        rows.append({
            "domain": group,
            "rho_full": rho_full,
            "rho_no_high3": rho_hi,
            "rho_no_low3": rho_lo,
            "delta_high3": delta_hi,
            "delta_low3": delta_lo,
            "pct_change_high3": pct_change_hi,
            "pct_change_low3": pct_change_lo,
            "unstable_flag": unstable,
            "N": len(tmp)
        })

        print(f"{group}: full={rho_full:.3f}, Δhi={delta_hi:.3f}, Δlo={delta_lo:.3f}, unstable={unstable}")

    df_out = pd.DataFrame(rows)

    # save
    out_path = os.path.join(OUTDIR, f"robustness_{tag}.tsv")
    df_out.to_csv(out_path, sep="\t", index=False)

    print(f"Saved: {out_path}")

    return df_out
# =============================================================================
# LOAD DATA
# =============================================================================

print("Loading data")

meta = pd.read_csv(META_FILE)
nets = pd.read_csv(NETWORK_FILE, sep="\t")

nets = nets.rename(columns={"subject":"Arunno"})

meta["Arunno"] = meta["Arunno"].apply(clean_id)
nets["Arunno"] = nets["Arunno"].apply(clean_id)

df = pd.merge(meta, nets, on="Arunno")

network_cols = sorted([c for c in df.columns if c.startswith("Amp_Net")])
df[network_cols] = StandardScaler().fit_transform(df[network_cols])

# =============================================================================
# BUILD CARDIAC DOMAINS
# =============================================================================

print("Building cardiac domain scores")

for group, metrics in CARDIAC_GROUPS.items():
    valid = [m for m in metrics if m in df.columns]
    tmp = df[valid].dropna()
    if len(tmp) < MIN_N:
        continue
    df.loc[tmp.index, f"{group}_meanz"] = safe_mean_z(tmp)

# =============================================================================
# CORE FUNCTION — RUN BCCI
# =============================================================================

def run_bcci(df_in, covars=None, suffix=""):

    df_model = df_in.copy()

    # ---------------------------
    # residualize (if covars)
    # ---------------------------
    for group in CARDIAC_GROUPS:

        meanz = f"{group}_meanz"
        target = f"{group}_target{suffix}"

        if meanz not in df_model:
            continue

        if covars is None:
            df_model[target] = df_model[meanz]
        else:
            df_model[target], _ = residualize(df_model, meanz, covars)

    # ---------------------------
    # build BCCI
    # ---------------------------
    for group in CARDIAC_GROUPS:

        ycol = f"{group}_target{suffix}"
        bcci_col = f"{group}_BCCI{suffix}"

        if ycol not in df_model:
            continue

        tmp = df_model[network_cols + [ycol]].dropna()

        if len(tmp) < MIN_N:
            continue

        X = tmp[network_cols].values
        y = tmp[ycol].values

        w = build_weights(X, y)

        df_model.loc[tmp.index, bcci_col] = X @ w

    return df_model

# =============================================================================
# LOOCV
# =============================================================================

def run_loocv(df_model, suffix):

    rows = []

    for group in CARDIAC_GROUPS:

        ycol = f"{group}_target{suffix}"
        bcci_col = f"{group}_BCCI{suffix}"

        if ycol not in df_model:
            continue

        tmp = df_model[network_cols + [ycol]].dropna()

        if len(tmp) < MIN_N:
            continue

        X = tmp[network_cols].values
        y = tmp[ycol].values

        loo = LeaveOneOut()
        preds = np.zeros(len(y))

        for train, test in loo.split(X):
            w = build_weights(X[train], y[train])
            preds[test] = np.dot(X[test], w)

        rho, p = spearmanr(preds, y)

        ss_res = np.sum((y - preds)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot

        rows.append([group, rho, p, r2, len(y)])

    return pd.DataFrame(rows, columns=["domain","rho","p","R2","N"])


# =============================================================================
# SAVE FIGURE 1 STATS
# =============================================================================

from sklearn.model_selection import LeaveOneOut
from scipy.stats import pearsonr
import numpy as np

def compute_loocv_stats(x, y):

    loo = LeaveOneOut()
    preds = []
    true = []

    for train_idx, test_idx in loo.split(x):

        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(x_train)) < 2:
            continue

        # simple linear fit
        beta = np.polyfit(x_train, y_train, 1)
        y_pred = np.polyval(beta, x_test)

        preds.append(y_pred[0])
        true.append(y_test[0])

    preds = np.array(preds)
    true = np.array(true)

    if len(preds) < 5:
        return np.nan, np.nan, np.nan

    rho, p = pearsonr(preds, true)

    ss_res = np.sum((true - preds)**2)
    ss_tot = np.sum((true - np.mean(true))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return rho, p, r2


# =============================================================================
# SAVE FIGURE 1 STATS (CSV VERSION)
# =============================================================================

from sklearn.model_selection import LeaveOneOut
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import os

def compute_loocv_stats(x, y):

    loo = LeaveOneOut()
    preds = []
    true = []

    for train_idx, test_idx in loo.split(x):

        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(x_train)) < 2:
            continue

        beta = np.polyfit(x_train, y_train, 1)
        y_pred = np.polyval(beta, x_test)

        preds.append(y_pred[0])
        true.append(y_test[0])

    preds = np.array(preds)
    true = np.array(true)

    if len(preds) < 5:
        return np.nan, np.nan, np.nan

    rho, p = pearsonr(preds, true)

    ss_res = np.sum((true - preds)**2)
    ss_tot = np.sum((true - np.mean(true))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return rho, p, r2


def save_figure1_stats_csv(df_model, suffix, outdir, label):

    rows = []

    for domain in DOMAINS:

        x = df_model[f"{domain}_BCCI{suffix}"].values
        y = df_model[f"{domain}_target{suffix}"].values

        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        if len(x) < 10:
            continue

        # correlation
        rho, p = pearsonr(x, y)

        # bootstrap CI
        boots = []
        for _ in range(1000):
            idx = np.random.choice(len(x), len(x), replace=True)
            r, _ = pearsonr(x[idx], y[idx])
            boots.append(r)

        ci_low = np.percentile(boots, 2.5)
        ci_high = np.percentile(boots, 97.5)

        # LOOCV
        rho_cv, p_cv, r2_cv = compute_loocv_stats(x, y)

        rows.append({
            "domain": domain,
            "rho": rho,
            "pval": p,
            "n": len(x),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "rho_cv": rho_cv,
            "p_cv": p_cv,
            "r2_cv": r2_cv
        })

    df_out = pd.DataFrame(rows)

    os.makedirs(outdir, exist_ok=True)

    save_path = os.path.join(outdir, f"Figure1_stats_{label}.csv")
    df_out.to_csv(save_path, index=False)

    print(f"Saved: {save_path}")

    return df_out



def save_figure1_stats(df_model, suffix, outdir, label):

    rows = []

    for domain in DOMAINS:

        x = df_model[f"{domain}_BCCI{suffix}"].values
        y = df_model[f"{domain}_target{suffix}"].values

        mask = np.isfinite(x) & np.isfinite(y)

        x = x[mask]
        y = y[mask]

        if len(x) < 10:
            continue

        # correlation
        rho, p = pearsonr(x, y)

        # bootstrap CI
        boots = []
        for _ in range(1000):
            idx = np.random.choice(len(x), len(x), replace=True)
            r, _ = pearsonr(x[idx], y[idx])
            boots.append(r)

        ci_low = np.percentile(boots, 2.5)
        ci_high = np.percentile(boots, 97.5)

        # LOOCV
        rho_cv, p_cv, r2_cv = compute_loocv_stats(x, y)

        rows.append({
            "domain": domain,
            "rho": rho,
            "pval": p,
            "n": len(x),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "rho_cv": rho_cv,
            "p_cv": p_cv,
            "r2_cv": r2_cv
        })

    df_out = pd.DataFrame(rows)

    save_path = os.path.join(outdir, f"Figure1_stats_{label}.tsv")
    df_out.to_csv(save_path, sep="\t", index=False)

    print(f"Saved: {save_path}")

    return df_out




# =============================================================================
# FIGURE 1
# =============================================================================

def plot_figure1(df_model, suffix, tag):

    fig, axes = plt.subplots(1, len(CARDIAC_GROUPS), figsize=(20,5))

    for ax, group in zip(axes, CARDIAC_GROUPS):

        x = f"{group}_BCCI{suffix}"
        y = f"{group}_target{suffix}"

        tmp = df_model[[x,y]].dropna()

        if len(tmp) < MIN_N:
            continue

        rho,p = spearmanr(tmp[x], tmp[y])
        model = sm.OLS(tmp[y], sm.add_constant(tmp[x])).fit()

        sns.regplot(x=tmp[x], y=tmp[y], ax=ax)

        ax.set_title(group)
        ax.text(0.05,0.95,f"ρ={rho:.2f}\np={p:.3g}\nR²={model.rsquared:.2f}",
                transform=ax.transAxes, va="top")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"Figure1_{tag}.png"))
    plt.close()

# =============================================================================
# FIGURE 4 (NETWORK WEIGHTS)
# =============================================================================

def compute_weights(df_model, suffix, tag):

    rows = []

    for group in CARDIAC_GROUPS:

        ycol = f"{group}_target{suffix}"

        if ycol not in df_model:
            continue

        for net in network_cols:

            tmp = df_model[[net, ycol]].dropna()

            if len(tmp) < MIN_N:
                continue

            r,p = pearsonr(tmp[net], tmp[ycol])

            rows.append({
                "domain":group,
                "network":net,
                "weight":r,
                "p":p
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUTDIR, f"weights_{tag}.tsv"), sep="\t", index=False)

    return df_out

# =============================================================================
# RUN PIPELINE
# =============================================================================

print("\nRunning UNADJUSTED")

df_unadj = run_bcci(df, covars=None, suffix="_unadj")

plot_figure1(df_unadj, "_unadj", "unadjusted")

loocv_unadj = run_loocv(df_unadj, "_unadj")
loocv_unadj.to_csv(os.path.join(OUTDIR, "LOOCV_unadjusted.tsv"), sep="\t", index=False)

weights_unadj = compute_weights(df_unadj, "_unadj", "unadjusted")

# -------------------------------------------------------------------------

print("\nRunning ADJUSTED")

covars = [c for c in COVARS_FULL if c in df.columns]

df_adj = run_bcci(df, covars=covars, suffix="_adj")

plot_figure1(df_adj, "_adj", "adjusted")

loocv_adj = run_loocv(df_adj, "_adj")
loocv_adj.to_csv(os.path.join(OUTDIR, "LOOCV_adjusted.tsv"), sep="\t", index=False)

weights_adj = compute_weights(df_adj, "_adj", "adjusted")

# =============================================================================
# DIFFERENCE (KEY RESULT)
# =============================================================================

print("\nComputing difference in weights")

diff = weights_adj.copy()
diff["weight_diff"] = weights_unadj["weight"] - weights_adj["weight"]

diff.to_csv(os.path.join(OUTDIR, "weights_difference.tsv"), sep="\t", index=False)

print("\nDONE")

robust_unadj = check_bcci_robustness(df_unadj, "_unadj", "unadjusted")
robust_adj   = check_bcci_robustness(df_adj, "_adj", "adjusted")


# UNADJUSTED
stats_unadj = save_figure1_stats(
    df_model=df_unadj,
    suffix="_unadj",
    outdir=OUTDIR,
    label="unadjusted"
)

# ADJUSTED
stats_adj = save_figure1_stats(
    df_model=df_adj,
    suffix="_adj",
    outdir=OUTDIR,
    label="adjusted"
)

# UNADJUSTED
stats_unadj = save_figure1_stats_csv(
    df_model=df_unadj,
    suffix="_unadj",
    outdir=OUTDIR,
    label="unadjusted"
)

# ADJUSTED
stats_adj = save_figure1_stats_csv(
    df_model=df_adj,
    suffix="_adj",
    outdir=OUTDIR,
    label="adjusted"
)
# =============================================================================
# FIGURE 4 — NETWORK WEIGHTS COMPARISON
# =============================================================================

def plot_figure4(weights_unadj, weights_adj):

    print("\nBuilding Figure 4")

    mat_unadj = build_weight_matrix(weights_unadj)
    mat_adj   = build_weight_matrix(weights_adj)

    common = mat_unadj.index.intersection(mat_adj.index)
    mat_unadj = mat_unadj.loc[common]
    mat_adj   = mat_adj.loc[common]

    diff = mat_unadj - mat_adj

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    for ax, mat, title in zip(
        axes,
        [mat_unadj, mat_adj, diff],
        ["A. Unadjusted", "B. Adjusted", "C. Difference"]
    ):
        sns.heatmap(mat, cmap="coolwarm", center=0, ax=ax)
        ax.set_title(title)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "Figure4_network_comparison.png"), dpi=300)
    plt.close()

    # save matrices
    mat_unadj.to_csv(os.path.join(OUTDIR, "weights_matrix_unadj.tsv"), sep="\t")
    mat_adj.to_csv(os.path.join(OUTDIR, "weights_matrix_adj.tsv"), sep="\t")
    diff.to_csv(os.path.join(OUTDIR, "weights_matrix_diff.tsv"), sep="\t")

    return mat_unadj, mat_adj, diff



mat_unadj, mat_adj, mat_diff = plot_figure4(weights_unadj, weights_adj)




# =============================================================================
# TOP NETWORK SHIFTS
# =============================================================================

def summarize_top_changes(diff):

    df_long = diff.reset_index().melt(
        id_vars="index",
        var_name="domain",
        value_name="delta"
    )

    df_long = df_long.rename(columns={"index": "network"})

    # 🔥 APPLY LABELS HERE TOO
    df_long["network"] = df_long["network"].map(lambda x: x)

    df_long["abs_delta"] = df_long["delta"].abs()

    top = df_long.sort_values("abs_delta", ascending=False).head(15)

    print("\nTop network shifts:")
    print(top)

    top.to_csv(
        os.path.join(OUTDIR, "top_network_changes.tsv"),
        sep="\t",
        index=False
    )

    return top


top_changes = summarize_top_changes(mat_diff)




import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# -----------------------------------------------------------------------------
# USER SETTINGS
# -----------------------------------------------------------------------------

#SIGNIFICANT_DOMAINS = ["rate_control", "pumping"]  # adjust if needed
SIGNIFICANT_DOMAINS = [
    "rate_control",
    "pumping",
    "systolic_function",
    "diastolic_function"
]
ALPHA = 0.05

# -----------------------------------------------------------------------------
# NETWORK LABELS (EDIT IF NEEDED)
# -----------------------------------------------------------------------------

NETWORK_LABELS = {
    "Amp_Net01": "Midline Autonomic Axis",
    "Amp_Net02": "Anterior Medial (mPFC–ACC)",
    "Amp_Net03": "Posterior Association",
    "Amp_Net04": "Sensorimotor–Insular",
    "Amp_Net05": "Primary Somatomotor",
    "Amp_Net06": "Ventral Temporal",
    "Amp_Net07": "Olfactory–Basal",
    "Amp_Net08": "Temporal–Insular",
    "Amp_Net09": "Thalamo–Brainstem",
    "Amp_Net10": "Tecto–Cerebellar",
    "Amp_Net11": "Cerebellar Crus",
    "Amp_Net12": "Olfactory Bulb"
}

# Optional ordering (recommended)
NETWORK_ORDER = [
    "Midline Autonomic Axis",
    "Anterior Medial (mPFC–ACC)",
    "Thalamo–Brainstem",
    "Sensorimotor–Insular",
    "Primary Somatomotor",
    "Posterior Association",
    "Temporal–Insular",
    "Ventral Temporal",
    "Olfactory–Basal",
    "Olfactory Bulb",
    "Tecto–Cerebellar",
    "Cerebellar Crus"
]

# -----------------------------------------------------------------------------
# 1. COMPUTE WEIGHT STATS (r, p)
# -----------------------------------------------------------------------------

def compute_weight_stats(df_model, suffix, network_cols, cardiac_groups, min_n=20):

    rows = []

    for group in cardiac_groups:

        y = f"{group}_target{suffix}"
        if y not in df_model:
            continue

        for net in network_cols:

            if net not in df_model:
                continue

            tmp = df_model[[net, y]].dropna()

            if len(tmp) < min_n:
                continue

            r, p = pearsonr(tmp[net], tmp[y])

            rows.append({
                "network": net,
                "domain": group,
                "r": r,
                "p": p,
                "N": len(tmp)
            })

    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# 2. BUILD MATRICES
# -----------------------------------------------------------------------------

def build_matrix_with_p(df_stats, network_cols):

    mat_r = df_stats.pivot(index="network", columns="domain", values="r")
    mat_p = df_stats.pivot(index="network", columns="domain", values="p")

    # ensure ordering
    mat_r = mat_r.loc[[n for n in network_cols if n in mat_r.index]]
    mat_p = mat_p.loc[mat_r.index]

    # apply labels
    labels = [NETWORK_LABELS.get(n, n) for n in mat_r.index]
    mat_r.index = labels
    mat_p.index = labels

    # reorder nicely
    mat_r = mat_r.loc[[n for n in NETWORK_ORDER if n in mat_r.index]]
    mat_p = mat_p.loc[mat_r.index]

    return mat_r, mat_p

# -----------------------------------------------------------------------------
# 3. PLOT FIGURE 4 WITH SIGNIFICANCE
# -----------------------------------------------------------------------------

def plot_figure4_sig(df_unadj,
                    df_adj,
                    network_cols,
                    cardiac_groups,
                    outdir):

    print("\nBuilding Figure 4 (with significance)")

    # -------------------------
    # compute stats
    # -------------------------
    stats_unadj = compute_weight_stats(df_unadj, "_unadj",
                                       network_cols, cardiac_groups)

    stats_adj = compute_weight_stats(df_adj, "_adj",
                                     network_cols, cardiac_groups)

    # -------------------------
    # matrices
    # -------------------------
    mat_r_u, mat_p_u = build_matrix_with_p(stats_unadj, network_cols)
    mat_r_a, mat_p_a = build_matrix_with_p(stats_adj, network_cols)

    # -------------------------
    # filter domains
    # -------------------------
    keep_domains = [d for d in SIGNIFICANT_DOMAINS if d in mat_r_u.columns]

    mat_r_u = mat_r_u[keep_domains]
    mat_r_a = mat_r_a[keep_domains]
    mat_p_u = mat_p_u[keep_domains]
    mat_p_a = mat_p_a[keep_domains]

    diff = mat_r_u - mat_r_a

    # -------------------------
    # plot
    # -------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    # ---- UNADJUSTED ----
    sns.heatmap(mat_r_u, cmap="coolwarm", center=0, ax=axes[0])
    axes[0].set_title("A. Unadjusted")

    for i in range(mat_p_u.shape[0]):
        for j in range(mat_p_u.shape[1]):
            if mat_p_u.iloc[i, j] < ALPHA:
                axes[0].text(j + 0.5, i + 0.5, "*",
                             ha="center", va="center", color="black")

    # ---- ADJUSTED ----
    sns.heatmap(mat_r_a, cmap="coolwarm", center=0, ax=axes[1])
    axes[1].set_title("B. Adjusted")

    for i in range(mat_p_a.shape[0]):
        for j in range(mat_p_a.shape[1]):
            if mat_p_a.iloc[i, j] < ALPHA:
                axes[1].text(j + 0.5, i + 0.5, "*",
                             ha="center", va="center", color="black")

    # ---- DIFFERENCE ----
    sns.heatmap(diff, cmap="bwr", center=0, ax=axes[2])
    axes[2].set_title("C. Difference (Unadj − Adj)")

    # formatting
    for ax in axes:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.tight_layout()

    save_path = os.path.join(outdir, "Figure4_sig.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")

    # -------------------------
    # save tables
    # -------------------------
    mat_r_u.to_csv(os.path.join(outdir, "weights_unadj_matrix.tsv"), sep="\t")
    mat_r_a.to_csv(os.path.join(outdir, "weights_adj_matrix.tsv"), sep="\t")
    diff.to_csv(os.path.join(outdir, "weights_diff_matrix.tsv"), sep="\t")

    mat_p_u.to_csv(os.path.join(outdir, "pvals_unadj.tsv"), sep="\t")
    mat_p_a.to_csv(os.path.join(outdir, "pvals_adj.tsv"), sep="\t")

    return mat_r_u, mat_r_a, diff

mat_u, mat_a, mat_diff = plot_figure4_sig(
    df_unadj=df_unadj,
    df_adj=df_adj,
    network_cols=network_cols,
    cardiac_groups=CARDIAC_GROUPS,
    outdir=OUTDIR
)





# =============================================================================
# FIGURE 4 — FINAL (ALL DOMAINS, STATS + STAR MASK)
# =============================================================================

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# -----------------------------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------------------------

ALPHA = 0.05
TREND = 0.10

DOMAINS = [
    "rate_control",
    "pumping",
    "systolic_function",
    "diastolic_function"
]

# -----------------------------------------------------------------------------
# NETWORK LABELS (EDIT IF NEEDED)
# -----------------------------------------------------------------------------

NETWORK_LABELS = {
    "Amp_Net01": "Midline Autonomic Axis",
    "Amp_Net02": "Anterior Medial (mPFC–ACC)",
    "Amp_Net03": "Posterior Association",
    "Amp_Net04": "Sensorimotor–Insular",
    "Amp_Net05": "Primary Somatomotor",
    "Amp_Net06": "Ventral Temporal",
    "Amp_Net07": "Olfactory–Basal",
    "Amp_Net08": "Temporal–Insular",
    "Amp_Net09": "Thalamo–Brainstem",
    "Amp_Net10": "Tecto–Cerebellar",
    "Amp_Net11": "Cerebellar Crus",
    "Amp_Net12": "Olfactory Bulb"
}

NETWORK_ORDER = [
    "Midline Autonomic Axis",
    "Anterior Medial (mPFC–ACC)",
    "Thalamo–Brainstem",
    "Sensorimotor–Insular",
    "Primary Somatomotor",
    "Posterior Association",
    "Temporal–Insular",
    "Ventral Temporal",
    "Olfactory–Basal",
    "Olfactory Bulb",
    "Tecto–Cerebellar",
    "Cerebellar Crus"
]

# -----------------------------------------------------------------------------
# 1. COMPUTE STATS
# -----------------------------------------------------------------------------

def compute_weight_stats(df_model, suffix, network_cols, cardiac_groups, min_n=20):

    rows = []

    for group in cardiac_groups:

        y = f"{group}_target{suffix}"
        if y not in df_model:
            continue

        for net in network_cols:

            if net not in df_model:
                continue

            tmp = df_model[[net, y]].dropna()

            if len(tmp) < min_n:
                continue

            r, p = pearsonr(tmp[net], tmp[y])

            rows.append({
                "network": net,
                "domain": group,
                "r": r,
                "p": p,
                "N": len(tmp)
            })

    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# 2. BUILD MATRICES
# -----------------------------------------------------------------------------

def build_matrix_with_p(df_stats, network_cols):

    mat_r = df_stats.pivot(index="network", columns="domain", values="r")
    mat_p = df_stats.pivot(index="network", columns="domain", values="p")

    # ensure consistent order
    mat_r = mat_r.loc[[n for n in network_cols if n in mat_r.index]]
    mat_p = mat_p.loc[mat_r.index]

    # apply labels
    labels = [NETWORK_LABELS.get(n, n) for n in mat_r.index]
    mat_r.index = labels
    mat_p.index = labels

    # reorder for interpretability
    mat_r = mat_r.loc[[n for n in NETWORK_ORDER if n in mat_r.index]]
    mat_p = mat_p.loc[mat_r.index]

    return mat_r, mat_p

# -----------------------------------------------------------------------------
# 3. MAIN FUNCTION — BUILD FIGURE 4
# -----------------------------------------------------------------------------

def run_figure4(df_unadj,
                df_adj,
                network_cols,
                cardiac_groups,
                outdir):

    print("\nBuilding FINAL Figure 4")

    # -------------------------
    # stats
    # -------------------------
    stats_unadj = compute_weight_stats(df_unadj, "_unadj",
                                       network_cols, cardiac_groups)

    stats_adj = compute_weight_stats(df_adj, "_adj",
                                     network_cols, cardiac_groups)

    # -------------------------
    # matrices
    # -------------------------
    mat_u, mat_p_u = build_matrix_with_p(stats_unadj, network_cols)
    mat_a, mat_p_a = build_matrix_with_p(stats_adj, network_cols)

    # keep all domains (ordered)
    keep = [d for d in DOMAINS if d in mat_u.columns]

    mat_u = mat_u[keep]
    mat_a = mat_a[keep]
    mat_p_u = mat_p_u[keep]
    mat_p_a = mat_p_a[keep]
    
    diff = mat_u - mat_a

    # -------------------------
    # plot
    # -------------------------
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    panels = [
        (mat_u, "A. Unadjusted"),
        (mat_a, "B. Adjusted"),
        (diff,  "C. Difference (Unadj − Adj)")
    ]

    for k, (mat, title) in enumerate(panels):

        ax = axes[k]

        sns.heatmap(
            mat,
            cmap="coolwarm",
            center=0,
            ax=ax,
            cbar_kws={"label": "Weight"}
        )

        ax.set_title(title, fontsize=14)

        # -------------------------
        # significance mask (UNADJ)
        # -------------------------
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):

                pval = mat_p_u.iloc[i, j]

                if pval < ALPHA:
                    ax.text(j + 0.5, i + 0.5, "●",
                            ha="center", va="center",
                            color="black", fontsize=10)

                elif pval < TREND:
                    ax.text(j + 0.5, i + 0.5, "+",
                            ha="center", va="center",
                            color="black", fontsize=9)

        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.tight_layout()

    # -------------------------
    # save outputs
    # -------------------------
    os.makedirs(outdir, exist_ok=True)

    plt.savefig(os.path.join(outdir, "Figure4_FINAL.png"), dpi=300)
    plt.close()

    mat_u.to_csv(os.path.join(outdir, "weights_unadj.tsv"), sep="\t")
    mat_a.to_csv(os.path.join(outdir, "weights_adj.tsv"), sep="\t")
    diff.to_csv(os.path.join(outdir, "weights_diff.tsv"), sep="\t")
    mat_p_u.to_csv(os.path.join(outdir, "pvals_unadj.tsv"), sep="\t")

    print("Saved Figure 4 + matrices")

    return mat_u, mat_a, diff, mat_p_u, mat_p_a



mat_u, mat_a, mat_diff, mat_p_u, mat_p_a = run_figure4(
    df_unadj=df_unadj,
    df_adj=df_adj,
    network_cols=network_cols,
    cardiac_groups=CARDIAC_GROUPS,
    outdir=OUTDIR
)



# =============================================================================
# FIGURE 4 — FINAL MULTIPANEL (UNADJ vs ADJ + STATS + PANEL D)
# =============================================================================

def build_figure4_final(
    mat_u,
    mat_a,
    mat_p_u,
    mat_p_a,
    out_path
):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    print("\nBuilding Figure 4 FINAL (multipanel)")

    # -------------------------
    # matrices
    # -------------------------
    mat_unadj = mat_u.copy()
    mat_adj   = mat_a.copy()
    mat_diff  = mat_unadj - mat_adj

    domains = mat_unadj.columns.tolist()

    # -------------------------
    # significance masks
    # -------------------------
    star_mask_unadj = mat_p_u.values < 0.05
    star_mask_adj   = mat_p_a.values < 0.05

    # -------------------------
    # PANEL D — top networks
    # -------------------------
    df_diff = mat_diff.copy()

    df_long = df_diff.reset_index().melt(id_vars="index")
    df_long.columns = ["network", "domain", "delta"]
    df_long["abs_delta"] = df_long["delta"].abs()

    top_nets = (
        df_long
        .sort_values("abs_delta", ascending=False)
        .groupby("network")
        .first()
        .sort_values("abs_delta", ascending=False)
        .head(10)
        .index
    )

    df_top = df_diff.loc[top_nets]

    # -------------------------
    # COLORS
    # -------------------------
    domain_colors = {
        "rate_control": "#1f77b4",
        "pumping": "#ff7f0e",
        "systolic_function": "#2ca02c",
        "diastolic_function": "#d62728"
    }

    # -------------------------
    # FIGURE
    # -------------------------
    fig = plt.figure(figsize=(16, 12))

    # =========================
    # A. UNADJUSTED
    # =========================
    ax1 = plt.subplot2grid((2,2),(0,0))

    sns.heatmap(mat_unadj, cmap="coolwarm", center=0, ax=ax1)
    ax1.set_title("A. Unadjusted", fontsize=13)

    for i in range(mat_unadj.shape[0]):
        for j in range(mat_unadj.shape[1]):
            if star_mask_unadj[i, j]:
                ax1.text(j+0.5, i+0.5, "*", ha="center", va="center")

    # =========================
    # B. ADJUSTED
    # =========================
    ax2 = plt.subplot2grid((2,2),(0,1))

    sns.heatmap(mat_adj, cmap="coolwarm", center=0, ax=ax2)
    ax2.set_title("B. Adjusted", fontsize=13)

    for i in range(mat_adj.shape[0]):
        for j in range(mat_adj.shape[1]):
            if star_mask_adj[i, j]:
                ax2.text(j+0.5, i+0.5, "*", ha="center", va="center")

    # =========================
    # C. DIFFERENCE
    # =========================
    ax3 = plt.subplot2grid((2,2),(1,0))

    sns.heatmap(mat_diff, cmap="bwr", center=0, ax=ax3)
    ax3.set_title("C. Difference (Unadj − Adj)", fontsize=13)

    # =========================
    # D. TOP NETWORK SHIFTS
    # =========================
    ax4 = plt.subplot2grid((2,2),(1,1))

    df_plot = df_top.iloc[::-1]

    width = 0.2
    y = np.arange(len(df_plot.index))

    for i, d in enumerate(domains):

        sig_mask = mat_p_u[d] < 0.05
        sig_subset = sig_mask.loc[df_plot.index]

        for j, val in enumerate(df_plot[d]):

            alpha = 1.0 if sig_subset.iloc[j] else 0.3

            ax4.barh(
                y[j] + i*width,
                val,
                height=width,
                color=domain_colors.get(d, "gray"),
                alpha=alpha
            )

    ax4.set_yticks(y + width)
    ax4.set_yticklabels(df_plot.index)
    ax4.axvline(0, color='k')
    ax4.set_title("D. Top Network Shifts (Δ weights)", fontsize=13)

    # -------------------------
    # LEGEND BELOW PANEL D
    # -------------------------
    handles = [
        plt.Line2D([0],[0], color=domain_colors[d], lw=6, label=d)
        for d in domains
    ]

    sig_handle = plt.Line2D([0],[0], color="black", lw=6, alpha=1.0, label="p < 0.05")
    nonsig_handle = plt.Line2D([0],[0], color="black", lw=6, alpha=0.3, label="n.s.")

    ax4.legend(
        handles=handles + [sig_handle, nonsig_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=3,
        frameon=False
    )

    # -------------------------
    # formatting
    # -------------------------
    for ax in [ax1, ax2, ax3]:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved: {out_path}")
    
print("\nBuilding Figure 4 FINAL (with Panel D)")



build_figure4_final(
    mat_u=mat_u,
    mat_a=mat_a,
    mat_p_u=mat_p_u,
    mat_p_a=mat_p_a,
    out_path=os.path.join(OUTDIR, "Figure4_FINAL_with_panelD.png")
)
print("Saved: Figure4_FINAL_with_panelD.png")


stats_unadj.to_csv(
    os.path.join(OUTDIR, "Figure4_stats_unadjusted_FDR.tsv"),
    sep="\t",
    index=False
)

stats_adj.to_csv(
    os.path.join(OUTDIR, "Figure4_stats_adjusted_FDR.tsv"),
    sep="\t",
    index=False
)

# =============================================================================
# FIGURE 4 — FINAL (FDR-CORRECTED, MULTIPANEL WITH PANEL D)
# =============================================================================

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

# -----------------------------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------------------------

ALPHA = 0.05
TREND = 0.10

DOMAINS = [
    "rate_control",
    "pumping",
    "systolic_function",
    "diastolic_function"
]

# -----------------------------------------------------------------------------
# NETWORK LABELS
# -----------------------------------------------------------------------------

NETWORK_LABELS = {
    "Amp_Net01": "Midline Autonomic Axis",
    "Amp_Net02": "Anterior Medial (mPFC–ACC)",
    "Amp_Net03": "Posterior Association",
    "Amp_Net04": "Sensorimotor–Insular",
    "Amp_Net05": "Primary Somatomotor",
    "Amp_Net06": "Ventral Temporal",
    "Amp_Net07": "Olfactory–Basal",
    "Amp_Net08": "Temporal–Insular",
    "Amp_Net09": "Thalamo–Brainstem",
    "Amp_Net10": "Tecto–Cerebellar",
    "Amp_Net11": "Cerebellar Crus",
    "Amp_Net12": "Olfactory Bulb"
}

NETWORK_ORDER = [
    "Midline Autonomic Axis",
    "Anterior Medial (mPFC–ACC)",
    "Thalamo–Brainstem",
    "Sensorimotor–Insular",
    "Primary Somatomotor",
    "Posterior Association",
    "Temporal–Insular",
    "Ventral Temporal",
    "Olfactory–Basal",
    "Olfactory Bulb",
    "Tecto–Cerebellar",
    "Cerebellar Crus"
]

# -----------------------------------------------------------------------------
# 1. COMPUTE STATS + FDR
# -----------------------------------------------------------------------------

def compute_weight_stats_fdr(df_model, suffix, network_cols, cardiac_groups, min_n=20):

    rows = []

    for group in cardiac_groups:

        y = f"{group}_target{suffix}"
        if y not in df_model:
            continue

        for net in network_cols:

            if net not in df_model:
                continue

            tmp = df_model[[net, y]].dropna()

            if len(tmp) < min_n:
                continue

            r, p = pearsonr(tmp[net], tmp[y])

            rows.append({
                "network": net,
                "domain": group,
                "r": r,
                "p": p,
                "N": len(tmp)
            })

    df_stats = pd.DataFrame(rows)

    # -------------------------
    # FDR (within domain)
    # -------------------------
    df_stats["p_fdr"] = np.nan
    df_stats["sig_fdr"] = False

    for d in df_stats["domain"].unique():

        mask = df_stats["domain"] == d

        pvals = df_stats.loc[mask, "p"].values

        reject, p_fdr, _, _ = multipletests(pvals, method="fdr_bh")

        df_stats.loc[mask, "p_fdr"] = p_fdr
        df_stats.loc[mask, "sig_fdr"] = reject

    return df_stats




def build_matrices(df_stats, network_cols):

    # --- pivot ---
    mat_r = df_stats.pivot(index="network", columns="domain", values="r")
    mat_p_raw = df_stats.pivot(index="network", columns="domain", values="p")
    mat_p_fdr = df_stats.pivot(index="network", columns="domain", values="p_fdr")

    # --- enforce network order ---
    ordered = [n for n in network_cols if n in mat_r.index]

    mat_r = mat_r.loc[ordered]
    mat_p_raw = mat_p_raw.loc[ordered]
    mat_p_fdr = mat_p_fdr.loc[ordered]

    # --- apply labels ---
    labels = [NETWORK_LABELS.get(n, n) for n in mat_r.index]

    mat_r.index = labels
    mat_p_raw.index = labels
    mat_p_fdr.index = labels

    # --- final nice ordering ---
    final_order = [n for n in NETWORK_ORDER if n in mat_r.index]

    mat_r = mat_r.loc[final_order]
    mat_p_raw = mat_p_raw.loc[final_order]
    mat_p_fdr = mat_p_fdr.loc[final_order]

    return mat_r, mat_p_raw, mat_p_fdr








def run_figure4_fdr(df_unadj,
                   df_adj,
                   network_cols,
                   cardiac_groups,
                   outdir):

    print("\nBuilding FINAL Figure 4 (FDR-corrected)")

    # =========================
    # STATS
    # =========================
    stats_unadj = compute_weight_stats_fdr(
        df_unadj, "_unadj", network_cols, cardiac_groups
    )

    stats_adj = compute_weight_stats_fdr(
        df_adj, "_adj", network_cols, cardiac_groups
    )

    # =========================
    # MATRICES
    # =========================
    mat_u, mat_p_u_raw, mat_p_u = build_matrices(stats_unadj, network_cols)
    mat_a, mat_p_a_raw, mat_p_a = build_matrices(stats_adj, network_cols)

    # keep consistent domains
    keep = [d for d in DOMAINS if d in mat_u.columns]

    mat_u = mat_u[keep]
    mat_a = mat_a[keep]
    mat_p_u = mat_p_u[keep]
    mat_p_a = mat_p_a[keep]

    mat_p_u_raw = mat_p_u_raw[keep]
    mat_p_a_raw = mat_p_a_raw[keep]

    mat_diff = mat_u - mat_a

    # =========================
    # SAVE MATRICES (KEY FIX)
    # =========================
    os.makedirs(outdir, exist_ok=True)

    mat_p_u_raw.to_csv(os.path.join(outdir, "Figure4A_pvals_raw.tsv"), sep="\t")
    mat_p_u.to_csv(os.path.join(outdir, "Figure4A_pvals_FDR.tsv"), sep="\t")

    mat_p_a_raw.to_csv(os.path.join(outdir, "Figure4B_pvals_raw.tsv"), sep="\t")
    mat_p_a.to_csv(os.path.join(outdir, "Figure4B_pvals_FDR.tsv"), sep="\t")

    print("Saved p-value matrices for Figure 4")

    # =========================
    # PANEL D DATA
    # =========================
    # =========================
    # PANEL D (IMPROVED)
    # =========================
    
    df_long = mat_diff.reset_index().melt(id_vars="index")
    df_long.columns = ["network", "domain", "delta"]
    
    df_long["abs_delta"] = df_long["delta"].abs()
    
    # add significance from BOTH models
    df_long["sig_unadj"] = df_long.apply(
        lambda r: mat_p_u.loc[r["network"], r["domain"]] < ALPHA, axis=1
    )
    
    df_long["sig_adj"] = df_long.apply(
        lambda r: mat_p_a.loc[r["network"], r["domain"]] < ALPHA, axis=1
    )
    
    # keep only meaningful rows
    df_long = df_long[df_long["sig_unadj"] | df_long["sig_adj"]]
    
    # rank
    top_nets = (
        df_long.sort_values("abs_delta", ascending=False)
        .groupby("network")
        .first()
        .sort_values("abs_delta", ascending=False)
        .head(10)
        .index
    )
    
    df_top = mat_diff.loc[top_nets]

    # =========================
    # COLORS
    # =========================
    domain_colors = {
        "rate_control": "#1f77b4",
        "pumping": "#ff7f0e",
        "systolic_function": "#2ca02c",
        "diastolic_function": "#d62728"
    }

    # =========================
    # FIGURE
    # =========================
    fig = plt.figure(figsize=(16, 12))

    # ---- A ----
    ax1 = plt.subplot2grid((2,2),(0,0))
    sns.heatmap(mat_u, cmap="coolwarm", center=0, ax=ax1)
    ax1.set_title("A. Unadjusted (FDR)", fontsize=13)

    for i in range(mat_u.shape[0]):
        for j in range(mat_u.shape[1]):
            pval = mat_p_u.iloc[i, j]
            if pval < ALPHA:
                ax1.text(j+0.5, i+0.5, "●", ha="center", va="center")
            elif pval < TREND:
                ax1.text(j+0.5, i+0.5, "+", ha="center", va="center")

    # ---- B ----
    ax2 = plt.subplot2grid((2,2),(0,1))
    sns.heatmap(mat_a, cmap="coolwarm", center=0, ax=ax2)
    ax2.set_title("B. Adjusted (FDR)", fontsize=13)

    for i in range(mat_a.shape[0]):
        for j in range(mat_a.shape[1]):
            pval = mat_p_a.iloc[i, j]
            if pval < ALPHA:
                ax2.text(j+0.5, i+0.5, "●", ha="center", va="center")
            elif pval < TREND:
                ax2.text(j+0.5, i+0.5, "+", ha="center", va="center")

    # ---- C ----
    ax3 = plt.subplot2grid((2,2),(1,0))
    sns.heatmap(mat_diff, cmap="bwr", center=0, ax=ax3)
    ax3.set_title("C. Difference (Unadj − Adj)", fontsize=13)

    # ---- D ----
 # =========================
    # PANEL D — PAIRED CONTRIBUTIONS
    # =========================
    
    from matplotlib.lines import Line2D
    
    def compute_contribution(mat_r, mat_p):
        df = mat_r.reset_index().melt(id_vars="index")
        df.columns = ["network", "domain", "r"]
    
        df["p"] = df.apply(
            lambda x: mat_p.loc[x["network"], x["domain"]], axis=1
        )
    
        # contribution score (effect + reliability)
        df["score"] = df["r"].abs() * (-np.log10(df["p"] + 1e-10))
    
        return df
    
    
    # --- compute contributions ---
    df_u = compute_contribution(mat_u, mat_p_u)
    df_a = compute_contribution(mat_a, mat_p_a)
    
    # --- merge for joint ranking ---
    df_merge = df_u.copy()
    df_merge["score_adj"] = df_a["score"].values
    
    df_merge["combined_score"] = df_merge["score"] + df_merge["score_adj"]
    
    # --- select top networks globally ---
    top_nets = (
        df_merge
        .groupby("network")["combined_score"]
        .max()
        .sort_values(ascending=False)
        .head(10)
        .index
    )
    
    df_plot_u = mat_u.loc[top_nets]
    df_plot_a = mat_a.loc[top_nets]
    
    # =========================
    # PLOT
    # =========================
    
    ax4 = plt.subplot2grid((2,2),(1,1))
    
    y = np.arange(len(top_nets))
    width = 0.18
    
    for i, d in enumerate(DOMAINS):
    
        offset = (i - 1.5) * width  # center around each row
    
        for j, net in enumerate(top_nets):
    
            val_u = df_plot_u.loc[net, d]
            val_a = df_plot_a.loc[net, d]
    
            p_u = mat_p_u.loc[net, d]
            p_a = mat_p_a.loc[net, d]
    
            # transparency encodes significance
            alpha_u = 1.0 if p_u < ALPHA else 0.3
            alpha_a = 1.0 if p_a < ALPHA else 0.3
    
            # UNADJ (lighter)
            ax4.barh(
                y[j] + offset - width/4,
                val_u,
                height=width/2,
                color=domain_colors[d],
                alpha=alpha_u * 0.5
            )
    
            # ADJ (darker)
            ax4.barh(
                y[j] + offset + width/4,
                val_a,
                height=width/2,
                color=domain_colors[d],
                alpha=alpha_a
            )
    
    # formatting
    ax4.set_yticks(y)
    ax4.set_yticklabels(top_nets)
    ax4.axvline(0, color='k')
    ax4.set_title("D. Network Contributions (Unadj vs Adj)", fontsize=13)

    # =========================
    # LEGEND
    # =========================
    
    legend_elements = [
        Line2D([0], [0], color="#1f77b4", lw=4, label="rate_control"),
        Line2D([0], [0], color="#ff7f0e", lw=4, label="pumping"),
        Line2D([0], [0], color="#2ca02c", lw=4, label="systolic_function"),
        Line2D([0], [0], color="#d62728", lw=4, label="diastolic_function"),
    
        Line2D([0], [0], color="black", lw=4, alpha=0.5, label="Unadjusted"),
        Line2D([0], [0], color="black", lw=4, alpha=1.0, label="Adjusted"),
    
        Line2D([0], [0], color="black", lw=4, alpha=1.0, label="FDR p < 0.05"),
        Line2D([0], [0], color="black", lw=4, alpha=0.3, label="n.s.")
    ]
    
    ax4.legend(
        handles=legend_elements,
        loc="lower right",
        frameon=False,
        fontsize=9
    )
    
    # formatting
    for ax in [ax1, ax2, ax3]:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)


    # -------------------------
    # LEGEND FOR PANEL D
    # -------------------------
    
    legend_elements = [
        Line2D([0], [0], color="#1f77b4", lw=4, label="rate_control"),
        Line2D([0], [0], color="#ff7f0e", lw=4, label="pumping"),
        Line2D([0], [0], color="#2ca02c", lw=4, label="systolic_function"),
        Line2D([0], [0], color="#d62728", lw=4, label="diastolic_function"),
    
        Line2D([0], [0], color="black", lw=4, alpha=1.0, label="FDR p < 0.05"),
        Line2D([0], [0], color="black", lw=4, alpha=0.3, label="n.s.")
    ]
    
    ax4.legend(
        handles=legend_elements,
        loc="lower right",
        frameon=False,
        fontsize=10
    )

    plt.tight_layout()

    save_path = os.path.join(outdir, "Figure4_FINAL_FDR.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")

    return mat_u, mat_a, mat_diff, mat_p_u, mat_p_a



mat_u, mat_a, mat_diff, mat_p_u, mat_p_a = run_figure4_fdr(
    df_unadj=df_unadj,
    df_adj=df_adj,
    network_cols=network_cols,
    cardiac_groups=CARDIAC_GROUPS,
    outdir=OUTDIR
)



# =============================================================================
# FIGURE B — EFFECT SIZES (ADJUSTED vs UNADJUSTED)
# =============================================================================

def run_figureB_effects(mat_unadj,
                       mat_adj,
                       OUTDIR,
                       domains=None,
                       save_prefix="FigureB_effects"):

    """
    Inputs:
    -------
    mat_unadj : DataFrame (networks × domains)
    mat_adj   : DataFrame (networks × domains)
    OUTDIR    : output directory
    domains   : list of domains (optional, default = all columns)

    Outputs:
    --------
    - Bar plot of adjusted effect sizes (ranked)
    - Optional side-by-side comparison (unadj vs adj)
    - TSV files with effect sizes
    """

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    os.makedirs(OUTDIR, exist_ok=True)

    # -------------------------
    # select domains
    # -------------------------
    if domains is None:
        domains = mat_adj.columns.tolist()

    mat_unadj = mat_unadj[domains]
    mat_adj   = mat_adj[domains]

    # -------------------------
    # EFFECT SIZE (mean abs β across domains)
    # -------------------------
    eff_unadj = mat_unadj.abs().mean(axis=1)
    eff_adj   = mat_adj.abs().mean(axis=1)

    df_eff = pd.DataFrame({
        "unadjusted": eff_unadj,
        "adjusted": eff_adj
    })

    # -------------------------
    # ranking (by adjusted)
    # -------------------------
    df_eff = df_eff.sort_values("adjusted", ascending=False)

    # -------------------------
    # SAVE TABLE
    # -------------------------
    df_eff.to_csv(os.path.join(OUTDIR, f"{save_prefix}_table.tsv"), sep="\t")

    # -------------------------
    # PANEL B1 — ADJUSTED ONLY (MAIN FIGURE)
    # -------------------------
    plt.figure(figsize=(6,5))

    plt.barh(df_eff.index[::-1],
             df_eff["adjusted"][::-1])

    plt.xlabel("Mean |effect size| (adjusted)")
    plt.title("Network Contributions (Adjusted)")
    plt.tight_layout()

    plt.savefig(os.path.join(OUTDIR,
                f"{save_prefix}_adjusted.png"),
                dpi=300)

    plt.close()

    # -------------------------
    # PANEL B2 — COMPARISON (SUPPLEMENTARY)
    # -------------------------
    x = np.arange(len(df_eff.index))

    plt.figure(figsize=(7,5))

    width = 0.4

    plt.barh(x - width/2,
             df_eff["unadjusted"],
             height=width,
             label="Unadjusted",
             alpha=0.7)

    plt.barh(x + width/2,
             df_eff["adjusted"],
             height=width,
             label="Adjusted",
             alpha=0.9)

    plt.yticks(x, df_eff.index)
    plt.xlabel("Mean |effect size|")
    plt.title("Network Contributions: Unadjusted vs Adjusted")
    plt.legend()

    plt.tight_layout()

    plt.savefig(os.path.join(OUTDIR,
                f"{save_prefix}_comparison.png"),
                dpi=300)

    plt.close()

    # -------------------------
    # OPTIONAL: per-domain tables
    # -------------------------
    for d in domains:
        df_d = pd.DataFrame({
            "unadjusted": mat_unadj[d],
            "adjusted": mat_adj[d]
        }).sort_values("adjusted", ascending=False)

        df_d.to_csv(os.path.join(OUTDIR,
                    f"{save_prefix}_{d}.tsv"),
                    sep="\t")

    print(f"Saved FigureB effects to: {OUTDIR}")

    return df_eff

df_eff = run_figureB_effects(
    mat_unadj=mat_u,
    mat_adj=mat_a,
    OUTDIR=OUTDIR,
    domains=["rate_control", "systolic_function", "diastolic_function"]
)





###############################################################################
##### 20 March 2026 ###########################################################
###############################################################################

# =============================================================================
# FIGURE 4 — SIX-PANEL FINAL
# A. Unadjusted
# B. Adjusted
# C. Difference (Unadjusted − Adjusted)
# D. Change summary (mean |Δ|)
# E. Effect summary (mean |adjusted r|)
# F. Stability summary (bootstrap mean |adjusted r| ± 95% CI)
# =============================================================================

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

# -----------------------------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------------------------

ALPHA = 0.05
TREND = 0.10
N_BOOT = 1000
MIN_N = 20

DOMAINS = [
    "rate_control",
    "pumping",
    "systolic_function",
    "diastolic_function"
]

# -----------------------------------------------------------------------------
# NETWORK LABELS
# -----------------------------------------------------------------------------

NETWORK_LABELS = {
    "Amp_Net01": "Midline Autonomic Axis",
    "Amp_Net02": "Anterior Medial (mPFC–ACC)",
    "Amp_Net03": "Posterior Association",
    "Amp_Net04": "Sensorimotor–Insular",
    "Amp_Net05": "Primary Somatomotor",
    "Amp_Net06": "Ventral Temporal",
    "Amp_Net07": "Olfactory–Basal",
    "Amp_Net08": "Temporal–Insular",
    "Amp_Net09": "Thalamo–Brainstem",
    "Amp_Net10": "Tecto–Cerebellar",
    "Amp_Net11": "Cerebellar Crus",
    "Amp_Net12": "Olfactory Bulb"
}

NETWORK_ORDER = [
    "Midline Autonomic Axis",
    "Anterior Medial (mPFC–ACC)",
    "Thalamo–Brainstem",
    "Sensorimotor–Insular",
    "Primary Somatomotor",
    "Posterior Association",
    "Temporal–Insular",
    "Ventral Temporal",
    "Olfactory–Basal",
    "Olfactory Bulb",
    "Tecto–Cerebellar",
    "Cerebellar Crus"
]

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def _ordered_labeled_networks(network_cols):
    labels = [NETWORK_LABELS.get(n, n) for n in network_cols]
    labels = [n for n in NETWORK_ORDER if n in labels]
    return labels


def _bootstrap_corr(x, y, n_boot=1000):
    """
    Bootstrap Pearson r and return:
    mean_boot, ci95_low, ci95_high, ci90_low, ci90_high
    """
    boots = []
    n = len(x)

    if n < MIN_N:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)

        x_b = x[idx]
        y_b = y[idx]

        if np.std(x_b) == 0 or np.std(y_b) == 0:
            continue

        try:
            r_b, _ = pearsonr(x_b, y_b)
            if np.isfinite(r_b):
                boots.append(r_b)
        except Exception:
            continue

    if len(boots) < 50:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    boots = np.array(boots)

    return (
        np.mean(boots),
        np.percentile(boots, 2.5),
        np.percentile(boots, 97.5),
        np.percentile(boots, 5.0),
        np.percentile(boots, 95.0),
    )


def _bootstrap_delta_same_rows(df_unadj, df_adj, net, domain, n_boot=1000):
    """
    Bootstrap delta correlation:
    delta = r_unadj - r_adj

    Uses the same sampled rows for both models to preserve pairing.
    """
    y_u = f"{domain}_target_unadj"
    y_a = f"{domain}_target_adj"

    cols_u = [net, y_u]
    cols_a = [net, y_a]

    if net not in df_unadj.columns or net not in df_adj.columns:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    if y_u not in df_unadj.columns or y_a not in df_adj.columns:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    tmp = pd.DataFrame({
        "x": df_unadj[net],
        "y_u": df_unadj[y_u],
        "y_a": df_adj[y_a]
    }).dropna()

    if len(tmp) < MIN_N:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    x = tmp["x"].values
    y_u_vals = tmp["y_u"].values
    y_a_vals = tmp["y_a"].values

    boots = []
    n = len(tmp)

    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)

        xb = x[idx]
        yub = y_u_vals[idx]
        yab = y_a_vals[idx]

        if np.std(xb) == 0 or np.std(yub) == 0 or np.std(yab) == 0:
            continue

        try:
            r_u, _ = pearsonr(xb, yub)
            r_a, _ = pearsonr(xb, yab)
            d = r_u - r_a
            if np.isfinite(d):
                boots.append(d)
        except Exception:
            continue

    if len(boots) < 50:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    boots = np.array(boots)

    return (
        np.mean(boots),
        np.percentile(boots, 2.5),
        np.percentile(boots, 97.5),
        np.percentile(boots, 5.0),
        np.percentile(boots, 95.0),
    )


# -----------------------------------------------------------------------------
# 1. COMPUTE MODEL STATS (UNADJUSTED / ADJUSTED)
# -----------------------------------------------------------------------------

def compute_weight_stats_boot(df_model, suffix, network_cols, domains, n_boot=1000):
    """
    Returns long table with:

    Core stats:
    ----------
    network, network_label, domain
    r, p, q

    Significance:
    -------------
    sig_fdr   : q < 0.05
    trend_fdr : q < 0.10
    sig_unc   : p < 0.05 (uncorrected)
    trend_unc : p < 0.10

    Bootstrap:
    ----------
    boot_mean
    ci95_low, ci95_high
    ci90_low, ci90_high
    """

    rows = []

    for domain in domains:

        ycol = f"{domain}_target{suffix}"

        if ycol not in df_model.columns:
            continue

        for net in network_cols:

            if net not in df_model.columns:
                continue

            tmp = df_model[[net, ycol]].dropna()

            if len(tmp) < MIN_N:
                continue

            x = tmp[net].values
            y = tmp[ycol].values

            # -------------------------
            # correlation
            # -------------------------
            try:
                r, p = pearsonr(x, y)
            except Exception:
                r, p = np.nan, np.nan

            # -------------------------
            # bootstrap CI
            # -------------------------
            boot_mean, ci95_low, ci95_high, ci90_low, ci90_high = _bootstrap_corr(
                x, y, n_boot=n_boot
            )

            rows.append({
                "network": net,
                "network_label": NETWORK_LABELS.get(net, net),
                "domain": domain,

                # raw stats
                "r": r,
                "p": p,

                # bootstrap
                "boot_mean": boot_mean,
                "ci95_low": ci95_low,
                "ci95_high": ci95_high,
                "ci90_low": ci90_low,
                "ci90_high": ci90_high,

                "N": len(tmp)
            })

    df_stats = pd.DataFrame(rows)

    if len(df_stats) == 0:
        return df_stats

    # =========================================================
    # FDR CORRECTION (within domain)
    # =========================================================
    df_stats["q"] = np.nan
    df_stats["sig_fdr"] = False
    df_stats["trend_fdr"] = False
    
    for d in df_stats["domain"].unique():
    
        mask = df_stats["domain"] == d
        sub_idx = df_stats.loc[mask].index
    
        pvals = df_stats.loc[sub_idx, "p"].values
    
        # handle NaNs safely
        valid_idx = np.where(np.isfinite(pvals))[0]
    
        if len(valid_idx) == 0:
            continue
    
        reject, qvals, _, _ = multipletests(pvals[valid_idx], method="fdr_bh")
    
        # map back to original indices
        idx_valid = sub_idx[valid_idx]
    
        df_stats.loc[idx_valid, "q"] = qvals
        df_stats.loc[idx_valid, "sig_fdr"] = qvals < ALPHA
        df_stats.loc[idx_valid, "trend_fdr"] = qvals < TREND

    # =========================================================
    # UNCORRECTED SIGNIFICANCE (FOR '+' SYMBOLS)
    # =========================================================
    df_stats["sig_unc"] = df_stats["p"] < 0.05
    df_stats["trend_unc"] = df_stats["p"] < 0.10

    # =========================================================
    # CLEANUP (avoid chained assignment warnings)
    # =========================================================
    df_stats["sig_fdr"] = df_stats["sig_fdr"].fillna(False)
    df_stats["trend_fdr"] = df_stats["trend_fdr"].fillna(False)
    df_stats["sig_unc"] = df_stats["sig_unc"].fillna(False)
    df_stats["trend_unc"] = df_stats["trend_unc"].fillna(False)

    return df_stats


# -----------------------------------------------------------------------------
# 2. COMPUTE DELTA STATS
# -----------------------------------------------------------------------------

def compute_delta_stats(df_unadj, df_adj, network_cols, domains, n_boot=1000):
    """
    delta = r_unadj - r_adj
    Significance is from bootstrap CI excluding 0.
    """
    rows = []

    for domain in domains:
        y_u = f"{domain}_target_unadj"
        y_a = f"{domain}_target_adj"

        if y_u not in df_unadj.columns or y_a not in df_adj.columns:
            continue

        for net in network_cols:
            tmp = pd.DataFrame({
                "x": df_unadj[net],
                "y_u": df_unadj[y_u],
                "y_a": df_adj[y_a]
            }).dropna()

            if len(tmp) < MIN_N:
                continue

            try:
                r_u, _ = pearsonr(tmp["x"], tmp["y_u"])
                r_a, _ = pearsonr(tmp["x"], tmp["y_a"])
                delta = r_u - r_a
            except Exception:
                r_u, r_a, delta = np.nan, np.nan, np.nan

            boot_mean, ci95_low, ci95_high, ci90_low, ci90_high = _bootstrap_delta_same_rows(
                df_unadj, df_adj, net, domain, n_boot=n_boot
            )

            sig95 = (
                np.isfinite(ci95_low) and np.isfinite(ci95_high) and
                ((ci95_low > 0 and ci95_high > 0) or (ci95_low < 0 and ci95_high < 0))
            )

            sig90 = (
                np.isfinite(ci90_low) and np.isfinite(ci90_high) and
                ((ci90_low > 0 and ci90_high > 0) or (ci90_low < 0 and ci90_high < 0))
            )

            rows.append({
                "network": net,
                "network_label": NETWORK_LABELS.get(net, net),
                "domain": domain,
                "r_unadj": r_u,
                "r_adj": r_a,
                "delta": delta,
                "boot_mean": boot_mean,
                "ci95_low": ci95_low,
                "ci95_high": ci95_high,
                "ci90_low": ci90_low,
                "ci90_high": ci90_high,
                "sig95": sig95,
                "sig90": sig90,
                "N": len(tmp)
            })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 3. BUILD MATRICES
# -----------------------------------------------------------------------------

def build_matrix(df_long, value_col, domains, network_cols):
    mat = df_long.pivot(index="network_label", columns="domain", values=value_col)

    ordered_rows = _ordered_labeled_networks(network_cols)
    ordered_cols = [d for d in domains if d in mat.columns]

    mat = mat.loc[[r for r in ordered_rows if r in mat.index], ordered_cols]
    return mat


def build_sig_matrices_model(df_stats, domains, network_cols):
    mat_sig = build_matrix(df_stats, "sig_fdr", domains, network_cols)
    mat_trd = build_matrix(df_stats, "trend_fdr", domains, network_cols)
    return mat_sig, mat_trd


def build_sig_matrices_delta(df_delta, domains, network_cols):
    mat_sig = build_matrix(df_delta, "sig95", domains, network_cols)
    mat_trd = build_matrix(df_delta, "sig90", domains, network_cols)
    return mat_sig, mat_trd


# -----------------------------------------------------------------------------
# 4. NETWORK-LEVEL SUMMARIES FOR PANELS D / E / F
# -----------------------------------------------------------------------------

def summarize_network_change(df_delta, network_cols):
    """
    Panel D:
    mean absolute delta across domains
    shading = fraction of domains with 95% or 90% delta significance
    """
    rows = []

    for net_label in _ordered_labeled_networks(network_cols):
        sub = df_delta[df_delta["network_label"] == net_label].copy()
        if len(sub) == 0:
            continue

        mean_abs_delta = sub["delta"].abs().mean()
        frac_sig95 = sub["sig95"].mean()
        frac_sig90 = sub["sig90"].mean()

        rows.append({
            "network_label": net_label,
            "mean_abs_delta": mean_abs_delta,
            "frac_sig95": frac_sig95,
            "frac_sig90": frac_sig90
        })

    return pd.DataFrame(rows)


def summarize_network_effect(df_adj_stats, network_cols):
    """
    Panel E:
    mean absolute adjusted effect across domains
    shading = fraction of domains significant after FDR
    """
    rows = []

    for net_label in _ordered_labeled_networks(network_cols):
        sub = df_adj_stats[df_adj_stats["network_label"] == net_label].copy()
        if len(sub) == 0:
            continue

        mean_abs_effect = sub["r"].abs().mean()
        frac_sig = sub["sig_fdr"].mean()
        frac_trend = sub["trend_fdr"].mean()

        rows.append({
            "network_label": net_label,
            "mean_abs_effect": mean_abs_effect,
            "frac_sig": frac_sig,
            "frac_trend": frac_trend
        })

    return pd.DataFrame(rows)


def compute_network_stability(df_adj, network_cols, domains, n_boot=1000):
    """
    Panel F:
    For each network, bootstrap the mean absolute adjusted effect across domains.
    """
    rows = []

    for net in network_cols:
        net_label = NETWORK_LABELS.get(net, net)
        boot_vals = []

        # check that at least one domain is available
        valid_domains = [d for d in domains if f"{d}_target_adj" in df_adj.columns]
        if len(valid_domains) == 0:
            continue

        # bootstrap over rows per domain independently and average |r|
        for _ in range(n_boot):
            rs = []

            for d in valid_domains:
                ycol = f"{d}_target_adj"

                tmp = df_adj[[net, ycol]].dropna()
                if len(tmp) < MIN_N:
                    continue

                idx = np.random.choice(len(tmp), len(tmp), replace=True)
                xb = tmp[net].values[idx]
                yb = tmp[ycol].values[idx]

                if np.std(xb) == 0 or np.std(yb) == 0:
                    continue

                try:
                    r, _ = pearsonr(xb, yb)
                    if np.isfinite(r):
                        rs.append(abs(r))
                except Exception:
                    continue

            if len(rs) > 0:
                boot_vals.append(np.mean(rs))

        if len(boot_vals) < 50:
            rows.append({
                "network_label": net_label,
                "mean_abs_effect_boot": np.nan,
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "ci90_low": np.nan,
                "ci90_high": np.nan,
                "stability_width": np.nan
            })
            continue

        boot_vals = np.array(boot_vals)

        rows.append({
            "network_label": net_label,
            "mean_abs_effect_boot": np.mean(boot_vals),
            "ci95_low": np.percentile(boot_vals, 2.5),
            "ci95_high": np.percentile(boot_vals, 97.5),
            "ci90_low": np.percentile(boot_vals, 5.0),
            "ci90_high": np.percentile(boot_vals, 95.0),
            "stability_width": np.percentile(boot_vals, 97.5) - np.percentile(boot_vals, 2.5)
        })

    df_stab = pd.DataFrame(rows)

    # preserve requested network order
    df_stab["network_label"] = pd.Categorical(
        df_stab["network_label"],
        categories=_ordered_labeled_networks(network_cols),
        ordered=True
    )
    df_stab = df_stab.sort_values("network_label")

    return df_stab


# -----------------------------------------------------------------------------
# 5. PLOTTING HELPERS
# -----------------------------------------------------------------------------

def annotate_heatmap_sig(ax, mat_sig, mat_trd, symbol_sig="*", symbol_trd="+"):
    for i in range(mat_sig.shape[0]):
        for j in range(mat_sig.shape[1]):

            sig = bool(mat_sig.iloc[i, j]) if pd.notnull(mat_sig.iloc[i, j]) else False
            trd = bool(mat_trd.iloc[i, j]) if pd.notnull(mat_trd.iloc[i, j]) else False

            if sig:
                ax.text(j + 0.5, i + 0.5, symbol_sig,
                        ha="center", va="center", color="black", fontsize=11, fontweight="bold")
            elif trd:
                ax.text(j + 0.5, i + 0.5, symbol_trd,
                        ha="center", va="center", color="black", fontsize=10)

def annotate_heatmap_full(ax, mat_fdr, mat_unc):

    for i in range(mat_fdr.shape[0]):
        for j in range(mat_fdr.shape[1]):

            fdr = bool(mat_fdr.iloc[i, j])
            unc = bool(mat_unc.iloc[i, j])

            if fdr:
                ax.text(j+0.5, i+0.5, "*",
                        ha="center", va="center",
                        fontsize=11, fontweight="bold")

            elif unc:
                ax.text(j+0.5, i+0.5, "+",
                        ha="center", va="center",
                        fontsize=10)
def bar_alpha(sig_main, sig_trend, base=0.30, trend=0.65, strong=1.00):
    """
    Utility for D/E/F:
    stronger shading if stronger support.
    """
    if pd.isnull(sig_main):
        return base
    if sig_main >= 0.5:
        return strong
    if pd.notnull(sig_trend) and sig_trend >= 0.5:
        return trend
    if sig_main > 0 or (pd.notnull(sig_trend) and sig_trend > 0):
        return trend
    return base


#def build_unc_matrix(df_stats, domains, network_cols):
#    return build_matrix(df_stats, "sig_unc", domains, network_cols)

def build_unc_matrix(df_stats, domains, network_cols):
    # "+" = ANY nominal signal (p < 0.10)
    return build_matrix(df_stats, "trend_unc", domains, network_cols)



# -----------------------------------------------------------------------------
# 6. MAIN FUNCTION
# -----------------------------------------------------------------------------

def run_figure4_sixpanel(
    df_unadj,
    df_adj,
    network_cols,
    outdir,
    domains=None,
    n_boot=1000
):
    if domains is None:
        domains = DOMAINS

    os.makedirs(outdir, exist_ok=True)

    print("\nComputing Figure 4 statistics...")

    # -------------------------------------------------------------------------
    # stats
    # -------------------------------------------------------------------------
    stats_unadj = compute_weight_stats_boot(
        df_model=df_unadj,
        suffix="_unadj",
        network_cols=network_cols,
        domains=domains,
        n_boot=n_boot
    )

    stats_adj = compute_weight_stats_boot(
        df_model=df_adj,
        suffix="_adj",
        network_cols=network_cols,
        domains=domains,
        n_boot=n_boot
    )

    delta_stats = compute_delta_stats(
        df_unadj=df_unadj,
        df_adj=df_adj,
        network_cols=network_cols,
        domains=domains,
        n_boot=n_boot
    )

    stability_stats = compute_network_stability(
        df_adj=df_adj,
        network_cols=network_cols,
        domains=domains,
        n_boot=n_boot
    )

    # save long tables
    stats_unadj.to_csv(os.path.join(outdir, "Figure4_stats_unadjusted.tsv"), sep="\t", index=False)
    stats_adj.to_csv(os.path.join(outdir, "Figure4_stats_adjusted.tsv"), sep="\t", index=False)
    delta_stats.to_csv(os.path.join(outdir, "Figure4_stats_delta.tsv"), sep="\t", index=False)
    stability_stats.to_csv(os.path.join(outdir, "Figure4_stats_stability.tsv"), sep="\t", index=False)

    # -------------------------------------------------------------------------
    # matrices
    # -------------------------------------------------------------------------
    mat_u = build_matrix(stats_unadj, "r", domains, network_cols)
    mat_a = build_matrix(stats_adj, "r", domains, network_cols)
    mat_d = build_matrix(delta_stats, "delta", domains, network_cols)

    mat_u_sig, mat_u_trd = build_sig_matrices_model(stats_unadj, domains, network_cols)
    mat_a_sig, mat_a_trd = build_sig_matrices_model(stats_adj, domains, network_cols)
    mat_d_sig, mat_d_trd = build_sig_matrices_delta(delta_stats, domains, network_cols)
    
    mat_u_unc = build_unc_matrix(stats_unadj, domains, network_cols)
    mat_a_unc = build_unc_matrix(stats_adj, domains, network_cols)

    # save matrices
    mat_u.to_csv(os.path.join(outdir, "Figure4A_unadjusted_matrix.tsv"), sep="\t")
    mat_a.to_csv(os.path.join(outdir, "Figure4B_adjusted_matrix.tsv"), sep="\t")
    mat_d.to_csv(os.path.join(outdir, "Figure4C_difference_matrix.tsv"), sep="\t")
    
    mat_u_unc.to_csv(os.path.join(outdir, "Figure4A_unadjusted_unc.tsv"), sep="\t")
    mat_a_unc.to_csv(os.path.join(outdir, "Figure4B_adjusted_unc.tsv"), sep="\t")

    # -------------------------------------------------------------------------
    # network summaries for D/E/F
    # -------------------------------------------------------------------------
    df_change = summarize_network_change(delta_stats, network_cols)
    df_effect = summarize_network_effect(stats_adj, network_cols)

    df_change.to_csv(os.path.join(outdir, "Figure4D_change_summary.tsv"), sep="\t", index=False)
    df_effect.to_csv(os.path.join(outdir, "Figure4E_effect_summary.tsv"), sep="\t", index=False)

    # preserve fixed order
    ordered_labels = _ordered_labeled_networks(network_cols)

    df_change["network_label"] = pd.Categorical(df_change["network_label"],
                                                categories=ordered_labels,
                                                ordered=True)
    df_change = df_change.sort_values("network_label")

    df_effect["network_label"] = pd.Categorical(df_effect["network_label"],
                                                categories=ordered_labels,
                                                ordered=True)
    df_effect = df_effect.sort_values("network_label")

    # -------------------------------------------------------------------------
    # plotting
    # -------------------------------------------------------------------------
    print("Building Figure 4 (six-panel)...")

    fig, axes = plt.subplots(2, 3, figsize=(22, 16))
    axes = axes.flatten()

    # -------------------------------------------------------------------------
    # A. UNADJUSTED
    # -------------------------------------------------------------------------
    ax = axes[0]
    sns.heatmap(
        mat_u,
        cmap="coolwarm",
        center=0,
        ax=ax,
        cbar_kws={"label": "Pearson r"}
    )
    #annotate_heatmap_sig(ax, mat_u_sig, mat_u_trd, symbol_sig="*", symbol_trd="+")
    annotate_heatmap_full(ax, mat_u_sig, mat_u_unc)
    ax.set_title("A. Unadjusted model", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    # -------------------------------------------------------------------------
    # B. ADJUSTED
    # -------------------------------------------------------------------------
    ax = axes[1]
    sns.heatmap(
        mat_a,
        cmap="coolwarm",
        center=0,
        ax=ax,
        cbar_kws={"label": "Pearson r"}
    )
    #annotate_heatmap_sig(ax, mat_a_sig, mat_a_trd, symbol_sig="*", symbol_trd="+")
    annotate_heatmap_full(ax, mat_a_sig, mat_a_unc)
    ax.set_title("B. Adjusted model", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    # -------------------------------------------------------------------------
    # C. DIFFERENCE
    # -------------------------------------------------------------------------
    ax = axes[2]
    sns.heatmap(
        mat_d,
        cmap="bwr",
        center=0,
        ax=ax,
        cbar_kws={"label": "Δr (Unadjusted − Adjusted)"}
    )
    annotate_heatmap_sig(ax, mat_d_sig, mat_d_trd, symbol_sig="*", symbol_trd="+")
    ax.set_title("C. Difference", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    # -------------------------------------------------------------------------
    # D. CHANGE SUMMARY
    # mean |Δ| across domains
    # shading = proportion of domains with bootstrap-supported delta
    # -------------------------------------------------------------------------
    ax = axes[3]

    y = np.arange(len(df_change))
    vals = df_change["mean_abs_delta"].values

    alphas = [
        bar_alpha(row["frac_sig95"], row["frac_sig90"])
        for _, row in df_change.iterrows()
    ]

    for i, (v, a) in enumerate(zip(vals, alphas)):
        ax.barh(i, v, alpha=a)

    # stars / plus at bar ends
    for i, row in enumerate(df_change.itertuples()):
        if row.frac_sig95 >= 0.5:
            ax.text(row.mean_abs_delta + 0.005, i, "*", va="center", fontsize=12, fontweight="bold")
        elif row.frac_sig90 >= 0.5:
            ax.text(row.mean_abs_delta + 0.005, i, "+", va="center", fontsize=11)

    ax.set_yticks(y)
    ax.set_yticklabels(df_change["network_label"])
    ax.invert_yaxis()
    ax.set_title("D. Change summary", fontsize=14)
    ax.set_xlabel("Mean |Δr| across domains")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.2)

    # -------------------------------------------------------------------------
    # E. EFFECT SUMMARY
    # mean |adjusted effect| across domains
    # shading = fraction of FDR-significant domains
    # -------------------------------------------------------------------------
    ax = axes[4]

    y = np.arange(len(df_effect))
    vals = df_effect["mean_abs_effect"].values

    alphas = [
        bar_alpha(row["frac_sig"], row["frac_trend"])
        for _, row in df_effect.iterrows()
    ]

    for i, (v, a) in enumerate(zip(vals, alphas)):
        ax.barh(i, v, alpha=a)

    for i, row in enumerate(df_effect.itertuples()):
        if row.frac_sig >= 0.5:
            ax.text(row.mean_abs_effect + 0.005, i, "*", va="center", fontsize=12, fontweight="bold")
        elif row.frac_trend >= 0.5:
            ax.text(row.mean_abs_effect + 0.005, i, "+", va="center", fontsize=11)

    ax.set_yticks(y)
    ax.set_yticklabels(df_effect["network_label"])
    ax.invert_yaxis()
    ax.set_title("E. Effect summary", fontsize=14)
    ax.set_xlabel("Mean |adjusted r| across domains")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.2)

    # -------------------------------------------------------------------------
    # F. STABILITY SUMMARY
    # bootstrap forest plot of mean |adjusted r|
    # shading based on CI width (narrower = more stable)
    # -------------------------------------------------------------------------
    ax = axes[5]

    df_stab = stability_stats.copy()

    # convert CI width to alpha: narrower intervals = darker
    widths = df_stab["stability_width"].values.astype(float)
    finite_mask = np.isfinite(widths)

    if finite_mask.sum() > 1:
        wmin = np.nanmin(widths)
        wmax = np.nanmax(widths)
        if wmax > wmin:
            stability_alpha = 1.0 - 0.7 * ((widths - wmin) / (wmax - wmin))
        else:
            stability_alpha = np.repeat(0.85, len(widths))
    else:
        stability_alpha = np.repeat(0.85, len(widths))

    stability_alpha = np.where(np.isfinite(stability_alpha), stability_alpha, 0.35)

    y = np.arange(len(df_stab))

    for i, row in enumerate(df_stab.itertuples()):
        mean_v = row.mean_abs_effect_boot
        lo = row.ci95_low
        hi = row.ci95_high
        a = stability_alpha[i]

        if np.isfinite(mean_v) and np.isfinite(lo) and np.isfinite(hi):
            ax.errorbar(
                mean_v,
                i,
                xerr=[[mean_v - lo], [hi - mean_v]],
                fmt='o',
                capsize=3,
                alpha=a
            )

            # star/plus for CI support relative to zero
            if np.isfinite(row.ci95_low) and row.ci95_low > 0:
                ax.text(hi + 0.005, i, "*", va="center", fontsize=12, fontweight="bold")
            elif np.isfinite(row.ci90_low) and row.ci90_low > 0:
                ax.text(hi + 0.005, i, "+", va="center", fontsize=11)

    ax.set_yticks(y)
    ax.set_yticklabels(df_stab["network_label"])
    ax.invert_yaxis()
    ax.set_title("F. Stability summary", fontsize=14)
    ax.set_xlabel("Bootstrap mean |adjusted r| ± 95% CI")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.2)

    # overall layout
    plt.tight_layout()

    fig_path = os.path.join(outdir, "Figure4_sixpanel.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {fig_path}")

    return {
        "stats_unadj": stats_unadj,
        "stats_adj": stats_adj,
        "delta_stats": delta_stats,
        "stability_stats": stability_stats,
        "mat_unadj": mat_u,
        "mat_adj": mat_a,
        "mat_diff": mat_d,
        "change_summary": df_change,
        "effect_summary": df_effect
    }


# =============================================================================
# RUN FIGURE 4
# =============================================================================

figure4_results = run_figure4_sixpanel(
    df_unadj=df_unadj,
    df_adj=df_adj,
    network_cols=network_cols,
    outdir=OUTDIR,
    domains=DOMAINS,
    n_boot=N_BOOT
)



#### MASTER SHEET ######

#### MASTER SHEET ######

stats_unadj = figure4_results["stats_unadj"]
stats_adj   = figure4_results["stats_adj"]
delta_stats = figure4_results["delta_stats"]

# ------------------------------------------------------------------
# Start from adjusted model (PRIMARY inference layer)
# ------------------------------------------------------------------
df_master = stats_adj.copy()

# rename adjusted stats (make inference explicit)
df_master = df_master.rename(columns={
    "r": "r_adj",
    "p": "p_adj",
    "q": "q_adj",
    "sig_fdr": "sig_fdr_adj",
    "trend_fdr": "trend_fdr_adj",
    "sig_unc": "sig_unc_adj",
    "trend_unc": "trend_unc_adj"
})

# ------------------------------------------------------------------
# Add UNADJUSTED stats (context layer)
# ------------------------------------------------------------------

# r_unadj
df_master = df_master.merge(
    stats_unadj[["network_label","domain","r"]]
        .rename(columns={"r": "r_unadj"}),
    on=["network_label","domain"],
    how="left"
)

# p_unadj
df_master = df_master.merge(
    stats_unadj[["network_label","domain","p"]]
        .rename(columns={"p": "p_unadj"}),
    on=["network_label","domain"],
    how="left"
)

# optional: unadjusted significance flags
df_master = df_master.merge(
    stats_unadj[[
        "network_label","domain",
        "sig_unc","trend_unc"
    ]].rename(columns={
        "sig_unc": "sig_unc_unadj",
        "trend_unc": "trend_unc_unadj"
    }),
    on=["network_label","domain"],
    how="left"
)

# ------------------------------------------------------------------
# Add DELTA stats (Panel C: shift between models)
# ------------------------------------------------------------------
df_master = df_master.merge(
    delta_stats[[
        "network_label", "domain",
        "delta",
        "ci95_low", "ci95_high",
        "ci90_low", "ci90_high",
        "sig95", "sig90",
        "boot_mean"
    ]].rename(columns={
        "ci95_low": "ci95_low_delta",
        "ci95_high": "ci95_high_delta",
        "ci90_low": "ci90_low_delta",
        "ci90_high": "ci90_high_delta",
        "sig95": "sig95_delta",
        "sig90": "sig90_delta"
    }),
    on=["network_label","domain"],
    how="left"
)

# ------------------------------------------------------------------
# (Optional but HIGHLY recommended) Column ordering
# ------------------------------------------------------------------
cols_order = [
    "network", "network_label", "domain",

    # unadjusted
    "r_unadj", "p_unadj",
    "sig_unc_unadj", "trend_unc_unadj",

    # adjusted (PRIMARY)
    "r_adj", "p_adj", "q_adj",
    "sig_fdr_adj", "trend_fdr_adj",

    # delta (mechanism)
    "delta",
    "boot_mean",
    "ci95_low_delta", "ci95_high_delta",
    "ci90_low_delta", "ci90_high_delta",
    "sig95_delta", "sig90_delta",

    # misc
    "N"
]

# keep only existing columns (safe)
cols_order = [c for c in cols_order if c in df_master.columns]
df_master = df_master[cols_order]

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------
df_master.to_csv(
    os.path.join(OUTDIR, "Figure4_MASTER_table.tsv"),
    sep="\t",
    index=False
)

print("Saved: Figure4_MASTER_table.tsv")
print(df_master.columns)

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------
df_master.to_csv(
    os.path.join(OUTDIR, "Figure4_MASTER_table.tsv"),
    sep="\t",
    index=False
)

print("Saved: Figure4_MASTER_table.tsv")

df_master.to_csv(
    os.path.join(OUTDIR, "Figure4_MASTER_table.tsv"),
    sep="\t",
    index=False
)


print(delta_stats.columns)
print("Saved: Figure4_MASTER_table.tsv")


df_master["class"] = "neutral"

# robust
df_master.loc[
    (df_master["q_adj"] < 0.05) &
    (~df_master["sig95_delta"]),
    "class"
] = "robust"

# confounded
df_master.loc[
    (df_master["p_unadj"] < 0.05) &
    (df_master["q_adj"] > 0.1) &
    (df_master["sig95_delta"]),
    "class"
] = "confounded"

# emergent
df_master.loc[
    (df_master["q_adj"] < 0.05) &
    (df_master["sig95_delta"]),
    "class"
] = "emergent"


df_master["abs_delta"] = df_master["delta"].abs()
df_master[df_master["class"] != "neutral"].sort_values("abs_delta", ascending=False)
outdir_groups = os.path.join(OUTDIR, "Figure4_groups")
os.makedirs(outdir_groups, exist_ok=True)

# subsets
df_robust     = df_master[df_master["class"] == "robust"]
df_confounded = df_master[df_master["class"] == "confounded"]
df_emergent   = df_master[df_master["class"] == "emergent"]
df_neutral    = df_master[df_master["class"] == "neutral"]

# save
df_robust.to_csv(os.path.join(outdir_groups, "robust.tsv"), sep="\t", index=False)
df_confounded.to_csv(os.path.join(outdir_groups, "confounded.tsv"), sep="\t", index=False)
df_emergent.to_csv(os.path.join(outdir_groups, "emergent.tsv"), sep="\t", index=False)
df_neutral.to_csv(os.path.join(outdir_groups, "neutral.tsv"), sep="\t", index=False)

print("Saved group tables")

df_non_neutral = df_master[df_master["class"] != "neutral"]

df_ranked = df_non_neutral.sort_values("abs_delta", ascending=False)

df_ranked.to_csv(
    os.path.join(outdir_groups, "ranked_non_neutral_by_delta.tsv"),
    sep="\t",
    index=False
)

print("Saved ranked networks by delta")

# Show non-neutral networks clearly
df_key = df_master[df_master["class"] != "neutral"].copy()

df_key = df_key.sort_values("abs_delta", ascending=False)

print(df_key[[
    "network_label",
    "domain",
    "class",
    "r_unadj",
    "r_adj",
    "delta",
    "q_adj"
]])

df_network_view = (
    df_key.groupby("network_label")
    .agg({
        "class": lambda x: list(x),
        "domain": lambda x: list(x),
        "delta": "mean",
        "abs_delta": "max"
    })
    .reset_index()
)

print(df_network_view)


df_master["abs_delta"].mean()
df_master["abs_delta"].max()

df_master["abs_delta"].mean()