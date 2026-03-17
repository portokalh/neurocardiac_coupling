#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =========================
# 1. IMPORTS
# =========================

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut


# =========================
# 2. PATHS
# =========================

BASE = "/mnt/newStor/paros/paros_WORK/aashika"

META_FILE = os.path.join(BASE, "data/metadata/cardiac_design_updated3.csv")
NETWORK_FILE = os.path.join(BASE, "results/march16_2026/network_amplitudes_matrix_COPY.tsv")
OUTDIR = os.path.join(BASE, "results/march16_2026")

os.makedirs(OUTDIR, exist_ok=True)


# =========================
# 3. LOAD + MERGE
# =========================

print("\nLoading files...")

meta = pd.read_csv(META_FILE)
nets = pd.read_csv(NETWORK_FILE, sep="\t")

if "subject" in nets.columns:
    nets = nets.rename(columns={"subject": "Arunno"})

meta["Arunno"] = meta["Arunno"].astype(str).str.strip()
nets["Arunno"] = nets["Arunno"].astype(str).str.strip()

df = pd.merge(meta, nets, on="Arunno", how="inner")

print("Merged shape:", df.shape)


# =========================
# 4. GENOTYPE GROUP
# =========================

geno_map = {
    "E22": "E2", "E2HN": "E2",
    "E33": "E3", "E3HN": "E3",
    "E44": "E4", "E4HN": "E4",
    "KO": "KO"
}

df["Genotype_group"] = df["Genotype"].map(geno_map)
df = df[df["Genotype_group"].isin(["E2", "E3", "E4"])]
df["Genotype_group"] = df["Genotype_group"].astype("category")


# =========================
# 5. STANDARDIZE NETWORKS
# =========================

network_cols = [c for c in df.columns if c.startswith("Amp_Net")]

scaler = StandardScaler()
df[network_cols] = scaler.fit_transform(df[network_cols])

print("Networks detected:", len(network_cols))


# =========================
# 6. CARDIAC DOMAINS
# =========================

CARDIAC_GROUPS = {
    "rate_control": ["Heart_Rate"],
    "systolic_function": [
        "Stroke_Volume", "Cardiac_Output", "Ejection_Fraction",
        "Systolic_LV_Volume", "Systolic_RV", "Systolic_LA",
        "Systolic_RA", "Systolic_Myo"
    ],
    "diastolic_function": [
        "Diastolic_LV_Volume", "Diastolic_RV",
        "Diastolic_LA", "Diastolic_RA", "Diastolic_Myo"
    ]
}


# =========================
# 7. CREATE DOMAIN SCORES
# =========================

scaler_dom = StandardScaler()

for group, metrics in CARDIAC_GROUPS.items():

    valid_metrics = [m for m in metrics if m in df.columns]

    if len(valid_metrics) == 0:
        print(f"{group}: no valid metrics")
        continue

    tmp = df[valid_metrics].dropna()

    if len(tmp) < 20:
        print(f"{group}: insufficient data (N={len(tmp)})")
        continue

    zvals = scaler_dom.fit_transform(tmp)

    df.loc[tmp.index, f"{group}_z"] = zvals.mean(axis=1)

    print(f"{group}: created (N={len(tmp)})")


# =========================
# 8. RESIDUALIZE DOMAINS
# =========================

covars = ["Age", "Sex_Male", "Mass"]
covars = [c for c in covars if c in df.columns]

for group in CARDIAC_GROUPS.keys():

    col = f"{group}_z"

    if col not in df.columns:
        continue

    tmp = df[[col] + covars].dropna()

    if len(tmp) < 20:
        print(f"{group}: insufficient for residualization")
        continue

    X = sm.add_constant(tmp[covars])
    model = sm.OLS(tmp[col], X).fit()

    df.loc[tmp.index, f"{group}_z_resid"] = model.resid

    print(f"{group}: residualized")


# =========================
# 9. BUILD BCCI PER DOMAIN
# =========================

domain_stats = {}

for group in CARDIAC_GROUPS.keys():

    y = f"{group}_z_resid"

    if y not in df.columns:
        print(f"{group}: missing residual")
        continue

    results = []

    for net in network_cols:

        tmp = df[[net, y]].dropna()

        if len(tmp) < 20:
            continue

        r, _ = pearsonr(tmp[net], tmp[y])
        results.append({"network": net, "r": r})

    res_df = pd.DataFrame(results)

    if res_df.empty:
        continue

    neg = res_df[res_df["r"] < -0.05]["network"].tolist()
    pos = res_df[res_df["r"] >  0.05]["network"].tolist()

    if len(neg) == 0 or len(pos) == 0:
        print(f"{group}: insufficient network split")
        continue

    df[f"{group}_BCCI"] = df[neg].mean(axis=1) - df[pos].mean(axis=1)

    domain_stats[group] = res_df

    print(f"\n{group}")
    print("Negative networks:", neg)
    print("Positive networks:", pos)


# =========================
# 10. TEST EACH DOMAIN
# =========================

summary = []

for group in CARDIAC_GROUPS.keys():

    x = f"{group}_BCCI"
    y = f"{group}_z_resid"

    if x not in df.columns or y not in df.columns:
        continue

    tmp = df[[x, y]].dropna()

    rho, p = spearmanr(tmp[x], tmp[y])

    X = sm.add_constant(tmp[x])
    model = sm.OLS(tmp[y], X).fit()

    print(f"\n==== {group.upper()} ====")
    print(f"rho = {rho:.3f}, p = {p:.4f}")
    print(f"R² = {model.rsquared:.3f}")

    summary.append({
        "Domain": group,
        "rho": rho,
        "p": p,
        "R2": model.rsquared
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(OUTDIR, "domain_summary.tsv"), sep="\t", index=False)


# =========================
# 11. LOOCV (RATE ONLY)
# =========================

if "rate_control_BCCI" in df.columns:

    loo = LeaveOneOut()
    betas = []

    tmp = df[["rate_control_BCCI", "rate_control_z_resid"]].dropna()

    X = tmp["rate_control_BCCI"].values
    y = tmp["rate_control_z_resid"].values

    for train_idx, _ in loo.split(X):

        X_train = sm.add_constant(X[train_idx])
        y_train = y[train_idx]

        model = sm.OLS(y_train, X_train).fit()
        betas.append(model.params[1])

    print("\nLOOCV beta mean:", np.mean(betas))
    print("LOOCV beta std:", np.std(betas))


# =========================
# 12. FIGURE
# =========================

plt.figure(figsize=(6, 5))

for group in CARDIAC_GROUPS.keys():

    x = f"{group}_BCCI"
    y = f"{group}_z_resid"

    if x not in df.columns or y not in df.columns:
        continue

    sns.regplot(
        data=df,
        x=x,
        y=y,
        label=group,
        scatter=False
    )

plt.legend()
plt.title("BCCI across cardiac domains")
plt.tight_layout()

plt.savefig(os.path.join(OUTDIR, "BCCI_domains.png"), dpi=300)
plt.show()

print("\nDONE — DOMAIN PIPELINE COMPLETE")



# ======================================================
# 9A. PCA DOMAIN SCORES (BETTER CARDIAC REPRESENTATION)
# ======================================================

from sklearn.decomposition import PCA

for group, metrics in CARDIAC_GROUPS.items():

    valid = [m for m in metrics if m in df.columns]

    if len(valid) < 2:
        continue

    tmp = df[valid].dropna()

    if len(tmp) < 20:
        continue

    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(tmp)

    df.loc[tmp.index, f"{group}_pc1"] = pc1[:, 0]

    print(f"{group}: PCA explained variance = {pca.explained_variance_ratio_[0]:.2f}")


# ======================================================
# 9B. RESIDUALIZE PCA SCORES
# ======================================================

for group in CARDIAC_GROUPS.keys():

    col = f"{group}_pc1"

    if col not in df.columns:
        continue

    tmp = df[[col] + covars].dropna()

    if len(tmp) < 20:
        continue

    X = sm.add_constant(tmp[covars])
    model = sm.OLS(tmp[col], X).fit()

    df.loc[tmp.index, f"{group}_pc1_resid"] = model.resid


# ======================================================
# 9C. WEIGHTED BCCI (KEY UPGRADE)
# ======================================================

weighted_results = {}

for group in CARDIAC_GROUPS.keys():

    y = f"{group}_z_resid"

    if y not in df.columns:
        continue

    weights = {}

    for net in network_cols:

        tmp = df[[net, y]].dropna()

        if len(tmp) < 20:
            continue

        r, _ = pearsonr(tmp[net], tmp[y])
        weights[net] = r

    if len(weights) == 0:
        continue

    # build weighted index
    df[f"{group}_BCCI_weighted"] = 0

    for net, w in weights.items():
        df[f"{group}_BCCI_weighted"] += df[net] * w

    weighted_results[group] = weights

    print(f"{group}: weighted BCCI built ({len(weights)} networks)")


# ======================================================
# 10. COMPARE ALL MODELS
# ======================================================

print("\n\n===== MODEL COMPARISON =====")

summary_all = []

for group in CARDIAC_GROUPS.keys():

    print(f"\n--- {group.upper()} ---")

    models = {
        "split": (f"{group}_BCCI", f"{group}_z_resid"),
        "weighted": (f"{group}_BCCI_weighted", f"{group}_z_resid"),
        "pca": (f"{group}_BCCI_weighted", f"{group}_pc1_resid"),
    }

    for name, (x, y) in models.items():

        if x not in df.columns or y not in df.columns:
            continue

        tmp = df[[x, y]].dropna()

        if len(tmp) < 20:
            continue

        rho, p = spearmanr(tmp[x], tmp[y])

        X = sm.add_constant(tmp[x])
        model = sm.OLS(tmp[y], X).fit()

        print(f"{name}: rho={rho:.3f}, p={p:.4f}, R²={model.rsquared:.3f}")

        summary_all.append({
            "Domain": group,
            "Model": name,
            "rho": rho,
            "p": p,
            "R2": model.rsquared
        })


summary_all = pd.DataFrame(summary_all)
summary_all.to_csv(os.path.join(OUTDIR, "domain_model_comparison.tsv"), sep="\t", index=False)
