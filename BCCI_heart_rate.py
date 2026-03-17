#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ======================================================
# 1. IMPORTS
# ======================================================

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut


# ======================================================
# 2. PATHS
# ======================================================

BASE = "/mnt/newStor/paros/paros_WORK/aashika"

META_FILE = os.path.join(
    BASE,
    "data/metadata/cardiac_design_updated3.csv"
)

NETWORK_FILE = os.path.join(
    BASE,
    "results/march16_2026/network_amplitudes_matrix_COPY.tsv"
)

OUTDIR = os.path.join(
    BASE,
    "results/march16_2026"
)

os.makedirs(OUTDIR, exist_ok=True)


# ======================================================
# 3. LOAD + MERGE (ROBUST VERSION)
# ======================================================

print("\nLoading files...")

meta = pd.read_csv(META_FILE)
nets = pd.read_csv(NETWORK_FILE, sep="\t")

# ---- Standardize ID column ----
if "subject" in nets.columns:
    nets = nets.rename(columns={"subject": "Arunno"})

if "Arunno" not in meta.columns or "Arunno" not in nets.columns:
    raise ValueError("Arunno column missing in one of the files")

# Clean IDs
meta["Arunno"] = meta["Arunno"].astype(str).str.strip()
nets["Arunno"] = nets["Arunno"].astype(str).str.strip()

# ---- Debug before merge ----
print("\nUnique IDs:")
print("Meta:", meta["Arunno"].nunique())
print("Nets:", nets["Arunno"].nunique())

common_ids = set(meta["Arunno"]).intersection(set(nets["Arunno"]))
print("Common IDs:", len(common_ids))

# ---- Merge ----
df = pd.merge(meta, nets, on="Arunno", how="inner")

print("\nMerged shape:", df.shape)

# ---- Post-merge checks ----
print("\nMissing values (top 10):")
print(df.isna().sum().sort_values(ascending=False).head(10))

if "Genotype" in df.columns:
    print("\nGenotype distribution:")
    print(df["Genotype"].value_counts())

# Save merged dataset
df.to_csv(os.path.join(OUTDIR, "merged_dataset.tsv"), sep="\t", index=False)


# ======================================================
# 4. GENOTYPE GROUP
# ======================================================

geno_map = {
    "E22": "E2", "E2HN": "E2",
    "E33": "E3", "E3HN": "E3",
    "E44": "E4", "E4HN": "E4",
    "KO": "KO"
}

df["Genotype_group"] = df["Genotype"].map(geno_map)

# Remove KO (optional but recommended)
df = df[df["Genotype_group"].isin(["E2", "E3", "E4"])]

df["Genotype_group"] = df["Genotype_group"].astype("category")

print("\nAfter genotype filtering:", df.shape)


# ======================================================
# 5. STANDARDIZE NETWORKS
# ======================================================

network_cols = [c for c in df.columns if c.startswith("Amp_Net")]

print("Networks detected:", len(network_cols))

scaler = StandardScaler()
df[network_cols] = scaler.fit_transform(df[network_cols])


# ======================================================
# 6. RESIDUALIZE HEART RATE
# ======================================================

HR_COL = "Heart_Rate"

covars = ["Age", "Sex_Male", "Mass"]
covars = [c for c in covars if c in df.columns]

tmp = df[[HR_COL] + covars].dropna()

X = sm.add_constant(tmp[covars])
model = sm.OLS(tmp[HR_COL], X).fit()

df.loc[tmp.index, "HR_resid"] = model.resid

print("\nHeart rate residualization complete")
print(model.summary())


# ======================================================
# 7. NETWORK → HR SCREENING
# ======================================================

results = []

for net_col in network_cols:

    tmp = df[[net_col, "HR_resid"]].dropna()

    if len(tmp) < 20:
        continue

    r, p = pearsonr(tmp[net_col], tmp["HR_resid"])

    results.append({
        "network": net_col,
        "r": r,
        "p": p
    })

res_df = pd.DataFrame(results).sort_values("r")

res_df.to_csv(os.path.join(OUTDIR, "network_hr_correlations.tsv"), sep="\t", index=False)


# ======================================================
# 8. DEFINE HR- / HR+ NETWORKS
# ======================================================

THRESH = 0.05

HR_neg_nets = res_df[res_df["r"] < -THRESH]["network"].tolist()
HR_pos_nets = res_df[res_df["r"] >  THRESH]["network"].tolist()

print("\nHR- networks:", HR_neg_nets)
print("HR+ networks:", HR_pos_nets)


# ======================================================
# 9. BUILD BCCI
# ======================================================

df["HR_minus_index"] = df[HR_neg_nets].mean(axis=1)
df["HR_plus_index"]  = df[HR_pos_nets].mean(axis=1)

df["BCCI"] = df["HR_minus_index"] - df["HR_plus_index"]


# ======================================================
# 10. GLOBAL ASSOCIATION
# ======================================================

tmp = df[["BCCI", "HR_resid"]].dropna()

rho, pval = spearmanr(tmp["BCCI"], tmp["HR_resid"])

X = sm.add_constant(tmp["BCCI"])
model_bcci = sm.OLS(tmp["HR_resid"], X).fit()

print("\n===== BCCI RESULT =====")
print("Spearman rho =", round(rho,3), "p =", round(pval,4))
print("R² =", round(model_bcci.rsquared,3))


# ======================================================
# 11. GENOTYPE MODEL
# ======================================================

formula = """
HR_resid ~ BCCI
+ C(Genotype_group, Treatment(reference='E3'))
+ BCCI:C(Genotype_group, Treatment(reference='E3'))
"""

model_geno = smf.ols(formula, data=df).fit()

print("\n===== GENOTYPE MODEL =====")
print(model_geno.summary())


# ======================================================
# 12. WITHIN-GENOTYPE EFFECTS
# ======================================================

for g in ["E2", "E3", "E4"]:

    sub = df[df["Genotype_group"] == g]
    tmp = sub[["BCCI", "HR_resid"]].dropna()

    if len(tmp) < 10:
        continue

    r, p = spearmanr(tmp["BCCI"], tmp["HR_resid"])

    print(f"{g}: rho={r:.3f}, p={p:.4f}, N={len(tmp)}")


# ======================================================
# 13. LOOCV STABILITY
# ======================================================

loo = LeaveOneOut()

betas = []

tmp = df[["BCCI", "HR_resid"]].dropna()

X = tmp["BCCI"].values
y = tmp["HR_resid"].values

for train_idx, test_idx in loo.split(X):

    X_train = sm.add_constant(X[train_idx])
    y_train = y[train_idx]

    model = sm.OLS(y_train, X_train).fit()
    betas.append(model.params[1])

print("\nLOOCV beta mean:", np.mean(betas))
print("LOOCV beta std:", np.std(betas))


# ======================================================
# 14. FIGURE — BCCI × GENOTYPE
# ======================================================

sns.lmplot(
    data=df,
    x="BCCI",
    y="HR_resid",
    hue="Genotype_group",
    height=5,
    aspect=1.2,
    scatter_kws={"s":50}
)

plt.title("Brain–Cardiac Coupling by Genotype")
plt.savefig(os.path.join(OUTDIR, "BCCI_by_genotype.png"), dpi=300)
plt.show()


# ======================================================
# 15. HEATMAP
# ======================================================

heatmap_data = res_df.set_index("network")[["r"]]

plt.figure(figsize=(4,6))
sns.heatmap(
    heatmap_data,
    cmap="coolwarm",
    center=0,
    annot=True
)

plt.title("Network–Heart Rate Coupling")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "network_heatmap.png"), dpi=300)
plt.show()


print("\nDONE — FULL PIPELINE COMPLETE")
