#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 15:22:12 2026

@author: alex
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

# Optional for clustering
from scipy.cluster.hierarchy import linkage, leaves_list

# Optional for brain maps
BRAINMAPS_ENABLED = False
ICA_4D_FILE = None  # e.g. "/mnt/.../melodic_IC.nii.gz"

try:
    import nibabel as nib
except Exception:
    nib = None

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

# ======================================================
# PATHS
# ======================================================

#IN_RESULTS = "/mnt/newStor/paros/paros_WORK/aashika/results/network_cardiac_stats/network_cardiac_results_hierFDR.tsv"
IN_RESULTS = "/mnt/newStor/paros/paros_WORK/aashika/results/network_cardiac_stats/network_cardiac_results.tsv"

NETWORK_FILE = "/mnt/newStor/paros/paros_WORK/aashika/results/network12_from_errts/Network12_amplitudes.tsv"

META_FILE = "/mnt/newStor/paros/paros_WORK/aashika/data/metadata/cardiac_design_updated3.csv"

OUTDIR = "/mnt/newStor/paros/paros_WORK/aashika/results/network_cardiac_postproc"

os.makedirs(OUTDIR, exist_ok=True)

# ======================================================
# LOAD DATA
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
# GROUP DEFINITIONS
# ======================================================

CARDIAC_GROUPS = {
    "pump_function": ["Heart_Rate", "Stroke_Volume", "Ejection_Fraction", "Cardiac_Output"],
    "left_ventricle": ["Diastolic_LV_Volume", "Systolic_LV_Volume"],
    "right_ventricle": ["Diastolic_RV", "Systolic_RV"],
    "atria": ["Diastolic_LA", "Systolic_LA", "Diastolic_RA", "Systolic_RA"],
    "myocardium": ["Diastolic_Myo", "Systolic_Myo"],
}

metric_to_group = {m: g for g, ms in CARDIAC_GROUPS.items() for m in ms}

# ======================================================
# HELPERS
# ======================================================

def simes_p(pvals):
    p = np.sort(np.asarray(pvals))
    m = len(p)
    if m == 0:
        return np.nan
    return np.min((m * p) / (np.arange(1, m + 1)))

# ======================================================
# LOAD RESULTS
# ======================================================

res = pd.read_csv(IN_RESULTS, sep="\t")
res["Group"] = res["CardiacMetric"].map(metric_to_group)

if res["Group"].isna().any():
    missing_metrics = sorted(res.loc[res["Group"].isna(), "CardiacMetric"].unique().tolist())
    if len(missing_metrics) > 0:
        raise ValueError(f"Some metrics are not assigned to a group: {missing_metrics}")

# ======================================================
# HIERARCHICAL FDR
# ======================================================

# Stage 1: group-level p-value using Simes across all tests in the group
group_p = (
    res.groupby("Group")["P"]
      .apply(lambda s: simes_p(s.dropna().values))
      .reset_index()
      .rename(columns={"P": "Group_P"})
)

group_p["Group_FDR"] = multipletests(group_p["Group_P"].values, method="fdr_bh")[1]
group_p = group_p.sort_values("Group_FDR")

group_p.to_csv(os.path.join(OUTDIR, "group_level_simes_fdr.tsv"), sep="\t", index=False)

# Stage 2: within each group, BH-FDR across all tests
res["FDR_within_group"] = np.nan
for g in res["Group"].unique():
    idx = res["Group"] == g
    res.loc[idx, "FDR_within_group"] = multipletests(res.loc[idx, "P"].values, method="fdr_bh")[1]

# Merge group stats
res = res.merge(group_p, on="Group", how="left")

# Hierarchical significance
res["Sig_hier"] = (res["Group_FDR"] < 0.05) & (res["FDR_within_group"] < 0.05)

res_out = os.path.join(OUTDIR, "network_cardiac_results_hierFDR.tsv")
res.sort_values(["Group_FDR", "FDR_within_group", "P"]).to_csv(res_out, sep="\t", index=False)

print("Saved hierarchical results:", res_out)

# ======================================================
# HEATMAPS
# ======================================================

# -log10(p) heatmap
mlogp = res.pivot_table(index="Network", columns="CardiacMetric", values="P", aggfunc="min")
mlogp = -np.log10(mlogp)

# Beta heatmap
beta = res.pivot_table(index="Network", columns="CardiacMetric", values="Beta", aggfunc="mean")

# Mask by hierarchical significance (optional)
sigmask = res.pivot_table(index="Network", columns="CardiacMetric", values="Sig_hier", aggfunc="max")
mlogp_masked = mlogp.where(sigmask, other=0)

def save_heatmap(mat, title, outname):
    mat_f = mat.fillna(0)
    plt.figure(figsize=(14, 8))
    plt.imshow(mat_f.values, aspect="auto")
    plt.xticks(range(mat_f.shape[1]), mat_f.columns, rotation=90)
    plt.yticks(range(mat_f.shape[0]), mat_f.index)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, outname), dpi=300)
    plt.close()

save_heatmap(mlogp, "Network–Cardiac (-log10 p)", "heatmap_network_cardiac_-log10p.png")
save_heatmap(mlogp_masked, "Network–Cardiac (-log10 p), masked by hierarchical FDR", "heatmap_network_cardiac_-log10p_hier_masked.png")
save_heatmap(beta, "Network–Cardiac (beta)", "heatmap_network_cardiac_beta.png")

# ======================================================
# CLUSTERING NETWORKS BY CARDIAC PROFILE
# ======================================================

# Cluster using beta profiles across metrics
X = beta.fillna(0).values
Z = linkage(X, method="ward")
order = leaves_list(Z)

beta_ord = beta.iloc[order, :]
mlogp_ord = mlogp.iloc[order, :]
mlogp_masked_ord = mlogp_masked.iloc[order, :]

save_heatmap(beta_ord, "Network–Cardiac (beta), clustered networks", "heatmap_beta_clustered_networks.png")
save_heatmap(mlogp_ord, "Network–Cardiac (-log10 p), clustered networks", "heatmap_-log10p_clustered_networks.png")
save_heatmap(mlogp_masked_ord, "Network–Cardiac (-log10 p) masked, clustered networks", "heatmap_-log10p_hier_masked_clustered_networks.png")

# Simple “cluster detection” via correlation threshold on beta profiles
corr = np.corrcoef(beta.fillna(0).values)
thr = 0.6
adj = (corr > thr).astype(int)
np.fill_diagonal(adj, 0)

visited = set()
components = []

for i in range(adj.shape[0]):
    if i in visited:
        continue
    stack = [i]
    comp = []
    while stack:
        u = stack.pop()
        if u in visited:
            continue
        visited.add(u)
        comp.append(u)
        nbrs = np.where(adj[u] == 1)[0].tolist()
        stack.extend(nbrs)
    if len(comp) >= 2:
        components.append(comp)

network_names = beta.index.tolist()
components_named = [[network_names[i] for i in comp] for comp in components]

# Save clusters
clusters_out = os.path.join(OUTDIR, "network_clusters_by_cardiac_profile_thr0p6.tsv")
with open(clusters_out, "w") as f:
    f.write("ClusterID\tNetworks\n")
    for k, comp in enumerate(components_named, 1):
        f.write(f"{k}\t{','.join(comp)}\n")

print("Saved network clusters:", clusters_out)



# ======================================================
# NETWORK CLUSTERING DENDROGRAM
# ======================================================

from scipy.cluster.hierarchy import dendrogram, linkage

# use beta matrix (network x cardiac metric)
X = beta.fillna(0).values

Z = linkage(X, method="ward")

plt.figure(figsize=(10,6))

dendrogram(
    Z,
    labels=beta.index.tolist(),
    leaf_rotation=90
)

plt.title("Clustering of brain networks by cardiac association profile")

plt.ylabel("Distance")

plt.tight_layout()

fig_out = os.path.join(OUTDIR, "network_cardiac_dendrogram.png")

plt.savefig(fig_out, dpi=300)

plt.close()

print("Saved dendrogram:", fig_out)

# ======================================================
# OPTIONAL: BRAIN MAPS (WEIGHTED CIRCUIT MAPS)
# ======================================================

if BRAINMAPS_ENABLED:
    if nib is None:
        raise RuntimeError("nibabel not available but BRAINMAPS_ENABLED=True")
    if ICA_4D_FILE is None or not os.path.exists(ICA_4D_FILE):
        raise RuntimeError("Set ICA_4D_FILE to an existing 4D ICA NIfTI file")

    ica_img = nib.load(ICA_4D_FILE)
    ica_data = ica_img.get_fdata()
    n_vols = ica_data.shape[3]

    # Build mapping from network name to ICA volume index
    # Assumes network columns look like Amp_ICA01, Amp_ICA02, ...
    def net_to_index(net_name):
        # Amp_ICA01 -> 0
        s = net_name.replace("Amp_", "").replace("ICA", "")
        return int(s) - 1

    for metric in sorted(res["CardiacMetric"].unique()):
        sub = res[(res["CardiacMetric"] == metric) & (res["Sig_hier"])].copy()
        if len(sub) == 0:
            continue

        weights = np.zeros(n_vols)
        for _, r in sub.iterrows():
            idx = net_to_index(r["Network"])
            if 0 <= idx < n_vols:
                weights[idx] = r["Beta"]

        weighted_map = np.tensordot(ica_data, weights, axes=([3], [0]))
        out_img = nib.Nifti1Image(weighted_map, affine=ica_img.affine, header=ica_img.header)
        out_fn = os.path.join(OUTDIR, f"cardiac_circuit_weightedBeta_{metric}.nii.gz")
        out_img.to_filename(out_fn)

    print("Saved brain maps to:", OUTDIR)
    
    
    # ======================================================
    # HEART RATE NETWORK EFFECT PLOT
    # ======================================================
    
    hr = res[res["CardiacMetric"] == "Heart_Rate"].copy()
    
    if len(hr) > 0:
    
        hr = hr.sort_values("Beta")
    
        plt.figure(figsize=(6,6))
    
        plt.barh(
            hr["Network"],
            hr["Beta"],
            color="steelblue"
        )
    
        plt.axvline(0, color="black", linewidth=1)
    
        plt.xlabel("Effect size (beta)")
        plt.title("Brain network association with heart rate")
    
        plt.tight_layout()
    
        fig_out = os.path.join(OUTDIR, "heart_rate_network_effects.png")
    
        plt.savefig(fig_out, dpi=300)
    
        plt.close()
    
        print("Saved:", fig_out)
        
        
        CARDIAC_GROUPS = {

        "Pump_Function":[
            "Stroke_Volume",
            "Ejection_Fraction",
            "Cardiac_Output"
        ],
    
        "Systolic_Function":[
            "Systolic_LV_Volume",
            "Systolic_RV",
            "Systolic_LA",
            "Systolic_RA",
            "Systolic_Myo"
        ],
    
        "Diastolic_Function":[
            "Diastolic_LV_Volume",
            "Diastolic_RV",
            "Diastolic_LA",
            "Diastolic_RA",
            "Diastolic_Myo"
        ]
    }
        
    group_results = []

    for group, metrics in CARDIAC_GROUPS.items():
    
        sub = res[res["CardiacMetric"].isin(metrics)]
    
        for net in sub["Network"].unique():
    
            beta_mean = sub[sub["Network"]==net]["Beta"].mean()
    
            group_results.append({
                "Group":group,
                "Network":net,
                "Beta":beta_mean
            })
    
    group_df = pd.DataFrame(group_results)
    
    for group in group_df["Group"].unique():

        sub = group_df[group_df["Group"]==group].sort_values("Beta")
    
        plt.figure(figsize=(6,6))
    
        plt.barh(sub["Network"], sub["Beta"])
    
        plt.axvline(0,color="black")
    
        plt.title(group)
    
        plt.tight_layout()
    
        plt.savefig(os.path.join(OUTDIR,f"{group}_network_effects.png"),dpi=300)
    
        plt.close()
        
        
  # ======================================================
    # BRAIN-CARDIAC COUPLING INDEX
    # ======================================================
    
    print("Computing brain–cardiac coupling index")
    
    # average absolute beta across cardiac metrics
    weights = beta.abs().mean(axis=1)
    
    weights = weights / weights.sum()
    
    # load original network amplitudes
    nets = pd.read_csv(NETWORK_FILE, sep="\t")
    
    nets.columns = nets.columns.str.strip()
    
    network_cols = [c for c in nets.columns if c.startswith("Amp_")]
    
    # align weights with network columns
    weight_vec = np.array([
        weights.get(net, 0) for net in network_cols
    ])
    
    # compute index
    nets["Brain_Cardiac_Index"] = np.dot(
        nets[network_cols].values,
        weight_vec
    )
    
    bcci_out = os.path.join(OUTDIR, "brain_cardiac_index.tsv")
    
    nets[["Arunno","Brain_Cardiac_Index"]].to_csv(
        bcci_out,
        sep="\t",
        index=False
    )
    
    print("Saved BCCI:", bcci_out)      
    
    # ======================================================
    # BCCI VS HEART RATE
    # ======================================================
    
    df_bcci = pd.merge(meta, nets, on="Arunno")
    
    plt.figure(figsize=(6,5))
    
    plt.scatter(
        df_bcci["Brain_Cardiac_Index"],
        df_bcci["Heart_Rate"],
        s=60
    )
    
    plt.xlabel("Brain–Cardiac Coupling Index")
    
    plt.ylabel("Heart Rate")
    
    plt.title("Brain–Cardiac Coupling")
    
    plt.tight_layout()
    
    fig_out = os.path.join(OUTDIR, "BCCI_vs_HeartRate.png")
    
    plt.savefig(fig_out, dpi=300)
    
    plt.close()
    
    print("Saved:", fig_out)
    
    
    # ======================================================
    # CARDIAC GROUP HEATMAPS
    # ======================================================
    
    CARDIAC_GROUPS = {
    
    "Pump_Function":[
        "Stroke_Volume",
        "Ejection_Fraction",
        "Cardiac_Output"
    ],
    
    "Systolic_Function":[
        "Systolic_LV_Volume",
        "Systolic_RV",
        "Systolic_LA",
        "Systolic_RA",
        "Systolic_Myo"
    ],
    
    "Diastolic_Function":[
        "Diastolic_LV_Volume",
        "Diastolic_RV",
        "Diastolic_LA",
        "Diastolic_RA",
        "Diastolic_Myo"
    ]
    }
    
    group_beta = []
    
    for group, metrics in CARDIAC_GROUPS.items():
    
        sub = res[res["CardiacMetric"].isin(metrics)]
    
        grp = sub.groupby("Network")["Beta"].mean()
    
        grp.name = group
    
        group_beta.append(grp)
    
    group_beta = pd.concat(group_beta, axis=1)
    
    plt.figure(figsize=(8,6))
    
    plt.imshow(group_beta.fillna(0), aspect="auto")
    
    plt.xticks(range(group_beta.shape[1]), group_beta.columns)
    
    plt.yticks(range(group_beta.shape[0]), group_beta.index)
    
    plt.colorbar()
    
    plt.title("Network associations with cardiac physiological systems")
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTDIR,"heatmap_network_cardiac_groups.png"),dpi=300)
    
    plt.close()
    
    
    
    # ======================================================
    # NETWORK CLUSTER STRUCTURE
    # ======================================================
    
    from scipy.cluster.hierarchy import dendrogram
    
    plt.figure(figsize=(10,6))
    
    dendrogram(
        Z,
        labels=beta.index.tolist(),
        leaf_rotation=90
    )
    
    plt.title("Brain networks controlling cardiac physiology")
    
    plt.ylabel("Similarity of cardiac modulation")
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTDIR,"network_cardiac_dendrogram.png"),dpi=300)
    
    plt.close()
    
    
    
    # ======================================================
    # BRAIN-CARDIAC COUPLING INDEX
    # ======================================================
    
    print("Computing Brain–Cardiac Coupling Index")
    
    from sklearn.preprocessing import StandardScaler
    import scipy.stats as stats
    
    network_cols = [c for c in nets.columns if c.startswith("Amp_")]
    
    # Standardize network amplitudes
    scaler = StandardScaler()
    nets[network_cols] = scaler.fit_transform(nets[network_cols])
    
    # compute weights from cardiac associations
    weights = beta.abs().mean(axis=1)
    weights = weights / weights.sum()
    
    weight_vec = np.array([weights.get(net,0) for net in network_cols])
    
    # compute coupling index
    nets["Brain_Cardiac_Index"] = np.dot(
        nets[network_cols].values,
        weight_vec
    )
    
    # save index
    nets[["Arunno","Brain_Cardiac_Index"]].to_csv(
        os.path.join(OUTDIR,"brain_cardiac_index.tsv"),
        sep="\t",
        index=False
    )
    
    print("Saved BCCI")
    
    # ======================================================
    # BCCI VS HEART RATE
    # ======================================================
    
    import statsmodels.api as sm
    import numpy as np
    
    import statsmodels.api as sm
    import numpy as np
    
    df_bcci = pd.merge(meta, nets, on="Arunno")
    
    x = df_bcci["Brain_Cardiac_Index"].values
    y = df_bcci["Heart_Rate"].values
    
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    
    # stats
    p = model.pvalues[1]
    r2 = model.rsquared
    
    # prediction grid
    xfit = np.linspace(x.min(), x.max(), 100)
    Xfit = sm.add_constant(xfit)
    
    pred = model.get_prediction(Xfit)
    pred_summary = pred.summary_frame(alpha=0.05)
    
    yfit = pred_summary["mean"]
    ci_lower = pred_summary["mean_ci_lower"]
    ci_upper = pred_summary["mean_ci_upper"]
    
    plt.figure(figsize=(6,5))
    
    plt.scatter(x, y, s=70)
    
    # regression line
    plt.plot(xfit, yfit, linewidth=2)
    
    # confidence interval
    plt.fill_between(
        xfit,
        ci_lower,
        ci_upper,
        alpha=0.2
    )
    
    plt.xlabel("Brain–Cardiac Coupling Index")
    plt.ylabel("Heart Rate (bpm)")
    plt.title("Neuro–cardiac coupling")
    plt.scatter(x, y, s=70, alpha=0.8)
    plt.text(
        0.05,
        0.95,
        f"$R^2$ = {r2:.2f}\np = {p:.3f}",
        transform=plt.gca().transAxes,
        verticalalignment="top"
    )

    plt.tight_layout()
    
    plt.savefig(
        os.path.join(OUTDIR,"BCCI_vs_HeartRate_CI.png"),
        dpi=300
    )
    
    plt.close()
    
    
    
    
    # ======================================================
    # FINAL: BCCI + TWO-INDEX VERSION (ALWAYS RUN)
    # ======================================================
    
    import statsmodels.api as sm
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    
    print("\n=== FINAL SECTION: BCCI + two-index coupling ===")
    
    # -------------------------------
    # 0) Ensure clean column names
    # -------------------------------
    nets.columns = nets.columns.astype(str).str.strip()
    meta.columns = meta.columns.astype(str).str.strip()
    
    # -------------------------------
    # 1) Standardize network amplitudes
    # -------------------------------
    network_cols = [c for c in nets.columns if c.startswith("Amp_")]
    if len(network_cols) == 0:
        raise ValueError("No network amplitude columns found (expected columns starting with 'Amp_').")
    
    nets_z = nets.copy()
    scaler = StandardScaler()
    nets_z[network_cols] = scaler.fit_transform(nets_z[network_cols].values)
    
    # -------------------------------
    # 2) Compute global BCCI (weights from beta table)
    #    NOTE: beta index must match network column names exactly.
    # -------------------------------
    weights = beta.abs().mean(axis=1)           # network -> weight
    weights = weights / weights.sum()
    
    weight_vec = np.array([weights.get(net, 0.0) for net in network_cols])
    
    nets_z["BCCI"] = np.dot(nets_z[network_cols].values, weight_vec)
    
    bcci_out = os.path.join(OUTDIR, "brain_cardiac_index.tsv")
    nets_z[["Arunno", "BCCI"]].to_csv(bcci_out, sep="\t", index=False)
    print("Saved BCCI:", bcci_out)
    
    # merge for plotting/regression
    df_bcci = pd.merge(meta, nets_z[["Arunno", "BCCI"]], on="Arunno", how="inner")
    print("Subjects in df_bcci:", df_bcci.shape[0])
    
    # -------------------------------
    # 3) Plot BCCI vs Heart Rate with regression + 95% CI + p + R²
    # -------------------------------
    if "Heart_Rate" not in df_bcci.columns:
        raise ValueError("Heart_Rate not found in metadata after merge.")
    
    x = df_bcci["BCCI"].astype(float).values
    y = df_bcci["Heart_Rate"].astype(float).values
    
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    p = model.pvalues[1]
    r2 = model.rsquared
    
    xfit = np.linspace(x.min(), x.max(), 200)
    Xfit = sm.add_constant(xfit)
    
    pred = model.get_prediction(Xfit).summary_frame(alpha=0.05)
    yfit = pred["mean"].values
    ci_lower = pred["mean_ci_lower"].values
    ci_upper = pred["mean_ci_upper"].values
    
    plt.figure(figsize=(6,5))
    plt.scatter(x, y, s=70, alpha=0.8)
    plt.plot(xfit, yfit, linewidth=2)
    plt.fill_between(xfit, ci_lower, ci_upper, alpha=0.2)
    
    plt.xlabel("Brain–Cardiac Coupling Index (BCCI)")
    plt.ylabel("Heart Rate (bpm)")
    plt.title("Neuro–cardiac coupling")
    plt.text(
        0.05, 0.95,
        f"$R^2$ = {r2:.2f}\np = {p:.3f}",
        transform=plt.gca().transAxes,
        va="top"
    )
    
    plt.tight_layout()
    fig_out = os.path.join(OUTDIR, "BCCI_vs_HeartRate_CI.png")
    plt.savefig(fig_out, dpi=300)
    plt.close()
    print("Saved:", fig_out)
    
    # -------------------------------
    # 4) Two-index version (HR+ networks vs HR- networks)
    #    Uses Heart_Rate betas from res table
    # -------------------------------
    hr_assoc = res[res["CardiacMetric"] == "Heart_Rate"].copy()
    if hr_assoc.empty:
        raise ValueError("No rows for CardiacMetric == 'Heart_Rate' found in res.")
    
    # Identify networks by sign of HR association
    pos_nets = hr_assoc.loc[hr_assoc["Beta"] > 0, "Network"].unique().tolist()
    neg_nets = hr_assoc.loc[hr_assoc["Beta"] < 0, "Network"].unique().tolist()
    
    # Ensure names match your network amplitude columns.
    # If your Network field is already like 'Amp_ICA01' then this is fine.
    # If your Network field is like 'ICA01', we convert to 'Amp_ICA01'.
    def to_amp_name(net_name):
        net_name = str(net_name).strip()
        return net_name if net_name.startswith("Amp_") else f"Amp_{net_name}"
    
    pos_cols = [to_amp_name(n) for n in pos_nets]
    neg_cols = [to_amp_name(n) for n in neg_nets]
    
    pos_cols = [c for c in pos_cols if c in nets_z.columns]
    neg_cols = [c for c in neg_cols if c in nets_z.columns]
    
    if len(pos_cols) == 0 or len(neg_cols) == 0:
        raise ValueError(
            f"Two-index split failed. Found pos_cols={len(pos_cols)}, neg_cols={len(neg_cols)}. "
            "Check that res['Network'] naming matches nets columns (Amp_*)."
        )
    
    # Simple mean-based indices (already z-scored networks)
    nets_z["Index_HR_Pos"] = nets_z[pos_cols].mean(axis=1)
    nets_z["Index_HR_Neg"] = nets_z[neg_cols].mean(axis=1)
    
    two_out = os.path.join(OUTDIR, "dual_cardiac_indices.tsv")
    nets_z[["Arunno", "Index_HR_Pos", "Index_HR_Neg"]].to_csv(two_out, sep="\t", index=False)
    print("Saved two indices:", two_out)
    
    df2 = pd.merge(meta, nets_z[["Arunno", "Index_HR_Pos", "Index_HR_Neg"]], on="Arunno", how="inner")
    
    # -------------------------------
    # 5) Plot each index vs HR (with regression + CI)
    # -------------------------------
    def plot_index_vs_hr(df, xcol, outname, title):
        x = df[xcol].astype(float).values
        y = df["Heart_Rate"].astype(float).values
    
        X = sm.add_constant(x)
        m = sm.OLS(y, X).fit()
    
        xfit = np.linspace(x.min(), x.max(), 200)
        pred = m.get_prediction(sm.add_constant(xfit)).summary_frame(alpha=0.05)
    
        plt.figure(figsize=(6,5))
        plt.scatter(x, y, s=70, alpha=0.8)
        plt.plot(xfit, pred["mean"].values, linewidth=2)
        plt.fill_between(xfit, pred["mean_ci_lower"].values, pred["mean_ci_upper"].values, alpha=0.2)
    
        plt.xlabel(xcol)
        plt.ylabel("Heart Rate (bpm)")
        plt.title(title)
        plt.text(
            0.05, 0.95,
            f"$R^2$ = {m.rsquared:.2f}\np = {m.pvalues[1]:.3f}",
            transform=plt.gca().transAxes,
            va="top"
        )
    
        plt.tight_layout()
        fn = os.path.join(OUTDIR, outname)
        plt.savefig(fn, dpi=300)
        plt.close()
        print("Saved:", fn)
    
    plot_index_vs_hr(df2, "Index_HR_Pos", "Index_HR_Pos_vs_HeartRate_CI.png", "HR+ network index vs Heart Rate")
    plot_index_vs_hr(df2, "Index_HR_Neg", "Index_HR_Neg_vs_HeartRate_CI.png", "HR− network index vs Heart Rate")
    
    # -------------------------------
    # 6) Optional: 2-predictor model (HR+ and HR− together)
    # -------------------------------
    X = df2[["Index_HR_Pos", "Index_HR_Neg"]].astype(float)
    X = sm.add_constant(X)
    m2 = sm.OLS(df2["Heart_Rate"].astype(float), X).fit()
    
    two_model_out = os.path.join(OUTDIR, "two_index_model_summary.txt")
    with open(two_model_out, "w") as f:
        f.write(m2.summary().as_text())
    
    print("Saved two-index model summary:", two_model_out)
    print("Two-index model R^2:", round(m2.rsquared, 3))
    
    
    
    
    
    
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    network_cols = [c for c in nets.columns if c.startswith("Amp_")]

    X = nets[network_cols].values
    y = meta["Heart_Rate"].values.reshape(-1,1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    pls = PLSRegression(n_components=1)

    pls.fit(X,y)

    y_pred = pls.predict(X)

    r2 = r2_score(y,y_pred)

    print("PLS R2:",r2)
    
    
    #PLS
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5,shuffle=True,random_state=1)
    
    pred = np.zeros(len(y))
    
    for train,test in kf.split(X):
    
        pls = PLSRegression(n_components=1)
        pls.fit(X[train],y[train])
        pred[test] = pls.predict(X[test]).ravel()
    
    r2_cv = r2_score(y,pred)
    
    print("Cross-validated R2:",r2_cv)
    
    
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler

    network_cols = [c for c in nets.columns if c.startswith("Amp_")]
    
    scaler = StandardScaler()
    
    X = scaler.fit_transform(nets[network_cols].values)
    
    weights = beta.abs().mean(axis=1)
    weights = weights / weights.sum()
    
    weight_vec = np.array([weights.get(net,0) for net in network_cols])
    
    nets["Brain_Cardiac_Index"] = np.dot(X, weight_vec)
    
    

    loo = LeaveOneOut()
    
    betas = []
    
    # ensure consistent column name
    if "Brain_Cardiac_Index" not in nets.columns:
        if "BCCI" in nets.columns:
            nets["Brain_Cardiac_Index"] = nets["BCCI"]
    
    df_bcci = pd.merge(meta, nets, on="Arunno")
    
    X = df_bcci["Brain_Cardiac_Index"].values.reshape(-1,1)
    y = df_bcci["Heart_Rate"].values
    
    for train,test in loo.split(X):
    
        model = sm.OLS(y[train], sm.add_constant(X[train])).fit()
        betas.append(model.params[1])
    
    print("Mean beta:", np.mean(betas))
    print("SD beta:", np.std(betas))
    
    
    plt.figure(figsize=(6,5))

plt.scatter(
    df_bcci["Brain_Cardiac_Index"].rank(),
    df_bcci["Heart_Rate"].rank(),
    s=70
)

plt.xlabel("Ranked Brain–Cardiac Index")
plt.ylabel("Ranked Heart Rate")

plt.title("Spearman relationship (ρ ≈ -0.28)")

plt.tight_layout()

plt.savefig(os.path.join(OUTDIR,"spearman_rank_plot.png"),dpi=300)


import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

x = df_bcci["Brain_Cardiac_Index"].values
y = df_bcci["Heart_Rate"].values

# Fit regression
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()

# Stats
p = model.pvalues[1]
r2 = model.rsquared

# Prediction grid
xfit = np.linspace(x.min(), x.max(), 200)
Xfit = sm.add_constant(xfit)

pred = model.get_prediction(Xfit).summary_frame()

yfit = pred["mean"]
ci_lower = pred["mean_ci_lower"]
ci_upper = pred["mean_ci_upper"]

plt.figure(figsize=(6,5))

# Scatter
plt.scatter(x, y, s=70, alpha=0.8)

# Regression line
plt.plot(xfit, yfit, linewidth=2)

# Confidence interval
plt.fill_between(
    xfit,
    ci_lower,
    ci_upper,
    alpha=0.2
)

plt.xlabel("Brain–Cardiac Coupling Index")
plt.ylabel("Heart Rate (bpm)")
plt.title("Brain–cardiac coupling")

# Annotation
plt.text(
    0.05,
    0.95,
    f"$R^2$ = {r2:.2f}\np = {p:.3f}",
    transform=plt.gca().transAxes,
    verticalalignment="top"
)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTDIR,"BCCI_vs_HeartRate_CI.png"),
    dpi=300
)

plt.close()