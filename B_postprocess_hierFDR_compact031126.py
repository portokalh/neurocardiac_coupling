#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 11:29:58 2026

@author: alex
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Network–Cardiac Coupling Analysis Pipeline
Author: Alex
"""



# ======================================================
# 1. IMPORTS
# ======================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from statsmodels.stats.multitest import multipletests
from scipy.cluster.hierarchy import dendrogram, linkage

import statsmodels.api as sm
import statsmodels.formula.api as smf

import nibabel as nib

# Optional brain map support
BRAINMAPS_ENABLED = True



try:
    import nibabel as nib
except Exception:
    nib = None


# ======================================================
# 2. PATHS
# ======================================================


IN_RESULTS = "/mnt/newStor/paros/paros_WORK/aashika/results/network_cardiac_stats/network_cardiac_results.tsv"
NETWORK_FILE = "/mnt/newStor/paros/paros_WORK/aashika/results/network12_from_errts/Network12_amplitudes.tsv"
META_FILE = "/mnt/newStor/paros/paros_WORK/aashika/data/metadata/cardiac_design_updated3.csv"

ICA_4D_FILE = "/mnt/newStor/paros/paros_WORK/aashika/data/ICA/4DNetwork/Networks_12_4D.nii.gz"

OUTDIR = "/mnt/newStor/paros/paros_WORK/aashika/results/network_cardiac_postproc031126"
os.makedirs(OUTDIR, exist_ok=True)

BRAINMAPS_ENABLED = True


# ======================================================
# 3. LOAD DATA
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
# 4. CARDIAC METRIC GROUP DEFINITIONS
# ======================================================
'''
CARDIAC_GROUPS = {
    "pump_function": ["Heart_Rate","Stroke_Volume","Ejection_Fraction","Cardiac_Output"],
    "left_ventricle": ["Diastolic_LV_Volume","Systolic_LV_Volume"],
    "right_ventricle": ["Diastolic_RV","Systolic_RV"],
    "atria": ["Diastolic_LA","Systolic_LA","Diastolic_RA","Systolic_RA"],
    "myocardium": ["Diastolic_Myo","Systolic_Myo"]
}
'''

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
metric_to_group = {m:g for g,ms in CARDIAC_GROUPS.items() for m in ms}




# ======================================================
# 5. HELPER FUNCTIONS
# ======================================================

def safe_z(x):
    x = pd.to_numeric(x, errors="coerce")
    return (x - np.nanmean(x)) / np.nanstd(x)

def simes_p(pvals):
    p = np.sort(np.asarray(pvals))
    m = len(p)
    if m == 0:
        return np.nan
    return np.min((m*p)/(np.arange(1,m+1)))

def save_heatmap(mat,title,filename):

    mat = mat.fillna(0)

    plt.figure(figsize=(12,7))
    sns.heatmap(mat,cmap="coolwarm",center=0)

    plt.title(title)
    plt.tight_layout()

    plt.savefig(os.path.join(OUTDIR,filename),dpi=300)
    plt.close()


# ======================================================
# 6. ASSIGN METRIC GROUPS
# ======================================================

res["Group"] = res["CardiacMetric"].map(metric_to_group)

if res["Group"].isna().any():
    raise ValueError("Some cardiac metrics not mapped")



# ======================================================
# 7. HIERARCHICAL FDR
# ======================================================

print("Running hierarchical FDR")

group_p = (
    res.groupby("Group")["P"]
    .apply(lambda s: simes_p(s.values))
    .reset_index()
    .rename(columns={"P":"Group_P"})
)

group_p["Group_FDR"] = multipletests(group_p["Group_P"],method="fdr_bh")[1]

res = res.merge(group_p,on="Group")

res["FDR_within_group"] = np.nan

for g in res["Group"].unique():

    idx = res["Group"]==g

    res.loc[idx,"FDR_within_group"] = multipletests(
        res.loc[idx,"P"],method="fdr_bh"
    )[1]

res["Sig_hier"] = (res["Group_FDR"]<0.05) & (res["FDR_within_group"]<0.05)

res.to_csv(os.path.join(OUTDIR,"network_cardiac_results_hierFDR.tsv"),sep="\t",index=False)




# ======================================================
# 8. HEATMAPS
# ======================================================

mlogp = res.pivot_table(index="Network",columns="CardiacMetric",values="P")
mlogp = -np.log10(mlogp)

beta = res.pivot_table(index="Network",columns="CardiacMetric",values="Beta")

save_heatmap(mlogp,"Network–Cardiac (-log10 p)","heatmap_logp.png")
save_heatmap(beta,"Network–Cardiac beta","heatmap_beta.png")


# ======================================================
# 9. NETWORK CLUSTERING
# ======================================================

X = StandardScaler().fit_transform(beta.fillna(0))

Z = linkage(X,method="ward")

plt.figure(figsize=(10,6))
dendrogram(Z,labels=beta.index.tolist(),leaf_rotation=90)

plt.title("Network clustering by cardiac profile")

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR,"network_dendrogram.png"),dpi=300)
plt.close()


# ======================================================
# 10. COMPUTE BCCI
# ======================================================

print("Computing BCCI")

network_cols = [c for c in nets.columns if c.startswith("Amp_")]

nets_z = nets.copy()
nets_z[network_cols] = StandardScaler().fit_transform(nets_z[network_cols])

weights = beta.abs().mean(axis=1)
weights = weights / weights.sum()

weights.index = [
    x if x.startswith("Amp_") else f"Amp_{x}"
    for x in weights.index
]

weight_vec = np.array([weights.get(c,0) for c in network_cols])

nets_z["BCCI"] = np.dot(nets_z[network_cols].values,weight_vec)

nets_z[["Arunno","BCCI"]].to_csv(
    os.path.join(OUTDIR,"brain_cardiac_index.tsv"),
    sep="\t",
    index=False
)




# ======================================================
# 11. MERGE METADATA
# ======================================================

df_bcci = pd.merge(meta,nets_z[["Arunno","BCCI"]],on="Arunno")

print("Subjects:",len(df_bcci))

df = df_bcci.copy()


# ======================================================
# DOMAIN Z SCORES
# ======================================================

for domain, metrics in CARDIAC_GROUPS.items():

    existing = [m for m in metrics if m in df.columns]

    if len(existing)==0:
        continue

    tmp = df[existing].apply(pd.to_numeric,errors="coerce")

    tmp_z = tmp.apply(safe_z)

    df[domain+"_z"] = tmp_z.mean(axis=1,skipna=True)

for m in ["Heart_Rate_z","pump_function_z","left_ventricle_z"]:

    if m not in df.columns:
        print("Missing:", m)
        continue

    r = np.corrcoef(df["BCCI_z"], df[m])[0,1]
    print(m, r)

# ======================================================
# 12. BCCI vs HEART RATE
# ======================================================

x = df["BCCI"].values
y = df["Heart_Rate"].values

X = sm.add_constant(x)

model = sm.OLS(y,X).fit()

r2 = model.rsquared
p = model.pvalues[1]

plt.figure(figsize=(6,5))
plt.scatter(x,y,s=70)

xfit = np.linspace(x.min(),x.max(),200)
pred = model.get_prediction(sm.add_constant(xfit)).summary_frame()

plt.plot(xfit,pred["mean"],linewidth=2)
plt.fill_between(xfit,pred["mean_ci_lower"],pred["mean_ci_upper"],alpha=0.2)

plt.xlabel("BCCI")
plt.ylabel("Heart Rate")

plt.text(0.05,0.95,f"R²={r2:.2f}\np={p:.3f}",transform=plt.gca().transAxes)

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR,"BCCI_vs_HeartRate.png"),dpi=300)
plt.close()



# ======================================================
# 13. SPEARMAN-STYLE RANK PLOT
# ======================================================

plt.figure(figsize=(6,5))

plt.scatter(
    df_bcci["BCCI"].rank(),
    df_bcci["Heart_Rate"].rank(),
    s=70
)

plt.xlabel("Ranked BCCI")
plt.ylabel("Ranked Heart Rate")

plt.tight_layout()

plt.savefig(os.path.join(OUTDIR,"spearman_rank_plot.png"),dpi=300)

plt.close()


# ======================================================
# 14. PLS MODEL
# ======================================================

X = nets_z[network_cols].values
y = df["Heart_Rate"].values.reshape(-1,1)

pls = PLSRegression(n_components=1)
pls.fit(X,y)

pred = pls.predict(X)

print("PLS R2:",r2_score(y,pred))


# ======================================================
# 15. CROSS VALIDATED PLS
# ======================================================

kf = KFold(n_splits=5,shuffle=True,random_state=1)

pred = np.zeros(len(y))

for train,test in kf.split(X):

    pls = PLSRegression(n_components=1)

    pls.fit(X[train],y[train])

    pred[test] = pls.predict(X[test]).ravel()

print("Cross validated R2:",r2_score(y,pred))


# ======================================================
# 16. INDIVIDUAL METRIC BRAIN MAPS
# ======================================================

if BRAINMAPS_ENABLED:

    print("Creating individual metric brain maps")

    img = nib.load(ICA_4D_FILE)
    data = img.get_fdata()

    for metric in res["CardiacMetric"].unique():

        # USE ALL NETWORKS (no Sig_hier filtering)
        sub = res[res["CardiacMetric"] == metric]

        weights = np.zeros(data.shape[3])

        for _, r in sub.iterrows():

            if r["Network"] not in network_cols:
                continue

            idx = network_cols.index(r["Network"])
            weights[idx] = r["Beta"]

        # build map
        weighted_map = np.tensordot(data, weights, axes=([3],[0]))

        out = nib.Nifti1Image(weighted_map, img.affine)

        outfile = os.path.join(OUTDIR, f"cardiac_map_ori_{metric}.nii.gz")

        out.to_filename(outfile)

        print("Saved:", outfile)

# ======================================================
# 16B. DOMAIN BRAIN MAPS
# ======================================================

if BRAINMAPS_ENABLED:

    print("Creating cardiac domain brain maps")

    domain_defs = {

        "HeartRate": ["Heart_Rate"],

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

    for domain, metrics in domain_defs.items():

        sub = res[res["CardiacMetric"].isin(metrics)]

        weights = np.zeros(data.shape[3])

        for _, r in sub.iterrows():

            if r["Network"] not in network_cols:
                continue

            idx = network_cols.index(r["Network"])
            weights[idx] += r["Beta"]

        weighted_map = np.tensordot(data, weights, axes=([3],[0]))

        out = nib.Nifti1Image(weighted_map, img.affine)

        out_file = os.path.join(
            OUTDIR,
            f"cardiac_domain_ori_map_{domain}.nii.gz"
        )

        out.to_filename(out_file)

        print("Saved:", out_file)
    



        
        
        
    # ======================================================
# 17. BRAIN–HEART CIRCUIT MAP (PLS WEIGHTS)
# ======================================================

if BRAINMAPS_ENABLED:

    print("Creating brain–heart circuit map")

    if ICA_4D_FILE is None:
        raise ValueError("ICA_4D_FILE must be defined")

    if not os.path.exists(ICA_4D_FILE):
        raise ValueError("ICA file not found")

    ica_img = nib.load(ICA_4D_FILE)
    ica_data = ica_img.get_fdata()

    n_components = ica_data.shape[3]

    # --------------------------------------------
    # Compute network weights from PLS model
    # --------------------------------------------

    pls_weights = pls.x_weights_.flatten()

    if len(pls_weights) != n_components:
        print("Warning: ICA components and network columns differ")
    
    weights = np.zeros(n_components)

  # --------------------------------------------
  # Align weights with ICA components by order
  # --------------------------------------------

    weights = np.zeros(n_components)

    n = min(len(network_cols), n_components)

    for i in range(n):
        weights[i] = pls_weights[i]
    
    if len(network_cols) != n_components:
        print(
            f"Warning: ICA components ({n_components}) "
            f"!= network columns ({len(network_cols)}). "
            "Using first matching components."
        )

    # --------------------------------------------
    # Build weighted circuit map
    # --------------------------------------------

    circuit_map = np.tensordot(
        ica_data,
        weights,
        axes=([3],[0])
    )

    out_img = nib.Nifti1Image(
        circuit_map,
        affine=ica_img.affine,
        header=ica_img.header
    )

    out_file = os.path.join(
        OUTDIR,
        "brain_heart_circuit_map_PLS.nii.gz"
    )

    out_img.to_filename(out_file)

    print("Saved circuit map:", out_file)
    
  
    
# ======================================================
# 16. INDIVIDUAL METRIC BRAIN MAPS
# ======================================================

if BRAINMAPS_ENABLED:

    print("Creating individual metric brain maps")

    img = nib.load(ICA_4D_FILE)
    data = img.get_fdata()

    for metric in res["CardiacMetric"].unique():

        sub = res[res["CardiacMetric"] == metric]

        weights = np.zeros(data.shape[3])

        for _, r in sub.iterrows():

            if r["Network"] not in network_cols:
                continue

            idx = network_cols.index(r["Network"])

            p = max(r["P"], 1e-10)
            weights[idx] = r["Beta"] * (-np.log10(p))

        if np.max(np.abs(weights)) == 0:
            continue

        weights = weights / np.max(np.abs(weights))

        weighted_map = np.tensordot(data, weights, axes=([3],[0]))

        # ----- THRESHOLD VERSION -----
        thr = np.percentile(np.abs(weighted_map), 80)

        thr_map = weighted_map.copy()
        thr_map[np.abs(thr_map) < thr] = 0

        # save continuous map
        nib.save(
            nib.Nifti1Image(weighted_map, img.affine),
            os.path.join(OUTDIR, f"cardiac_map_{metric}.nii.gz")
        )

        # save thresholded map
        nib.save(
            nib.Nifti1Image(thr_map, img.affine),
            os.path.join(OUTDIR, f"cardiac_map_{metric}_thr80.nii.gz")
        )

        print("Saved:", metric)  
    
 # ======================================================
# 16B. DOMAIN BRAIN MAPS
# ======================================================

if BRAINMAPS_ENABLED:

    print("Creating cardiac domain brain maps")

    domain_defs = {

        "HeartRate": ["Heart_Rate"],

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

    img = nib.load(ICA_4D_FILE)
    data = img.get_fdata()

    for domain, metrics in domain_defs.items():

        sub = res[res["CardiacMetric"].isin(metrics)]

        weights = np.zeros(data.shape[3])

        for _, r in sub.iterrows():

            if r["Network"] not in network_cols:
                continue

            idx = network_cols.index(r["Network"])

            p = max(r["P"], 1e-10)
            weights[idx] += r["Beta"] * (-np.log10(p))

        if np.max(np.abs(weights)) == 0:
            continue

        weights = weights / np.max(np.abs(weights))

        weighted_map = np.tensordot(data, weights, axes=([3],[0]))

        # ----- THRESHOLD VERSION -----
        thr = np.percentile(np.abs(weighted_map), 80)

        thr_map = weighted_map.copy()
        thr_map[np.abs(thr_map) < thr] = 0

        nib.save(
            nib.Nifti1Image(weighted_map, img.affine),
            os.path.join(OUTDIR, f"cardiac_domain_map_{domain}.nii.gz")
        )

        nib.save(
            nib.Nifti1Image(thr_map, img.affine),
            os.path.join(OUTDIR, f"cardiac_domain_map_{domain}_thr80.nii.gz")
        )

        print("Saved domain:", domain) 
    
 # ======================================================
# 17. PLS BRAIN–HEART CIRCUIT MAP
# ======================================================

if BRAINMAPS_ENABLED:

    print("Creating PLS brain–heart circuit map")

    img = nib.load(ICA_4D_FILE)
    data = img.get_fdata()

    # PLS network weights
    weights = pls.x_weights_.flatten()

    # normalize for visualization
    if np.max(np.abs(weights)) > 0:
        weights = weights / np.max(np.abs(weights))

    circuit_map = np.tensordot(data, weights, axes=([3],[0]))

    # threshold for figure visualization
    thr = np.percentile(np.abs(circuit_map), 80)

    thr_map = circuit_map.copy()
    thr_map[np.abs(thr_map) < thr] = 0

    # save continuous map
    nib.save(
        nib.Nifti1Image(circuit_map, img.affine),
        os.path.join(OUTDIR, "brain_heart_circuit_map_PLS.nii.gz")
    )

    # save thresholded map
    nib.save(
        nib.Nifti1Image(thr_map, img.affine),
        os.path.join(OUTDIR, "brain_heart_circuit_map_PLS_thr80.nii.gz")
    )

    print("Saved PLS brain–heart circuit map")   
   
    
   # ======================================================
# 18. CARDIAC DOMAIN CIRCUIT MAPS
# ======================================================

if BRAINMAPS_ENABLED:

    print("Creating cardiac domain circuit maps")

    domain_defs = {

        "HeartRate": ["Heart_Rate"],

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

    ica_img = nib.load(ICA_4D_FILE)
    ica_data = ica_img.get_fdata()
    n_components = ica_data.shape[3]

    for domain, metrics in domain_defs.items():

        sub = res[res["CardiacMetric"].isin(metrics)]

        if len(sub)==0:
            continue

        # average beta per network
        net_beta = sub.groupby("Network")["Beta"].mean()

        weights = np.zeros(n_components)

        for i, col in enumerate(network_cols):
        
            if i >= n_components:
                break
        
            weights[i] = net_beta.get(col,0)
        
        # ------------------------------------
        # Debug: show network importance
        # ------------------------------------
        print("\nDomain:", domain)
        print(pd.Series(weights, index=network_cols).sort_values(ascending=False))
        
        # ------------------------------------------------
        # keep only strongest network contributions
        # ------------------------------------------------
        thr = np.percentile(np.abs(weights), 60)
        weights[np.abs(weights) < thr] = 0
        
        # normalize for stable maps
        if np.max(np.abs(weights)) > 0:
            weights = weights / np.max(np.abs(weights))
        
        # ------------------------------------------------
        # Build weighted circuit map
        # ------------------------------------------------
        circuit_map = np.tensordot(
            ica_data,
            weights,
            axes=([3],[0])
        )

        out_img = nib.Nifti1Image(
            circuit_map,
            affine=ica_img.affine,
            header=ica_img.header
        )

        out_file = os.path.join(
            OUTDIR,
            f"{domain}_brain_cardiac_map.nii.gz"
        )

        out_img.to_filename(out_file)

        print("Saved:", out_file)
        
        
        
        
        
        # ======================================================
        # 19. NETWORK IMPORTANCE BY CARDIAC DOMAIN (FIGURE)
        # ======================================================
        
        print("Creating network importance summary figure")
        
        domain_matrix = pd.DataFrame(index=network_cols)
        
        for domain, metrics in domain_defs.items():
        
            sub = res[res["CardiacMetric"].isin(metrics)]
        
            if len(sub) == 0:
                continue
        
            net_beta = sub.groupby("Network")["Beta"].mean()
            net_beta.index = ["Amp_" + x.replace("Amp_","") for x in net_beta.index]
        
            vals = [net_beta.get(c,0) for c in network_cols]
        
            domain_matrix[domain] = vals
        
        domain_matrix = domain_matrix.fillna(0)
        
        plt.figure(figsize=(8,6))
        
        plt.imshow(domain_matrix.values, aspect="auto")
        
        plt.xticks(range(domain_matrix.shape[1]), domain_matrix.columns, rotation=45)
        plt.yticks(range(domain_matrix.shape[0]), domain_matrix.index)
        
        plt.colorbar(label="Beta")
        
        plt.title("Network Importance by Cardiac Domain")
        
        plt.tight_layout()
        
        plt.savefig(
            os.path.join(OUTDIR,"network_importance_by_domain.png"),
            dpi=300
        )
        
        plt.close()
        
        print("Saved domain importance figure")
        
        
        
       # ======================================================
        # NETWORK IMPORTANCE FIGURES (METRICS + DOMAINS)
        # ======================================================
        
        print("Creating network importance figures")
        
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        import os
        
        '''
        # ------------------------------------------------------
        # Cardiac domains
        # ------------------------------------------------------
        
        domain_defs = {
            "HeartRate": ["HeartRate"],
            "Pump_Function": ["EjectionFraction"],
            "Systolic_Function": ["StrokeVolume","SystolicVolume"],
            "Diastolic_Function": [
                "Diastolic_LV_Volume",
                "Diastolic_RA",
                "Diastolic_Myo"
            ]
        }
        '''
        # ------------------------------------------------------
        # 1. NETWORK × METRIC MATRIX
        # ------------------------------------------------------
        
        metric_matrix = res.pivot_table(
            index="Network",
            columns="CardiacMetric",
            values="Beta"
        )
        
        metric_matrix = metric_matrix.reindex(network_cols)
        
        # ------------------------------------------------------
        # 2. NETWORK × DOMAIN MATRIX
        # ------------------------------------------------------
        
        # ------------------------------------------------------
        # NETWORK × DOMAIN MATRIX
        # ------------------------------------------------------
        
        domain_matrix = pd.DataFrame(index=network_cols)
        
        for domain, metrics in domain_defs.items():
        
            sub = res[res["CardiacMetric"].isin(metrics)]
        
            if len(sub) == 0:
                continue
        
            # mean beta per network
            net_beta = sub.groupby("Network")["Beta"].mean()
        
            # align with network order
            vals = [net_beta.get(n,0) for n in network_cols]
        
            domain_matrix[domain] = vals
        
        domain_matrix = domain_matrix.fillna(0)
        
        # ------------------------------------------------------
        # 3. STANDARDIZE FOR VISUALIZATION
        # ------------------------------------------------------
        
        def scale_matrix(mat):
        
            scaled = mat.copy()
        
            for col in scaled.columns:
                sd = scaled[col].std()
                if sd > 0:
                    scaled[col] = (scaled[col] - scaled[col].mean()) / sd
        
            return scaled
        
        metric_scaled = scale_matrix(metric_matrix)
        domain_scaled = scale_matrix(domain_matrix)
        
        # robust symmetric color limits
        v_metric = np.percentile(np.abs(metric_scaled.values),95)
        v_domain = np.percentile(np.abs(domain_scaled.values),95)
        
        # ------------------------------------------------------
        # 4. HEATMAP — ALL METRICS
        # ------------------------------------------------------
        
        plt.figure(figsize=(10,7))
        
        sns.heatmap(
            metric_scaled,
            cmap="coolwarm",
            center=0,
            vmin=-v_metric,
            vmax=v_metric,
            linewidths=0.5,
            cbar_kws={"label":"Standardized Beta"}
        )
        
        plt.title("Brain–Heart Coupling by Cardiac Metric")
        
        plt.xticks(rotation=40, ha="right")
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        plt.savefig(
            os.path.join(OUTDIR,"network_importance_by_metric.png"),
            dpi=300
        )
        
        plt.close()
        
        # ------------------------------------------------------
        # 5. HEATMAP — CARDIAC DOMAINS
        # ------------------------------------------------------
        
        plt.figure(figsize=(8,6))
        
        sns.heatmap(
            domain_scaled,
            cmap="coolwarm",
            center=0,
            vmin=-v_domain,
            vmax=v_domain,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"label":"Standardized Beta"}
        )
        
        plt.title("Brain–Heart Coupling by Cardiac Domain")
        
        plt.xticks(rotation=35, ha="right")
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        plt.savefig(
            os.path.join(OUTDIR,"network_importance_by_domain.png"),
            dpi=300
        )
        
        plt.close()
        
        print("Saved:")
        print("  network_importance_by_metric.png")
        print("  network_importance_by_domain.png")
        
        
        # ------------------------------------------------------
        # 5. HEATMAP — CARDIAC DOMAINS (CLUSTERED)
        # ------------------------------------------------------
        
        import scipy.cluster.hierarchy as sch
        
        # ensure scaling
        domain_scaled = (domain_matrix - domain_matrix.mean()) / domain_matrix.std()
        
        # robust symmetric color limits
        v_domain = np.percentile(np.abs(domain_scaled.values), 95)
        
        # clustering
        g = sns.clustermap(
            domain_scaled,
            cmap="coolwarm",
            center=0,
            vmin=-v_domain,
            vmax=v_domain,
            linewidths=0.5,
            annot=True,
            fmt=".2f",
            figsize=(8,7),
            row_cluster=True,
            col_cluster=False,
            cbar_kws={"label":"Standardized Beta"}
        )
        
        g.fig.suptitle("Brain–Heart Coupling by Cardiac Domain", y=1.02)
        
        plt.savefig(
            os.path.join(OUTDIR,"network_importance_by_domain_clustered.png"),
            dpi=300
        )
        
        plt.close()
        
        
        # ======================================================
        # 11B. CLEAN / STANDARDIZE VARIABLES FOR MODERATION MODELS
        # ======================================================
        
        print("Preparing variables for moderation models")
        
        df = df_bcci.copy()
        
        # ------------------------------------------------------
        # Define moderators present in your dataset
        # ------------------------------------------------------
        
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
        
        # ------------------------------------------------------
        # Ensure numeric format
        # ------------------------------------------------------
        
        for col in mods + ["BCCI"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # ------------------------------------------------------
        # Safe zscore function (works with NaNs and old scipy)
        # ------------------------------------------------------
        
        def safe_z(x):
            x = pd.to_numeric(x, errors="coerce")
            return (x - np.nanmean(x)) / np.nanstd(x)
        
        # ------------------------------------------------------
        # Standardize continuous variables
        # ------------------------------------------------------
        
        df["BCCI_z"] = safe_z(df["BCCI"])
        
        if "Age" in df.columns:
            df["Age_z"] = safe_z(df["Age"])
        
        if "Mass" in df.columns:
            df["Mass_z"] = safe_z(df["Mass"])
        
        # ------------------------------------------------------
        # Identify cardiac metrics present
        # ------------------------------------------------------
        
        all_metrics = [m for m in metric_to_group.keys() if m in df.columns]
        
        print("Cardiac metrics found:", all_metrics)
        
        # create standardized versions
        for m in all_metrics:
            df[m + "_z"] = safe_z(df[m])
            
    # ======================================================
    # 12. BCCI vs ALL CARDIAC METRICS
    # ======================================================
    
    print("Running BCCI vs cardiac metrics")
    
    bcci_metric_results = []
    
    for metric in all_metrics:
        tmp = df[["BCCI_z", metric]].copy()
        tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
        tmp = tmp.dropna()
    
        if len(tmp) < 15:
            continue
    
        model = smf.ols(f"{metric} ~ BCCI_z", data=tmp).fit()
    
        bcci_metric_results.append({
            "Metric": metric,
            "N": len(tmp),
            "Beta_BCCI": model.params.get("BCCI_z", np.nan),
            "P_BCCI": model.pvalues.get("BCCI_z", np.nan),
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
            sep="\t", index=False
        )
    
    print(bcci_metric_results)     
    
    
    # ======================================================
    # 13. MODERATION MODELS: DOES EACH FACTOR CHANGE BCCI–METRIC COUPLING?
    # ======================================================
    
    print("Running moderation models")
    
    moderation_results = []
    
    # use one moderator at a time first
    mods_to_test = []
    if "Age_z" in df.columns:
        mods_to_test.append("Age_z")
    for c in ["Sex", "Exercise_Yes", "HFD_Yes", "APOE"]:
        if c in df.columns:
            mods_to_test.append(c)
    
    for metric in all_metrics:
        metric_z = metric + "_z"
        if metric_z not in df.columns:
            continue
    
        for mod in mods_to_test:
    
            cols = ["BCCI_z", metric_z, mod]
            tmp = df[cols].dropna().copy()
    
            if len(tmp) < 20:
                continue
    
            formula = f"{metric_z} ~ BCCI_z * {mod}"
            model = smf.ols(formula, data=tmp).fit()
    
            interaction_term = f"BCCI_z:{mod}"
            if interaction_term not in model.params.index:
                # sometimes statsmodels may store term order differently
                alt_term = f"{mod}:BCCI_z"
                interaction_term = alt_term if alt_term in model.params.index else interaction_term
    
            moderation_results.append({
                "Metric": metric,
                "Moderator": mod,
                "N": len(tmp),
                "Beta_BCCI": model.params.get("BCCI_z", np.nan),
                "P_BCCI": model.pvalues.get("BCCI_z", np.nan),
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
            sep="\t", index=False
        )
    
    print(moderation_results.head(20))
    
    
    
    # ======================================================
    # 14. FULL ADJUSTED MODELS FOR TOP METRICS
    # ======================================================
    
    print("Running full adjusted models")
    
    top_metrics = []
    if len(bcci_metric_results) > 0:
        top_metrics = bcci_metric_results.loc[
            bcci_metric_results["FDR_BCCI"] < 0.10, "Metric"
        ].tolist()
    
    # fallback: take top 5 nominal if none survive
    if len(top_metrics) == 0 and len(bcci_metric_results) > 0:
        top_metrics = bcci_metric_results.head(5)["Metric"].tolist()
    
    full_results = []
    
    base_covars = []
    if "Age_z" in df.columns:
        base_covars.append("Age_z")
    for c in ["Sex", "Exercise_Yes", "HFD_Yes", "APOE"]:
        if c in df.columns:
            base_covars.append(c)
    
    for metric in top_metrics:
        metric_z = metric + "_z"
    
        # main adjusted model
        main_formula = f"{metric_z} ~ BCCI_z"
        if len(base_covars) > 0:
            main_formula += " + " + " + ".join(base_covars)
    
        cols = ["BCCI_z", metric_z] + base_covars
        tmp = df[cols].dropna().copy()
    
        if len(tmp) < 25:
            continue
    
        model_main = smf.ols(main_formula, data=tmp).fit()
    
        full_results.append({
            "Metric": metric,
            "Model": "Adjusted_main",
            "N": len(tmp),
            "Term": "BCCI_z",
            "Beta": model_main.params.get("BCCI_z", np.nan),
            "P": model_main.pvalues.get("BCCI_z", np.nan),
            "R2": model_main.rsquared
        })
    
        # interaction models: one moderator at a time, adjusted for others
        for mod in base_covars:
            other_covars = [c for c in base_covars if c != mod]
            formula = f"{metric_z} ~ BCCI_z * {mod}"
            if len(other_covars) > 0:
                formula += " + " + " + ".join(other_covars)
    
            cols2 = ["BCCI_z", metric_z, mod] + other_covars
            tmp2 = df[cols2].dropna().copy()
    
            if len(tmp2) < 25:
                continue
    
            model_int = smf.ols(formula, data=tmp2).fit()
    
            interaction_term = f"BCCI_z:{mod}"
            if interaction_term not in model_int.params.index:
                alt_term = f"{mod}:BCCI_z"
                interaction_term = alt_term if alt_term in model_int.params.index else interaction_term
    
            full_results.append({
                "Metric": metric,
                "Model": "Adjusted_interaction",
                "N": len(tmp2),
                "Term": interaction_term,
                "Beta": model_int.params.get(interaction_term, np.nan),
                "P": model_int.pvalues.get(interaction_term, np.nan),
                "R2": model_int.rsquared
            })
    
    full_results = pd.DataFrame(full_results)
    
    if len(full_results) > 0:
        full_results["FDR"] = multipletests(full_results["P"].fillna(1.0), method="fdr_bh")[1]
        full_results.to_csv(
            os.path.join(OUTDIR, "BCCI_full_adjusted_models.tsv"),
            sep="\t", index=False
        )
    
    print(full_results.head(20))
    
    
    # ======================================================
    # 15. PLOT SIGNIFICANT INTERACTIONS
    # ======================================================
    
    print("Plotting significant interactions")
    
    # ensure output directory exists
    os.makedirs(OUTDIR, exist_ok=True)
    
    sig_int = moderation_results.copy()
    sig_int = sig_int[sig_int["FDR_Interaction"] < 0.10]
    
    print("Significant interactions:", len(sig_int))
    
    for _, row in sig_int.iterrows():
    
        metric = row["Metric"]
        mod = row["Moderator"]
        metric_z = metric + "_z"
    
        cols = ["BCCI_z", metric_z, mod]
    
        tmp = df[cols].dropna().copy()
    
        if len(tmp) < 20:
            print(f"Skipping {metric} × {mod} (too few samples)")
            continue
    
        plt.figure(figsize=(6,5))
    
        # -------------------------------------
        # binary moderator
        # -------------------------------------
        unique_vals = sorted(tmp[mod].dropna().unique())
    
        if len(unique_vals) <= 2:
    
            for val in unique_vals:
    
                sub = tmp[tmp[mod] == val]
    
                plt.scatter(
                    sub["BCCI_z"],
                    sub[metric_z],
                    s=70,
                    alpha=0.8,
                    label=f"{mod}={val}"
                )
    
                fit = smf.ols(f"{metric_z} ~ BCCI_z", data=sub).fit()
    
                xfit = np.linspace(
                    sub["BCCI_z"].min(),
                    sub["BCCI_z"].max(),
                    100
                )
    
                yfit = fit.params["Intercept"] + fit.params["BCCI_z"] * xfit
    
                plt.plot(xfit, yfit, linewidth=2)
    
            plt.legend()
    
        # -------------------------------------
        # continuous moderator
        # -------------------------------------
        else:
    
            median_val = tmp[mod].median()
    
            tmp["Group"] = np.where(
                tmp[mod] <= median_val,
                "Low",
                "High"
            )
    
            for grp in ["Low", "High"]:
    
                sub = tmp[tmp["Group"] == grp]
    
                plt.scatter(
                    sub["BCCI_z"],
                    sub[metric_z],
                    s=70,
                    alpha=0.8,
                    label=f"{mod} {grp}"
                )
    
                fit = smf.ols(f"{metric_z} ~ BCCI_z", data=sub).fit()
    
                xfit = np.linspace(
                    sub["BCCI_z"].min(),
                    sub["BCCI_z"].max(),
                    100
                )
    
                yfit = fit.params["Intercept"] + fit.params["BCCI_z"] * xfit
    
                plt.plot(xfit, yfit, linewidth=2)
    
            plt.legend()
    
        # -------------------------------------
        # labels
        # -------------------------------------
    
        plt.xlabel("Brain–Cardiac Coupling Index (z)")
        plt.ylabel(f"{metric} (z)")
        plt.title(f"{metric}: BCCI × {mod}")
    
        plt.tight_layout()
    
        # -------------------------------------
        # safe filename
        # -------------------------------------
    
        metric_safe = metric.replace(" ", "_").replace("/", "_")
        mod_safe = mod.replace(" ", "_")
    
        outfile = os.path.join(
            OUTDIR,
            f"interaction_{metric_safe}_by_{mod_safe}.png"
        )
    
        plt.savefig(outfile, dpi=300)
    
        plt.close()
    
        print("Saved:", outfile)
            
        
# ======================================================
# 16. INDIVIDUAL METRIC BRAIN MAPS
# ======================================================

if BRAINMAPS_ENABLED:

    print("Creating individual metric brain maps")

    img = nib.load(ICA_4D_FILE)
    data = img.get_fdata()
    n_components = data.shape[3]

    for metric in res["CardiacMetric"].unique():

        sub = res[res["CardiacMetric"] == metric]

        weights = np.zeros(n_components)

        for _, r in sub.iterrows():

            if r["Network"] not in network_cols:
                continue

            idx = network_cols.index(r["Network"])

            # effect size × statistical support
            p = max(r["P"], 1e-10)
            weights[idx] = r["Beta"] * (-np.log10(p))

        if np.max(np.abs(weights)) == 0:
            continue

        # normalize weights
        weights = weights / np.max(np.abs(weights))

        weighted_map = np.tensordot(
            data,
            weights,
            axes=([3],[0])
        )

        # -----------------------------
        # Save continuous map
        # -----------------------------

        out_file = os.path.join(
            OUTDIR,
            f"cardiac_map_{metric}.nii.gz"
        )

        nib.save(
            nib.Nifti1Image(weighted_map, img.affine),
            out_file
        )

        print("Saved:", out_file)

        # -----------------------------
        # Thresholded map
        # -----------------------------

        thr = np.percentile(np.abs(weighted_map), 80)

        thr_map = weighted_map.copy()
        thr_map[np.abs(thr_map) < thr] = 0

        thr_file = os.path.join(
            OUTDIR,
            f"cardiac_map_{metric}_thr80.nii.gz"
        )

        nib.save(
            nib.Nifti1Image(thr_map, img.affine),
            thr_file
        )

        print("Saved thresholded:", thr_file)



# ======================================================
# 16B. DOMAIN BRAIN MAPS
# ======================================================

if BRAINMAPS_ENABLED:

    print("Creating cardiac domain brain maps")

    domain_defs = {

        "HeartRate": ["Heart_Rate"],

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

    img = nib.load(ICA_4D_FILE)
    data = img.get_fdata()
    n_components = data.shape[3]

    for domain, metrics in domain_defs.items():

        sub = res[res["CardiacMetric"].isin(metrics)]

        weights = np.zeros(n_components)

        for _, r in sub.iterrows():

            if r["Network"] not in network_cols:
                continue

            idx = network_cols.index(r["Network"])

            p = max(r["P"], 1e-10)
            weights[idx] += r["Beta"] * (-np.log10(p))

        if np.max(np.abs(weights)) == 0:
            continue

        weights = weights / np.max(np.abs(weights))

        weighted_map = np.tensordot(
            data,
            weights,
            axes=([3],[0])
        )

        # -----------------------------
        # Save continuous map
        # -----------------------------

        out_file = os.path.join(
            OUTDIR,
            f"cardiac_domain_map_{domain}.nii.gz"
        )

        nib.save(
            nib.Nifti1Image(weighted_map, img.affine),
            out_file
        )

        print("Saved:", out_file)

        # -----------------------------
        # Thresholded map
        # -----------------------------

        thr = np.percentile(np.abs(weighted_map), 80)

        thr_map = weighted_map.copy()
        thr_map[np.abs(thr_map) < thr] = 0

        thr_file = os.path.join(
            OUTDIR,
            f"cardiac_domain_map_{domain}_thr80.nii.gz"
        )

        nib.save(
            nib.Nifti1Image(thr_map, img.affine),
            thr_file
        )

        print("Saved thresholded:", thr_file)
        
# ======================================================
# 17. PLS BRAIN–HEART CIRCUIT MAP
# ======================================================

if BRAINMAPS_ENABLED:

    print("Creating PLS brain–heart circuit map")

    img = nib.load(ICA_4D_FILE)
    data = img.get_fdata()
    n_components = data.shape[3]

    # PLS network weights
    weights = pls.x_weights_.flatten()

    if len(weights) > n_components:
        weights = weights[:n_components]

    # normalize weights
    if np.max(np.abs(weights)) > 0:
        weights = weights / np.max(np.abs(weights))

    # reconstruct spatial circuit
    circuit_map = np.tensordot(
        data,
        weights,
        axes=([3],[0])
    )

    # --------------------------------------
    # Save continuous PLS circuit
    # --------------------------------------

    out_file = os.path.join(
        OUTDIR,
        "brain_heart_circuit_map_PLS.nii.gz"
    )

    nib.save(
        nib.Nifti1Image(circuit_map, img.affine),
        out_file
    )

    print("Saved:", out_file)

    # --------------------------------------
    # Thresholded version for visualization
    # --------------------------------------

    thr = np.percentile(np.abs(circuit_map), 80)

    thr_map = circuit_map.copy()
    thr_map[np.abs(thr_map) < thr] = 0

    thr_file = os.path.join(
        OUTDIR,
        "brain_heart_circuit_map_PLS_thr80.nii.gz"
    )

    nib.save(
        nib.Nifti1Image(thr_map, img.affine),
        thr_file
    )

    print("Saved thresholded:", thr_file)        
# ======================================================
# 18. GENOTYPE-SPECIFIC BRAIN–HEART CIRCUITS
# ======================================================

if BRAINMAPS_ENABLED:

    print("Creating genotype-specific brain–heart circuit maps")

    img = nib.load(ICA_4D_FILE)
    ica_data = img.get_fdata()
    n_components = ica_data.shape[3]

    genotype_groups = {
        "NonE4": df[df["E4_Genotype"] == 0],
        "E4": df[df["E4_Genotype"] == 1]
    }

    maps = {}

    for gname, gdf in genotype_groups.items():

        print("\nProcessing genotype:", gname, "N =", len(gdf))

        weights = np.zeros(n_components)

        for i, net in enumerate(network_cols):

            if i >= n_components:
                break

            # merge cardiac + network data to align indices
            tmp = pd.merge(
                gdf[["Arunno","Heart_Rate"]],
                nets[["Arunno", net]],
                on="Arunno"
            ).dropna()

            if len(tmp) < 10:
                continue

            X = sm.add_constant(tmp[net])
            y = tmp["Heart_Rate"]

            model = sm.OLS(y, X).fit()

            beta = model.params[1]
            p = max(model.pvalues[1], 1e-10)

            # effect size × statistical support
            weights[i] = beta * (-np.log10(p))

        if np.max(np.abs(weights)) == 0:
            continue

        # normalize for stable visualization
        weights = weights / np.max(np.abs(weights))

        # reconstruct spatial circuit
        circuit_map = np.tensordot(
            ica_data,
            weights,
            axes=([3],[0])
        )

        maps[gname] = circuit_map

        # save continuous map
        out_file = os.path.join(
            OUTDIR,
            f"brain_heart_map_{gname}.nii.gz"
        )

        nib.save(
            nib.Nifti1Image(circuit_map, img.affine),
            out_file
        )

        print("Saved:", out_file)

        # ---------------------------------------
        # Thresholded map for visualization
        # ---------------------------------------

        thr = np.percentile(np.abs(circuit_map), 80)

        thr_map = circuit_map.copy()
        thr_map[np.abs(thr_map) < thr] = 0

        thr_file = os.path.join(
            OUTDIR,
            f"brain_heart_map_{gname}_thr80.nii.gz"
        )

        nib.save(
            nib.Nifti1Image(thr_map, img.affine),
            thr_file
        )

        print("Saved thresholded:", thr_file)


    # =================================================
    # Difference map
    # =================================================

    if "E4" in maps and "NonE4" in maps:

        diff_map = maps["E4"] - maps["NonE4"]

        diff_file = os.path.join(
            OUTDIR,
            "brain_heart_map_E4_minus_NonE4.nii.gz"
        )

        nib.save(
            nib.Nifti1Image(diff_map, img.affine),
            diff_file
        )

        print("\nSaved difference map:", diff_file)

        # thresholded difference map

        thr = np.percentile(np.abs(diff_map), 80)

        diff_thr = diff_map.copy()
        diff_thr[np.abs(diff_thr) < thr] = 0

        diff_thr_file = os.path.join(
            OUTDIR,
            "brain_heart_map_E4_minus_NonE4_thr80.nii.gz"
        )

        nib.save(
            nib.Nifti1Image(diff_thr, img.affine),
            diff_thr_file
        )

        print("Saved thresholded difference map:", diff_thr_file)



# ======================================================
# 19. GENOTYPE-SPECIFIC MAPS PER CARDIAC METRIC
# ======================================================

if BRAINMAPS_ENABLED:

    print("\nCreating genotype-specific maps for each cardiac metric")

    img = nib.load(ICA_4D_FILE)
    ica_data = img.get_fdata()
    n_components = ica_data.shape[3]

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

            weights = np.zeros(n_components)

            for i, net in enumerate(network_cols):

                if i >= n_components:
                    break

                tmp = pd.merge(
                    gdf[["Arunno", metric]],
                    nets[["Arunno", net]],
                    on="Arunno"
                ).dropna()

                if len(tmp) < 10:
                    continue

                X = sm.add_constant(tmp[net])
                y = tmp[metric]

                model = sm.OLS(y, X).fit()

                beta = model.params[1]
                p = max(model.pvalues[1], 1e-10)

                weights[i] = beta * (-np.log10(p))

            if np.max(np.abs(weights)) == 0:
                continue

            weights = weights / np.max(np.abs(weights))

            circuit_map = np.tensordot(
                ica_data,
                weights,
                axes=([3],[0])
            )

            maps[gname] = circuit_map

            out_file = os.path.join(
                OUTDIR,
                f"brain_heart_{metric}_{gname}.nii.gz"
            )

            nib.save(
                nib.Nifti1Image(circuit_map, img.affine),
                out_file
            )

            print("Saved:", out_file)

            # thresholded version
            thr = np.percentile(np.abs(circuit_map), 80)

            thr_map = circuit_map.copy()
            thr_map[np.abs(thr_map) < thr] = 0

            thr_file = os.path.join(
                OUTDIR,
                f"brain_heart_{metric}_{gname}_thr80.nii.gz"
            )

            nib.save(
                nib.Nifti1Image(thr_map, img.affine),
                thr_file
            )

            print("Saved thresholded:", thr_file)

        # difference map

        if "E4" in maps and "NonE4" in maps:

            diff_map = maps["E4"] - maps["NonE4"]

            diff_file = os.path.join(
                OUTDIR,
                f"brain_heart_{metric}_E4_minus_NonE4.nii.gz"
            )

            nib.save(
                nib.Nifti1Image(diff_map, img.affine),
                diff_file
            )

            print("Saved difference:", diff_file)

            thr = np.percentile(np.abs(diff_map), 80)

            diff_thr = diff_map.copy()
            diff_thr[np.abs(diff_thr) < thr] = 0

            diff_thr_file = os.path.join(
                OUTDIR,
                f"brain_heart_{metric}_E4_minus_NonE4_thr80.nii.gz"
            )

            nib.save(
                nib.Nifti1Image(diff_thr, img.affine),
                diff_thr_file
            )

            print("Saved thresholded difference:", diff_thr_file)
    # ======================================================
    # 20. TARGETED INTERACTION MODELS
    # ======================================================
    
    print("Running targeted interaction models")
    
    target_models = [
        ("Pump_Function_z", "E4_Genotype"),
        ("Diastolic_Function_z", "Sex_Male"),
        ("Heart_Rate_z", "Exercise_Yes")
    ]
    
    target_results = []
    
    for metric, mod in target_models:
    
        if metric not in df.columns:
            print("Missing metric:", metric)
            continue
    
        if mod not in df.columns:
            print("Missing moderator:", mod)
            continue
    
        cols = ["BCCI_z", metric, mod]
        tmp = df[cols].dropna()
    
        if len(tmp) < 20:
            continue
    
        model = smf.ols(f"{metric} ~ BCCI_z * {mod}", data=tmp).fit()
    
        interaction_term = f"BCCI_z:{mod}"
    
        target_results.append({
            "Metric": metric,
            "Moderator": mod,
            "N": len(tmp),
            "Beta_BCCI": model.params["BCCI_z"],
            "P_BCCI": model.pvalues["BCCI_z"],
            "Beta_Interaction": model.params.get(interaction_term, np.nan),
            "P_Interaction": model.pvalues.get(interaction_term, np.nan),
            "R2": model.rsquared
        })
    
        print("\n", metric, "~ BCCI *", mod)
        print(model.summary())
    
    target_results = pd.DataFrame(target_results)
    
    target_results.to_csv(
        os.path.join(OUTDIR,"targeted_coupling_models.tsv"),
        sep="\t",
        index=False
    )             
    
    
    
    import statsmodels.formula.api as smf

    models = [
        ("pump_function_z", "E4_Genotype"),
        ("left_ventricle_z", "Sex_Male"),
        ("Heart_Rate_z", "Exercise_Yes")
    ]
    
    for metric, mod in models:
    
        if metric not in df.columns:
            print("Missing metric:", metric)
            continue
    
        if mod not in df.columns:
            print("Missing moderator:", mod)
            continue
    
        model = smf.ols(f"{metric} ~ BCCI_z * {mod}", data=df).fit()
    
        print("\n")
        print(metric, "~ BCCI_z *", mod)
        print(model.summary())
        
        
        
        for m in ["Heart_Rate_z","pump_function_z","left_ventricle_z"]:
            r = np.corrcoef(df["BCCI_z"], df[m])[0,1]
            print(m, r)
            
        #Heart_Rate_z -0.28458237674473413
        #pump_function_z -0.2533250113156922
        #left_ventricle_z 0.17321425048320324

        for group, mask in {
            "E4": df["E4_Genotype"] == 1,
            "NonE4": df["E4_Genotype"] == 0
        }.items():

            sub = df[mask]

            model = smf.ols("Heart_Rate_z ~ BCCI_z", data=sub).fit()

            print("\n", group)
            print(model.summary())
            
            
            
            
            
        import numpy as np

        n_perm = 5000
        diffs = []
        
        for i in range(n_perm):
        
            shuffled = df.copy()
            shuffled["E4_Genotype"] = np.random.permutation(shuffled["E4_Genotype"])
        
            beta1 = smf.ols(
                "Heart_Rate_z ~ BCCI_z",
                data=shuffled[shuffled["E4_Genotype"]==1]
            ).fit().params["BCCI_z"]
        
            beta0 = smf.ols(
                "Heart_Rate_z ~ BCCI_z",
                data=shuffled[shuffled["E4_Genotype"]==0]
            ).fit().params["BCCI_z"]
        
            diffs.append(beta1 - beta0)
        
        obs = (-0.147) - (-0.412)
        
        p_perm = np.mean(np.abs(diffs) >= abs(obs))
        
        print("Permutation p =", p_perm)
        
        
        
# ======================================================
# 20. BCCI MODEL WITH COVARIATES + MODEL COMPARISON
# ======================================================

# ======================================================
# FIGURE 4 – BRAIN–CARDIAC COUPLING
# ======================================================

import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

print("\n==============================")
print("Running Figure 4 analysis")
print("==============================")

# ------------------------------------------------------
# Prepare dataframe
# ------------------------------------------------------

df = df_bcci.copy()

# standardize BCCI
df["BCCI_z"] = (df["BCCI"] - df["BCCI"].mean()) / df["BCCI"].std()


# ======================================================
# DEFINE GENOTYPE GROUPS
# ======================================================

geno_map = {
    "E22":"E2",
    "E2HN":"E2",
    "E33":"E3",
    "E3HN":"E3",
    "E44":"E4",
    "E4HN":"E4",
    "KO":"KO"
}

df["Genotype_group"] = df["Genotype"].map(geno_map)

df["Genotype_group"] = df["Genotype_group"].astype("category")
df["Sex"] = df["Sex"].astype("category")

print("\nGenotype counts:")
print(df["Genotype_group"].value_counts())


# ======================================================
# PANEL A – BCCI vs HEART RATE
# ======================================================

print("\nRunning covariate-adjusted model")

X = df[["BCCI_z","Age","Sex","Genotype_group"]]

X = pd.get_dummies(X,drop_first=True)

X = sm.add_constant(X).astype(float)

y = df["Heart_Rate"].astype(float)

print("Design matrix columns:",X.columns)

model = sm.OLS(y,X).fit()

beta = model.params["BCCI_z"]
pval = model.pvalues["BCCI_z"]
r2 = model.rsquared

ci_low,ci_high = model.conf_int().loc["BCCI_z"]

print(model.summary())

# prediction line
xfit = np.linspace(df["BCCI_z"].min(),df["BCCI_z"].max(),200)

# create prediction matrix with same columns
Xpred = pd.DataFrame(0,index=np.arange(len(xfit)),columns=X.columns)

Xpred["const"] = 1
Xpred["BCCI_z"] = xfit
Xpred["Age"] = df["Age"].mean()

Xpred = Xpred.astype(float)

pred = model.get_prediction(Xpred).summary_frame()

plt.figure(figsize=(6,5))

plt.scatter(df["BCCI_z"],y,s=60)

plt.plot(xfit,pred["mean"],linewidth=2)

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
    f"β = {beta:.2f}\n95% CI [{ci_low:.2f},{ci_high:.2f}]\np = {pval:.3g}\nR² = {r2:.2f}",
    transform=plt.gca().transAxes,
    verticalalignment="top"
)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTDIR,"Figure4A_BCCI_vs_HeartRate.png"),
    dpi=300
)

plt.close()


# ======================================================
# PANEL B – MODEL ROBUSTNESS
# ======================================================

print("\nRunning KO sensitivity analysis")

# model with KO
X1 = df[["BCCI_z","Age","Sex","Genotype_group"]]
X1 = pd.get_dummies(X1,drop_first=True)
X1 = sm.add_constant(X1).astype(float)

m1 = sm.OLS(df["Heart_Rate"],X1).fit()

# model without KO
df_noKO = df[df["Genotype_group"]!="KO"]

X2 = df_noKO[["BCCI_z","Age","Sex","Genotype_group"]]
X2 = pd.get_dummies(X2,drop_first=True)
X2 = sm.add_constant(X2).astype(float)

m2 = sm.OLS(df_noKO["Heart_Rate"],X2).fit()

coef_vals = [
    m1.params["BCCI_z"],
    m2.params["BCCI_z"]
]

p_vals = [
    m1.pvalues["BCCI_z"],
    m2.pvalues["BCCI_z"]
]

r2_vals = [
    m1.rsquared,
    m2.rsquared
]

ci1 = m1.conf_int().loc["BCCI_z"]
ci2 = m2.conf_int().loc["BCCI_z"]

errors = [
    coef_vals[0]-ci1[0],
    coef_vals[1]-ci2[0]
]

labels = ["All animals","No KO"]

plt.figure(figsize=(5,4))

bars = plt.bar(
    labels,
    coef_vals,
    yerr=errors,
    capsize=6
)

plt.ylabel("BCCI coefficient (Heart Rate)")
plt.title("Model robustness")

for i,b in enumerate(bars):

    plt.text(
        b.get_x()+b.get_width()/2,
        coef_vals[i],
        f"β={coef_vals[i]:.2f}\np={p_vals[i]:.3g}\nR²={r2_vals[i]:.2f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(OUTDIR,"Figure4B_model_robustness.png"),
    dpi=300
)

plt.close()


# ======================================================
# PANEL C – CARDIAC DOMAIN EFFECTS
# ======================================================

print("\nTesting cardiac domains")

domain_beta = {}
domain_p = {}
domain_r2 = {}
domain_ci = {}

for domain,metrics in CARDIAC_GROUPS.items():

    valid = [m for m in metrics if m in df.columns]

    Xdom = df[valid]

    Xdom = (Xdom - Xdom.mean()) / Xdom.std()

    y = Xdom.mean(axis=1)

    X = df[["BCCI_z","Age","Sex","Genotype_group"]]

    X = pd.get_dummies(X,drop_first=True)

    X = sm.add_constant(X).astype(float)

    model = sm.OLS(y,X).fit()

    domain_beta[domain] = model.params["BCCI_z"]
    domain_p[domain] = model.pvalues["BCCI_z"]
    domain_r2[domain] = model.rsquared
    domain_ci[domain] = model.conf_int().loc["BCCI_z"]

plt.figure(figsize=(6,4))

domains = list(domain_beta.keys())
betas = [domain_beta[d] for d in domains]

errors = [
    betas[i] - domain_ci[domains[i]][0]
    for i in range(len(domains))
]

bars = plt.bar(
    domains,
    betas,
    yerr=errors,
    capsize=6
)

plt.ylabel("BCCI effect")
plt.title("Brain–heart coupling across cardiac domains")

for i,d in enumerate(domains):

    ci_low,ci_high = domain_ci[d]

    plt.text(
        bars[i].get_x()+bars[i].get_width()/2,
        betas[i],
        f"β={betas[i]:.2f}\n95%CI[{ci_low:.2f},{ci_high:.2f}]\np={domain_p[d]:.3g}\nR²={domain_r2[d]:.2f}",
        ha="center",
        va="bottom"
    )

plt.xticks(rotation=25)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTDIR,"Figure4C_domain_effects.png"),
    dpi=300
)

plt.close()

print("\nFigure 4 completed")