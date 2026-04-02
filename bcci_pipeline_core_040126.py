#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FINAL BCCI PIPELINE

- Robust metadata/network matching
- Cardiac domain construction
- BCCI construction: unadjusted + adjusted
- LOOCV validation
- Factor-specific network × factor interaction models
- Human-readable network labels in outputs
- Saves merged raw dataset, merged dataset with BCCI, analysis matrix, network dictionary
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import argparse
import datetime as dt
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.random.seed(42)
# =============================================================================
# GLOBAL COLOR SCHEME (USE EVERYWHERE)
# =============================================================================
COL_UNADJ = "#9ecae1"   # light blue
COL_ADJ   = "#08519c"   # dark blue
# =============================================================================
# ARGUMENTS
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--meta", default=None, help="Path to metadata CSV")
parser.add_argument("--nets", default=None, help="Path to network amplitudes TSV")
parser.add_argument("--out", default=None, help="Base output directory")
args = parser.parse_args()

# =============================================================================
# DEFAULT PATHS
# =============================================================================
DEFAULT_META = "/mnt/newStor/paros/paros_WORK/aashika/data/metadata/cardiac_design_updated3.csv"
DEFAULT_NETS = "/mnt/newStor/paros/paros_WORK/aashika/results/march16_2026/network_amplitudes_matrix_COPY.tsv"
DEFAULT_OUT  = "/mnt/newStor/paros/paros_WORK/aashika/results/BCCI_040126"

meta_path = args.meta if args.meta else DEFAULT_META
nets_path = args.nets if args.nets else DEFAULT_NETS
out_base  = args.out  if args.out  else DEFAULT_OUT

if not os.path.exists(meta_path):
    raise FileNotFoundError(f"Metadata file not found: {meta_path}")
if not os.path.exists(nets_path):
    raise FileNotFoundError(f"Network file not found: {nets_path}")

DATE = dt.datetime.now().strftime("%Y%m%d")
OUTDIR = os.path.join(out_base, f"run_{DATE}")
os.makedirs(OUTDIR, exist_ok=True)

print(f"Meta: {meta_path}")
print(f"Nets: {nets_path}")
print(f"Out:  {OUTDIR}")

# =============================================================================
# SETTINGS
# =============================================================================
MIN_N = 20

BCCI_COVARS = ["Age", "Sex_Male", "Mass"]
FACTORS = ["Exercise_Yes", "Diet_HFD", "E4_Genotype", "E2_Genotype", "HN"]

FACTOR_MODELS = {
    "E4_Genotype": ["Age", "Sex_Male", "Mass"],
    "E2_Genotype": ["Age", "Sex_Male", "Mass"],
    "Diet_HFD": ["Age", "Sex_Male", "Mass", "E4_Genotype", "E2_Genotype"],
    "Exercise_Yes": ["Age", "Sex_Male", "Mass", "E4_Genotype", "E2_Genotype", "Diet_HFD"],
    "HN": ["Age", "Sex_Male", "Mass", "E4_Genotype", "E2_Genotype"],
}

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
    "Amp_Net12": "Olfactory Bulb",
}

CARDIAC_GROUPS = {
    "rate_control": ["Heart_Rate"],
    "pumping": ["Stroke_Volume", "Cardiac_Output", "Ejection_Fraction"],
    "systolic_function": [
        "Systolic_LV_Volume", "Systolic_RV", "Systolic_LA", "Systolic_RA", "Systolic_Myo"
    ],
    "diastolic_function": [
        "Diastolic_LV_Volume", "Diastolic_RV", "Diastolic_LA", "Diastolic_RA", "Diastolic_Myo"
    ],
}

# =============================================================================
# HELPERS
# =============================================================================

def bootstrap_weights(X, y, n_boot=500):

    W = []

    for _ in range(n_boot):
        idx = np.random.choice(len(y), len(y), replace=True)
        w = correlation_weights(X[idx], y[idx])
        W.append(w)

    W = np.array(W)

    ci_low = np.nanpercentile(W, 2.5, axis=0)
    ci_high = np.nanpercentile(W, 97.5, axis=0)

    return ci_low, ci_high


def load_weights(domain, suffix):

    f = os.path.join(OUTDIR, f"weights_{domain}_{suffix}.csv")

    if not os.path.exists(f):
        print("Missing:", f)
        return None

    df = pd.read_csv(f)

    if "ci_low" in df.columns:
        df["sig"] = (df["ci_low"] * df["ci_high"] > 0)
    else:
        df["sig"] = True

    return df


def load_loocv():
    f1 = pd.read_csv(os.path.join(OUTDIR, "LOOCV_unadjusted_stats.csv"))
    f2 = pd.read_csv(os.path.join(OUTDIR, "LOOCV_adjusted_stats.csv"))
    return f1.merge(f2, on="domain", suffixes=("_unadj", "_adj"))


def aggregate_system_level(df_delta):

    df = df_delta.copy()

    df["system"] = df["network"].apply(map_network_to_system)

    agg = (
        df.groupby(["factor", "mode", "system"])
        .agg(
            delta=("delta", "mean"),
            ci_low=("ci_low", "mean"),
            ci_high=("ci_high", "mean"),
            sig_frac=("sig", "mean")
        )
        .reset_index()
    )

    return agg



import nibabel as nib

def save_weighted_ica_map(weights, network_cols, ica_4d_path, out_path):

    img = nib.load(ica_4d_path)
    data = img.get_fdata()

    weighted = np.zeros(data.shape[:3])

    for i, net in enumerate(network_cols):

        if net not in weights.index:
            continue

        w = weights.loc[net]

        if np.isnan(w):
            continue

        weighted += data[:, :, :, i] * w

    nib.save(nib.Nifti1Image(weighted, img.affine), out_path)
    
    
def map_network_to_system(net):

    label = NETWORK_LABELS.get(net, "")

    if "Autonomic" in label or "Brainstem" in label:
        return "Autonomic / Brainstem"

    elif "Olfactory" in label:
        return "Olfactory–Basal"

    elif "Cerebellar" in label or "Tecto" in label:
        return "Cerebellar"

    elif "Somatomotor" in label or "Sensorimotor" in label:
        return "Sensorimotor"

    elif "Temporal" in label:
        return "Temporal–Limbic"

    elif "Prefrontal" in label or "Medial" in label or "ACC" in label:
        return "Prefrontal–Cingulate"

    elif "Association" in label:
        return "Association Cortex"

    elif "Thalamo" in label:
        return "Subcortical Relay"

    else:
        return "Other"

def clean_id(x):
    if pd.isna(x):
        return np.nan
    return str(x).replace(".0", "").strip()

def detect_id_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def zscore_array(x):
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(x, dtype=float)
    return (x - np.nanmean(x)) / sd

def safe_mean_z(df_sub):
    if df_sub.shape[0] == 0:
        return pd.Series(dtype=float)
    arr = StandardScaler().fit_transform(df_sub)
    return pd.Series(arr.mean(axis=1), index=df_sub.index)

def residualize_series(df, ycol, covars):
    use_covars = [c for c in covars if c in df.columns]
    cols = [ycol] + use_covars
    tmp = df[cols].dropna().copy()
    out = pd.Series(index=df.index, dtype=float)

    if len(tmp) < MIN_N:
        return out

    X = sm.add_constant(tmp[use_covars], has_constant="add")
    y = tmp[ycol].astype(float)
    model = sm.OLS(y, X).fit()
    out.loc[tmp.index] = model.resid
    return out

def correlation_weights(X, y):
    w = []
    for i in range(X.shape[1]):
        xi = X[:, i]
        if np.std(xi) == 0 or np.std(y) == 0:
            w.append(np.nan)
        else:
            try:
                w.append(pearsonr(xi, y)[0])
            except Exception:
                w.append(np.nan)
    return np.asarray(w, dtype=float)

def interaction_model(df, yvar, net, factor, covars):
    use_covars = [c for c in covars if c in df.columns]
    cols = [yvar, net, factor] + use_covars
    tmp = df[cols].dropna().copy()

    if len(tmp) < MIN_N:
        return None, None

    tmp["net"] = tmp[net].astype(float)
    tmp["factor"] = tmp[factor].astype(float)
    tmp["interaction"] = tmp["net"] * tmp["factor"]

    Xcols = ["net", "factor", "interaction"] + use_covars
    X = sm.add_constant(tmp[Xcols], has_constant="add")
    y = tmp[yvar].astype(float)

    model = sm.OLS(y, X).fit()
    return model, tmp

# =============================================================================
# LOAD + ROBUST MERGE
# =============================================================================
meta = pd.read_csv(meta_path)
nets = pd.read_csv(nets_path, sep="\t")

meta_id = detect_id_column(meta, ["Arunno", "ID", "subject"])
nets_id = detect_id_column(nets, ["Arunno", "ID", "subject"])

if meta_id is None:
    raise ValueError("Could not find an ID column in metadata.")
if nets_id is None:
    raise ValueError("Could not find an ID column in network file.")

meta = meta.rename(columns={meta_id: "MERGE_ID"}).copy()
nets = nets.rename(columns={nets_id: "MERGE_ID"}).copy()

meta["MERGE_ID"] = meta["MERGE_ID"].apply(clean_id)
nets["MERGE_ID"] = nets["MERGE_ID"].apply(clean_id)

common = set(meta["MERGE_ID"]).intersection(set(nets["MERGE_ID"]))

print("\n=== MERGE DEBUG ===")
print("meta shape:", meta.shape)
print("nets shape:", nets.shape)
print("common IDs:", len(common))
print("meta IDs example:", meta["MERGE_ID"].dropna().head().tolist())
print("nets IDs example:", nets["MERGE_ID"].dropna().head().tolist())

if len(common) == 0:
    raise ValueError("No matching IDs between metadata and networks after cleaning.")

df = pd.merge(meta, nets, on="MERGE_ID", how="inner")
print("merged shape:", df.shape)

if len(df) < 10:
    raise ValueError("Too few subjects after merge. Check ID alignment.")

df["ID"] = df["MERGE_ID"]

# =============================================================================
# DETECT NETWORK COLUMNS
# =============================================================================
network_cols = sorted([c for c in df.columns if c.startswith("Amp_Net")])
if len(network_cols) == 0:
    raise ValueError("No network columns found starting with 'Amp_Net'.")

print("Detected network columns:", network_cols)

for c in network_cols:
    df[c] = zscore_array(df[c].values)

# =============================================================================
# SAVE NETWORK LABEL DICTIONARY
# =============================================================================
labels_df = pd.DataFrame({
    "network": list(NETWORK_LABELS.keys()),
    "network_name": list(NETWORK_LABELS.values())
})
labels_df.to_csv(os.path.join(OUTDIR, "network_labels.csv"), index=False)

# =============================================================================
# BUILD CARDIAC DOMAIN SCORES
# =============================================================================
built_domains = []

for domain, vars_ in CARDIAC_GROUPS.items():
    valid = [v for v in vars_ if v in df.columns]
    if len(valid) == 0:
        print(f"Skipping domain {domain}: no variables found")
        continue

    tmp = df[valid].dropna()
    if len(tmp) < MIN_N:
        print(f"Skipping domain {domain}: insufficient non-missing rows ({len(tmp)})")
        continue

    df.loc[tmp.index, f"{domain}_meanz"] = safe_mean_z(tmp)
    built_domains.append(domain)

if len(built_domains) == 0:
    raise ValueError("No cardiac domains could be constructed.")

print("Built domains:", built_domains)

# save merged raw dataset
df.to_csv(os.path.join(OUTDIR, "merged_metadata_networks_raw.csv"), index=False)

# =============================================================================
# BUILD BCCI
# =============================================================================
def build_bcci(df_in, covars=None, suffix=""):
    out = df_in.copy()

    for domain in built_domains:
        source = f"{domain}_meanz"
        target = f"{domain}_target{suffix}"

        if covars is None:
            out[target] = out[source]
        else:
            out[target] = residualize_series(out, source, covars)

    for domain in built_domains:
        target = f"{domain}_target{suffix}"
        bcci_col = f"{domain}_BCCI{suffix}"

        tmp = out[network_cols + [target]].dropna().copy()
        if len(tmp) < MIN_N:
            print(f"Skipping BCCI for {domain}{suffix}: N={len(tmp)}")
            continue

        X = tmp[network_cols].values
        y = tmp[target].values

        w = correlation_weights(X, y)
        valid = np.isfinite(w)
        if valid.sum() == 0:
            print(f"Skipping BCCI for {domain}{suffix}: no valid weights")
            continue

        score = np.dot(X[:, valid], w[valid])
        out.loc[tmp.index, bcci_col] = score

        valid_networks = np.array(network_cols)[valid]
        ci_low, ci_high = bootstrap_weights(X, y)

        weights_df = pd.DataFrame({
            "network": valid_networks,
            "network_name": [NETWORK_LABELS.get(n, n) for n in valid_networks],
            "weight": w[valid],
            "abs_weight": np.abs(w[valid]),
            "ci_low": ci_low[valid],
            "ci_high": ci_high[valid],
            "domain": domain,
            "model": suffix.replace("_", "")
        }).sort_values("abs_weight", ascending=False)
        
        weights_df.to_csv(
            os.path.join(OUTDIR, f"weights_{domain}{suffix}.csv"),
            index=False
        )


        w_series = pd.Series(w, index=network_cols)

        save_weighted_ica_map(
            w_series,
            network_cols,
            "/mnt/newStor/paros/paros_WORK/aashika/data/ICA/4DNetwork/Networks_12_4D.nii.gz",
            os.path.join(OUTDIR, f"BCCI_map_{domain}{suffix}.nii.gz")
        )

    return out

core_covars = [c for c in BCCI_COVARS if c in df.columns]
df_unadj = build_bcci(df, covars=None, suffix="_unadj")
df_adj = build_bcci(df, covars=core_covars, suffix="_adj")

# =============================================================================
# APPEND BCCI TO MERGED DATAFRAME
# =============================================================================
df_final = df.copy()

for domain in built_domains:
    bcci_un = f"{domain}_BCCI_unadj"
    bcci_ad = f"{domain}_BCCI_adj"
    tgt_un = f"{domain}_target_unadj"
    tgt_ad = f"{domain}_target_adj"

    if bcci_un in df_unadj.columns:
        df_final[bcci_un] = df_unadj[bcci_un]
    if bcci_ad in df_adj.columns:
        df_final[bcci_ad] = df_adj[bcci_ad]
    if tgt_un in df_unadj.columns:
        df_final[tgt_un] = df_unadj[tgt_un]
    if tgt_ad in df_adj.columns:
        df_final[tgt_ad] = df_adj[tgt_ad]

# safety checks
if df_final.shape[0] == 0:
    raise ValueError("Merged dataframe is empty.")

for col in [f"{d}_BCCI_unadj" for d in built_domains]:
    if col in df_final.columns and df_final[col].isnull().all():
        raise ValueError(f"{col} is all NaN")

for col in [f"{d}_BCCI_adj" for d in built_domains]:
    if col in df_final.columns and df_final[col].isnull().all():
        raise ValueError(f"{col} is all NaN")

# save merged dataset with BCCI
df_final.to_csv(os.path.join(OUTDIR, "merged_dataset_with_BCCI.csv"), index=False)

# analysis matrix
analysis_cols = (
    ["ID", "MERGE_ID"] +
    network_cols +
    [f"{d}_meanz" for d in built_domains] +
    [f for f in FACTORS if f in df_final.columns] +
    [c for c in BCCI_COVARS if c in df_final.columns] +
    [f"{d}_BCCI_unadj" for d in built_domains if f"{d}_BCCI_unadj" in df_final.columns] +
    [f"{d}_BCCI_adj" for d in built_domains if f"{d}_BCCI_adj" in df_final.columns] +
    [f"{d}_target_unadj" for d in built_domains if f"{d}_target_unadj" in df_final.columns] +
    [f"{d}_target_adj" for d in built_domains if f"{d}_target_adj" in df_final.columns]
)

analysis_cols = list(dict.fromkeys([c for c in analysis_cols if c in df_final.columns]))
analysis_df = df_final[analysis_cols].copy()
analysis_df.to_csv(os.path.join(OUTDIR, "analysis_matrix.csv"), index=False)

# =============================================================================
# LOOCV
# =============================================================================
def run_loocv_bcci(df_model, suffix):
    rows = []
    pred_tables = []

    for domain in built_domains:
        ycol = f"{domain}_target{suffix}"
        tmp = df_model[network_cols + [ycol]].dropna().copy()

        if len(tmp) < MIN_N:
            continue

        X = tmp[network_cols].values
        y = tmp[ycol].values

        loo = LeaveOneOut()
        preds = np.full(len(y), np.nan)

        for train, test in loo.split(X):
            Xtr, Xte = X[train], X[test]
            ytr = y[train]

            w = correlation_weights(Xtr, ytr)
            valid = np.isfinite(w)
            if valid.sum() == 0:
                continue

            preds[test] = np.dot(Xte[:, valid], w[valid])

        mask = np.isfinite(preds) & np.isfinite(y)
        if mask.sum() < MIN_N:
            continue

        rho, p = spearmanr(preds[mask], y[mask])
        ss_res = np.sum((y[mask] - preds[mask]) ** 2)
        ss_tot = np.sum((y[mask] - np.mean(y[mask])) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        rows.append({
            "domain": domain,
            "model": suffix.replace("_", ""),
            "rho_spearman": rho,
            "pval": p,
            "R2": r2,
            "N": int(mask.sum())
        })

        pred_tables.append(pd.DataFrame({
            "domain": domain,
            "y_true": y[mask],
            "y_pred": preds[mask]
        }))

    df_stats = pd.DataFrame(rows)
    df_pred = pd.concat(pred_tables, axis=0, ignore_index=True) if len(pred_tables) else pd.DataFrame()

    return df_stats, df_pred

loocv_stats_unadj, loocv_pred_unadj = run_loocv_bcci(df_unadj, "_unadj")
loocv_stats_adj, loocv_pred_adj = run_loocv_bcci(df_adj, "_adj")

loocv_stats_unadj.to_csv(os.path.join(OUTDIR, "LOOCV_unadjusted_stats.csv"), index=False)
loocv_pred_unadj.to_csv(os.path.join(OUTDIR, "LOOCV_unadjusted_predictions.csv"), index=False)
loocv_stats_adj.to_csv(os.path.join(OUTDIR, "LOOCV_adjusted_stats.csv"), index=False)
loocv_pred_adj.to_csv(os.path.join(OUTDIR, "LOOCV_adjusted_predictions.csv"), index=False)





# =============================================================================
# FACTOR SHIFT (network × factor interaction)
# =============================================================================
def run_factor_shift(df_in, factor, covars, model_label):
    rows = []

    for domain in built_domains:
        yvar = f"{domain}_meanz"
        if yvar not in df_in.columns or factor not in df_in.columns:
            continue

        for net in network_cols:
            model, tmp = interaction_model(df_in, yvar, net, factor, covars)
            if model is None or tmp is None:
                continue

            beta_int = model.params.get("interaction", np.nan)
            p_int = model.pvalues.get("interaction", np.nan)

            rows.append({
                "model_type": model_label,
                "factor": factor,
                "domain": domain,
                "network": net,
                "network_name": NETWORK_LABELS.get(net, net),
                "delta_beta": beta_int,
                "p_raw": p_int,
                "N": len(tmp),
                "covars_used": ", ".join([c for c in covars if c in df_in.columns])
            })

    df_out = pd.DataFrame(rows)
    if len(df_out) == 0:
        return df_out

    df_out["q_fdr"] = np.nan
    for domain in df_out["domain"].unique():
        mask = df_out["domain"] == domain
        pvals = df_out.loc[mask, "p_raw"].values
        valid = np.isfinite(pvals)
        if valid.sum() == 0:
            continue
        _, qvals, _, _ = multipletests(pvals[valid], method="fdr_bh")
        idx = df_out.loc[mask].index[valid]
        df_out.loc[idx, "q_fdr"] = qvals

    return df_out

factor_tables = []

for factor, adj_covars in FACTOR_MODELS.items():
    if factor not in df.columns:
        print(f"Skipping factor {factor}: not found")
        continue

    res_un = run_factor_shift(df, factor, covars=[], model_label="unadjusted")
    if len(res_un):
        res_un.to_csv(os.path.join(OUTDIR, f"stats_shift_{factor}_unadjusted.csv"), index=False)
        factor_tables.append(res_un)

    use_covars = [c for c in adj_covars if c in df.columns]
    res_ad = run_factor_shift(df, factor, covars=use_covars, model_label="adjusted")
    if len(res_ad):
        res_ad.to_csv(os.path.join(OUTDIR, f"stats_shift_{factor}_adjusted.csv"), index=False)
        factor_tables.append(res_ad)

df_shift_all = pd.concat(factor_tables, axis=0, ignore_index=True) if len(factor_tables) else pd.DataFrame()
df_shift_all.to_csv(os.path.join(OUTDIR, "stats_shift_ALL_factors.csv"), index=False)

# =============================================================================
# RUN MANIFEST
# =============================================================================
manifest = {
    "OUTDIR": OUTDIR,
    "N_subjects_merged": int(len(df)),
    "N_networks": int(len(network_cols)),
    "domains_built": ", ".join(built_domains),
    "BCCI_covars": ", ".join(core_covars),
    "meta_file": meta_path,
    "nets_file": nets_path
}
pd.DataFrame([manifest]).to_csv(os.path.join(OUTDIR, "RUN_MANIFEST.csv"), index=False)

# =============================================================================
# LATEST SYMLINK
# =============================================================================
latest = os.path.join(out_base, "latest")
try:
    if os.path.islink(latest) or os.path.exists(latest):
        os.remove(latest)
    os.symlink(OUTDIR, latest)
except Exception:
    pass

print("\nDONE")
print(f"Saved to: {OUTDIR}")




# =============================================================================
# FIGURE 1 — BCCI vs Cardiac Domains with FULL STATS
# =============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

FIGDIR = os.path.join(OUTDIR, "figures")
os.makedirs(FIGDIR, exist_ok=True)

def compute_stats(df, domain, suffix):

    y = df[f"{domain}_meanz"]
    x = df[f"{domain}_BCCI{suffix}"]

    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(tmp) < MIN_N:
        return None

    # Spearman
    rho, pval = spearmanr(tmp["x"], tmp["y"])

    # OLS for beta + CI
    X = sm.add_constant(tmp["x"])
    model = sm.OLS(tmp["y"], X).fit()

    beta = model.params["x"]
    ci_low, ci_high = model.conf_int().loc["x"]

    return {
        "rho": rho,
        "pval": pval,
        "beta": beta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "N": len(tmp)
    }


 
  
    
# =============================================================================
# FIGURE 2 — NETWORK + RISK + LOOCV (FINAL)
# =============================================================================

def move_legend_top(ax):
    leg = ax.get_legend()
    if leg:
        leg.set_bbox_to_anchor((0.5, 1.25))
        leg.set_loc("upper center")
        leg.set_ncol(2)

def plot_loocv_forest(ax):

    df_un = loocv_stats_unadj.copy()
    df_ad = loocv_stats_adj.copy()

    df_un["model"] = "Unadjusted"
    df_ad["model"] = "Adjusted"

    df = pd.concat([df_un, df_ad])

    df["effect"] = np.sign(df["R2"]) * np.sqrt(np.abs(df["R2"]))

    sns.pointplot(
        data=df,
        y="domain",
        x="effect",
        hue="model",
        dodge=0.4,
        join=False,
        ax=ax
    )

    ax.axvline(0, linestyle="--", color="black")
    ax.set_xlabel("Predictive strength (signed √R²)")
    ax.set_title("LOOCV Performance")

    #ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2)
    ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,
    frameon=False
)

# =============================================================================
# FIGURE 1 — BCCI vs Cardiac Domains (FINAL, CORRECTED)
# =============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import spearmanr

FIGDIR = os.path.join(OUTDIR, "figures")
os.makedirs(FIGDIR, exist_ok=True)


def plot_figure1_final(df, OUTDIR):

    domains = ["rate_control", "pumping", "systolic_function", "diastolic_function"]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for j, domain in enumerate(domains):

        for i, suffix in enumerate(["_unadj", "_adj"]):

            ax = axes[i, j]

            x = f"{domain}_BCCI{suffix}"
            y = f"{domain}_meanz"

            # =========================
            # SAFETY CHECK
            # =========================
            if x not in df.columns or y not in df.columns:
                print(f"Skipping {domain}{suffix}: missing columns")
                ax.set_visible(False)
                continue

            df_use = df[[x, y]].dropna()

            if len(df_use) < 10:
                print(f"Skipping {domain}{suffix}: too few samples")
                ax.set_visible(False)
                continue

            # =========================
            # STATS
            # =========================
            rho, p = spearmanr(df_use[x], df_use[y])

            X = np.vstack([np.ones(len(df_use)), df_use[x]]).T
            beta = np.linalg.lstsq(X, df_use[y], rcond=None)[0][1]

            y_pred = X @ np.linalg.lstsq(X, df_use[y], rcond=None)[0]
            ss_res = np.sum((df_use[y] - y_pred) ** 2)
            ss_tot = np.sum((df_use[y] - df_use[y].mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

            # =========================
            # PLOT (BLUE ONLY)
            # =========================
            color = COL_ADJ if suffix == "_adj" else COL_UNADJ
            
            sns.regplot(
                data=df_use,
                x=x,
                y=y,
                ax=ax,
                scatter_kws={"alpha": 0.35, "color": color},
                line_kws={
                    "color": "blue",
                    "linestyle": "-" if suffix == "_adj" else "--",
                    "linewidth": 2
                }
            )

            # =========================
            # TITLE WITH FULL STATS
            # =========================
            ax.set_title(
                f"{domain.replace('_',' ').title()} ({'Adjusted' if suffix=='_adj' else 'Unadjusted'})\n"
                f"ρ={rho:.2f} | p={p:.2e} | β={beta:.2f} | R²={r2:.2f}",
                fontsize=11
            )

            ax.set_xlabel("")
            ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "Figure1_BCCI_domains_FINAL.png"), dpi=300)
    plt.close()

    print("Saved Figure 1 →", FIGDIR)


# =============================================================================
# CALL FIGURE 1
# =============================================================================

print("\nGenerating Figure 1...")

plot_figure1_final(df_final, OUTDIR)

print("Figure 1 DONE")


def plot_figure1_final2(df, OUTDIR):

    domains = ["rate_control", "pumping", "systolic_function", "diastolic_function"]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    panel_labels = list("ABCDEFGH")
    panel_idx = 0

    for j, domain in enumerate(domains):

        for i, suffix in enumerate(["_unadj", "_adj"]):

            ax = axes[i, j]

            # =========================
            # PANEL LABEL
            # =========================
            ax.text(
                -0.18, 1.05, panel_labels[panel_idx],
                transform=ax.transAxes,
                fontsize=14,
                fontweight="bold",
                va="top"
            )
            panel_idx += 1

            x = f"{domain}_BCCI{suffix}"
            y = f"{domain}_meanz"

            # =========================
            # SAFETY CHECK
            # =========================
            if x not in df.columns or y not in df.columns:
                print(f"Skipping {domain}{suffix}: missing columns")
                ax.set_visible(False)
                continue

            df_use = df[[x, y]].dropna()

            if len(df_use) < 10:
                print(f"Skipping {domain}{suffix}: too few samples")
                ax.set_visible(False)
                continue

            # =========================
            # STATS
            # =========================
            rho, p = spearmanr(df_use[x], df_use[y])

            X = np.vstack([np.ones(len(df_use)), df_use[x]]).T
            beta = np.linalg.lstsq(X, df_use[y], rcond=None)[0][1]

            y_pred = X @ np.linalg.lstsq(X, df_use[y], rcond=None)[0]
            ss_res = np.sum((df_use[y] - y_pred) ** 2)
            ss_tot = np.sum((df_use[y] - df_use[y].mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

            # =========================
            # COLOR (CONSISTENT WITH FIGURE 2)
            # =========================
            color = COL_ADJ if suffix == "_adj" else COL_UNADJ

            # =========================
            # PLOT (CRISP, NOT PALE)
            # =========================
            sns.regplot(
                data=df_use,
                x=x,
                y=y,
                ax=ax,
                scatter_kws={
                    "alpha": 0.35,
                    "color": color,
                    "s": 45
                },
                line_kws={
                    "color": color,
                    "linewidth": 2.5
                },
                ci=95
            )

            # slightly strengthen CI visibility
            for coll in ax.collections:
                coll.set_alpha(0.2)

            # =========================
            # TITLE + STATS (CLEAN)
            # =========================
            ax.set_title(
                f"{domain.replace('_',' ').title()} ({'Adj' if suffix=='_adj' else 'Unadj'})",
                fontsize=11
            )

            ax.text(
                0.02, 0.98,
                f"ρ={rho:.2f}\nβ={beta:.2f}\nR²={r2:.2f}\np={p:.1e}",
                transform=ax.transAxes,
                va="top",
                fontsize=9
            )

            ax.set_xlabel("")
            ax.set_ylabel("")

            ax.grid(alpha=0.3)
            sns.despine(ax=ax)

    # =========================
    # FINAL LAYOUT
    # =========================
    plt.subplots_adjust(wspace=0.25, hspace=0.35)

    out = os.path.join(FIGDIR, "Figure1_BCCI_domains_FINAL2.png")
    plt.savefig(out, dpi=300)
    plt.close()

    print("Saved Figure 1 →", out)


plot_figure1_final2(df_final, OUTDIR)



# =============================================================================
# FIGURE 2 — FINAL (ROBUST + PUBLICATION READY)
# =============================================================================

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob
import seaborn as sns

FIGDIR = os.path.join(OUTDIR, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# =============================================================================
# NETWORK SYSTEM GROUPING
# =============================================================================

NETWORK_SYSTEMS = {
    "Amp_Net01": "autonomic_subcortical",
    "Amp_Net09": "autonomic_subcortical",

    "Amp_Net02": "association",
    "Amp_Net03": "association",
    "Amp_Net06": "association",

    "Amp_Net04": "sensorimotor",
    "Amp_Net05": "sensorimotor",

    "Amp_Net07": "limbic_olfactory",
    "Amp_Net12": "limbic_olfactory",

    "Amp_Net08": "temporal_insular",

    "Amp_Net10": "cerebellar",
    "Amp_Net11": "cerebellar"
}

# =============================================================================
# LOAD FUNCTIONS
# =============================================================================

def load_weights(domain, suffix):
    f = os.path.join(OUTDIR, f"weights_{domain}_{suffix}.csv")

    if not os.path.exists(f):
        print("Missing:", f)
        return None

    df = pd.read_csv(f)

    # significance from CI
    if "ci_low" in df.columns:
        df["sig"] = (df["ci_low"] * df["ci_high"] > 0)
    else:
        df["sig"] = True

    return df


def load_delta_all(OUTDIR):

    files = glob.glob(os.path.join(OUTDIR, "stats_shift_*_*.csv"))
    files = [f for f in files if "ALL" not in f]

    rows = []

    for f in files:

        name = os.path.basename(f)
        parts = name.replace(".tsv","").split("_")

        factor = "_".join(parts[2:-1])
        mode = parts[-1]

        df = pd.read_csv(f)

        # --- FIX delta ---
        if "delta_beta" in df.columns:
            df["delta"] = df["delta_beta"]
        if "Delta" in df.columns:
            df["delta"] = df["Delta"]

        if "delta" not in df.columns:
            print("Skipping:", f)
            continue

        # --- stats ---
        if "qval" in df.columns:
            df["q"] = df["qval"]
        else:
            df["q"] = np.nan

        if "sig_fdr" in df.columns:
            df["sig"] = df["sig_fdr"]
        else:
            df["sig"] = df["q"] < 0.05

        df["factor"] = factor
        df["mode"] = mode

        rows.append(df)

    if len(rows) == 0:
        raise ValueError("No valid delta files found.")

    return pd.concat(rows, ignore_index=True)


def load_loocv():
    f1 = pd.read_csv(os.path.join(OUTDIR, "LOOCV_unadjusted_stats.tsv"), sep="\t")
    f2 = pd.read_csv(os.path.join(OUTDIR, "LOOCV_adjusted_stats.tsv"), sep="\t")
    return f1.merge(f2, on="domain", suffixes=("_unadj", "_adj"))


def aggregate_system_level(df):

    df = df.copy()

    df["system"] = df["network"].map(NETWORK_SYSTEMS)

    # ---- handle missing CI columns ----
    has_ci = ("ci_low" in df.columns) and ("ci_high" in df.columns)

    if has_ci:

        df_sys = (
            df.groupby(["system", "mode"])
            .agg(
                delta=("delta", "mean"),
                ci_low=("ci_low", "mean"),
                ci_high=("ci_high", "mean"),
                sig_frac=("sig", "mean")
            )
            .reset_index()
        )

    else:
        print("⚠️ No CI columns found → skipping error bars in Panel D")

        df_sys = (
            df.groupby(["system", "mode"])
            .agg(
                delta=("delta", "mean"),
                sig_frac=("sig", "mean")
            )
            .reset_index()
        )

        # create dummy CI so plotting doesn't crash
        df_sys["ci_low"] = df_sys["delta"]
        df_sys["ci_high"] = df_sys["delta"]

    return df_sys


def plot_weights(ax, df, title):

    import numpy as np

    if df is None or len(df) == 0:
        ax.set_title(title + " (no data)")
        return

    df = df.copy()

    if "abs_weight" not in df.columns:
        df["abs_weight"] = np.abs(df["weight"])

    df = df.sort_values("abs_weight").reset_index(drop=True)

    for i, row in df.iterrows():

        sig = bool(row["sig"]) if "sig" in df.columns else False
        alpha = 1.0 if sig else 0.3

        ax.barh(i, row["weight"], color="#4C72B0", alpha=alpha)

        if sig:
            ax.text(row["weight"], i, " *", va="center", fontsize=12)

    ax.set_yticks(range(len(df)))

    if "network_name" in df.columns:
        ax.set_yticklabels(df["network_name"])
    else:
        ax.set_yticklabels(df["network"])

    ax.set_title(title)
    ax.axvline(0, linestyle="--", linewidth=1, color="black")

    try:
        import seaborn as sns
        sns.despine(ax=ax)
    except:
        pass




# =============================================================================
# RUN
# =============================================================================

# =============================================================================
# FIGURE 2 — FINAL (FULLY ALIGNED + CONSISTENT)
# =============================================================================
# =============================================================================
# FIGURE 2 — FINAL (EQUAL PANELS, CLEAN, WITH SIGNIFICANCE)
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
import seaborn as sns

FIGDIR = os.path.join(OUTDIR, "figures")
os.makedirs(FIGDIR, exist_ok=True)

COL_UNADJ = "#9ecae1"
COL_ADJ   = "#08519c"


import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],

    "axes.titlesize": 16,
    "axes.labelsize": 14,

    "xtick.labelsize": 12,
    "ytick.labelsize": 12,

    "legend.fontsize": 12,

    "axes.titleweight": "regular"
})

# =============================================================================
# PANEL LABEL
# =============================================================================
def add_panel_label(ax, label):
    ax.text(
        -0.18, 1.05, label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top"
    )

# =============================================================================
# LOAD DELTA
# =============================================================================
def load_delta_all(OUTDIR):

    files = glob.glob(os.path.join(OUTDIR, "stats_shift_*_*.csv"))
    files = [f for f in files if "ALL" not in f]

    rows = []

    for f in files:
        df = pd.read_csv(f)

        if "delta_beta" in df.columns:
            df["delta"] = df["delta_beta"]

        if "delta" not in df.columns:
            continue

        name = os.path.basename(f).replace(".csv", "")
        parts = name.split("_")

        df["factor"] = "_".join(parts[2:-1])
        df["mode"] = parts[-1]

        rows.append(df)

    return pd.concat(rows, ignore_index=True)

# =============================================================================
# PANEL A
# =============================================================================
def plot_weights(ax, df, title):

    df = df.copy()

    # readable labels
    df["label"] = df["network"].map(lambda x: NETWORK_LABELS.get(x, x))

    # sort by absolute weight
    df = df.sort_values("abs_weight").reset_index(drop=True)

    for i, row in df.iterrows():

        sig = False
        if "ci_low" in df.columns:
            sig = (row["ci_low"] > 0) or (row["ci_high"] < 0)

        # softer appearance for weights
        color = COL_ADJ if sig else COL_UNADJ
        alpha = 0.85 if sig else 0.4

        ax.barh(i, row["weight"], color=color, alpha=alpha)

        # CI (subtle)
        if "ci_low" in df.columns:
            ax.errorbar(
                row["weight"], i,
                xerr=[[row["weight"] - row["ci_low"]],
                      [row["ci_high"] - row["weight"]]],
                fmt="none",
                ecolor="gray",
                elinewidth=1,
                capsize=2
            )

        # significance star (small + offset)
        if sig:
            offset = 0.02 * np.sign(row["weight"])
            ax.text(row["weight"] + offset, i, "*", va="center", fontsize=9)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["label"], fontsize=9)

    ax.axvline(0, linestyle="--", color="black", linewidth=1)

    ax.set_title(title, fontsize=11)

    sns.despine(ax=ax)

# =============================================================================
# LOOCV EFFECT
# =============================================================================
def compute_loocv_effect_ci(pred_file):

    df = pd.read_csv(pred_file)
    rows = []

    for domain, d in df.groupby("domain"):

        y = d["y_true"].values
        yhat = d["y_pred"].values

        if len(y) < 10:
            continue

        r2 = 1 - np.sum((y-yhat)**2)/np.sum((y-np.mean(y))**2)
        effect = np.sign(r2) * np.sqrt(abs(r2))

        boot = []
        for _ in range(200):
            idx = np.random.choice(len(y), len(y), replace=True)
            yb, yhb = y[idx], yhat[idx]
            r2b = 1 - np.sum((yb-yhb)**2)/np.sum((yb-np.mean(yb))**2)
            boot.append(np.sign(r2b)*np.sqrt(abs(r2b)))

        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

        rows.append(dict(domain=domain, effect=effect, ci_low=ci_low, ci_high=ci_high))

    return pd.DataFrame(rows)

# =============================================================================
# MAIN FIGURE
# =============================================================================
def plot_figure2_final(OUTDIR):

    w_un = pd.read_csv(os.path.join(OUTDIR, "weights_rate_control_unadj.csv"))
    w_ad = pd.read_csv(os.path.join(OUTDIR, "weights_rate_control_adj.csv"))

    df_delta = load_delta_all(OUTDIR)

    loocv_un = compute_loocv_effect_ci(os.path.join(OUTDIR, "LOOCV_unadjusted_predictions.csv"))
    loocv_ad = compute_loocv_effect_ci(os.path.join(OUTDIR, "LOOCV_adjusted_predictions.csv"))

    loocv_un["mode"] = "unadjusted"
    loocv_ad["mode"] = "adjusted"
    loocv = pd.concat([loocv_un, loocv_ad])

    # -------------------------
    # LAYOUT (EQUAL PANELS)
    # -------------------------
    # -------------------------
    # LAYOUT (TRULY UNIFORM)
    # -------------------------
    fig = plt.figure(figsize=(24, 14))
    
    
    
        
    gs = fig.add_gridspec(
        2, 3,   # ← IMPORTANT: 2 rows ONLY
        width_ratios=[1.4, 1.4, 1.4],
        height_ratios=[1, 1]
    )
    
    # LEFT (A uses internal split)
    gs_left = gs[:, 0].subgridspec(2, 1)
    ax_w_un = fig.add_subplot(gs_left[0]) #A
    ax_w_ad = fig.add_subplot(gs_left[1]) #B
    
    # CENTER
    ax_delta = fig.add_subplot(gs[0, 1])   # C
    ax_heat  = fig.add_subplot(gs[1, 1])   # D heatmap
    
    # RIGHT (NOW PERFECTLY MATCHED)
    ax_loocv = fig.add_subplot(gs[0, 2])   # E
    ax_sys   = fig.add_subplot(gs[1, 2])   # F
    # -------------------------
    # PANEL A
    # -------------------------
    plot_weights(ax_w_un, w_un, "Weights (Unadjusted)")
    plot_weights(ax_w_ad, w_ad, "Weights (Adjusted)")
    
    
    
    add_panel_label(ax_w_un, "A")
    add_panel_label(ax_w_ad, "B")
    add_panel_label(ax_delta, "C")
    add_panel_label(ax_heat, "D")
    add_panel_label(ax_loocv, "E")
    add_panel_label(ax_sys, "F")

    # -------------------------
    # PANEL B
    # -------------------------
    # -------------------------
    # PANEL B — FINAL
    # -------------------------
    rows = []
    
    for (factor, mode), d in df_delta.groupby(["factor", "mode"]):
    
        vals = d["delta"].values
        if len(vals) < 5:
            continue
    
        mean = np.mean(vals)
    
        boot = [np.mean(np.random.choice(vals, len(vals), True)) for _ in range(300)]
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    
        rows.append(dict(
            factor=factor,
            mode=mode,
            mean=mean,
            ci_low=ci_low,
            ci_high=ci_high
        ))
    
    df_factor = pd.DataFrame(rows)
    
    # consistent ordering
    factor_order = ["E4_Genotype", "E2_Genotype", "Diet_HFD", "Exercise_Yes", "HN"]
    df_factor["factor"] = pd.Categorical(df_factor["factor"], factor_order, True)
    df_factor = df_factor.sort_values("factor")
    
    y_map = {f: i for i, f in enumerate(factor_order)}
    
    for mode, offset, color in zip(
        ["unadjusted", "adjusted"],
        [-0.2, 0.2],
        [COL_UNADJ, COL_ADJ]
    ):
    
        d = df_factor[df_factor["mode"] == mode]
        y = np.array([y_map[f] for f in d["factor"]])
    
        # main bars (stronger than Panel A)
        ax_delta.barh(
            y + offset,
            d["mean"],
            height=0.35,
            color=color,
            alpha=1.0,
            label=mode
        )
    
        # CI (clear)
        ax_delta.errorbar(
            d["mean"],
            y + offset,
            xerr=[d["mean"] - d["ci_low"], d["ci_high"] - d["mean"]],
            fmt="none",
            ecolor="black",
            elinewidth=1.5,
            capsize=3
        )
    
        # significance stars (larger)
        for xi, yi, lo, hi in zip(d["mean"], y, d["ci_low"], d["ci_high"]):
            if (lo > 0) or (hi < 0):
                offset_star = 0.03 * np.sign(xi)
                ax_delta.text(xi + offset_star, yi + offset, "*", fontsize=12)
    
    # formatting
    ax_delta.set_yticks(range(len(factor_order)))
    ax_delta.set_yticklabels(factor_order)
    
    ax_delta.axvline(0, linestyle="--", color="black")
    
    ax_delta.set_title("Risk Factors Shift Brain–Cardiac Coupling", fontsize=11)
    
    ax_delta.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=2,
        frameon=False
    )
    
    sns.despine(ax=ax_delta)

    # -------------------------
    # HEATMAP
    # -------------------------
    pivot = df_delta.pivot_table(index="network", columns="factor", values="delta", aggfunc="mean")
    pivot.index = pivot.index.map(lambda x: NETWORK_LABELS.get(x,x))

    sns.heatmap(pivot, ax=ax_heat, cmap="coolwarm", center=0, cbar_kws={"label":"Δβ"})
    ax_heat.set_title("Network-Level Effects")
    ax_heat.tick_params(axis='x', rotation=45)
    ax_heat.tick_params(axis='y', labelsize=9)

    # -------------------------
    # PANEL C
    # -------------------------
    domains = loocv["domain"].unique()
    y_map = {d:i for i,d in enumerate(domains)}

    for mode, offset, color in zip(
        ["unadjusted","adjusted"],[-0.2,0.2],[COL_UNADJ,COL_ADJ]
    ):
        d = loocv[loocv["mode"]==mode]
        y = np.array([y_map[x] for x in d["domain"]])

        ax_loocv.barh(y+offset, d["effect"], height=0.35, color=color, label=mode)
        ax_loocv.errorbar(
            d["effect"], y+offset,
            xerr=[d["effect"]-d["ci_low"], d["ci_high"]-d["effect"]],
            fmt="none", ecolor="black", capsize=3
        )

        for xi, yi, lo, hi in zip(d["effect"], y, d["ci_low"], d["ci_high"]):
            if (lo > 0) or (hi < 0):
                ax_loocv.text(xi, yi+offset, "*")

    ax_loocv.axvline(0, linestyle="--")
    ax_loocv.set_yticks(range(len(domains)))
    ax_loocv.set_yticklabels(domains)
    ax_loocv.set_title("LOOCV (signed √R²)")
    ax_loocv.legend(loc="lower center", bbox_to_anchor=(0.5,1.05), ncol=2, frameon=False)
    add_panel_label(ax_loocv, "C")

    # -------------------------
    # PANEL D
    # -------------------------
    df_sys = df_delta.copy()
    df_sys["system"] = df_sys["network"].map(NETWORK_SYSTEMS)

    rows = []
    for (system, mode), d in df_sys.groupby(["system","mode"]):
        vals = d["delta"].values
        if len(vals)<5: continue

        mean = np.mean(vals)
        boot = [np.mean(np.random.choice(vals,len(vals),True)) for _ in range(200)]
        ci_low, ci_high = np.percentile(boot,[2.5,97.5])

        rows.append(dict(system=system,mode=mode,mean=mean,ci_low=ci_low,ci_high=ci_high))

    df_sys_agg = pd.DataFrame(rows)
    systems = df_sys_agg["system"].unique()
    y_map = {s:i for i,s in enumerate(systems)}

    for mode, offset, color in zip(
        ["unadjusted","adjusted"],[-0.2,0.2],[COL_UNADJ,COL_ADJ]
    ):
        d = df_sys_agg[df_sys_agg["mode"]==mode]
        y = np.array([y_map[s] for s in d["system"]])

        ax_sys.barh(y+offset, d["mean"], height=0.35, color=color, label=mode)
        ax_sys.errorbar(
            d["mean"], y+offset,
            xerr=[d["mean"]-d["ci_low"], d["ci_high"]-d["mean"]],
            fmt="none", ecolor="black", capsize=3
        )

        for xi, yi, lo, hi in zip(d["mean"], y, d["ci_low"], d["ci_high"]):
            if (lo > 0) or (hi < 0):
                ax_sys.text(xi, yi+offset, "*")

    ax_sys.axvline(0, linestyle="--")
    ax_sys.set_yticks(range(len(systems)))
    ax_sys.set_yticklabels(systems)
    ax_sys.set_title("System-Level Effects")
    ax_sys.legend(loc="lower center", bbox_to_anchor=(0.5,1.05), ncol=2, frameon=False)
    add_panel_label(ax_sys, "D")

    # -------------------------
    # FINAL
    # -------------------------
    sns.despine()
    plt.subplots_adjust(wspace=0.5, hspace=0.35)

    out = os.path.join(FIGDIR, "Figure2_FINAL.png")
    plt.savefig(out, dpi=300)
    plt.close()

    print("Saved:", out)

# RUN
plot_figure2_final(OUTDIR)




#####################################

##########FIGURE 3 ##############

####################################


import nibabel as nib
import numpy as np
import os


from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from scipy.stats import spearmanr
import numpy as np
import pandas as pd


DOMAINS = list(CARDIAC_GROUPS.keys())
import os

TEMPLATE = os.path.expandvars("$WORK/aashika/data/atlas/chass_atlas.nii")
LABELS = os.path.expandvars("$WORK/aashika/data/atlas/chass_atlas_labels.nii")
print(TEMPLATE)


from nilearn import plotting

def plot_mouse_map(stat_map, template, out_file, title):

    display = plotting.plot_stat_map(
        stat_map,
        bg_img=template,          # ← mouse brain background
        display_mode='ortho',
        cut_coords=(0, 0, 0),     # adjust if needed
        cmap='cold_hot',
        colorbar=True,
        threshold=0,
        black_bg=False,
        title=title
    )

    display.savefig(out_file)
    display.close()
    
def run_loocv_ridge(df_model, suffix):

    results = []
    pred_tables = []
    weight_maps = {}

    for domain in DOMAINS:   # ← clean and explicit

        ycol = f"{domain}_target{suffix}"

        if ycol not in df_model.columns:
            print(f"Missing {ycol}")
            continue

        tmp = df_model[network_cols + [ycol]].dropna()

        if len(tmp) < MIN_N:
            print(f"Too small N for {domain}")
            continue

        X = tmp[network_cols].values
        y = tmp[ycol].values

        loo = LeaveOneOut()
        preds = np.full(len(y), np.nan)
        weights_all = []

        for train_idx, test_idx in loo.split(X):

            Xtr, Xte = X[train_idx], X[test_idx]
            ytr = y[train_idx]

            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xtr)
            Xte = scaler.transform(Xte)

            model = RidgeCV(alphas=np.logspace(-3, 3, 50))
            model.fit(Xtr, ytr)

            preds[test_idx] = model.predict(Xte)
            weights_all.append(model.coef_)

        weights_mean = np.nanmean(weights_all, axis=0)
        weight_maps[domain] = weights_mean

        mask = np.isfinite(preds)

        rho, p = spearmanr(preds[mask], y[mask])

        ss_res = np.sum((y[mask] - preds[mask])**2)
        ss_tot = np.sum((y[mask] - np.mean(y[mask]))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        print(f"{domain}: rho={rho:.2f}, p={p:.3g}, R2={r2:.2f}")

        results.append({
            "domain": domain,
            "rho": rho,
            "pval": p,
            "R2": r2,
            "N": mask.sum()
        })

        pred_tables.append(pd.DataFrame({
            "domain": domain,
            "y_true": y[mask],
            "y_pred": preds[mask]
        }))

    return pd.DataFrame(results), pd.concat(pred_tables), weight_maps

def threshold_map(weighted_map, percentile=97):

    valid = weighted_map[np.isfinite(weighted_map)]

    thr = np.percentile(np.abs(valid), percentile)

    mask = np.abs(weighted_map) >= thr

    thresh_map = np.full_like(weighted_map, np.nan)   # ← KEY CHANGE
    thresh_map[mask] = weighted_map[mask]

    print(f"Threshold = {thr:.3f}, kept voxels = {np.sum(mask)}")

    return thresh_map, thr
def save_maps(weight_maps, ICA_FILE, OUTDIR):

    img = nib.load(ICA_FILE)
    data = img.get_fdata()
    affine = img.affine

    saved_maps = {}

    for domain, weights in weight_maps.items():

        weighted_map = np.zeros(data.shape[:3])

        for i in range(len(weights)):
            weighted_map += data[:,:,:,i] * weights[i]

        # normalize (important)
        weighted_map = (weighted_map - np.mean(weighted_map)) / np.std(weighted_map)

        thresh_map, thr = threshold_map(weighted_map, percentile=95)

        raw_img = nib.Nifti1Image(weighted_map, affine)
        thr_img = nib.Nifti1Image(thresh_map, affine)

        raw_path = os.path.join(OUTDIR, f"BCCI_map_{domain}_ridge.nii.gz")
        thr_path = os.path.join(OUTDIR, f"BCCI_map_{domain}_thresh.nii.gz")

        nib.save(raw_img, raw_path)
        nib.save(thr_img, thr_path)

        saved_maps[domain] = {
            "raw": raw_path,
            "thresh": thr_path,
            "threshold": thr
        }

        print(f"Saved maps for {domain} (thr={thr:.3f})")

    return saved_maps


from nilearn import plotting

def plot_voxel_maps(saved_maps, OUTDIR):

    for domain, paths in saved_maps.items():

        # GLASS BRAIN
        display = plotting.plot_glass_brain(
            paths["thresh"],
            display_mode='lyrz',
            colorbar=True,
            cmap='cold_hot',
            plot_abs=False,
            title=f"{domain} BCCI map (top 5%)"
        )

        glass_path = os.path.join(OUTDIR, f"Figure5_{domain}_glass.png")
        display.savefig(glass_path)
        display.close()

        # SLICE VIEW
        display = plotting.plot_stat_map(
            paths["thresh"],
            display_mode='ortho',
            cut_coords=(0, -2, 2),
            colorbar=True,
            cmap='cold_hot',
            threshold=0,
            title=f"{domain} BCCI map (thresholded)"
        )

        slice_path = os.path.join(OUTDIR, f"Figure5_{domain}_slices.png")
        display.savefig(slice_path)
        display.close()

        print(f"Saved figures for {domain}")
        
        
        
ICA_FILE = "/mnt/newStor/paros/paros_WORK/aashika/data/ICA/4DNetwork/Networks_12_4D.nii.gz"

# 1. run ridge
ridge_stats, ridge_pred, weight_maps = run_loocv_ridge(df_adj, "_adj")

# 2. save maps
saved_maps = save_maps(weight_maps, ICA_FILE, OUTDIR)

# 3. plot figures
plot_voxel_maps(saved_maps, OUTDIR)        





saved_maps = save_maps(weight_maps, ICA_FILE, OUTDIR)
plot_voxel_maps(saved_maps, OUTDIR)



import nibabel as nib
import os

TEMPLATE = os.path.expandvars("$WORK/aashika/data/atlas/chass_atlas.nii")

# LOAD BOTH (IMPORTANT)
atlas_img = nib.load(TEMPLATE)     # ← needed for resampling & plotting
ATLAS = atlas_img.get_fdata()      # ← numeric array for masking




for domain, paths in saved_maps.items():

    plot_mouse_map(
        paths["thresh"],
        TEMPLATE,
        os.path.join(OUTDIR, f"Figure5_{domain}_mouse.png"),
        f"{domain} BCCI map (mouse)"
    )
    
    
    
import nibabel as nib
import numpy as np
import pandas as pd
import os

ATLAS = nib.load(TEMPLATE).get_fdata()    


import pandas as pd
import os

LABELS_TXT = os.path.expandvars(
    "$WORK/aashika/data/atlas/CHASSSYMM3AtlasLegends.xlsx"
)

df_labels = pd.read_excel(LABELS_TXT)

print(df_labels.columns)
print(df_labels.head())

label_map = dict(zip(df_labels["index2"], df_labels["Abbreviation"]))


# ======================================================
# FINAL FIGURE 5 — RATE CONTROL (WITH LEVEL_4)
# ======================================================

import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.image import resample_to_img
from nilearn import plotting
from scipy.stats import spearmanr

# -----------------------------
# PATHS
# -----------------------------
ATLAS_FILE = os.path.expandvars("$WORK/aashika/data/atlas/chass_atlas_labels.nii")
LABELS_XLSX = os.path.expandvars("$WORK/aashika/data/atlas/CHASSSYMM3AtlasLegends.xlsx")

# -----------------------------
# LOAD ATLAS (LABELS!)
# -----------------------------
atlas_img = nib.load(ATLAS_FILE)
ATLAS = atlas_img.get_fdata().astype(int)

atlas_ids = np.unique(ATLAS)
atlas_ids = atlas_ids[atlas_ids > 0]

print("Atlas regions:", len(atlas_ids))

# -----------------------------
# LOAD LABELS
# -----------------------------
df_labels = pd.read_excel(LABELS_XLSX)

df_labels = df_labels[df_labels["index2"].notna()].copy()
df_labels["index2"] = df_labels["index2"].astype(int)

# -----------------------------
# BUILD LABEL MAPS
# -----------------------------
label_map = {}
level4_map = {}

for _, row in df_labels.iterrows():
    rid = int(row["index2"])

    label_map[rid] = row["Abbreviation"]
    level4_map[rid] = row["Level_4"]

# -----------------------------
# RESAMPLE FUNCTION
# -----------------------------
def resample_stat(stat_map_path, atlas_img):
    stat_img = nib.load(stat_map_path)
    return resample_to_img(stat_img, atlas_img, interpolation="continuous")

# -----------------------------
# REGION EXTRACTION
# -----------------------------
def extract_regions(stat_map_path):

    stat_img = resample_stat(stat_map_path, atlas_img)
    stat = stat_img.get_fdata()

    rows = []

    for r in atlas_ids:

        mask = ATLAS == r
        vals = stat[mask]
        vals = vals[np.isfinite(vals)]

        if len(vals) < 20:
            continue

        rows.append({
            "region_id": r,
            "region_name": label_map.get(r, f"R{r}"),
            "Level_4": level4_map.get(r, "Unknown"),
            "n_voxels": len(vals),
            "mean_beta": np.mean(vals),
            "abs_mean": np.mean(np.abs(vals))
        })

    df = pd.DataFrame(rows).sort_values("abs_mean", ascending=False)

    return df, stat_img

# -----------------------------
# FIGURE BUILDER
# -----------------------------
def make_figure5(saved_maps, ridge_pred, OUTDIR):

    domain = "rate_control"

    print("\nBuilding Figure 5...")

    stat_map = saved_maps[domain]["thresh"]

    # extract regions
    df, stat_img = extract_regions(stat_map)

    # save tables
    df.to_csv(os.path.join(OUTDIR, "regions_rate_control.tsv"), sep="\t", index=False)
    df.head(10).to_csv(os.path.join(OUTDIR, "top10_rate_control.tsv"), sep="\t", index=False)

    # split pos / neg
    df_pos = df[df["mean_beta"] > 0].head(10)
    df_neg = df[df["mean_beta"] < 0].head(10)

    # -----------------------------
    # CREATE FIGURE
    # -----------------------------
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)

    # A — voxel map
    ax1 = fig.add_subplot(gs[0, 0])

    vmax = np.nanmax(np.abs(stat_img.get_fdata()))

    plotting.plot_stat_map(
        stat_img,
        bg_img=atlas_img,
        display_mode='ortho',
        cut_coords=(0, -2, 2),
        cmap='cold_hot',
        colorbar=True,
        threshold=1e-6,
        vmax=vmax,
        axes=ax1,
        title="A. Voxel map"
    )

    # B — positive regions
    ax2 = fig.add_subplot(gs[0, 1])

    sns.barplot(
        data=df_pos,
        y="region_name",
        x="abs_mean",
        hue="Level_4",
        dodge=False,
        ax=ax2
    )

    ax2.set_title("B. Positive regions")
    ax2.set_xlabel("|Effect size|")
    ax2.legend(title="System", bbox_to_anchor=(1.05, 1))

    # C — negative regions
    ax3 = fig.add_subplot(gs[1, 0])

    sns.barplot(
        data=df_neg,
        y="region_name",
        x="abs_mean",
        hue="Level_4",
        dodge=False,
        ax=ax3
    )

    ax3.set_title("C. Negative regions")
    ax3.set_xlabel("|Effect size|")
    ax3.legend(title="System", bbox_to_anchor=(1.05, 1))

    # D — prediction
    ax4 = fig.add_subplot(gs[1, 1])

    sub = ridge_pred[ridge_pred["domain"] == domain]

    sns.scatterplot(data=sub, x="y_true", y="y_pred", ax=ax4)
    sns.regplot(data=sub, x="y_true", y="y_pred", scatter=False, ax=ax4)

    rho, p = spearmanr(sub["y_true"], sub["y_pred"])

    ax4.set_title(f"D. Prediction\nρ={rho:.2f}, p={p:.3g}")
    ax4.set_xlabel("Observed")
    ax4.set_ylabel("Predicted")

    plt.tight_layout()

    out_path = os.path.join(OUTDIR, "Figure5_rate_control_final.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)

# -----------------------------
# RUN
# -----------------------------
make_figure5(saved_maps, ridge_pred, OUTDIR)






# ======================================================
# FINAL FIGURE 5 — CLEAN (BLACK A, WHITE B–D)
# ======================================================

import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.image import resample_to_img
from nilearn import plotting
from scipy.stats import spearmanr
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# -----------------------------
# STYLE
# -----------------------------
plt.rcParams.update({"font.size": 10})
sns.set_style("white")

# -----------------------------
# LOAD DATA
# -----------------------------
atlas_img = nib.load(os.path.expandvars("$WORK/aashika/data/atlas/chass_atlas_labels.nii"))
ATLAS = atlas_img.get_fdata().astype(int)

anat_img = nib.load(os.path.expandvars("$WORK/aashika/data/atlas/chass_atlas.nii"))

df_labels = pd.read_excel(os.path.expandvars("$WORK/aashika/data/atlas/CHASSSYMM3AtlasLegends.xlsx"))
df_labels = df_labels[df_labels["index2"].notna()]
df_labels["index2"] = df_labels["index2"].astype(int)

label_map = dict(zip(df_labels["index2"], df_labels["Abbreviation"]))
level4_map = dict(zip(df_labels["index2"], df_labels["Level_4"]))

atlas_ids = np.unique(ATLAS)
atlas_ids = atlas_ids[atlas_ids > 0]

# -----------------------------
# FUNCTIONS
# -----------------------------
def resample_stat(path):
    return resample_to_img(nib.load(path), atlas_img)

def bootstrap_ci(vals, n_boot=200):
    vals = vals[np.isfinite(vals)]
    if len(vals) < 5:
        return np.nan, np.nan
    means = [np.mean(np.random.choice(vals, len(vals), True)) for _ in range(n_boot)]
    return np.percentile(means, 2.5), np.percentile(means, 97.5)

def bilateral_id(r):
    return r - 1000 if r >= 1000 else r

def extract_regions_raw(stat_img):
    stat = stat_img.get_fdata()
    rows = []

    for r in atlas_ids:
        vals = stat[ATLAS == r]
        vals = vals[np.isfinite(vals)]
        if len(vals) < 20:
            continue

        mean = np.mean(vals)
        ci_low, ci_high = bootstrap_ci(vals)
        sig = (ci_low > 0) or (ci_high < 0)

        rows.append({
            "region_id": r,
            "bilateral_id": bilateral_id(r),
            "region": label_map.get(r, f"R{r}"),
            "Level_4": level4_map.get(r, "Unknown"),
            "mean": mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "sig": sig,
            "n_voxels": len(vals)
        })

    return pd.DataFrame(rows)

def collapse_bilateral(df):
    return (
        df.groupby("bilateral_id")
        .agg(
            region=("region", "first"),
            Level_4=("Level_4", "first"),
            mean=("mean", "mean"),
            ci_low=("ci_low", "mean"),
            ci_high=("ci_high", "mean"),
            n_voxels=("n_voxels", "sum"),
            sig=("sig", "max")
        )
        .reset_index()
        .sort_values("mean", ascending=False)
    )

# -----------------------------
# PANEL A (BLACK)
# -----------------------------
def plot_coronal(stat_img, fig, gs):

    vmax = np.nanmax(np.abs(stat_img.get_fdata()))
    coords = np.linspace(8, -4, 12)

    axesA = []

    for i, y in enumerate(coords):

        ax = fig.add_subplot(gs[i // 6, i % 6])
        axesA.append(ax)

        ax.set_facecolor("black")

        plotting.plot_stat_map(
            stat_img,
            bg_img=anat_img,
            display_mode='y',
            cut_coords=[y],
            cmap='cold_hot',
            threshold=1e-6,
            vmax=vmax,
            axes=ax,
            annotate=False,
            colorbar=False,
            black_bg=True
        )

        ax.set_title(f"{y:.1f}", fontsize=7, color="white")

    # colorbar (only panel A axes)
    norm = Normalize(vmin=-vmax, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap='cold_hot')

    cbar = fig.colorbar(
        sm,
        ax=axesA,
        orientation='horizontal',
        location='top',
        fraction=0.05,
        pad=0.02
    )

    cbar.set_label("Effect size")

    fig.text(0.02, 0.93, "A", fontsize=14, weight="bold", color="white")

# -----------------------------
# BAR
# -----------------------------
def plot_bar(ax, df, color, title):

    y = np.arange(len(df))

    err_low = np.maximum(0, df["mean"] - df["ci_low"])
    err_high = np.maximum(0, df["ci_high"] - df["mean"])

    ax.barh(y, df["mean"], xerr=[err_low, err_high], color=color)

    ax.set_yticks(y)
    ax.set_yticklabels(df["region"])
    ax.invert_yaxis()
    ax.set_title(title)

    for i, (_, r) in enumerate(df.iterrows()):
        if r["sig"]:
            ax.text(r["mean"] + 0.02, i, "*")

# -----------------------------
# MAIN
# -----------------------------
def make_figure5(saved_maps, ridge_pred, OUTDIR):

    os.makedirs(OUTDIR, exist_ok=True)

    stat_img = resample_stat(saved_maps["rate_control"]["thresh"])

    df_raw = extract_regions_raw(stat_img)
    df = collapse_bilateral(df_raw)

    # save tables
    df_raw.to_csv(os.path.join(OUTDIR, "regions_raw.tsv"), sep="\t", index=False)
    df.to_csv(os.path.join(OUTDIR, "regions_bilateral.tsv"), sep="\t", index=False)

    df[df["sig"]].to_csv(os.path.join(OUTDIR, "regions_significant.tsv"), sep="\t", index=False)

    systems = (
        df.groupby("Level_4")
        .agg(mean_effect=("mean", "mean"),
             n_regions=("region", "count"),
             total_voxels=("n_voxels", "sum"))
        .sort_values("mean_effect", ascending=False)
        .reset_index()
    )

    systems.to_csv(os.path.join(OUTDIR, "systems.tsv"), sep="\t", index=False)
    systems.head(8).to_csv(os.path.join(OUTDIR, "Table1.tsv"), sep="\t", index=False)

    # -----------------------------
    # FIGURE
    # -----------------------------
    fig = plt.figure(figsize=(16, 10), facecolor="white")  # 🔥 white base
    gs = fig.add_gridspec(3, 6)

    # Panel A (black internally)
    plot_coronal(stat_img, fig, gs)

    # Panel B
    axB = fig.add_subplot(gs[2, 0:2])
    sns.barplot(data=systems.head(6), y="Level_4", x="mean_effect", color="gray", ax=axB)
    axB.set_title("B  System-level effects")

    # Panel C
    axC = fig.add_subplot(gs[2, 2:4])
    plot_bar(axC, df.head(8), "red", "C  Top regions")

    # Panel D
    axD = fig.add_subplot(gs[2, 4:6])
    sub = ridge_pred[ridge_pred["domain"] == "rate_control"]

    sns.scatterplot(data=sub, x="y_true", y="y_pred", ax=axD)
    sns.regplot(data=sub, x="y_true", y="y_pred", scatter=False, ax=axD)

    rho, p = spearmanr(sub["y_true"], sub["y_pred"])

    axD.set_title(f"D  Prediction (ρ={rho:.2f}, p={p:.2g})")
    axD.set_xlabel("Observed cardiac measure")
    axD.set_ylabel("Predicted (LOOCV)")

    plt.tight_layout()

    out = os.path.join(OUTDIR, "Figure5_final.png")
    plt.savefig(out, dpi=300)
    plt.close()

    print("Saved:", out)

# -----------------------------
# RUN
# -----------------------------
make_figure5(saved_maps, ridge_pred, OUTDIR)







# ======================================================
# FINAL PIPELINE:
#  - Figure 5 per domain
#  - Exact merging (bilateral + name) with voxel-pooled CI
#  - Tables saved
#  - Cross-domain heatmap (system × domain)
# ======================================================

import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.image import resample_to_img
from nilearn import plotting
from scipy.stats import spearmanr
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# -----------------------------
# STYLE
# -----------------------------
plt.rcParams.update({"font.size": 10})
sns.set_style("white")

# -----------------------------
# PATHS
# -----------------------------
ATLAS_LABELS = os.path.expandvars("$WORK/aashika/data/atlas/chass_atlas_labels.nii")
ATLAS_ANAT   = os.path.expandvars("$WORK/aashika/data/atlas/chass_atlas.nii")
LABELS_XLSX  = os.path.expandvars("$WORK/aashika/data/atlas/CHASSSYMM3AtlasLegends.xlsx")

# -----------------------------
# LOAD DATA
# -----------------------------
atlas_img = nib.load(ATLAS_LABELS)
ATLAS = atlas_img.get_fdata().astype(int)
anat_img = nib.load(ATLAS_ANAT)

df_labels = pd.read_excel(LABELS_XLSX)
df_labels = df_labels[df_labels["index2"].notna()].copy()
df_labels["index2"] = df_labels["index2"].astype(int)

label_map  = dict(zip(df_labels["index2"], df_labels["Abbreviation"]))
level4_map = dict(zip(df_labels["index2"], df_labels["Level_4"]))

atlas_ids = np.unique(ATLAS)
atlas_ids = atlas_ids[atlas_ids > 0]

# -----------------------------
# UTIL
# -----------------------------
def resample_stat(path):
    return resample_to_img(nib.load(path), atlas_img, interpolation="continuous")

def bootstrap_ci(vals, n_boot=200):
    vals = vals[np.isfinite(vals)]
    if len(vals) < 5:
        return np.nan, np.nan
    means = [np.mean(np.random.choice(vals, len(vals), True)) for _ in range(n_boot)]
    return np.percentile(means, 2.5), np.percentile(means, 97.5)

# -----------------------------
# EXACT MERGING (voxel pooled)
# -----------------------------
def build_voxel_dict(stat_img):
    stat = stat_img.get_fdata()
    voxel_dict = {}
    for r in atlas_ids:
        vals = stat[ATLAS == r]
        vals = vals[np.isfinite(vals)]
        if len(vals) >= 20:
            voxel_dict[int(r)] = vals
    return voxel_dict

def collapse_exact(voxel_dict):
    # bilateral
    bilateral_vox = {}
    for rid, vals in voxel_dict.items():
        bid = rid - 1000 if rid >= 1000 else rid
        bilateral_vox.setdefault(bid, []).append(vals)
    for bid in bilateral_vox:
        bilateral_vox[bid] = np.concatenate(bilateral_vox[bid])

    # merge by name
    name_vox = {}
    for bid, vals in bilateral_vox.items():
        name = label_map.get(bid, f"R{bid}")
        name_vox.setdefault(name, []).append(vals)
    for name in name_vox:
        name_vox[name] = np.concatenate(name_vox[name])

    rows = []
    for name, vals in name_vox.items():
        mean = np.mean(vals)
        ci_low, ci_high = bootstrap_ci(vals)
        sig = (ci_low > 0) or (ci_high < 0)

        # find Level_4
        level4 = None
        for k, v in label_map.items():
            if v == name:
                level4 = level4_map.get(k, "Unknown")
                break

        rows.append({
            "region": name,
            "Level_4": level4,
            "mean": mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "sig": sig,
            "n_voxels": len(vals)
        })

    df = pd.DataFrame(rows).sort_values("mean", ascending=False)
    return df

def clean_systems(df):
    df["system"] = (
        df["Level_4"]
        .str.replace(r"^\d+_", "", regex=True)
        .str.replace("_", " ")
    )
    return df

# -----------------------------
# PANEL A
# -----------------------------
def plot_coronal(stat_img, fig, gs):
    vmax = np.nanmax(np.abs(stat_img.get_fdata()))
    coords = np.linspace(8, -4, 12)

    axesA = []
    for i, y in enumerate(coords):
        ax = fig.add_subplot(gs[i // 6, i % 6])
        axesA.append(ax)
        ax.set_facecolor("black")

        plotting.plot_stat_map(
            stat_img,
            bg_img=anat_img,
            display_mode='y',
            cut_coords=[y],
            cmap='cold_hot',
            threshold=1e-6,
            vmax=vmax,
            axes=ax,
            annotate=False,
            colorbar=False,
            black_bg=True
        )
        ax.set_title(f"{y:.1f}", fontsize=7, color="white")

    norm = Normalize(vmin=-vmax, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap='cold_hot')

    fig.colorbar(
        sm,
        ax=axesA,
        orientation='horizontal',
        location='top',
        fraction=0.05,
        pad=0.02
    )

    fig.text(0.02, 0.93, "A", fontsize=14, weight="bold", color="white")

# -----------------------------
# BAR
# -----------------------------
def plot_bar(ax, df, color, title):
    y = np.arange(len(df))

    err_low  = np.maximum(0, df["mean"] - df["ci_low"])
    err_high = np.maximum(0, df["ci_high"] - df["mean"])

    ax.barh(y, df["mean"], xerr=[err_low, err_high], color=color)

    ax.set_yticks(y)
    ax.set_yticklabels(df["region"])
    ax.invert_yaxis()
    ax.set_title(title)

    # axis labels (requested)
    ax.set_xlabel("Mean effect size (BCCI weight)")
    ax.set_ylabel("Region")

    for i, (_, r) in enumerate(df.iterrows()):
        if r["sig"]:
            ax.text(r["mean"] + 0.02, i, "*", fontsize=10)

# -----------------------------
# MAIN DOMAIN LOOP
# -----------------------------
def run_all_domains(saved_maps, ridge_pred, OUTDIR):

    os.makedirs(OUTDIR, exist_ok=True)

    DOMAINS = [
        "rate_control",
        "pumping",
        "systolic_function",
        "diastolic_function"
    ]

    all_systems = []

    for domain in DOMAINS:

        print(f"\n=== {domain} ===")

        stat_img = resample_stat(saved_maps[domain]["thresh"])

        voxel_dict = build_voxel_dict(stat_img)
        df = collapse_exact(voxel_dict)
        df = clean_systems(df)

        df["domain"] = domain

        domain_dir = os.path.join(OUTDIR, domain)
        os.makedirs(domain_dir, exist_ok=True)

        # ---------------- TABLES
        df.to_csv(os.path.join(domain_dir, f"{domain}_regions.tsv"),
                  sep="\t", index=False)

        df[df["sig"]].to_csv(os.path.join(domain_dir, f"{domain}_significant.tsv"),
                            sep="\t", index=False)

        df.head(10).to_csv(os.path.join(domain_dir, f"{domain}_top10.tsv"),
                           sep="\t", index=False)

        systems = (
            df.groupby("system")
            .agg(
                mean_effect=("mean", "mean"),
                n_regions=("region", "count"),
                total_voxels=("n_voxels", "sum")
            )
            .sort_values("mean_effect", ascending=False)
            .reset_index()
        )

        systems["domain"] = domain
        systems.to_csv(os.path.join(domain_dir, f"{domain}_systems.tsv"),
                       sep="\t", index=False)

        all_systems.append(systems)

        # ---------------- FIGURE 5 PER DOMAIN
        fig = plt.figure(figsize=(16, 10), facecolor="white")
        gs = fig.add_gridspec(3, 6)

        plot_coronal(stat_img, fig, gs)

        axB = fig.add_subplot(gs[2, 0:2])
        sns.barplot(data=systems.head(6), y="system", x="mean_effect",
                    color="gray", ax=axB)
        axB.set_title(f"B  System effects ({domain})")

        axC = fig.add_subplot(gs[2, 2:4])
        plot_bar(axC, df.head(8), "red", f"C  Top regions ({domain})")

        axD = fig.add_subplot(gs[2, 4:6])
        sub = ridge_pred[ridge_pred["domain"] == domain]

        sns.scatterplot(data=sub, x="y_true", y="y_pred", ax=axD)
        sns.regplot(data=sub, x="y_true", y="y_pred", scatter=False, ax=axD)

        rho, p = spearmanr(sub["y_true"], sub["y_pred"])

        axD.set_title(f"D  Prediction (ρ={rho:.2f}, p={p:.2g})")
        axD.set_xlabel("Observed")
        axD.set_ylabel("Predicted")

        plt.tight_layout()

        plt.savefig(os.path.join(domain_dir, f"Figure5_{domain}.png"), dpi=300)
        plt.close()

    # -----------------------------
    # CROSS-DOMAIN HEATMAP
    # -----------------------------
    systems_all = pd.concat(all_systems)

    heatmap_df = systems_all.pivot(
        index="system",
        columns="domain",
        values="mean_effect"
    ).fillna(0)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        heatmap_df,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f"
    )

    plt.title("System × Domain Effects")
    plt.xlabel("Domain")
    plt.ylabel("System")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "cross_domain_heatmap.png"), dpi=300)
    plt.close()

    print("\nSaved cross-domain heatmap")

# -----------------------------
# RUN
# -----------------------------
run_all_domains(saved_maps, ridge_pred, OUTDIR)





# ======================================================
# FINAL PIPELINE — ALL DOMAINS + CLUSTERED HEATMAP
# ======================================================

import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.image import resample_to_img
from nilearn import plotting
from scipy.stats import spearmanr
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

plt.rcParams.update({"font.size": 10})
sns.set_style("white")

# -----------------------------
# LOAD
# -----------------------------
atlas_img = nib.load(os.path.expandvars("$WORK/aashika/data/atlas/chass_atlas_labels.nii"))
ATLAS = atlas_img.get_fdata().astype(int)
anat_img = nib.load(os.path.expandvars("$WORK/aashika/data/atlas/chass_atlas.nii"))

df_labels = pd.read_excel(os.path.expandvars("$WORK/aashika/data/atlas/CHASSSYMM3AtlasLegends.xlsx"))
df_labels = df_labels[df_labels["index2"].notna()]
df_labels["index2"] = df_labels["index2"].astype(int)

label_map = dict(zip(df_labels["index2"], df_labels["Abbreviation"]))
level4_map = dict(zip(df_labels["index2"], df_labels["Level_4"]))

atlas_ids = np.unique(ATLAS)
atlas_ids = atlas_ids[atlas_ids > 0]

# -----------------------------
# UTILS
# -----------------------------
def resample_stat(path):
    return resample_to_img(nib.load(path), atlas_img)

def bootstrap_ci(vals, n_boot=300):
    vals = vals[np.isfinite(vals)]
    if len(vals) < 5:
        return np.nan, np.nan
    means = [np.mean(np.random.choice(vals, len(vals), True)) for _ in range(n_boot)]
    return np.percentile(means, 2.5), np.percentile(means, 97.5)

# -----------------------------
# EXACT MERGING
# -----------------------------
def build_voxel_dict(stat_img):
    stat = stat_img.get_fdata()
    return {
        int(r): stat[ATLAS == r][np.isfinite(stat[ATLAS == r])]
        for r in atlas_ids
        if np.sum(np.isfinite(stat[ATLAS == r])) >= 20
    }

def collapse_exact(voxel_dict):
    # bilateral
    bil = {}
    for rid, vals in voxel_dict.items():
        bid = rid - 1000 if rid >= 1000 else rid
        bil.setdefault(bid, []).append(vals)
    for k in bil:
        bil[k] = np.concatenate(bil[k])

    # merge by name
    name_vox = {}
    for bid, vals in bil.items():
        name = label_map.get(bid, f"R{bid}")
        name_vox.setdefault(name, []).append(vals)
    for k in name_vox:
        name_vox[k] = np.concatenate(name_vox[k])

    rows = []
    for name, vals in name_vox.items():
        mean = np.mean(vals)
        ci_low, ci_high = bootstrap_ci(vals)

        # bootstrap p
        boot = [np.mean(np.random.choice(vals, len(vals), True)) for _ in range(300)]
        p = 2 * min(np.mean(np.array(boot) > 0), np.mean(np.array(boot) < 0))

        sig = "**" if p < 0.01 else "*" if p < 0.05 else ""

        level4 = next((level4_map[k] for k, v in label_map.items() if v == name), "Unknown")

        rows.append({
            "region": name,
            "Level_4": level4,
            "mean": mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p": p,
            "sig": sig,
            "n_voxels": len(vals)
        })

    return pd.DataFrame(rows).sort_values("mean", ascending=False)

def clean_systems(df):
    df["system"] = (
        df["Level_4"]
        .str.replace(r"^\d+_", "", regex=True)
        .str.replace("_", " ")
    )
    return df

# -----------------------------
# SYSTEM STATS
# -----------------------------
def compute_system_stats(df):
    rows = []

    for system in df["system"].unique():
        vals = df[df["system"] == system]["mean"].values

        mean = np.mean(vals)
        boot = [np.mean(np.random.choice(vals, len(vals), True)) for _ in range(500)]

        ci_low = np.percentile(boot, 2.5)
        ci_high = np.percentile(boot, 97.5)

        p = 2 * min(np.mean(np.array(boot) > 0), np.mean(np.array(boot) < 0))
        sig = "**" if p < 0.01 else "*" if p < 0.05 else ""

        rows.append({
            "system": system,
            "mean_effect": mean,
            "p": p,
            "sig": sig
        })

    return pd.DataFrame(rows)

# -----------------------------
# FIGURE PANELS
# -----------------------------
def plot_coronal(stat_img, fig, gs):
    vmax = np.nanmax(np.abs(stat_img.get_fdata()))
    coords = np.linspace(8, -4, 12)
    axes = []

    for i, y in enumerate(coords):
        ax = fig.add_subplot(gs[i // 6, i % 6])
        axes.append(ax)
        ax.set_facecolor("black")

        plotting.plot_stat_map(
            stat_img, bg_img=anat_img,
            display_mode='y', cut_coords=[y],
            cmap='cold_hot', vmax=vmax,
            axes=ax, annotate=False, colorbar=False, black_bg=True
        )

    norm = Normalize(vmin=-vmax, vmax=vmax)
    fig.colorbar(ScalarMappable(norm=norm, cmap='cold_hot'),
                 ax=axes, orientation='horizontal', location='top')

def plot_bar(ax, df, color):
    y = np.arange(len(df))
    ax.barh(y, df["mean"], color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(df["region"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean effect size")

# -----------------------------
# MAIN
# -----------------------------
def run_pipeline(saved_maps, ridge_pred, OUTDIR):

    DOMAINS = ["rate_control", "pumping", "systolic_function", "diastolic_function"]
    all_systems = []

    for domain in DOMAINS:

        stat_img = resample_stat(saved_maps[domain]["thresh"])

        voxel_dict = build_voxel_dict(stat_img)
        df = collapse_exact(voxel_dict)
        df = clean_systems(df)

        df.to_csv(os.path.join(OUTDIR, f"{domain}_regions.tsv"), sep="\t", index=False)

        systems = compute_system_stats(df)
        systems["domain"] = domain
        all_systems.append(systems)

        # ---- FIGURE 5
        fig = plt.figure(figsize=(16, 10), facecolor="white")
        gs = fig.add_gridspec(3, 6)

        plot_coronal(stat_img, fig, gs)

        axB = fig.add_subplot(gs[2, 0:2])
        sns.barplot(data=systems, y="system", x="mean_effect", color="gray", ax=axB)

        axC = fig.add_subplot(gs[2, 2:4])
        plot_bar(axC, df.head(8), "red")

        axD = fig.add_subplot(gs[2, 4:6])
        sub = ridge_pred[ridge_pred["domain"] == domain]
        sns.regplot(data=sub, x="y_true", y="y_pred", ax=axD)

        rho, p = spearmanr(sub["y_true"], sub["y_pred"])
        axD.set_title(f"ρ={rho:.2f}, p={p:.2g}")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"Figure5_{domain}.png"), dpi=300)
        plt.close()

    # -----------------------------
    # CLUSTERED HEATMAP
    # -----------------------------
    systems_all = pd.concat(all_systems)

    heat = systems_all.pivot(index="system", columns="domain", values="mean_effect")
    sig  = systems_all.pivot(index="system", columns="domain", values="sig")

    g = sns.clustermap(
        heat,
        cmap="coolwarm",
        center=0,
        figsize=(8, 8),
        annot=False
    )

    ax = g.ax_heatmap

    for i in range(sig.shape[0]):
        for j in range(sig.shape[1]):
            s = sig.iloc[i, j]
            if s != "":
                ax.text(j + 0.5, i + 0.5, s,
                        ha='center', va='center', fontsize=10)

    plt.savefig(os.path.join(OUTDIR, "cross_domain_clustered.png"), dpi=300)
    plt.close()

    print("Pipeline complete.")

# -----------------------------
# RUN
# -----------------------------
run_pipeline(saved_maps, ridge_pred, OUTDIR)
