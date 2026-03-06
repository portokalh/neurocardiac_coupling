#!/usr/bin/env python3

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import os

# ----------------------------------------------------
# PATHS
# ----------------------------------------------------

COUPLING_MAP = "/mnt/newStor/paros/paros_WORK/aashika/results/brain_cardiac_maps/brain_cardiac_coupling_map.nii.gz"

ATLAS_LABELS = "/mnt/newStor/paros/paros_WORK/aashika/chass/chass_symmetric3_labels_PLI_res.nii.gz"

ATLAS_LEGEND = "/mnt/newStor/paros/paros_WORK/aashika/chass/CHASSSYMM3AtlasLegends021826.csv"

OUT_DIR = "/mnt/newStor/paros/paros_WORK/aashika/results/brain_cardiac_maps/"

os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------

print("Loading coupling map")
coupling = nib.load(COUPLING_MAP).get_fdata()

print("Loading atlas labels")
atlas = nib.load(ATLAS_LABELS).get_fdata().astype(int)

print("Loading atlas legend")
legend = pd.read_csv(ATLAS_LEGEND)

legend = legend.rename(columns={"index2":"LabelValue"})

# ----------------------------------------------------
# THRESHOLD COUPLING MAP
# ----------------------------------------------------

thr = np.percentile(np.abs(coupling),95)

mask = np.abs(coupling) > thr

print("Coupling voxels:",mask.sum())

# ----------------------------------------------------
# REGION ANALYSIS
# ----------------------------------------------------

labels = np.unique(atlas)
labels = labels[labels>0]

rows = []

for lab in labels:

    region_mask = atlas == lab

    region_voxels = region_mask.sum()

    if region_voxels == 0:
        continue

    overlap_mask = region_mask & mask

    overlap_voxels = overlap_mask.sum()

    coverage = overlap_voxels / region_voxels

    region_values = np.abs(coupling[region_mask])

    contribution_strength = region_values.sum()

    mean_strength = region_values.mean()

    rows.append({
        "LabelValue": lab,
        "OverlapVoxels": overlap_voxels,
        "RegionVoxels": region_voxels,
        "CoverageFraction": coverage,
        "ContributionStrength": contribution_strength,
        "MeanStrength": mean_strength
    })

df = pd.DataFrame(rows)

# ----------------------------------------------------
# MERGE WITH ATLAS LEGEND
# ----------------------------------------------------

df = df.merge(
    legend[[
        "LabelValue",
        "RegionName",
        "Structure",
        "Volume_vox",
        "Volume_mm3"
    ]],
    on="LabelValue",
    how="left"
)

# ----------------------------------------------------
# RANK REGIONS
# ----------------------------------------------------

df = df.sort_values("ContributionStrength",ascending=False)

df.insert(0,"Rank",range(1,len(df)+1))

# ----------------------------------------------------
# SAVE CSV
# ----------------------------------------------------

csv_file = os.path.join(OUT_DIR,"coupling_region_ranking.csv")

df.to_csv(csv_file,index=False)

print("Saved ranking table:",csv_file)

# ----------------------------------------------------
# BAR PLOT — CONTRIBUTION STRENGTH
# ----------------------------------------------------

top = df.head(20)

plt.figure(figsize=(8,6))

plt.barh(
    top["RegionName"][::-1],
    top["ContributionStrength"][::-1]
)

plt.xlabel("Coupling Contribution Strength")
plt.title("Top Brain Regions in Neuro-Cardiac Coupling")

plt.tight_layout()

fig1 = os.path.join(OUT_DIR,"top_regions_contribution.png")

plt.savefig(fig1,dpi=300)

print("Saved figure:",fig1)

# ----------------------------------------------------
# BAR PLOT — COVERAGE FRACTION
# ----------------------------------------------------

plt.figure(figsize=(8,6))

plt.barh(
    top["RegionName"][::-1],
    top["CoverageFraction"][::-1]
)

plt.xlabel("Coverage Fraction")
plt.title("Regional Coverage of Neuro-Cardiac Coupling")

plt.tight_layout()

fig2 = os.path.join(OUT_DIR,"top_regions_coverage.png")

plt.savefig(fig2,dpi=300)

print("Saved figure:",fig2)

plt.show()

print("\nDone.")


# ----------------------------------------------------
# SORT BY CONTRIBUTION
# ----------------------------------------------------

top = df.sort_values("ContributionStrength", ascending=False).head(20)

regions = top["RegionName"]

# ----------------------------------------------------
# FIGURE 1 — CONTRIBUTION
# ----------------------------------------------------

plt.figure(figsize=(8,6))

plt.barh(
    regions[::-1],
    top["ContributionStrength"][::-1]
)

plt.xlabel("Coupling Contribution Strength")
plt.title("Top Brain Regions in Neuro-Cardiac Coupling")

plt.tight_layout()

plt.savefig(os.path.join(OUT_DIR,"top_regions_contribution.png"),dpi=300)


# ----------------------------------------------------
# FIGURE 2 — COVERAGE (same ordering)
# ----------------------------------------------------

plt.figure(figsize=(8,6))

plt.barh(
    regions[::-1],
    top["CoverageFraction"][::-1]
)

plt.xlabel("Coverage Fraction")
plt.title("Spatial Coverage of Neuro-Cardiac Coupling")

plt.tight_layout()

plt.savefig(os.path.join(OUT_DIR,"top_regions_coverage.png"),dpi=300)