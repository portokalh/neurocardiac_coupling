#!/usr/bin/env python3

import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import plotting

# =====================================================
# PATHS
# =====================================================

ICA_DIR = "/mnt/newStor/paros/paros_WORK/aashika/data/ICA/3DNetworks"

RESULTS_FILE = "/mnt/newStor/paros/paros_WORK/aashika/results/network_cardiac_postproc/network_cardiac_results_hierFDR.tsv"

OUTDIR = "/mnt/newStor/paros/paros_WORK/aashika/results/brain_cardiac_maps"

os.makedirs(OUTDIR, exist_ok=True)

# =====================================================
# LOAD NETWORK–CARDIAC RESULTS
# =====================================================

print("Loading:", RESULTS_FILE)

res = pd.read_csv(RESULTS_FILE, sep="\t")

print("Rows:", len(res))

# =====================================================
# COMPUTE NETWORK WEIGHTS
# =====================================================

beta = res.groupby("Network")["Beta"].mean()

weights = np.abs(beta)
weights = weights / weights.sum()

print("\nNetwork weights:")
print(weights)

# =====================================================
# BUILD NETWORK WEIGHT DICTIONARY
# =====================================================

weights_dict = {}

for net, w in weights.items():

    # Amp_MidlineAutonomicAxis → MidlineAutonomicAxis
    network = net.replace("Amp_", "")

    weights_dict[network] = w

print("\nNetwork dictionary keys:")
print(weights_dict.keys())

# =====================================================
# LOAD ICA FILES
# =====================================================

ica_files = sorted(glob.glob(os.path.join(ICA_DIR,"*.nii.gz")))

print("\nFound", len(ica_files), "ICA maps")

# =====================================================
# BUILD WEIGHTED MAP
# =====================================================

weighted_map = None
ref_img = None

for f in ica_files:

    fname = os.path.basename(f)

    parts = fname.split("_")

    if len(parts) < 2:
        continue

    network_name = parts[1]

    if network_name not in weights_dict:
        print("Skipping:", network_name)
        continue

    weight = weights_dict[network_name]

    print("Using", network_name, "weight", weight)

    img = nib.load(f)
    data = img.get_fdata()

    # -----------------------------------------------
    # THRESHOLD ICA MAP (top 5% magnitude)
    # -----------------------------------------------

    thr = np.percentile(np.abs(data),95)
    data[np.abs(data) < thr] = 0

    # -----------------------------------------------
    # BUILD WEIGHTED SUM
    # -----------------------------------------------

    if weighted_map is None:

        weighted_map = weight * data
        ref_img = img

    else:

        weighted_map += weight * data


# =====================================================
# CHECK RESULT
# =====================================================

if weighted_map is None:
    raise RuntimeError("No ICA maps matched network names")

# =====================================================
# SAVE MAP
# =====================================================

out_file = os.path.join(
    OUTDIR,
    "brain_cardiac_coupling_map.nii.gz"
)

img_out = nib.Nifti1Image(
    weighted_map,
    ref_img.affine,
    ref_img.header
)

nib.save(img_out,out_file)

print("\nSaved NIfTI:", out_file)

# =====================================================
# VISUALIZE MAP
# =====================================================

thr = np.percentile(np.abs(weighted_map),95)

display = plotting.plot_stat_map(
    img_out,
    display_mode="ortho",
    threshold=thr,
    title="Brain–Cardiac Coupling Network"
)

png_file = os.path.join(
    OUTDIR,
    "brain_cardiac_coupling_map.png"
)

display.savefig(png_file)

print("Saved figure:", png_file)

plotting.show()

print("\nFinished.")