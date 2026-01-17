#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated on Dec 17 2024
@author: alex
"""

import os
import numpy as np
from nilearn import image, masking
from nilearn.input_data import NiftiMasker
from nilearn import image
from nibabel import load

# Step 1: Define directories
canonical_dir = '/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/myICA_separate100'
time_series_dir = '/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/errts_trim/'
mask_img = load('/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/atlas/reference_maps/chass_atlas_mask.nii') 

output_root_dir = "/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/myICA_separate100/"

# Step 2: Load ICA components
canonical_files = sorted([os.path.join(canonical_dir, f) for f in os.listdir(canonical_dir) 
                          if f.endswith('.nii') or f.endswith('.nii.gz')])

for j in range(20, 21):  # Loop through ICA components 1 to 5
    ica_suffix = f"ica{j:02}"  # Format ICA index as ica01, ica02, etc.

    # Create dynamically named output directory
    output_dir = os.path.join(output_root_dir, f"level1_{ica_suffix}")
    os.makedirs(output_dir, exist_ok=True)

    # Load seed ICA component
    seed_img = image.load_img(canonical_files[j-1])  # Select ICA component (0-based index)
    print(f"Processing ICA component {j}: {canonical_files[j-1]}")

    # Prepare masker for seed and time series processing
    masker = NiftiMasker(mask_img=mask_img, standardize=True, memory='nilearn_cache')
    seed_time_series = masker.fit_transform(seed_img)  # Extract seed time series

    # --- Step 2: Process fMRI Time Series Files ---
    fmri_files = sorted([os.path.join(time_series_dir, f) for f in os.listdir(time_series_dir) 
                         if f.endswith('.nii.gz')])
    for i, fmri_file in enumerate(fmri_files):
        # Extract base filename and preserve substring (e.g., A23091004)
        base_filename = os.path.basename(fmri_file)
        preserved_name = "_".join(base_filename.split('_')[:1])

        print(f"Processing file {i+1}/{len(fmri_files)}: {fmri_file}")
        
        # Load fMRI image
        fmri_img = image.load_img(fmri_file)

        # Extract time series for seed and fMRI
        fmri_time_series = masker.fit_transform(fmri_img)
        seed_weights = masker.transform(seed_img).flatten()
        
        # Compute seed time series projection
        seed_ts = np.dot(fmri_time_series, seed_weights) / seed_weights.sum()

        # Compute voxel-wise correlation
        correlations = np.array([
            np.corrcoef(seed_ts, fmri_time_series[:, voxel])[0, 1]
            for voxel in range(fmri_time_series.shape[1])
        ])

        # Reconstruct and save the connectivity map
        output_file = os.path.join(output_dir, f"seed_connectivity_map_{preserved_name}.nii.gz")
        connectivity_map = masker.inverse_transform(correlations)
        connectivity_map.to_filename(output_file)
        print(f"Saved: {output_file}")

print("All ICA components and fMRI files processed.")