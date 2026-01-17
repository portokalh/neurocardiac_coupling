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


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Step 2: Load ICA components
canonical_files = sorted([os.path.join(canonical_dir, f) for f in os.listdir(canonical_dir) 
                          if f.endswith('.nii') or f.endswith('.nii.gz')])
seed_img = image.load_img(canonical_files[j])  # Component 1 as seed

output_dir = os.path.join(canonical_dir, 'level1_', j)


#rest_components = image.concat_imgs(canonical_files[1:])  # Combine other components

print(f"Loaded 1 seed component and {len(canonical_files)-1} comparison components.")

# Step 3: Process time series files
time_series_files = sorted([os.path.join(time_series_dir, f) for f in os.listdir(time_series_dir) 
                            if f.endswith('.nii') or f.endswith('.nii.gz')])

# Step 4: Masking setup
masker = NiftiMasker(mask_img=mask_img, standardize=True, memory='nilearn_cache')
seed_time_series = masker.fit_transform(seed_img)  # Extract seed time series

# Step 4: Loop over all fMRI files
fmri_files = [os.path.join(time_series_dir, f) for f in os.listdir(time_series_dir) if f.endswith('.nii.gz')]

for i, fmri_file in enumerate(fmri_files):
    # Extract base filename and preserve substring (e.g., A23091004)
    base_filename = os.path.basename(fmri_file)  # Full filename
    preserved_name = "_".join(base_filename.split('_')[:1])  # Extract first part (e.g., A23091004)
    
    print(f"Processing file {i+1}/{len(fmri_files)}: {fmri_file}")
    
    # Load fMRI image
    fmri_img = image.load_img(fmri_file)

    # Project ICA1 spatial map onto fMRI data to get seed time series
    fmri_time_series = masker.fit_transform(fmri_img)
    seed_weights = masker.transform(seed_img).flatten()  # Flattened ICA1 spatial map
    seed_ts = np.dot(fmri_time_series, seed_weights) / seed_weights.sum()

    # Compute voxel-wise correlation
    correlations = np.array([
        np.corrcoef(seed_ts, fmri_time_series[:, j])[0, 1]
        for j in range(fmri_time_series.shape[1])
    ])

    # Reconstruct and save the connectivity map
    output_file = os.path.join(output_dir, f"seed_connectivity_map_{preserved_name}.nii.gz")
    connectivity_map = masker.inverse_transform(correlations)
    connectivity_map.to_filename(output_file)
    print(f"Saved: {output_file}")

print("All fMRI files processed.")