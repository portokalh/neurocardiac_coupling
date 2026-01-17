#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 15:39:09 2024

@author: alex
"""
"""
Created on Fri Dec 27 15:08:13 2024

@author: alex
"""

import os
import pandas as pd
from nilearn.glm.second_level import SecondLevelModel
from nilearn import image
from nilearn.glm import threshold_stats_img
from scipy.stats import norm
import numpy as np
from nilearn.image import new_img_like

# --- Step 1: Define paths ---
output_root_dir = "/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/myICA_separate100_clean_0p1/"
mask_img_path = '/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/atlas/reference_maps/chass_atlas_mask_0p1.nii.gz'

# Load the mask image
mask_img = image.load_img(mask_img_path)

# Load subject-level metadata
#metadata_file = "/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/metadata/cardiac_design.csv"
metadata_file = "/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/metadata/cardiac_design_updated.csv"

metadata = pd.read_csv(metadata_file)

metrics = [
    "Diastolic_LV_Volume", "Systolic_LV_Volume", "Heart_Rate",
    "Stroke_Volume", "Ejection_Fraction", "Cardiac_Output",
    "Diastolic_RV", "Systolic_RV", "Diastolic_LA", "Systolic_LA",
    "Diastolic_RA", "Systolic_RA", "Diastolic_Myo"
]
# Loop through each level of Exercise_Yes
exercise_levels = metadata['Exercise_Yes'].unique()

for exercise in exercise_levels:
    print(f"Processing Exercise Level: {exercise}")
    exercise_metadata = metadata[metadata['Exercise_Yes'] == exercise]

    # Loop through each level of Diet_HFD
    diet_levels = exercise_metadata['Diet_HFD'].unique()
    for diet in diet_levels:
        print(f"Processing Diet Level: {diet} for Exercise Level: {exercise}")
        diet_metadata = exercise_metadata[exercise_metadata['Diet_HFD'] == diet]

        for metric in metrics:
            print(f"Processing metric: {metric} for Exercise Level: {exercise}, Diet Level: {diet}")
            
            for j in range(1, 21):  # Iterate over ICA components
                level1_dir = os.path.join(output_root_dir, 'level1', f"level1_ica{j:02}")
                level2_dir = os.path.join(
                    output_root_dir, 
                    'level2', 
                    f"exercise_{exercise}_diet_{diet}", 
                    metric, 
                    f"level2_ica{j:02}"
                )
                
                os.makedirs(level2_dir, exist_ok=True)

                level1_files = [f for f in os.listdir(level1_dir) if f.endswith(".nii.gz")]
                matched_files = []
                matched_metadata = []

                for _, row in diet_metadata.iterrows():
                    subject_id = row["Arunno"]
                    matched_file = [os.path.join(level1_dir, f) for f in level1_files if subject_id in f]
                    if matched_file:
                        matched_files.append(matched_file[0])
                        matched_metadata.append(row)
                
                matched_metadata_df = pd.DataFrame(matched_metadata)
                if matched_metadata_df.empty:
                    print(f"No matched metadata for Exercise: {exercise}, Diet: {diet}, Metric: {metric}, ICA: {j}")
                    continue

                design_matrix = matched_metadata_df[[
                    metric, "Age", "Sex_Male", "Exercise_Yes", "Diet_HFD"
                ]]
                design_matrix = pd.get_dummies(design_matrix, drop_first=True)
                
                second_level_input = [image.load_img(f) for f in matched_files]
                
                # Run SecondLevelModel
                second_level_model = SecondLevelModel(smoothing_fwhm=0.3)
                second_level_model = second_level_model.fit(second_level_input, design_matrix=design_matrix)

                z_map = second_level_model.compute_contrast(second_level_contrast=metric, output_type="z_score")
                z_map.to_filename(os.path.join(level2_dir, f"{metric}_zmap.nii.gz"))

                # Mask the z-map
                masked_z_map = image.math_img("img1 * img2", img1=z_map, img2=mask_img)
                masked_z_map.to_filename(os.path.join(level2_dir, f"{metric}_masked_zmap.nii.gz"))

                # Compute p-value map
                z_data = masked_z_map.get_fdata()
                p_data = 2 * (1 - norm.cdf(np.abs(z_data)))
                p_map = new_img_like(masked_z_map, p_data)
                p_map.to_filename(os.path.join(level2_dir, f"{metric}_pmap.nii.gz"))

                # FDR-corrected Z-map
                thresholded_map, _ = threshold_stats_img(
                    stat_img=masked_z_map, alpha=0.05, height_control='fdr', cluster_threshold=10, two_sided=True
                )
                thresholded_map.to_filename(os.path.join(level2_dir, f"{metric}_masked_z_fdr_corrected.nii.gz"))

                # FDR-corrected P-map
                thresholded_data = thresholded_map.get_fdata()
                p_data_corrected = 2 * (1 - norm.cdf(np.abs(thresholded_data)))  # Two-sided p-values
                p_map_corrected = new_img_like(thresholded_map, p_data_corrected)
                p_map_corrected_path = os.path.join(level2_dir, f"{metric}_p_map_fdr_corrected.nii.gz")
                p_map_corrected.to_filename(p_map_corrected_path)

                print(f"Saved corrected p-value map to: {p_map_corrected_path}")

            print(f"All ICA components processed for Metric: {metric}, Exercise: {exercise}, Diet: {diet}.")
