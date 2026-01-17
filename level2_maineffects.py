#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
metadata_file = "/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/metadata/cardiac_design_updated3.csv"
#output_metadata_file = "/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/metadata/cardiac_design_updated3.csv"
metadata = pd.read_csv(metadata_file)

# # Create new columns based on the Genotype column
metadata['E2_genotype'] = metadata['Genotype'].apply(lambda x: 1 if x == 'E22' or x == 'E22HN' else 0)
metadata['E3_genotype'] = metadata['Genotype'].apply(lambda x: 1 if x == 'E33' or x == 'E33HN' else 0)
metadata['E4_genotype'] = metadata['Genotype'].apply(lambda x: 1 if x == 'E44' or x == 'E44HN' else 0)
metadata['HN'] = metadata['Genotype'].apply(lambda x: 1 if 'HN' in str(x) else 0)

'''
metadata = metadata.rename(columns={
    "E2_genotype": "E2_Genotype",
    "E3_genotype": "E3_Genotype",
    "E4_genotype": "E4_Genotype"
})
'''

# # Add Age_cat column based on median age
median_age = metadata['Age'].median()
metadata['Age_cat'] = metadata['Age'].apply(lambda x: 0 if x < median_age else 1)

# # Save the updated metadata file
# metadata.to_csv(output_metadata_file, index=False)
# print(f"Updated metadata saved to: {output_metadata_file}")

metrics = [
    "Diastolic_LV_Volume", "Systolic_LV_Volume", "Heart_Rate",
    "Stroke_Volume", "Ejection_Fraction", "Cardiac_Output",
    "Diastolic_RV", "Systolic_RV", "Diastolic_LA", "Systolic_LA",
    "Diastolic_RA", "Systolic_RA", "Diastolic_Myo"
]

# Main effects and interactions to analyze
main_effects = ["Age_cat", "E4_Genotype", "Sex_Male", "HN", "Diet_HFD", "Exercise_Yes", "E2_Genotype", "E3_Genotype"]




interactions = [
    "Diet:Exercise_Yes",
    "Diet:Age_cat",
    "Diet:Sex_Male",
    "Exercise_Yes:Age_cat",
    "Exercise_Yes:Sex_Male",
    "Age_cat:Sex_Male"
]

#effects_to_analyze = main_effects + interactions
effects_to_analyze = main_effects

for effect in effects_to_analyze:
    print(f"Processing effect: {effect}")
    
    for metric in metrics:
        print(f"Processing metric: {metric} for effect: {effect}")

        for j in range(1, 21):  # Iterate over ICA components
            level1_dir = os.path.join(output_root_dir, 'level1', f"level1_ica{j:02}")
            level2_dir = os.path.join(output_root_dir, 'level2', f"{effect}", metric, f"level2_ica{j:02}")

            os.makedirs(level2_dir, exist_ok=True)

            level1_files = [f for f in os.listdir(level1_dir) if f.endswith(".nii.gz")]
            matched_files = []
            matched_metadata = []

            for _, row in metadata.iterrows():
                subject_id = row["Arunno"]
                matched_file = [os.path.join(level1_dir, f) for f in level1_files if subject_id in f]
                if matched_file:
                    matched_files.append(matched_file[0])
                    matched_metadata.append(row)

            matched_metadata_df = pd.DataFrame(matched_metadata)
            if matched_metadata_df.empty:
                print(f"No matched metadata for Metric: {metric}, Effect: {effect}, ICA: {j}")
                continue

            # Prepare the design matrix for the current effect
            if ":" in effect:  # Interaction term
                factors = effect.split(":")
                for factor in factors:
                    if factor not in matched_metadata_df.columns:
                        print(f"Missing factor {factor} for interaction {effect}. Skipping.")
                        continue
                # Create interaction term as a product of factors
                matched_metadata_df[effect] = matched_metadata_df[factors[0]] * matched_metadata_df[factors[1]]
                design_matrix = matched_metadata_df[[metric] + factors + [effect]]
            else:  # Main effect
                design_matrix = matched_metadata_df[[metric, effect]]

            # Convert to numeric, coercing errors to NaN, then drop NaNs
            design_matrix = design_matrix.apply(pd.to_numeric, errors='coerce').dropna()

            # Ensure there are no remaining non-numeric values
            if design_matrix.empty or design_matrix.isnull().values.any():
                print(f"Invalid design matrix for Metric: {metric}, Effect: {effect}, ICA: {j}")
                continue

            # Add intercept 
            #design_matrix["Intercept"] = 1
            second_level_input = [image.load_img(f) for f in matched_files]

            # Run SecondLevelModel
            second_level_model = SecondLevelModel(smoothing_fwhm=0.3)
            second_level_model = second_level_model.fit(second_level_input, design_matrix=design_matrix)

            z_map = second_level_model.compute_contrast(second_level_contrast=effect, output_type="z_score")
            z_map.to_filename(os.path.join(level2_dir, f"{metric}_{effect}_zmap.nii.gz"))

            # Mask the z-map
            masked_z_map = image.math_img("img1 * img2", img1=z_map, img2=mask_img)
            masked_z_map.to_filename(os.path.join(level2_dir, f"{metric}_{effect}_masked_zmap.nii.gz"))

            # Compute p-value map
            z_data = masked_z_map.get_fdata()
            p_data = 2 * (1 - norm.cdf(np.abs(z_data)))
            p_map = new_img_like(masked_z_map, p_data)
            p_map.to_filename(os.path.join(level2_dir, f"{metric}_{effect}_pmap.nii.gz"))

            # FDR-corrected Z-map
            thresholded_map, _ = threshold_stats_img(
                stat_img=masked_z_map, alpha=0.05, height_control='fdr', cluster_threshold=10, two_sided=True
            )
            thresholded_map.to_filename(os.path.join(level2_dir, f"{metric}_{effect}_masked_z_fdr_corrected.nii.gz"))

            # FDR-corrected P-map
            thresholded_data = thresholded_map.get_fdata()
            p_data_corrected = 2 * (1 - norm.cdf(np.abs(thresholded_data)))  # Two-sided p-values
            p_map_corrected = new_img_like(thresholded_map, p_data_corrected)
            p_map_corrected_path = os.path.join(level2_dir, f"{metric}_{effect}_p_map_fdr_corrected.nii.gz")
            p_map_corrected.to_filename(p_map_corrected_path)

            print(f"Saved corrected p-value map to: {p_map_corrected_path}")

        print(f"All ICA components processed for Metric: {metric}, Effect: {effect}.")
