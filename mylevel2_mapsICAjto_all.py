#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:55:04 2024

@author: alex
"""

import os
import pandas as pd
from nilearn.glm.second_level import SecondLevelModel
from nilearn import image
from nilearn.reporting import make_glm_report
import numpy as np
import os
import numpy as np
from nilearn import image, masking
from nilearn.input_data import NiftiMasker
from nilearn import image
from nibabel import load
from nilearn.reporting import make_glm_report
from nilearn.glm import threshold_stats_img
from scipy.stats import norm



# --- Step 1: Define paths ---
output_root_dir = "/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/myICA_separate100/"

#level1_dir = "/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/myICA_separate100/level1_ica01/"  # Directory with first-level results
#output_dir = "/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/myICA_separate100/level2_ica01/"  # Directory to save second-level results
mask_img = load('/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/atlas/reference_maps/chass_atlas_mask.nii') 




# Load subject-level metadata
metadata_file = "/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/metadata/coon_for_subjects_346.csv"
metadata = pd.read_csv(metadata_file)

# --- Step 2: Match imaging files with metadata IDs ---
# Get list of brain imaging files in level1
for j in range(20, 21):
    
    level1_dir = os.path.join(output_root_dir, f"level1_ica{j:02}")
    level1_files = [f for f in os.listdir(level1_dir) if f.endswith(".nii.gz")]
    
    output_dir = os.path.join(output_root_dir, f"level2_ica{j:02}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    
    
    # Match files with IDs in the metadata
    matched_files = []
    matched_metadata = []
    for _, row in metadata.iterrows():
        subject_id = row["ID"]  # Subject ID from metadata
        matched_file = [os.path.join(level1_dir, f) for f in level1_files if subject_id in f]
        if matched_file:
            matched_files.append(matched_file[0])
            matched_metadata.append(row)
    
    # Convert matched metadata to DataFrame
    matched_metadata_df = pd.DataFrame(matched_metadata)
    
    # --- Step 3: Set up design matrix ---
    # Define the covariates and Diet variable
    design_matrix = matched_metadata_df[["Age", "male", "Excercise", "E4", "CTRL"]]
    design_matrix.columns = ["Age", "Male", "Exercise", "E4", "Diet"]  # Clean column names
    design_matrix["Intercept"] = 1.0
    
    # --- Step 4: Fit the second-level GLM ---
    # Load brain imaging files into a list
    second_level_input = [image.load_img(f) for f in matched_files]
    
    # Initialize the SecondLevelModel
    second_level_model = SecondLevelModel(smoothing_fwhm=0.1)
    second_level_model = second_level_model.fit(second_level_input, design_matrix=design_matrix)
    
    # --- Step 5: Compute the effect of Diet ---
    z_map = second_level_model.compute_contrast(second_level_contrast="Diet", output_type="z_score")
    
    # Save the resulting statistical map
    z_map_path = os.path.join(output_dir, "diet_effect_zmap.nii.gz")
    z_map.to_filename(z_map_path)
    print(f"Saved Diet effect Z-map to: {z_map_path}")
    
    # --- Step 2: Mask the Z-Map ---
    masked_z_map = image.math_img("img1 * img2", img1=z_map, img2=mask_img)
    masked_z_map.to_filename(os.path.join(output_dir, "masked_z_map.nii.gz"))
    print("Saved masked Z-map.")
    
    # --- Step 3: Convert Masked Z-Map to P-Value Map ---
    z_data = masked_z_map.get_fdata()
    p_data = 2 * (1 - norm.cdf(np.abs(z_data)))  # Two-tailed p-values
    
    # Create a new masked P-map
    masked_p_map = image.new_img_like(masked_z_map, p_data)
    masked_p_map.to_filename(os.path.join(output_dir, "masked_p_map.nii.gz"))
    print("Saved masked P-value map.")
    
    # --- Step 4: Apply Cluster-Level FDR Correction ---
    thresholded_map, threshold = threshold_stats_img(
        stat_img=masked_z_map,   # Input the masked Z-map
        alpha=0.05,              # FDR threshold
        height_control='fdr',    # Apply FDR correction
        cluster_threshold=10,    # Minimum cluster size (voxels)
        two_sided=True           # Two-tailed test
    )
    
    # Save the FDR-corrected Z-map
    thresholded_map.to_filename(os.path.join(output_dir, "masked_z_fdr_corrected.nii.gz"))
    print("Saved FDR-corrected Z-map.")

# --- Step 5: Generate Corrected P-Map (FDR-Corrected) ---
# Restrict p-values to masked area
thresholded_data = thresholded_map.get_fdata()
corrected_p_data = 2 * (1 - norm.cdf(np.abs(thresholded_data)))

# Mask and save the corrected P-map
corrected_p_map = image.new_img_like(masked_z_map, corrected_p_data)
corrected_p_map.to_filename(os.path.join(output_dir, "masked_fdr_corrected_p_map.nii.gz"))
print("Saved FDR-corrected P-value map.")

# --- Step 6: Generate an HTML Report ---
report = make_glm_report(second_level_model, contrasts="Diet")
report_path = os.path.join(output_dir, "diet_effect_report.html")
report.save_as_html(report_path)
print(f"Saved GLM report to: {report_path}")


'''
# Convert Z-map to P-value map
z_data = z_map.get_fdata()  # Extract the Z values as a NumPy array
p_data = 2 * (1 - norm.cdf(np.abs(z_data)))  # Two-tailed p-values

# Save the p-value map
p_map = image.new_img_like(z_map, p_data)
p_map.to_filename(os.path.join(output_dir, "diet_effect_pmap.nii.gz"))
print("Saved P-value map.")

# --- Step 5: FDR Correction at Cluster Level ---
# Apply cluster-level thresholding with FDR correction
thresholded_map, threshold = threshold_stats_img(
    stat_img=z_map,         # Input Z-map
    alpha=0.05,             # FDR threshold at 5%
    height_control='fdr',   # Apply FDR correction
    cluster_threshold=10,   # Minimum cluster size (in voxels)
    two_sided=True          # Two-tailed test
)

# Save the thresholded map
thresholded_map.to_filename(os.path.join(output_dir, "diet_effect_cluster_z_fdr.nii.gz"))
print("Saved FDR-corrected cluster-level map.")

z_data = z_map.get_fdata()
p_data = 2 * (1 - norm.cdf(np.abs(z_data)))  # Two-tailed p-values

# Apply FDR correction
p_data_flat = p_data[mask_img.get_fdata() > 0].flatten()  # Restrict to mask
fdr_corrected_p = np.zeros_like(p_data)
_, corrected_p_flat = masking.threshold_img(z_map, alpha=0.05, cluster_threshold=20)
corrected_fdr_map = corrected_p_flat  # Input the corrected_flat map
# Save as a new P-map
p_map = image.new_img_like(z_map, p_data)
p_map.to_filename("diet_effect_cluster_fdr_pmap.nii.gz")
print("P-map saved as diet_effect_cluster_fdr_pmap.nii.gz")


# --- Step 6: Generate an HTML report ---
report = make_glm_report(second_level_model, contrasts="Diet")
report_path = os.path.join(output_dir, "diet_effect_report.html")
report.save_as_html(report_path)
print(f"Saved GLM report to: {report_path}")



# --- Step 6: Generate an HTML report ---
report = make_glm_report(second_level_model, contrasts="Diet")
report_path = os.path.join(output_dir, "diet_effect_report.html")
report.save_as_html(report_path)
print(f"Saved GLM report to: {report_path}")
'''