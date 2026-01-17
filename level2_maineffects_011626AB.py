#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 17:33:43 2026

@author: alex
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Second-level VBA (main effects) with explicit intercept and contrasts
"""

import os
import pandas as pd
import numpy as np
from nilearn.glm.second_level import SecondLevelModel
from nilearn import image
from nilearn.glm import threshold_stats_img
from nilearn.image import new_img_like
from scipy.stats import norm

# --- Paths ---
output_root_dir = "/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/myICA_separate100_clean_0p1/"
mask_img_path = "/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/atlas/reference_maps/chass_atlas_mask_0p1.nii.gz"
metadata_file = "/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/metadata/cardiac_design_updated3.csv"

mask_img = image.load_img(mask_img_path)
metadata = pd.read_csv(metadata_file)

metadata = metadata.rename(columns={
    "E2_genotype": "E2_Genotype",
    "E3_genotype": "E3_Genotype",
    "E4_genotype": "E4_Genotype"
})

# --- Genotype coding (safe even if columns already exist) ---
if "E2_Genotype" not in metadata.columns:
    metadata["E2_Genotype"] = metadata["Genotype"].apply(
        lambda x: 1 if x in ["E22", "E22HN"] else 0
    )
if "E3_Genotype" not in metadata.columns:
    metadata["E3_Genotype"] = metadata["Genotype"].apply(
        lambda x: 1 if x in ["E33", "E33HN"] else 0
    )
if "E4_Genotype" not in metadata.columns:
    metadata["E4_Genotype"] = metadata["Genotype"].apply(
        lambda x: 1 if x in ["E44", "E44HN"] else 0
    )

metadata["HN"] = metadata["Genotype"].apply(lambda x: 1 if "HN" in str(x) else 0)

# --- Age category ---
median_age = metadata["Age"].median()
metadata["Age_cat"] = (metadata["Age"] >= median_age).astype(int)

# --- Metrics ---
metrics = [
    "Diastolic_LV_Volume", "Systolic_LV_Volume", "Heart_Rate",
    "Stroke_Volume", "Ejection_Fraction", "Cardiac_Output",
    "Diastolic_RV", "Systolic_RV", "Diastolic_LA", "Systolic_LA",
    "Diastolic_RA", "Systolic_RA", "Diastolic_Myo"
]

# --- Main effects ---
effects_to_analyze = [
    "Age_cat", "Sex_Male", "Diet_HFD", "Exercise_Yes", "HN",
    "E2_Genotype", "E3_Genotype", "E4_Genotype"
]

# --- Loop ---
for effect in effects_to_analyze:
    print(f"\nProcessing effect: {effect}")

    for metric in metrics:
        print(f"  Metric: {metric}")

        for j in range(1, 21):
            level1_dir = os.path.join(
                output_root_dir, "level1", f"level1_ica{j:02}"
            )
            level2_dir = os.path.join(
                output_root_dir, "level2", effect, metric, f"level2_ica{j:02}"
            )
            os.makedirs(level2_dir, exist_ok=True)

            level1_files = sorted(
                f for f in os.listdir(level1_dir) if f.endswith(".nii.gz")
            )

            matched_files = []
            matched_rows = []

            for _, row in metadata.iterrows():
                sid = row["Arunno"]
                hit = [os.path.join(level1_dir, f) for f in level1_files if sid in f]
                if hit:
                    matched_files.append(hit[0])
                    matched_rows.append(row)

            if not matched_rows:
                continue

            md = pd.DataFrame(matched_rows)

            # --- Design matrix ---
            design_matrix = md[[metric, effect]].apply(
                pd.to_numeric, errors="coerce"
            ).dropna()

            if design_matrix.empty:
                continue

            # --- Align images to design matrix rows ---
            valid_idx = design_matrix.index.tolist()
            second_level_input = [
                image.load_img(matched_files[i]) for i in valid_idx
            ]

            # --- Intercept ---
            design_matrix["Intercept"] = 1

            # --- Fit model ---
            model = SecondLevelModel(smoothing_fwhm=0.3)
            model = model.fit(second_level_input, design_matrix=design_matrix)

            # --- Explicit contrast ---
            contrast = {effect: 1, "Intercept": 0}

            z_map = model.compute_contrast(
                second_level_contrast=contrast,
                output_type="z_score"
            )
            z_map_path = os.path.join(level2_dir, f"{metric}_{effect}_zmap.nii.gz")
            z_map.to_filename(z_map_path)

            # --- Mask ---
            masked_z = image.math_img("img1 * img2", img1=z_map, img2=mask_img)
            masked_z_path = os.path.join(
                level2_dir, f"{metric}_{effect}_masked_zmap.nii.gz"
            )
            masked_z.to_filename(masked_z_path)

            # --- P map ---
            z_data = masked_z.get_fdata()
            p_data = 2 * (1 - norm.cdf(np.abs(z_data)))
            p_map = new_img_like(masked_z, p_data)
            p_map.to_filename(
                os.path.join(level2_dir, f"{metric}_{effect}_pmap.nii.gz")
            )

            # --- FDR ---
            z_fdr, _ = threshold_stats_img(
                masked_z, alpha=0.05, height_control="fdr",
                cluster_threshold=10, two_sided=True
            )
            z_fdr.to_filename(
                os.path.join(level2_dir, f"{metric}_{effect}_masked_z_fdr.nii.gz")
            )

            p_fdr = new_img_like(
                z_fdr, 2 * (1 - norm.cdf(np.abs(z_fdr.get_fdata())))
            )
            p_fdr.to_filename(
                os.path.join(level2_dir, f"{metric}_{effect}_p_fdr.nii.gz")
            )

        print(f"    ICA components done for {metric}")

print("\nAll analyses complete.")
