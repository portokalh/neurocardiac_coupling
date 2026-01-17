#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 11:40:07 2024

@author: alex
"""

from nilearn.decomposition import CanICA
import os
import subprocess
from nibabel import load
from nilearn.image import iter_img
from pathlib import Path

from nilearn.plotting import plot_prob_atlas
from nilearn.image import iter_img
from nilearn.plotting import plot_stat_map, show
from nilearn.plotting import plot_prob_atlas, plot_stat_map
from nilearn.image import iter_img
import matplotlib.pyplot as plt
from pathlib import Path


mask_img = load('/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/atlas/reference_maps/chass_atlas_mask.nii') 

# Define the directory containing the .nii.gz files
directory = '/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/errts_trim/'

# List all .nii.gz files in the directory
nii_files = [f for f in os.listdir(directory) if f.startswith('A') and f.endswith('.nii.gz')]
#func_filenames = rest_dataset.func  # list of 4D nifti files for each subject

ncomp=len(nii_files) #100

selected_files = nii_files[:ncomp]  # This selects the first 20 files
if len(selected_files) < ncomp:
    raise ValueError("Not enough files in the directory that meet the criteria.")

nii_files = selected_files

# Construct the base part of the command
myprefix = "$PAROS/paros_WORK/aashika/resample_mouse_fmri/concateneted_errts.nii.gz"
myICAout = "$PAROS/paros_WORK/aashika/resample_mouse_fmri/myICA100/"

# Create the full command by concatenating the files
input_files = [os.path.join(directory, f) for f in nii_files]
func_filenames = input_files

print("Files used for ICA:")
for idx, file in enumerate(func_filenames, start=1):
    print(f"{idx}: {file}")




# Run the command using subprocess
#subprocess.run(command, shell=True)

canica = CanICA(
    n_components=20,
    memory="nilearn_cache",
    memory_level=2,
    mask=mask_img,
    verbose=10,
    #mask_strategy="whole-brain-template", # if in MNI space
    random_state=42,
    standardize="zscore_sample",
    n_jobs=10,
)
canica.fit(func_filenames)

# Retrieve the independent components in brain space. Directly
# accessible through attribute `components_img_`.
canica_components_img = canica.components_img_



# Define output directory as a Path object
output_dir = Path("/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/myICA100/")
output_dir.mkdir(exist_ok=True, parents=True)  # Ensure the directory exists
print(f"Output will be saved to: {output_dir}")

canica_components_img.to_filename(output_dir / "canica_resting_state.nii.gz")


output_dir = Path("/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/myICA_separate100/")
output_dir.mkdir(exist_ok=True, parents=True)  # Ensure the directory exists

# Iterate over each component in the 4D image and save them separately
for i, img in enumerate(iter_img(canica_components_img), start=1):
    output_file = output_dir / f"canica_component_{i:02d}.nii.gz"
    img.to_filename(str(output_file))  # Save each 3D component
    print(f"Saved: {output_file}")
    


# Define output directory for saved plots
output_dir = Path("/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/myICA_plots/")
output_dir.mkdir(parents=True, exist_ok=True)

# Save all ICA components together as a single plot
all_components_plot_file = output_dir / "all_ica_components.png"
display = plot_prob_atlas(canica_components_img, title="All ICA components")
plt.savefig(all_components_plot_file, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {all_components_plot_file}")

# Save each ICA component as a separate plot
for i, cur_img in enumerate(iter_img(canica_components_img)):
    single_component_plot_file = output_dir / f"ica_component_{i:02d}.png"
    display = plot_stat_map(
        cur_img,
        display_mode="z",
        title=f"IC {i}",
        cut_coords=1,
        colorbar=False,
    )
    plt.savefig(single_component_plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {single_component_plot_file}")