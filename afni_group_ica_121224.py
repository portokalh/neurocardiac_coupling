#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:57:30 2024

@author: bass
"""
import os
import subprocess

# Define the directory containing the .nii.gz files
directory = '/mnt/newStor/paros/paros_WORK/aashika/resample_mouse_fmri/errts_trim/'

# List all .nii.gz files in the directory
nii_files = [f for f in os.listdir(directory) if f.startswith('A') and f.endswith('.nii.gz')]

# Construct the base part of the command
myprefix = "$PAROS/paros_WORK/aashika/resample_mouse_fmri/concateneted_errts.nii.gz"
myICAout = "$PAROS/paros_WORK/aashika/resample_mouse_fmri/myICA/"

# Create the full command by concatenating the files
input_files = [os.path.join(directory, f) for f in nii_files]

# Ensure the command includes all files in the directory
command = f"3dTcat -prefix {myprefix} " + " ".join(input_files)

# Print the command for verification (optional)
print(f"Running command: {command}")

# Run the command using subprocess
#subprocess.run(command, shell=True)

command2=f"3dICA -prefix {myICAout} -infile {myprefix} -num_ICs 20 "
subprocess.run(command2, shell=True)
# 3dICA -prefix group_ica_output -infile group_data+orig -num_ICs 20

# 1dplot group_ica_output_A+orig'[0]'  # For IC 0

'''
#regress noisy components
3dROIstats -mask group_ica_output+orig'[0]' group_ica_output_A+orig > IC0_time_series.1D
3dROIstats -mask group_ica_output+orig'[1]' group_ica_output_A+orig > IC1_time_series.1D

3dDeconvolve \
  -input group_data+orig \
  -polort 2 \
  -num_stimts 2 \
  -stim_file 1 IC0_time_series.1D \
  -stim_label 1 IC0 \
  -stim_file 2 IC1_time_series.1D \
  -stim_label 2 IC1 \
  -bucket cleaned_output \
  -rin motion_timeseries.1D  # optional: if you have motion parameters as regressors
  '''
  
  