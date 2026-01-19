Neurocardiac Coupling Analysis Pipeline
This repository contains Python scripts for analyzing neuro–cardiac coupling using resting-state fMRI–derived functional connectivity and cardiac imaging metrics in mouse models. The workflow integrates first-level ICA-based processing with second-level voxelwise and network-level GLMs to identify brain circuits whose functional connectivity varies systematically with cardiac structure and function.
The code is designed for research use and reflects the analysis pipeline used in associated manuscripts.
Overview of the Analysis Workflow
The pipeline follows a standard two-level neuroimaging framework:
First-level (subject-level) processing
Resting-state fMRI time series are denoised using ICA-based approaches.
Residual time series (e.g., FSL ERRTS or equivalent) are used to derive subject-level functional connectivity or ICA spatial maps.
Second-level (group-level) modeling
Subject-level maps are entered into voxelwise or network-level GLMs.
Cardiac metrics (e.g., diastolic volume, ejection fraction), along with covariates (age, sex, diet, exercise, genotype), are used as predictors.


neurocardiac_coupling/
│
├── group_ica_121224.py
│   Runs group ICA and generates group-level components.
│
├── mycanica.py
│   Custom wrapper utilities for CanICA-based ICA workflows.
│
├── mylevel1_mapsICA*.py
│   Generates subject-level ICA spatial maps from denoised residuals.
│
├── level2_maineffects.py
│   Performs second-level voxelwise GLMs for main effects.
│
├── level2_maineffects_011626AB.py
│   Updated/annotated version of the second-level analysis script.
│
├── fMRI2cardiacSBC_allmetrics_diet_exercise.py
│   Links fMRI connectivity metrics to multiple cardiac measures,
│   including diet and exercise effects.
│
└── 
