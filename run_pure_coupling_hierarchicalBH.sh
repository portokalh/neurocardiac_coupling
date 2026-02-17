#!/bin/bash
#SBATCH --job-name=coupling_allICA
#SBATCH --output=/mnt/newStor/paros/paros_WORK/aashika/logs/coupling_%j.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/aashika/logs/coupling_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --partition=normal
#SBATCH --array=0-12


echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "==========================================="

START_TIME=$(date +%s)

# Load environment if needed
# module purge
# source ~/.bashrc
# conda activate nilearn_env

#METRICS=(Heart_Rate Ejection_Fraction Cardiac_Output Stroke_Volume Diastolic_LV_Volume Systolic_LV_Volume)

METRICS=(
Diastolic_LV_Volume
Systolic_LV_Volume
Heart_Rate
Stroke_Volume
Ejection_Fraction
Cardiac_Output
Diastolic_RV
Systolic_RV
Diastolic_LA
Systolic_LA
Diastolic_RA
Systolic_RA
Diastolic_Myo
)

N_METRICS=${#METRICS[@]}

if [ $SLURM_ARRAY_TASK_ID -ge $N_METRICS ]; then
    echo "Error: SLURM_ARRAY_TASK_ID exceeds METRICS length"
    exit 1
fi

METRIC=${METRICS[$SLURM_ARRAY_TASK_ID]}

SCRIPT=/mnt/newStor/paros/paros_WORK/aashika/mycode/cardiacfmri_021626/8_coupling_models_ICA_HierBH.py

echo "Running metric: $METRIC"
echo "-------------------------------------------"

python $SCRIPT \
    --metric $METRIC \
    --ica all \
    --n_ica_total 20 \
    --smoothing_fwhm 0.3 \
    --alpha_fdr 0.05

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "=============================================="
echo "End time: $(date)"
echo "Elapsed time: $ELAPSED seconds"
echo "Elapsed time: $(($ELAPSED / 60)) minutes"
echo "Elapsed time: $(($ELAPSED / 3600)) hours"
echo "=============================================="