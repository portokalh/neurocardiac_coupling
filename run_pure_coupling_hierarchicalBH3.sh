#!/bin/bash
#SBATCH --job-name=coupling_grid
#SBATCH --partition=normal
#SBATCH --array=0-259%40              # Max 40 running at once
#SBATCH --cpus-per-task=1             # Jobs are single-core
#SBATCH --mem=8G                      # Based on observed 2–6G usage
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/aashika/logs/coupling_%A_%a.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/aashika/logs/coupling_%A_%a.err

START_TIME=$(date +%s)

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Started at: $(date)"
echo "Node: $(hostname)"
echo "=============================================="

# -----------------------------------------------------
# Prevent BLAS oversubscription (critical)
# -----------------------------------------------------

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# -----------------------------------------------------
# Define environment python directly (HPC-safe)
# -----------------------------------------------------

ENV_PYTHON=/home/apps/ubuntu-22.04/anaconda3/envs/columns_analyses/bin/python

$ENV_PYTHON -c "import nilearn; print('Nilearn:', nilearn.__version__)"

# -----------------------------------------------------
# Define metrics
# -----------------------------------------------------

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

NUM_METRICS=${#METRICS[@]}
NUM_ICAS=20
TOTAL_JOBS=$((NUM_METRICS * NUM_ICAS))

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_JOBS" ]; then
    echo "Array index exceeds job count."
    exit 1
fi

# -----------------------------------------------------
# Grid indexing
# -----------------------------------------------------

METRIC_INDEX=$(( SLURM_ARRAY_TASK_ID / NUM_ICAS ))
ICA_INDEX=$(( SLURM_ARRAY_TASK_ID % NUM_ICAS ))

CURRENT_METRIC=${METRICS[$METRIC_INDEX]}
CURRENT_ICA=$((ICA_INDEX + 1))

echo "Running metric: $CURRENT_METRIC"
echo "Running ICA:    $CURRENT_ICA"

# -----------------------------------------------------
# Run Python
# -----------------------------------------------------

$ENV_PYTHON /mnt/newStor/paros/paros_WORK/aashika/mycode/cardiacfmri_021626/8_coupling_models_ICA_HierBH_withinICAFDR.py \
    --metric $CURRENT_METRIC \
    --ica $CURRENT_ICA \
    --smoothing_fwhm 0.3 \
    --alpha_fdr 0.05

# -----------------------------------------------------
# Timing
# -----------------------------------------------------

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "=============================================="
echo "Finished at: $(date)"
echo "Elapsed time: $ELAPSED seconds"
echo "=============================================="
