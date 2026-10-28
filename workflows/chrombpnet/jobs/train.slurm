#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: SLURM array job for step 02. Trains ChromBPNet for one cell type
#          (line SLURM_ARRAY_TASK_ID of the run list) on every fold: GC-matched
#          background regions (chrombpnet prep nonpeaks), then chrombpnet pipeline
#          with the pretrained bias model. Finished folds are skipped unless
#          OVERWRITE=1; unfinished folds are cleaned up and restarted.
# Input:   $1 - Run list (one cell type per line)
#          $2 - BAM folder           $3 - Prepared peaks and blacklist (step 01)
#          $4 - Output folder        $5 - Bias model (.h5)
#          $6 - Folder with <fold>.json splits
#          $7 - Genome FASTA         $8 - Chromosome sizes
#          $9 - OVERWRITE (0/1)      $10 - Folds (space-separated)
# ==============================================================================

#SBATCH --job-name=ChromBPNet_train
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=64G
#SBATCH --time=2-00:00:00

set -eo pipefail   # no -u: conda activate scripts use unset variables
source "${CS2F_REPO_ROOT:?Submit via workflows/chrombpnet/02_train.sh}/workflows/common/load_site_env.sh"

CONFIG_FILE="$1"
BAM_DIR="$2"
INPUT_DIR="$3"
OUT_DIR="$4"
BIAS_MODEL="$5"
SPLITS_DIR="$6"
REFERENCE_GENOME="$7"
CHROM_SIZES="$8"
OVERWRITE="$9"
FOLDS="${10}"

CELL_TYPE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CONFIG_FILE" | awk '{print $1}')
BAM_FILE="${BAM_DIR}/${CELL_TYPE}.bam"
PEAKS="${INPUT_DIR}/${CELL_TYPE}_peaks_no_blacklist.narrowPeak"
BLACKLIST="${INPUT_DIR}/blacklist.chr.bed"   # expanded by chrombpnet prep nonpeaks itself

echo "======================================================"
echo "Starting ChromBPNet training for: ${CELL_TYPE}"
echo "======================================================"
echo "Date:           $(date)"
echo "Array Task ID:  ${SLURM_ARRAY_TASK_ID}"
echo "Job ID:         ${SLURM_JOB_ID}"
echo "Execution Node: ${SLURMD_NODENAME}"
echo "======================================================"

cs2f_activate "$CS2F_ENV_CHROMBPNET"
python -c "import tensorflow as tf; print('TensorFlow', tf.__version__, '| GPUs:', tf.config.list_physical_devices('GPU'))"
mkdir -p "$OUT_DIR"

for fold in $FOLDS; do
    FOLD_DIR="${OUT_DIR}/${CELL_TYPE}_${fold}"
    NEGATIVES_BED="${FOLD_DIR}_negatives.bed"
    AUX_DIR="${FOLD_DIR}_auxiliary"
    MODEL_H5_FILE="${FOLD_DIR}/models/chrombpnet_nobias.h5"

    if [[ "$OVERWRITE" != "1" && -f "$MODEL_H5_FILE" ]]; then
        echo -e "\nFold ${fold} for ${CELL_TYPE} is already complete (found chrombpnet_nobias.h5). Skipping..."
        continue
    fi

    echo -e "\nStarting with ${fold} for ${CELL_TYPE}.."

    # Clean up incomplete files/folders from previous crashed runs to avoid corruption
    if [[ -d "$FOLD_DIR" || -d "$AUX_DIR" || -f "$NEGATIVES_BED" ]]; then
        echo "   Cleaning up existing/incomplete files for this fold before starting..."
        rm -rf "$FOLD_DIR" "$AUX_DIR"
        rm -f "$NEGATIVES_BED"
    fi
    mkdir -p "$FOLD_DIR"

    # GC-matched background regions (writes <FOLD_DIR>_negatives.bed and <FOLD_DIR>_auxiliary/)
    chrombpnet prep nonpeaks -g "$REFERENCE_GENOME" \
        -p "$PEAKS" \
        -c "$CHROM_SIZES" \
        -fl "${SPLITS_DIR}/${fold}.json" \
        -br "$BLACKLIST" \
        -o "$FOLD_DIR"

    # Train, evaluate and interpret the bias-factorized model (the BAM is only read)
    chrombpnet pipeline \
        -ibam "$BAM_FILE" \
        -d "ATAC" \
        -g "$REFERENCE_GENOME" \
        -c "$CHROM_SIZES" \
        -p "$PEAKS" \
        -n "$NEGATIVES_BED" \
        -fl "${SPLITS_DIR}/${fold}.json" \
        -b "$BIAS_MODEL" \
        -o "$FOLD_DIR"
done

conda deactivate
echo -e "\nJob finished for: ${CELL_TYPE} ($(date))"
