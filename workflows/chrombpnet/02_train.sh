#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: Submits ChromBPNet training (SLURM array job, one task per cell type,
#          5 folds each) with the pretrained ENCODE bias model. Pseudobulks are
#          filtered by cell count and peak count first.
# Usage:   bash workflows/chrombpnet/02_train.sh
#          CELL_TYPES="all" N_CELLS=500 bash workflows/chrombpnet/02_train.sh
#          OVERWRITE=1 (retrain finished folds), FOLDS="fold_0 fold_1",
#          N_GPUS=2, GPU_CONSTRAINT="gpu=A100" (TensorFlow 2.8 / CUDA 11; "" = site default)
# Input:   <atac_bam_dir>/<cell type>.bam, <chrombpnet_input_dir> (step 01),
#          <chrombpnet_bias_model>, configs/splits/<fold>.json, <genome_fasta>, <chrom_sizes>,
#          optional input_prep/configs/nCells.csv
# Output:  <chrombpnet_train_dir>/<cell type>_<fold>/ (models/chrombpnet_nobias.h5, evaluation/, ...),
#          <cell type>_<fold>_negatives.bed, <cell type>_<fold>_auxiliary/;
#          run list and submission log in <logs_dir>/chrombpnet/
# ==============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/load_site_env.sh"
WORKFLOW_DIR="${CS2F_REPO_ROOT}/workflows/chrombpnet"

# ------------------------------------------------------------------------------
# I/O & Parameters
# ------------------------------------------------------------------------------
BAM_DIR="${BAM_DIR:-$(cs2f_path atac_bam_dir)}"
INPUT_DIR="${INPUT_DIR:-$(cs2f_path chrombpnet_input_dir)}"
OUTPUT_DIR="${OUTPUT_DIR:-$(cs2f_path chrombpnet_train_dir)}"
BIAS_MODEL="${BIAS_MODEL:-$(cs2f_path chrombpnet_bias_model)}"
SPLITS_DIR="${SPLITS_DIR:-${WORKFLOW_DIR}/configs/splits}"
GENOME_FASTA="$(cs2f_path genome_fasta)"
CHROM_SIZES="$(cs2f_path chrom_sizes)"

# Cell types (BAM file names without .bam), or "all" for every pseudobulk with prepared peaks
read -r -a CELL_TYPES <<< "${CELL_TYPES:-Cardiomyocytes Fibroblasts Cardiac_progenitors}"
CELL_NUMBER_FILE="${CELL_NUMBER_FILE-${CS2F_REPO_ROOT}/workflows/input_prep/configs/nCells.csv}"
N_CELLS="${N_CELLS:-300}"     # minimum number of cells
N_PEAKS="${N_PEAKS:-10000}"   # minimum number of peaks
FOLDS="${FOLDS:-fold_0 fold_1 fold_2 fold_3 fold_4}"
OVERWRITE="${OVERWRITE:-0}"
N_GPUS="${N_GPUS:-2}"
GPU_CONSTRAINT="${GPU_CONSTRAINT-gpu=A100}"

RUN_DIR="$(cs2f_path logs_dir)/chrombpnet"
CONFIG_FILE="${RUN_DIR}/valid_pseudobulks.txt"
LOG_FILE="${RUN_DIR}/submitted_pseudobulks.txt"

# ------------------------------------------------------------------------------
# Checks
# ------------------------------------------------------------------------------
if [[ ! -f "$BIAS_MODEL" ]]; then
    echo "Error: bias model ${BIAS_MODEL} not found." >&2
    exit 1
fi
for fold in $FOLDS; do
    if [[ ! -f "${SPLITS_DIR}/${fold}.json" ]]; then
        echo "Error: split ${SPLITS_DIR}/${fold}.json not found." >&2
        exit 1
    fi
done

# ------------------------------------------------------------------------------
# Filter pseudobulks
# ------------------------------------------------------------------------------
mkdir -p "$RUN_DIR"
> "$CONFIG_FILE"
> "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

if [[ "${CELL_TYPES[0],,}" == "all" ]]; then
    CELL_TYPES=()
    for peaks in "$INPUT_DIR"/*_peaks_no_blacklist.narrowPeak; do
        [[ -f "$peaks" ]] && CELL_TYPES+=("$(basename "$peaks" _peaks_no_blacklist.narrowPeak)")
    done
fi

echo "Scanning for valid pseudobulks..."
printf "%-25s %-10s %-10s %-15s\n" "CELL_TYPE" "CELLS" "PEAKS" "STATUS"
echo "--------------------------------------------------------------------------------"

for CELL_TYPE in "${CELL_TYPES[@]}"; do
    BAM_FILE="${BAM_DIR}/${CELL_TYPE}.bam"
    PEAKS="${INPUT_DIR}/${CELL_TYPE}_peaks_no_blacklist.narrowPeak"

    if [[ ! -f "$BAM_FILE" || ! -f "$PEAKS" ]]; then
        printf "%-25s %-10s %-10s %-15s\n" "$CELL_TYPE" "N/A" "N/A" "SKIPPED (Missing Files)"
        continue
    fi

    # Cell count from the CSV (without the file, only the peak filter applies)
    CELL_COUNT="N/A"
    CELLS_OK=1
    if [[ -n "$CELL_NUMBER_FILE" && -f "$CELL_NUMBER_FILE" ]]; then
        CELL_COUNT=$(awk -F',' -v col="n_cells" -v ct="$CELL_TYPE" '
        NR==1 {
            for(i=1;i<=NF;i++) {
                gsub(/"/, "", $i); # Remove quotes from header
                if($i == col) col=i;
            }
        }
        NR>1 {
            gsub(/"/, "", $NF); # Remove quotes from cell type column
            if($NF == ct && col) {
                print $col;
            }
        }' "$CELL_NUMBER_FILE")
        CELL_COUNT=${CELL_COUNT:-0}
        (( CELL_COUNT >= N_CELLS )) || CELLS_OK=0
    fi
    PEAK_COUNT=$(wc -l < "$PEAKS")

    if (( CELLS_OK == 1 && PEAK_COUNT >= N_PEAKS )); then
        echo "$CELL_TYPE" >> "$CONFIG_FILE"
        printf "%-25s %-10s %-10s %-15s\n" "$CELL_TYPE" "$CELL_COUNT" "$PEAK_COUNT" "KEPT"
    else
        printf "%-25s %-10s %-10s %-15s\n" "$CELL_TYPE" "$CELL_COUNT" "$PEAK_COUNT" "SKIPPED (Filtered)"
    fi
done
echo "--------------------------------------------------------------------------------"

NUM_JOBS=$(wc -l < "$CONFIG_FILE")
if [[ "$NUM_JOBS" -eq 0 ]]; then
    echo "Error: No pseudobulks passed the filters. Exiting."
    exit 1
fi
echo "Found ${NUM_JOBS} valid pseudobulks. Submitting Slurm array job..."

SBATCH_OPTS=(--array=1-"$NUM_JOBS" --gpus="$N_GPUS")
[[ -n "$GPU_CONSTRAINT" ]] && SBATCH_OPTS+=(--constraint="$GPU_CONSTRAINT")

cs2f_sbatch chrombpnet gpu "${SBATCH_OPTS[@]}" "${WORKFLOW_DIR}/jobs/train.slurm" \
    "$CONFIG_FILE" "$BAM_DIR" "$INPUT_DIR" "$OUTPUT_DIR" "$BIAS_MODEL" "$SPLITS_DIR" \
    "$GENOME_FASTA" "$CHROM_SIZES" "$OVERWRITE" "$FOLDS"
