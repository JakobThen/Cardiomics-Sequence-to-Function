#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: Takes pseudobulked ATAC .bam files and submits a SLURM array job to
#          compute base-resolution ATAC bigwigs (+4/-4 shift) for AlphaGenome/Borzoi.
#          Reads are normalized to 100M counts per .bw file.
#          Optionally filters pseudobulks based on a minimum cell count.
#          Assumes DEDUPLICATED .bam files.
# Usage:   bash workflows/input_prep/01_atac_bigwig.sh
#          (parameters below can be overridden, e.g. N_CELLS=200 bash ...)
# Input:   <atac_bam_dir>/<cell type>.bam, optional configs/nCells.csv
# Output:  <bigwig_dir>/multiome_ATAC_bigwig_deduplicated/ATAC_<cell type>_100M_norm.bw
#          Run list and submission log in <logs_dir>/input_prep/
# ==============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/load_site_env.sh"
WORKFLOW_DIR="${CS2F_REPO_ROOT}/workflows/input_prep"

# ------------------------------------------------------------------------------
# I/O & Parameters
# ------------------------------------------------------------------------------
BAM_DIR="${BAM_DIR:-$(cs2f_path atac_bam_dir)}"
BW_OUTPUT_DIR="${BW_OUTPUT_DIR:-$(cs2f_path bigwig_dir)/multiome_ATAC_bigwig_deduplicated}"
GENOME_FASTA="$(cs2f_path genome_fasta)"
CHROM_SIZES="$(cs2f_path chrom_sizes)"

# Define pseudobulks using cell type (contained in file name). Use "all" to process the whole directory.
CELL_TYPES=("all")

# Optional Filtering: Set CELL_NUMBER_FILE="" to disable filtering entirely.
CELL_NUMBER_FILE="${CELL_NUMBER_FILE-${WORKFLOW_DIR}/configs/nCells.csv}"
N_CELLS="${N_CELLS:-400}"

# Run list (read by the array job) & submission log
RUN_DIR="$(cs2f_path logs_dir)/input_prep"
CONFIG_FILE="${RUN_DIR}/valid_ATAC_pseudobulks.txt"
LOG_FILE="${RUN_DIR}/submitted_ATAC_pseudobulks.txt"

# ------------------------------------------------------------------------------
# Execution
# ------------------------------------------------------------------------------
mkdir -p "$RUN_DIR"

# Clear previous run logs
> "$CONFIG_FILE"
> "$LOG_FILE"

# Redirect all stdout + stderr to both the console and the log file simultaneously
exec > >(tee -a "$LOG_FILE") 2>&1

# Resolve "all" to actual cell types by scanning the directory
if [[ "${CELL_TYPES[0],,}" == "all" ]]; then
    declare -A UNIQUE_CELL_TYPES=()
    for bam in "$BAM_DIR"/*.bam; do
        if [[ -f "$bam" ]]; then
            CT_NAME=$(basename "$bam" .bam)
            UNIQUE_CELL_TYPES["$CT_NAME"]=1
        fi
    done
    CELL_TYPES=("${!UNIQUE_CELL_TYPES[@]}")
fi

echo "Scanning for valid data combinations in ${BAM_DIR}..."
printf "%-35s %-10s %-15s\n" "CELL_TYPE" "CELLS" "STATUS"
echo "--------------------------------------------------------------------------------"

for CELL_TYPE in "${CELL_TYPES[@]}"; do
    BAM_FILE="${BAM_DIR}/${CELL_TYPE}.bam"

    # 1. Verify the BAM file exists
    if [[ -f "$BAM_FILE" ]]; then

        # 2. Check if we should apply metadata filtering
        if [[ -n "$CELL_NUMBER_FILE" && -f "$CELL_NUMBER_FILE" ]]; then
            # Extract cell count from the CSV
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

            # Apply thresholds
            if (( CELL_COUNT >= N_CELLS )); then
                echo "$CELL_TYPE" >> "$CONFIG_FILE"
                printf "%-35s %-10s %-15s\n" "$CELL_TYPE" "$CELL_COUNT" "KEPT"
            else
                printf "%-35s %-10s %-15s\n" "$CELL_TYPE" "$CELL_COUNT" "SKIPPED (Filtered)"
            fi
        else
            # 3. Bypass Filtering: Keep everything
            echo "$CELL_TYPE" >> "$CONFIG_FILE"
            printf "%-35s %-10s %-15s\n" "$CELL_TYPE" "N/A" "KEPT (No Filter)"
        fi

    else
        printf "%-35s %-10s %-15s\n" "$CELL_TYPE" "N/A" "SKIPPED (Missing BAM)"
    fi
done

echo "--------------------------------------------------------------------------------"

NUM_JOBS=$(wc -l < "$CONFIG_FILE")

if [ "$NUM_JOBS" -eq 0 ]; then
    echo "Error: No combinations passed the checks. Exiting."
    exit 1
fi

echo "Found $NUM_JOBS valid combinations. Submitting Slurm array job..."

# Launch the array job, passing our dynamic directories and config path as arguments
cs2f_sbatch input_prep cpu --array=1-"$NUM_JOBS" "${WORKFLOW_DIR}/jobs/atac_bigwig.slurm" \
    "$BAM_DIR" "$BW_OUTPUT_DIR" "$CONFIG_FILE" "$GENOME_FASTA" "$CHROM_SIZES"
