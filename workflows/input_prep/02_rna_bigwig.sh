#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: Takes pseudobulked scRNA .bam files and submits a SLURM array job to
#          compute base-resolution bigwigs for AlphaGenome/Borzoi.
#          Optionally filters pseudobulks based on a minimum cell count.
#          Two modes (RNA_MODE):
#            3prime_stranded (default): 3' ends split by strand; both strands
#                                       together sum to 100M counts per pseudobulk.
#            unstranded:                full read coverage, unstranded, normalized
#                                       to 100M (e.g. for bulk RNA sequencing).
# Usage:   bash workflows/input_prep/02_rna_bigwig.sh
#          RNA_MODE=unstranded bash workflows/input_prep/02_rna_bigwig.sh
# Input:   <rna_bam_dir>/<cell type>.bam, optional configs/nCells.csv
# Output:  3prime_stranded: <bigwig_dir>/multiome_RNA_3prime_bigwig_deduplicated/
#                           RNA_<cell type>_{plus,minus}_100M.bw
#          unstranded:      <bigwig_dir>/multiome_RNA_unstranded_bigwig/
#                           RNA_<cell type>_100M_norm.bw
#          Run list and submission log in <logs_dir>/input_prep/
# ==============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/load_site_env.sh"
WORKFLOW_DIR="${CS2F_REPO_ROOT}/workflows/input_prep"

# ------------------------------------------------------------------------------
# I/O & Parameters
# ------------------------------------------------------------------------------
RNA_MODE="${RNA_MODE:-3prime_stranded}"
case "$RNA_MODE" in
    3prime_stranded)
        JOB_SCRIPT="${WORKFLOW_DIR}/jobs/rna_3prime_bigwig.slurm"
        DEFAULT_OUTPUT="multiome_RNA_3prime_bigwig_deduplicated" ;;
    unstranded)
        JOB_SCRIPT="${WORKFLOW_DIR}/jobs/rna_unstranded_bigwig.slurm"
        DEFAULT_OUTPUT="multiome_RNA_unstranded_bigwig" ;;
    *)
        echo "Error: RNA_MODE must be '3prime_stranded' or 'unstranded', got '${RNA_MODE}'." >&2
        exit 1 ;;
esac

BAM_DIR="${BAM_DIR:-$(cs2f_path rna_bam_dir)}"
BW_OUTPUT_DIR="${BW_OUTPUT_DIR:-$(cs2f_path bigwig_dir)/${DEFAULT_OUTPUT}}"
GENOME_FASTA="$(cs2f_path genome_fasta)"
CHROM_SIZES="$(cs2f_path chrom_sizes)"

# Define pseudobulks to train using celltype. Use "all" to process the whole directory.
CELL_TYPES=("all")

# Optional Filtering: Set CELL_NUMBER_FILE="" to disable filtering entirely.
CELL_NUMBER_FILE="${CELL_NUMBER_FILE-${WORKFLOW_DIR}/configs/nCells.csv}"
N_CELLS="${N_CELLS:-400}"

# Run list (read by the array job) & submission log
RUN_DIR="$(cs2f_path logs_dir)/input_prep"
CONFIG_FILE="${RUN_DIR}/valid_RNA_pseudobulks.txt"
LOG_FILE="${RUN_DIR}/submitted_RNA_pseudobulks.txt"

# ------------------------------------------------------------------------------
# Execution
# ------------------------------------------------------------------------------
mkdir -p "$RUN_DIR"

# Clear previous logs
> "$CONFIG_FILE"
> "$LOG_FILE"

# Redirect stdout + stderr to both console and log
exec > >(tee -a "$LOG_FILE") 2>&1

echo "RNA mode: ${RNA_MODE}"

# Resolve "all" wildcard to actual cell types
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

    # 1. Check if BAM exists
    if [[ -f "$BAM_FILE" ]]; then

        # 2. Check if we should apply metadata filtering
        if [[ -n "$CELL_NUMBER_FILE" && -f "$CELL_NUMBER_FILE" ]]; then
            # Extract cell count
            CELL_COUNT=$(awk -F',' -v col="n_cells" -v ct="$CELL_TYPE" '
            NR==1 {
                for(i=1;i<=NF;i++) {
                    gsub(/"/, "", $i);
                    if($i == col) col=i;
                }
            }
            NR>1 {
                gsub(/"/, "", $NF);
                if($NF == ct && col) {
                    print $col;
                }
            }' "$CELL_NUMBER_FILE")

            CELL_COUNT=${CELL_COUNT:-0}

            # Apply Filter
            if (( CELL_COUNT >= N_CELLS )); then
                echo "$CELL_TYPE" >> "$CONFIG_FILE"
                printf "%-35s %-10s %-15s\n" "$CELL_TYPE" "$CELL_COUNT" "KEPT"
            else
                printf "%-35s %-10s %-15s\n" "$CELL_TYPE" "$CELL_COUNT" "SKIPPED (Filtered)"
            fi
        else
            # 3. Bypass Filtering
            echo "$CELL_TYPE" >> "$CONFIG_FILE"
            printf "%-35s %-10s %-15s\n" "$CELL_TYPE" "N/A" "KEPT (No Filter)"
        fi

    else
        printf "%-35s %-10s %-15s\n" "$CELL_TYPE" "N/A" "SKIPPED (No BAM)"
    fi
done

echo "--------------------------------------------------------------------------------"

NUM_JOBS=$(wc -l < "$CONFIG_FILE")

if [ "$NUM_JOBS" -eq 0 ]; then
    echo "Error: No combinations passed the filtering thresholds. Exiting."
    exit 1
fi

echo "Found $NUM_JOBS valid RNA combinations. Submitting Slurm array job..."

# Launch the array job, passing our dynamic directories and config path as arguments
cs2f_sbatch input_prep cpu --array=1-"$NUM_JOBS" "$JOB_SCRIPT" \
    "$BAM_DIR" "$BW_OUTPUT_DIR" "$CONFIG_FILE" "$GENOME_FASTA" "$CHROM_SIZES"
