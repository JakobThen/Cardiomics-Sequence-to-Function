#!/bin/bash
# ==============================================================================
# Code by Jakob
# 
# Purpose: Takes pseudobulked scRNA .bam files and sets up a SLURM array job to 
#          compute base-resolution bigwigs for AlphaGenome/Borzoi.
#          Reads are split by strand and both strands sum to 100M counts per pseudobulk.   
#          Optionally filters pseudobulks based on a minimum cell count.
#          Assumes DEDUPLICATED .bam files.
# Input:   BAM_DIR and (optional) CELL_NUMBER_FILE metadata.
# Output:  Generates job configs and submits make_RNA_3prime_bw.slurm array jobs.
# ==============================================================================

# ------------------------------------------------------------------------------
# I/O & Parameters
# ------------------------------------------------------------------------------
BASE_DIR="../.."
BAM_DIR="${BASE_DIR}/input_data/raw_bam/multiome_RNA"
BW_OUTPUT_DIR="${BASE_DIR}/input_data/AlphaGenome_Borzoi_input/multiome_RNA_3prime_bigwig_deduplicated"

# Define pseudobulks to train using celltype. Use "all" to process the whole directory.
CELL_TYPES=("all") 

# Optional Filtering: Set CELL_NUMBER_FILE="" to disable filtering entirely.
CELL_NUMBER_FILE="configs/nCells.csv" 
N_CELLS=400   

# Config & Logging files
CONFIG_FILE="utils/preprocessing/valid_RNA_pseudobulks.txt"
LOG_FILE="utils/preprocessing/submitted_RNA_pseudobulks.txt"

# ------------------------------------------------------------------------------
# Execution
# ------------------------------------------------------------------------------

# Clear previous logs
> "$CONFIG_FILE"
> "$LOG_FILE"

# Redirect stdout + stderr to both console and log
exec > >(tee -a "$LOG_FILE") 2>&1

# Resolve "all" wildcard to actual cell types
if [[ "${CELL_TYPES[0],,}" == "all" ]]; then
    declare -A UNIQUE_CELL_TYPES
    for bam in "$BAM_DIR"/*.bam; do
        if [[ -f "$bam" ]]; then
            CT_NAME=$(basename "$bam" .bam)
            UNIQUE_CELL_TYPES["$CT_NAME"]=1
        fi
    done
    CELL_TYPES=("${!UNIQUE_CELL_TYPES[@]}")
fi

echo "Scanning for valid data combinations..."
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

echo "----------------------------------------------------------------------"

NUM_JOBS=$(wc -l < "$CONFIG_FILE")

if [ "$NUM_JOBS" -eq 0 ]; then
    echo "Error: No combinations passed the filtering thresholds. Exiting."
    exit 1
fi

echo "Found $NUM_JOBS valid RNA combinations. Submitting Slurm array job..."

# Launch the array job, passing our dynamic directories and config path as arguments
# Use make_RNA_bw.slurm to make unstranded RNA bw files
sbatch --array=1-$NUM_JOBS ../utils/preprocessing/make_RNA_3prime_bw.slurm "$BAM_DIR" "$BW_OUTPUT_DIR" "$CONFIG_FILE"