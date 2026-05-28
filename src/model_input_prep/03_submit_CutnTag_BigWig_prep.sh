#!/bin/bash
# ==============================================================================
# Code by Jakob
# 
# Purpose: Takes bulk Cut&Tag .bam files and sets up a SLURM array job to 
#          compute base-resolution bigwigs for AlphaGenome/Borzoi. 
#          Leverages chromBPNet derived pipeline and transfers it to Cut&Tag.
#          Reads are normalized to 100M counts per .bw file.
# Input:   INPUT_SOURCE can be either:
#          1. A directory path (will scan for all .bam files)
#          2. A text file containing manually curated full paths to .bam files
# Output:  Generates job configs and submits make_CutnTag_bw.slurm array jobs.
# ==============================================================================

# ------------------------------------------------------------------------------
# I/O & Parameters
# ------------------------------------------------------------------------------
BASE_DIR="../.."
BW_OUTPUT_DIR="${BASE_DIR}/input_data/AlphaGenome_Borzoi_input/CutnTag_bigwig"

# Set this to EITHER a directory OR a .txt file
INPUT_SOURCE="utils/preprocessing/valid_CutnTag_bams.txt" 
# Example directory usage: INPUT_SOURCE="${BASE_DIR}/input_data/raw_bam/CutnTag"

# Temporary config file that the array job will actually read
RUN_CONFIG="utils/preprocessing/running_CutnTag_bams.txt"
LOG_FILE="utils/preprocessing/submitted_CutnTag.txt"

# ------------------------------------------------------------------------------
# Execution
# ------------------------------------------------------------------------------

# Clear previous files
> "$RUN_CONFIG"
> "$LOG_FILE"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "======================================================================"
echo "Evaluating Input Source: $INPUT_SOURCE"
echo "======================================================================"

# Determine if the input is a directory or a file
if [[ -d "$INPUT_SOURCE" ]]; then
    echo "Mode: Directory detected. Scanning for .bam files..."
    find "$INPUT_SOURCE" -type f -name "*.bam" > "$RUN_CONFIG"
    
elif [[ -f "$INPUT_SOURCE" ]]; then
    echo "Mode: Config file detected. Copying manually curated paths..."
    # Filter out empty lines just in case
    grep -v '^$' "$INPUT_SOURCE" > "$RUN_CONFIG"
    
else
    echo "Error: INPUT_SOURCE is neither a valid directory nor a file."
    exit 1
fi

echo "----------------------------------------------------------------------"
# Show a preview of the files queued for processing
head -n 5 "$RUN_CONFIG"
echo "... (showing up to 5 paths)"
echo "----------------------------------------------------------------------"

NUM_JOBS=$(wc -l < "$RUN_CONFIG")

if [ "$NUM_JOBS" -eq 0 ]; then
    echo "Error: No BAM files found. Exiting."
    exit 1
fi

echo "Found $NUM_JOBS Cut&Tag samples. Submitting Slurm array job..."

# Launch as array job, passing the output dir and the active config file
sbatch --array=1-$NUM_JOBS ../utils/preprocessing/make_CutnTag_bw.slurm "$BW_OUTPUT_DIR" "$RUN_CONFIG"