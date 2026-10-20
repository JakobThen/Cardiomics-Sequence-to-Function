#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: Takes bulk Cut&Tag .bam files and submits a SLURM array job to
#          compute base-resolution bigwigs for AlphaGenome/Borzoi.
#          Leverages chromBPNet derived pipeline and transfers it to Cut&Tag.
#          Reads are normalized to 100M counts per .bw file.
# Usage:   bash workflows/input_prep/03_cutntag_bigwig.sh
#          CUTNTAG_INPUT=/path/to/curated_bams.txt bash workflows/input_prep/03_cutntag_bigwig.sh
# Input:   CUTNTAG_INPUT can be either:
#          1. A directory path (will scan for all .bam files)
#          2. A text file containing manually curated full paths to .bam files
#          [default: <cutntag_bam_input> from config/paths.yaml]
# Output:  <bigwig_dir>/CutnTag_bigwig/<sample>_100M_norm.bw
#          Run list and submission log in <logs_dir>/input_prep/
# ==============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/load_site_env.sh"
WORKFLOW_DIR="${CS2F_REPO_ROOT}/workflows/input_prep"

# ------------------------------------------------------------------------------
# I/O & Parameters
# ------------------------------------------------------------------------------
INPUT_SOURCE="${CUTNTAG_INPUT:-$(cs2f_path cutntag_bam_input)}"
BW_OUTPUT_DIR="${BW_OUTPUT_DIR:-$(cs2f_path bigwig_dir)/CutnTag_bigwig}"
GENOME_FASTA="$(cs2f_path genome_fasta)"
CHROM_SIZES="$(cs2f_path chrom_sizes)"

# Run list (read by the array job) & submission log
RUN_DIR="$(cs2f_path logs_dir)/input_prep"
RUN_CONFIG="${RUN_DIR}/running_CutnTag_bams.txt"
LOG_FILE="${RUN_DIR}/submitted_CutnTag.txt"

# ------------------------------------------------------------------------------
# Execution
# ------------------------------------------------------------------------------
mkdir -p "$RUN_DIR"

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
    grep -v '^$' "$INPUT_SOURCE" > "$RUN_CONFIG" || true

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
cs2f_sbatch input_prep cpu --array=1-"$NUM_JOBS" "${WORKFLOW_DIR}/jobs/cutntag_bigwig.slurm" \
    "$BW_OUTPUT_DIR" "$RUN_CONFIG" "$GENOME_FASTA" "$CHROM_SIZES"
