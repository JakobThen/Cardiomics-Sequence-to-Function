#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: Computes the mean of non-zero track values for all BigWig (.bw) files,
#          restricted to canonical chromosomes (chr1-22, X, Y). The means are
#          written into the AlphaGenome targets config by step 01.
# Usage:   bash workflows/alphagenome_ft/00_compute_nonzero_means.sh
# Input:   All .bw files below <bigwig_dir> (override: INPUT_DIR=...)
# Output:  <nonzero_means_csv> (override: OUTPUT_CSV=...)
# ==============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/load_site_env.sh"
WORKFLOW_DIR="${CS2F_REPO_ROOT}/workflows/alphagenome_ft"

INPUT_DIR="${INPUT_DIR:-$(cs2f_path bigwig_dir)}"
OUTPUT_CSV="${OUTPUT_CSV:-$(cs2f_path nonzero_means_csv)}"

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: BigWig directory not found: ${INPUT_DIR}" >&2
    exit 1
fi

cs2f_sbatch alphagenome_ft cpu "${WORKFLOW_DIR}/jobs/compute_nonzero_means.slurm" \
    --input_dir "$INPUT_DIR" \
    --output_csv "$OUTPUT_CSV"
