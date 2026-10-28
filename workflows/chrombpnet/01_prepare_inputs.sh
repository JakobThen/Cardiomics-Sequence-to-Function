#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: Prepares the ChromBPNet training inputs (CPU job): removes peaks that
#          overlap the blacklist expanded by 1057 bp (half the 2114 bp model input,
#          as in the ChromBPNet tutorial) and writes a chr-prefixed blacklist for
#          the background regions of step 02. The pseudobulk BAMs are used in place.
# Usage:   bash workflows/chrombpnet/01_prepare_inputs.sh
# Input:   <atac_peak_dir>/<cell type>.filtered.short.BAM_peaks.narrowPeak (MACS2),
#          <atac_bam_dir>/<cell type>.bam (only checked), <blacklist_bed>, <chrom_sizes>
# Output:  <chrombpnet_input_dir>/<cell type>_peaks_no_blacklist.narrowPeak,
#          blacklist.chr.bed, blacklist.slop.bed
# ==============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/load_site_env.sh"
WORKFLOW_DIR="${CS2F_REPO_ROOT}/workflows/chrombpnet"

PEAK_DIR="${PEAK_DIR:-$(cs2f_path atac_peak_dir)}"
BAM_DIR="${BAM_DIR:-$(cs2f_path atac_bam_dir)}"
OUTPUT_DIR="${OUTPUT_DIR:-$(cs2f_path chrombpnet_input_dir)}"
BLACKLIST="${BLACKLIST:-$(cs2f_path blacklist_bed)}"
CHROM_SIZES="$(cs2f_path chrom_sizes)"
SLOP_SIZE="${SLOP_SIZE:-1057}"

for required in "$BLACKLIST" "$CHROM_SIZES"; do
    if [[ ! -f "$required" ]]; then
        echo "Error: ${required} not found." >&2
        exit 1
    fi
done
for dir in "$PEAK_DIR" "$BAM_DIR"; do
    if [[ ! -d "$dir" ]]; then
        echo "Error: folder ${dir} not found." >&2
        exit 1
    fi
done

cs2f_sbatch chrombpnet cpu "${WORKFLOW_DIR}/jobs/prepare_inputs.slurm" \
    "$PEAK_DIR" "$BAM_DIR" "$OUTPUT_DIR" "$BLACKLIST" "$CHROM_SIZES" "$SLOP_SIZE"
