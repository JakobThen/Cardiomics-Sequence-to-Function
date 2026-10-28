#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: SLURM job for step 01. Filters the MACS2 peaks of every pseudobulk
#          against the expanded blacklist.
# Input:   $1 - Peak folder (MACS2 *.filtered.short.BAM_peaks.narrowPeak)
#          $2 - BAM folder (<cell type>.bam; a peak file is only used if its BAM exists)
#          $3 - Output folder
#          $4 - Blacklist BED
#          $5 - Chromosome sizes
#          $6 - Blacklist expansion in bp
# ==============================================================================

#SBATCH --job-name=ChromBPNet_prepare_inputs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -eo pipefail   # no -u: conda activate scripts use unset variables
source "${CS2F_REPO_ROOT:?Submit via workflows/chrombpnet/01_prepare_inputs.sh}/workflows/common/load_site_env.sh"

PEAK_DIR="$1"
BAM_DIR="$2"
OUTPUT_DIR="$3"
BLACKLIST="$4"
CHROM_SIZES="$5"
SLOP_SIZE="$6"

cs2f_activate "$CS2F_ENV_CHROMBPNET"
mkdir -p "$OUTPUT_DIR"

# Prepare the blacklist: add the chr prefix if missing, then expand it
echo "Preparing blacklist..."
BL_CHR="${OUTPUT_DIR}/blacklist.chr.bed"
BL_SLOP="${OUTPUT_DIR}/blacklist.slop.bed"
if ! head -n 1 "$BLACKLIST" | grep -q "^chr"; then
    sed 's/^/chr/' "$BLACKLIST" > "$BL_CHR"
else
    cp "$BLACKLIST" "$BL_CHR"
fi
bedtools slop -i "$BL_CHR" -g "$CHROM_SIZES" -b "$SLOP_SIZE" > "$BL_SLOP"

echo "Processing peak files in ${PEAK_DIR}..."
n_done=0
for PEAK_FILE in "$PEAK_DIR"/*.filtered.short.BAM_peaks.narrowPeak; do
    [[ -e "$PEAK_FILE" ]] || continue
    CELL="$(basename "$PEAK_FILE" .filtered.short.BAM_peaks.narrowPeak)"

    if [[ ! -f "${BAM_DIR}/${CELL}.bam" ]]; then
        echo "  Warning: missing BAM for ${CELL} (expected ${BAM_DIR}/${CELL}.bam), skipped"
        continue
    fi

    echo "  Processing - ${CELL}"
    bedtools intersect -v -a "$PEAK_FILE" -b "$BL_SLOP" > "${OUTPUT_DIR}/${CELL}_peaks_no_blacklist.narrowPeak"
    n_done=$((n_done + 1))
done

conda deactivate
echo "Preprocessing complete: ${n_done} pseudobulks written to ${OUTPUT_DIR}"
