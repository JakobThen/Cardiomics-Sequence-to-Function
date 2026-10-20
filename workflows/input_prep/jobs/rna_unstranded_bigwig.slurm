#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: Worker script for unstranded RNA BAM to BigWig conversion (full read
#          coverage, e.g. for bulk RNA sequencing). Runs per cell type in a
#          SLURM array. Computes 1 bp coverage with deepTools, filters for
#          canonical chromosomes, and normalizes so counts sum to 100M.
#          Submitted by workflows/input_prep/02_rna_bigwig.sh with
#          RNA_MODE=unstranded. Requires deepTools (bamCoverage) in the env.
# Input:   $1 - BAM Input Directory
#          $2 - BigWig Output Directory
#          $3 - Path to the Config file containing valid cell types
#          $4 - Reference genome FASTA (unused, kept for a uniform interface)
#          $5 - Chromosome sizes file
# ==============================================================================

#SBATCH --job-name=RNA_BW_preprocess
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

source "${CS2F_REPO_ROOT:?Submit via workflows/input_prep/02_rna_bigwig.sh}/workflows/common/load_site_env.sh"

# ------------------------------------------------------------------------------
# Argument Parsing
# ------------------------------------------------------------------------------
if [ "$#" -ne 5 ]; then
    echo "Error: Expected 5 arguments (BAM_DIR, BW_OUTPUT_DIR, CONFIG_FILE, GENOME_FASTA, CHROM_SIZES). Got $#."
    exit 1
fi

BAM_DIR="$1"
BW_OUTPUT_DIR="$2"
CONFIG_FILE="$3"
CHROM_SIZES="$5"

# Read specific cell type parameter for this specific Array Task ID
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CONFIG_FILE")
CELL_TYPE=$(echo "$PARAMS" | awk '{print $1}')

echo "================================================="
echo "Running Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing unstranded RNA for: $CELL_TYPE"
echo "BAM Directory:         $BAM_DIR"
echo "Output Directory:      $BW_OUTPUT_DIR"
echo "================================================="

# File definitions
BAM_FILE="${BAM_DIR}/${CELL_TYPE}.bam"
OUT_PREFIX="${BW_OUTPUT_DIR}/RNA_${CELL_TYPE}"

RAW_BW="${OUT_PREFIX}_raw_coverage.bw"
NORM_BW="${OUT_PREFIX}_100M_norm.bw"
TMP_BG="${OUT_PREFIX}_temp.bg"
FILTERED_RAW_BW="${OUT_PREFIX}_filtered_raw.bw"
TMP_NORM_BG="${OUT_PREFIX}_temp_norm.bg"

mkdir -p "$BW_OUTPUT_DIR"

# ------------------------------------------------------------------------------
# Processing Pipeline
# ------------------------------------------------------------------------------
cs2f_activate "$CS2F_ENV_CHROMBPNET"

# make raw counts BigWig using deepTools
echo "  Generating raw RNA coverage for ${CELL_TYPE}..."
bamCoverage \
    --bam "$BAM_FILE" \
    --outFileName "$RAW_BW" \
    --binSize 1 \
    --normalizeUsing None \
    --numberOfProcessors "$SLURM_CPUS_PER_TASK"

echo "  Filtering non-canonical chromosomes for ${CELL_TYPE}..."

# convert to bedgraph for chormosme filtering keep ONLY chr1-22, chrX
bigWigToBedGraph "$RAW_BW" stdout | awk '$1 ~ /^chr([1-9]|1[0-9]|2[0-2]|X)$/' > "$TMP_BG"

# save filtered raw BigWig
bedGraphToBigWig "$TMP_BG" "$CHROM_SIZES" "$FILTERED_RAW_BW"

echo "  Normalizing ${CELL_TYPE} to 100 million total signal..."

# ectract total signal from filtred BigWig
TOTAL_SIGNAL=$(python -c "
import pyBigWig
try:
    bw = pyBigWig.open('$FILTERED_RAW_BW')
    print(int(bw.header()['sumData']))
    bw.close()
except Exception as e:
    pass
")

if [[ -z "$TOTAL_SIGNAL" || "$TOTAL_SIGNAL" == "0" ]]; then
    echo "Error: Total signal parsed as zero or empty. Exiting."
    exit 1
fi

SCALE_FACTOR=$(awk -v total="$TOTAL_SIGNAL" 'BEGIN {printf "%.10f\n", 100000000 / total}')

echo "    Total filtered signal: ${TOTAL_SIGNAL}"
echo "    Scaling factor: ${SCALE_FACTOR}"

awk -v OFS="\t" -v scale="$SCALE_FACTOR" '{$4 = $4 * scale; print $0}' "$TMP_BG" > "$TMP_NORM_BG"

# convert to norm bigwig
bedGraphToBigWig "$TMP_NORM_BG" "$CHROM_SIZES" "$NORM_BW"

# cleanup
rm -f "$TMP_BG" "$TMP_NORM_BG" "$RAW_BW" "$FILTERED_RAW_BW"

echo "RNA Preprocessing complete for ${CELL_TYPE}."
conda deactivate
