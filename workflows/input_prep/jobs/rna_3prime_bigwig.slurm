#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: Worker script for RNA BAM to 3' BigWig conversion. Runs per cell type
#          in a SLURM array. Extracts 3' ends, splits by strand, filters for
#          canonical chromosomes, and normalizes so BOTH strands sum to 100M.
#          Submitted by workflows/input_prep/02_rna_bigwig.sh (default mode).
# Input:   $1 - BAM Input Directory
#          $2 - BigWig Output Directory
#          $3 - Path to the Config file containing valid cell types
#          $4 - Reference genome FASTA (unused, kept for a uniform interface)
#          $5 - Chromosome sizes file
# ==============================================================================

#SBATCH --job-name=RNA_3prime_BW
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
echo "Processing 3' RNA for: $CELL_TYPE"
echo "BAM Directory:         $BAM_DIR"
echo "Output Directory:      $BW_OUTPUT_DIR"
echo "================================================="

# File definitions
BAM_FILE="${BAM_DIR}/${CELL_TYPE}.bam"
OUT_PREFIX="${BW_OUTPUT_DIR}/RNA_${CELL_TYPE}"

# File names for stranded output
RAW_PLUS_BG="${OUT_PREFIX}_plus_raw.bg"
RAW_MINUS_BG="${OUT_PREFIX}_minus_raw.bg"
NORM_PLUS_BW="${OUT_PREFIX}_plus_100M.bw"
NORM_MINUS_BW="${OUT_PREFIX}_minus_100M.bw"

# Ensure output directory exists
mkdir -p "$BW_OUTPUT_DIR"

# ------------------------------------------------------------------------------
# Processing Pipeline
# ------------------------------------------------------------------------------

# Load Environment containing bedtools and UCSC tools (bedGraphToBigWig)
cs2f_activate "$CS2F_ENV_CHROMBPNET"

echo "  1/4 Extracting 3' ends and filtering for canonical chromosomes..."
# Plus strand
bedtools genomecov -ibam "$BAM_FILE" -bg -3 -strand + | \
awk '$1 ~ /^chr([1-9]|1[0-9]|2[0-2]|X)$/' > "$RAW_PLUS_BG"

# Minus strand
bedtools genomecov -ibam "$BAM_FILE" -bg -3 -strand - | \
awk '$1 ~ /^chr([1-9]|1[0-9]|2[0-2]|X)$/' > "$RAW_MINUS_BG"

echo "  2/4 Calculating combined total signal..."
TOTAL_PLUS=$(awk '{s+=$4} END {print s+0}' "$RAW_PLUS_BG")
TOTAL_MINUS=$(awk '{s+=$4} END {print s+0}' "$RAW_MINUS_BG")

# Use awk to sum them safely to avoid integer limits in Bash
TOTAL_SIGNAL=$(awk -v p="$TOTAL_PLUS" -v m="$TOTAL_MINUS" 'BEGIN {printf "%.0f", p + m}')

if [[ -z "$TOTAL_SIGNAL" || "$TOTAL_SIGNAL" == "0" ]]; then
    echo "Error: Total combined 3' signal is zero. Exiting."
    exit 1
fi

# 3. Compute the shared scaling factor across both strands
SCALE_FACTOR=$(awk -v total="$TOTAL_SIGNAL" 'BEGIN {printf "%.10f\n", 100000000 / total}')

echo "      Total Plus 3' Ends:  ${TOTAL_PLUS}"
echo "      Total Minus 3' Ends: ${TOTAL_MINUS}"
echo "      Combined Total:      ${TOTAL_SIGNAL}"
echo "      Scaling factor:      ${SCALE_FACTOR}"

# Define temp files for sorting
TMP_PLUS_BG="${OUT_PREFIX}_plus_temp.bg"
TMP_MINUS_BG="${OUT_PREFIX}_minus_temp.bg"

echo "  3/4 Normalizing and converting Plus strand..."
awk -v OFS="\t" -v scale="$SCALE_FACTOR" '{$4 = $4 * scale; print $0}' "$RAW_PLUS_BG" | \
LC_COLLATE="C" sort -k1,1 -k2,2n > "$TMP_PLUS_BG"

bedGraphToBigWig "$TMP_PLUS_BG" "$CHROM_SIZES" "$NORM_PLUS_BW"

echo "  4/4 Normalizing and converting Minus strand..."
awk -v OFS="\t" -v scale="$SCALE_FACTOR" '{$4 = $4 * scale; print $0}' "$RAW_MINUS_BG" | \
LC_COLLATE="C" sort -k1,1 -k2,2n > "$TMP_MINUS_BG"

bedGraphToBigWig "$TMP_MINUS_BG" "$CHROM_SIZES" "$NORM_MINUS_BW"

# Cleanup massive intermediate bedGraphs
rm -f "$RAW_PLUS_BG" "$RAW_MINUS_BG" "$TMP_PLUS_BG" "$TMP_MINUS_BG"

echo "RNA 3' Preprocessing complete for ${CELL_TYPE}."
conda deactivate
