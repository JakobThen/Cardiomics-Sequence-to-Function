#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: Worker script for ATAC BAM to BigWig conversion. Runs per cell type
#          in a SLURM array. Converts, filters canonical chromosomes, and
#          normalizes to 100M total insertions.
#          Submitted by workflows/input_prep/01_atac_bigwig.sh (partition and
#          log files are set there from config/site.env and config/paths.yaml).
# Input:   $1 - BAM Input Directory
#          $2 - BigWig Output Directory
#          $3 - Path to the Config file containing valid cell types
#          $4 - Reference genome FASTA
#          $5 - Chromosome sizes file
# ==============================================================================

#SBATCH --job-name=ATAC_BW_preprocess
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

source "${CS2F_REPO_ROOT:?Submit via workflows/input_prep/01_atac_bigwig.sh}/workflows/common/load_site_env.sh"

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
REFERENCE_GENOME="$4"
CHROM_SIZES="$5"

# Read specific cell type parameter for this specific Array Task ID
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CONFIG_FILE")
CELL_TYPE=$(echo "$PARAMS" | awk '{print $1}')

echo "================================================="
echo "Running Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing Cell Type:  $CELL_TYPE"
echo "BAM Directory:         $BAM_DIR"
echo "Output Directory:      $BW_OUTPUT_DIR"
echo "================================================="

# File definitions
BAM_FILE="${BAM_DIR}/${CELL_TYPE}.bam"
OUT_PREFIX="${BW_OUTPUT_DIR}/ATAC_${CELL_TYPE}"

ASSAY_TYPE="ATAC"

# Ensure output directory exists before writing to it
mkdir -p "$BW_OUTPUT_DIR"

# ------------------------------------------------------------------------------
# Processing Pipeline
# ------------------------------------------------------------------------------
cs2f_activate "$CS2F_ENV_CHROMBPNET"

echo "  1/3 Converting ${CELL_TYPE} to BigWig..."

#use -m module falg to load the chrombpnet script as module
python -m chrombpnet.helpers.preprocessing.reads_to_bigwig \
    -ibam "$BAM_FILE" \
    -g "$REFERENCE_GENOME" \
    -c "$CHROM_SIZES" \
    -op "$OUT_PREFIX" \
    -d "$ASSAY_TYPE"

# Intermediate and Final File Tracking
RAW_BW="${OUT_PREFIX}_unstranded.bw"
FILTERED_RAW_BW="${OUT_PREFIX}_filtered_raw.bw"
NORM_BW="${OUT_PREFIX}_100M_norm.bw"
TMP_BG="${OUT_PREFIX}_temp.bg"
TMP_NORM_BG="${OUT_PREFIX}_temp_norm.bg"

echo "  2/3 Filtering non-canonical chromosomes for ${CELL_TYPE}..."
# Convert to bedGraph, keep ONLY chr1-22 and chrX, output to temporary background file
bigWigToBedGraph "$RAW_BW" stdout | awk '$1 ~ /^chr([1-9]|1[0-9]|2[0-2]|X)$/' > "$TMP_BG"

# Save filtered RAW BigWig
bedGraphToBigWig "$TMP_BG" "$CHROM_SIZES" "$FILTERED_RAW_BW"

echo "  3/3 Normalizing ${CELL_TYPE} to 100 million insertions..."
# Extract Total Inserts directly from the filtered BigWig header
TOTAL_INSERTS=$(python -c "
import pyBigWig
try:
    bw = pyBigWig.open('$FILTERED_RAW_BW')
    print(int(bw.header()['sumData']))
    bw.close()
except Exception as e:
    pass
")

if [[ -z "$TOTAL_INSERTS" || "$TOTAL_INSERTS" == "0" ]]; then
    echo "Error: Total inserts parsed as zero or empty. Check the PyBigWig extraction step."
    exit 1
fi

# Calculate normalization factor
SCALE_FACTOR=$(awk -v total="$TOTAL_INSERTS" 'BEGIN {printf "%.10f\n", 100000000 / total}')

echo "      Total filtered insertions: ${TOTAL_INSERTS}"
echo "      Scaling factor:            ${SCALE_FACTOR}"

# Apply scaling factor column by column
awk -v OFS="\t" -v scale="$SCALE_FACTOR" '{$4 = $4 * scale; print $0}' "$TMP_BG" > "$TMP_NORM_BG"

# Convert to the final normalized BigWig
bedGraphToBigWig "$TMP_NORM_BG" "$CHROM_SIZES" "$NORM_BW"

# Clean up massive intermediate files to save disk space
rm -f "$TMP_BG" "$TMP_NORM_BG" "$RAW_BW" "$FILTERED_RAW_BW"

echo "Preprocessing complete for ${CELL_TYPE}."
conda deactivate
