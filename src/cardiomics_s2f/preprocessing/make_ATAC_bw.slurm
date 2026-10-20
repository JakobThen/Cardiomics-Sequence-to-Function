#!/bin/bash
# ==============================================================================
# Code by Jakob
# 
# Purpose: Worker script for ATAC BAM to BigWig conversion. Runs per cell type 
#          in a SLURM array. Converts, filters canonical chromosomes, and 
#          normalizes to 100M total insertions.
# Input:   $1 - BAM Input Directory
#          $2 - BigWig Output Directory
#          $3 - Path to the Config file containing valid cell types
# ==============================================================================

#SBATCH --job-name=ATAC_BW_preprocess
#SBATCH --output=logs/ATAC_BW_preprocess_%A_%a.out
#SBATCH --error=logs/ATAC_BW_preprocess_%A_%a.err
#SBATCH --time=5:00:00
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# ------------------------------------------------------------------------------
# Argument Parsing
# ------------------------------------------------------------------------------
if [ "$#" -ne 3 ]; then
    echo "Error: Expected 3 arguments (BAM_DIR, BW_OUTPUT_DIR, CONFIG_FILE). Got $#."
    exit 1
fi

BAM_DIR="$1"
BW_OUTPUT_DIR="$2"
CONFIG_FILE="$3"

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

# Reference genome parameters
REFERENCE_DIR="../../input_data/metadata/GRCh38_gencode_release29_genome_fasta"
CHROM_SIZES="${REFERENCE_DIR}/genome.chrom.sizes"
REFERENCE_GENOME="${REFERENCE_DIR}/genome.fa"

ASSAY_TYPE="ATAC"

# Ensure output directory exists before writing to it
mkdir -p "$BW_OUTPUT_DIR"

# ------------------------------------------------------------------------------
# Processing Pipeline
# ------------------------------------------------------------------------------
source /g/easybuild/x86_64/Rocky/8/haswell/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate /g/steinmetz/projects/variant2function_project/env/chrombpnet
#conda activate chrombpnet


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