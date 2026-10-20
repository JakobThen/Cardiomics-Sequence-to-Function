#!/bin/bash
# ==============================================================================
# Code by Jakob
# 
# Purpose: Worker script for Cut&Tag BAM to BigWig conversion. Runs per sample 
#          in a SLURM array. Normalizes to 100M insertions and filters for 
#          canonical chromosomes. Currently applies +4/-4 shift correction.
# Input:   $1 - BigWig Output Directory
#          $2 - Path to the Config file containing full BAM paths
# ==============================================================================

#SBATCH --job-name=CutnTag_BW_preprocess
#SBATCH --output=logs/CnT_BW_preprocess_%A_%a.out
#SBATCH --error=logs/CnT_BW_preprocess_%A_%a.err
#SBATCH --time=3:00:00
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# ------------------------------------------------------------------------------
# Argument Parsing
# ------------------------------------------------------------------------------
if [ "$#" -ne 2 ]; then
    echo "Error: Expected 2 arguments (BW_OUTPUT_DIR, RUN_CONFIG). Got $#."
    exit 1
fi

BW_OUTPUT_DIR="$1"
RUN_CONFIG="$2"

# Read the full BAM path directly from the config based on Array Task ID
BAM_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$RUN_CONFIG")

# Extract a clean name for the output
SAMPLE_NAME=$(basename "$BAM_FILE" .bam)

echo "================================================="
echo "Running Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing Cut&Tag Sample: $SAMPLE_NAME"
echo "BAM File Path:         $BAM_FILE"
echo "Output Directory:      $BW_OUTPUT_DIR"
echo "================================================="

# I/O File Paths
OUT_PREFIX="${BW_OUTPUT_DIR}/${SAMPLE_NAME}"
RAW_BW="${OUT_PREFIX}_unstranded.bw"
FILTERED_RAW_BW="${OUT_PREFIX}_filtered_raw.bw"
NORM_BW="${OUT_PREFIX}_100M_norm.bw"
TMP_BG="${OUT_PREFIX}_temp.bg"
TMP_NORM_BG="${OUT_PREFIX}_temp_norm.bg"

# Reference genome
REFERENCE_DIR="../../input_data/metadata/GRCh38_gencode_release29_genome_fasta"
CHROM_SIZES="${REFERENCE_DIR}/genome.chrom.sizes"
REFERENCE_GENOME="${REFERENCE_DIR}/genome.fa"

# "ATAC" applies a +4/-4 Tn5 shift.
ASSAY_TYPE="ATAC"

mkdir -p "$BW_OUTPUT_DIR"

# ------------------------------------------------------------------------------
# Processing Pipeline
# ------------------------------------------------------------------------------
source /g/easybuild/x86_64/Rocky/8/haswell/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate /g/steinmetz/projects/variant2function_project/env/chrombpnet

# Call chrombpnet script dynamically via the python module flag (-m)
echo "  1/3 Converting ${SAMPLE_NAME} to BigWig..."
python -m chrombpnet.helpers.preprocessing.reads_to_bigwig \
    -ibam "$BAM_FILE" \
    -g "$REFERENCE_GENOME" \
    -c "$CHROM_SIZES" \
    -op "$OUT_PREFIX" \
    -d "$ASSAY_TYPE"    

echo "  2/3 Filtering non-canonical chromosomes for ${SAMPLE_NAME}..."
bigWigToBedGraph "$RAW_BW" stdout | awk '$1 ~ /^chr([1-9]|1[0-9]|2[0-2]|X)$/' > "$TMP_BG"
bedGraphToBigWig "$TMP_BG" "$CHROM_SIZES" "$FILTERED_RAW_BW"

echo "  3/3 Normalizing ${SAMPLE_NAME} to 100 million insertions..."
# Extract total inserts
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
    echo "Error: Total inserts parsed as zero or empty. Exiting."
    exit 1
fi

SCALE_FACTOR=$(awk -v total="$TOTAL_INSERTS" 'BEGIN {printf "%.10f\n", 100000000 / total}')

echo "      Total filtered insertions: ${TOTAL_INSERTS}"
echo "      Scaling factor:            ${SCALE_FACTOR}"

# Scale
awk -v OFS="\t" -v scale="$SCALE_FACTOR" '{$4 = $4 * scale; print $0}' "$TMP_BG" > "$TMP_NORM_BG"
bedGraphToBigWig "$TMP_NORM_BG" "$CHROM_SIZES" "$NORM_BW"

# Cleanup
rm -f "$TMP_BG" "$TMP_NORM_BG" "$RAW_BW" "$FILTERED_RAW_BW"

echo "Cut&Tag Preprocessing complete for ${SAMPLE_NAME}."
conda deactivate