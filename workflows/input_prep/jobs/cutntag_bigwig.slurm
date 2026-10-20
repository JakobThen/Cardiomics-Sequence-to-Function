#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: Worker script for Cut&Tag BAM to BigWig conversion. Runs per sample
#          in a SLURM array. Normalizes to 100M insertions and filters for
#          canonical chromosomes. Currently applies +4/-4 shift correction.
#          Submitted by workflows/input_prep/03_cutntag_bigwig.sh.
# Input:   $1 - BigWig Output Directory
#          $2 - Path to the Config file containing full BAM paths
#          $3 - Reference genome FASTA
#          $4 - Chromosome sizes file
# ==============================================================================

#SBATCH --job-name=CutnTag_BW_preprocess
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

source "${CS2F_REPO_ROOT:?Submit via workflows/input_prep/03_cutntag_bigwig.sh}/workflows/common/load_site_env.sh"

# ------------------------------------------------------------------------------
# Argument Parsing
# ------------------------------------------------------------------------------
if [ "$#" -ne 4 ]; then
    echo "Error: Expected 4 arguments (BW_OUTPUT_DIR, RUN_CONFIG, GENOME_FASTA, CHROM_SIZES). Got $#."
    exit 1
fi

BW_OUTPUT_DIR="$1"
RUN_CONFIG="$2"
REFERENCE_GENOME="$3"
CHROM_SIZES="$4"

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

# "ATAC" applies a +4/-4 Tn5 shift.
ASSAY_TYPE="ATAC"

mkdir -p "$BW_OUTPUT_DIR"

# ------------------------------------------------------------------------------
# Processing Pipeline
# ------------------------------------------------------------------------------
cs2f_activate "$CS2F_ENV_CHROMBPNET"

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
