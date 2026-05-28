#!/bin/bash
# ==============================================================================
# Code by Jakob
# 
# Purpose: LEGACY Worker script for RNA BAM to 3' BigWig conversion. Runs per cell type 
#          in a SLURM array. DOES NOT SPLIT BY STRAND, filters for 
#          canonical chromosomes, and normalizes so counts sum to 100M.
# ==============================================================================

#SBATCH --job-name=RNA_BW_preprocess
#SBATCH --output=logs/RNA_BW_preprocess_%A_%a.out
#SBATCH --error=logs/RNA_BW_preprocess_%A_%a.err
#SBATCH --time=3:00:00
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G


# Read params
CONFIG_FILE="utils/preprocessing/valid_RNA_pseudobulks.txt"
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CONFIG_FILE")
DAY_NAME=$(echo "$PARAMS" | awk '{print $1}')
CELL_TYPE=$(echo "$PARAMS" | awk '{print $2}')

echo "================================================="
echo "Running Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing RNA for Day: $DAY_NAME | Cell Type: $CELL_TYPE"
echo "================================================="

# I/O Paths
BASE_DIR="/g/steinmetz/projects/variant2function_project"
BAM_DIR="${BASE_DIR}/data/processed/multiome/multiome_pseuodobulk_RNA"
BW_OUTPUT_DIR="${BASE_DIR}/data/processed/gReLu_input/multiome_RNA_bigwig"

BAM_FILE="${BAM_DIR}/${DAY_NAME}/${CELL_TYPE}.bam"
OUT_PREFIX="${BW_OUTPUT_DIR}/${DAY_NAME}_${CELL_TYPE}"

RAW_BW="${OUT_PREFIX}_raw_coverage.bw"
NORM_BW="${OUT_PREFIX}_100M_norm.bw"
TMP_BG="${OUT_PREFIX}_temp.bg"
FILTERED_RAW_BW="${OUT_PREFIX}_filtered_raw.bw"
TMP_NORM_BG="${OUT_PREFIX}_temp_norm.bg"

# Reference genome
REFERENCE_DIR="../../input_data/metadata/GRCh38_gencode_release29_genome_fasta"
CHROM_SIZES="${REFERENCE_DIR}/genome.chrom.sizes"
REFERENCE_GENOME="${REFERENCE_DIR}/genome.fa"

mkdir -p "$BW_OUTPUT_DIR"

# Load Environment (assuming deepTools is in this environment)
source /g/easybuild/x86_64/Rocky/8/haswell/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate /g/steinmetz/projects/variant2function_project/env/chrombpnet

# make raw counts BigWig using deepTools
echo "  Generating raw RNA coverage for ${DAY_NAME}_${CELL_TYPE}..."
bamCoverage \
    --bam "$BAM_FILE" \
    --outFileName "$RAW_BW" \
    --binSize 1 \
    --normalizeUsing None \
    --numberOfProcessors "$SLURM_CPUS_PER_TASK"

echo "  Filtering non-canonical chromosomes for ${DAY_NAME}_${CELL_TYPE}..."

# convert to bedgraph for chormosme filtering keep ONLY chr1-22, chrX
bigWigToBedGraph "$RAW_BW" stdout | awk '$1 ~ /^chr([1-9]|1[0-9]|2[0-2]|X)$/' > "$TMP_BG"

# save filtered raw BigWig
bedGraphToBigWig "$TMP_BG" "$CHROM_SIZES" "$FILTERED_RAW_BW"

echo "  Normalizing ${DAY_NAME}_${CELL_TYPE} to 100 million total signal..."

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
rm -f "$TMP_BG"
rm -f "$TMP_NORM_BG"
rm -f "$RAW_BW"

echo "RNA Preprocessing complete for ${DAY_NAME}_${CELL_TYPE}."
conda deactivate