#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: SLURM job for step 03. Converts the variants once, then runs the
#          variant-scorer scripts per cell type and fold and summarizes across folds.
#          Temporary files (converted variants, per-fold scores) are removed on exit.
# Input:   $1 - Variant CSV         $2 - Model folder (<cell type>_<fold>/models/)
#          $3 - Output folder       $4 - Genome FASTA       $5 - Chromosome sizes
#          $6 - Cell types (space-separated)                $7 - Folds (space-separated)
#          $8 - SHAP (1: compute variant SHAP scores, 0: skip)
# ==============================================================================

#SBATCH --job-name=ChromBPNet_VEP
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=64G
#SBATCH --time=1-00:00:00

set -eo pipefail   # no -u: conda activate scripts use unset variables
source "${CS2F_REPO_ROOT:?Submit via workflows/chrombpnet/03_vep.sh}/workflows/common/load_site_env.sh"
source "${CS2F_REPO_ROOT}/workflows/chrombpnet/variant_scorer.sh"

INPUT_VCF="$1"
MODEL_DIR="$2"
OUTPUT_DIR="$3"
GENOME="$4"
CHROM_SIZES="$5"
CELL_TYPES="$6"
FOLDS="$7"
SHAP="$8"

VCF_BASENAME=$(basename "${INPUT_VCF}" .csv)
TMP_TSV_FILE="${OUTPUT_DIR}/TMP_variants_for_chrombpnet_${VCF_BASENAME}.tsv"
SCORER="${VARIANT_SCORER_DIR}/src"

log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Remove temporary files on exit (success or failure)
cleanup() {
    log "--- Triggering exit cleanup ---"
    rm -f "${TMP_TSV_FILE}"
    for cell_type in $CELL_TYPES; do
        rm -f "${OUTPUT_DIR}/${cell_type}_fold_"*"_TMP_${VCF_BASENAME}.variant_scores.tsv"
    done
    log "Cleanup complete."
}
trap cleanup EXIT

log "--- Starting ChromBPNet VEP Pipeline (job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}) ---"
mkdir -p "$OUTPUT_DIR"
cs2f_activate "$CS2F_ENV_CHROMBPNET"
cs2f_check_variant_scorer

log "--- Step 1: Converting variants to the variant-scorer format ---"
python "${CS2F_REPO_ROOT}/workflows/chrombpnet/convert_variants.py" \
    --input_file "${INPUT_VCF}" \
    --output_file "${TMP_TSV_FILE}" \
    --genome "${GENOME}"

for cell_type in $CELL_TYPES; do
    log "--- Step 2: Scoring variants across folds for ${cell_type} ---"
    score_list=()
    for fold in $FOLDS; do
        log ">> Processing ${fold}..."
        FOLD_MODEL_PATH="${MODEL_DIR}/${cell_type}_${fold}/models/chrombpnet_nobias.h5"

        python "${SCORER}/variant_scoring.py" \
            --list "${TMP_TSV_FILE}" \
            --genome "${GENOME}" \
            --model "${FOLD_MODEL_PATH}" \
            --out_prefix "${OUTPUT_DIR}/${cell_type}_${fold}_TMP_${VCF_BASENAME}" \
            --chrom_sizes "${CHROM_SIZES}" \
            --schema chrombpnet
        score_list+=("${cell_type}_${fold}_TMP_${VCF_BASENAME}.variant_scores.tsv")

        if [[ "$SHAP" == "1" ]]; then
            log "   Running variant_shap.py for ${fold}..."
            python "${SCORER}/variant_shap.py" \
                --list "${TMP_TSV_FILE}" \
                --genome "${GENOME}" \
                --chrom_sizes "${CHROM_SIZES}" \
                --model "${FOLD_MODEL_PATH}" \
                --out_prefix "${OUTPUT_DIR}/${VCF_BASENAME}_${cell_type}_${fold}_" \
                --shap_type counts \
                --schema chrombpnet
        fi
    done

    log "--- Step 3: Summarizing variant scores across folds for ${cell_type} ---"
    python "${SCORER}/variant_summary_across_folds.py" \
        --score_dir "${OUTPUT_DIR}" \
        --score_list "${score_list[@]}" \
        --out_prefix "${OUTPUT_DIR}/${VCF_BASENAME}_${cell_type}" \
        --schema chrombpnet
done

conda deactivate
log "VEP done."
