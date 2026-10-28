#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: Submits ChromBPNet variant effect prediction with the Kundaje lab
#          variant-scorer: variants are converted once, then scored (and SHAP
#          contributions computed) with every fold of every cell type and averaged
#          across folds.
# Usage:   bash workflows/chrombpnet/03_vep.sh
#          INPUT_VCF=/path/to/variants.csv CELL_TYPES="Cardiomyocytes" bash workflows/chrombpnet/03_vep.sh
#          SHAP=0 (skip contribution scores), FOLDS=..., N_GPUS=2,
#          GPU_CONSTRAINT="gpu=A100" (TensorFlow 2.8 / CUDA 11; "" = site default)
# Input:   INPUT_VCF (CSV with variant_id, CHROM, POS, REF, ALT) [default: <vep_input_vcf>],
#          <chrombpnet_model_dir>/<cell type>_<fold>/models/chrombpnet_nobias.h5,
#          <genome_fasta>, <chrom_sizes>, variant-scorer (install_variant_scorer.sh)
# Output:  <vep_dir>/chromBPNet_VEP_out/<vcf name>_<cell type>.* (summary across folds),
#          <vcf name>_<cell type>_<fold>_* (SHAP, if SHAP=1)
# ==============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/load_site_env.sh"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/variant_scorer.sh"
WORKFLOW_DIR="${CS2F_REPO_ROOT}/workflows/chrombpnet"

INPUT_VCF="${INPUT_VCF:-$(cs2f_path vep_input_vcf)}"
MODEL_DIR="${MODEL_DIR:-$(cs2f_path chrombpnet_model_dir)}"
OUTPUT_DIR="${OUTPUT_DIR:-$(cs2f_path vep_dir)/chromBPNet_VEP_out}"
CELL_TYPES="${CELL_TYPES:-Cardiomyocytes Fibroblasts Cardiac_progenitors}"
FOLDS="${FOLDS:-fold_0 fold_1 fold_2 fold_3 fold_4}"
SHAP="${SHAP:-1}"
N_GPUS="${N_GPUS:-2}"
GPU_CONSTRAINT="${GPU_CONSTRAINT-gpu=A100}"

if [[ ! -f "$INPUT_VCF" ]]; then
    echo "Error: ${INPUT_VCF} not found." >&2
    exit 1
fi
cs2f_check_variant_scorer

# Check every model before submitting
for cell_type in $CELL_TYPES; do
    for fold in $FOLDS; do
        model="${MODEL_DIR}/${cell_type}_${fold}/models/chrombpnet_nobias.h5"
        if [[ ! -f "$model" ]]; then
            echo "Error: model not found: ${model}" >&2
            exit 1
        fi
    done
done

SBATCH_OPTS=(--gpus="$N_GPUS")
[[ -n "$GPU_CONSTRAINT" ]] && SBATCH_OPTS+=(--constraint="$GPU_CONSTRAINT")

cs2f_sbatch chrombpnet gpu "${SBATCH_OPTS[@]}" "${WORKFLOW_DIR}/jobs/vep.slurm" \
    "$INPUT_VCF" "$MODEL_DIR" "$OUTPUT_DIR" "$(cs2f_path genome_fasta)" "$(cs2f_path chrom_sizes)" \
    "$CELL_TYPES" "$FOLDS" "$SHAP"
