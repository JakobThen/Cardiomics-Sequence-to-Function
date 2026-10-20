#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: Submits variant effect prediction (VEP) with the fine-tuned
#          AlphaGenome model (8 GPUs).
# Usage:   bash workflows/alphagenome_ft/04_vep.sh
#          INPUT_VCF=/path/to/variants.csv bash workflows/alphagenome_ft/04_vep.sh
# Input:   INPUT_VCF (CSV with variant_id, CHROM, POS, REF, ALT)
#          [default: <variant_data_dir>/VCF_all_Richter_vars.csv],
#          <ag_targets_config>, best checkpoint in <ag_ft_checkpoint_dir>
# Output:  <vep_dir>/AG_ft_VEP_out/VEP_AG_ft_fold_1_<vcf name>.csv
# ==============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/load_site_env.sh"
WORKFLOW_DIR="${CS2F_REPO_ROOT}/workflows/alphagenome_ft"

INPUT_VCF="${INPUT_VCF:-$(cs2f_path variant_data_dir)/VCF_all_Richter_vars.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$(cs2f_path vep_dir)/AG_ft_VEP_out}"
TARGETS_CONFIG="${TARGETS_CONFIG:-$(cs2f_path ag_targets_config)}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$(cs2f_path ag_ft_checkpoint_dir)}"  # uses its 'best' checkpoint
BATCH_SIZE="${BATCH_SIZE:-24}"

for required in "$INPUT_VCF" "$TARGETS_CONFIG"; do
    if [[ ! -f "$required" ]]; then
        echo "Error: ${required} not found." >&2
        exit 1
    fi
done
if [[ ! -d "${CHECKPOINT_DIR}/best" ]]; then
    echo "Error: no 'best' checkpoint found in ${CHECKPOINT_DIR}." >&2
    exit 1
fi

cs2f_sbatch alphagenome_ft gpu "${WORKFLOW_DIR}/jobs/vep.slurm" \
    "$(cs2f_path xla_cache_dir)" \
    --input_vcf "$INPUT_VCF" \
    --out_dir "$OUTPUT_DIR" \
    --fasta_path "$(cs2f_path genome_fasta)" \
    --chrom_sizes "$(cs2f_path chrom_sizes)" \
    --config "$TARGETS_CONFIG" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --model_version "fold_1" \
    --batch_size "$BATCH_SIZE"
