#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: Submits variant effect prediction (VEP) with a fine-tuned Borzoi
#          model (1 GPU).
# Usage:   bash workflows/borzoi/04_vep.sh
#          INPUT_VCF=/path/to/variants.csv bash workflows/borzoi/04_vep.sh
#          RNA_SCALE=linear CHECKPOINT=/path/to/model.ckpt bash workflows/borzoi/04_vep.sh
# Input:   INPUT_VCF (CSV with variant_id, CHROM, POS, REF, ALT) [default: <vep_input_vcf>],
#          CHECKPOINT (.ckpt, or a folder with exactly one epoch*.ckpt) [default: <borzoi_checkpoint_dir>],
#          <gtf>, <chrom_sizes>
#          RNA_SCALE: squashed (default; models trained on squashed RNA) or linear
# Output:  <vep_dir>/Borzoi_VEP/VEP_Borzoi_<vcf name>.csv
# ==============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/load_site_env.sh"
WORKFLOW_DIR="${CS2F_REPO_ROOT}/workflows/borzoi"

INPUT_VCF="${INPUT_VCF:-$(cs2f_path vep_input_vcf)}"
CHECKPOINT="${CHECKPOINT:-$(cs2f_path borzoi_checkpoint_dir)}"
OUTPUT_DIR="${OUTPUT_DIR:-$(cs2f_path vep_dir)/Borzoi_VEP}"
RNA_SCALE="${RNA_SCALE:-squashed}"
WORKERS="${WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"

if [[ ! -f "$INPUT_VCF" ]]; then
    echo "Error: ${INPUT_VCF} not found." >&2
    exit 1
fi
if [[ -d "$CHECKPOINT" ]]; then
    n_ckpt="$(find "$CHECKPOINT" -name 'epoch*.ckpt' | wc -l)"
    if [[ "$n_ckpt" -ne 1 ]]; then
        echo "Error: ${CHECKPOINT} contains ${n_ckpt} epoch*.ckpt files (need exactly one); set CHECKPOINT to a .ckpt file." >&2
        exit 1
    fi
elif [[ ! -f "$CHECKPOINT" ]]; then
    echo "Error: CHECKPOINT=${CHECKPOINT} not found." >&2
    exit 1
fi

cs2f_sbatch borzoi gpu "${WORKFLOW_DIR}/jobs/vep.slurm" \
    --input_vcf "$INPUT_VCF" \
    --checkpoint "$CHECKPOINT" \
    --out_dir "$OUTPUT_DIR" \
    --gtf "$(cs2f_path gtf)" \
    --chrom_sizes "$(cs2f_path chrom_sizes)" \
    --rna_scale "$RNA_SCALE" \
    --devices 0 \
    --workers "$WORKERS" \
    --batch_size "$BATCH_SIZE"
