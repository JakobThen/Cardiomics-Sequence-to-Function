#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: Submits DeepSHAP attributions of the fine-tuned AlphaGenome model for
#          the regions of a BED file (1 GPU).
# Usage:   REGIONS=workflows/alphagenome_ft/configs/sdr_amplicons.bed bash workflows/alphagenome_ft/05_attributions.sh
#          REGIONS=... TRACKS="ATAC_Cardiomyocytes ATAC_Fibroblasts" OUT_PREFIX=sdr_amplicons bash workflows/alphagenome_ft/05_attributions.sh
# Input:   REGIONS (BED without header: chrom, start, end[, name]; template:
#          configs/attribution_regions.example.bed), <ag_targets_config>,
#          best checkpoint in <ag_ft_checkpoint_dir>, <genome_fasta>
# Output:  <attribution_dir>/<OUT_PREFIX>_<track>.npy, <OUT_PREFIX>_regions.tsv
# ==============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/load_site_env.sh"
WORKFLOW_DIR="${CS2F_REPO_ROOT}/workflows/alphagenome_ft"

REGIONS="${REGIONS:?Set REGIONS=<regions.bed>}"
OUT_DIR="${OUT_DIR:-$(cs2f_path attribution_dir)}"
OUT_PREFIX="${OUT_PREFIX:-$(basename "$REGIONS" .bed)}"
TARGETS_CONFIG="${TARGETS_CONFIG:-$(cs2f_path ag_targets_config)}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$(cs2f_path ag_ft_checkpoint_dir)}"  # uses its 'best' checkpoint
HEAD="${HEAD:-multiome_ATAC}"
read -r -a TRACKS <<< "${TRACKS:-ATAC_Cardiomyocytes}"
WINDOW_SIZE="${WINDOW_SIZE:-16384}"
CENTER_MASK_SIZE="${CENTER_MASK_SIZE:-500}"      # 0 = attribute the whole window
N_REFERENCES="${N_REFERENCES:-30}"

for required in "$REGIONS" "$TARGETS_CONFIG"; do
    if [[ ! -f "$required" ]]; then
        echo "Error: ${required} not found." >&2
        exit 1
    fi
done
if [[ ! -d "${CHECKPOINT_DIR}/best" ]]; then
    echo "Error: no 'best' checkpoint found in ${CHECKPOINT_DIR}." >&2
    exit 1
fi

cs2f_sbatch alphagenome_ft gpu "${WORKFLOW_DIR}/jobs/attributions.slurm" \
    "$(cs2f_path xla_cache_dir)" \
    --regions "$REGIONS" \
    --config "$TARGETS_CONFIG" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --fasta_path "$(cs2f_path genome_fasta)" \
    --out_dir "$OUT_DIR" \
    --out_prefix "$OUT_PREFIX" \
    --head "$HEAD" \
    --tracks "${TRACKS[@]}" \
    --window_size "$WINDOW_SIZE" \
    --center_mask_size "$CENTER_MASK_SIZE" \
    --n_references "$N_REFERENCES" \
    --model_version "fold_1"
