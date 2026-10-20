#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: Submits AlphaGenome head fine-tuning (8 GPUs). Data is staged to the
#          GPU node's local disk for fast loading.
# Usage:   bash workflows/alphagenome_ft/02_finetune_heads.sh
#          Resume a crashed / epoch-limited run from its last checkpoint:
#          RESUME_FROM=/path/to/previous/checkpoints bash workflows/alphagenome_ft/02_finetune_heads.sh
# Input:   <ag_targets_config> (step 01), <genome_fasta>
# Output:  Checkpoints for the best and last epoch in OUTPUT_DIR:
#          <ag_ft_train_dir>, or <ag_ft_train_dir>_resumed when resuming
# ==============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/load_site_env.sh"
WORKFLOW_DIR="${CS2F_REPO_ROOT}/workflows/alphagenome_ft"

# Setup core paths and training hyperparameters
TARGETS_CONFIG="${TARGETS_CONFIG:-$(cs2f_path ag_targets_config)}"
GENOME_FASTA="$(cs2f_path genome_fasta)"
RESUME_FROM="${RESUME_FROM:-}"
if [[ -n "$RESUME_FROM" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR:-$(cs2f_path ag_ft_train_dir)_resumed}"
    JOB_NAME="AG_resume_finetune_heads"
else
    OUTPUT_DIR="${OUTPUT_DIR:-$(cs2f_path ag_ft_train_dir)}"
    JOB_NAME="AG_finetune_heads"
fi

BATCH_SIZE="${BATCH_SIZE:-16}"  # 2 intervals per GPU. Must be a multiple of the number of GPUs.
MAX_EPOCHS="${MAX_EPOCHS:-10}"
LR="${LR:-3e-4}"

if [[ ! -f "$TARGETS_CONFIG" ]]; then
    echo "Error: targets config ${TARGETS_CONFIG} not found. Run 01_make_targets_config.sh first." >&2
    exit 1
fi
if [[ -n "$RESUME_FROM" && ! -d "${RESUME_FROM}/last" ]]; then
    echo "Error: no 'last' checkpoint found in RESUME_FROM=${RESUME_FROM}." >&2
    exit 1
fi

TRAIN_ARGS=(
    --checkpoint_dir "$OUTPUT_DIR"
    --batch_size "$BATCH_SIZE"
    --epochs "$MAX_EPOCHS"
    --lr "$LR"
)
[[ -n "$RESUME_FROM" ]] && TRAIN_ARGS+=(--resume_from "$RESUME_FROM")

cs2f_sbatch alphagenome_ft gpu --job-name="$JOB_NAME" "${WORKFLOW_DIR}/jobs/finetune_heads.slurm" \
    "$TARGETS_CONFIG" "$GENOME_FASTA" "$(cs2f_path node_tmp_dir)" "$(cs2f_path xla_cache_dir)" \
    "${TRAIN_ARGS[@]}"
