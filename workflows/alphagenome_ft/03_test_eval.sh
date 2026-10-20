#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: Submits the test-set evaluation of a fine-tuned AlphaGenome model
#          (streamed correlation analysis against the BigWig tracks, 8 GPUs).
# Usage:   bash workflows/alphagenome_ft/03_test_eval.sh
#          Optional switches: SAVE_PREDICTIONS=1 (write predictions to HDF5),
#          MINIMAL_TEST=1 (first 8 batches only, for debugging),
#          RESOLUTION=128 (evaluate the 128 bp transformer embedding)
# Input:   <ag_targets_config>, best checkpoint in <ag_ft_checkpoint_dir>
#          (override: CHECKPOINT_DIR=...), <genome_fasta>, <gtf>
# Output:  Test evaluation metrics and figures in <model_eval_dir>/AlphaGenome_ft
# ==============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/load_site_env.sh"
WORKFLOW_DIR="${CS2F_REPO_ROOT}/workflows/alphagenome_ft"

# Setup core paths
TARGETS_CONFIG="${TARGETS_CONFIG:-$(cs2f_path ag_targets_config)}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$(cs2f_path ag_ft_checkpoint_dir)}"  # uses its 'best' checkpoint
OUTPUT_DIR="${OUTPUT_DIR:-$(cs2f_path model_eval_dir)/AlphaGenome_ft}"
GENOME_FASTA="$(cs2f_path genome_fasta)"
GTF_FILE="$(cs2f_path gtf)"

RESOLUTION="${RESOLUTION:-1}"  # 1 (decoder) or 128 (transformer) based on what model embedding to evaluate
BATCH_SIZE="${BATCH_SIZE:-16}"

if [[ ! -f "$TARGETS_CONFIG" ]]; then
    echo "Error: targets config ${TARGETS_CONFIG} not found." >&2
    exit 1
fi
if [[ ! -d "${CHECKPOINT_DIR}/best" ]]; then
    echo "Error: no 'best' checkpoint found in ${CHECKPOINT_DIR}." >&2
    exit 1
fi

EVAL_ARGS=(
    --out_dir "$OUTPUT_DIR"
    --checkpoint_dir "$CHECKPOINT_DIR"
    --gtf_file "$GTF_FILE"
    --fold 1
    --model_version "fold_1"
    --batch_size "$BATCH_SIZE"
    --resolution "$RESOLUTION"
)
[[ "${SAVE_PREDICTIONS:-0}" == "1" ]] && EVAL_ARGS+=(--save_predictions)
[[ "${MINIMAL_TEST:-0}" == "1" ]] && EVAL_ARGS+=(--minimal_test)

cs2f_sbatch alphagenome_ft gpu "${WORKFLOW_DIR}/jobs/test_eval.slurm" \
    "$TARGETS_CONFIG" "$GENOME_FASTA" "$(cs2f_path node_tmp_dir)" "$(cs2f_path xla_cache_dir)" \
    "${EVAL_ARGS[@]}"
