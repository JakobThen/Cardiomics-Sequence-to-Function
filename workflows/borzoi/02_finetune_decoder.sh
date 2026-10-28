#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: Submits Borzoi stage 2: resumes from a stage-1 checkpoint and unfreezes
#          the decoder (U-Net tower + pointwise conv), 4 GPUs. As in the original
#          run, the whole bigwig_dir is staged; tracks and their order come from
#          the checkpoint, so every track file name must occur in only one
#          *bigwig* folder there (see configs/README.md).
# Usage:   bash workflows/borzoi/02_finetune_decoder.sh
#          PREV_CHECKPOINT=/path/to/run_or.ckpt bash workflows/borzoi/02_finetune_decoder.sh
#          Transformer stage (no own launcher; it was worse than the decoder stage):
#          PREV_CHECKPOINT=<decoder run> OUTPUT_DIR=<dir> UNFREEZE="transformer_tower unet_tower pointwise_conv" \
#              LR=1e-5 JOB_NAME=Borzoi_finetune_transformer bash workflows/borzoi/02_finetune_decoder.sh
# Input:   PREV_CHECKPOINT (.ckpt, or a folder with exactly one epoch*.ckpt) [default: <borzoi_heads_dir>],
#          <bigwig_dir>, <intervals_dir>/Borzoi_human_rep0_{train,val,test}_intervals.bed
# Output:  <borzoi_decoder_dir>/<run>/version_0/checkpoints/,
#          <borzoi_decoder_dir>/<PREFIX>_squashed_predictions.h5 (test intervals)
# ==============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/load_site_env.sh"
WORKFLOW_DIR="${CS2F_REPO_ROOT}/workflows/borzoi"

BIGWIG_DIR="${BIGWIG_DIR:-$(cs2f_path bigwig_dir)}"
INTERVALS_DIR="${INTERVALS_DIR:-$(cs2f_path intervals_dir)}"
PREV_CHECKPOINT="${PREV_CHECKPOINT:-$(cs2f_path borzoi_heads_dir)}"
OUTPUT_DIR="${OUTPUT_DIR:-$(cs2f_path borzoi_decoder_dir)}"
PREFIX="${PREFIX:-Borzoi_rep0_decoder}"
JOB_NAME="${JOB_NAME:-Borzoi_finetune_decoder}"
UNFREEZE="${UNFREEZE:-unet_tower pointwise_conv}"
N_GPUS="${N_GPUS:-4}"
LR="${LR:-1e-4}"
MAX_EPOCHS="${MAX_EPOCHS:-15}"

if [[ -d "$PREV_CHECKPOINT" ]]; then
    n_ckpt="$(find "$PREV_CHECKPOINT" -name 'epoch*.ckpt' | wc -l)"
    if [[ "$n_ckpt" -ne 1 ]]; then
        echo "Error: ${PREV_CHECKPOINT} contains ${n_ckpt} epoch*.ckpt files (need exactly one); set PREV_CHECKPOINT to a .ckpt file." >&2
        exit 1
    fi
elif [[ ! -f "$PREV_CHECKPOINT" ]]; then
    echo "Error: PREV_CHECKPOINT=${PREV_CHECKPOINT} not found. Run 01_finetune_heads.sh first." >&2
    exit 1
fi
if [[ ! -d "$BIGWIG_DIR" ]]; then
    echo "Error: bigwig_dir ${BIGWIG_DIR} not found." >&2
    exit 1
fi
for split in train val test; do
    bed="${INTERVALS_DIR}/Borzoi_human_rep0_${split}_intervals.bed"
    if [[ ! -f "$bed" ]]; then
        echo "Error: ${bed} not found. Run: python workflows/borzoi/export_borzoi_intervals.py" >&2
        exit 1
    fi
done

read -r -a UNFREEZE_ARGS <<< "$UNFREEZE"

cs2f_sbatch borzoi gpu --job-name="$JOB_NAME" \
    --gpus="$N_GPUS" --cpus-per-gpu=8 \
    "${WORKFLOW_DIR}/jobs/finetune.slurm" \
    "$BIGWIG_DIR" ALL "$(cs2f_path node_tmp_dir)" \
    --out_dir "$OUTPUT_DIR" \
    --prefix "$PREFIX" \
    --prev_checkpoint "$PREV_CHECKPOINT" \
    --train_bed "${INTERVALS_DIR}/Borzoi_human_rep0_train_intervals.bed" \
    --val_bed "${INTERVALS_DIR}/Borzoi_human_rep0_val_intervals.bed" \
    --test_bed "${INTERVALS_DIR}/Borzoi_human_rep0_test_intervals.bed" \
    --unfreeze "${UNFREEZE_ARGS[@]}" \
    --lr "$LR" \
    --epochs "$MAX_EPOCHS"
