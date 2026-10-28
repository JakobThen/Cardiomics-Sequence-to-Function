#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: Submits Borzoi stage 1: new output heads on the pretrained human_rep0
#          trunk (embedding frozen), 8 GPUs. The BigWig folders listed in
#          configs/bigwig_folders.txt are staged to the GPU node's local disk.
# Usage:   bash workflows/borzoi/01_finetune_heads.sh
#          FOLDERS_FILE=/path/to/folders.txt LR=1e-4 bash workflows/borzoi/01_finetune_heads.sh
# Input:   <bigwig_dir>/<folders in FOLDERS_FILE>,
#          <intervals_dir>/Borzoi_human_rep0_{train,val,test}_intervals.bed
#          (written by export_borzoi_intervals.py)
# Output:  <borzoi_heads_dir>/<run>/version_0/checkpoints/ (best epoch*.ckpt + last.ckpt),
#          <borzoi_heads_dir>/<PREFIX>_squashed_predictions.h5 (test intervals)
# ==============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/load_site_env.sh"
WORKFLOW_DIR="${CS2F_REPO_ROOT}/workflows/borzoi"

BIGWIG_DIR="${BIGWIG_DIR:-$(cs2f_path bigwig_dir)}"
INTERVALS_DIR="${INTERVALS_DIR:-$(cs2f_path intervals_dir)}"
FOLDERS_FILE="${FOLDERS_FILE:-${WORKFLOW_DIR}/configs/bigwig_folders.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-$(cs2f_path borzoi_heads_dir)}"
PREFIX="${PREFIX:-Borzoi_rep0_heads}"
N_GPUS="${N_GPUS:-8}"
LR="${LR:-3e-4}"
MAX_EPOCHS="${MAX_EPOCHS:-15}"

if [[ ! -f "$FOLDERS_FILE" ]]; then
    echo "Error: ${FOLDERS_FILE} not found. Copy configs/bigwig_folders.example.txt to configs/bigwig_folders.txt and adapt it." >&2
    exit 1
fi
FOLDERS=()
while IFS= read -r line; do
    line="${line%%#*}"
    line="$(echo "$line" | xargs)"   # trim whitespace
    [[ -z "$line" ]] && continue
    if [[ ! -d "${BIGWIG_DIR}/${line}" ]]; then
        echo "Error: BigWig folder ${BIGWIG_DIR}/${line} (from ${FOLDERS_FILE}) not found." >&2
        exit 1
    fi
    FOLDERS+=("$line")
done < "$FOLDERS_FILE"
if [[ ${#FOLDERS[@]} -eq 0 ]]; then
    echo "Error: no folders listed in ${FOLDERS_FILE}." >&2
    exit 1
fi

for split in train val test; do
    bed="${INTERVALS_DIR}/Borzoi_human_rep0_${split}_intervals.bed"
    if [[ ! -f "$bed" ]]; then
        echo "Error: ${bed} not found. Run: python workflows/borzoi/export_borzoi_intervals.py" >&2
        exit 1
    fi
done

STAGE_FOLDERS="$(IFS=','; echo "${FOLDERS[*]}")"
echo "Staging folders: ${STAGE_FOLDERS}"

cs2f_sbatch borzoi gpu --job-name="Borzoi_finetune_heads" \
    --gpus="$N_GPUS" --cpus-per-gpu=4 \
    "${WORKFLOW_DIR}/jobs/finetune.slurm" \
    "$BIGWIG_DIR" "$STAGE_FOLDERS" "$(cs2f_path node_tmp_dir)" \
    --out_dir "$OUTPUT_DIR" \
    --prefix "$PREFIX" \
    --train_bed "${INTERVALS_DIR}/Borzoi_human_rep0_train_intervals.bed" \
    --val_bed "${INTERVALS_DIR}/Borzoi_human_rep0_val_intervals.bed" \
    --test_bed "${INTERVALS_DIR}/Borzoi_human_rep0_test_intervals.bed" \
    --lr "$LR" \
    --epochs "$MAX_EPOCHS"
