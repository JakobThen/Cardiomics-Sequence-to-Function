#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: SLURM job for steps 01 and 02. Stages BigWig folders to the node-local
#          disk and runs finetune_borzoi.py on all GPUs of the allocation.
#          The launchers set the GPU count (step 01: 8, step 02: 4).
# Input:   $1 - bigwig_dir (source of the staged folders)
#          $2 - Folders to stage, comma-separated, or ALL (whole bigwig_dir)
#          $3 - Node-local scratch root (a per-job subfolder is created)
#          "${@:4}" - further arguments for finetune_borzoi.py (without --data_dir)
# ==============================================================================

#SBATCH --job-name=Borzoi_finetune
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=185G
#SBATCH --time=5-00:00:00

source "${CS2F_REPO_ROOT:?Submit via workflows/borzoi/01_finetune_heads.sh or 02_finetune_decoder.sh}/workflows/common/load_site_env.sh"

INPUT_DATA_DIR="$1"
STAGE_FOLDERS="$2"
LOCAL_TMP_DIR="$3/borzoi_bws_${SLURM_JOB_ID}"
shift 3

echo "======================================================"
echo "Starting Borzoi Finetuning Job"
echo "======================================================"
echo "Date:           $(date)"
echo "User:           $(whoami)"
echo "Job ID:         ${SLURM_JOB_ID}"
echo "Execution Node: ${SLURMD_NODENAME}"
echo "======================================================"
echo ""

# ALWAYS clean up the local tmp directory on exit, success or failure
trap 'echo "Cleaning up local tmp directory..."; rm -rf "${LOCAL_TMP_DIR}"; echo "Cleanup done."' EXIT

# I/O OPTIMIZATION: node-local data staging
echo "Staging data from ${INPUT_DATA_DIR} to ${SLURMD_NODENAME}:${LOCAL_TMP_DIR}..."
mkdir -p "$LOCAL_TMP_DIR"
if [[ "$STAGE_FOLDERS" == "ALL" ]]; then
    # trailing slash: copy the contents, not the folder itself
    rsync -a --info=progress2 "${INPUT_DATA_DIR}/" "${LOCAL_TMP_DIR}/"
else
    IFS=',' read -r -a FOLDERS <<< "$STAGE_FOLDERS"
    RSYNC_SOURCES=()
    for folder in "${FOLDERS[@]}"; do
        RSYNC_SOURCES+=("${INPUT_DATA_DIR}/${folder}")
    done
    rsync -a --info=progress2 "${RSYNC_SOURCES[@]}" "${LOCAL_TMP_DIR}/"
fi
echo "Data staging complete!"

cs2f_activate "$CS2F_ENV_BORZOI"

# Let PyTorch memory segments grow, reducing fragmentation crashes
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Confirm the CUDA setup before training
python "${CS2F_REPO_ROOT}/workflows/common/check_torch_cuda.py"

echo "Starting model finetuning..."
srun python "${CS2F_REPO_ROOT}/workflows/borzoi/finetune_borzoi.py" \
    --data_dir "$LOCAL_TMP_DIR" \
    "$@"

conda deactivate

echo ""
echo "======================================================"
echo "Training done! Ending job on: $(date)"
echo "======================================================"
