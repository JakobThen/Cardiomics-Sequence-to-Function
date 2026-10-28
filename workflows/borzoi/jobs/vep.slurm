#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: SLURM job for step 04. Runs run_vep_borzoi.py on one GPU.
# Input:   "$@" - arguments for run_vep_borzoi.py
# ==============================================================================

#SBATCH --job-name=Borzoi_VEP
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=256G
#SBATCH --time=2-00:00:00

source "${CS2F_REPO_ROOT:?Submit via workflows/borzoi/04_vep.sh}/workflows/common/load_site_env.sh"

echo "======================================================"
echo "Starting Borzoi VEP"
echo "======================================================"
echo "Date:           $(date)"
echo "User:           $(whoami)"
echo "Job ID:         ${SLURM_JOB_ID}"
echo "Execution Node: ${SLURMD_NODENAME}"
echo "======================================================"
echo ""

cs2f_activate "$CS2F_ENV_BORZOI"

# Let PyTorch memory segments grow, reducing fragmentation crashes
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python "${CS2F_REPO_ROOT}/workflows/common/check_torch_cuda.py"

srun python "${CS2F_REPO_ROOT}/workflows/borzoi/run_vep_borzoi.py" "$@"

conda deactivate
echo "VEP done!"
