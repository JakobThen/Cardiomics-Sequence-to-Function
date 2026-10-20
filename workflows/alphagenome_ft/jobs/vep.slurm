#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: SLURM job for step 04. Runs run_vep.py on 8 GPUs.
# Input:   $1 - XLA autotune cache directory
#          "${@:2}" - arguments for run_vep.py
# ==============================================================================

#SBATCH --job-name=AG_ft_VEP
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=1-00:00:00

source "${CS2F_REPO_ROOT:?Submit via workflows/alphagenome_ft/04_vep.sh}/workflows/common/load_site_env.sh"

XLA_CACHE_DIR="$1"
shift

echo "======================================================"
echo "Starting AlphaGenome Finetuned VEP"
echo "======================================================"
echo "Date:           $(date)"
echo "User:           $(whoami)"
echo "Job ID:         ${SLURM_JOB_ID}"
echo "Execution Node: ${SLURMD_NODENAME}"
echo "======================================================"
echo ""

cs2f_activate "$CS2F_ENV_ALPHAGENOME_FT"

# Set cache for XLA autotuning (caches benchmarks of the fastest computations on
# this GPU type, enabling faster start-up when reused)
mkdir -p "$XLA_CACHE_DIR"
export XLA_FLAGS="--xla_gpu_per_fusion_autotune_cache_dir=${XLA_CACHE_DIR}"

srun python "${CS2F_REPO_ROOT}/workflows/alphagenome_ft/run_vep.py" "$@"

conda deactivate
echo "VEP done!"
