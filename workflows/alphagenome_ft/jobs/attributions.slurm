#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: SLURM job for step 05. Runs compute_attributions.py on 1 GPU.
# Input:   $1 - XLA autotune cache directory
#          "${@:2}" - arguments for compute_attributions.py
# ==============================================================================

#SBATCH --job-name=AG_ft_attributions
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=1-00:00:00

source "${CS2F_REPO_ROOT:?Submit via workflows/alphagenome_ft/05_attributions.sh}/workflows/common/load_site_env.sh"

XLA_CACHE_DIR="$1"
shift

cs2f_activate "$CS2F_ENV_ALPHAGENOME_FT"

mkdir -p "$XLA_CACHE_DIR"
export XLA_FLAGS="--xla_gpu_per_fusion_autotune_cache_dir=${XLA_CACHE_DIR}"

python "${CS2F_REPO_ROOT}/workflows/alphagenome_ft/compute_attributions.py" "$@"

conda deactivate
echo "Attributions done!"
