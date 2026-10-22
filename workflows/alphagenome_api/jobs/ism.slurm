#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: SLURM job for step 04. Runs run_ism.py with the arguments passed by
#          04_ism.sh.
# Input:   "$@" - arguments for run_ism.py
# ==============================================================================

#SBATCH --job-name=AG_ISM
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

source "${CS2F_REPO_ROOT:?Submit via workflows/alphagenome_api/04_ism.sh}/workflows/common/load_site_env.sh"

cs2f_activate "$CS2F_ENV_ALPHAGENOME"

python "${CS2F_REPO_ROOT}/workflows/alphagenome_api/run_ism.py" "$@"

conda deactivate
