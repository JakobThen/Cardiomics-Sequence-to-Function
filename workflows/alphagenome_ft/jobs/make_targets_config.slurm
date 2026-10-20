#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: SLURM job for step 01. Runs make_targets_config.py with the
#          arguments passed by 01_make_targets_config.sh.
# Input:   "$@" - arguments for make_targets_config.py
# ==============================================================================

#SBATCH --job-name=make_config
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

source "${CS2F_REPO_ROOT:?Submit via workflows/alphagenome_ft/01_make_targets_config.sh}/workflows/common/load_site_env.sh"

cs2f_activate "$CS2F_ENV_ALPHAGENOME_FT"

python "${CS2F_REPO_ROOT}/workflows/alphagenome_ft/make_targets_config.py" "$@"

conda deactivate
