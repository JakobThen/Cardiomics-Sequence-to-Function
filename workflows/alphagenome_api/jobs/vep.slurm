#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: SLURM job for step 03. Runs run_vep_pretrained.py with the arguments passed by
#          03_vep.sh.
# Input:   "$@" - arguments for run_vep_pretrained.py
# ==============================================================================

#SBATCH --job-name=AG_VEP
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G

source "${CS2F_REPO_ROOT:?Submit via workflows/alphagenome_api/03_vep.sh}/workflows/common/load_site_env.sh"

cs2f_activate "$CS2F_ENV_ALPHAGENOME"

python "${CS2F_REPO_ROOT}/workflows/alphagenome_api/run_vep_pretrained.py" "$@" \
    --max_agg_workers "$((SLURM_CPUS_PER_TASK - 2))"

conda deactivate
