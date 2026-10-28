#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: SLURM job for step 03. Runs evaluate_on_test_pretrained.py with the
#          arguments passed by 03_test_eval.sh (CPU only; predictions are
#          streamed from the HDF5 in batches).
# Input:   "$@" - arguments for evaluate_on_test_pretrained.py
# ==============================================================================

#SBATCH --job-name=AG_pretrained_test_eval
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

source "${CS2F_REPO_ROOT:?Submit via workflows/alphagenome_api/03_test_eval.sh}/workflows/common/load_site_env.sh"

cs2f_activate "$CS2F_ENV_ALPHAGENOME"

python "${CS2F_REPO_ROOT}/workflows/alphagenome_api/evaluate_on_test_pretrained.py" "$@"

conda deactivate
