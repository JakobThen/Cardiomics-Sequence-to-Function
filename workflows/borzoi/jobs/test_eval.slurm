#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: SLURM job for step 03. Runs evaluate_on_test_borzoi.py (CPU only;
#          predictions are streamed from the HDF5 in batches).
# Input:   "$@" - arguments for evaluate_on_test_borzoi.py
# ==============================================================================

#SBATCH --job-name=Borzoi_test_eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00

source "${CS2F_REPO_ROOT:?Submit via workflows/borzoi/03_test_eval.sh}/workflows/common/load_site_env.sh"

echo "======================================================"
echo "Starting Borzoi Test Evaluation"
echo "======================================================"
echo "Date:           $(date)"
echo "User:           $(whoami)"
echo "Job ID:         ${SLURM_JOB_ID}"
echo "Execution Node: ${SLURMD_NODENAME}"
echo "======================================================"
echo ""

cs2f_activate "$CS2F_ENV_BORZOI"

python "${CS2F_REPO_ROOT}/workflows/borzoi/evaluate_on_test_borzoi.py" "$@"

conda deactivate
echo "Test evaluation done!"
