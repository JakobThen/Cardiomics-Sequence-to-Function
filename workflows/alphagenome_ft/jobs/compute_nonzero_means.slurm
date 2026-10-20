#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: SLURM job for step 00. Runs compute_nonzero_means.py with the
#          arguments passed by 00_compute_nonzero_means.sh.
# Input:   "$@" - arguments for compute_nonzero_means.py
# ==============================================================================

#SBATCH --job-name=bw_mean_calc
#SBATCH --time=04:00:00          # Adjust depending on the total number of BigWig files
#SBATCH --cpus-per-task=1        # Sequential processing per file
#SBATCH --mem=64G                # Allocated generously to load large chromosome arrays into memory safely

source "${CS2F_REPO_ROOT:?Submit via workflows/alphagenome_ft/00_compute_nonzero_means.sh}/workflows/common/load_site_env.sh"

echo "Starting job on $(date)"

cs2f_activate "$CS2F_ENV_ALPHAGENOME_FT"

python "${CS2F_REPO_ROOT}/workflows/alphagenome_ft/compute_nonzero_means.py" "$@"

conda deactivate
echo "Job finished on $(date)."
