#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: SLURM job for step 01 (RNA). Runs match_tracks_rna.py with the arguments passed by
#          01_track_matching.sh.
# Input:   "$@" - arguments for match_tracks_rna.py
# ==============================================================================

#SBATCH --job-name=AG_RNA_match
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=1024G

source "${CS2F_REPO_ROOT:?Submit via workflows/alphagenome_api/01_track_matching.sh}/workflows/common/load_site_env.sh"

cs2f_activate "$CS2F_ENV_ALPHAGENOME"

python "${CS2F_REPO_ROOT}/workflows/alphagenome_api/match_tracks_rna.py" "$@"

conda deactivate
