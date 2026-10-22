#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: SLURM job for step 02. Runs predict_tracks.py with the arguments passed by
#          02_track_prediction.sh.
# Input:   "$@" - arguments for predict_tracks.py
# ==============================================================================

#SBATCH --job-name=AG_pretrained_tracks
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=2048G

source "${CS2F_REPO_ROOT:?Submit via workflows/alphagenome_api/02_track_prediction.sh}/workflows/common/load_site_env.sh"

cs2f_activate "$CS2F_ENV_ALPHAGENOME"

python "${CS2F_REPO_ROOT}/workflows/alphagenome_api/predict_tracks.py" "$@"

conda deactivate
