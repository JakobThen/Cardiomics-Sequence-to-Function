#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: Predicts pretrained AlphaGenome ATAC and stranded RNA-seq tracks for
#          the test-fold intervals, averaged per track group, for the evaluation
#          against the cardioid BigWigs.
# Usage:   bash workflows/alphagenome_api/02_track_prediction.sh
#          REFERENCE_TRACKS="CM_and_earlyCM_tracks=Cardiomyocytes" bash workflows/alphagenome_api/02_track_prediction.sh
# Input:   INTERVALS [default: <intervals_dir>/AG_test_intervals_fold1_HOMO_SAPIENS.bed],
#          <ag_track_aggregations> (TRACK_AGGREGATIONS=...),
#          REFERENCE_TRACKS: GROUP=CELL_TYPE pairs; the groups to predict (in order) and
#          the cell type whose BigWigs each group is compared with (the naive heart
#          aggregation is compared with Cardiomyocytes, the major cell type)
# Output:  <model_eval_dir>/AlphaGenome_pretrained/<intervals>_<config>.h5
# ==============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/load_site_env.sh"
WORKFLOW_DIR="${CS2F_REPO_ROOT}/workflows/alphagenome_api"

INTERVALS="${INTERVALS:-$(cs2f_path intervals_dir)/AG_test_intervals_fold1_HOMO_SAPIENS.bed}"
TRACK_AGGREGATIONS="${TRACK_AGGREGATIONS:-$(cs2f_path ag_track_aggregations)}"
OUT_DIR="${OUT_DIR:-$(cs2f_path model_eval_dir)/AlphaGenome_pretrained}"
read -r -a REFERENCE_TRACKS <<< "${REFERENCE_TRACKS:-CM_and_earlyCM_tracks=Cardiomyocytes pluripotent_tracks=Cardiac_progenitors naive_heart_tracks=Cardiomyocytes}"

for required in "$INTERVALS" "$TRACK_AGGREGATIONS"; do
    if [[ ! -f "$required" ]]; then
        echo "Error: ${required} not found." >&2
        exit 1
    fi
done

cs2f_sbatch alphagenome_api bigmem "${WORKFLOW_DIR}/jobs/track_prediction.slurm" \
    --intervals "$INTERVALS" \
    --config "$TRACK_AGGREGATIONS" \
    --reference_tracks "${REFERENCE_TRACKS[@]}" \
    --out_dir "$OUT_DIR"
