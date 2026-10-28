#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: Submits the test-set evaluation of the pretrained AlphaGenome track
#          predictions (step 02): streamed against the same BigWigs as the
#          fine-tuned AlphaGenome model, on the AlphaGenome squashed scale (CPU job).
# Usage:   bash workflows/alphagenome_api/03_test_eval.sh
#          PREDICTIONS=/path/to/<intervals>_<config>.h5 bash workflows/alphagenome_api/03_test_eval.sh
# Input:   PREDICTIONS [default: the single .h5 in <model_eval_dir>/AlphaGenome_pretrained],
#          <ag_targets_config> (reference BigWigs + non-zero means), <gtf>
# Output:  <model_eval_dir>/AlphaGenome_pretrained/<predictions file name>/
#          (track_correlations.csv, summary, fig/)
# ==============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/load_site_env.sh"
WORKFLOW_DIR="${CS2F_REPO_ROOT}/workflows/alphagenome_api"

PRED_DIR="${PRED_DIR:-$(cs2f_path model_eval_dir)/AlphaGenome_pretrained}"
TARGETS_CONFIG="${TARGETS_CONFIG:-$(cs2f_path ag_targets_config)}"
OUTPUT_DIR="${OUTPUT_DIR:-$(cs2f_path model_eval_dir)}"
BIN_SIZE="${BIN_SIZE:-1}"

if [[ -z "${PREDICTIONS:-}" ]]; then
    mapfile -t found < <(find "$PRED_DIR" -maxdepth 1 -name '*.h5' 2>/dev/null)
    if [[ ${#found[@]} -ne 1 ]]; then
        echo "Error: ${PRED_DIR} contains ${#found[@]} .h5 files (need exactly one); set PREDICTIONS. Run 02_track_prediction.sh first." >&2
        exit 1
    fi
    PREDICTIONS="${found[0]}"
fi
for required in "$PREDICTIONS" "$TARGETS_CONFIG"; do
    if [[ ! -f "$required" ]]; then
        echo "Error: ${required} not found." >&2
        exit 1
    fi
done

cs2f_sbatch alphagenome_api cpu "${WORKFLOW_DIR}/jobs/test_eval.slurm" \
    --predictions "$PREDICTIONS" \
    --targets_config "$TARGETS_CONFIG" \
    --out_dir "$OUTPUT_DIR" \
    --gtf "$(cs2f_path gtf)" \
    --bin_size "$BIN_SIZE"
