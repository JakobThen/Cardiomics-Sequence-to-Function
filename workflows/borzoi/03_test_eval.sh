#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: Submits the test-set evaluation of a fine-tuned Borzoi model: the test
#          predictions saved by step 01/02 are streamed against the BigWig tracks
#          (CPU job).
# Usage:   bash workflows/borzoi/03_test_eval.sh
#          PREDICTIONS=/path/to/<prefix>_squashed_predictions.h5 bash workflows/borzoi/03_test_eval.sh
# Input:   PREDICTIONS [default: the single *_predictions.h5 in <borzoi_checkpoint_dir>],
#          <bigwig_dir> (BigWigs the model was trained on), <gtf>
#          RNA_SCALE: squashed (default) or linear
# Output:  <model_eval_dir>/Borzoi/<predictions file name>/ (track_correlations.csv, summary, fig/)
# ==============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/load_site_env.sh"
WORKFLOW_DIR="${CS2F_REPO_ROOT}/workflows/borzoi"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$(cs2f_path borzoi_checkpoint_dir)}"
BIGWIG_DIR="${BIGWIG_DIR:-$(cs2f_path bigwig_dir)}"
OUTPUT_DIR="${OUTPUT_DIR:-$(cs2f_path model_eval_dir)}"
RNA_SCALE="${RNA_SCALE:-squashed}"
BIN_SIZE="${BIN_SIZE:-32}"

if [[ -z "${PREDICTIONS:-}" ]]; then
    if [[ ! -d "$CHECKPOINT_DIR" ]]; then
        echo "Error: set PREDICTIONS; ${CHECKPOINT_DIR} is not a folder." >&2
        exit 1
    fi
    mapfile -t found < <(find "$CHECKPOINT_DIR" -maxdepth 1 -name '*_predictions.h5')
    if [[ ${#found[@]} -ne 1 ]]; then
        echo "Error: ${CHECKPOINT_DIR} contains ${#found[@]} *_predictions.h5 files (need exactly one); set PREDICTIONS." >&2
        exit 1
    fi
    PREDICTIONS="${found[0]}"
fi
if [[ ! -f "$PREDICTIONS" ]]; then
    echo "Error: PREDICTIONS=${PREDICTIONS} not found." >&2
    exit 1
fi

cs2f_sbatch borzoi cpu "${WORKFLOW_DIR}/jobs/test_eval.slurm" \
    --predictions "$PREDICTIONS" \
    --bigwig_dir "$BIGWIG_DIR" \
    --out_dir "$OUTPUT_DIR" \
    --gtf "$(cs2f_path gtf)" \
    --bin_size "$BIN_SIZE" \
    --rna_scale "$RNA_SCALE"
