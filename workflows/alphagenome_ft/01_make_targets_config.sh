#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: Builds the AlphaGenome targets config (one head per modality, listing
#          its BigWig tracks and their non-zero means from step 00).
# Usage:   TARGETS_CONFIG=workflows/alphagenome_ft/configs/my_run.yaml bash workflows/alphagenome_ft/01_make_targets_config.sh
#          (refuses to overwrite an existing config unless OVERWRITE=1)
# Input:   Modality folders in <bigwig_dir>, <nonzero_means_csv>
# Output:  <ag_targets_config> (override: TARGETS_CONFIG=...)
# ==============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/load_site_env.sh"
WORKFLOW_DIR="${CS2F_REPO_ROOT}/workflows/alphagenome_ft"

# Setup core paths and variables
DATA_DIR="${DATA_DIR:-$(cs2f_path bigwig_dir)}"
NONZERO_MEANS_PATH="${NONZERO_MEANS_PATH:-$(cs2f_path nonzero_means_csv)}"
TARGETS_CONFIG="${TARGETS_CONFIG:-$(cs2f_path ag_targets_config)}"
NORM_TAG="${NORM_TAG:-_100M}"

# Folders in DATA_DIR (space-separated override, e.g. to rebuild the final model's config:
# MODALITIES="CutnTag_bigwig multiome_ATAC_bigwig multiome_RNA_3prime_bigwig")
read -r -a MODALITIES <<< "${MODALITIES:-multiome_RNA_3prime_bigwig_deduplicated multiome_ATAC_bigwig_deduplicated CutnTag_bigwig}"

if [[ ! -f "$NONZERO_MEANS_PATH" ]]; then
    echo "Error: ${NONZERO_MEANS_PATH} not found. Run 00_compute_nonzero_means.sh first." >&2
    exit 1
fi
# Never silently replace an existing config (it may be the config of record of a trained model)
if [[ -f "$TARGETS_CONFIG" && "${OVERWRITE:-0}" != "1" ]]; then
    echo "Error: ${TARGETS_CONFIG} already exists. Set TARGETS_CONFIG=<new file> or OVERWRITE=1." >&2
    exit 1
fi

cs2f_sbatch alphagenome_ft cpu "${WORKFLOW_DIR}/jobs/make_targets_config.slurm" \
    --data_dir "$DATA_DIR" \
    --config_prefix "$TARGETS_CONFIG" \
    --modality_folders "${MODALITIES[@]}" \
    --norm_tag "$NORM_TAG" \
    --nonzero_means "$NONZERO_MEANS_PATH"
