#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: Installs the Kundaje lab variant-scorer as a git checkout pinned to the
#          commit used in this project (see variant_scorer.sh). Run once on the
#          login node; safe to re-run (fetches and checks out the pinned commit).
# Usage:   bash workflows/chrombpnet/install_variant_scorer.sh
#          CS2F_VARIANT_SCORER_DIR=/path/to/variant-scorer bash workflows/chrombpnet/install_variant_scorer.sh
# Output:  $CS2F_VARIANT_SCORER_DIR (default: <repo>/external/variant-scorer, gitignored)
# Note:    Its Python dependencies (TensorFlow, deepdish, kundajelab-shap, pyfaidx, ...)
#          come with chrombpnet in CS2F_ENV_CHROMBPNET (pip install -e ".[chrombpnet]").
# ==============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/load_site_env.sh"
source "${CS2F_REPO_ROOT}/workflows/chrombpnet/variant_scorer.sh"

if [[ ! -d "${VARIANT_SCORER_DIR}/.git" ]]; then
    echo "Cloning ${VARIANT_SCORER_REPO} into ${VARIANT_SCORER_DIR}..."
    mkdir -p "$(dirname "$VARIANT_SCORER_DIR")"
    git clone "$VARIANT_SCORER_REPO" "$VARIANT_SCORER_DIR"
else
    git -C "$VARIANT_SCORER_DIR" fetch origin
fi
git -C "$VARIANT_SCORER_DIR" checkout --detach "$VARIANT_SCORER_COMMIT"

cs2f_check_variant_scorer
echo "variant-scorer ready at ${VARIANT_SCORER_DIR} (commit ${VARIANT_SCORER_COMMIT})."
