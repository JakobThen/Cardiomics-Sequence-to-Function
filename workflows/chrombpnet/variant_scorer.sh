#!/bin/bash
# ==============================================================================
# Pinned version of the Kundaje lab variant-scorer (source, don't run).
#
# variant-scorer is not a pip package (its scripts are run as
# `python src/variant_scoring.py`), so it is installed as a pinned git checkout
# with install_variant_scorer.sh instead of being copied into this repository.
#
# Provides:
#   VARIANT_SCORER_REPO, VARIANT_SCORER_COMMIT
#   VARIANT_SCORER_DIR          $CS2F_VARIANT_SCORER_DIR, default <repo>/external/variant-scorer
#   cs2f_check_variant_scorer   fails unless the checkout exists at the pinned commit
# ==============================================================================

VARIANT_SCORER_REPO="https://github.com/kundajelab/variant-scorer.git"
# Merge of PR #29 (2025-07-29): the version used for the project's ChromBPNet VEP
VARIANT_SCORER_COMMIT="0e1e34199e63112aa618748bb79a206fc491300a"
VARIANT_SCORER_DIR="${CS2F_VARIANT_SCORER_DIR:-${CS2F_REPO_ROOT}/external/variant-scorer}"

cs2f_check_variant_scorer() {
    if [[ ! -f "${VARIANT_SCORER_DIR}/src/variant_scoring.py" ]]; then
        echo "Error: variant-scorer not found in ${VARIANT_SCORER_DIR}. Run: bash workflows/chrombpnet/install_variant_scorer.sh" >&2
        return 1
    fi
    local head
    head="$(git -C "$VARIANT_SCORER_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
    if [[ "$head" != "$VARIANT_SCORER_COMMIT" ]]; then
        echo "Error: ${VARIANT_SCORER_DIR} is at ${head}, expected ${VARIANT_SCORER_COMMIT}. Run install_variant_scorer.sh." >&2
        return 1
    fi
    if [[ -n "$(git -C "$VARIANT_SCORER_DIR" status --porcelain --untracked-files=no 2>/dev/null)" ]]; then
        echo "Warning: ${VARIANT_SCORER_DIR} has local modifications." >&2
    fi
}
