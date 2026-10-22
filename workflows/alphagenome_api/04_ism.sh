#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: In silico mutagenesis with the pretrained AlphaGenome model (API) for
#          the regions of a BED file: ATAC, plus ChIP-TF (TFS) and RNA-seq (GENES).
# Usage:   BED=regions.bed CURIES="UBERON:0002078 UBERON:0002079" bash workflows/alphagenome_api/04_ism.sh
#          BED=regions.bed CURIES="..." TFS="CTCF GATA4" GENES=genes.csv bash workflows/alphagenome_api/04_ism.sh
# Input:   BED (no header; chrom, start, end[, name]), CURIES (ontology CURIEs to average),
#          optional TFS (names or one .txt file) and GENES (CSV, one row per region)
# Output:  <ism_dir>/AG_pretrained_ISM_<bed name>.h5 (OUT_PREFIX=...)
# ==============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/load_site_env.sh"
WORKFLOW_DIR="${CS2F_REPO_ROOT}/workflows/alphagenome_api"

BED="${BED:?Set BED=<regions.bed>}"
CURIES="${CURIES:?Set CURIES to one or more ontology CURIEs}"
read -r -a CURIE_LIST <<< "$CURIES"
SEQ_LEN="${SEQ_LEN:-1MB}"              # 1MB, 500KB, 100KB or 16KB
OUT_PREFIX="${OUT_PREFIX:-$(cs2f_path ism_dir)/AG_pretrained_ISM}"

for required in "$BED"; do
    if [[ ! -f "$required" ]]; then
        echo "Error: ${required} not found." >&2
        exit 1
    fi
done

ARGS=(--bed "$BED" --curies "${CURIE_LIST[@]}" --seq_len "$SEQ_LEN" --out_prefix "$OUT_PREFIX")
if [[ -n "${TFS:-}" ]]; then
    read -r -a TF_LIST <<< "$TFS"
    ARGS+=(--tfs "${TF_LIST[@]}")
fi
if [[ -n "${GENES:-}" ]]; then
    ARGS+=(--genes "$GENES")
fi

cs2f_sbatch alphagenome_api cpu "${WORKFLOW_DIR}/jobs/ism.slurm" "${ARGS[@]}"
