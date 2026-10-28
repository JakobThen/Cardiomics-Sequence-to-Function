#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: Variant effect prediction with the pretrained AlphaGenome model (API),
#          aggregated per track group. Interrupted runs resume from the last
#          finished batch when started again with the same input.
# Usage:   bash workflows/alphagenome_api/04_vep.sh
#          INPUT_VCF=/path/to/variants.csv MODALITIES="atac chip_tf" bash workflows/alphagenome_api/04_vep.sh
# Input:   INPUT_VCF (CSV/TSV with CHROM, POS, REF, ALT) [default: <vep_input_vcf>],
#          <ag_track_aggregations> (TRACK_AGGREGATIONS=...)
# Output:  <vep_dir>/AG_VEP_out/<vcf name>_<group>_aggr.csv
# ==============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/load_site_env.sh"
WORKFLOW_DIR="${CS2F_REPO_ROOT}/workflows/alphagenome_api"

INPUT_VCF="${INPUT_VCF:-$(cs2f_path vep_input_vcf)}"
TRACK_AGGREGATIONS="${TRACK_AGGREGATIONS:-$(cs2f_path ag_track_aggregations)}"
OUTPUT_DIR="${OUTPUT_DIR:-$(cs2f_path vep_dir)/AG_VEP_out}"
INTERVAL="${INTERVAL:-1MB}"            # 1MB, 500KB, 100KB or 16KB
API_WORKERS="${API_WORKERS:-5}"
ORGANISM="${ORGANISM:-HOMO_SAPIENS}"
read -r -a MODALITIES <<< "${MODALITIES:-atac rna_seq chip_histone chip_tf}"

for required in "$INPUT_VCF" "$TRACK_AGGREGATIONS"; do
    if [[ ! -f "$required" ]]; then
        echo "Error: ${required} not found." >&2
        exit 1
    fi
done

cs2f_sbatch alphagenome_api bigmem "${WORKFLOW_DIR}/jobs/vep.slurm" \
    --input_file "$INPUT_VCF" \
    --output_dir "$OUTPUT_DIR" \
    --interval_size "$INTERVAL" \
    --organism "$ORGANISM" \
    --max_api_workers "$API_WORKERS" \
    --modalities "${MODALITIES[@]}" \
    --config "$TRACK_AGGREGATIONS"
