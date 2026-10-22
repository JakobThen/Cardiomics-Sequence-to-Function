#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: Matches the pretrained AlphaGenome RNA-seq and ATAC tracks to the
#          multiome data (one bigmem job per modality). The track-matching notebook
#          analyses the results and derives the track aggregations (CURIE lists).
# Usage:   bash workflows/alphagenome_api/01_track_matching.sh
#          MODALITIES=atac BW_DIR=/path/to/ATAC_bigwigs bash workflows/alphagenome_api/01_track_matching.sh
# Input:   RNA:  <multiome_gene_counts_csv>, <multiome_gene_pct_csv>, <gtf>, <chrom_sizes>
#          ATAC: <abc_enhancer_table> (REGIONS=...), BigWigs in
#                <bigwig_dir>/multiome_ATAC_bigwig_deduplicated (BW_DIR=...)
# Output:  <track_matching_dir>/RNA/*.h5ad, <track_matching_dir>/ATAC/*.parquet
# ==============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/load_site_env.sh"
WORKFLOW_DIR="${CS2F_REPO_ROOT}/workflows/alphagenome_api"

read -r -a MODALITIES <<< "${MODALITIES:-rna atac}"
OUT_DIR="${OUT_DIR:-$(cs2f_path track_matching_dir)}"
SEQ_LEN="${SEQ_LEN:-500KB}"

for modality in "${MODALITIES[@]}"; do
    case "$modality" in
        rna)
            COUNTS_CSV="${COUNTS_CSV:-$(cs2f_path multiome_gene_counts_csv)}"
            PCT_CSV="${PCT_CSV:-$(cs2f_path multiome_gene_pct_csv)}"
            GTF="$(cs2f_path gtf)"
            CHROM_SIZES="$(cs2f_path chrom_sizes)"
            for required in "$COUNTS_CSV" "$PCT_CSV" "$GTF" "$CHROM_SIZES"; do
                if [[ ! -f "$required" ]]; then
                    echo "Error: ${required} not found." >&2
                    exit 1
                fi
            done
            cs2f_sbatch alphagenome_api bigmem "${WORKFLOW_DIR}/jobs/match_tracks_rna.slurm" \
                --counts_csv "$COUNTS_CSV" \
                --pct_csv "$PCT_CSV" \
                --gtf "$GTF" \
                --chrom_sizes "$CHROM_SIZES" \
                --out_dir "${OUT_DIR}/RNA" \
                --seq_len "$SEQ_LEN" \
                --n_top_var_genes "${N_TOP_VAR_GENES:-5000}"
            ;;
        atac)
            REGIONS="${REGIONS:-$(cs2f_path abc_enhancer_table)}"
            BW_DIR="${BW_DIR:-$(cs2f_path bigwig_dir)/multiome_ATAC_bigwig_deduplicated}"
            if [[ ! -f "$REGIONS" || ! -d "$BW_DIR" ]]; then
                echo "Error: ${REGIONS} or ${BW_DIR} not found." >&2
                exit 1
            fi
            cs2f_sbatch alphagenome_api bigmem "${WORKFLOW_DIR}/jobs/match_tracks_atac.slurm" \
                --regions "$REGIONS" \
                --bw_dir "$BW_DIR" \
                --out_dir "${OUT_DIR}/ATAC" \
                --seq_len "$SEQ_LEN"
            ;;
        *)
            echo "Error: unknown modality '${modality}' (use rna and/or atac)." >&2
            exit 1
            ;;
    esac
done
