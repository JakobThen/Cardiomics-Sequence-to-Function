# Input-prep configs

Your data-specific files in this folder are **gitignored**; the tracked
`*.example.*` files show the expected format.

| File | Example | Content |
|---|---|---|
| `nCells.csv` | `nCells.example.csv` | Cells per pseudobulk (`"n_cells","celltype"`, cell type = BAM file name without `.bam`). Used by `01_atac_bigwig.sh` / `02_rna_bigwig.sh` to skip pseudobulks below `N_CELLS` (default 400). Without the file, all pseudobulks are kept. Override with `CELL_NUMBER_FILE=...`. |
| `cutntag_bams.txt` | `cutntag_bams.example.txt` | One Cut&Tag BAM path per line; usable as `cutntag_bam_input` in `config/paths.yaml` (or a folder of BAMs instead). |

If you work with several cell type definitions that share names, write each to
its own BigWig folder (`BW_OUTPUT_DIR=...`) so files are not overwritten.
