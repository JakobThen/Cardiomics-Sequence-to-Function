# Input-prep configs

| File | Tracked | Content |
|---|---|---|
| `nCells.csv` | yes | Cells per pseudobulk (`"n_cells","celltype"`), used for the `N_CELLS` filter (default 400). Currently the **simple** (coarse) cell type definition with 7 cell types. |
| `cutntag_bams.txt` | no (cluster paths) | Curated Cut&Tag BAM list (batches 1 and 2), the default `cutntag_bam_input`. |

## Cell type definitions

The project switched between two pseudobulk definitions:

- **simple** (coarse): 7 cell types (Cardiac_mesoderm, Cardiac_progenitors,
  Cardiomyocytes, Ectoderm, Endoderm, Endothelial_cells, Fibroblasts).
- **normal** (finer): 13 cell types, 11 of them with ≥400 cells (adds
  Early_cardiomyocytes, Early_endoderm, Early_fibroblasts, Mesendoderm; drops
  Early_ectoderm and SOX2_Ectoderm). The final fine-tuned AlphaGenome model was
  trained on this definition (deduplicated reads; BigWigs in
  `seq2fun_train_input_baseres/multiome_{ATAC,RNA_3prime}_bigwig`).

Both write BigWigs with the same file names for shared cell types (e.g.
`ATAC_Cardiomyocytes_100M_norm.bw`), so write each definition to its own output
folder (`BW_OUTPUT_DIR=...`) to avoid overwriting.
