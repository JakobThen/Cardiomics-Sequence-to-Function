# chrombpnet configs

| File | Tracked | Used by |
|---|---|---|
| `splits/fold_{0..4}.json` | yes | step 02: chromosome splits (`test`, `valid`, `train`) of the five ChromBPNet folds |

The splits are the standard hg38 five-fold splits used with ChromBPNet (e.g. fold 0: test
chr1, chr3, chr6; validation chr8, chr20). They can be regenerated or changed with
`chrombpnet prep splits -c <chrom sizes> -tcr <test chroms> -vcr <valid chroms> -op <prefix>`;
models of different folds are only comparable if their splits are kept.

The cell-count filter of step 02 reads `workflows/input_prep/configs/nCells.csv`
(override with `CELL_NUMBER_FILE=...`; empty to disable).
