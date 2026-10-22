# Analysis notebooks

Notebooks for the downstream analyses. Reusable code lives in the
`cardiomics_s2f` package; the notebooks import it and only hold the analysis
steps and figures.

## Setup

```bash
pip install -e ".[analysis]"          # notebook dependencies
pip install -e ".[alphagenome]"       # additionally for track_matching/ (AlphaGenome API)
```

- **Paths** are read from `config/paths.yaml` with `cardiomics_s2f.config.load_paths()`;
  each notebook lists the keys it needs at the top.
- **API key** (track matching only): set `ALPHAGENOME_API_KEY` or put it in `.env`.
- **Saving** of figures and data files is commented out in all notebooks; uncomment
  the `savefig` / `to_csv` lines to write them.
- **Outputs** are stripped before each commit by the `nbstripout` pre-commit hook.

## Notebooks

### `variant_datasets/`

| Notebook | What it does |
|---|---|
| `richter_variant_dataset.ipynb` | Builds the annotated de novo variant tables of CHD probands and controls (Richter et al. 2020): exclusion of probands with causal coding variants, annotation with ENCODE4 cCREs, GTF features, phyloP, coding effects, CADD, cardioid ATAC peaks, ABC enhancers and the SDR007 selection, strand alignment to the reference. Writes `VCF_all_Richter_vars.csv`, `VCF_new_enhancer_Richter_vars.csv` and the CRISPRi screen VCF (±5 bp around enhancer SNVs). |
| `richter_cohort_overview.ipynb` | Descriptive figures comparing CHD and control variants (burden per proband, annotation, conservation, cCREs, ATAC peak and enhancer overlap, QC panels). Reads `VCF_all_Richter_vars.csv`. |

### `track_matching/`

| Notebook | What it does |
|---|---|
| `alphagenome_track_matching.ipynb` | Compares the pretrained AlphaGenome RNA and ATAC tracks with the cardioid multiome tracks (outputs of the track-matching scripts) and selects the best-matching AlphaGenome tracks per cell type; the resulting CURIE lists define the track aggregations for pretrained-AlphaGenome VEP. |

### `attribution/`

| Notebook | What it does |
|---|---|
| `alphagenome_ft_deepshap_amplicons.ipynb` | Analyses DeepSHAP attributions of the fine-tuned AlphaGenome ATAC head at the SDR amplicons 39–42: cell-type-specific attributions (CM, FB, ED), normalisation by local activity, TF-MoDISco seqlets with HOCOMOCO motif annotation, and attribution logos around the SDR variants next to their measured effects. |
