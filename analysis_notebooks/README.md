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
| `sdr_variant_dataset.ipynb` | Builds the SDR variant tables from the SDR007AB and SDR010ABC differential expression results (positive: adjusted p ≤ 0.05; negative: adjusted p = 1, ≥ 100 cells, \|log2FC\| ≤ 0.1; MAST and Wilcoxon agree; replicates averaged) and annotates the CM/FB variant-gene pairs (ENCODE4 cCREs, phyloP, GTF features and coding effects, distance to the target gene TSS, CADD incl. indels from the CADD website, ABC enhancers, cardioid ATAC peaks). Writes `VCF_SDR007AB_DLI_SDR010ABC_Comb.csv` (VEP input) and `..._Meta.csv`. |
| `sdr_cohort_overview.ipynb` | Descriptive figure comparing the positive and negative SDR variants (variant types, target genes, annotation, conservation, cCREs, ATAC peak and enhancer overlap). Reads `..._Meta.csv`. |

### `track_matching/`

| Notebook | What it does |
|---|---|
| `alphagenome_track_matching.ipynb` | Compares the pretrained AlphaGenome RNA and ATAC tracks with the cardioid multiome tracks (outputs of `workflows/alphagenome_api/01_track_matching.sh`) and selects the best-matching AlphaGenome tracks per cell type; the resulting CURIE lists define the track aggregations for pretrained-AlphaGenome VEP. |

### `benchmarks/`

| Notebook | What it does |
|---|---|
| `track_benchmark.ipynb` | Compares the models' track predictions on their held-out test intervals: Pearson r per track (ATAC, Cut&Tag, RNA) and gene-level RNA correlations of pretrained AlphaGenome, fine-tuned AlphaGenome (1 bp and 128 bp embeddings) and Borzoi, plus the ChromBPNet peak count correlation (mean over folds). Reads the outputs of the workflows' test-evaluation steps (`model_eval_dir`) and the ChromBPNet evaluation (`chrombpnet_model_dir`). |

### `model_evaluation/`

| Notebook | What it does |
|---|---|
| `borzoi_test_evaluation.ipynb` | Evaluates the final fine-tuned Borzoi model (`heads_only_deduplicated_squash`) on its test intervals: training and validation loss per epoch, coverage QC of the observed tracks, Pearson r per track over the concatenated intervals and per interval (heatmap, distributions, best and worst examples), and gene-level RNA correlations of summed exon counts (raw and quantile-normalised). Reads the test predictions and `metrics.csv` in `borzoi_checkpoint_dir` and the BigWigs in `bigwig_dir`; the per-track summary is also written by `workflows/borzoi/03_test_eval.sh`. Needs `pip install -e ".[borzoi]"` for the BigWig reader. |

### `attribution/`

| Notebook | What it does |
|---|---|
| `alphagenome_ft_deepshap_amplicons.ipynb` | Analyses DeepSHAP attributions (from `workflows/alphagenome_ft/05_attributions.sh`) of the fine-tuned AlphaGenome ATAC head at the SDR amplicons 39–42: cell-type-specific attributions (CM, FB, ED), normalisation by local activity, TF-MoDISco seqlets with HOCOMOCO motif annotation, and attribution logos around the SDR variants next to their measured effects. |
