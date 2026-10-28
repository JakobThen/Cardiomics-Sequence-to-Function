# Workflows

Runnable pipelines of the project. The reusable code they call lives in the
`cardiomics_s2f` package (`src/`); this folder only holds launchers, SLURM job
scripts and thin command-line scripts.

## One-time setup

```bash
cp config/paths.example.yaml config/paths.yaml   # data, model and output locations
cp config/site.example.env   config/site.env     # conda envs and SLURM settings
pip install -e .                                 # inside the AlphaGenome env (CS2F_ENV_ALPHAGENOME_FT)
```

`config/paths.yaml`, `config/site.env` and `.env` are gitignored. The login-node
Python (`CS2F_PYTHON`, default `python3`) needs PyYAML to read `paths.yaml`.

Data-specific inputs are gitignored too; each has a tracked `*.example.*` next to it:

| File | Example |
|---|---|
| `input_prep/configs/nCells.csv` (optional cell-count filter) | `nCells.example.csv` |
| `input_prep/configs/cutntag_bams.txt` (optional Cut&Tag BAM list) | `cutntag_bams.example.txt` |
| AlphaGenome targets config (`ag_targets_config`, built by step 01) | `alphagenome_ft/configs/targets_config.example.yaml` |
| Attribution regions (`alphagenome_ft` step 05, e.g. `sdr_amplicons.bed`) | `alphagenome_ft/configs/attribution_regions.example.bed` |
| Pretrained AlphaGenome track groups (`ag_track_aggregations`) | `alphagenome_api/configs/track_aggregations.example.yaml` |
| Borzoi stage-1 BigWig folders (`borzoi/configs/bigwig_folders.txt`) | `borzoi/configs/bigwig_folders.example.txt` |

## How a step runs

Every numbered step is started with `bash` on the login node, from any directory:

```bash
bash workflows/alphagenome_ft/02_finetune_heads.sh
```

The **launcher** (`NN_*.sh`) resolves all paths from `config/paths.yaml`, checks
its inputs and submits the matching **job** (`jobs/*.slurm`) via `cs2f_sbatch`,
which adds the partition, QoS, GPU constraint and account from `config/site.env`.
Job files only request resources (GPUs, memory, time). Logs and run lists go to
`<logs_dir>/<workflow>/` (default `logs/<workflow>/`).

Settings at the top of a launcher can be overridden per run with environment
variables, e.g. `N_CELLS=200 bash workflows/input_prep/01_atac_bigwig.sh`.

## input_prep (env: `CS2F_ENV_CHROMBPNET`)

| Step | Launcher | Output |
|---|---|---|
| 1 | `01_atac_bigwig.sh` | `bigwig_dir/multiome_ATAC_bigwig_deduplicated/ATAC_<cell type>_100M_norm.bw` |
| 2 | `02_rna_bigwig.sh` (`RNA_MODE=3prime_stranded` default, or `unstranded`) | `bigwig_dir/multiome_RNA_3prime_bigwig_deduplicated/` or `.../multiome_RNA_unstranded_bigwig/` |
| 3 | `03_cutntag_bigwig.sh` (`CUTNTAG_INPUT=` folder or list of BAMs) | `bigwig_dir/CutnTag_bigwig/<sample>_100M_norm.bw` |

Inputs: `atac_bam_dir`, `rna_bam_dir`, `cutntag_bam_input` (folder or list of BAM paths);
cell-count filter `input_prep/configs/nCells.csv` (`N_CELLS`, default 400).

## alphagenome_ft (env: `CS2F_ENV_ALPHAGENOME_FT`)

| Step | Launcher | Reads | Writes |
|---|---|---|---|
| 0 | `00_compute_nonzero_means.sh` | `bigwig_dir` | `nonzero_means_csv` |
| 1 | `01_make_targets_config.sh` | `bigwig_dir`, `nonzero_means_csv` | `ag_targets_config` |
| 2 | `02_finetune_heads.sh` (`RESUME_FROM=<ckpt dir>` to resume) | `ag_targets_config`, `genome_fasta` | `ag_ft_train_dir` (`_resumed`) |
| 3 | `03_test_eval.sh` (`SAVE_PREDICTIONS=1`, `MINIMAL_TEST=1`, `RESOLUTION=128`) | `ag_targets_config`, `ag_ft_checkpoint_dir/best`, `gtf` | `model_eval_dir/AlphaGenome_ft` |
| 4 | `04_vep.sh` (`INPUT_VCF=<csv>`) | `ag_targets_config`, `ag_ft_checkpoint_dir/best`, `vep_input_vcf` | `vep_dir/AG_ft_VEP_out` |
| 5 | `05_attributions.sh` (`REGIONS=<bed>`, `HEAD=`, `TRACKS=`) | `ag_targets_config`, `ag_ft_checkpoint_dir/best`, `genome_fasta` | `attribution_dir/<prefix>_<track>.npy` |
| – | `python export_fold_intervals.py` (login node; `--fold`, `--overwrite`) | – | `intervals_dir/AG_{train,valid,test}_intervals_fold1_HOMO_SAPIENS.bed` (1 Mbp; not used by Borzoi) |

GPU steps (2-5; step 5 on one GPU) stage the BigWigs to `node_tmp_dir` on the compute node and use
`xla_cache_dir` as a persistent XLA autotune cache.

Only **fold 1** of the four AlphaGenome folds is trained and evaluated (fixed in steps 2-5);
repeat for the other folds and average the predictions for final results.

## alphagenome_api (env: `CS2F_ENV_ALPHAGENOME`, CPU only)

Pretrained AlphaGenome through the API (key from `ALPHAGENOME_API_KEY` or `.env`).
Steps 1, 2 and 4 run on the high-memory partition (`CS2F_SLURM_BIGMEM_PARTITION`); the streamed test evaluation (3) on the CPU partition.

| Step | Launcher | Reads | Writes |
|---|---|---|---|
| 1 | `01_track_matching.sh` (`MODALITIES="rna atac"`) | `multiome_gene_counts_csv`, `multiome_gene_pct_csv`, `gtf`, `abc_enhancer_table`, ATAC BigWigs | `track_matching_dir/{RNA,ATAC}/` |
| 2 | `02_track_prediction.sh` (`REFERENCE_TRACKS="GROUP=CELL_TYPE ..."`) | `intervals_dir` (test fold), `ag_track_aggregations` | `model_eval_dir/AlphaGenome_pretrained` |
| 3 | `03_test_eval.sh` (CPU; `PREDICTIONS=`) | step-2 `.h5`, `ag_targets_config` (reference BigWigs, non-zero means), `gtf` | `model_eval_dir/AlphaGenome_pretrained/<predictions name>` |
| 4 | `04_vep.sh` (`INPUT_VCF=<csv>`, `MODALITIES=`) | `vep_input_vcf`, `ag_track_aggregations` | `vep_dir/AG_VEP_out` |
| 5 | `05_ism.sh` (`BED=<bed>`, `CURIES=`, optional `TFS=`, `GENES=`) | regions BED | `ism_dir` |

The track groups (`ag_track_aggregations`) are chosen from the step 1 results with
`analysis_notebooks/track_matching/alphagenome_track_matching.ipynb`.

## borzoi (env: `CS2F_ENV_BORZOI`)

Borzoi `human_rep0` fine-tuned through gReLU (`pip install -e ".[borzoi]"`).
Only the `human_rep0` replicate and its split are fine-tuned; repeat from `human_rep1`-`human_rep3`
and average the predictions for final results.

| Step | Launcher | Reads | Writes |
|---|---|---|---|
| – | `python export_borzoi_intervals.py` (login node; `--overwrite`) | `chrom_sizes`, pretrained Borzoi (downloaded) | `intervals_dir/Borzoi_human_rep0_{train,val,test}_intervals.bed` (524 kbp) |
| 1 | `01_finetune_heads.sh` (8 GPUs; `FOLDERS_FILE=`, `LR=`, `MAX_EPOCHS=`) | `bigwig_dir/<folders in configs/bigwig_folders.txt>`, Borzoi BEDs | `borzoi_heads_dir` |
| 2 | `02_finetune_decoder.sh` (4 GPUs; `PREV_CHECKPOINT=`, `UNFREEZE=`) | whole `bigwig_dir`, stage-1 checkpoint, Borzoi BEDs | `borzoi_decoder_dir` |
| 3 | `03_test_eval.sh` (CPU; `PREDICTIONS=`, `RNA_SCALE=`) | `borzoi_checkpoint_dir/*_predictions.h5`, `bigwig_dir`, `gtf` | `model_eval_dir/Borzoi/<predictions name>` |
| 4 | `04_vep.sh` (1 GPU; `INPUT_VCF=`, `CHECKPOINT=`, `RNA_SCALE=squashed\|linear`) | `borzoi_checkpoint_dir`, `vep_input_vcf`, `gtf`, `chrom_sizes` | `vep_dir/Borzoi_VEP` |

Steps 1-2 stage the BigWigs to `node_tmp_dir` and write the test-interval predictions
(squashed scale) next to the checkpoints. Step 2 takes its tracks, in order, from the
checkpoint, so a track file name must occur in only one `*bigwig*` folder of `bigwig_dir`.
The transformer stage has no launcher; run step 2 with
`UNFREEZE="transformer_tower unet_tower pointwise_conv" LR=1e-5`.

## chrombpnet (env: `CS2F_ENV_CHROMBPNET`)

ChromBPNet 0.1.7 (TensorFlow 2.8; GPU jobs default to `GPU_CONSTRAINT="gpu=A100"`). VEP uses the
Kundaje lab variant-scorer as a pinned git checkout: run `bash install_variant_scorer.sh` once.

| Step | Launcher | Reads | Writes |
|---|---|---|---|
| 1 | `01_prepare_inputs.sh` (CPU) | `atac_peak_dir` (MACS2), `atac_bam_dir`, `blacklist_bed`, `chrom_sizes` | `chrombpnet_input_dir` (peaks without blacklist overlap, `blacklist.chr.bed`) |
| 2 | `02_train.sh` (array, 1 task per cell type; `CELL_TYPES=`, `N_CELLS=300`, `N_PEAKS=10000`, `FOLDS=`, `OVERWRITE=1`) | `atac_bam_dir`, `chrombpnet_input_dir`, `chrombpnet_bias_model`, `configs/splits/`, `input_prep/configs/nCells.csv` | `chrombpnet_train_dir/<cell type>_<fold>/` |
| 3 | `03_vep.sh` (`INPUT_VCF=`, `CELL_TYPES=`, `SHAP=0`) | `chrombpnet_model_dir`, `vep_input_vcf` | `vep_dir/chromBPNet_VEP_out` |

The BAMs are read in place (no copies). Training reruns the BAM-to-BigWig conversion for every fold
(`chrombpnet pipeline`).

## common

- `load_site_env.sh`: sourced by every launcher and job; provides `cs2f_path`,
  `cs2f_activate` and `cs2f_sbatch`.
- `get_path.py`: prints one path from `config/paths.yaml` (used by `cs2f_path`).
- `stage_to_node.sh`, `stage_to_tmp.py`: copy BigWigs and the genome to the
  node-local disk and write a config that points to the copies.
- `check_torch_cuda.py`: prints the PyTorch/CUDA setup at the start of PyTorch GPU jobs.
