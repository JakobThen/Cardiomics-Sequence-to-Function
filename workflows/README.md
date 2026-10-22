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
| – | `python export_fold_intervals.py` (login node, env `CS2F_ENV_ALPHAGENOME_FT`; `--fold`, `--overwrite`) | `intervals_dir/AG_{train,valid,test}_intervals_fold1_HOMO_SAPIENS.bed` |

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

GPU steps (2-5; step 5 on one GPU) stage the BigWigs to `node_tmp_dir` on the compute node and use
`xla_cache_dir` as a persistent XLA autotune cache.

## alphagenome_api (env: `CS2F_ENV_ALPHAGENOME`, CPU only)

Pretrained AlphaGenome through the API (key from `ALPHAGENOME_API_KEY` or `.env`).
Steps 1-3 run on the high-memory partition (`CS2F_SLURM_BIGMEM_PARTITION`).

| Step | Launcher | Reads | Writes |
|---|---|---|---|
| 1 | `01_track_matching.sh` (`MODALITIES="rna atac"`) | `multiome_gene_counts_csv`, `multiome_gene_pct_csv`, `gtf`, `abc_enhancer_table`, ATAC BigWigs | `track_matching_dir/{RNA,ATAC}/` |
| 2 | `02_track_prediction.sh` (`REFERENCE_TRACKS="GROUP=CELL_TYPE ..."`) | `intervals_dir` (test fold), `ag_track_aggregations` | `model_eval_dir/AlphaGenome_pretrained` |
| 3 | `03_vep.sh` (`INPUT_VCF=<csv>`, `MODALITIES=`) | `vep_input_vcf`, `ag_track_aggregations` | `vep_dir/AG_VEP_out` |
| 4 | `04_ism.sh` (`BED=<bed>`, `CURIES=`, optional `TFS=`, `GENES=`) | regions BED | `ism_dir` |

The track groups (`ag_track_aggregations`) are chosen from the step 1 results with
`analysis_notebooks/track_matching/alphagenome_track_matching.ipynb`.

## common

- `load_site_env.sh`: sourced by every launcher and job; provides `cs2f_path`,
  `cs2f_activate` and `cs2f_sbatch`.
- `get_path.py`: prints one path from `config/paths.yaml` (used by `cs2f_path`).
- `stage_to_node.sh`, `stage_to_tmp.py`: copy BigWigs and the genome to the
  node-local disk and write a config that points to the copies.
