# Cardiomics Sequence-to-Function

[![Documentation Status](https://readthedocs.org/projects/cardiomics-sequence-to-function/badge/?version=latest)](https://cardiomics-sequence-to-function.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Tools and pipelines to train, fine-tune and evaluate sequence-to-function models
on pseudobulked single-cell multiome (ATAC, RNA) and bulk Cut&Tag data, and to
predict the effects of genetic variants on chromatin accessibility, histone marks
and gene expression.

The repository has three parts:

- **`cardiomics_s2f`**: a Python package with the reusable code: data
  input/output, streamed model evaluation, variant effect prediction (VEP)
  scoring and plotting.
- **`workflows/`**: SLURM pipelines built on the package that prepare model
  inputs and fine-tune, evaluate and apply the models on an HPC cluster.
- **`analysis_notebooks/`**: Jupyter notebooks for the downstream analyses
  (variant datasets, track matching, attributions); see
  [`analysis_notebooks/README.md`](analysis_notebooks/README.md).

## Documentation

Full documentation, including pipeline guides and the API reference, is on
[Read the Docs](https://cardiomics-sequence-to-function.readthedocs.io/).

## Repository layout

```
src/cardiomics_s2f/   Python package (library code only)
workflows/            SLURM launchers, job scripts and command-line tools
analysis_notebooks/   Analysis notebooks (outputs stripped)
config/               Templates for local paths and cluster settings
env/                  Conda environments of the model families
docs/                 Sphinx documentation
tests/                Unit tests
input_data/ models/ results/   Local data folders (not tracked, see their READMEs)
```

## Pipelines

**Model input preparation.** Pseudobulked BAM files are converted into
base-resolution, depth-normalised BigWig tracks (ATAC and Cut&Tag with Tn5 shift
correction, stranded 3' RNA).

**AlphaGenome fine-tuning.** New output heads (one per modality) are trained on
top of the pretrained AlphaGenome model, using the BigWig tracks as targets and
AlphaGenome's genomic fold splits. Data is staged to node-local disks and
training runs data-parallel on multiple GPUs; interrupted runs can be resumed.

**Evaluation.** Fine-tuned models are evaluated on held-out test intervals by
streaming predictions and computing track- and gene-level correlations against
the observed signal, without storing the full-resolution predictions.

**Variant effect prediction.** Reference and alternative sequences are scored
with the fine-tuned model; effects are aggregated per variant in a window around
it (ATAC, Cut&Tag) or over the exons of nearby genes (RNA), and can be
benchmarked against measured variant effects.

**Pretrained AlphaGenome.** Through the AlphaGenome API, the pretrained model's
output tracks are matched to the cardioid data, and the best-matching tracks are
used for track predictions, variant effect prediction and in silico mutagenesis.

Borzoi and ChromBPNet analyses are being migrated into the package.

## Quick start

```bash
git clone https://github.com/JakobThen/Cardiomics-Sequence-to-Function.git
cd Cardiomics-Sequence-to-Function

# Environment and package
conda env create -f env/alphagenome_ft_env.yaml
conda activate alphagenome_ft
pip install -e ".[alphagenome-ft]"

# Local configuration (gitignored)
cp config/paths.example.yaml config/paths.yaml   # where your data lives
cp config/site.example.env   config/site.env     # conda envs and SLURM settings
cp .env.example .env                             # API keys (pretrained AlphaGenome)
```

Pipeline steps are started from the login node, for example
`bash workflows/alphagenome_ft/02_finetune_heads.sh`. See
[`workflows/README.md`](workflows/README.md) for all steps, and the
documentation for [installation](https://cardiomics-sequence-to-function.readthedocs.io/en/latest/installation.html)
and [configuration](https://cardiomics-sequence-to-function.readthedocs.io/en/latest/configuration.html).
