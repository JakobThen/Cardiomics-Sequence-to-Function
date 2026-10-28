Configuration
=============

All machine- and data-specific settings live in three local files, which are
gitignored. Each has a committed template:

.. code-block:: bash

    cp config/paths.example.yaml config/paths.yaml   # data, model and output locations
    cp config/site.example.env   config/site.env     # conda environments and SLURM settings
    cp .env.example .env                             # API keys

No library function or workflow script contains hard-coded paths: Python code
reads them with :func:`cardiomics_s2f.config.load_paths`, shell scripts with
``cs2f_path <key>``.


Data and output paths: ``config/paths.yaml``
--------------------------------------------

Resolution rules:

* ``data_root``: a relative value resolves against the folder of
  ``paths.yaml`` (``config/``), so the template's ``..`` is the repository.
* All other entries: relative values resolve against ``data_root``; absolute
  values are used as they are. ``~`` and environment variables (e.g.
  ``$SCRATCH``) are expanded.
* Point entries to the real data locations rather than to symlinks.
* The file is found via ``$CS2F_PATHS``, else ``config/paths.yaml`` in the
  current folder or a parent, else in the repository.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Content (used by)
   * - ``data_root``
     - Base for relative entries
   * - ``atac_bam_dir``, ``rna_bam_dir``
     - Pseudobulk BAMs, one ``<cell type>.bam`` each (input prep 1, 2)
   * - ``cutntag_bam_input``
     - Folder of Cut&Tag BAMs, or a text file listing BAM paths (input prep 3)
   * - ``bigwig_dir``
     - Folder holding the modality BigWig folders (input prep output; AlphaGenome 0, 1)
   * - ``nonzero_means_csv``
     - Non-zero mean per BigWig (written by AlphaGenome 0, read by 1)
   * - ``genome_fasta``, ``chrom_sizes``
     - Reference genome (``.fai`` index next to the FASTA) and chromosome sizes
   * - ``gtf``
     - Gene annotation for gene-level evaluation (AlphaGenome 3)
   * - ``ag_targets_config``
     - AlphaGenome targets config (written by AlphaGenome 1, read by 2-4)
   * - ``ag_ft_train_dir``
     - Where fine-tuning writes checkpoints (AlphaGenome 2; resumed runs: ``<dir>_resumed``)
   * - ``ag_ft_checkpoint_dir``
     - Checkpoint whose ``best`` subfolder is evaluated (AlphaGenome 3, 4)
   * - ``vep_input_vcf``
     - Default VEP input: CSV with ``variant_id, CHROM, POS, REF, ALT`` (AlphaGenome 4, Borzoi 4)
   * - ``borzoi_heads_dir``, ``borzoi_decoder_dir``
     - Where Borzoi stage 1 (heads) and stage 2 (decoder) write their runs; stage 2 resumes from ``borzoi_heads_dir``
   * - ``borzoi_checkpoint_dir``
     - Borzoi model for test evaluation (Borzoi 3, its ``*_predictions.h5``) and VEP (Borzoi 4): a ``.ckpt`` file or a folder with exactly one ``epoch*.ckpt``
   * - ``model_eval_dir``, ``vep_dir``
     - Outputs of test evaluation and VEP
   * - ``xla_cache_dir``
     - Persistent XLA autotune cache for GPU jobs
   * - ``logs_dir``
     - SLURM logs and run lists, one subfolder per workflow
   * - ``node_tmp_dir``
     - Node-local disk for staging BigWigs on GPU nodes
   * - ``intervals_dir``
     - Interval BEDs: AlphaGenome fold intervals (written by ``alphagenome_ft/export_fold_intervals.py``, read by pretrained track prediction) and the Borzoi ``human_rep0`` windows (written by ``borzoi/export_borzoi_intervals.py``, read by Borzoi 1-2)
   * - ``ag_track_aggregations``
     - Track groups (ontology CURIEs) for pretrained-AlphaGenome track prediction and VEP
   * - ``multiome_gene_counts_csv``, ``multiome_gene_pct_csv``
     - Pseudobulk RNA counts and percentage of expressing cells per gene (RNA track matching)
   * - ``abc_enhancer_table``, ``atac_peak_dir``
     - Cardioid candidate enhancers (ATAC track matching, variant annotation) and pseudobulk ATAC peaks (also ChromBPNet 1)
   * - ``chrombpnet_input_dir``, ``blacklist_bed``
     - ChromBPNet inputs written by ChromBPNet 1 (filtered peaks, ``chr`` blacklist) from the genome blacklist
   * - ``chrombpnet_bias_model``, ``chrombpnet_train_dir``, ``chrombpnet_model_dir``
     - Pretrained ChromBPNet bias model; where ChromBPNet 2 writes new models; models used by ChromBPNet 3 (VEP)
   * - ``track_matching_dir``, ``ism_dir``, ``attribution_dir``
     - Outputs of track matching, ISM and fine-tuned AlphaGenome attributions
   * - ``variant_data_dir``, ``motif_meme_file``, ``figures_dir``
     - Variant tables and annotation tracks, motif database and figure folder of the analysis notebooks
   * - ``sdr007_de_dir``, ``sdr010_de_dir``
     - SDR-seq differential expression results (MAST and Wilcoxon tables) read by ``sdr_variant_dataset.ipynb``
   * - ``models_dir``, ``results_dir``
     - Not read by the current workflows; available to notebooks via :func:`~cardiomics_s2f.config.load_paths`

.. code-block:: python

    from cardiomics_s2f.config import load_paths

    paths = load_paths()
    paths.genome_fasta          # or paths["genome_fasta"], a pathlib.Path


Cluster settings: ``config/site.env``
-------------------------------------

Sourced by every launcher and job script (``workflows/common/load_site_env.sh``).
Set ``$CS2F_SITE_ENV`` to use a different file. Values containing ``|`` must be
quoted.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Variable
     - Meaning
   * - ``CS2F_PYTHON``
     - Python on the login node used to read ``paths.yaml`` (needs PyYAML); default ``python3``
   * - ``CS2F_CONDA_SH``
     - conda shell hook (``.../etc/profile.d/conda.sh``), needed for ``conda activate`` in jobs
   * - ``CS2F_ENV_ALPHAGENOME_FT``
     - Environment (name or prefix) for ``workflows/alphagenome_ft``
   * - ``CS2F_ENV_ALPHAGENOME``
     - Lean, GPU-free environment (name or prefix) for ``workflows/alphagenome_api``
   * - ``CS2F_ENV_CHROMBPNET``
     - Environment (name or prefix) for ``workflows/input_prep`` and ``workflows/chrombpnet`` (Python 3.10,
       ``pip install chrombpnet==0.1.7`` and ``pip install -e ".[chrombpnet]"``)
   * - ``CS2F_VARIANT_SCORER_DIR``
     - Pinned variant-scorer checkout for ChromBPNet VEP (empty: ``<repo>/external/variant-scorer``;
       ``bash workflows/chrombpnet/install_variant_scorer.sh``)
   * - ``CS2F_ENV_BORZOI``
     - Environment (name or prefix) for ``workflows/borzoi`` (``pip install -e ".[borzoi]"``)
   * - ``CS2F_SLURM_ACCOUNT``
     - ``--account`` for all jobs (empty: omitted)
   * - ``CS2F_SLURM_CPU_PARTITION``
     - ``--partition`` for CPU jobs
   * - ``CS2F_SLURM_BIGMEM_PARTITION``
     - ``--partition`` for high-memory jobs (pretrained AlphaGenome track matching, prediction, VEP)
   * - ``CS2F_SLURM_GPU_PARTITION``, ``CS2F_SLURM_GPU_QOS``, ``CS2F_SLURM_GPU_CONSTRAINT``
     - ``--partition``, ``--qos`` and ``--constraint`` for GPU jobs (e.g. ``"gpu=H100|gpu=A100"``)

Empty values are left out of the ``sbatch`` call.


API keys: ``.env``
------------------

The pretrained AlphaGenome API needs a key.
:func:`cardiomics_s2f.config.get_alphagenome_api_key` returns, in this order,
an explicit argument, the ``ALPHAGENOME_API_KEY`` environment variable, or
``ALPHAGENOME_API_KEY`` from the nearest ``.env`` file. Keys are never passed on
the command line, where they would appear in process lists and SLURM logs.

.. code-block:: python

    from alphagenome.models import dna_client
    from cardiomics_s2f.config import get_alphagenome_api_key

    model = dna_client.create(get_alphagenome_api_key())


Data-specific input files
-------------------------

Inputs that describe your data are gitignored; a tracked ``*.example.*`` file
next to each shows the format:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - File
     - Purpose
   * - ``workflows/input_prep/configs/nCells.csv``
     - Optional cell count per pseudobulk; pseudobulks below ``N_CELLS`` (default 400) are skipped
   * - ``workflows/input_prep/configs/cutntag_bams.txt``
     - Optional list of Cut&Tag BAM paths (for ``cutntag_bam_input``)
   * - ``workflows/alphagenome_ft/configs/*.yaml``
     - AlphaGenome targets configs (built by step 1, see :doc:`AlphaGenome_ft_pipeline`)
   * - ``workflows/alphagenome_ft/configs/*.bed``
     - Regions for fine-tuned AlphaGenome attributions (step 5)
   * - ``workflows/alphagenome_api/configs/track_aggregations.yaml``
     - Track groups for pretrained AlphaGenome (see :doc:`AlphaGenome_api_pipeline`)


Running a workflow step
-----------------------

Each numbered step is a launcher, run with ``bash`` on the login node from any
folder. It resolves paths, checks its inputs and submits the job in ``jobs/``
with the settings from ``site.env``; logs go to ``<logs_dir>/<workflow>/``.
Settings at the top of a launcher can be overridden per run:

.. code-block:: bash

    N_CELLS=200 bash workflows/input_prep/01_atac_bigwig.sh
    RESUME_FROM=/path/to/checkpoints bash workflows/alphagenome_ft/02_finetune_heads.sh

The steps are described in :doc:`input_prep_pipeline` and
:doc:`AlphaGenome_ft_pipeline`.
