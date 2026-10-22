Model Input Generation Pipeline
===============================

Pipeline of bash scripts to generate base resolution, normalised bigwig files for model training and finetuning.

All steps live in ``workflows/input_prep/``. Each launcher is run with ``bash`` on the login node,
selects the samples to process and submits a SLURM array job (``jobs/*.slurm``, one task per sample)
running in the ``CS2F_ENV_CHROMBPNET`` environment from ``config/site.env``. Paths come from
``config/paths.yaml``; run lists and logs are written to ``logs/input_prep/``. Parameters can be
overridden per run, e.g. ``N_CELLS=200 bash workflows/input_prep/01_atac_bigwig.sh``.

1. ATAC inputs
--------------

Takes pseudobulked ATAC .bam files and sets up a SLURM array job to
compute base-resolution ATAC bigwigs (+4/-4 shift) for AlphaGenome/Borzoi.
Reads are normalized to 100M counts per .bw file.
Optionally filters pseudobulks based on a minimum cell count (``configs/nCells.csv``).
Assumes DEDUPLICATED .bam files.

* **Input:** ``atac_bam_dir/<cell type>.bam``
* **Output:** ``bigwig_dir/multiome_ATAC_bigwig_deduplicated/ATAC_<cell type>_100M_norm.bw``

.. literalinclude:: ../../workflows/input_prep/01_atac_bigwig.sh
   :language: bash
   :linenos:
   :caption: workflows/input_prep/01_atac_bigwig.sh


2. RNA inputs
-------------

Takes pseudobulked scRNA .bam files and sets up a SLURM array job to
compute base-resolution bigwigs for AlphaGenome/Borzoi.
Optionally filters pseudobulks based on a minimum cell count.
Assumes DEDUPLICATED .bam files. Two modes are available via ``RNA_MODE``:

* ``3prime_stranded`` (default): 3' ends split by strand; both strands together sum to 100M counts per
  pseudobulk. Output: ``RNA_<cell type>_{plus,minus}_100M.bw``.
* ``unstranded``: full read coverage without strand information (e.g. for bulk RNA sequencing),
  normalized to 100M. Requires deepTools. Output: ``RNA_<cell type>_100M_norm.bw``.

.. code-block:: bash

    bash workflows/input_prep/02_rna_bigwig.sh
    RNA_MODE=unstranded bash workflows/input_prep/02_rna_bigwig.sh

.. literalinclude:: ../../workflows/input_prep/02_rna_bigwig.sh
   :language: bash
   :linenos:
   :caption: workflows/input_prep/02_rna_bigwig.sh


3. CutnTag inputs
-----------------

Takes bulk Cut&Tag .bam files and sets up a SLURM array job to
compute base-resolution bigwigs for AlphaGenome/Borzoi.
Leverages chromBPNet derived pipeline and transfers it to Cut&Tag.
Reads are normalized to 100M counts per .bw file.
The input (``CUTNTAG_INPUT``, default ``cutntag_bam_input``) can be either:

* A directory path (will scan for all .bam files)
* A text file containing manually curated full paths to .bam files

.. literalinclude:: ../../workflows/input_prep/03_cutntag_bigwig.sh
   :language: bash
   :linenos:
   :caption: workflows/input_prep/03_cutntag_bigwig.sh


4. Fold intervals
-----------------

Writes AlphaGenome's genomic fold split (train/valid/test intervals of 1 Mbp) as BED files, so
that every model (AlphaGenome fine-tuning, Borzoi) and the pretrained-AlphaGenome benchmarks use
the same intervals. Unlike steps 1-3 this is a plain Python script: it runs in seconds on the
login node, in the AlphaGenome environment (``CS2F_ENV_ALPHAGENOME_FT``, which provides
``alphagenome_ft``). Existing files are only replaced with ``--overwrite``.

* **Output:** ``intervals_dir/AG_{train,valid,test}_intervals_fold<fold>_<organism>.bed``
  (tab-separated, header ``chrom start end``)

.. code-block:: bash

    python workflows/input_prep/export_fold_intervals.py
    python workflows/input_prep/export_fold_intervals.py --fold 2 --out_dir /path/to/intervals

.. argparse::
   :module: export_fold_intervals
   :func: get_parser
   :prog: export_fold_intervals.py
