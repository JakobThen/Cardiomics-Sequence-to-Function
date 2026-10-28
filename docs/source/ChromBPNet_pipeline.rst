ChromBPNet Pipeline
===================

**SLURM launchers to train ChromBPNet models on pseudobulk ATAC data and score variant effects
with them.**

`ChromBPNet <https://github.com/kundajelab/chrombpnet>`_ (0.1.7) predicts base-resolution chromatin
accessibility from sequence and separates the Tn5 enzyme bias from the biological signal with a
pretrained bias model. One model is trained per cell type and fold (five folds); variant effects are
averaged across folds with the Kundaje lab
`variant-scorer <https://github.com/kundajelab/variant-scorer>`_.

All steps live in ``workflows/chrombpnet/``. Each launcher is run with ``bash`` on the login node and
submits a job (``jobs/*.slurm``) in the conda environment ``CS2F_ENV_CHROMBPNET`` (Python 3.10,
``pip install chrombpnet==0.1.7`` and ``pip install -e ".[chrombpnet]"``; bedtools, samtools and the
UCSC tools from conda). ChromBPNet 0.1.7 uses TensorFlow 2.8 with CUDA 11, so the GPU jobs request
A100 GPUs by default (``GPU_CONSTRAINT="gpu=A100"``; set it to ``""`` for the site default). Logs and
run lists are written to ``logs/chrombpnet/``.

**variant-scorer** is not a pip package; it is installed once as a git checkout pinned to the commit
used in this project (``workflows/chrombpnet/variant_scorer.sh``):

.. code-block:: bash

    bash workflows/chrombpnet/install_variant_scorer.sh   # into CS2F_VARIANT_SCORER_DIR or external/variant-scorer


1. Input preparation
--------------------
ChromBPNet does not remove peaks in blacklisted regions itself. As in the ChromBPNet tutorial, the
MACS2 peaks of every pseudobulk are filtered against the genome blacklist expanded by 1057 bp (half
the 2114 bp model input). The blacklist is also written with ``chr`` prefixes for the background
regions of step 2. The pseudobulk BAMs are not copied; training reads them in place.

* **Input:** ``atac_peak_dir/<cell type>.filtered.short.BAM_peaks.narrowPeak``,
  ``atac_bam_dir/<cell type>.bam``, ``blacklist_bed``, ``chrom_sizes``
* **Output:** ``chrombpnet_input_dir/<cell type>_peaks_no_blacklist.narrowPeak``,
  ``blacklist.chr.bed``, ``blacklist.slop.bed``

.. code-block:: bash

    bash workflows/chrombpnet/01_prepare_inputs.sh


2. Training
-----------
Pseudobulks are kept if they have at least ``N_CELLS`` cells (default 300, from
``workflows/input_prep/configs/nCells.csv``) and ``N_PEAKS`` peaks (default 10,000). One array task
per cell type trains all folds (``workflows/chrombpnet/configs/splits/fold_{0..4}.json``, the
standard hg38 splits):

#. ``chrombpnet prep nonpeaks``: GC-matched background regions outside peaks and the blacklist;
#. ``chrombpnet pipeline``: converts the BAM to a Tn5-shifted BigWig, trains the bias-factorized model
   with the pretrained ENCODE ATAC bias model (``chrombpnet_bias_model``), evaluates it on the test
   chromosomes (``evaluation/chrombpnet_metrics.json``) and computes contribution scores and motifs.

Finished folds are skipped (``OVERWRITE=1`` retrains them); incomplete folds are removed and restarted.

* **Output:** ``chrombpnet_train_dir/<cell type>_<fold>/`` (``models/chrombpnet_nobias.h5``,
  ``evaluation/``, ``logs/``), ``<cell type>_<fold>_negatives.bed``, ``<cell type>_<fold>_auxiliary/``

.. code-block:: bash

    bash workflows/chrombpnet/02_train.sh
    CELL_TYPES="all" N_CELLS=500 bash workflows/chrombpnet/02_train.sh


3. Variant Effect Prediction
----------------------------
Variants from a VCF-style CSV (``variant_id, CHROM, POS, REF, ALT``) are converted to the
variant-scorer format: variants whose REF does not match the genome are removed, and the bases REF
and ALT share at the start of indels are trimmed (:mod:`cardiomics_s2f.vep.chrombpnet`). For every
cell type in ``CELL_TYPES`` (default Cardiomyocytes, Fibroblasts, Cardiac_progenitors) and fold,
``variant_scoring.py`` computes the allele log fold change (``logfc``), the Jensen-Shannon distance of
the profiles (``jsd``) and their product, ``variant_shap.py`` the counts contribution scores
(``SHAP=0`` to skip), and ``variant_summary_across_folds.py`` averages the scores across folds. The
summaries are read with :func:`cardiomics_s2f.vep.processing.format_chrombpnet_vep`.

* **Input:** ``INPUT_VCF`` (default ``vep_input_vcf``),
  ``chrombpnet_model_dir/<cell type>_<fold>/models/chrombpnet_nobias.h5``
* **Output:** ``vep_dir/chromBPNet_VEP_out/<vcf name>_<cell type>.*`` (fold summary) and
  ``<vcf name>_<cell type>_<fold>_*`` (SHAP)

.. code-block:: bash

    INPUT_VCF=/path/to/variants.csv bash workflows/chrombpnet/03_vep.sh

.. argparse::
   :module: convert_variants
   :func: get_parser
   :prog: convert_variants.py
