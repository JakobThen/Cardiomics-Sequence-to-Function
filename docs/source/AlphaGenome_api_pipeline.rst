Pretrained AlphaGenome Pipeline
===============================

Analyses with the pretrained AlphaGenome model through the AlphaGenome API: which of its
output tracks match the cardioid data, track predictions for the model benchmark, variant
effect prediction (VEP) and in silico mutagenesis (ISM).

All steps live in ``workflows/alphagenome_api/``. Each launcher is run with ``bash`` on the
login node and submits a CPU job (``jobs/*.slurm``) running in the lean, GPU-free
``CS2F_ENV_ALPHAGENOME`` environment (``env/alphagenome_env.yaml`` with
``pip install -e ".[alphagenome]"``). Track matching and prediction hold all predictions in
memory and, like VEP, run on the high-memory partition (``CS2F_SLURM_BIGMEM_PARTITION``; 1-2 TB). The
API key is read from ``ALPHAGENOME_API_KEY`` or ``.env`` (see :doc:`configuration`); logs go
to ``logs/alphagenome_api/``.

Steps 2 and 3 average AlphaGenome tracks over groups of ontology CURIEs defined in the
track aggregations config (``ag_track_aggregations``, gitignored; template
``workflows/alphagenome_api/configs/track_aggregations.example.yaml``). The groups are
chosen from the results of step 1 with the track-matching notebook
(``analysis_notebooks/track_matching/``).


1. Track matching
-----------------

Compares every AlphaGenome RNA-seq and ATAC track with the multiome pseudobulks, one job per
modality (``MODALITIES="rna atac"``):

* **RNA:** for the 5,000 most variable expressed genes, AlphaGenome predicts a 500 kb window
  around the TSS (one third upstream); the predicted signal is summed over the gene's exons
  and correlated with the observed counts across genes (log and quantile-normalised counts).
* **ATAC:** candidate enhancer regions are grouped into shared 500 kb windows; predicted and
  observed (BigWig) signal are correlated at base resolution per window.

* **Output:** ``track_matching_dir/RNA/AG_Multiome_RNA_feature_sum_exon_counts_corr_500KB.h5ad``,
  ``track_matching_dir/ATAC/AG_Multiome_ATAC_stats_enhancer_regions_500KB.parquet``

.. code-block:: bash

    bash workflows/alphagenome_api/01_track_matching.sh
    MODALITIES=atac BW_DIR=/path/to/ATAC_bigwigs bash workflows/alphagenome_api/01_track_matching.sh

.. argparse::
   :module: match_tracks_rna
   :func: get_parser
   :prog: match_tracks_rna.py

.. argparse::
   :module: match_tracks_atac
   :func: get_parser
   :prog: match_tracks_atac.py


2. Track prediction
-------------------

Predicts ATAC and stranded total RNA-seq for the test-fold intervals (``intervals_dir``,
see :doc:`input_prep_pipeline`), averaged per track group, for the benchmark against the
cardioid BigWigs. ``REFERENCE_TRACKS`` names, per group, the cell type whose BigWigs it is
compared with; the naive heart aggregation is compared with Cardiomyocytes, the major cell type.

* **Output:** ``model_eval_dir/AlphaGenome_pretrained/<intervals>_<config>.h5``

.. code-block:: bash

    bash workflows/alphagenome_api/02_track_prediction.sh
    REFERENCE_TRACKS="CM_and_earlyCM_tracks=Cardiomyocytes naive_heart_tracks=Cardiomyocytes" \
        bash workflows/alphagenome_api/02_track_prediction.sh

.. argparse::
   :module: predict_tracks
   :func: get_parser
   :prog: predict_tracks.py


3. Variant effect prediction
----------------------------

Scores variants with the recommended AlphaGenome scorers of the chosen modalities and
averages the scores per variant over each track group (:mod:`cardiomics_s2f.vep.alphagenome_api`).
Variants are scored in batches; finished batches are checkpointed, so an interrupted run
continues when started again with the same input.

* **Output:** ``vep_dir/AG_VEP_out/<vcf name>_<group>_aggr.csv``

.. code-block:: bash

    INPUT_VCF=/path/to/variants.csv bash workflows/alphagenome_api/03_vep.sh
    # optional: MODALITIES="atac rna_seq chip_histone chip_tf", INTERVAL=1MB, API_WORKERS=5

.. argparse::
   :module: run_vep_pretrained
   :func: get_parser
   :prog: run_vep_pretrained.py


4. In silico mutagenesis
------------------------

Scores every single-nucleotide change in each region of a BED file (ATAC, plus ChIP-TF for
given transcription factors and RNA-seq for given genes), averaged over the given ontology
CURIEs (:mod:`cardiomics_s2f.ism`).

* **Output:** ``ism_dir/AG_pretrained_ISM_<bed name>.h5``

.. code-block:: bash

    BED=regions.bed CURIES="UBERON:0002078 UBERON:0002079" TFS="CTCF GATA4" \
        bash workflows/alphagenome_api/04_ism.sh

.. argparse::
   :module: run_ism
   :func: get_parser
   :prog: run_ism.py
