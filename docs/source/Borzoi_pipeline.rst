Borzoi Finetuning Pipeline
==========================

**CLI-based Python scripts and SLURM launchers to fine-tune Borzoi on the cardioid
BigWig tracks and score variant effects with it.**

Borzoi (``human_rep0``) is fine-tuned through `gReLU <https://github.com/Genentech/gReLU>`_ on
the BigWig files prepared by the :doc:`input_prep_pipeline`. Training runs in stages: stage 1
trains new output heads on the frozen pretrained trunk; stage 2 continues from a stage-1
checkpoint with the decoder (U-Net tower) unfrozen. RNA targets are trained on the Borzoi
squashed scale (:class:`~cardiomics_s2f.io.data_input.SelectiveSquashTransform`), and labels are
streamed from the BigWigs during training
(:class:`~cardiomics_s2f.finetuning.borzoi.datasets.StreamingBigWigSeqDataset`).

All steps live in ``workflows/borzoi/``. Each numbered launcher is run with ``bash`` on the login
node and submits a GPU job (``jobs/*.slurm``) in the conda environment ``CS2F_ENV_BORZOI``, in which
``cardiomics_s2f`` must be installed (``pip install -e ".[borzoi]"``). Logs are written to
``logs/borzoi/``. Settings can be overridden per run through environment variables, e.g.
``LR=1e-4 bash workflows/borzoi/01_finetune_heads.sh``.


0. Training intervals
---------------------
Borzoi keeps its own ``human_rep0`` data split (524 kbp windows), which differs from the
AlphaGenome fold intervals. The windows stored in the pretrained model are restricted to autosomes
and chrX, windows past a chromosome end are dropped, and windows within 1 kb of the blacklist are
removed (:func:`~cardiomics_s2f.finetuning.borzoi.intervals.build_borzoi_fold_intervals`). The script
runs on the login node and downloads the pretrained model on first use.

* **Input:** ``chrom_sizes``
* **Output:** ``intervals_dir/Borzoi_human_rep0_{train,val,test}_intervals.bed``

.. code-block:: bash

    python workflows/borzoi/export_borzoi_intervals.py

.. argparse::
   :module: export_borzoi_intervals
   :func: get_parser
   :prog: export_borzoi_intervals.py


1. Fine-tuning the heads
------------------------
Stage 1 adds a new output head with one task per normalised BigWig (``*100M*``) in the folders listed
in ``workflows/borzoi/configs/bigwig_folders.txt`` (gitignored; template
``bigwig_folders.example.txt``). The folders are staged to the GPU node's local disk; tracks are
ordered by folder and file name. Training uses 8 GPUs (learning rate 3e-4, 15 epochs). After
training, the best checkpoint predicts the test intervals.

* **Input:** ``bigwig_dir/<listed folders>``, the Borzoi interval BEDs
* **Output:** ``borzoi_heads_dir/<run>/version_0/checkpoints/`` (best ``epoch*.ckpt`` and
  ``last.ckpt``), ``borzoi_heads_dir/<prefix>_squashed_predictions.h5``

.. code-block:: bash

    bash workflows/borzoi/01_finetune_heads.sh

.. argparse::
   :module: finetune_borzoi
   :func: get_parser
   :prog: finetune_borzoi.py


2. Fine-tuning the decoder
--------------------------
Stage 2 resumes from a stage-1 checkpoint (``PREV_CHECKPOINT``, default ``borzoi_heads_dir``; a
``.ckpt`` file or a folder with exactly one ``epoch*.ckpt``) and unfreezes the U-Net tower and the
pointwise convolution (4 GPUs, learning rate 1e-4). As in the original run, the whole
``bigwig_dir`` is staged. The tracks and their order are taken from the checkpoint's task list
(:func:`~cardiomics_s2f.finetuning.borzoi.training.tracks_from_checkpoint`), so each track file name
must occur in only one ``*bigwig*`` folder there: the script warns when a folder and its
``*_deduplicated`` version are both present and stops on ambiguous names.

The same launcher runs the transformer stage, which has no launcher of its own (it performed worse
than the decoder stage):

.. code-block:: bash

    bash workflows/borzoi/02_finetune_decoder.sh
    PREV_CHECKPOINT=/path/to/decoder/run OUTPUT_DIR=/path/to/transformer \
        UNFREEZE="transformer_tower unet_tower pointwise_conv" LR=1e-5 \
        bash workflows/borzoi/02_finetune_decoder.sh

* **Output:** ``borzoi_decoder_dir/<run>/version_0/checkpoints/``,
  ``borzoi_decoder_dir/<prefix>_squashed_predictions.h5``


3. Test Evaluation
------------------
Steps 1 and 2 save the predictions of the best checkpoint on the test intervals (squashed scale).
This step streams them against the observed BigWig tracks, as the AlphaGenome test evaluation does
(:func:`~cardiomics_s2f.eval.streamed_correlation_analysis.evaluate_from_h5`): Pearson r per track over
the concatenated test intervals (squashed scale), coverage QC and gene-level RNA correlations of
summed exon counts. The BigWigs are taken from the staged paths stored with the predictions
(:func:`~cardiomics_s2f.finetuning.borzoi.training.resolve_staged_track_paths`), so each track maps to
the folder it was trained on. The job runs on CPUs only.

* **Input:** ``PREDICTIONS`` (default: the single ``*_predictions.h5`` in ``borzoi_checkpoint_dir``),
  ``bigwig_dir``, ``gtf``
* **Output:** ``model_eval_dir/Borzoi/<predictions file name>/`` (``track_correlations.csv``, summary
  ``.txt``, ``fig/``)

.. code-block:: bash

    bash workflows/borzoi/03_test_eval.sh

.. argparse::
   :module: evaluate_on_test_borzoi
   :func: get_parser
   :prog: evaluate_on_test_borzoi.py

More plots of the same evaluation (per-interval correlations, coverage QC, gene-level scatters) and the
training curves are in ``analysis_notebooks/model_evaluation/borzoi_test_evaluation.ipynb``.


4. Variant Effect Prediction
----------------------------
The fine-tuned model (``borzoi_checkpoint_dir``) scores variants from a VCF-style CSV (columns
``variant_id, CHROM, POS, REF, ALT``) on one GPU. SNVs are predicted as REF/ALT pairs; indels and
MNVs follow the Borzoi stitching strategy, in which the shifted allele's left and right halves are
combined so that bins stay aligned. RNA predictions are unsquashed first (``RNA_SCALE=squashed``,
the default; ``linear`` for models trained without the squash transform). Scores
(:mod:`cardiomics_s2f.vep.borzoi`):

* ``SUM``, ``logSUM`` and ``L2`` over the whole output window;
* ``logSED`` and ``L2ED`` in a 501 bp (ATAC) or 2001 bp (Cut&Tag) window around the variant and,
  for RNA, over the exons of each overlapping gene (strand-matched tracks only).

* **Output:** ``vep_dir/Borzoi_VEP/VEP_Borzoi_<vcf name>.csv``

.. code-block:: bash

    INPUT_VCF=/path/to/variants.csv bash workflows/borzoi/04_vep.sh

.. argparse::
   :module: run_vep_borzoi
   :func: get_parser
   :prog: run_vep_borzoi.py
