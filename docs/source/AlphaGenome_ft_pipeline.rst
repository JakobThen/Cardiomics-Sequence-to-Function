AlphaGenome Finetuning Pipeline
===============================

**CLI-based Python scripts and SLURM launchers to fine-tune AlphaGenome heads,
evaluate the model on test intervals and score variant effects.**

This tutorial demonstrates how to fine-tune the heads of an AlphaGenome model from previously prepared
BigWig files (see the :doc:`input_prep_pipeline` guide). Heads must be fine-tuned following the 4-fold split
defined in the original paper. This example shows how to train a single fold. For accurate predictions,
models should be trained on all four folds and their predictions averaged.

All steps live in ``workflows/alphagenome_ft/``. Each numbered launcher is run with ``bash`` on the
login node: it reads ``config/paths.yaml`` and ``config/site.env`` and submits the matching job in
``jobs/`` with the site's partition, QoS and GPU constraint. Logs are written to
``logs/alphagenome_ft/``. Settings can be overridden per run through environment variables, e.g.
``BATCH_SIZE=8 bash workflows/alphagenome_ft/02_finetune_heads.sh``.

The jobs run in the conda environment ``CS2F_ENV_ALPHAGENOME_FT`` from ``config/site.env``, in
which ``cardiomics_s2f`` must be installed (``pip install -e .``).


0. Non-zero Mean Computation
----------------------------
Assuming base-resolution BigWig (``.bw``) files are already generated for all modalities, we first need
to compute the non-zero track mean of every file for proper input scaling, restricted to canonical
chromosomes (chr1-22, X, Y).

* **Input:** all ``.bw`` files below ``bigwig_dir``
* **Output:** ``nonzero_means_csv`` (basename, non-zero mean, full path)

.. code-block:: bash

    bash workflows/alphagenome_ft/00_compute_nonzero_means.sh

.. argparse::
   :module: compute_nonzero_means
   :func: get_parser
   :prog: compute_nonzero_means.py


1. Config File Generation
-------------------------
Next, we create a ``.yaml`` configuration file specifying the tracks and heads to fine-tune.

We specify a directory and modality subfolders corresponding to the output heads. All ``.bw`` files in
these folders become tracks of the heads, annotated with their non-zero means from step 0. This file is
used every time we initialize the model to attach the fine-tuned heads to the pre-trained trunk.

* **Input:** modality folders in ``bigwig_dir`` (set in the launcher), ``nonzero_means_csv``
* **Output:** ``ag_targets_config``

.. code-block:: bash

    bash workflows/alphagenome_ft/01_make_targets_config.sh

.. argparse::
   :module: make_targets_config
   :func: get_parser
   :prog: make_targets_config.py


2. Fine-tuning
--------------
After the configuration file is generated, we can begin fine-tuning our AlphaGenome model on 8 GPUs.

To make this run quickly on any cluster, the job first stages the genome and the BigWig files listed in
the targets config to the GPU node's local disk (``node_tmp_dir``) and writes a config pointing to the
staged files (``workflows/common/stage_to_tmp.py``). Checkpoints of the best and last epoch are saved to
``ag_ft_train_dir``.

.. code-block:: bash

    bash workflows/alphagenome_ft/02_finetune_heads.sh

In cases where fine-tuning fails or is aborted (due to a low max epoch count, timeout errors, etc.),
or if the user desires iterative fine-tuning with altered learning rates, the same launcher resumes
from the last checkpoint of a previous run. New checkpoints go to ``<ag_ft_train_dir>_resumed``.

.. code-block:: bash

    RESUME_FROM=/path/to/previous/checkpoints bash workflows/alphagenome_ft/02_finetune_heads.sh

**Caution:** Resuming currently resets optimizer states and only loads weights and biases.
Expect small spikes in loss during the first few epochs.

.. argparse::
   :module: finetune_heads
   :func: get_parser
   :prog: finetune_heads.py

.. argparse::
   :module: stage_to_tmp
   :func: get_parser
   :prog: stage_to_tmp.py


3. Fine-tuning Evaluation
-------------------------
After fine-tuning is complete, we need to ensure its success by assessing the model's predictions
on its test intervals. To avoid having to store all output predictions (which can exceed 1TB at 1bp
resolution as compressed ``.h5`` files), the correlation is computed in a streamed manner while making
forward passes, keeping a running sum for every batch of outputs.

This script integrates the evaluation pipeline (shared across models, :mod:`cardiomics_s2f.eval`)
with AlphaGenome forward passes. It evaluates the ``best`` checkpoint in ``ag_ft_checkpoint_dir``
and writes metrics and figures to ``model_eval_dir/AlphaGenome_ft``.

.. code-block:: bash

    bash workflows/alphagenome_ft/03_test_eval.sh
    # optional: SAVE_PREDICTIONS=1, MINIMAL_TEST=1, RESOLUTION=128

.. argparse::
   :module: evaluate_on_test
   :func: get_parser
   :prog: evaluate_on_test.py


4. Variant Effect Prediction
----------------------------
Finally, the fine-tuned model scores variants from a VCF-style CSV (columns ``variant_id, CHROM, POS,
REF, ALT``). ATAC and Cut&Tag heads are scored by the log2 fold change of the signal summed in a window
centred on the variant; RNA heads by the log fold change of the mean signal over each gene's exons
(:mod:`cardiomics_s2f.vep.scoring`).

.. code-block:: bash

    INPUT_VCF=/path/to/variants.csv bash workflows/alphagenome_ft/04_vep.sh

.. argparse::
   :module: run_vep
   :func: get_parser
   :prog: run_vep.py


5. Attributions
---------------
DeepSHAP-style nucleotide attributions of the fine-tuned model for the regions of a BED file
(e.g. the SDR tiling amplicons): a window (default 16 kb) is centred on each region and the
gradient of one track's summed output (default: the central 500 bp) is compared with the
gradients of dinucleotide-shuffled reference sequences
(:mod:`cardiomics_s2f.finetuning.alphagenome.attributions`). Runs on one GPU; one ``.npy``
per track is written to ``attribution_dir``. Region files are gitignored like the targets
configs (template ``configs/attribution_regions.example.bed``).

.. code-block:: bash

    REGIONS=workflows/alphagenome_ft/configs/sdr_amplicons.bed \
    TRACKS="ATAC_Cardiomyocytes ATAC_Fibroblasts" \
        bash workflows/alphagenome_ft/05_attributions.sh

.. argparse::
   :module: compute_attributions
   :func: get_parser
   :prog: compute_attributions.py
