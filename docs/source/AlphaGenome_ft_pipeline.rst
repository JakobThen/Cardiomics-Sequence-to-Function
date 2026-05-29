AlphaGenome Finetuning Pipeline
===============================

**CLI-based Python scripts and shell wrappers to build and run an AlphaGenome fine-tuning pipeline
and evaluate the model outputs on test intervals.**  

This tutorial demonstrates how to fine-tune the heads of an AlphaGenome model from previously prepared 
BigWig files (see the :doc:`input_prep_pipeline` guide). Heads must be fine-tuned following the 4-fold split 
defined in the original paper. This example shows how to train a single fold. For accurate predictions, 
models should be trained on all four folds and their predictions averaged.


Non-zero Mean Computation
----------------------------
Assuming base-resolution BigWig (``.bw``) files are already generated for all modalities, we first need 
to compute the non-zero track mean for proper input scaling. This is handled directly in bash. 

Computes the mean of non-zero track values for BigWig files, specifically restricted to canonical 
chromosomes (chr1-22, X). 

* **Input:** ``$1`` - Directory containing input ``.bw`` files
* **Output:** ``$2`` - CSV file capturing basename, non-zero mean, and full path

.. literalinclude:: ../../src/AlphaGenome_finetuning/00_compute_nonzero_means_per_bw.sh
   :language: bash
   :linenos:
   :caption: src/AlphaGenome_finetuning/00_compute_nonzero_means_per_bw.sh


1. Config File Generation
-------------------------
Next, we create a ``.yaml`` configuration file specifying the tracks and heads to fine-tune.

We specify a directory and modality subfolders corresponding to the output heads. This Python script
loads all ``.bw`` files in the directory as tracks for the heads and generates a configuration ``.yaml`` file. 
This file is used every time we initialize the model to attach the fine-tuned heads to the pre-trained trunk.

Python script:

.. argparse::
   :module: utils.preprocessing.make_AG_input_config
   :func: get_parser
   :prog: make_AG_input_config.py

Alternatively, we can use a bash wrapper to execute the Python config generator.

* **Input:** Hardcoded variables pointing to directories and metadata
* **Output:** A formatted ``.yaml`` configuration file (e.g., ``test_config.yaml``)

.. literalinclude:: ../../src/AlphaGenome_finetuning/01_make_config_file.sh
   :language: bash
   :linenos:
   :caption: src/AlphaGenome_finetuning/01_make_config_file.sh


2. Fine-tuning
--------------
After the configuration file is generated, we can begin fine-tuning our AlphaGenome model. 
The fine-tuning is handled by the following script:

.. argparse::
   :module: utils.finetuning.AlphaGenome.finetune_heads
   :func: get_parser
   :prog: finetune_heads.py

To make this run quickly on any cluster, it is highly recommended to stage the BigWig data to the 
``tmp`` directory of the GPU node before starting the fine-tuning.

We can do this with the following helper script, which stages the ``.bw`` files specified in the 
``config.yaml`` from the input directory to a ``tmp`` directory on the node. It also generates a new 
``config.yaml`` in the ``tmp`` directory with updated data paths pointing to the staged ``.bw`` files.

.. argparse::
   :module: utils.finetuning.AlphaGenome.stage_to_tmp
   :func: get_parser
   :prog: stage_to_tmp.py

As an example, this bash script can be used to execute the entire data staging and fine-tuning 
process on a SLURM cluster:

.. literalinclude:: ../../src/AlphaGenome_finetuning/02_submit_finetune_heads.sh
   :language: bash
   :linenos:
   :caption: src/AlphaGenome_finetuning/02_submit_finetune_heads.sh

In cases where fine-tuning fails or is aborted (due to a low max epoch count, timeout errors, etc.), 
or if the user desires iterative fine-tuning with altered learning rates, loss weights, or additional 
unfrozen layers, we created a resume fine-tuning script. This allows you to restart the fine-tuning 
from an existing model checkpoint.

**Caution:** This script currently resets optimizer states (parameters optimized during validation steps) 
and only loads weights and biases. Expect small spikes in loss during the first few epochs.

.. argparse::
   :module: utils.finetuning.AlphaGenome.resume_finetune_heads
   :func: get_parser
   :prog: resume_finetune_heads.py

An example of a full bash wrapper for resuming a job can be found at: 
``../../src/AlphaGenome_finetuning/02b_resume_submit_finetune_heads.sh``


3. Fine-tuning Evaluation
-------------------------
After fine-tuning is complete, we need to ensure its success by assessing the model's predictions 
on its test intervals. To avoid having to store all output predictions (which can exceed 1TB at 1bp 
resolution as compressed ``.h5`` files), the correlation is computed in a streamed manner while making 
forward passes, keeping a running sum for every batch of outputs.

This script integrates the evaluation pipeline (shared across models, found at ``../../src/utils/eval``) 
with AlphaGenome forward passes.

.. argparse::
   :module: utils.finetuning.AlphaGenome.evaluate_AG_on_test_streamed
   :func: get_parser
   :prog: evaluate_AG_on_test_streamed.py

To speed up computation, this can be integrated with data staging (just like the fine-tuning step) 
using a bash script like this:

.. literalinclude:: ../../src/AlphaGenome_finetuning/03_submit_test_eval.sh
   :language: bash
   :linenos:
   :caption: src/AlphaGenome_finetuning/03_submit_test_eval.sh