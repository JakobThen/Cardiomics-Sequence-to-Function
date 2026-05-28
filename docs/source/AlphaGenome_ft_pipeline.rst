AlphaGenome Finetuning Pipeline
===============================

CLI-input based python script and shell wrappers to build and run an AlphaGenome finetuning pipeline
and evaluate the model outputs on test intervals.  

0. Nonzero-mean computation
----------------------------------
Assuming we already generated base resolution bigwig fiel sfor all modalities.
Before starign we need to compute the nonezero track mean for proper input scaling.
THis is handled directly in bash. 
Computes the mean of non-zero track values for BigWig (.bw) files, 
specifically restricted to canonical chromosomes (chr1-22, X, Y). 
* Input:   $1 - Directory containing input .bw files
* Output:  $2 - CSV file capturing basename, non-zero mean, and full path

.. literalinclude:: ../../src/AlphaGenome_finetuning/00_compute_nonzero_means_per_bw.sh
   :language: bash
   :linenos:
   :caption: src/AlphaGenome_finetuning/00_compute_nonzero_means_per_bw.sh


1. Config file generation
-----------------------------
Now we create a .ymal config file specifying the tracks and heads to finetune.
Python script:
.. argparse::
   :module: utils/preprocessing/make_AG_input_config.py
   :func: get_parser
   :prog: make_AG_input_config.py


Alternatiely we can submit using a bash wrapper execute the Python config generator.
* Input:   Hardcoded variables pointing to directories and metadata
* Output:  A formatted .yaml configuration file (e.g., test_config.yaml)

.. literalinclude:: ../../src/AlphaGenome_finetuning/01_make_config_file.sh
   :language: bash
   :linenos:
   :caption: src/AlphaGenome_finetuning/01_make_config_file.sh


3. Finetuning evaluation
----------------------------------
.. argparse::
   :module: utils.finetuning.AlphaGenome.evaluate_AG_on_test_streamed
   :func: get_parser
   :prog: evaluate_AG_on_test_streamed.py