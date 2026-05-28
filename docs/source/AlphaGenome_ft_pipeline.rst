AlphaGenome Finetuning Pipeline
===============================

This section details the shell and SLURM scripts used to execute the AlphaGenome finetuning pipeline. 


Nonzero-mean computation
----------------------------------

Script to compute the nonzero-track means per .bw file used for the AlphaGenome squashed scale 
data transformation.

.. literalinclude:: ../../src/AlphaGenome_finetuning/00_compute_nonzero_means_per_bw.sh
   :language: bash
   :linenos:
   :caption: src/AlphaGenome_finetuning/00_compute_nonzero_means_per_bw.sh


Config file generation
------------------------

Creates a .yaml config file containg the head and track configureation for AlphaGenome finetuning.

.. literalinclude:: ../../src/AlphaGenome_finetuning/01_make_config_file.sh
   :language: bash
   :linenos:
   :caption: src/AlphaGenome_finetuning/01_make_config_file.sh