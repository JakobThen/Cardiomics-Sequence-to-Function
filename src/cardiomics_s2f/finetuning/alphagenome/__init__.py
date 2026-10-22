"""
AlphaGenome Finetuning Module
=============================

Library code shared by the AlphaGenome fine-tuning, test evaluation and VEP
workflows (the runnable scripts live in ``workflows/alphagenome_ft/``).

Submodules:
-----------
* `inference`: Multi-GPU device mesh and JIT-compiled data-parallel forward pass.
* `training`: Data-parallel multi-GPU training loop (adapted from alphagenome-ft 0.1.5, MIT).
* `predictions`: Selection of prediction outputs and alignment of per-track non-zero means.
* `attributions`: DeepSHAP-style nucleotide attributions and their local-activity normalisation.
"""
