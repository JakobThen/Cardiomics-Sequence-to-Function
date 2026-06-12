"""
AlphaGenome Finetuning Module
=============================

This module provides scripts and classes for staging datasets, running heads-only or
joint finetuning, resuming interrupted training sessions, and performing post-finetuning
evaluation on test intervals.

CLI Scripts:
------------
* `evaluate_AG_on_test_streamed`: Runs streaming test metrics evaluation using a checkpoint.
* `finetune_heads`: Finetunes new heads on top of pretrained AlphaGenome models.
* `resume_finetune_heads`: Resumes a training run from a PyTorch Lightning checkpoint.
* `stage_to_tmp`: Staged datasets and configuration YAMLs into node-local temporary SSDs.
"""
