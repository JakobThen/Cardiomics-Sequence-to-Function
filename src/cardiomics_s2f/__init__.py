"""
cardiomics_s2f: Cardiomics Sequence-to-Function
===============================================

This package contains the library code for the cardiomics sequence-to-function pipeline.
Runnable pipelines (SLURM launchers and CLI scripts) live in ``workflows/``.

Subpackages:
------------
* **io**: Functions to annotate genetic variants (VCF-style) and handle data input/output.
* **preprocessing**: Utilities to generate input configurations and preprocess data for AlphaGenome models.
* **finetuning**: Inference and prediction helpers for fine-tuned AlphaGenome models.
* **eval**: Streamed correlation analysis, track prediction, and accumulators for comprehensive model evaluation.
* **vep**: Variant Effect Prediction tools to generate, score, process, and plot in silico variant effects.
* **ism**: Lightweight wrappers around the pretrained AlphaGenome API to add functionality to ATAC, ChIP-TF, and RNA modalities, and run multiple interval ISMs at scale while writing and scaling predictions.
"""
