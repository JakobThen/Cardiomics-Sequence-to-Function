"""
Cardiomics Sequence-to-Function Utilities
=========================================

This package contains various utility subpackages for the cardiomics sequence-to-function pipeline.

Subpackages:
------------
* **io**: Functions to annotate genetic variants (VCF-style) and handle data input/output.
* **preprocessing**: Utilities to generate input configurations and preprocess data for AlphaGenome models.
* **finetuning**: Scripts and utilities for fine-tuning AlphaGenome models, including head fine-tuning and evaluation.
* **eval**: Streamed correlation analysis, track prediction, and accumulators for comprehensive model evaluation.
* **VEP**: Variant Effect Prediction tools and scripts to process, analyze, and plot in silico variant effects.
* **ISM**: Lightweight wrappers around the pretrained AlphaGenome API to add functionality to ATAC, ChIP-TF, and RNA modalities, and run multiple interval ISMs at scale while writing and scaling predictions.
"""
