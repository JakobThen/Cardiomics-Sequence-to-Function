"""
Variant Effect Prediction (VEP) Package
=======================================

This package contains utilities to perform In Silico Variant Effect Prediction
(VEP) using fine-tuned AlphaGenome and Borzoi models and to analyse the resulting
scores. The runnable VEP scripts are ``workflows/alphagenome_ft/run_vep.py`` and
``workflows/borzoi/run_vep.py``.

Submodules:
-----------
* `alphagenome_ft`: Chromosome sizes and batched REF/ALT sequence generation for fine-tuned AlphaGenome VEP.
* `borzoi`: Stitched indel dataset, chromosome-end filter, RNA unsquashing and whole-window / masked scores for fine-tuned Borzoi VEP.
* `scoring`: Per-variant scores from aligned REF/ALT predictions (centre window and exon masks).
* `processing`: Utilities to preprocess variant tables and align prediction arrays.
* `plotting`: Visualization scripts to plot VEP scores (heatmaps, density plots, scatters).
* `cohort_plotting`: Descriptive figures of the CHD and control variant cohorts (Richter et al. 2020).
"""
