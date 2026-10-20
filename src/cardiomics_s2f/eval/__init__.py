"""
Evaluation Package
==================

This package provides utilities for calculating correlation metrics, tracking model
predictions, and accumulating validation stats over streaming genomic intervals.

Submodules:
-----------
* `accumulators`: Implements accumulator classes (e.g., for Pearson correlation,
  stranded gene expression) to calculate metrics over streamed batches without keeping
  all predictions in memory.
* `streamed_correlation_analysis`: Orchestrates validation loops to run evaluation and
  write out correlation summary reports.
* `track_correlation`: Provides analysis and plotting scripts to evaluate and visualize
  prediction correlations (beeswarms, heatmap comparisons, scatterplots).
* `track_prediction`: Handles loading predictions from HDF5 and GTF extraction of counts.
"""
