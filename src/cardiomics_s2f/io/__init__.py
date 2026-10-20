"""
IO Package
==========

This package contains modules and classes for reading genomic data,
handling scaling/transformations (such as Borzoi and AlphaGenome squash formats),
and saving predictions or annotations.

Submodules:
-----------
* `batch_bw_loader`: Contains `BatchBigWigLoader`, a class to load multiple BigWig tracks in parallel.
* `data_input`: Contains mathematical squash operations and data loading helpers.
* `data_output`: Contains utilities to save predictions into formats like annotated HDF5.
* `variant_annotation`: Contains functions to annotate genetic variant dataframes with genomic features.
"""
