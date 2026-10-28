Cardiomics Sequence-to-Function
===============================

Tools and pipelines to train, fine-tune and evaluate sequence-to-function models
(AlphaGenome, Borzoi, ChromBPNet) on pseudobulked single-cell multiome (ATAC,
RNA) and bulk Cut&Tag data, and to predict the effects of genetic variants.

The project has two parts:

* ``cardiomics_s2f`` (``src/``): the Python package with the reusable code, i.e.
  data input/output, model evaluation, variant effect prediction and plotting.
* ``workflows/``: SLURM pipelines that prepare model inputs and fine-tune,
  evaluate and apply the models, built on the package.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   configuration

.. toctree::
   :maxdepth: 2
   :caption: Pipelines

   input_prep_pipeline
   AlphaGenome_ft_pipeline
   AlphaGenome_api_pipeline
   Borzoi_pipeline

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/cardiomics_s2f
