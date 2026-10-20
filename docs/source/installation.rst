Installation
============

The project consists of the Python package ``cardiomics_s2f`` (``src/``) and the
SLURM pipelines in ``workflows/`` that use it. The pipelines are written for a
SLURM cluster with GPU nodes; the package itself also installs on a laptop.

Requirements
------------

* Python 3.10 or newer
* conda (e.g. Miniforge) for the model environments
* For the pipelines: a SLURM cluster; AlphaGenome fine-tuning, evaluation and
  VEP use 8 GPUs per job

Get the code
------------

.. code-block:: bash

    git clone https://github.com/JakobThen/Cardiomics-Sequence-to-Function.git
    cd Cardiomics-Sequence-to-Function

Environments
------------

Each model family runs in its own conda environment, defined in ``env/``:

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - File
     - Environment
     - Used by
   * - ``env/alphagenome_ft_env.yaml``
     - ``alphagenome_ft``
     - AlphaGenome fine-tuning, test evaluation and VEP (``workflows/alphagenome_ft``)
   * - ``env/chrombpnet_env.yaml``
     - ``chrombpnet``
     - BigWig generation (``workflows/input_prep``)
   * - ``env/alphagenome_env.yaml``
     - ``alphagenome``
     - Pretrained AlphaGenome API (``cardiomics_s2f.ism``)
   * - ``env/borzoi_env.yaml``
     - ``borzoi``
     - Borzoi / gReLU models

.. code-block:: bash

    conda env create -f env/alphagenome_ft_env.yaml

Tell the pipelines which environment to use for which workflow in
``config/site.env`` (see :doc:`configuration`).

Install the package
-------------------

Install ``cardiomics_s2f`` in editable mode inside each environment that runs
package code (at least ``alphagenome_ft``):

.. code-block:: bash

    conda activate alphagenome_ft
    pip install -e ".[alphagenome-ft]"

Optional dependency groups:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Extra
     - Adds
   * - ``alphagenome-ft``
     - ``alphagenome``, ``alphagenome-ft``, ``alphagenome-research``, JAX and torch
       (fine-tuning, test evaluation, VEP)
   * - ``alphagenome``
     - ``alphagenome`` and ``anndata`` (pretrained AlphaGenome API, ``cardiomics_s2f.ism``)
   * - ``torch``
     - torch (``cardiomics_s2f.io.data_input``, streamed correlation analysis)
   * - ``docs``
     - Sphinx and extensions to build this documentation
   * - ``dev``
     - pytest

Without extras, the core install covers data handling, evaluation metrics, VEP
scoring and plotting.

Check the installation
----------------------

.. code-block:: bash

    python -c "import cardiomics_s2f.vep.scoring, cardiomics_s2f.config"
    pip install -e ".[dev]" && pytest

Build the documentation
-----------------------

.. code-block:: bash

    pip install -r docs/requirements.txt
    cd docs
    sphinx-build -a -E -b html source build/html
