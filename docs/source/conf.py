# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import re
import sys

# The docs are built from the source tree with heavy dependencies mocked (see
# autodoc_mock_imports), so the package does not need to be installed.
sys.path.insert(0, os.path.abspath('../../src'))
# Workflow CLI scripts (documented with sphinx-argparse in the pipeline pages)
sys.path.insert(0, os.path.abspath('../../workflows/alphagenome_ft'))
sys.path.insert(0, os.path.abspath('../../workflows/common'))
sys.path.insert(0, os.path.abspath('../../workflows/alphagenome_api'))
sys.path.insert(0, os.path.abspath('../../workflows/borzoi'))
sys.path.insert(0, os.path.abspath('../../workflows/chrombpnet'))


def _package_version():
    # Single source of truth: [project] version in pyproject.toml
    pyproject = os.path.join(os.path.dirname(__file__), '..', '..', 'pyproject.toml')
    with open(pyproject, encoding='utf-8') as f:
        return re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.M).group(1)


project = 'cardiomics-sequence-to-function'
copyright = '2026, Jakob Then'
author = 'Jakob Then'
release = _package_version()
version = release

extensions = [
    'sphinx.ext.autodoc',     # Pulls in documentation from docstrings
    'sphinx.ext.napoleon',    # Supports Google/NumPy style docstrings (makes them readable)
    'sphinx.ext.viewcode',    # Adds [source] links to the API pages
    'myst_nb',                # Renders Jupyter notebooks (.ipynb)
    'sphinxarg.ext',          # Documents argparse CLIs (pipeline pages)
]

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']


# MyST-NB: execute notebooks only if they have no saved outputs
nb_execution_mode = "cache"

templates_path = ['_templates']
exclude_patterns = []

autodoc_mock_imports = [
    "alphagenome",
    "alphagenome_ft",
    "h5py",
    "jax",
    "numpy",
    "pandas",
    "yaml",
    "torch",
    "pytorch_lightning",
    "matplotlib",
    "alphagenome_research",
    "optax",
    "seaborn",
    "pyBigWig",
    "pyranges",
    "ncls",
    "requests",
    "tqdm",
    "scipy",
    "sklearn",
    "mpl_toolkits",
    "mpl_toolkits.axes_grid1",
    "anndata",
    # [analysis] extra
    "statsmodels",
    "adjustText",
    "matplotlib_venn",
    "tangermeme",
    "joblib",
    "pyfaidx",
    # [chrombpnet] extra
    "pysam",
    # [borzoi] extra
    "grelu",
    "einops",
    "psutil",
]
