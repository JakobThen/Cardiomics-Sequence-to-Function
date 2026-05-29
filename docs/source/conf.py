# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
# Tell Sphinx to look one folder up to find your Python modules
sys.path.insert(0, os.path.abspath('../../src')) 

project = 'cardiomics-sequence-to-function'
copyright = '2026, Jakob Then'
author = 'Jakob Then'
release = '0.01'

extensions = [
    'sphinx.ext.autodoc',     # Pulls in documentation from docstrings
    'sphinx.ext.napoleon',    # Supports Google/NumPy style docstrings (makes them readable)
    'sphinx.ext.viewcode',    # Adds the [source] links you liked
    'myst_nb',                # Parses and executes Jupyter Notebooks (.ipynb)
    'sphinxarg.ext',          # Makes argparse arguments readable 
]

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']


# Configure MyST-NB this setting means notebooks will only be executed if they don't already have outputs saved
nb_execution_mode = "cache"

templates_path = ['_templates']
exclude_patterns = []

#"fake" packages autodoc assumes are import to run my code
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
    "seaborn"
]
