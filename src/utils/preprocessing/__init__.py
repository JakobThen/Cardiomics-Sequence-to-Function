"""
Preprocessing Package
=====================

This package contains scripts and utilities for preparing raw sequencing data 
(like ATAC and RNA) into the specific input formats required by AlphaGenome 
and Borzoi models.

It handles:
* BigWig generation
* Cell counting and metadata extraction
* Final configuration `.yaml` construction

CLI Scripts:
------------
* `make_AG_input_config`: CLI script to map directories and generate AlphaGenome YAML configuration files.
"""