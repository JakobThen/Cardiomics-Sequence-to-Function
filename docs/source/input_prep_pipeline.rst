Model Input Generation Pipeline
================================

Pipeline of bash scripts to generate base resolution, normlised bigwig files for model training and finetuning.

1. ATAC inputs
----------------------------------

Takes pseudobulked ATAC .bam files and sets up a SLURM array job to 
compute base-resolution ATAC bigwigs (+4/-4 shift) for AlphaGenome/Borzoi.
Reads are normalized to 100M counts per .bw file.
Optionally filters pseudobulks based on a minimum cell count.
Assumes DEDUPLICATED .bam files.
Input:   BAM_DIR and (optional) CELL_NUMBER_FILE metadata.
Output:  Generates job configs and submits make_ATAC_bw.slurm array jobs.

.. literalinclude:: ../../src/model_input_prep/01_submit_ATAC_BigWig_prep.sh
   :language: bash
   :linenos:
   :caption: src/model_input_prep/01_submit_ATAC_BigWig_prep.sh


2. RNA inputs
---------------------

Takes pseudobulked scRNA .bam files and sets up a SLURM array job to 
compute base-resolution bigwigs for AlphaGenome/Borzoi.
Reads are split by strand and both strands sum to 100M counts per pseudobulk.   
Optionally filters pseudobulks based on a minimum cell count.
Assumes DEDUPLICATED .bam files.
Input:   BAM_DIR and (optional) CELL_NUMBER_FILE metadata.
Output:  Generates job configs and submits make_RNA_3prime_bw.slurm array jobs.

.. literalinclude:: ../../src/model_input_prep/02_submit_RNA_BigWig_prep.sh
   :language: bash
   :linenos:
   :caption: src/model_input_prep/02_submit_RNA_BigWig_prep.sh


3. CutnTag inputs
---------------------

Takes bulk Cut&Tag .bam files and sets up a SLURM array job to 
compute base-resolution bigwigs for AlphaGenome/Borzoi. 
Leverages chromBPNet derived pipeline and transfers it to Cut&Tag.
Reads are normalized to 100M counts per .bw file.
Input:   INPUT_SOURCE can be either:

         * A directory path (will scan for all .bam files)
         * A text file containing manually curated full paths to .bam files

Output:  Generates job configs and submits make_CutnTag_bw.slurm array jobs.

.. literalinclude:: ../../src/model_input_prep/03_submit_CutnTag_BigWig_prep.sh
   :language: bash
   :linenos:
   :caption: src/model_input_prep/03_submit_CutnTag_BigWig_prep.sh