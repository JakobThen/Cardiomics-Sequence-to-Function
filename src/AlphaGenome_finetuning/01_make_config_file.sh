#!/bin/bash
# ==============================================================================
# Code by Jakob Then
# 
# Purpose: Wrapper script to execute the Python config generator for 
#          AlphaGenome finetuning. It feeds predefined paths and modality
#          lists into the Python parser to generate the final YAML.
# Input:   Hardcoded variables pointing to directories and metadata
# Output:  A formatted .yaml configuration file (e.g., test_config.yaml)
# ==============================================================================

# SLURM Configuration
#SBATCH --job-name=make_config
#SBATCH --output=logs/make_config_%j.log 
#SBATCH --error=logs/make_config_%j.err
#SBATCH --time=01:00:00          
#SBATCH --cpus-per-task=1        # Sequential processing per file
#SBATCH --mem=16G                
#SBATCH --partition=bigmem       # Ensure this matches your specific cluster topology

# Setup core paths and variables
TARGETS_CONFIG_PATH="configs/test_config.yaml"
DATA_DIR="../../input_data/AlphaGenome_Borzoi_input"
NONZERO_MEANS_PATH="../../input_data/bw_nonzero_means.csv"
#folders in DATA_DIR
MODALITIES=(
    "multiome_RNA_3prime_bigwig_deduplicated" 
    "multiome_ATAC_bigwig_deduplicated" 
    "CutnTag_bigwig"
)

# Load conda environments
source /g/easybuild/x86_64/Rocky/8/haswell/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
source env/initiate_alphagenome_ft_env.sh

# Execute the Python config builder
python ../utils/preprocessing/make_AG_input_config.py \
    --data_dir "${DATA_DIR}" \
    --config_prefix "${TARGETS_CONFIG_PATH}" \
    --modality_folders "${MODALITIES[@]}" \
    --norm_tag "_100M" \
    --nonzero_means "${NONZERO_MEANS_PATH}"

conda deactivate