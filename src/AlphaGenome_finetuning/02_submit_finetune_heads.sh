#!/bin/bash
# ==============================================================================
# Code by Jakob Then
# 
# Purpose: Wrapper script to finetune heads in AlphaGenome. Stages data to a temporary 
#          directory on the GPU node for highly accelerated data loading.
# Input:   Hardcoded variables pointing to bigwig directories, metadata, 
#          and the input config file.
# Output:  Checkpoints for the best and last model epoch saved to OUTPUT_DIR.
# ==============================================================================

# SLURM Configuration
#SBATCH -J AG_finetune_heads        
#SBATCH -N 1                        # Single node for 8-GPU training
#SBATCH -p gpu-training             
#SBATCH --qos=high
#SBATCH --gpus=8                    
#SBATCH --cpus-per-gpu=8       
#SBATCH --mem-per-gpu=180G          
#SBATCH -C ("gpu=B200"|"gpu=H200"|"gpu=H100"|"gpu=A100")  
#SBATCH -t 6-00:00:00               # 6-day maximum runtime limit 
#SBATCH --output=logs/finetune_heads_%j.out
#SBATCH --error=logs/finetune_heads_%j.err

# Setup core paths and training hyperparameters
DATA_DIR="../../input_data/AlphaGenome_Borzoi_input"
OUTPUT_DIR="../../models/AlphaGenome_ft"
INPUT_CONFIG_FILE="configs/simple_merged_deduplicated.yaml"
GENOME_FASTA="../../input_data/metadata/GRCh38_gencode_release29_genome_fasta/genome.fa"

# Dynamically assign a node-local temp directory based on the unique Slurm job ID
LOCAL_TMP_DIR="/tmp/alphagenome_bws_${SLURM_JOB_ID}" #adjust to node tmp dir if needed

BATCH_SIZE=16  # Represents 2 intervals per GPU. Adjust if out-of-memory errors occur, but fine for H100s. Must be a multiple of n_GPUS
MAX_EPOCHS=10 

echo "======================================================"
echo "Starting AlphaGenome Finetuning Job"
echo "======================================================"
echo "Date:           $(date)"
echo "User:           $(whoami)"
echo "Job ID:         ${SLURM_JOB_ID}"
echo "Execution Node: ${SLURMD_NODENAME}"
echo "======================================================"
echo ""

# Load Conda environment
source /g/easybuild/x86_64/Rocky/8/haswell/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
source /g/steinmetz/projects/variant2function_project/env/alphagenome_ft/initiate_alphagenome_ft_env.sh
#source ../../env/initiate_alphagenome_ft_env.sh

# Confirm correct JAX and CUDA setup before we start copying massive files
bash ../../env/test_alphagenome_ft_env_config.sh

# Enable XLA autotune caching. This benchmarks operations on the specific GPU 
# architecture to find the fastest execution paths, saving time on restarts.
export XLA_FLAGS="--xla_gpu_per_fusion_autotune_cache_dir=xla_cache"

# ==============================================================================
# I/O OPTIMIZATION: Node-local Data Staging
# ==============================================================================

# ALWAYS clean up the local /tmp directory when the script exits
trap 'echo "Cleaning up node-local tmp directory..."; rm -rf "${LOCAL_TMP_DIR}"; echo "Cleanup complete."' EXIT

mkdir -p "${LOCAL_TMP_DIR}"

# Stage genome .fa and its .fai index.
echo "Staging Genome files to ${LOCAL_TMP_DIR}..."
rsync -a "${GENOME_FASTA}"* "${LOCAL_TMP_DIR}"/

# Stage BigWig files and generate a new temporary YAML config that points
# the model to the local /tmp/ paths instead of the slow network drive.
echo "Staging BigWig files and generating temp YAML..."
LOCAL_YAML="${LOCAL_TMP_DIR}/running_config.yaml"
python ../utils/finetuning/AlphaGenome/stage_to_tmp.py \
    --master_yaml "${INPUT_CONFIG_FILE}" \
    --tmp_dir "${LOCAL_TMP_DIR}" \
    --out_yaml "${LOCAL_YAML}"

# ==============================================================================
# MODEL TRAINING
# ==============================================================================

echo "Starting model finetuning..."
# Using srun here ensures proper CPU/GPU binding and resource distribution within the allocated SLURM step.
srun python ../utils/finetuning/AlphaGenome/finetune_heads.py \
    --fasta_path "${LOCAL_TMP_DIR}/genome.fa" \
    --config "${LOCAL_YAML}" \
    --base_dir "${LOCAL_TMP_DIR}" \
    --checkpoint_dir "${OUTPUT_DIR}" \
    --batch_size $BATCH_SIZE \
    --epochs $MAX_EPOCHS \
    --lr 3e-4

# Cleanup
conda deactivate

echo ""
echo "======================================================"
echo "Training done! Ending job on: $(date)"
echo "======================================================"