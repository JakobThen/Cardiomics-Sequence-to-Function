#!/bin/bash
# ==============================================================================
# Code by Jakob Then
# 
# Purpose: Wrapper script to evaluate the track predcition performance of finetune
#          heads in AlphaGenome on test intervals. Stages data to a temporary
#          directory on the GPU node for highly accelerated data loading. Make
#          forward passes on batches of test intervals and simulatanouesly computes
#          correaltion and other evaluation metrics using streaming accumulators to
#          avoid out-of-mem issues and saving 1TB of tmp data.
# Input:   Hardcoded variables pointing to bigwig directories, metadata, 
#          and the input config file plus best finetuning chepoint.
# Output:  Test evaulationmetric files and figures stored in the output directroy.
# ==============================================================================

# SLURM Configuration
#SBATCH -J AG_test_eval        
#SBATCH -N 1                        
#SBATCH -p gpu-training             
#SBATCH --qos=high
#SBATCH --gpus=8                    
#SBATCH --cpus-per-gpu=8       
#SBATCH --mem-per-gpu=64G          
#SBATCH -t 1-00:00:00               
#SBATCH --output=logs/AG_test_eval%j.out
#SBATCH --error=logs/AG_test_eval%j.err

# Setup core paths
DATA_DIR="../../input_data/AlphaGenome_Borzoi_input"
CKPT_DIR="../../models/AlphaGenome_ft" #automatically finds best checkpoint
OUTPUT_DIR="../../results/model_evaluation/AlphaGenome_ft"
INPUT_CONFIG_FILE="configs/simple_merged_deduplicated.yaml"
GENOME_FASTA="../../input_data/metadata/GRCh38_gencode_release29_genome_fasta/genome.fa"
GTF_FILE="../../input_data/metadata/GRCh38_gencode_release29_genome_fasta/genes.gtf"

RESOLUTION=1 #1 (decoder) or 128 (transformer) based on what model embedding  to evaluate
BATCH_SIZE=16

# Dynamically assign a node-local temp directory based on the unique Slurm job ID
LOCAL_TMP_DIR="/tmp/alphagenome_bws_${SLURM_JOB_ID}" #adjust to node tmp dir if needed


echo "======================================================"
echo "Starting AlphaGenome Finetuned Evaluation"
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

#Run processing script
srun python ../utils/finetuning/AlphaGenome/evaluate_AG_on_test_streamed.py \
    --fasta_path "${GENOME_FASTA}" \
    --config "${LOCAL_YAML}" \
    --input_dir "${LOCAL_TMP_DIR}" \
    --out_dir "${OUTPUT_DIR}" \
    --checkpoint_dir "${CKPT_DIR}" \
    --gtf_file "${GTF_FILE}" \
    --fold 1 \
    --model_version "fold_1" \
    --batch_size $BATCH_SIZE \
    --resolution $RESOLUTION \
    --save_predictions False \
    --minimal_test False

# Cleanup
conda deactivate

echo ""
echo "======================================================"
echo "Eval done! Ending job on: $(date)"
echo "======================================================"