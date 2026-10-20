#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: SLURM job for step 02. Stages data to the node-local disk and runs
#          finetune_heads.py on 8 GPUs.
# Input:   $1 - Targets config (YAML)
#          $2 - Genome FASTA
#          $3 - Node-local scratch root (a per-job subfolder is created)
#          $4 - XLA autotune cache directory
#          "${@:5}" - further arguments for finetune_heads.py
# ==============================================================================

#SBATCH --job-name=AG_finetune_heads
#SBATCH --nodes=1                   # Single node for 8-GPU training
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=180G
#SBATCH --time=6-00:00:00           # 6-day maximum runtime limit

source "${CS2F_REPO_ROOT:?Submit via workflows/alphagenome_ft/02_finetune_heads.sh}/workflows/common/load_site_env.sh"
source "${CS2F_REPO_ROOT}/workflows/common/stage_to_node.sh"

TARGETS_CONFIG="$1"
GENOME_FASTA="$2"
LOCAL_TMP_DIR="$3/alphagenome_bws_${SLURM_JOB_ID}"
XLA_CACHE_DIR="$4"
shift 4

echo "======================================================"
echo "Starting AlphaGenome Finetuning Job"
echo "======================================================"
echo "Date:           $(date)"
echo "User:           $(whoami)"
echo "Job ID:         ${SLURM_JOB_ID}"
echo "Execution Node: ${SLURMD_NODENAME}"
echo "======================================================"
echo ""

cs2f_activate "$CS2F_ENV_ALPHAGENOME_FT"

# Confirm correct JAX and CUDA setup before we start copying massive files
bash "${CS2F_REPO_ROOT}/env/test_alphagenome_ft_env_config.sh"

# Enable XLA autotune caching. This benchmarks operations on the specific GPU
# architecture to find the fastest execution paths, saving time on restarts.
mkdir -p "$XLA_CACHE_DIR"
export XLA_FLAGS="--xla_gpu_per_fusion_autotune_cache_dir=${XLA_CACHE_DIR}"

# I/O OPTIMIZATION: Node-local Data Staging (sets LOCAL_FASTA and LOCAL_YAML)
cs2f_stage_to_node "$TARGETS_CONFIG" "$GENOME_FASTA" "$LOCAL_TMP_DIR"

echo "Starting model finetuning..."
# Using srun here ensures proper CPU/GPU binding and resource distribution within the allocated SLURM step.
srun python "${CS2F_REPO_ROOT}/workflows/alphagenome_ft/finetune_heads.py" \
    --fasta_path "$LOCAL_FASTA" \
    --config "$LOCAL_YAML" \
    --base_dir "$LOCAL_TMP_DIR" \
    "$@"

conda deactivate

echo ""
echo "======================================================"
echo "Training done! Ending job on: $(date)"
echo "======================================================"
