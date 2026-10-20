#!/bin/bash
# ==============================================================================
# Code by Jakob Then
#
# Purpose: SLURM job for step 03. Stages the BigWig tracks to the node-local
#          disk and runs evaluate_on_test.py on 8 GPUs.
# Input:   $1 - Targets config (YAML)
#          $2 - Genome FASTA
#          $3 - Node-local scratch root (a per-job subfolder is created)
#          $4 - XLA autotune cache directory
#          "${@:5}" - further arguments for evaluate_on_test.py
# ==============================================================================

#SBATCH --job-name=AG_test_eval
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=1-00:00:00

source "${CS2F_REPO_ROOT:?Submit via workflows/alphagenome_ft/03_test_eval.sh}/workflows/common/load_site_env.sh"
source "${CS2F_REPO_ROOT}/workflows/common/stage_to_node.sh"

TARGETS_CONFIG="$1"
GENOME_FASTA="$2"
LOCAL_TMP_DIR="$3/alphagenome_bws_${SLURM_JOB_ID}"
XLA_CACHE_DIR="$4"
shift 4

echo "======================================================"
echo "Starting AlphaGenome Finetuned Evaluation"
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

# Run evaluation (the genome is read from the network copy, as before)
srun python "${CS2F_REPO_ROOT}/workflows/alphagenome_ft/evaluate_on_test.py" \
    --fasta_path "$GENOME_FASTA" \
    --config "$LOCAL_YAML" \
    --input_dir "$LOCAL_TMP_DIR" \
    --base_dir "$LOCAL_TMP_DIR" \
    "$@"

conda deactivate

echo ""
echo "======================================================"
echo "Eval done! Ending job on: $(date)"
echo "======================================================"
