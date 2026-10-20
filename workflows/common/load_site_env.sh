#!/bin/bash
# ==============================================================================
# Shared setup for all workflow launchers and SLURM job scripts (source, don't run).
#
# Provides:
#   CS2F_REPO_ROOT              Repository root (exported, so SLURM jobs inherit it)
#   site settings               From config/site.env, or the file named by
#                               CS2F_SITE_ENV (conda hook, envs, SLURM defaults)
#   cs2f_path <key>             Prints a resolved path from config/paths.yaml
#   cs2f_activate <env>         Activates a conda env incl. its CUDA/library paths
#   cs2f_sbatch <workflow> <cpu|gpu> [sbatch options] <job.slurm> [job args]
#                               Submits with site partition/QoS/constraint and writes
#                               logs to <logs_dir>/<workflow>/
#
# Launchers (run with bash on the login node) locate the repo from this file's
# path. SLURM jobs run a spooled copy of their script, so they rely on
# CS2F_REPO_ROOT, which cs2f_sbatch exports to them.
# ==============================================================================

if [[ -z "${CS2F_REPO_ROOT:-}" ]]; then
    CS2F_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
export CS2F_REPO_ROOT

_cs2f_site_env="${CS2F_SITE_ENV:-${CS2F_REPO_ROOT}/config/site.env}"
if [[ ! -f "$_cs2f_site_env" ]]; then
    echo "Error: ${_cs2f_site_env} not found. Copy config/site.example.env to config/site.env and adapt it (or set CS2F_SITE_ENV)." >&2
    return 1 2>/dev/null || exit 1
fi
set -a
# shellcheck source=/dev/null
source "$_cs2f_site_env"
set +a

# Prints the resolved path for <key> from config/paths.yaml (needs Python with PyYAML).
cs2f_path() {
    "${CS2F_PYTHON:-python3}" "${CS2F_REPO_ROOT}/workflows/common/get_path.py" "$1"
}

# Activates a conda environment (name or prefix) and exposes its CUDA libraries.
cs2f_activate() {
    local env="$1"
    if [[ -z "$env" ]]; then
        echo "Error: no conda environment given (check config/site.env)." >&2
        return 1
    fi
    # shellcheck source=/dev/null
    source "${CS2F_CONDA_SH:?CS2F_CONDA_SH is not set in config/site.env}"
    conda activate "$env"
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export CUDA_HOME="${CONDA_PREFIX}"
}

# Submits a job with the site's SLURM settings and per-workflow log files.
cs2f_sbatch() {
    local workflow="$1" kind="$2"
    shift 2
    local log_dir
    log_dir="$(cs2f_path logs_dir)/${workflow}" || return 1
    mkdir -p "$log_dir"

    local opts=(
        "--output=${log_dir}/%x_%j.out"
        "--error=${log_dir}/%x_%j.err"
        "--export=ALL,CS2F_REPO_ROOT=${CS2F_REPO_ROOT}"
    )
    if [[ "$kind" == "gpu" ]]; then
        [[ -n "${CS2F_SLURM_GPU_PARTITION:-}" ]] && opts+=("--partition=${CS2F_SLURM_GPU_PARTITION}")
        [[ -n "${CS2F_SLURM_GPU_QOS:-}" ]] && opts+=("--qos=${CS2F_SLURM_GPU_QOS}")
        [[ -n "${CS2F_SLURM_GPU_CONSTRAINT:-}" ]] && opts+=("--constraint=${CS2F_SLURM_GPU_CONSTRAINT}")
    else
        [[ -n "${CS2F_SLURM_CPU_PARTITION:-}" ]] && opts+=("--partition=${CS2F_SLURM_CPU_PARTITION}")
    fi
    [[ -n "${CS2F_SLURM_ACCOUNT:-}" ]] && opts+=("--account=${CS2F_SLURM_ACCOUNT}")

    echo "Submitting: sbatch ${opts[*]} $*"
    sbatch "${opts[@]}" "$@"
}
