#!/bin/bash
# ==============================================================================
# Node-local data staging for AlphaGenome jobs (source, don't run).
#
# cs2f_stage_to_node <targets_config> <genome_fasta> <local_dir>
#   Copies the genome (.fa + index) and every BigWig listed in the targets config
#   to <local_dir> on the compute node, which is much faster to read than the
#   network drive. Writes a config pointing to the staged files and removes
#   <local_dir> when the job exits.
#   Sets: LOCAL_YAML (staged config), LOCAL_FASTA (staged genome)
# ==============================================================================

cs2f_stage_to_node() {
    local targets_config="$1" genome_fasta="$2" local_dir="$3"

    # ALWAYS clean up the local tmp directory when the script exits
    trap 'echo "Cleaning up node-local tmp directory..."; rm -rf "'"${local_dir}"'"; echo "Cleanup complete."' EXIT
    mkdir -p "$local_dir"

    # Stage genome .fa and its .fai index.
    echo "Staging Genome files to ${local_dir}..."
    rsync -a "${genome_fasta}"* "${local_dir}"/
    LOCAL_FASTA="${local_dir}/$(basename "$genome_fasta")"

    # Stage BigWig files and generate a new temporary YAML config that points
    # the model to the local paths instead of the slow network drive.
    echo "Staging BigWig files and generating temp YAML..."
    LOCAL_YAML="${local_dir}/running_config.yaml"
    python "${CS2F_REPO_ROOT}/workflows/common/stage_to_tmp.py" \
        --master_yaml "$targets_config" \
        --tmp_dir "$local_dir" \
        --out_yaml "$LOCAL_YAML"
}
