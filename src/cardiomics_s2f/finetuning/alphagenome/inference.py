"""
AlphaGenome Multi-GPU Inference
===============================

Shared JAX setup for running a fine-tuned AlphaGenome model data-parallel across
all local GPUs. Used by the test-set evaluation and by variant effect prediction.
"""

import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P


def setup_data_parallel_mesh():
    """Creates a 1D device mesh that shards the batch axis across all local devices.

    Returns:
        tuple:
            - jax.sharding.Mesh: Mesh with a single ``'data'`` axis.
            - PartitionSpec: Sharding for batched inputs (split along ``'data'``).
            - PartitionSpec: Sharding for replicated inputs (parameters, state).
    """
    num_devices = jax.local_device_count()
    print(f"Setting up {num_devices}-GPU Device Mesh...", flush=True)
    mesh = Mesh(mesh_utils.create_device_mesh((num_devices,)), axis_names=('data',))
    data_sharding, replicated_sharding = P('data'), P()
    return mesh, data_sharding, replicated_sharding


def make_parallel_predict(model, head_specs, data_sharding, replicated_sharding):
    """Builds a JIT-compiled forward pass that shards the batch axis across GPUs.

    Args:
        model: Loaded fine-tuned AlphaGenome model (from ``alphagenome_ft.load_checkpoint``).
        head_specs (list): Head specifications; only these heads are returned.
        data_sharding (PartitionSpec): Sharding for batched inputs.
        replicated_sharding (PartitionSpec): Sharding for replicated inputs.

    Returns:
        Callable: ``parallel_predict(params, state, seq_batch, org_batch, mask_batch, strand_idx)``
        returning a dict mapping each head ID to its raw predictions.
    """
    @jax.jit(
        in_shardings=(replicated_sharding, replicated_sharding, data_sharding, data_sharding, data_sharding, replicated_sharding),
        out_shardings=replicated_sharding
    )
    def parallel_predict(params, state, seq_batch, org_batch, mask_batch, strand_idx):
        raw_preds = model._predict(params, state, seq_batch, org_batch, negative_strand_mask=mask_batch, strand_reindexing=strand_idx)
        return {head.head_id: raw_preds[head.head_id] for head in head_specs}

    return parallel_predict
