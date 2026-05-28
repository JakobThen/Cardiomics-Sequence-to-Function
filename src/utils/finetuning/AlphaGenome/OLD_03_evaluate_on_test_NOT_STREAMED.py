# import os
# import h5py
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# from pathlib import Path
# from collections import defaultdict
# from pandas.api.types import is_string_dtype, is_object_dtype

# import jax
# import jax.numpy as jnp
# from jax.experimental import mesh_utils
# from jax.sharding import Mesh, PartitionSpec as P

# from alphagenome_ft import load_checkpoint
# from alphagenome_ft.finetune.config import load_targets_config, prepare_head_specs
# from alphagenome_ft.finetune.data import prepare_intervals_from_fold, BigWigDataModule
# from alphagenome_ft.finetune.train import register_predefined_heads
# from alphagenome.models import dna_model as ag_dna_model

# # Script to run a test inference pass on test intervals for fintuned AG models. Saves preds as .h5 file for the correaltion pipeline
# # Code by Jakob Then 

# # ---------------------------------------------------------------------------
# # Iterative Saving Class
# # ---------------------------------------------------------------------------
# class HDF5BatchWriter:
#     """
#     Manages an open HDF5 file to incrementally write batches of predictions to disk,
#     keeping CPU RAM usage minimal.
#     """
#     def __init__(self, out_dir, file_prefix, total_samples):
#         self.out_dir = Path(out_dir)
#         self.out_dir.mkdir(parents=True, exist_ok=True)
#         self.h5_path = self.out_dir / f"{file_prefix}.h5"
#         self.total_samples = total_samples
        
#         # Open the file in write mode
#         print(f"Opening HDF5 file for iterative writing at {self.h5_path}...", flush=True)
#         self.file = h5py.File(self.h5_path, 'w')
#         self.datasets = {}

#     def write_batch(self, head_name, batch_preds, start_idx):
#         """Writes a single batch of predictions to the HDF5 dataset."""
#         # Lazy initialization: Create the dataset on the first batch when shape is known
#         if head_name not in self.datasets:
#             feature_shape = batch_preds.shape[1:]
#             full_shape = (self.total_samples, *feature_shape)
            
#             head_grp = self.file.create_group(head_name)
#             self.datasets[head_name] = head_grp.create_dataset(
#                 "predictions", 
#                 shape=full_shape, 
#                 dtype=batch_preds.dtype,
#                 compression="gzip" # "lzf" is much faster than gzip but makes it only radable in python and nolonger R
#             )
        
#         # Calculate end index (handles cases where the last batch is smaller)
#         end_idx = start_idx + batch_preds.shape[0]
        
#         # Write directly to disk
#         self.datasets[head_name][start_idx:end_idx] = batch_preds

#     def write_metadata(self, intervals_df=None, tracks_df_dict=None):
#         """Writes all track and interval metadata to the HDF5 file after loop finishes."""
#         print("Writing metadata to HDF5...", flush=True)
#         if intervals_df is not None:
#             interval_grp = self.file.create_group("intervals")
#             for col in intervals_df.columns:
#                 data = intervals_df[col].values
#                 if is_string_dtype(intervals_df[col]) or is_object_dtype(intervals_df[col]):
#                     data = data.astype(str).astype('S')
#                 interval_grp.create_dataset(col, data=data)
        
#         if tracks_df_dict is not None:
#             for head_name, tracks_df in tracks_df_dict.items():
#                 if head_name in self.file:
#                     head_grp = self.file[head_name]
#                     track_grp = head_grp.create_group("tracks")
#                     for col in tracks_df.columns:
#                         data = tracks_df[col].values
#                         if is_string_dtype(tracks_df[col]) or is_object_dtype(tracks_df[col]):
#                             data = data.astype(str).astype('S')
#                         track_grp.create_dataset(col, data=data)

#     def close(self):
#         """Closes the HDF5 file pointer."""
#         self.file.close()
#         print(f"Successfully saved and closed HDF5 file!", flush=True)

# def get_prediction_key(raw_p, bin_size=128, return_scaled=True):
#     prefix = "scaled_predictions" if return_scaled else "predictions"
#     preferred = [f"{prefix}_{bin_size}bp"]
#     fallback_bin = 128 if bin_size == 1 else 1
#     preferred.append(f"{prefix}_{fallback_bin}bp")
#     for key in preferred:
#         if key in raw_p:
#             return key
#     available = list(raw_p.keys())
#     raise KeyError(
#         f"No matching prediction key found. "
#         f"Tried: {preferred}. "
#         f"Available keys: {available}"
#     )

# # ---------------------------------------------------------------------------
# # 1. Setup & GPU Validation
# # ---------------------------------------------------------------------------
# print(f"JAX is using: {jax.devices()}")
# if jax.devices()[0].platform == 'cpu':
#     raise RuntimeError("JAX is defaulting to CPU! Please check your CUDA installation and GPU allocation.")

# PROJECT_DIR = Path("/g/steinmetz/projects/variant2function_project/src/analysis/AG_ft")

# FASTA_PATH = Path("/tmp/then_alphagenome_bws/genome.fa") 
# TARGETS_CONFIG_PATH = Path("/tmp/then_alphagenome_bws/tmp_bw_config.yaml")
# CHECKPOINT_DIR = Path("/g/steinmetz/projects/then/AG/heads_only_deduplicated_resumed")

# FOLD = "1"
# WINDOW_SIZE = 1_048_576
# ORGANISM = "HOMO_SAPIENS"
# ORGANISM_IDX = 0  
# MODEL_VERSION = "fold_1"
# BATCH_SIZE = 16
# DROP_LAST = True 
# BIN_SIZE = 1 
# RETURN_SCALED = True 

# OUTPUT_DATA_PREFIX = f"AG_fold{FOLD}_test_eval_{CHECKPOINT_DIR.name}"

# # ---------------------------------------------------------------------------
# # 2. Config & Dataloader Setup
# # ---------------------------------------------------------------------------
# config_dict = load_targets_config(TARGETS_CONFIG_PATH, base_dir= Path("/tmp/then_alphagenome_bws/"))
# head_specs = prepare_head_specs(config_dict)
# register_predefined_heads(head_specs)

# intervals = prepare_intervals_from_fold(fold=FOLD, window_size=WINDOW_SIZE, organism=ORGANISM)
# test_intervals_list = intervals.get('test', [])
# total_test_intervals = len(test_intervals_list)
# print(f"Loaded {total_test_intervals} test intervals.")

# # Calculate exact number of intervals that will be saved based on DROP_LAST
# if DROP_LAST:
#     total_saved_intervals = (total_test_intervals // BATCH_SIZE) * BATCH_SIZE
# else:
#     total_saved_intervals = total_test_intervals

# data_module = BigWigDataModule(
#     intervals=intervals, 
#     fasta_path=FASTA_PATH,
#     head_specs=head_specs,
#     batch_size=BATCH_SIZE,
#     shuffle=False,  
#     drop_last=DROP_LAST
# )

# tracks_metadata = {}
# for head in config_dict.get('heads', []):
#     tracks_metadata[head['id']] = pd.DataFrame(head.get('targets', []))

# # ---------------------------------------------------------------------------
# # 3. Load Model & Setup Device Mesh
# # ---------------------------------------------------------------------------
# print("Loading model onto GPU...", flush = True)
# model = load_checkpoint(
#     CHECKPOINT_DIR / "best",
#     base_model_version=MODEL_VERSION,
#     init_seq_len=WINDOW_SIZE
# )
# print("Model loaded successfully.", flush = True)

# organism_enum = getattr(ag_dna_model.Organism, ORGANISM)
# strand_reindexing = jax.device_put(
#     model._metadata[organism_enum].strand_reindexing,
#     model._device_context._device,
# )

# num_devices = jax.local_device_count()
# print(f"Setting up {num_devices}-GPU Device Mesh for evaluation...", flush = True)
# devices = mesh_utils.create_device_mesh((num_devices,))
# mesh = Mesh(devices, axis_names=('data',))

# data_sharding = P('data')
# replicated_sharding = P()

# @jax.jit(
#     in_shardings=(
#         replicated_sharding, replicated_sharding, data_sharding, 
#         data_sharding, data_sharding, replicated_sharding
#     ),
#     out_shardings=replicated_sharding 
# )
# def parallel_predict(params, state, seq_batch, org_batch, mask_batch, strand_idx):
#     raw_preds = model._predict(
#         params, state, seq_batch, org_batch,
#         negative_strand_mask=mask_batch,
#         strand_reindexing=strand_idx
#     )
#     return {head.head_id: raw_preds[head.head_id] for head in head_specs}

# # ---------------------------------------------------------------------------
# # 4. Evaluation & Iterative Writing Loop
# # ---------------------------------------------------------------------------
# # Initialize the HDF5 writer
# h5_writer = HDF5BatchWriter(
#     out_dir=PROJECT_DIR, 
#     file_prefix=OUTPUT_DATA_PREFIX, 
#     total_samples=total_saved_intervals
# )

# print("Starting evaluation...", flush = True)

# with model._device_context, jax.set_mesh(mesh):
#     for batch_idx, batch in enumerate(data_module.iter_batches(split="test")):
        
#         seq = jnp.asarray(batch['sequences'])
#         org_idx = jnp.full((seq.shape[0],), ORGANISM_IDX, dtype=jnp.int32)
#         neg_mask = jnp.zeros((seq.shape[0],), dtype=jnp.bool_)
        
#         seq_sharded = jax.device_put(seq, data_sharding)
#         org_idx_sharded = jax.device_put(org_idx, data_sharding)
#         neg_mask_sharded = jax.device_put(neg_mask, data_sharding)
        
#         preds = parallel_predict(
#             model._params, model._state, seq_sharded, 
#             org_idx_sharded, neg_mask_sharded, strand_reindexing
#         )
        
#         # Calculate the starting index in the final HDF5 array for this batch
#         start_idx = batch_idx * BATCH_SIZE

#         for head_spec in head_specs:
#             head_name = head_spec.head_id
#             raw_p = preds[head_name]

#             if batch_idx == 0:
#                 print(f"Head {head_name} pred , type(raw_p) = {type(raw_p)}", flush = True)

#             # Get correct resolution embedding key and convert to NumPy
#             resolution_key = get_prediction_key(raw_p, bin_size=BIN_SIZE, return_scaled=RETURN_SCALED)
#             p = np.array(raw_p[resolution_key]) 
            
#             # Write directly to disk buffer instead of RAM list
#             h5_writer.write_batch(head_name, p, start_idx)

#         if (batch_idx + 1) % 10 == 0:
#             print(f"Processed {batch_idx + 1} batches...", flush = True)

# # ---------------------------------------------------------------------------
# # 5. Save Metadata & Cleanup
# # ---------------------------------------------------------------------------
# # Slice intervals to match exact count saved (handles DROP_LAST condition)
# intervals_df = pd.DataFrame([
#     {"chromosome": iv.chromosome, "start": iv.start, "end": iv.end} 
#     for iv in test_intervals_list[:total_saved_intervals]
# ])

# # Finalize the HDF5 file
# h5_writer.write_metadata(
#     intervals_df=intervals_df,
#     tracks_df_dict=tracks_metadata
# )
# h5_writer.close()