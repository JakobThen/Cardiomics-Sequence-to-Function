# import os
# import sys
# import h5py
# import numpy as np
# import pandas as pd
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

# parent_dir = os.path.abspath("/home/then")
# if parent_dir not in sys.path:
#     sys.path.append(parent_dir)

# from utils.eval.streamed_correlation_analysis import evaluate_from_inference
# from utils.eval.track_prediction import process_track_metadata

# # ---------------------------------------------------------------------------
# # Iterative Saving Class (for small models or low resolution)
# # ---------------------------------------------------------------------------
# class HDF5BatchWriter:
#     def __init__(self, out_dir, file_prefix, total_samples):
#         self.out_dir = Path(out_dir)
#         self.out_dir.mkdir(parents=True, exist_ok=True)
#         self.h5_path = self.out_dir / f"{file_prefix}.h5"
#         self.total_samples = total_samples
#         print(f"Opening HDF5 file for iterative writing at {self.h5_path}...", flush=True)
#         self.file = h5py.File(self.h5_path, 'w')
#         self.datasets = {}

#     def write_batch(self, head_name, batch_preds, start_idx):
#         if head_name not in self.datasets:
#             full_shape = (self.total_samples, *batch_preds.shape[1:])
#             head_grp = self.file.create_group(head_name)
#             self.datasets[head_name] = head_grp.create_dataset(
#                 "predictions", shape=full_shape, dtype=batch_preds.dtype, compression="gzip"
#             )
#         end_idx = start_idx + batch_preds.shape[0]
#         self.datasets[head_name][start_idx:end_idx] = batch_preds
    
#     def write_metadata(self, intervals_df, tracks_df):
#         from pandas.api.types import is_string_dtype, is_object_dtype
#         print("Writing metadata to HDF5...", flush=True)
        
#         # Save intervals
#         interval_grp = self.file.create_group("intervals")
#         for col in intervals_df.columns:
#             data = intervals_df[col].values
#             if is_string_dtype(intervals_df[col]) or is_object_dtype(intervals_df[col]):
#                 data = data.astype(str).astype('S')
#             interval_grp.create_dataset(col, data=data)
            
#         # Save tracks
#         track_grp = self.file.create_group("tracks")
#         for col in tracks_df.columns:
#             data = tracks_df[col].values
#             if is_string_dtype(tracks_df[col]) or is_object_dtype(tracks_df[col]):
#                 data = data.astype(str).astype('S')
#             track_grp.create_dataset(col, data=data)

#     def close(self):
#         self.file.close()
#         print(f"Successfully saved and closed HDF5 file!", flush=True)

# def get_prediction_key(raw_p, bin_size=128, return_scaled=True):
#     prefix = "scaled_predictions" if return_scaled else "predictions"
#     preferred = [f"{prefix}_{bin_size}bp", f"{prefix}_{128 if bin_size == 1 else 1}bp"]
#     for key in preferred:
#         if key in raw_p: return key
#     raise KeyError(f"No matching prediction key found. Tried: {preferred}.")

# def get_aligned_nonzero_means(tracks_df: pd.DataFrame, config_dict: dict) -> np.ndarray:
#     """
#     Extracts nonzero_means from config dictionary and aligns to tracks_df.
#     Args:
#         tracks_df (pd.DataFrame): DataFrame containing a 'path' column.
#         config_dict (dict): Nested dictionary containing 'heads' and 'targets'. 
#     Returns:
#         np.ndarray: A NumPy array of nonzero_means in the same order as tracks_df.
#     """
#     path_to_mean = {}
#     for head in config_dict.get('heads', []):
#         for target in head.get('targets', []):
#             path = target.get('path')
#             mean_val = target.get('nonzero_mean')
#             if path is not None and mean_val is not None:
#                 path_to_mean[path] = mean_val
#     aligned_means = []
#     for path in tracks_df['path']:
#         if path not in path_to_mean:
#             raise KeyError(f"Path from tracks_df not found in dictionary: {path}")
            
#         aligned_means.append(path_to_mean[path])
#     return np.array(aligned_means)

# # ---------------------------------------------------------------------------
# # Setup & Config
# # ---------------------------------------------------------------------------
# print(f"JAX is using: {jax.devices()}")
# if jax.devices()[0].platform == 'cpu':
#     raise RuntimeError("JAX is defaulting to CPU! Please check your CUDA installation.")

# # USER CONFIGURATIONS
# PROJECT_DIR = Path("/g/steinmetz/projects/variant2function_project/src/analysis/AG_ft")
# OUT_DIR = Path("/g/steinmetz/projects/variant2function_project/results/variant.selection/track_prediction/AG_ft")
# FASTA_PATH = Path("/tmp/then_alphagenome_bws/genome.fa") 
# TARGETS_CONFIG_PATH = Path("/tmp/then_alphagenome_bws/tmp_bw_config.yaml")
# CHECKPOINT_DIR = Path("/g/steinmetz/projects/then/AG/heads_only_deduplicated_resumed")
# BW_DIR = Path("/tmp/then_alphagenome_bws/") # Directory containing BigWigs
# GTF_FILE = Path("/g/steinmetz/calfonso/shared/reference_genomes/GRCh38_gencode_release29/annotation/genes.gtf") 

# FOLD = "1"
# WINDOW_SIZE = 1_048_576
# ORGANISM = "HOMO_SAPIENS"
# ORGANISM_IDX = 0  
# MODEL_VERSION = "fold_1"
# BATCH_SIZE = 16 #2 batches per gpu with JAX mesh
# DROP_LAST = True 
# BIN_SIZE = 128 #or use 128 for coarse embedding
# RETURN_SCALED = True 
# MINIMAL_TEST = False

# SAVE_PREDICTIONS = False #for all test intervals at binsize 1 this is 700GB compressed!!
# OUTPUT_DATA_PREFIX = f"AG_fold{FOLD}_{BIN_SIZE}bp_test_eval_{CHECKPOINT_DIR.name}"

# # ---------------------------------------------------------------------------
# # 2. Prepare Data & Metadata
# # ---------------------------------------------------------------------------
# config_dict = load_targets_config(TARGETS_CONFIG_PATH, base_dir=Path("/tmp/then_alphagenome_bws/"))
# head_specs = prepare_head_specs(config_dict)
# register_predefined_heads(head_specs)

# intervals = prepare_intervals_from_fold(fold=FOLD, window_size=WINDOW_SIZE, organism=ORGANISM)
# test_intervals_list = intervals.get('test', [])
# total_test_intervals = len(test_intervals_list)

# if DROP_LAST:
#     total_saved_intervals = (total_test_intervals // BATCH_SIZE) * BATCH_SIZE
# else:
#     total_saved_intervals = total_test_intervals

# # Build exact DataFrame of testing intervals
# intervals_df = pd.DataFrame([
#     {"chrom": iv.chromosome, "start": iv.start, "end": iv.end} 
#     for iv in test_intervals_list[:total_saved_intervals]
# ])

# if MINIMAL_TEST:
#     n = min(BATCH_SIZE * 8, total_saved_intervals)
#     print(f"WARNING: USING MINIMAL TEST WITH ONLY THE FIRST {n} TEST INTERVALS!!!", flush = True)    
#     total_saved_intervals = n
#     test_intervals_list = test_intervals_list[:n]
#     intervals_df = intervals_df.iloc[:n, :]
#     intervals["test"] = test_intervals_list
#     OUT_DIR = OUT_DIR / "MIN_TEST"
#     OUTPUT_DATA_PREFIX = f"MIN_TEST_AG_fold{FOLD}_{BIN_SIZE}bp_test_eval_{CHECKPOINT_DIR.name}"
# print(f"Loaded {total_saved_intervals} test intervals.")


# data_module = BigWigDataModule(
#     intervals=intervals, fasta_path=FASTA_PATH, head_specs=head_specs,
#     batch_size=BATCH_SIZE, shuffle=False, drop_last=DROP_LAST
# ) #Data mdodule is initalized with test, val and train splits as there is an error otherwise but during inference this is split to test only!
# #see batch_generator() in streamed_correlation_analysis.evaluate_from_inference()

# # Combine multi-head tracks into a single ordered DataFrame
# tracks_list = []
# for head in config_dict.get('heads', []):
#     tracks_list.append(pd.DataFrame(head.get('targets', [])))
# tracks_df = pd.concat(tracks_list, ignore_index=True)
# tracks_df = process_track_metadata(tracks_df)
# tracks_df = tracks_df.set_index("id")

# # Scan BW_DIR to create bw_paths_df (matching tracks_df order)
# print("Scanning BigWig directory...")
# bw_files = list(BW_DIR.rglob("*_100M*.bw"))
# bw_paths_dict = {p.name: str(p) for p in bw_files}

# # Build paths array strictly aligned to tracks_df
# aligned_paths = []
# for fname in tracks_df['filename']:
#     if fname not in bw_paths_dict:
#         raise FileNotFoundError(f"Missing BigWig for track: {fname}")
#     aligned_paths.append(bw_paths_dict[fname])

# bw_paths_df = pd.DataFrame({'path': aligned_paths})
# bw_paths_df["nonzero_means"] = get_aligned_nonzero_means(bw_paths_df, config_dict)


# # ---------------------------------------------------------------------------
# # 3. Load Model & Setup Device Mesh
# # ---------------------------------------------------------------------------
# print("Loading model onto GPU...", flush=True)
# model = load_checkpoint(CHECKPOINT_DIR / "best", base_model_version=MODEL_VERSION, init_seq_len=WINDOW_SIZE)
# print("Model loaded successfully.", flush = True)

# organism_enum = getattr(ag_dna_model.Organism, ORGANISM)
# strand_reindexing = jax.device_put(model._metadata[organism_enum].strand_reindexing, model._device_context._device)

# num_devices = jax.local_device_count()
# print(f"Setting up {num_devices}-GPU Device Mesh for evaluation...", flush = True)
# devices = mesh_utils.create_device_mesh((num_devices,))
# mesh = Mesh(devices, axis_names=('data',))
# data_sharding, replicated_sharding = P('data'), P()

# @jax.jit(
#     in_shardings=(replicated_sharding, replicated_sharding, data_sharding, data_sharding, data_sharding, replicated_sharding),
#     out_shardings=replicated_sharding 
# )
# def parallel_predict(params, state, seq_batch, org_batch, mask_batch, strand_idx):
#     raw_preds = model._predict(params, state, seq_batch, org_batch, negative_strand_mask=mask_batch, strand_reindexing=strand_idx)
#     return {head.head_id: raw_preds[head.head_id] for head in head_specs}

# # ---------------------------------------------------------------------------
# # 4. Evaluation Execution
# # ---------------------------------------------------------------------------
# h5_writer = None
# if SAVE_PREDICTIONS:
#     h5_writer = HDF5BatchWriter(OUT_DIR, OUTPUT_DATA_PREFIX, total_saved_intervals)
    

# # Define the forward wrapper that the Streaming Correaltion Pipeline will call
# batch_counter = 0
# def model_forward_fn(batch, start_idx):
#     seq = jnp.asarray(batch['sequences'])
#     org_idx = jnp.full((seq.shape[0],), ORGANISM_IDX, dtype=jnp.int32)
#     neg_mask = jnp.zeros((seq.shape[0],), dtype=jnp.bool_)
    
#     seq_sharded = jax.device_put(seq, data_sharding)
#     org_idx_sharded = jax.device_put(org_idx, data_sharding)
#     neg_mask_sharded = jax.device_put(neg_mask, data_sharding)
    
#     preds = parallel_predict(model._params, model._state, seq_sharded, org_idx_sharded, neg_mask_sharded, strand_reindexing)
    
#     #start_idx = batch_counter * BATCH_SIZE
#     head_arrays = []
    
#     for head_spec in head_specs:
#         head_name = head_spec.head_id
#         raw_p = preds[head_name]
#         res_key = get_prediction_key(raw_p, BIN_SIZE, RETURN_SCALED)
#         p = np.array(raw_p[res_key]) # p is originally (Batch, Bins, Head_Tracks)
#         p = np.transpose(p, (0, 2, 1))  # transpose to (Batch, Head_Tracks, Bins) to match all other scritps
        
#         if SAVE_PREDICTIONS:
#             h5_writer.write_batch(head_name, p, start_idx)
            
#         head_arrays.append(p)

#     #batch_counter += 1
#     # Concatenate across heads to yield shape (Batch, Tracks, Bins)
#     return np.concatenate(head_arrays, axis=1)

# with model._device_context, jax.set_mesh(mesh):
#     print("Starting Streaming Evaluation...", flush=True)
#     evaluate_from_inference(
#         data_module=data_module,
#         model_forward_fn=model_forward_fn,
#         intervals_df=intervals_df,
#         tracks_df=tracks_df,
#         bw_paths_df=bw_paths_df,
#         bin_size=BIN_SIZE,
#         out_dir=OUT_DIR,
#         model_name=f"AG_ft_fold{FOLD}",
#         analysis_name=f"{BIN_SIZE}bp_embedding_test_eval_{CHECKPOINT_DIR.name}",
#         gtf_file=GTF_FILE,
#         batch_size=BATCH_SIZE,
#         is_squashed_scale = RETURN_SCALED
#     )
# print("Eval done :D", flush = True)

# if SAVE_PREDICTIONS:
#     print("Saving predictions...", flush = True)
#     h5_writer.write_metadata(intervals_df, tracks_df)
#     h5_writer.close()