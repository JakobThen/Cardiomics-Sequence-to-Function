"""Evaluates a fine-tuned AlphaGenome model using streaming correlation analysis (workflow step 03).

It loads a specified checkpoint, performs multi-GPU distributed inference across 
test intervals, computes metrics against BigWig tracks, and optionally saves 
the predictions to an HDF5 file.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import jax
import jax.numpy as jnp

from alphagenome_ft import load_checkpoint
from alphagenome_ft.finetune.config import load_targets_config, prepare_head_specs
from alphagenome_ft.finetune.data import prepare_intervals_from_fold, BigWigDataModule
from alphagenome_ft.finetune.train import register_predefined_heads
from alphagenome.models import dna_model as ag_dna_model

from cardiomics_s2f.eval.streamed_correlation_analysis import evaluate_from_inference
from cardiomics_s2f.eval.track_prediction import process_track_metadata
from cardiomics_s2f.finetuning.alphagenome.inference import make_parallel_predict, setup_data_parallel_mesh
from cardiomics_s2f.finetuning.alphagenome.predictions import get_aligned_nonzero_means, get_prediction_key
from cardiomics_s2f.io.data_output import HDF5BatchWriter

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def get_parser():
    """Creates and configures the argument parser for the evaluation script.
    
    Separated from the main execution logic to allow Sphinx's argparse extension
    (`sphinxarg.ext`) to auto-generate CLI documentation natively.

    Returns:
        argparse.ArgumentParser: The configured argument parser object.
    """
    parser = argparse.ArgumentParser(description="AlphaGenome Finetuned Streaming Evaluation Script")

    # Core I/O Arguments
    parser.add_argument("--fasta_path", type=str, required=True, help="Path to the genome.fa file") 
    parser.add_argument("--config", type=str, required=True, help="Path to the target YAML config file") 
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input BigWig files") 
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for figures and results") 
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory containing the best model checkpoint") 
    parser.add_argument("--gtf_file", type=str, default=None, help="Optional GTF file for Gene-based correlation mapping") 
    parser.add_argument("--base_dir", type=str, default=None,
                        help="Base directory for resolving relative paths in the config (defaults to the config's folder)")
    
    # Data & Model Options
    parser.add_argument("--fold", type=str, default="1", help="Data fold to use (0, 1, 2, 3)")
    parser.add_argument("--window_size", type=int, default=1_048_576, help="Window size in bp (must map to an AlphaGenome config)")
    parser.add_argument("--organism", type=str, default="HOMO_SAPIENS", help="Target organism")
    parser.add_argument("--model_version", type=str, default="fold_1", help="Base model version (e.g., fold_1, all_folds)")
    parser.add_argument("--batch_size", type=int, default=16, help="Global batch size (must be divisible by N_GPUs)")

    # Evaluation Options
    parser.add_argument("--resolution", type=int, default=1, choices=[1, 128], help="Embedding resolution to evaluate (1, 128)")
    parser.add_argument("--save_predictions", action="store_true", help="Save test interval predictions to HDF5.")
    parser.add_argument("--minimal_test", action="store_true", help="Truncate run to the first 8 batches for quick debugging.")

    return parser

def main(args):
    """Executes the main evaluation pipeline.

    Sets up hardware, prepares data intervals and metadata, loads the model
    and its sharding configuration, and executes multi-GPU streaming evaluation.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Raises:
        RuntimeError: If JAX defaults to CPU execution.
        FileNotFoundError: If a required BigWig file is missing from the directory.
    """
    # ---------------------------------------------------------------------------
    # 1. Setup & Hardware Config
    # ---------------------------------------------------------------------------
    print(f"JAX is using: {jax.devices()}")
    if jax.devices()[0].platform == 'cpu':
        raise RuntimeError("JAX is defaulting to CPU! Please check your CUDA installation.")
        
    # Standardize input variables
    FASTA_PATH = Path(args.fasta_path)
    TARGETS_CONFIG_PATH = Path(args.config)
    BW_DIR = Path(args.input_dir)
    CHECKPOINT_DIR = Path(args.checkpoint_dir)
    OUT_DIR = Path(args.out_dir)
    GTF_FILE = Path(args.gtf_file) if args.gtf_file else None
    BASE_DIR = Path(args.base_dir) if args.base_dir else TARGETS_CONFIG_PATH.parent

    FOLD = args.fold
    MODEL_VERSION = args.model_version
    ORGANISM = args.organism
    ORGANISM_IDX = 0 if ORGANISM == "HOMO_SAPIENS" else 1
    
    WINDOW_SIZE = args.window_size
    BIN_SIZE = args.resolution
    MIN_TEST = args.minimal_test
    SAVE_PREDICTIONS = args.save_predictions
    BATCH_SIZE = args.batch_size # Extracted from args to resolve missing variable

    DROP_LAST = True 
    RETURN_SCALED = True 
    
    OUTPUT_DATA_PREFIX = f"AG_fold{FOLD}_{BIN_SIZE}bp_test_eval_{CHECKPOINT_DIR.name}"

    # ---------------------------------------------------------------------------
    # 2. Prepare Data & Metadata
    # ---------------------------------------------------------------------------
    config_dict = load_targets_config(TARGETS_CONFIG_PATH, base_dir=BASE_DIR)
    head_specs = prepare_head_specs(config_dict)
    register_predefined_heads(head_specs)

    intervals = prepare_intervals_from_fold(fold=FOLD, window_size=WINDOW_SIZE, organism=ORGANISM)
    test_intervals_list = intervals.get('test', [])
    total_test_intervals = len(test_intervals_list)

    # Calculate exact interval boundaries to prevent batch shape mismatches if DROP_LAST is active
    if DROP_LAST:
        total_saved_intervals = (total_test_intervals // BATCH_SIZE) * BATCH_SIZE
    else:
        total_saved_intervals = total_test_intervals

    # Build the tracking DataFrame for intervals
    intervals_df = pd.DataFrame([
        {"chrom": iv.chromosome, "start": iv.start, "end": iv.end} 
        for iv in test_intervals_list[:total_saved_intervals]
    ])

    # Minimal test flag sets 2 batches per GPU for rapid prototyping
    if MIN_TEST:
        n = min(BATCH_SIZE * 8, total_saved_intervals)
        print(f"WARNING: USING MINIMAL TEST WITH ONLY THE FIRST {n} TEST INTERVALS!!!", flush=True)    
        total_saved_intervals = n
        test_intervals_list = test_intervals_list[:n]
        intervals_df = intervals_df.iloc[:n, :]
        intervals["test"] = test_intervals_list
        OUT_DIR = OUT_DIR / "MIN_TEST"
        OUTPUT_DATA_PREFIX = f"MIN_TEST_AG_fold{FOLD}_{BIN_SIZE}bp_test_eval_{CHECKPOINT_DIR.name}"
        
    print(f"Loaded {total_saved_intervals} test intervals.")

    # Initialize data module (creates train/val/test splits, but only test is inferred)
    data_module = BigWigDataModule(
        intervals=intervals, fasta_path=FASTA_PATH, head_specs=head_specs,
        batch_size=BATCH_SIZE, shuffle=False, drop_last=DROP_LAST
    )

    # Collate track metadata across all specified heads
    tracks_list = []
    for head in config_dict.get('heads', []):
        tracks_list.append(pd.DataFrame(head.get('targets', [])))
        
    tracks_df = pd.concat(tracks_list, ignore_index=True)
    tracks_df = process_track_metadata(tracks_df)
    tracks_df = tracks_df.set_index("id")

    # Map BigWig paths and ensure strict ordering alignment with tracks_df
    print("Scanning BigWig directory...")
    bw_files = list(BW_DIR.rglob("*.bw"))
    bw_paths_dict = {p.name: str(p) for p in bw_files}

    aligned_paths = []
    for fname in tracks_df['filename']:
        if fname not in bw_paths_dict:
            raise FileNotFoundError(f"Missing BigWig for track: {fname}")
        aligned_paths.append(bw_paths_dict[fname])

    bw_paths_df = pd.DataFrame({'path': aligned_paths})
    bw_paths_df["nonzero_means"] = get_aligned_nonzero_means(bw_paths_df, config_dict)

    # ---------------------------------------------------------------------------
    # 3. Load Model & Setup JAX Device Mesh
    # ---------------------------------------------------------------------------
    print("Loading model onto GPU...", flush=True)
    model = load_checkpoint(CHECKPOINT_DIR / "best", base_model_version=MODEL_VERSION, init_seq_len=WINDOW_SIZE)
    print("Model loaded successfully.", flush=True)

    # Extract strand reindexing array and pin it to the correct device
    organism_enum = getattr(ag_dna_model.Organism, ORGANISM)
    strand_reindexing = jax.device_put(model._metadata[organism_enum].strand_reindexing, model._device_context._device)

    # Configure multi-GPU sharding layout and JIT-compile the data-parallel forward pass
    mesh, data_sharding, replicated_sharding = setup_data_parallel_mesh()
    parallel_predict = make_parallel_predict(model, head_specs, data_sharding, replicated_sharding)

    # ---------------------------------------------------------------------------
    # 4. Evaluation Execution
    # ---------------------------------------------------------------------------
    h5_writer = None
    if SAVE_PREDICTIONS:
        h5_writer = HDF5BatchWriter(OUT_DIR, OUTPUT_DATA_PREFIX, total_saved_intervals)

    # Define the forward wrapper function expected by the evaluation pipeline
    def model_forward_fn(batch, start_idx):
        # Format and push inputs to the correct device/sharding spec
        seq = jnp.asarray(batch['sequences'])
        org_idx = jnp.full((seq.shape[0],), ORGANISM_IDX, dtype=jnp.int32)
        neg_mask = jnp.zeros((seq.shape[0],), dtype=jnp.bool_)
        
        seq_sharded = jax.device_put(seq, data_sharding)
        org_idx_sharded = jax.device_put(org_idx, data_sharding)
        neg_mask_sharded = jax.device_put(neg_mask, data_sharding)
        
        preds = parallel_predict(model._params, model._state, seq_sharded, org_idx_sharded, neg_mask_sharded, strand_reindexing)
        
        head_arrays = []
        for head_spec in head_specs:
            head_name = head_spec.head_id
            raw_p = preds[head_name]
            res_key = get_prediction_key(raw_p, BIN_SIZE, RETURN_SCALED)
            
            # Extract and transpose from (Batch, Bins, Head_Tracks) -> (Batch, Head_Tracks, Bins)
            p = np.array(raw_p[res_key]) 
            p = np.transpose(p, (0, 2, 1))  
            
            if SAVE_PREDICTIONS:
                h5_writer.write_batch(head_name, p, start_idx)
                
            head_arrays.append(p)

        # Stitch all head outputs together into a single master array
        return np.concatenate(head_arrays, axis=1)

    # Execute the streaming evaluation inside the specific JAX context limits
    with model._device_context, jax.set_mesh(mesh):
        print("Starting Streaming Evaluation...", flush=True)
        evaluate_from_inference(
            data_module=data_module,
            model_forward_fn=model_forward_fn,
            intervals_df=intervals_df,
            tracks_df=tracks_df,
            bw_paths_df=bw_paths_df,
            bin_size=BIN_SIZE,
            out_dir=OUT_DIR,
            model_name=f"AG_ft_fold{FOLD}",
            analysis_name=f"{BIN_SIZE}bp_embedding_test_eval_{CHECKPOINT_DIR.name}",
            gtf_file=GTF_FILE,
            batch_size=BATCH_SIZE,
            is_squashed_scale=RETURN_SCALED,
            transform="alphagenome",
        )
        
    print("Eval done :D", flush=True)

    if SAVE_PREDICTIONS:
        print("Saving predictions...", flush=True)
        h5_writer.write_metadata(intervals_df, tracks_df)
        h5_writer.close()

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
