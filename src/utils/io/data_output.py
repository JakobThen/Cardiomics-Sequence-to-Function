"""
Data Output Module

This module provides helper functions to save predictions to compressed
and structured formats (like annotated HDF5), with fallbacks if dependencies
like h5py are missing.
"""
from pathlib import Path
import numpy as np
from pandas.api.types import is_string_dtype, is_object_dtype

def save_borzoi_predictions_with_fallback(preds, out_dir, file_prefix, intervals_df=None, tracks_df=None):
    """Attempts to save predictions as an annotated HDF5 file; falls back to numpy (.npy) on failure.

    This function compiles predictions and optional metadata (interval and track annotations)
    into a structured HDF5 dataset, with gzip compression. If `h5py` is not installed or
    the write fails, it falls back to saving as a plain NumPy `.npy` file.

    Args:
        preds (np.ndarray): The predictions array to save.
        out_dir (str or Path): Output directory where the file will be saved.
        file_prefix (str): Prefix name of the output file (excluding extension).
        intervals_df (pandas.DataFrame, optional): DataFrame containing annotations
            for the genomic intervals. Saved as the "intervals" group in HDF5.
        tracks_df (pandas.DataFrame, optional): DataFrame containing annotations
            for the tracks. Saved as the "tracks" group in HDF5.
    """
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    npy_path = out_dir / f"{file_prefix}.npy"
    h5_path = out_dir / f"{file_prefix}.h5"
    
    print(f"\nAttempting to compile annotated HDF5 object at {h5_path}...", flush=True)
    
    try:
        import h5py
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset("predictions", data=preds, compression="gzip")
            
            if intervals_df is not None:
                interval_grp = f.create_group("intervals")
                for col in intervals_df.columns:
                    data = intervals_df[col].values
                    if is_string_dtype(intervals_df[col]) or is_object_dtype(intervals_df[col]):
                        data = data.astype(str).astype('S')
                    interval_grp.create_dataset(col, data=data)
            
            if tracks_df is not None:
                track_grp = f.create_group("tracks")
                for col in tracks_df.columns:
                    data = tracks_df[col].values
                    if is_string_dtype(tracks_df[col]) or is_object_dtype(tracks_df[col]):
                        data = data.astype(str).astype('S')
                    track_grp.create_dataset(col, data=data)
                    
        print("Successfully saved annotated HDF5!", flush=True)

    except ImportError:
        print("'h5py' module not found. Falling back to .npy...", flush=True)
        np.save(npy_path, preds)
        print(f"Saved fallback .npy to {npy_path}", flush=True)

    except Exception as e:
        print("Failed to save HDF5 file. Falling back to .npy...", flush=True)
        print(f"Error details: {e}", flush=True)
        np.save(npy_path, preds)
        print(f"Saved fallback .npy to {npy_path}", flush=True)