from pathlib import Path
import numpy as np
from pandas.api.types import is_string_dtype, is_object_dtype

def save_borzoi_predictions_with_fallback(preds, out_dir, file_prefix, intervals_df=None, tracks_df=None):
    """Attempts to save predictions as annotated HDF5; falls back to .npy only if that fails."""
    
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