"""
Data Output Module

This module provides helpers to save predictions to compressed and structured
formats (like annotated HDF5), either in one go with a ``.npy`` fallback if
dependencies like h5py are missing, or iteratively batch by batch for outputs
that do not fit into memory.
"""
from pathlib import Path
import numpy as np
from pandas.api.types import is_string_dtype, is_object_dtype


def _write_dataframe_group(h5_file, group_name, df):
    # HDF5 requires string arrays to be explicitly typed as bytes ('S')
    group = h5_file.create_group(group_name)
    for col in df.columns:
        data = df[col].values
        if is_string_dtype(df[col]) or is_object_dtype(df[col]):
            data = data.astype(str).astype('S')
        group.create_dataset(col, data=data)


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
                _write_dataframe_group(f, "intervals", intervals_df)

            if tracks_df is not None:
                _write_dataframe_group(f, "tracks", tracks_df)

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


class HDF5BatchWriter:
    """Iterative saving class designed to handle massive output arrays.

    By writing to disk iteratively, we avoid out-of-memory (OOM) errors
    when dealing with 1bp resolution data across thousands of intervals.

    Args:
        out_dir (str or Path): Directory where the HDF5 file will be saved.
        file_prefix (str): Prefix for the generated HDF5 file name.
        total_samples (int): Total number of samples (intervals) to pre-allocate.

    Attributes:
        out_dir (Path): Resolved path for output.
        h5_path (Path): Full path to the targeted `.h5` file.
        total_samples (int): Total expected size of the first dimension.
        file (h5py.File): Open file handle for writing.
        datasets (dict): Dictionary tracking initialized datasets per head.
    """
    def __init__(self, out_dir, file_prefix, total_samples):
        import h5py

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.h5_path = self.out_dir / f"{file_prefix}.h5"
        self.total_samples = total_samples

        print(f"Opening HDF5 file for iterative writing at {self.h5_path}...", flush=True)
        self.file = h5py.File(self.h5_path, 'w')
        self.datasets = {}

    def write_batch(self, head_name, batch_preds, start_idx):
        """Writes a batch of predictions for a specific head to disk.

        Dynamically creates the dataset if it is the first batch received for
        the given model head.

        Args:
            head_name (str): Identifier for the model output head.
            batch_preds (np.ndarray): Predictions array for the current batch.
            start_idx (int): The starting index along the sample dimension where
                this batch should be inserted.
        """
        # Create the dataset dynamically upon receiving the first batch for a specific head
        if head_name not in self.datasets:
            full_shape = (self.total_samples, *batch_preds.shape[1:])
            head_grp = self.file.create_group(head_name)
            self.datasets[head_name] = head_grp.create_dataset(
                "predictions", shape=full_shape, dtype=batch_preds.dtype, compression="gzip"
            )

        # Calculate the chunk slice and dump the data to disk
        end_idx = start_idx + batch_preds.shape[0]
        self.datasets[head_name][start_idx:end_idx] = batch_preds

    def write_metadata(self, intervals_df, tracks_df):
        """Saves interval and track metadata to the HDF5 file.

        Converts string/object dtypes to byte strings (`'S'`) to ensure
        compatibility with HDF5 standards.

        Args:
            intervals_df (pd.DataFrame): DataFrame containing interval coordinates.
            tracks_df (pd.DataFrame): DataFrame containing track metadata.
        """
        print("Writing metadata to HDF5...", flush=True)
        _write_dataframe_group(self.file, "intervals", intervals_df)
        _write_dataframe_group(self.file, "tracks", tracks_df)

    def close(self):
        """Safely closes the underlying HDF5 file handle."""
        self.file.close()
        print("Successfully saved and closed HDF5 file!", flush=True)
