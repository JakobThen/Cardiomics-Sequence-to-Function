"""
Batch BigWig Loader Module

This module provides a class to load data from multiple BigWig files
in parallel, keeping file handles open for efficient batch retrieval.
"""
import pyBigWig
import numpy as np
from concurrent.futures import ThreadPoolExecutor


class BatchBigWigLoader:
    """A context-managed loader for reading multiple BigWig files in parallel.

    This class opens multiple BigWig files during context entry, keeps them open
    in memory, and uses a ThreadPool to read them concurrently when `load_batch`
    is called.
    """

    def __init__(self, bw_paths_df, num_bins, bin_size, n_workers=8):
        """Initializes the BatchBigWigLoader.

        Args:
            bw_paths_df (pandas.DataFrame): DataFrame containing a 'path' column
                specifying the file path for each track, in track order.
            num_bins (int): Number of bins to divide each genomic interval into.
            bin_size (int): Size of each bin in base pairs.
            n_workers (int, optional): Number of concurrent workers for reading
                BigWig files. Defaults to 8.
        """
        self.bw_paths = bw_paths_df['path'].tolist()
        self.num_tracks = len(self.bw_paths)
        self.num_bins = num_bins
        self.bin_size = bin_size
        self.n_workers = min(n_workers, self.num_tracks)
        self.files = []

    def __enter__(self):
        """Opens all BigWig files and returns the loader instance.

        Returns:
            BatchBigWigLoader: The opened loader instance.
        """
        print(f"Opening {self.num_tracks} BigWig files...", flush = True)
        for path in self.bw_paths:
            try:
                bw = pyBigWig.open(str(path))
                self.files.append(bw)
            except Exception as e:
                print(f"Failed to open {path}: {e}", flush = True)
                self.files.append(None) # Keep None to maintain index alignment
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Closes all open BigWig file handles."""
        print(f"Closing {self.num_tracks} BigWig files...", flush = True)
        for bw in self.files:
            if bw is not None:
                bw.close()

    def _read_track_for_batch(self, track_idx, coords):
        """Reads one track across a batch of coordinates."""
        bw = self.files[track_idx]
        track_data = np.full((len(coords), self.num_bins), np.nan, dtype=np.float32)
        
        if bw is None:
            return track_idx, track_data
            
        valid_chroms = bw.chroms()
        
        for int_idx, (chrom, start, end) in enumerate(coords):
            if chrom not in valid_chroms or start < 0 or end > valid_chroms[chrom]:
                continue
                
            vals = bw.values(chrom, int(start), int(end), numpy=True)
            if vals is None or len(vals) == 0:
                continue
                
            reshaped = vals.reshape(self.num_bins, self.bin_size)
            all_nan = np.isnan(reshaped).all(axis=1)
            bin_sums = np.nansum(reshaped, axis=1)
            
            bin_sums[all_nan] = np.nan
            track_data[int_idx, :] = bin_sums
            
        return track_idx, track_data

    def load_batch(self, coords):
        """Loads a batch of intervals for all tracks in parallel.

        Args:
            coords (list of tuple): List of genomic intervals as `[(chrom, start, end), ...]`.

        Returns:
            np.ndarray: A 3D array of shape `(batch_size, num_tracks, num_bins)` containing the
            accumulated signals.
        """
        batch_size = len(coords)
        T_batch = np.full((batch_size, self.num_tracks, self.num_bins), np.nan, dtype=np.float32)
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [
                executor.submit(self._read_track_for_batch, t, coords) 
                for t in range(self.num_tracks)
            ]
            for future in futures:
                track_idx, track_data = future.result()
                T_batch[:, track_idx, :] = track_data
                
        return T_batch

    
