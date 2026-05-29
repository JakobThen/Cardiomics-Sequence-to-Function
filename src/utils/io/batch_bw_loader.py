import pyBigWig
import numpy as np
from concurrent.futures import ThreadPoolExecutor

"""
    This class opens multiple BigWig files once during __enter__, keeps them open in memory, and uses a ThreadPool to read them concurrently when load_batch is called.
    This is done for model evaulation during inference and correaltion analysis.
"""

class BatchBigWigLoader:
    def __init__(self, bw_paths_df, num_bins, bin_size, n_workers=8):
        """
        bw_paths_df: DataFrame containing a 'path' column in the exact order of your tracks.
        """
        self.bw_paths = bw_paths_df['path'].tolist()
        self.num_tracks = len(self.bw_paths)
        self.num_bins = num_bins
        self.bin_size = bin_size
        self.n_workers = min(n_workers, self.num_tracks)
        self.files = []

    def __enter__(self):
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
        """
        coords: List of tuples [(chrom, start, end), ...] for the batch.
        Returns: T_batch array of shape (batch_size, num_tracks, num_bins)
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

    
