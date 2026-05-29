import os
import sys
import numpy as np
import torch

parent_dir = os.path.abspath("/home/then")
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from utils.io.batch_bw_loader import BatchBigWigLoader 

from typing import Union, List, Optional
TensorOrArray = Union[np.ndarray, torch.Tensor]


# CORE MATHEMATICAL OPERATIONS (Internal Helpers)
def _borzoi_core_squash(y: TensorOrArray) -> TensorOrArray:
    if isinstance(y, torch.Tensor):
        y_34 = torch.pow(torch.relu(y), 0.75) # relu prevents NaN on negative inputs
        return torch.where(y_34 <= 384.0, y_34, 384.0 + torch.sqrt(torch.relu(y_34 - 384.0)))
    else:
        y_34 = np.power(np.maximum(y, 0.0), 0.75)
        return np.where(y_34 <= 384.0, y_34, 384.0 + np.sqrt(np.maximum(y_34 - 384.0, 0.0)))

def _borzoi_core_inverse(y: TensorOrArray) -> TensorOrArray:
    if isinstance(y, torch.Tensor):
        y_34 = torch.where(y <= 384.0, y, torch.square(y - 384.0) + 384.0)
        return torch.pow(torch.relu(y_34), 4.0 / 3.0)
    else:
        y_34 = np.where(y <= 384.0, y, np.square(y - 384.0) + 384.0)
        return np.power(np.maximum(y_34, 0.0), 4.0 / 3.0)

def _alphagenome_core_squash(y: TensorOrArray) -> TensorOrArray:
    if isinstance(y, torch.Tensor):
        y = torch.pow(torch.relu(y), 0.75)
        return torch.where(y > 10.0, 2.0 * torch.sqrt(y * 10.0) - 10.0, y)
    else:
        y = np.power(np.maximum(y, 0.0), 0.75)
        return np.where(y > 10.0, 2.0 * np.sqrt(y * 10.0) - 10.0, y)

def _alphagenome_core_inverse(y: TensorOrArray) -> TensorOrArray:
    if isinstance(y, torch.Tensor):
        y = torch.where(y > 10.0, torch.square(y + 10.0) / 40.0, y)
        return torch.pow(torch.relu(y), 4.0 / 3.0)
    else:
        y = np.where(y > 10.0, np.square(y + 10.0) / 40.0, y)
        return np.power(np.maximum(y, 0.0), 4.0 / 3.0)


# PUBLIC STANDALONE FUNCTIONS
def borzoi_squash(y: TensorOrArray) -> TensorOrArray:
    """Applies the Borzoi 'squashed scale' transformation."""
    if isinstance(y, torch.Tensor): y = y.to(torch.float32)
    else: y = np.array(y, dtype=np.float32)
    return _borzoi_core_squash(y)

def inverse_borzoi_squash(y_sq: TensorOrArray) -> TensorOrArray:
    """Inverts the Borzoi 'squashed scale' transformation."""
    if isinstance(y_sq, torch.Tensor): y_sq = y_sq.to(torch.float32)
    else: y_sq = np.array(y_sq, dtype=np.float32)
    return _borzoi_core_inverse(y_sq)

def alphagenome_squash(
    targets: TensorOrArray, 
    track_means: Union[TensorOrArray, float], 
    apply_rna_squashing: bool = True
) -> TensorOrArray:
    """Scales targets by track means and optionally applies AlphaGenome squashing."""
    if isinstance(targets, torch.Tensor):
        targets = targets.to(torch.float32)
        if isinstance(track_means, torch.Tensor):
            track_means = track_means.to(dtype=torch.float32, device=targets.device)
    else:
        targets = np.array(targets, dtype=np.float32)

    targets = targets / track_means
    if apply_rna_squashing:
        targets = _alphagenome_core_squash(targets)
    return targets

def invert_alphagenome_squash(
    x: TensorOrArray, 
    track_means: Union[TensorOrArray, float], 
    apply_rna_squashing: bool = True
) -> TensorOrArray:
    """Inverts the AlphaGenome scaling and squashing to return raw counts."""
    if isinstance(x, torch.Tensor):
        x = x.to(torch.float32)
        if isinstance(track_means, torch.Tensor):
            track_means = track_means.to(dtype=torch.float32, device=x.device)
    else:
        x = np.array(x, dtype=np.float32)

    if apply_rna_squashing:
        x = _alphagenome_core_inverse(x)
    return x * track_means

def get_non_zero_track_means(preds: np.ndarray) -> np.ndarray:
    """
    Calculates the non-zero mean per track for an array of shape (intervals, tracks, bins).
    Safely ignores NaN values (treating them as missing/zero coverage).
    Highly memory efficient.
    """
    track_sums = np.nansum(preds, axis=(0, 2))
    #mask that ensures the value is NOT 0 and NOT NaN.
    valid_mask = (preds != 0) & ~np.isnan(preds)
    track_counts = np.sum(valid_mask, axis=(0, 2))
    track_means = np.divide(
        track_sums, 
        track_counts, 
        out=np.zeros_like(track_sums, dtype=np.float32), 
        where=track_counts != 0
    )
    return track_means

def get_streaming_non_zero_track_means(bw_paths_df, intervals_df, bin_size, batch_size=256):
    """
    Computes non-zero track means across all intervals of a bunch of bw files without loading everything into RAM.
    Uses float64 for accumulators to prevent overflow on large datasets.
    """
    print("Pre-computing track means from BigWigs for AlphaGenome transform...", flush=True)
    num_bins = (intervals_df['end'].iloc[0] - intervals_df['start'].iloc[0]) // bin_size
    num_tracks = len(bw_paths_df)
    
    track_sums = np.zeros(num_tracks, dtype=np.float64)
    track_counts = np.zeros(num_tracks, dtype=np.float64)
       
    with BatchBigWigLoader(bw_paths_df, num_bins, bin_size) as bw_loader:
        for start_idx in range(0, len(intervals_df), batch_size):
            end_idx = min(start_idx + batch_size, len(intervals_df))
            coords = intervals_df.iloc[start_idx:end_idx][['chrom', 'start', 'end']].apply(tuple, axis=1).tolist()
            T_batch = bw_loader.load_batch(coords)
            valid_mask = (T_batch != 0) & ~np.isnan(T_batch)  # Find valid data (not NaN, not 0)
            
            # Accumulate sums and counts across Batch (axis 0) and Bins (axis 2)
            track_sums += np.nansum(T_batch, axis=(0, 2))
            track_counts += np.sum(valid_mask, axis=(0, 2))
            
    track_means = np.divide(
        track_sums, 
        track_counts, 
        out=np.zeros_like(track_sums, dtype=np.float32), 
        where=track_counts != 0
    )
    
    print("Track means computation complete.", flush=True)
    return track_means.astype(np.float32)

# UNIFIED SELECTIVE TRANSFORM CLASS
class SelectiveSquashTransform:
    """
    A callable transform that applies selective squashing to specific tasks (e.g. RNA-seq).
    Compatible with PyTorch Lightning DataLoaders and native model transforms.
    """
    def __init__(
        self, 
        rna_task_indices: List[int], 
        task_axis: int = -2, 
        transform: str = "Borzoi",
        track_means: Optional[TensorOrArray] = None,
        inverse: bool = False
    ):
        """
        Args:
            rna_task_indices: List of integer indices corresponding to the RNA-seq tracks.
            task_axis: The axis containing the different tasks (-2 works for B,T,L and T,L).
            transform: "Borzoi" or "AlphaGenome".
            track_means: Required if transform="AlphaGenome". A 1D array of shape (N_Tasks,).
            inverse: If True, reverses the transformation (useful for prediction/inference).
        """
        self.rna_indices = list(rna_task_indices)
        self.task_axis = task_axis
        self.transform = transform.strip().lower()
        self.inverse = inverse
        
        if self.transform not in ["borzoi", "alphagenome"]:
            raise ValueError(f'Transform must be "Borzoi" or "AlphaGenome", got "{transform}".')
            
        if self.transform == "alphagenome" and track_means is None:
            raise ValueError("The 'AlphaGenome' transform requires 'track_means' to be provided.")
            
        self.track_means = track_means

    def _format_track_means(self, target_array: TensorOrArray) -> TensorOrArray:
        """Helper to ensure track_means broadcasts against the target array properly."""
        tm = self.track_means
        is_torch = isinstance(target_array, torch.Tensor)
        
        # 1. Match type and device
        if is_torch:
            if not isinstance(tm, torch.Tensor):
                tm = torch.tensor(tm, dtype=torch.float32, device=target_array.device)
            else:
                tm = tm.to(dtype=torch.float32, device=target_array.device)
        else:
            tm = np.array(tm, dtype=np.float32)
            
        # 2. Reshape for broadcasting (e.g., from [T] to [1, T, 1])
        shape = [1] * target_array.ndim
        shape[self.task_axis] = -1
        return tm.reshape(shape)

    def __call__(self, y: TensorOrArray) -> TensorOrArray:
        # 1. Cast and clone to avoid modifying original dataloader memory in-place
        if isinstance(y, torch.Tensor):
            out = y.clone().to(torch.float32)
        else:
            out = np.array(y, copy=True, dtype=np.float32)
            
        # 2. Create dynamic slicing tuple for the RNA tracks
        slices = [slice(None)] * out.ndim
        slices[self.task_axis] = self.rna_indices
        slices = tuple(slices)

        # 3. Apply Transformations
        if self.transform == "borzoi":
            if not self.inverse:
                if self.rna_indices:
                    out[slices] = _borzoi_core_squash(out[slices])
            else:
                if self.rna_indices:
                    out[slices] = _borzoi_core_inverse(out[slices])

        elif self.transform == "alphagenome":
            tm = self._format_track_means(out)
            
            if not self.inverse:
                # Forward: Scale ALL tracks by means, then squash RNA tracks
                out = out / tm
                if self.rna_indices:
                    out[slices] = _alphagenome_core_squash(out[slices])
            else:
                # Inverse: Inverse squash RNA tracks, then scale ALL tracks back by means
                if self.rna_indices:
                    out[slices] = _alphagenome_core_inverse(out[slices])
                out = out * tm

        return out