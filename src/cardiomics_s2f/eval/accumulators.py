"""
Validation and Evaluation Metric Accumulators

This module provides accumulator classes to compute correlation metrics, gene-level
expression, coverage statistics, and track the best intervals incrementally during
streaming model evaluation.
"""
import numpy as np
import pandas as pd

class StreamingPearsonAccumulator:
    """Computes exact Pearson correlation incrementally per track.

    This class tracks Pearson correlation coefficient components and keeps a count of
    valid (non-NaN) bins to generate QC metrics without storing all batch predictions
    in memory.
    """
    def __init__(self, num_tracks, track_names=None):
        """Initializes the StreamingPearsonAccumulator.

        Args:
            num_tracks (int): Number of tracks to evaluate.
            track_names (list of str, optional): Human-readable names of the tracks.
                Defaults to None.
        """
        self.num_tracks = num_tracks
        self.track_names = track_names
        
        # Accumulators for Pearson r
        self.n = np.zeros(num_tracks, dtype=np.float64)
        self.sum_x = np.zeros(num_tracks, dtype=np.float64)
        self.sum_y = np.zeros(num_tracks, dtype=np.float64)
        self.sum_xy = np.zeros(num_tracks, dtype=np.float64)
        self.sum_x2 = np.zeros(num_tracks, dtype=np.float64)
        self.sum_y2 = np.zeros(num_tracks, dtype=np.float64)
        
        # Accumulators for coverage QC
        self.total_bins_seen = 0
        self.valid_bins_per_track = np.zeros(num_tracks, dtype=np.int64)

    def update(self, P_batch, T_batch):
        """Updates the accumulators with a new batch of predictions and targets.

        Args:
            P_batch (np.ndarray): Predictions batch of shape `(batch_size, num_tracks, num_bins)`.
            T_batch (np.ndarray): Targets batch of shape `(batch_size, num_tracks, num_bins)`.
        """
        batch_size, num_tracks, num_bins = P_batch.shape
        self.total_bins_seen += (batch_size * num_bins)
        
        for t in range(num_tracks):
            p_track = P_batch[:, t, :].ravel()
            t_track = T_batch[:, t, :].ravel()
            
            # Mask out NaNs
            valid_mask = ~np.isnan(p_track) & ~np.isnan(t_track)
            p_valid = p_track[valid_mask]
            t_valid = t_track[valid_mask]
            
            # Update counts
            self.valid_bins_per_track[t] += np.sum(~np.isnan(t_track))
            
            n_batch = len(p_valid)
            if n_batch == 0:
                continue
                          
            # Force dtype=np.float64 to prevent overflow during the sum reduction
            self.n[t] += n_batch
            self.sum_x[t] += np.sum(p_valid, dtype=np.float64)
            self.sum_y[t] += np.sum(t_valid, dtype=np.float64)
            self.sum_xy[t] += np.sum(p_valid * t_valid, dtype=np.float64)
            self.sum_x2[t] += np.sum(p_valid ** 2, dtype=np.float64)
            self.sum_y2[t] += np.sum(t_valid ** 2, dtype=np.float64)

    def compute(self, coverage_cutoff_pct=0.1):
        """Computes the final Pearson correlation coefficient and coverage per track.

        Args:
            coverage_cutoff_pct (float, optional): Bins seen threshold below which
                correlation is returned as NaN. Defaults to 0.1.

        Returns:
            tuple:
                - pandas.Series: Pearson correlation coefficients per track.
                - pandas.Series: Percent coverage per track.
        """
        r_values = np.zeros(self.num_tracks, dtype=np.float64)
        coverage = self.valid_bins_per_track / self.total_bins_seen
        
        for t in range(self.num_tracks):
            if coverage[t] < coverage_cutoff_pct or self.n[t] == 0:
                r_values[t] = np.nan
                continue
                
            n = self.n[t]
            numerator = (n * self.sum_xy[t]) - (self.sum_x[t] * self.sum_y[t])
            denominator_x = (n * self.sum_x2[t]) - (self.sum_x[t] ** 2)
            denominator_y = (n * self.sum_y2[t]) - (self.sum_y[t] ** 2)
            
            if denominator_x <= 0 or denominator_y <= 0:
                r_values[t] = np.nan
            else:
                r_values[t] = numerator / np.sqrt(denominator_x * denominator_y)
                
        r_series = pd.Series(r_values, index=self.track_names)
        cov_series = pd.Series(coverage, index=self.track_names)
        return r_series, cov_series


class StreamingGeneAccumulator:
    """Accumulates predicted and actual counts over gene exons incrementally.

    This class precomputes overlaps between genomic intervals and gene exons to allow
    fast per-batch aggregation of expression targets and predictions.
    """
    def __init__(self, intervals_df, gtf_df, bin_size, track_names, track_strands):
        """Initializes the StreamingGeneAccumulator.

        Args:
            intervals_df (pandas.DataFrame): DataFrame containing genomic intervals.
            gtf_df (pandas.DataFrame): GTF DataFrame containing exon boundaries.
            bin_size (int): Size of bins in base pairs.
            track_names (list of str): List of track identifiers.
            track_strands (list of str): List of strands ('+', '-', or '.') for each track.
        """
        self.bin_size = bin_size
        self.track_names = track_names
        self.track_strands = np.array(track_strands)
        self.num_tracks = len(track_names)
        
        # Extract unique genes
        self.genes = gtf_df.index.tolist()
        self.gene_strands = gtf_df['strand'].values
        self._seen_gene_indices = set()
        
        # State: sums per gene per track
        self.gene_sums_P = np.zeros((len(self.genes), self.num_tracks), dtype=np.float64)
        self.gene_sums_T = np.zeros((len(self.genes), self.num_tracks), dtype=np.float64)
        
        # Precompute mapping: interval_idx -> list of (gene_idx, b_start, b_end)
        self.gene_to_idx = {g: i for i, g in enumerate(self.genes)}
        self.mapping = self._precompute_mapping(intervals_df, gtf_df)

    def _precompute_mapping(self, intervals_df, gtf_df):
        print("Pre-computing gene-interval overlaps...", flush = True)
        mapping = {i: [] for i in range(len(intervals_df))}
        
        # Basic merging function from your original code
        def merge_bins(intervals):
            if not intervals: return []
            intervals.sort(key=lambda x: x[0])
            merged = [intervals[0]]
            for current in intervals[1:]:
                prev = merged[-1]
                if current[0] <= prev[1]:
                    merged[-1] = (prev[0], max(prev[1], current[1]))
                else:
                    merged.append(current)
            return merged

        for i, row in intervals_df.iterrows():
            i_start, i_end, chrom = row['start'], row['end'], row['chrom']
            
            chrom_genes = gtf_df[gtf_df['chrom'] == chrom]
            if chrom_genes.empty: continue
                
            overlap_start = np.maximum(chrom_genes['start'], i_start)
            overlap_end = np.minimum(chrom_genes['end'], i_end)
            overlap_len = np.maximum(0, overlap_end - overlap_start)
            gene_len = chrom_genes['end'] - chrom_genes['start']
            
            # Genes > 50% in interval
            valid_genes = chrom_genes[overlap_len > (0.5 * gene_len)]
            
            for gene_idx, (gene_name, gene_row) in enumerate(valid_genes.iterrows()):
                # Get absolute gene index
                abs_gene_idx = self.gene_to_idx[gene_name]
                bin_intervals = []
                
                for (e_start, e_end) in gene_row['exon_intervals']:
                    o_start, o_end = max(i_start, e_start), min(i_end, e_end)
                    if o_start >= o_end: continue
                    
                    b_start = (o_start - i_start) // self.bin_size
                    b_end = (o_end - i_start + self.bin_size - 1) // self.bin_size
                    
                    if b_start < b_end:
                        bin_intervals.append((b_start, b_end))
                        
                merged_bins = merge_bins(bin_intervals)
                if merged_bins:
                    mapping[i].append((abs_gene_idx, merged_bins))
                    
        return mapping

    def update(self, start_interval_idx, P_batch, T_batch):
        """Updates the gene expression totals with a new batch of predictions and targets.

        Args:
            start_interval_idx (int): Starting global index of intervals in this batch.
            P_batch (np.ndarray): Batch predictions array of shape `(batch_size, num_tracks, num_bins)`.
            T_batch (np.ndarray): Batch targets array of shape `(batch_size, num_tracks, num_bins)`.
        """
        batch_size = P_batch.shape[0]
        
        for batch_i in range(batch_size):
            global_i = start_interval_idx + batch_i
            ops = self.mapping.get(global_i, [])
            if not ops: continue
                
            # Cumsum for fast segment queries (handling NaNs in T)
            # pad with 0 at start
            p_pad = np.pad(P_batch[batch_i], ((0, 0), (1, 0)), mode='constant')
            t_pad = np.pad(T_batch[batch_i], ((0, 0), (1, 0)), mode='constant')
            
            p_cs = np.nancumsum(p_pad, axis=1)
            t_cs = np.nancumsum(t_pad, axis=1)
            
            for gene_idx, merged_bins in ops:
                for (b_start, b_end) in merged_bins:
                    self.gene_sums_P[gene_idx] += (p_cs[:, b_end] - p_cs[:, b_start])
                    self.gene_sums_T[gene_idx] += (t_cs[:, b_end] - t_cs[:, b_start])
                    self._seen_gene_indices.add(gene_idx)

    def compute(self, return_log1p=True):
        """Computes the final accumulated gene expression values and applies strand masking.

        Args:
            return_log1p (bool, optional): Whether to apply `log1p` scaling to outputs.
                Defaults to True.

        Returns:
            tuple:
                - pandas.DataFrame: Predicted expression table of shape `(seen_genes, tracks)`.
                - pandas.DataFrame: Target expression table of shape `(seen_genes, tracks)`.
        """
        if not self._seen_gene_indices:
            return pd.DataFrame(), pd.DataFrame()
            
        seen = sorted(self._seen_gene_indices)
        P_final = self.gene_sums_P[seen].copy()
        T_final = self.gene_sums_T[seen].copy()       
        gene_names = [self.genes[i] for i in seen]
        
        # Strand masking only over seen indices
        for i, original_idx in enumerate(seen):
            g_strand_str = self.gene_strands[original_idx]
            g_strand = 1 if g_strand_str in ["+", 1, "1", "plus"] else -1 if g_strand_str in ["-", -1, "-1", "minus"] else 0
            mismatched_mask = (g_strand * self.track_strands) == -1
            
            P_final[i, mismatched_mask] = np.nan
            T_final[i, mismatched_mask] = np.nan
            
        if return_log1p:
            P_final = np.log1p(P_final)
            T_final = np.log1p(T_final)
            
        df_P = pd.DataFrame(P_final, index=gene_names, columns=self.track_names)
        df_T = pd.DataFrame(T_final, index=gene_names, columns=self.track_names)
        df_P.index.name = 'gene_name'
        df_T.index.name = 'gene_name'
        
        return df_P, df_T


class BestIntervalTracker:
    """Tracks and captures the genomic intervals with the highest correlation.

    Finds the interval achieving the highest Pearson correlation for specific assay
    modalities (e.g. ATAC, RNA, CutnTag) to allow visualization of top-performing regions.
    """
    def __init__(self, track_df):
        """Initializes the BestIntervalTracker.

        Args:
            track_df (pandas.DataFrame): Metadata DataFrame of the tracks. Must contain
                an 'Assay_type' column.
        """
        self.modalities = ["ATAC", "RNA", "CnT"]
        self.masks = {mod: (track_df['Assay_type'] == mod).values for mod in self.modalities}
        self.track_df = track_df
        
        self.best_scores = {mod: -np.inf for mod in self.modalities}
        self.best_data = {mod: None for mod in self.modalities}

    def _pearson_batch(self, p_mod, t_mod, min_coverage=0.1):
        """Vectorized Pearson r for [batch, tracks, bins] → [batch, tracks]"""
        num_bins = p_mod.shape[2]
        
        # Valid mask: non-NaN in both P and T
        valid = ~np.isnan(p_mod) & ~np.isnan(t_mod)
        n = valid.sum(axis=2, dtype=np.float64)      
        
        # FIX: Force float64 here so all subsequent squares and sums use 64-bit precision
        P_v = np.where(valid, p_mod, 0.0).astype(np.float64)
        T_v = np.where(valid, t_mod, 0.0).astype(np.float64)       
        
        sp = P_v.sum(axis=2)
        sq = T_v.sum(axis=2)
        spp = (P_v**2).sum(axis=2)
        sqq = (T_v**2).sum(axis=2)
        spq = (P_v * T_v).sum(axis=2)
        
        numer = n * spq - sp * sq
        denom = np.sqrt(np.maximum(0, n*spp - sp**2) * np.maximum(0, n*sqq - sq**2))       
        r_vals = np.where(denom > 0, numer / denom, np.nan)
        
        # Apply the 10% coverage logic set to NaN if coverage is too low
        r_vals = np.where(n > (num_bins * min_coverage), r_vals, np.nan)
        
        return r_vals

    def update(self, P_batch, T_batch, coords_batch):
        """Checks the batch for intervals with better correlation scores.

        Args:
            P_batch (np.ndarray): Predictions batch of shape `(batch_size, num_tracks, num_bins)`.
            T_batch (np.ndarray): Targets batch of shape `(batch_size, num_tracks, num_bins)`.
            coords_batch (list of tuple): Coordinate tuples representing genomic locations of each
                batch element.
        """
        batch_size = P_batch.shape[0]
        
        for mod in self.modalities:
            mask = self.masks[mod]
            if not mask.any(): continue
            
            p_mod = P_batch[:, mask, :]
            t_mod = T_batch[:, mask, :]
            r_batch = self._pearson_batch(p_mod, t_mod, min_coverage=0.1)
            
            for i in range(batch_size):
                valid_r_vals = r_batch[i][~np.isnan(r_batch[i])]
                if len(valid_r_vals) > 0:
                    score = np.median(valid_r_vals)
                    if score > self.best_scores[mod]:
                        self.best_scores[mod] = score
                        # Store a snapshot
                        self.best_data[mod] = {
                            "coord": coords_batch[i],
                            "P": p_mod[i].copy(),
                            "T": t_mod[i].copy(),
                            "track_names": self.track_df[mask].index.tolist()
                        }


class StreamingCoverageAccumulator:
    """Accumulates and tracks the fraction of covered (non-NaN) bins per interval/track.

    Useful for diagnostic coverage metrics during streaming model evaluation.
    """
    def __init__(self):
        """Initializes the StreamingCoverageAccumulator."""
        # Using a list of arrays is highly memory efficient for this 
        # because the arrays are 2D (Batch x Tracks) and very small.
        self.interval_coverages = []

    def update(self, T_batch_unsq):
        """Calculates the fraction of non-NaN bins for a batch.

        Expects raw, unsquashed BigWig targets to accurately measure sparsity.
        
        Args:
            T_batch_unsq (np.ndarray): Raw targets of shape `(batch, tracks, bins)`.
        """
        num_bins = T_batch_unsq.shape[2]
        
        # Count valid bins per interval per track
        valid_bins_batch = np.sum(~np.isnan(T_batch_unsq), axis=2)
        
        batch_coverage_pct = (valid_bins_batch / num_bins).astype(np.float64)
        self.interval_coverages.append(batch_coverage_pct)

    def compute(self):
        """Concatenates all batch coverages into a single matrix.
        
        Returns:
            np.ndarray: Combined coverage array of shape `(total_intervals, tracks)`.
        """
        if not self.interval_coverages:
            raise ValueError("Accumulator is empty. Did you call update()?")
            
        return np.concatenate(self.interval_coverages, axis=0)
    
    
#DO NOT USE ANYMORE AS COVERAGE PER BIN IS NOT REALLY INTERESTING                        
# class ReservoirSampler:
#     """Maintains a random sample of (P, T) bins for density scatters using fast vectorized reservoir sampling."""
#     def __init__(self, num_tracks, k=10000):
#         self.num_tracks = num_tracks
#         self.k = k
#         self.reservoir_P = np.full((num_tracks, k), np.nan, dtype=np.float32)
#         self.reservoir_T = np.full((num_tracks, k), np.nan, dtype=np.float32)
#         self.items_seen = np.zeros(num_tracks, dtype=np.int64)

#     def update(self, P_batch, T_batch):
#         # P_batch: [batch, tracks, bins] -> reshape to [tracks, batch*bins]
#         P_flat = P_batch.transpose(1, 0, 2).reshape(self.num_tracks, -1)
#         T_flat = T_batch.transpose(1, 0, 2).reshape(self.num_tracks, -1)

#         for t in range(self.num_tracks):
#             valid = ~(np.isnan(P_flat[t]) | np.isnan(T_flat[t]))
#             p_v = P_flat[t, valid]
#             t_v = T_flat[t, valid]
            
#             n_new = len(p_v)
#             if n_new == 0:
#                 continue

#             idx_start = self.items_seen[t]
            
#             # Fill the reservoir if its not full yet
#             fill_count = min(n_new, self.k - idx_start) if idx_start < self.k else 0
#             if fill_count > 0:
#                 self.reservoir_P[t, idx_start:idx_start+fill_count] = p_v[:fill_count]
#                 self.reservoir_T[t, idx_start:idx_start+fill_count] = t_v[:fill_count]
                
#             # Vectorized random replacement for any remaining items
#             remaining = n_new - fill_count
#             if remaining > 0:
#                 # Global indices of these new items (M)
#                 M = idx_start + fill_count + np.arange(remaining)
#                 j = np.random.randint(0, M + 1) # Generate a random integer between 0 and M for each item
#                 selected = j < self.k # Items with a random number < k get to replace an item in the reservoir
                
#                 if selected.any():
#                     replace_idx = j[selected]
#                     new_p = p_v[fill_count:][selected]
#                     new_t = t_v[fill_count:][selected]
#                     self.reservoir_P[t, replace_idx] = new_p
#                     self.reservoir_T[t, replace_idx] = new_t
                    
#             self.items_seen[t] += n_new