# function to run th track evalution of seunce to function models
import os
import gc
import pyBigWig
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, List, Tuple, Union, Optional, Iterable

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm, trange

def get_gtf(
    gtf_file: str, 
    chrom_filter: Optional[Union[str, List[str]]] = None, 
    gene_type_filter: Optional[List[str]] = ["protein_coding"], 
    feature: str = "exon"
) -> pd.DataFrame:
    """
    Parses a GTF file, filters for a specific feature, extracts key attributes, 
    and applies optional chromosome and gene type filters.

    Args:
        gtf_file (str): Path to the GTF file.
        chrom_filter (str or List[str], optional): Chromosomes to keep. 
            Accepts preset strings: 'autosomes' (chr1-22), 'autosomesX' (chr1-22 + chrX), 
            or 'autosomesXY' (chr1-22 + chrX, chrY). 
            Also accepts a custom list of exact names (e.g., ['chr1', 'chrM']).
        gene_type_filter (List[str], optional): List of gene types to keep 
            (e.g., ['protein_coding']). If None, all are kept. Defaults to protein coding.
        feature (str, optional): The feature type to filter for. If None, all are kept. Defaults to "exon".

    Returns:
        pd.DataFrame: A filtered DataFrame with extracted gene_name, gene_type, 
        gene_id, and transcript_id columns.
    """
    
    target_chroms = chrom_filter
    if isinstance(chrom_filter, str):
        autosomes = [f"chr{i}" for i in range(1, 23)]
        if chrom_filter.lower() == "autosomes":
            target_chroms = autosomes
        elif chrom_filter.lower() == "autosomesx":
            target_chroms = autosomes + ["chrX"]
        elif chrom_filter.lower() == "autosomesxy":
            target_chroms = autosomes + ["chrX", "chrY"]
        else:
            raise ValueError(
                "Invalid chrom_filter string. Use 'autosomes', 'autosomesX', "
                "'autosomesXY', or provide a list of specific chromosomes."
            )

    try:
        gtf = pd.read_csv(gtf_file, sep="\t", comment="#", header=None, dtype=str)
        gtf.columns = ["chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]
        
        gtf["start"] = gtf["start"].astype(int)
        gtf["end"] = gtf["end"].astype(int)
        
        if feature:
            gtf = gtf[gtf["feature"] == feature].copy()
        if target_chroms is not None:
            gtf = gtf[gtf["chrom"].isin(target_chroms)]
            
        gtf["gene_name"] = gtf["attribute"].str.extract(r'gene_name\s+"([^"]+)"')
        gtf["gene_type"] = gtf["attribute"].str.extract(r'gene_type\s+"([^"]+)"')
        gtf["gene_id"] = gtf["attribute"].str.extract(r'gene_id\s+"([^"]+)"')
        gtf["transcript_id"] = gtf["attribute"].str.extract(r'transcript_id\s+"([^"]+)"')
        
        if gene_type_filter:
            gtf = gtf[gtf["gene_type"].isin(gene_type_filter)]
            
        gtf = gtf.drop_duplicates().reset_index(drop=True)      
        return gtf

    except FileNotFoundError:
        print(f"Error: The file '{gtf_file}' was not found.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise
        
def merge_bin_intervals(intervals):
    """Merges overlapping or adjacent bin intervals to prevent double-counting.
    Handles this issue: Exon A spans base pairs 10 to 40. This means it falls into Bins 0 and 1.
    Exon B spans base pairs 50 to 80. This falls into Bins 1 and 2. This prevents this"""
    if not intervals:
        return []
    # Sort by start bin
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        previous = merged[-1]
        # If current start overlaps or touches previous end, merge them
        if current[0] <= previous[1]:
            merged[-1] = (previous[0], max(previous[1], current[1]))
        else:
            merged.append(current)
    return merged

def sum_exon_counts(
    pred_array: np.ndarray, 
    intervals_df: pd.DataFrame, 
    genes_df: pd.DataFrame, 
    bin_size: int = 1, 
    exon_col: str = 'exons',
    track_names: Optional[List[str]] = None,
    use_strand: bool = True,
    track_strands: Optional[Union[List[int], np.ndarray]] = None,
    return_log1p_counts: bool = True
) -> pd.DataFrame:
    """
    Sums predicted counts over exons and returns a DaraFrame of Genes x Tracks.
    
    Args:
        pred_array (np.ndarray): Shape (Intervals, tracks, bins).
        intervals_df (pd.DataFrame): Genomic coordinates with 'chrom', 'start', 'end'.
        genes_df (pd.DataFrame): Gene info with 'chrom', 'start', 'end', and `exon_col`. Must include 'strand' if use_strand is True.
        bin_size (int): Resolution of the bins.
        exon_col (str): Column name in genes_df containing the exon coordinates.
        track_names (List[str], optional): List of names for the columns.
        use_strand (bool): Whether to match tracks by strand to genes. Assumes track_names contains minus and plus if track_strands is not defined.
        track_strands (List[int] or np.ndarray, optional): Explicit vector of track strands (1 (plus), -1 (minus), or 0 (unstranded)). 
            If None and use_strand=True, inferred from track_names.
        return_log1p_counts (bool): Whether to return the summed counts as log1p(sum).
        
    Returns:
        pd.DataFrame: Matrix where Index = gene_id and Columns = tracks.
    """
    num_intervals, num_tracks, num_bins = pred_array.shape
    
    if track_names is not None and len(track_names) != num_tracks:
        raise ValueError(f"Length of track_names ({len(track_names)}) must match number of tracks ({num_tracks}).")
        
    if num_intervals != len(intervals_df):
        raise ValueError(f"Number of predictions ({num_intervals}) must match number of genomic coordinates ({len(intervals_df)}).")
        
    if use_strand and "strand" not in genes_df.columns:
        raise ValueError("column 'strand' required in genes_df if use_strand=True")

    if use_strand:
        if track_strands is not None:
            if len(track_strands) != num_tracks:
                raise ValueError("Length of track_strands must match number of tracks.")
            t_strands = np.array(track_strands)
        else:
            if track_names is None:
                raise ValueError("track_names required to infer strands if track_strands is not provided.")
            #infer strand from track name
            t_strands = np.array([
                1 if "plus" in track.lower() else -1 if "minus" in track.lower() else 0 
                for track in track_names
            ])
            
            if np.all(t_strands == 0):
                raise ValueError("track_names must contain 'plus' or 'minus', or provide track_strands explicitly.")
    else:
        t_strands = np.zeros(num_tracks)
        
    results_dict = {}
    for i in trange(num_intervals, desc="Processing Intervals"):
        i_start = intervals_df.iloc[i]['start']
        i_end = intervals_df.iloc[i]['end']
        chrom = intervals_df.iloc[i]['chrom']

        chrom_genes = genes_df[genes_df['chrom'] == chrom]
        if chrom_genes.empty:
            continue

        # Find genes > 50% in interval (avoid double counting genes)
        overlap_start = np.maximum(chrom_genes['start'], i_start)
        overlap_end = np.minimum(chrom_genes['end'], i_end)
        overlap_len = np.maximum(0, overlap_end - overlap_start)
        gene_len = chrom_genes['end'] - chrom_genes['start']

        mask = overlap_len > (0.5 * gene_len)
        valid_genes = chrom_genes[mask]

        if valid_genes.empty:
            continue

        # Cumsum for fast querying
        pred_i = pred_array[i] 
        pred_pad = np.pad(pred_i, ((0, 0), (1, 0)), mode='constant') #add 0 at the beginning for correct indexing
        pred_cumsum = np.nancumsum(pred_pad, axis=1) #nan cumsum handles missing valued when using observed bw tracks as input

        for gene_idx, gene_row in valid_genes.iterrows():
            exons = gene_row[exon_col]
            bin_intervals = []
            
            for (e_start, e_end) in exons:
                o_start = max(i_start, e_start)
                o_end = min(i_end, e_end)

                if o_start >= o_end:
                    continue 

                b_start = (o_start - i_start) // bin_size
                b_end = (o_end - i_start + bin_size - 1) // bin_size

                b_start = max(0, min(b_start, num_bins))
                b_end = max(0, min(b_end, num_bins))

                if b_start < b_end:
                    bin_intervals.append((b_start, b_end))

            merged_bins = merge_bin_intervals(bin_intervals) #avoid double counting bins overlapping multiple exons

            # Sum over the merged bins
            gene_sum = np.zeros(num_tracks)
            for (b_start, b_end) in merged_bins:
                gene_sum += (pred_cumsum[:, b_end] - pred_cumsum[:, b_start])

            #set tracks with wrong strand for gene to NA
            if use_strand:
                g_strand = 1 if gene_row["strand"] in ["+", 1, "1", "plus"] else -1 if gene_row["strand"] in ["-", -1, "-1", "minus"] else 0
                mismatched_mask = (g_strand * t_strands) == -1 #for all tracks thta mathc this vector is 1, mismatch is -1 unstranded is 0
                gene_sum[mismatched_mask] = np.nan
            
            if return_log1p_counts:
                gene_sum = np.log1p(gene_sum)
            results_dict[gene_idx] = gene_sum

    df_matrix = pd.DataFrame.from_dict(results_dict, orient='index')
    
    if track_names is not None:
        df_matrix.columns = track_names
    else:
        df_matrix.columns = [f"Track_{i}" for i in range(num_tracks)]
        
    df_matrix.index.name = 'gene_id'
    return df_matrix


def get_counts_from_bw(
    track_idx: int,
    bw_path: Union[str, Path],
    coords: List[tuple],
    num_bins: int,
    bin_size: int
) -> Tuple[int, np.ndarray]:
    track_data = np.full((len(coords), num_bins), np.nan, dtype=np.float32)
    try:
        bw_file = pyBigWig.open(str(bw_path)) 
        valid_chroms = bw_file.chroms()
        
        for int_idx, (chrom, start, end) in enumerate(coords):
            if chrom not in valid_chroms or start < 0 or end > valid_chroms[chrom]:
                continue

            vals = bw_file.values(chrom, int(start), int(end), numpy=True)
            if vals is None or len(vals) == 0:
                continue
                
            reshaped = vals.reshape(num_bins, bin_size)
            all_nan = np.isnan(reshaped).all(axis=1)
            bin_sums = np.nansum(reshaped, axis=1)
            
            # Restore NaNs for fully NaN bins
            bin_sums[all_nan] = np.nan 
            track_data[int_idx, :] = bin_sums
            
    finally:
        bw_file.close() # Ensure file is closed even if an error occurs
        
    return track_idx, track_data


def get_counts_from_bws(
    bw_paths: Iterable[Union[str, Path]],
    coordinates_df: pd.DataFrame, 
    n_workers: int = 1, 
    bin_size: int = 1  
) -> np.ndarray:
    '''
    Extracts bigwig counts from multiple bigwig files over identical intervals in parallel.
    '''
    assert all(c in coordinates_df.columns for c in ['chrom', 'start', 'end']), "coordinates_df must contain 'chrom', 'start', and 'end'."
    
    lengths = coordinates_df["end"] - coordinates_df["start"]
    interval_len = lengths.iloc[0]
    assert (lengths == interval_len).all(), "All intervals in coordinates_df must have the exact same length."
    assert interval_len % bin_size == 0, f"Interval length ({interval_len}) must be perfectly divisible by bin_size ({bin_size})."
    
    bw_paths = list(bw_paths)
    num_tracks = len(bw_paths)
    num_intervals = len(coordinates_df)
    num_bins = interval_len // bin_size

    coords = list(coordinates_df[['chrom', 'start', 'end']].itertuples(index=False, name=None))
    max_workers = min(os.cpu_count() or 1, n_workers, num_tracks)

    # Initialize array in shape (Intervals, Tracks, Bins)
    T = np.full((num_intervals, num_tracks, num_bins), np.nan, dtype=np.float32)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(get_counts_from_bw, track_idx, path, coords, num_bins, bin_size): track_idx
            for track_idx, path in enumerate(bw_paths)
        }

        # Process results as they finish to populate the main array
        for future in tqdm(as_completed(futures), total=num_tracks, desc="Extracting BigWig Tracks"):
            track_idx, track_data = future.result()
            T[:, track_idx, :] = track_data

    total_bins = T.size
    valid_bins = total_bins - np.isnan(T).sum()
    print(f"Found {valid_bins} non NaN bins of {total_bins} total bins ({np.round((valid_bins/total_bins)*100, 2)}%).")
    
    return T

def _extract_h5_group_to_df(group):
    """
    Helper function to convert an HDF5 group of 1D datasets into a pandas DataFrame.
    """
    if group is None:
        return None
        
    data_dict = {}
    for key in group.keys():
        arr = group[key][()]
        if len(arr) > 0 and isinstance(arr[0], bytes):
            arr = [val.decode('utf-8') for val in arr]
        data_dict[key] = arr
        
    if not data_dict:
        return None
        
    return pd.DataFrame(data_dict)

def _align_prediction_axes(P, num_intervals=None, num_tracks=None, head_name="Single Head"):
    """
    Infers the axes of the prediction array P using metadata lengths and 
    safely transposes it to (intervals, tracks, bins).
    """
    if P.ndim != 3:
        raise ValueError(f"[{head_name}] Expected 3D prediction array, got {P.ndim}D.")

    shape = P.shape
    
    # If we have no metadata to guide return as is
    if num_intervals is None and num_tracks is None:
        return P

    # Default axis assumptions
    ax_intervals = 0
    ax_tracks = 1
    ax_bins = 2
    
    # find the intervals Axis
    if num_intervals is not None:
        if shape[0] == num_intervals:
            ax_intervals = 0
        elif num_intervals in shape:
            ax_intervals = shape.index(num_intervals)
        else:
            raise ValueError(f"[{head_name}] Shape {shape} does not contain an axis matching {num_intervals} intervals.")
            
    # fid the Tracks Axis
    if num_tracks is not None:
        # Look only at the axes that are NOT the intervals axis
        remaining_axes = [i for i in range(3) if i != ax_intervals]
        
        # Check which of the remaining axes matches the track count
        if shape[remaining_axes[0]] == num_tracks and shape[remaining_axes[1]] != num_tracks:
            ax_tracks = remaining_axes[0]
            ax_bins = remaining_axes[1]
        elif shape[remaining_axes[1]] == num_tracks and shape[remaining_axes[0]] != num_tracks:
            ax_tracks = remaining_axes[1]
            ax_bins = remaining_axes[0]
        elif shape[remaining_axes[0]] == num_tracks and shape[remaining_axes[1]] == num_tracks:
            # Edge case: Tracks and Bins have the exact same size. 
            # We assume it was originally (intervals, tracks, bins) relative to remaining axes.
            ax_tracks = remaining_axes[0]
            ax_bins = remaining_axes[1]
        else:
            raise ValueError(f"[{head_name}] Shape {shape} does not contain an axis matching {num_tracks} tracks (excluding intervals axis).")

    # handle case where we know intervals but have no track metadata
    elif num_intervals is not None:
        remaining_axes = [i for i in range(3) if i != ax_intervals]
        ax_tracks = remaining_axes[0]
        ax_bins = remaining_axes[1]

    # transpose if the axes are out of the expected (0, 1, 2) order
    if (ax_intervals, ax_tracks, ax_bins) != (0, 1, 2):
        print(f"[{head_name}] Auto-aligning axes: Transposing shape {shape} using mapping "
              f"intervals->axis{ax_intervals}, tracks->axis{ax_tracks}, bins->axis{ax_bins}.")
        P = np.transpose(P, (ax_intervals, ax_tracks, ax_bins))
        
    return P


def load_predictions_from_h5(h5_path):
    """
    Reads an HDF5 file containing genomic predictions and metadata.
    Handles single-head and multi-head files, and automatically realigns 
    scrambled array dimensions to (intervals x tracks x bins).
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        intervals_df = None
        num_intervals = None
        if "intervals" in f:
            intervals_df = _extract_h5_group_to_df(f["intervals"])
            num_intervals = len(intervals_df) if intervals_df is not None else None
        else:
            print(f"Warning: 'intervals' group missing in {h5_path.name}")
       
        # Single-Head (Borzoi)
        if "predictions" in f:
            P = f["predictions"][()]
            tracks_df = _extract_h5_group_to_df(f.get("tracks"))
            num_tracks = len(tracks_df) if tracks_df is not None else None
            
            # Auto-align the single prediction array
            P = _align_prediction_axes(P, num_intervals, num_tracks, head_name="SingleHead")
            
        # Multi-Head (AG)
        else:
            preds_list = []
            tracks_list = []
            
            for head_name in f.keys():
                if head_name == "intervals":
                    continue 
                    
                head_grp = f[head_name]
                if "predictions" not in head_grp:
                    continue 
                    
                p_array = head_grp["predictions"][()]
                t_df = _extract_h5_group_to_df(head_grp.get("tracks"))
                num_tracks = len(t_df) if t_df is not None else None
                
                p_array = _align_prediction_axes(p_array, num_intervals, num_tracks, head_name)
                
                preds_list.append(p_array)
                if t_df is not None:
                    tracks_list.append(t_df)
                    
            if not preds_list:
                raise ValueError(f"Could not find any 'predictions' datasets in {h5_path.name}")
            
            # Verify shape alignment before concatenating
            # Because we just ran _align_prediction_axes, we KNOW it's (intervals, tracks, bins)
            # So we just ensure intervals (axis 0) and bins (axis 2) match across heads.
            base_shape = preds_list[0].shape
            for i, p in enumerate(preds_list):
                if p.shape[0] != base_shape[0] or p.shape[2] != base_shape[2]:
                    raise ValueError(f"Shape mismatch in multi-head concatenation! "
                                     f"Expected intervals/bins to match (Intervals: {base_shape[0]}, Bins: {base_shape[2]}), "
                                     f"but got shape {p.shape}.")

            # Concatenate predictions along the tracks axis (axis=1)
            P = np.concatenate(preds_list, axis=1)
            
            # Concatenate tracks DataFrames row-wise
            if tracks_list and len(tracks_list) == len(preds_list):
                tracks_df = pd.concat(tracks_list, ignore_index=True)
            else:
                tracks_df = None
                
    return P, intervals_df, tracks_df


def process_track_metadata(tracks_df, folder_to_assay=None):
    """
    Finds the .bw column in a tracks DataFrame, standardizes filename/path columns, 
    and extracts cell_type and Assay_type metadata.
    """
    df = tracks_df.copy()
    
    # column containing '.bw' files
    bw_col = None
    for col in df.columns:
        valid_vals = df[col].dropna().astype(str)
        if len(valid_vals) > 0 and valid_vals.iloc[0].endswith('.bw'):
            bw_col = col
            break
            
    if bw_col is None:
        raise ValueError("Could not find any column containing '.bw' files.")
        
    # determine if the column contains full paths or just filenames
    # We check if any of the strings contain a forward or backward slash
    has_path_separators = df[bw_col].astype(str).str.contains(r'[/\\]', regex=True).any()
    if has_path_separators:
        df.rename(columns={bw_col: 'path'}, inplace=True)
        df['filename'] = df['path'].apply(lambda p: Path(p).name)
    else:
        df.rename(columns={bw_col: 'filename'}, inplace=True)
        
    if "cell_type" not in df.columns:    
        df['cell_type'] = (df['filename']
                           .str.replace('_100M_norm.bw', '', regex=False)
                           .str.replace('_100M.bw', '', regex=False)
                           .str.replace('ATAC_', '', regex=False)
                           .str.replace('RNA_', '', regex=False))
               
    assay = []
    for ass in df['filename']:
        if "RNA" in ass: assay.append("RNA")
        elif "ATAC" in ass: assay.append("ATAC")
        else: assay.append("CnT")
    df['Assay_type'] = assay
                       
    if 'path' in df.columns:
        df['folder'] = df['path'].apply(lambda p: Path(p).parent.name)
        
    if df["filename"].nunique == len(df):
        df["id"] = df["filename"]
    else:
        df["id"] = df["cell_type"] + "_" + df["filename"] 
           
                
    return df