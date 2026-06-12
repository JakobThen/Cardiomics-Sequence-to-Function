"""Plotting functions and processing helpers to analyze and visualize model track prediction performance"""
import gc
import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Any, Optional, Dict

from scipy.stats import gaussian_kde

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from mpl_toolkits.axes_grid1 import make_axes_locatable

def compute_na_masked_pearson_distribution(P, T, min_valid_pct=0.1, return_sum_counts = False):
    """Calculates the Pearson correlation coefficient matrix between predictions and targets.

    Computes correlation coefficients between all prediction tracks and target tracks across
    non-NaN bins, using vectorized operations.

    Args:
        P (np.ndarray): Predictions array of shape `(intervals, P_tracks, bins)`.
        T (np.ndarray): Targets array of shape `(intervals, T_tracks, bins)`.
        min_valid_pct (float, optional): Minimum required valid (non-NaN) bins fraction
            per interval for calculating correlation. Defaults to 0.1.
        return_sum_counts (bool, optional): If True, returns target and prediction sum matrices.
            Defaults to False.

    Returns:
        np.ndarray or tuple:
            - np.ndarray: Pearson correlation matrix of shape `(intervals, P_tracks, T_tracks)`.
            - np.ndarray: Sum of targets per interval/track (if `return_sum_counts=True`).
            - np.ndarray: Sum of predictions per interval/track (if `return_sum_counts=True`).
    """
    # Create masks and safe-T
    T_valid_mask = (~np.isnan(T)).astype(np.uint8) # 1.0 if valid, 0.0 if NaN
    T_0 = np.nan_to_num(T, nan=0.0)             # Safe for math; masked out later
    
    # Count valid elements per batch (summing over Bins at axis=2)
    # Resulting shape: (Batch, 1, N_tracks) to broadcast against M_tracks
    C = np.sum(T_valid_mask, axis=2)[:, np.newaxis, :]
    C_safe = np.maximum(C, 1)
    
    # Calculate sums per interval
    # Einsum variables: b=batch, m=P_tracks, n=T_tracks, l=bins (now at the end)
    sum_P = np.einsum('bml,bnl->bmn', P, T_valid_mask)
    mu_P = sum_P / C_safe
    if not return_sum_counts: del sum_P
    sum_P2 = np.einsum('bml,bnl->bmn', P**2, T_valid_mask)
    
    del T_valid_mask; gc.collect() 
    
    sum_T = np.sum(T_0, axis=2)[:, np.newaxis, :]
    mu_T = sum_T / C_safe
    if not return_sum_counts: del sum_T
    sum_T2 = np.sum(T_0**2, axis=2)[:, np.newaxis, :]
    
    sum_PT = np.einsum('bml,bnl->bmn', P, T_0)
    
    cov = sum_PT - (C * mu_P * mu_T)
    var_P = sum_P2 - (C * (mu_P**2))
    var_T = sum_T2 - (C * (mu_T**2))
    
    std_product = np.sqrt(np.maximum(var_P, 0) * np.maximum(var_T, 0))
    valid_combo_mask = (C >= int(T.shape[2]*min_valid_pct)) & (std_product > 1e-12)
    
    # R_indiv shape: (Batch, M_pred_tracks, N_bw_tracks)
    R_indiv = np.divide(cov, std_product, out=np.full_like(cov, np.nan), where=valid_combo_mask)
    
    if return_sum_counts:
        if sum_T.shape != sum_P.shape: sum_T = np.broadcast_to(sum_T, sum_P.shape)     # Broadcast sum_T to match sum_P's 3D shape (Batch, M, N) so they align in the long dataframe
        return R_indiv, sum_T, sum_P
    else:
        return R_indiv


#quantlile normalizeing the df for nromlaized correaltion analysis
def quantile_normalize_RNA_counts(log_counts_data):
    """Quantile normalizes RNA counts across tracks and centers them by gene means.

    Aligns count distributions across tracks to a shared target distribution
    (the mean distribution across tracks) and centers them by subtracting the average
    count of each gene. Robust to NaN values.

    Args:
        log_counts_data (pandas.DataFrame or np.ndarray): Input counts table of shape
            `(genes, tracks)`. Expects log-transformed counts.

    Returns:
        pandas.DataFrame or np.ndarray: Normalized and gene-centered counts matching the input type.
    """
    is_df = isinstance(log_counts_data, pd.DataFrame)
    arr = log_counts_data.values.astype(float) if is_df else np.asarray(log_counts_data).astype(float)
    sorted_genes = np.sort(arr, axis=0)
    
    # # context manager to suppress warnings if a row is entirely NaN
    # with np.errstate(mean='ignore', invalid='ignore'):
    gene_target_distribution = np.nanmean(sorted_genes, axis=1)
    
    temp_df = log_counts_data if is_df else pd.DataFrame(arr)
    gene_ranks = temp_df.rank(method='min').values - 1
    norm_pred = np.full_like(arr, np.nan, dtype=float)
    
    for col_idx in range(arr.shape[1]):
        valid_mask = ~np.isnan(arr[:, col_idx])
        valid_ranks = gene_ranks[valid_mask, col_idx].astype(int)
        norm_pred[valid_mask, col_idx] = gene_target_distribution[valid_ranks]
   
    # Center the predictions ignoring NaNs
    #with np.errstate(mean='ignore', invalid='ignore'):
    gene_means = np.nanmean(norm_pred, axis=1, keepdims=True)
    centered_norm_pred = norm_pred - gene_means
    
    if is_df:
        return pd.DataFrame(centered_norm_pred, index=log_counts_data.index, columns=log_counts_data.columns)
    
    return centered_norm_pred

import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Optional
from scipy.ndimage import gaussian_filter

import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Optional
from scipy.ndimage import gaussian_filter

def density_scatter(
    pred: Any, 
    obs: Any, 
    ax: Optional[plt.Axes] = None, 
    figsize: tuple = (5, 5), 
    xlabel: Optional[str] = None, 
    ylabel: Optional[str] = None, 
    title: Optional[str] = None,
    density: bool = True,
    bins: int = 100,         
    smooth_sigma: float = 1.0 
) -> plt.Axes:
    """
    Creates a square scatter plot of predicted vs observed values, colored by point density.
    """
    x = np.asarray(pred).flatten()
    y = np.asarray(obs).flatten()
    
    valid_mask = np.isfinite(x) & np.isfinite(y)
    x = x[valid_mask]
    y = y[valid_mask]
    
    if len(x) < 2:
        raise ValueError("Not enough valid points to plot after removing NaNs/Infs.")

    if density:
        H, xedges, yedges = np.histogram2d(x, y, bins=bins)
        H_smoothed = gaussian_filter(H, sigma=smooth_sigma)
        
        # FIX: Clip directly to the generated dimensions of H_smoothed
        # This guarantees you never get an out-of-bounds error regardless of the bins used
        x_idx = np.clip(np.digitize(x, xedges) - 1, 0, H_smoothed.shape[0] - 1)
        y_idx = np.clip(np.digitize(y, yedges) - 1, 0, H_smoothed.shape[1] - 1)
        
        z = H_smoothed[x_idx, y_idx]

        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    else:
        z = np.ones_like(x)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(x, y, c=z, s=10, cmap='viridis', edgecolor='none', rasterized=True)
    
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    pad = (max_val - min_val) * 0.05
    ax.set_xlim(min_val - pad, max_val + pad)
    ax.set_ylim(min_val - pad, max_val + pad)
    ax.set_aspect('equal', adjustable='box')

    ax.plot(
        [min_val - pad, max_val + pad], 
        [min_val - pad, max_val + pad], 
        color='lightgrey', 
        linestyle='-', 
        linewidth=1, 
        alpha=0.8, 
        zorder=0
    )

    ax.set_xlabel(xlabel if xlabel is not None else "log predicted counts")
    ax.set_ylabel(ylabel if ylabel is not None else "log observed counts")
    if title:
        ax.set_title(title)
    ax.grid(False)
    return ax

def plot_correlation_beeswarm(
    values: Any, 
    labels: Any, 
    ax: Optional[plt.Axes] = None, 
    figsize: Optional[tuple] = None, 
    ylabel: Optional[str] = None, 
    title: Optional[str] = None
) -> plt.Axes:
    """
    Creates side-by-side beeswarm plots for categorical data, highlighting the mean.
    Automatically scales y-limits and handles both flat and grouped inputs.
    
    Args:
        values: Can be either:
                1. A single 1D array of values (e.g., [0.9, 0.8, 0.5...]).
                2. A list/tuple of arrays (e.g., [arr_A, arr_B]).
        labels: Can be either:
                1. A 1D array of labels matching the length of values (e.g., ['A', 'A', 'B'...]).
                2. A list of category names matching the number of arrays in values (e.g., ['A', 'B']).
        ax: Matplotlib axes object. If None, creates a new figure and axis.
        figsize: Tuple specifying (width, height). Defaults to (1 + n, 5).
        ylabel: Custom y-axis label. Defaults to 'Pearson Correlation'.
        title: Optional title for the plot.
        
    Returns:
        ax: The matplotlib Axes object containing the plot.
    """
    
    is_grouped_input = False
    if len(values) == len(labels) and len(values) > 0:
        first_elem = values[0]
        if isinstance(first_elem, (list, tuple, np.ndarray, pd.Series)) and not isinstance(first_elem, str):
            is_grouped_input = True

    if is_grouped_input:
        flat_values = []
        flat_labels = []
        for i, arr in enumerate(values):
            arr_flat = np.asarray(arr).flatten()
            flat_values.extend(arr_flat)
            # Duplicate the category label for every item in this array
            flat_labels.extend([labels[i]] * len(arr_flat)) 
            
        # Reassign to our standard variables
        values = flat_values
        labels = flat_labels

    df = pd.DataFrame({'value': values, 'label': labels}).dropna()    
    if df.empty:
        raise ValueError("No valid data points left to plot after dropping NaNs.")
        
    unique_labels = df['label'].unique()
    n = len(unique_labels)
    
    if ax is None:
        if figsize is None:
            figsize = (2 + 0.6*n, 5) # Minimum width of 5, scales up with n
        fig, ax = plt.subplots(figsize=figsize)

    sns.swarmplot(
        data=df, 
        x='label', 
        y='value', 
        hue='label', 
        palette='husl', 
        size=4,         
        legend=False,
        ax=ax,
        zorder=1
    )

    means = df.groupby('label')['value'].mean()
    plotted_categories = [tick.get_text() for tick in ax.get_xticklabels()]
    
    for i, category in enumerate(plotted_categories):
        mean_val = means.get(category, np.nan)
        if pd.notna(mean_val):
            ax.hlines(
                y=mean_val, 
                xmin=i - 0.1, 
                xmax=i + 0.1, 
                color='black', 
                linewidth=2, 
                zorder=5 
            )

    min_val = df['value'].min()
    if min_val < 0:
        ax.set_ylim(min_val-0.05, 1.05)
    else:
        ax.set_ylim(0, 1.05)
    ax.axhline(y=0, color='grey', linewidth=1, alpha=0.8, zorder=0)
    ax.set_xlabel("") 
    ax.set_ylabel(ylabel if ylabel is not None else "Pearson Correlation")
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    if title:
        ax.set_title(title)
        
    ax.grid(False)

    return ax

def plot_interval_correlation_heatmap(
    R_indiv: np.ndarray,
    intervals_df: pd.DataFrame,
    track_names: List[str],
    coverage_per_track: Union[np.ndarray, List[float]],
    title: str = 'Individual Pearson Correlation per Interval and Track',
    cmap: str = 'viridis'
) -> Tuple[Figure, Axes]:
    """
    Generates an annotated heatmap of interval-track correlations.
    
    The heatmap is automatically sorted by genomic position (chromosome, then start).
    It includes a left-side annotation for track coverage, and top annotations for 
    chromosome and start positions.
    
    Args:
        R_indiv (np.ndarray): 2D array of correlations of shape (n_intervals, n_tracks).
        intervals_df (pd.DataFrame): DataFrame of shape (n_intervals, ...). Must 
                                     contain 'chrom' and 'start' columns.
        track_names (List[str]): List of track names, length equal to n_tracks.
        coverage_per_track (Union[np.ndarray, List[float]]): 1D array of coverage 
                                                             values (0 to 1), length equal to n_tracks.
        title (str): The main title of the plot.
        cmap (str): The matplotlib colormap to use for the main correlation heatmap.

    Returns:
        Tuple[Figure, Axes]: The matplotlib Figure and main Axes object.
    
    Raises:
        ValueError: If the dimensions of the inputs do not align.
        KeyError: If 'chrom' or 'start' are missing from intervals_df.
    """
    # ==========================================
    # 0. Safety Checks
    # ==========================================
    n_intervals, n_tracks = R_indiv.shape
    if len(intervals_df) != n_intervals:
        raise ValueError(f"intervals_df length ({len(intervals_df)}) does not match R_indiv rows ({n_intervals}).")
    if len(track_names) != n_tracks:
        raise ValueError(f"track_names length ({len(track_names)}) does not match R_indiv columns ({n_tracks}).")
    if len(coverage_per_track) != n_tracks:
        raise ValueError(f"coverage_per_track length ({len(coverage_per_track)}) does not match n_tracks ({n_tracks}).")
    if not {'chrom', 'start'}.issubset(intervals_df.columns):
        raise KeyError("intervals_df must contain 'chrom' and 'start' columns.")

    # ==========================================
    # 1. Sort the Data
    # ==========================================
    # Work on a copy to avoid mutating the original DataFrame
    df = intervals_df[['chrom', 'start']].copy()
    
    chrom_order = [f"chr{i}" for i in range(1, 23)] + ['chrX', 'chrY', 'chrM']
    
    df['chrom'] = pd.Categorical(
        df['chrom'], 
        categories=chrom_order, 
        ordered=True
    )
    
    sorted_df = df.reset_index(drop=True).sort_values(by=['chrom', 'start'])
    sort_idx = sorted_df.index.to_numpy()
    
    # Reorder the main array (Columns/Intervals)
    # R_indiv is (intervals, tracks), so R_indiv.T is (tracks, intervals)
    R_sorted = R_indiv.T[:, sort_idx]

    # ==========================================
    # 2. Prepare Annotation Data
    # ==========================================
    cov_array = np.array(coverage_per_track).reshape(-1, 1)
    chrom_integers = sorted_df['chrom'].cat.codes.values.reshape(1, -1)
    start_values = sorted_df['start'].values.reshape(1, -1)

    # ==========================================
    # 3. Setup Figure and Divider
    # ==========================================
    # Dynamically scale height based on number of tracks
    fig_height = 1 + 6 * (len(track_names) / 30)
    fig, ax_main = plt.subplots(figsize=(24, fig_height))

    divider = make_axes_locatable(ax_main)

    ax_left = divider.append_axes("left", size="2%", pad=0.1)
    ax_top_start = divider.append_axes("top", size="2%", pad=0.05)
    ax_top_chrom = divider.append_axes("top", size="2%", pad=0.05)
    
    cax_main = divider.append_axes("right", size="1%", pad=0.1)       
    cax_stacked = divider.append_axes("right", size="1%", pad=0.9)   
    cax_stacked.axis('off') 
    
    cax_chrom = cax_stacked.inset_axes([0, 0.55, 1, 0.45]) 
    cax_start = cax_stacked.inset_axes([0, 0.00, 1, 0.45]) 

    # ==========================================
    # 4. Plot the Data
    # ==========================================
    im_main = ax_main.imshow(R_sorted, aspect='auto', cmap=cmap, interpolation='nearest')
    im_left = ax_left.imshow(cov_array, aspect='auto', cmap='Reds', interpolation='nearest', vmin=0, vmax=1)
    im_start = ax_top_start.imshow(start_values, aspect='auto', cmap='magma', interpolation='nearest')
    im_chrom = ax_top_chrom.imshow(chrom_integers, aspect='auto', cmap='tab20', interpolation='nearest')

    # ==========================================
    # 5. Clean Up Axes, Ticks, and Labels
    # ==========================================
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    ax_main.set_xlabel('Intervals (Sorted by Chrom & Start)', fontsize=12)
    
    ax_left.set_xticks([])
    ax_left.set_yticks(np.arange(len(track_names)))
    ax_left.set_yticklabels(track_names)
    ax_left.set_ylabel('Tracks', fontsize=12)
    ax_left.set_xlabel('Coverage\n(non NaN bin %)', fontsize=10, rotation=90, labelpad=10)
    
    ax_top_start.set_xticks([])
    ax_top_start.set_yticks([])
    ax_top_start.set_ylabel('Start', rotation=0, ha='right', va='center', fontsize=10)
    
    ax_top_chrom.set_xticks([])
    ax_top_chrom.set_yticks([])
    ax_top_chrom.set_ylabel('Chrom', rotation=0, ha='right', va='center', fontsize=10)
    ax_top_chrom.set_title(title, fontsize=16, pad=15)

    # ==========================================
    # 6. Assign and Format Colorbars
    # ==========================================
    fig.colorbar(im_main, cax=cax_main, label='Interval Pearson R')
    
    cb_chrom = fig.colorbar(im_chrom, cax=cax_chrom, label='Interval Chromosome')
    unique_chroms = np.unique(chrom_integers[0]) 
    cb_chrom.set_ticks(unique_chroms)
    cb_chrom.set_ticklabels([chrom_order[i] for i in unique_chroms])
    cb_chrom.ax.tick_params(labelsize=8) 
    
    cb_start = fig.colorbar(im_start, cax=cax_start, label='Interval Start Position')
    cb_start.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x/1e6:g}M"))

    # ==========================================
    # 7. Add Chromosome Boundaries
    # ==========================================
    chrom_changes = np.where(chrom_integers[0][:-1] != chrom_integers[0][1:])[0]
    for boundary in chrom_changes:
        ax_main.axvline(x=boundary + 0.5, color='black', linewidth=0.7, alpha=1)

    return fig, ax_main

def compare_track_coverage(
    preds: Any,
    obs: Any,
    ax: Optional[Axes] = None,
    pos: Optional[Tuple[str, int, int]] = None,
    title: Optional[str] = None,
    log_transform: bool = True,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 3)
) -> Tuple[Figure, Axes]:
    """
    Plots predicted against observed coverage for a genomic interval as overlapping tracks.
    
    Args:
        preds: Array-like of predicted values (plotted on top).
        obs: Array-like of observed values (plotted underneath).
        ax: Optional matplotlib Axes. If None, creates a new Figure and Axes.
        pos: Optional tuple of (chrom, start, end). If None, uses relative coordinates.
        title: Optional title for the plot.
        log_transform: If True, applies np.log1p to both arrays.
        xlabel: Custom x-axis label. Defaults to 'Genomic Position' or 'Relative Position'.
        ylabel: Custom y-axis label. Defaults to 'Coverage (log1p counts)' or 'Coverage (counts)'.
        figsize: Tuple specifying figure dimensions if ax is not provided.
        
    Returns:
        Tuple[Figure, Axes]: The matplotlib Figure and Axes object.
    """
    preds_arr = np.asarray(preds, dtype=np.float64).flatten()
    obs_arr = np.asarray(obs, dtype=np.float64).flatten()
    
    if len(preds_arr) != len(obs_arr):
        raise ValueError(f"Length mismatch: preds ({len(preds_arr)}) vs obs ({len(obs_arr)})")
        
    n_bins = len(preds_arr)
    
    #compute cor
    valid_mask = ~np.isnan(preds_arr) & ~np.isnan(obs_arr)
    T_valid = obs_arr[valid_mask]
    P_valid = preds_arr[valid_mask]
    if len(T_valid) > 2:
        r = np.corrcoef(P_valid, T_valid)[0, 1]
    else:
        r = np.nan
    
    preds_arr = np.nan_to_num(preds_arr, nan=0.0, posinf=0.0, neginf=0.0)
    obs_arr = np.nan_to_num(obs_arr, nan=0.0, posinf=0.0, neginf=0.0)
    
    if log_transform:
        preds_arr = np.log1p(preds_arr)
        obs_arr = np.log1p(obs_arr)
        default_ylabel = "Coverage (log1p)"
    else:
        default_ylabel = "Coverage (squashed scale)"
        
    # Genomic Coordinates (X-axis)
    if pos is not None:
        chrom, start, end = pos
        if (end - start) % n_bins != 0:
            print(f"Warning: Interval length ({end - start}) is not perfectly divisible by n_bins ({n_bins}). Check your positions.")
        x_coords = np.linspace(start, end, n_bins, endpoint=False)
        default_xlabel = f"Interval {chrom}:{start}-{end}"
    else:
        x_coords = np.arange(n_bins)
        default_xlabel = "Relative Bin Position"
        
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
  
    ax.plot(x_coords, obs_arr, color='darkblue', alpha = 0.8, linewidth=1, label='Observed', zorder=1)
    ax.plot(x_coords, preds_arr, color='red', alpha = 0.8, linewidth=1, label='Predicted', zorder=2)

    ax.set_xlim(x_coords[0], x_coords[-1])
    ax.set_ylim(bottom=0)
    
    ax.set_ylabel(ylabel if ylabel is not None else default_ylabel)
    ax.set_xlabel(xlabel if xlabel is not None else default_xlabel)
    
    if pos is not None:
        if start > 1e6:
             ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x/1e6:g}M"))
        elif start > 1e3:
             ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x/1e3:g}k"))

    if title: title = f"{title} (r = {np.round(r, 2)})"
    else: title = f"r = {np.round(r, 2)}"
        
    ax.set_title(title)
        
    ax.grid(False)
    ax.legend(loc='upper right', frameon=False)
    
    return fig, ax   

def plot_strand_expression_heatmaps(
    gene_P: pd.DataFrame,
    gene_T: pd.DataFrame,
    gene_cors: Dict[str, pd.Series],
    gene_cors_by_gene: Optional[Dict[str, pd.Series]] = None,  
    figsize: Tuple[int, int] = (26, 15),
    cmap_exp: str = 'viridis',
    cmap_cent: str = 'RdBu_r',
    cmap_cor: str = 'magma',
    label_exp: str = 'log1p(Sum Exon Counts)',
    label_cent: str = 'Gene Centered Exon Counts [Δ]',
    label_cor: str = 'Pearson r'
) -> Figure:
    """Generates interleaved heatmaps of gene expression, centered expression, and correlations.

    Interleaves predicted and observed expression profiles for plus and minus strand genes,
    sorted by descending average expression. Side heatmaps display track-level correlations,
    and optional top heatmaps show gene-level correlations.

    Args:
        gene_P (pandas.DataFrame): Predicted gene expression matrix of shape `(genes, tracks)`.
        gene_T (pandas.DataFrame): Observed gene expression matrix of shape `(genes, tracks)`.
        gene_cors (dict of str to pandas.Series): Dict of track-level correlation series
            (keys are metrics, values are series indexed by track).
        gene_cors_by_gene (dict of str to pandas.Series, optional): Dict of gene-level
            correlation series (keys are metrics, values are series indexed by gene).
            Defaults to None.
        figsize (Tuple[int, int], optional): Matplotlib figure size. Defaults to (26, 15).
        cmap_exp (str, optional): Colormap for absolute expression. Defaults to 'viridis'.
        cmap_cent (str, optional): Colormap for gene-centered expression. Defaults to 'RdBu_r'.
        cmap_cor (str, optional): Colormap for correlation heatmaps. Defaults to 'magma'.
        label_exp (str, optional): Label for the expression colorbar.
            Defaults to 'log1p(Sum Exon Counts)'.
        label_cent (str, optional): Label for the centered expression colorbar.
            Defaults to 'Gene Centered Exon Counts [Δ]'.
        label_cor (str, optional): Label for the correlation colorbar. Defaults to 'Pearson r'.

    Returns:
        matplotlib.figure.Figure: The generated figure containing the heatmaps.
    """
    # ==========================================
    # 1. Data Preparation & Masking
    # ==========================================
    track_names = gene_P.columns
    is_plus_mask = np.array([True if "plus" in str(t).lower() else False for t in track_names])

    pP = gene_P.loc[:, is_plus_mask].dropna().T
    pT = gene_T.loc[:, is_plus_mask].dropna().T
    mP = gene_P.loc[:, ~is_plus_mask].dropna().T
    mT = gene_T.loc[:, ~is_plus_mask].dropna().T

    plus_sorted_genes  = pT.mean(axis=0).sort_values(ascending=False).index
    minus_sorted_genes = mT.mean(axis=0).sort_values(ascending=False).index

    pP, pT = pP[plus_sorted_genes],   pT[plus_sorted_genes]
    mP, mT = mP[minus_sorted_genes],  mT[minus_sorted_genes]

    pP_cent, pT_cent = pP - pP.mean(axis=0), pT - pT.mean(axis=0)
    mP_cent, mT_cent = mP - mP.mean(axis=0), mT - mT.mean(axis=0)

    # ==========================================
    # 2. Internal Helpers
    # ==========================================
    def interleave_pt(P_df, T_df):
        P_vals, T_vals = P_df.values, T_df.values
        n_tracks, n_genes = P_vals.shape
        interleaved = np.empty((n_tracks * 2, n_genes), dtype=P_vals.dtype)
        interleaved[0::2, :] = P_vals
        interleaved[1::2, :] = T_vals
        track_labels = [str(n).replace('_minus','').replace('_plus','').replace('.bw','') for n in P_df.index]
        track_ticks = np.arange(0.5, n_tracks * 2, 2)
        row_ticks   = np.arange(n_tracks * 2)
        row_labels  = ["Pred", "Obs"] * n_tracks
        return interleaved, track_labels, track_ticks, row_labels, row_ticks

    def build_cor_matrix(P_df):
        """Track-level correlations, interleaved to match Pred/Obs rows."""
        n_tracks  = len(P_df.index)
        cor_names = list(gene_cors.keys())
        cor_matrix = np.empty((n_tracks * 2, len(cor_names)))
        for j, col in enumerate(cor_names):
            vals = gene_cors[col].reindex(P_df.index).values
            cor_matrix[0::2, j] = vals
            cor_matrix[1::2, j] = vals
        return cor_matrix, cor_names

    def build_gene_annot_matrix(sorted_genes):
        """Gene-level correlations → (n_gene_cors, n_genes) matrix."""
        if not gene_cors_by_gene:
            return None, []
        names  = list(gene_cors_by_gene.keys())
        matrix = np.array([gene_cors_by_gene[k].reindex(sorted_genes).values
                           for k in names], dtype=float)
        return matrix, names

    # ==========================================
    # 3. Matrix Generation & Scaling
    # ==========================================
    plus_matrix,      p_labels, p_ticks, row_labels, row_ticks = interleave_pt(pP, pT)
    minus_matrix,     m_labels, m_ticks, _,          _         = interleave_pt(mP, mT)
    plus_cent_matrix, *_  = interleave_pt(pP_cent, pT_cent)
    minus_cent_matrix, *_ = interleave_pt(mP_cent, mT_cent)

    plus_cor_matrix,  cor_names = build_cor_matrix(pP)
    minus_cor_matrix, _         = build_cor_matrix(mP)

    plus_gene_matrix,  gene_annot_names = build_gene_annot_matrix(plus_sorted_genes)
    minus_gene_matrix, _                = build_gene_annot_matrix(minus_sorted_genes)
    has_gene_annot = gene_cors_by_gene is not None and len(gene_cors_by_gene) > 0

    # Global colour bounds
    all_exp  = np.concatenate([plus_matrix.ravel(), minus_matrix.ravel()])
    exp_vmin, exp_vmax = np.nanpercentile(all_exp, 1), np.nanpercentile(all_exp, 99)

    all_cent = np.concatenate([plus_cent_matrix.ravel(), minus_cent_matrix.ravel()])
    cent_vmax = np.nanpercentile(np.abs(all_cent), 99)
    cent_vmin = -cent_vmax

    # Include gene-level annotations in the shared correlation colour scale
    all_cor_parts = [plus_cor_matrix.ravel(), minus_cor_matrix.ravel()]
    if has_gene_annot:
        all_cor_parts += [plus_gene_matrix.ravel(), minus_gene_matrix.ravel()]
    all_cor = np.concatenate(all_cor_parts)
    cor_vmin, cor_vmax = min(0, np.nanmin(all_cor)), 1

    # ==========================================
    # 4. Layout Setup
    # ==========================================
    # Gene annotation rows get a small height fraction; expression rows get 1.
    # Layout (6 rows when annotation present, 4 rows otherwise):
    #   0: gene annot – plus
    #   1: p1  (plus raw)
    #   2: p2  (plus centred)
    #   3: gene annot – minus
    #   4: m1  (minus raw)
    #   5: m2  (minus centred)
    n_gene_annot_rows = len(gene_cors_by_gene) if has_gene_annot else 0
    annot_h = max(0.12, 0.12 * n_gene_annot_rows)  # scales with number of annotations

    if has_gene_annot:
        n_rows = 6
        height_ratios = [annot_h, 1, 1, annot_h, 1, 1]
        row_idx = {'annot_p': 0, 'p1': 1, 'p2': 2, 'annot_m': 3, 'm1': 4, 'm2': 5}
    else:
        n_rows = 4
        height_ratios = [1, 1, 1, 1]
        row_idx = {'p1': 0, 'p2': 1, 'm1': 2, 'm2': 3}

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(
        n_rows, 4,
        width_ratios=[1, 30, 0.8, 0.3],
        height_ratios=height_ratios,
        wspace=0.03, hspace=0.2
    )

    def make_row_axes(r):
        return fig.add_subplot(gs[r, 0]), fig.add_subplot(gs[r, 1])

    axes_map = {
        'p1': make_row_axes(row_idx['p1']),
        'p2': make_row_axes(row_idx['p2']),
        'm1': make_row_axes(row_idx['m1']),
        'm2': make_row_axes(row_idx['m2']),
    }
    if has_gene_annot:
        # annotation axes span only the expression column (col 1);
        # col 0 is hidden for these thin rows
        ax_annot_p = fig.add_subplot(gs[row_idx['annot_p'], 1])
        ax_annot_m = fig.add_subplot(gs[row_idx['annot_m'], 1])
        # hide the unused left cell
        for r in [row_idx['annot_p'], row_idx['annot_m']]:
            fig.add_subplot(gs[r, 0]).set_visible(False)

    gs_cb = gridspec.GridSpecFromSubplotSpec(
        5, 1, subplot_spec=gs[:, 3],
        height_ratios=[1, 2, 2, 2, 1], hspace=0.6
    )
    cax_cor  = fig.add_subplot(gs_cb[1, 0])
    cax_exp  = fig.add_subplot(gs_cb[2, 0])
    cax_cent = fig.add_subplot(gs_cb[3, 0])

    # ==========================================
    # 5. Gene-Annotation Plotter (NEW)
    # ==========================================
    def plot_gene_annotation(ax, gene_matrix, names, title):
        """Renders a (n_cors × n_genes) row-annotation heatmap."""
        im = ax.imshow(gene_matrix, aspect='auto',
                       cmap=cmap_cor, vmin=cor_vmin, vmax=cor_vmax,
                       interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.tick_params(axis='y', length=0)
        ax.set_title(title, fontsize=14, pad=6)
        return im

    # ==========================================
    # 6. Row Plotter (unchanged internally)
    # ==========================================
    def plot_row(ax_cor, ax_exp, cor_matrix, exp_matrix,
                 labels, ticks, cmap, vmin, vmax, title='', show_x=False):
        im_c = ax_cor.imshow(cor_matrix, aspect='auto',
                             cmap=cmap_cor, vmin=cor_vmin, vmax=cor_vmax)
        ax_cor.set_yticks(ticks)
        ax_cor.set_yticklabels(labels, fontsize=10)
        ax_cor.tick_params(axis='y', length=0)
        ax_cor.set_xticks(range(len(cor_names)))
        if show_x:
            ax_cor.set_xticklabels(cor_names, rotation=90, fontsize=9)
        else:
            ax_cor.set_xticklabels([])

        im_e = ax_exp.imshow(exp_matrix, aspect='auto',
                             cmap=cmap, vmin=vmin, vmax=vmax)
        if title and not has_gene_annot:   # title moves to annotation row when present
            ax_exp.set_title(title, fontsize=14, pad=10)
        ax_exp.set_yticks([])
        ax_exp.set_xticks([])
        if show_x:
            ax_exp.set_xlabel('Genes (Sorted Descending)')

        ax_right = ax_exp.twinx()
        ax_right.set_ylim(ax_exp.get_ylim())
        ax_right.set_yticks(row_ticks)
        ax_right.set_yticklabels(row_labels, fontsize=7, alpha=0.7, fontstyle='italic')
        ax_right.tick_params(axis='y', length=0)

        for y_div in range(1, exp_matrix.shape[0], 2):
            ax_cor.axhline(y=y_div + 0.5, color='black', linewidth=0.8, alpha=0.8)
            ax_exp.axhline(y=y_div + 0.5, color='black', linewidth=0.8, alpha=0.8)

        return im_c, im_e

    # ==========================================
    # 7. Execute Plotting
    # ==========================================
    # Gene-level annotations (drawn first so titles land correctly)
    if has_gene_annot:
        plot_gene_annotation(ax_annot_p, plus_gene_matrix,  gene_annot_names, 'Plus Strand Genes')
        plot_gene_annotation(ax_annot_m, minus_gene_matrix, gene_annot_names, 'Minus Strand Genes')

    im_cor, im_exp = plot_row(*axes_map['p1'], plus_cor_matrix,   plus_matrix,
                              p_labels, p_ticks, cmap_exp, exp_vmin, exp_vmax,
                              'Plus Strand Genes')
    _,      im_cent = plot_row(*axes_map['p2'], plus_cor_matrix,  plus_cent_matrix,
                               p_labels, p_ticks, cmap_cent, cent_vmin, cent_vmax, '')
    plot_row(*axes_map['m1'], minus_cor_matrix, minus_matrix,
             m_labels, m_ticks, cmap_exp, exp_vmin, exp_vmax, 'Minus Strand Genes')
    plot_row(*axes_map['m2'], minus_cor_matrix, minus_cent_matrix,
             m_labels, m_ticks, cmap_cent, cent_vmin, cent_vmax, '', show_x=True)

    # Colorbars
    fig.colorbar(im_cor,  cax=cax_cor,  label=label_cor)
    fig.colorbar(im_exp,  cax=cax_exp,  label=label_exp)
    fig.colorbar(im_cent, cax=cax_cent, label=label_cent)

    return fig