"""
Track Matching
==============

Helpers to compare pretrained AlphaGenome output tracks with the cardioid
multiome tracks and to pick the best-matching AlphaGenome tracks per cell type
(the CURIE lists used to aggregate pretrained-AlphaGenome VEP scores).

Inputs are the outputs of the track-matching scripts:

* RNA: an AnnData of exon counts per gene (``obs``) for predicted (``data_source == "AG"``)
  and observed (``"multiome"``) tracks (``var``), with the layer ``log_norm_counts``
  and Pearson correlations in ``varm["pear_r_log_norm_across_genes"]``.
* ATAC: a long table with one row per interval and track pair (``alphagenome_track``,
  ``BigWig_track``, ``pearson_r``, ``log_sum_pred_count``, ``log_sum_obs_count``).
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib_venn import venn2, venn2_circles
from scipy.stats import gaussian_kde


def plot_kde_density_scatter(x, y, ax, title):
    """Draws a scatter plot coloured by 2D Gaussian kernel density.

    NaN pairs are dropped and dense points are drawn on top. Falls back to a
    plain scatter for fewer than two points or constant input.

    Args:
        x (np.ndarray): Observed values (x axis).
        y (np.ndarray): Predicted values (y axis).
        ax (matplotlib.axes.Axes): Axes to draw on.
        title (str): Axes title.
    """
    # Filter out NaNs
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    
    if len(x) < 2 or np.var(x) == 0 or np.var(y) == 0:
        ax.scatter(x, y, color='blue', alpha=0.5, s=15)
    else:
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        # Sort by density so dense points are plotted last (on top)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        ax.scatter(x, y, c=z, cmap='viridis', s=15, edgecolor=None)
        
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Observed (log1p)")
    ax.set_ylabel("Predicted (log1p)")


def extreme_correlation_pairs(corr_matrix_df):
    """Returns the two highest and the two lowest correlated track pairs.

    Args:
        corr_matrix_df (pd.DataFrame): Correlation matrix. Its columns end up in
            ``Target_Track`` and its index in ``AG_Track``.

    Returns:
        list of dict: Four records with ``Target_Track``, ``AG_Track`` and
        ``Correlation``: the two highest pairs (ascending), then the two lowest.
    """
    melted = corr_matrix_df.unstack().reset_index()
    melted.columns = ['Target_Track', 'AG_Track', 'Correlation']
    melted = melted.dropna()
    
    sorted_df = melted.sort_values(by='Correlation')
    bottom_2 = sorted_df.head(2).to_dict('records')
    top_2 = sorted_df.tail(2).to_dict('records')
    
    return top_2 + bottom_2 # List of 4 dictionaries


def top_matching_tracks_per_target(df, top_n=5):
    """Lists the best-correlated columns for every row of a correlation matrix.

    Args:
        df (pd.DataFrame): Correlations with targets as rows and candidate tracks
            as columns.
        top_n (int, optional): Number of tracks per target. Defaults to 5.

    Returns:
        dict: ``{row label: [(column, correlation), ...]}``, highest first.
    """
    result_dict = {}
    
    # Iterate through each row in the dataframe
    for row_label, row_series in df.iterrows():
        # Get the top N values for the current row
        top_n_series = row_series.nlargest(top_n)
        
        # Zip the column names (index of the series) with their correlation values
        top_tuples = list(zip(top_n_series.index, top_n_series.values))
        
        # Store in the dictionary
        result_dict[row_label] = top_tuples
        
    return result_dict


def print_top_matching_tracks(consensus_tracks):
    """Prints the output of `top_matching_tracks_per_target`, five tracks per line.

    Args:
        consensus_tracks (dict): ``{target: [(track, correlation), ...]}``.
    """
    print("Consensus Top AG Tracks Across All Timepoints:\n" + "="*80)
    for ct, tracks in consensus_tracks.items():
        print(f"\n{ct}:")
        if tracks:
            track_strs = [f"✅ {track} (Avg R={avg_r:.3f})" for track, avg_r in tracks]
            for i in range(0, len(track_strs), 5):
                chunk = track_strs[i:i+5]
                print("  " + " | ".join(chunk))
        else:
            print("  ❌ No shared tracks found. Try increasing 'top_n'.")


def plot_track_match_correlations_by_assay(df_atac, df_rna, rna_adata, target_cell_types):
    """Plots ATAC and RNA track-match correlations side by side per cell type.

    For each target cell type, one ATAC and one RNA box (with all AlphaGenome
    tracks as points) are drawn and the top five AlphaGenome tracks are labelled.
    Underscores in target names are replaced by hyphens before matching. Sets
    the seaborn style to ``"white"``.

    Args:
        df_atac (pd.DataFrame): ATAC correlations (BigWig tracks x AlphaGenome tracks).
        df_rna (pd.DataFrame): RNA correlations (AlphaGenome tracks x observed tracks).
        rna_adata (anndata.AnnData): RNA AnnData; ``var["biosample_name"]`` labels the RNA tracks.
        target_cell_types (list of str): Target tracks to show, in plotting order.

    Returns:
        matplotlib.axes.Axes: The plot axes.
    """
    
    melted_atac = df_atac.T.unstack().reset_index()
    melted_atac.columns = ['Target_Track', 'AG_Track', 'Correlation']
    melted_atac['Assay'] = 'ATAC'
    melted_atac['Annotation_Label'] = melted_atac['AG_Track'].astype(str)

    melted_rna = df_rna.T.unstack().reset_index()
    melted_rna.columns = ['AG_Track', 'Target_Track', 'Correlation']
    curie_to_name = rna_adata.var['biosample_name'].to_dict()
    melted_rna['Assay'] = 'RNA'
    melted_rna['Annotation_Label'] = melted_rna['AG_Track'].map(curie_to_name).astype(str)

    df_combined = pd.concat([
        melted_atac[['Target_Track', 'Correlation', 'Assay', 'Annotation_Label']],
        melted_rna[['Target_Track', 'Correlation', 'Assay', 'Annotation_Label']]
    ])
    
    df_combined['Target_Track'] = df_combined['Target_Track'].str.replace('_', '-')
    standardized_targets = [ct.replace('_', '-') for ct in target_cell_types]
    
    df_subset = df_combined[df_combined['Target_Track'].isin(standardized_targets)].copy()

    sns.set_style("white")
    plt.figure(figsize=(10, 6))

    shared_hue_order = ['ATAC', 'RNA']
    box_width = 0.5
    
    # The offset is strictly a quarter of the total width when dealing with two 
    # categories. This isolates the exact geometric center of each dodged box so 
    # we can anchor both the scatter points and text accurately.
    center_offset = box_width / 4

    ax = sns.boxplot(
        data=df_subset, 
        x='Target_Track', 
        y='Correlation', 
        hue='Assay',
        hue_order=shared_hue_order,
        order=standardized_targets,
        palette={'ATAC': 'lightblue', 'RNA': 'orange'},
        showfliers=False,
        linewidth=1,
        width=box_width
    )

    for patch in ax.patches:
        patch.set_edgecolor('black')
        patch.set_linewidth(1)

    # Seaborn's stripplot decouples its dodge spacing from the boxplot width,
    # which causes visual drift. Calculating the coordinates manually via scatter
    # ensures the dots are locked to the center of the customized boxes.
    np.random.seed(42)
    for i, target in enumerate(standardized_targets):
        for assay in shared_hue_order:
            group_df = df_subset[(df_subset['Target_Track'] == target) & (df_subset['Assay'] == assay)]
            if not group_df.empty:
                y_vals = group_df['Correlation'].values
                
                offset = -center_offset if assay == 'ATAC' else center_offset
                x_center = i + offset
                
                # The jitter bounds are kept proportionally tight relative to the new 
                # width so points do not spill over the box edges.
                jitter = np.random.uniform(-0.06, 0.06, size=len(y_vals))
                x_vals = x_center + jitter
                
                ax.scatter(x_vals, y_vals, color='black', alpha=0.6, s=4, zorder=3)

    ax.axhline(0, color='black', linewidth=1, zorder=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)

    #plt.xticks(rotation=30, ha='right')
    #plt.title("Pearson Correlation Distributions: ATAC vs RNA", fontsize=16)

    texts = []
    for i, target in enumerate(standardized_targets):
        for assay in shared_hue_order:
            group_df = df_subset[(df_subset['Target_Track'] == target) & (df_subset['Assay'] == assay)]
            if not group_df.empty:
                offset = -center_offset if assay == 'ATAC' else center_offset
                x_pos = i + offset

                top_5 = group_df.nlargest(5, 'Correlation')
                for _, row in top_5.iterrows():
                    text_color = 'darkblue' if assay == 'ATAC' else 'darkred'
                    texts.append(ax.text(
                        x_pos, 
                        row['Correlation'], 
                        row['Annotation_Label'], 
                        fontsize=8, 
                        color=text_color
                    ))

    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5), ax=ax)
    
    plt.tight_layout()
    plt.show()

    return ax


def plot_atac_rna_track_overlap(atac_set, rna_set, title=None, figsize=(5, 4)):
    """Plots a Venn diagram of the AlphaGenome tracks selected via ATAC and via RNA.

    Args:
        atac_set (set or list): Track names selected from the ATAC matching.
        rna_set (set or list): Track names selected from the RNA matching.
        title (str, optional): Plot title. Defaults to None.
        figsize (tuple, optional): Figure size. Defaults to (5, 4).
    """
    # Converting inputs to sets ensures the Venn calculation accurately 
    # computes unique and intersecting elements regardless of the input data type.
    atac_elements = set(atac_set)
    rna_elements = set(rna_set)

    plt.figure(figsize=figsize)

    # We map the established color scheme directly into the venn tool.
    # Alpha is adjusted to 0.7 so the overlap region forms a distinct composite color.
    v = venn2(
        [atac_elements, rna_elements], 
        set_labels=('ATAC', 'RNA'), 
        set_colors=('lightblue', 'orange'),
        alpha=0.7
    )

    # Forcing all subset numbers to render in black guarantees high contrast 
    # against both the lightblue and orange backgrounds.
    for subset_id in ['10', '01', '11']:
        label = v.get_label_by_id(subset_id)
        if label:
            label.set_color('black')
            label.set_fontsize(12)

    # The venn2_circles function maps the explicit black outlines independently 
    # of the colored patches generated by venn2.
    venn2_circles(
        [atac_elements, rna_elements], 
        color='black', 
        linewidth=1.5
    )

    if title:
        plt.title(title, fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_atac_track_match_scatter(ATAC_data, df_atac_pearson, ag_tracks, bw_target, n_rows = 1):
    """Plots predicted vs observed ATAC counts of several AlphaGenome tracks for one target.

    Each panel shows the per-interval log1p summed counts of one AlphaGenome
    track against the observed BigWig ``bw_target``, coloured by density.

    Args:
        ATAC_data (pd.DataFrame): Long ATAC table (``BigWig_track``, ``alphagenome_track``,
            ``log_sum_pred_count``, ``log_sum_obs_count``).
        df_atac_pearson (pd.DataFrame): Correlations (BigWig tracks x AlphaGenome
            tracks), shown in the panel titles.
        ag_tracks (list of str): AlphaGenome tracks (biosample names) to plot.
        bw_target (str): Observed BigWig track.
        n_rows (int, optional): Rows of the panel grid. Defaults to 1.
    """
    n_tracks = len(ag_tracks)
    n_cols = math.ceil(n_tracks / n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    fig.suptitle(f"ATAC Log1p Counts: Predicted vs Observed {bw_target} all Intervals", fontsize=16)

    target_data = ATAC_data[(ATAC_data.BigWig_track == bw_target)]
    axes = axes.flatten()
    for ax, ag_track in zip(axes, ag_tracks):
        pred_vals = target_data.loc[(target_data.alphagenome_track == ag_track), "log_sum_pred_count"].values
        obs_vals = target_data.loc[(target_data.alphagenome_track == ag_track), "log_sum_obs_count"].values

        r_val = df_atac_pearson.loc[bw_target, ag_track]
        title = f"Target: {bw_target}\nPred: {ag_track}\nbp res. Pearson R: {r_val:.3f}"

        plot_kde_density_scatter(obs_vals, pred_vals, ax, title)

    for i in range(n_tracks, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def plot_rna_track_match_scatter(RNA_adata, df_rna_pearson, ag_tracks, bw_target, n_rows = 1):
    """Plots predicted vs observed RNA gene counts of several AlphaGenome tracks for one target.

    Uses the layer ``log_norm_counts`` (quantile-normalised, gene-centred log
    counts). All AlphaGenome tracks whose ``biosample_name`` is in ``ag_tracks``
    are shown.

    Args:
        RNA_adata (anndata.AnnData): Genes x tracks, predicted and observed.
        df_rna_pearson (pd.DataFrame): Correlations (AlphaGenome tracks x observed
            tracks), shown in the panel titles.
        ag_tracks (list of str): AlphaGenome biosample names to plot.
        bw_target (str): Observed track (a ``var_name`` of ``RNA_adata``).
        n_rows (int, optional): Rows of the panel grid. Defaults to 1.
    """
    track_idx = RNA_adata.var[RNA_adata.var.biosample_name.isin(ag_tracks)].index

    n_tracks = len(ag_tracks)
    n_cols = math.ceil(n_tracks / n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    fig.suptitle(f"RNA Log1p Counts: Predicted vs Observed {bw_target} all Genes", fontsize=16)

    obs_vals = RNA_adata[:, bw_target].layers["log_norm_counts"].flatten()
    axes = axes.flatten()
    for ax, ag_track in zip(axes, track_idx):
        pred_vals = RNA_adata[:, ag_track].layers["log_norm_counts"].flatten()
        r_val = df_rna_pearson.loc[ag_track, bw_target]
        title = f"Target: {bw_target}\nPred: {ag_track}\nPearson R: {r_val:.3f}"
        plot_kde_density_scatter(obs_vals, pred_vals, ax, title)

    for i in range(n_tracks, len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.show()


def plot_track_match_scatter_by_assay(
    atac_data, 
    df_atac_pearson, 
    atac_tracks, 
    rna_adata, 
    df_rna_pearson, 
    rna_tracks, 
    bw_target, 
    n_cols=4
):
    """Plots ATAC and RNA predicted-vs-observed scatters for one target in one grid.

    ATAC panels are outlined in light blue, RNA panels in orange; the figure has
    shared axis labels.

    Args:
        atac_data (pd.DataFrame): Long ATAC table (see `plot_atac_track_match_scatter`).
        df_atac_pearson (pd.DataFrame): ATAC correlations (not shown; kept for symmetry).
        atac_tracks (list of str): AlphaGenome ATAC tracks to plot.
        rna_adata (anndata.AnnData): RNA AnnData (see `plot_rna_track_match_scatter`).
        df_rna_pearson (pd.DataFrame): RNA correlations (not shown; kept for symmetry).
        rna_tracks (list of str): AlphaGenome RNA biosample names to plot.
        bw_target (str): Observed target track (same name in ATAC and RNA data).
        n_cols (int, optional): Columns of the panel grid. Defaults to 4.
    """
    
    # We map the RNA track names to their corresponding AnnData index values upfront
    # so we know exactly how many valid RNA plots will be generated.
    rna_track_indices = rna_adata.var[rna_adata.var.biosample_name.isin(rna_tracks)].index
    
    total_plots = len(atac_tracks) + len(rna_track_indices)
    n_rows = math.ceil(total_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))    
    # Flattening allows us to treat the 2D axis grid as a simple 1D list, 
    # making continuous iteration across two different datasets straightforward.
    axes_flat = axes.flatten()
    ax_idx = 0
    
    # Pre-subsetting the ATAC data by target prevents redundant filtering inside the loop
    target_data = atac_data[(atac_data.BigWig_track == bw_target)]
    
    # Plotting ATAC data
    for ag_track in atac_tracks:
        ax = axes_flat[ax_idx]
        
        pred_vals = target_data.loc[(target_data.alphagenome_track == ag_track), "log_sum_pred_count"].values
        obs_vals = target_data.loc[(target_data.alphagenome_track == ag_track), "log_sum_obs_count"].values        
        plot_kde_density_scatter(obs_vals, pred_vals, ax, title = "")
        
        # Stripping local axis labels to rely on the global figure labels later
        ax.set_xlabel("")
        ax.set_ylabel("")
        
        # Applying a thick lightblue outline to denote ATAC plots
        for spine in ax.spines.values():
            spine.set_color('lightblue')
            spine.set_linewidth(3)
            
        ax_idx += 1

    # Pre-extracting RNA observed data since the target remains constant for all RNA plots
    rna_obs_vals = rna_adata[:, bw_target].layers["log_norm_counts"].flatten()

    # Plotting RNA data
    for ag_track in rna_track_indices:
        ax = axes_flat[ax_idx]
        
        pred_vals = rna_adata[:, ag_track].layers["log_norm_counts"].flatten()        
        plot_kde_density_scatter(rna_obs_vals, pred_vals, ax, title = "")
        
        ax.set_xlabel("")
        ax.set_ylabel("")
        
        # Applying a thick orange outline to denote RNA plots
        for spine in ax.spines.values():
            spine.set_color('orange')
            spine.set_linewidth(3)
            
        ax_idx += 1

    # Cleaning up any unused subplots created by the grid rounding
    for i in range(ax_idx, len(axes_flat)):
        fig.delaxes(axes_flat[i])

    # Setting global axis labels for the entire figure
    fig.supxlabel("Observed log1p Counts", fontsize=26, y = 0)
    fig.supylabel("Predicted log1p Counts", fontsize=26, x = 0)

    plt.tight_layout()
    plt.show()
