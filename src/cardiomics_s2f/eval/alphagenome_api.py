"""
Pretrained AlphaGenome Tracks
=============================

Helpers for track predictions of the pretrained AlphaGenome model (AlphaGenome
API) and for matching its output tracks to the cardioid multiome data:

* aggregation of predicted tracks over groups of ontology CURIEs and the
  matching track table for evaluation against BigWigs;
* RNA track matching: gene filtering, TSS-anchored prediction windows, exon sums
  of predicted signal, and the combined AnnData with track-by-track correlations;
* ATAC track matching: grouping of regions into prediction windows, reading the
  observed BigWig signal and per-window, NaN-masked Pearson correlations.

The runnable scripts are in ``workflows/alphagenome_api/``; the analysis of the
matching results is in ``cardiomics_s2f.eval.track_matching``.
"""

import gc
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyBigWig
from alphagenome.data import genome
from tqdm import tqdm

from cardiomics_s2f.eval.track_correlation import quantile_normalize_RNA_counts


def aggregate_tracks_by_curies(P, meta, config, keys, strand=None):
    """Averages predicted tracks over groups of ontology CURIEs.

    Args:
        P (np.ndarray): Predictions of shape ``(n_tracks, n_bins)``.
        meta (pd.DataFrame): Track metadata (``ontology_curie``, ``strand``), one row per track.
        config (dict): ``{group: [CURIE, ...]}``.
        keys (list of str): Groups to aggregate, in output order.
        strand (str, optional): Only use tracks on this strand (``"+"`` or ``"-"``).
            Defaults to None (all tracks).

    Returns:
        np.ndarray: Float32 array of shape ``(len(keys), n_bins)``.
    """
    t, b = P.shape
    P_agg = np.empty((len(keys), b), dtype=np.float32)
    
    strand_mask = meta["strand"] == strand if strand is not None else np.ones(len(meta), dtype=bool)
    
    for a_idx, key in enumerate(keys):
        c_mask = strand_mask & meta["ontology_curie"].isin(config[key])
        P_agg[a_idx, :] = np.mean(P[c_mask, :], axis=0)
    
    return P_agg


def masked_pearson_per_window(P, T, min_valid=50000):
    # Create masks and safe-T
    """Pearson correlation of every predicted with every observed track, per window.

    NaNs in the observed signal are masked. A correlation is only computed if at
    least ``min_valid`` positions are valid and both tracks vary; otherwise it is
    NaN. ``T`` is modified in place (NaNs set to 0) to save memory.

    Args:
        P (np.ndarray): Predictions of shape ``(n_windows, seq_len, n_pred_tracks)``.
        T (np.ndarray): Observed signal of shape ``(n_windows, seq_len, n_obs_tracks)``.
        min_valid (int, optional): Minimum number of valid positions. Defaults to 50000.

    Returns:
        tuple: ``(R, sum_T, sum_P)``, each of shape ``(n_windows, n_pred_tracks, n_obs_tracks)``:
        the correlations and the summed observed and predicted signal over valid positions.
    """
    T_valid_mask = (~np.isnan(T)).astype(np.uint8) # 1.0 if valid, 0.0 if NaN use unsigned int datatype to save more mem
    np.nan_to_num(T, nan=0.0, copy=False)             # Safe for math; masked out later modify in place to save RAM
    
    # Count valid elements per batch -> shape: (Batch, 1, N_tracks)
    C = np.sum(T_valid_mask, axis=1)[:, np.newaxis, :]
    C_safe = np.maximum(C, 1)
    
    # Calculate sums per interval
    sum_P = np.einsum('blm,bln->bmn', P, T_valid_mask)
    sum_P2 = np.einsum('blm,bln->bmn', P**2, T_valid_mask)

    del T_valid_mask; gc.collect()

    sum_T = np.sum(T, axis=1)[:, np.newaxis, :]
    sum_PT = np.einsum('blm,bln->bmn', P, T)
    sum_T2 = np.sum(T**2, axis=1)[:, np.newaxis, :]

    mu_P = sum_P / C_safe
    mu_T = sum_T / C_safe
    
    cov = sum_PT - (C * mu_P * mu_T)
    var_P = sum_P2 - (C * (mu_P**2))
    var_T = sum_T2 - (C * (mu_T**2))
    
    std_product = np.sqrt(np.maximum(var_P, 0) * np.maximum(var_T, 0))
    valid_combo_mask = (C >= min_valid) & (std_product > 1e-12)
    
    # R_indiv shape: (Batch, M_pred_tracks, N_bw_tracks)
    R_indiv = np.divide(cov, std_product, out=np.full_like(cov, np.nan), where=valid_combo_mask)
    
    # Broadcast sum_T to match sum_P's 3D shape (Batch, M, N) so they align in the long dataframe
    sum_T = np.broadcast_to(sum_T, sum_P.shape)
    
    return R_indiv, sum_T, sum_P


def prediction_track_table(reference_tracks):
    """Builds the track table of aggregated pretrained-AlphaGenome predictions.

    The predictions of each aggregation group are evaluated against the BigWig of
    one reference cell type, given explicitly by ``reference_tracks`` (e.g. the
    naive heart aggregation against Cardiomyocytes). Rows follow the prediction
    layout: all groups for ATAC, then RNA plus strand, then RNA minus strand.

    Args:
        reference_tracks (dict): ``{group: cell type}``, in output order.

    Returns:
        pd.DataFrame: ``cell_type`` (group), ``Assay_type``, ``strand`` and the
        reference BigWig ``filename`` per predicted track.
    """
    keys = list(reference_tracks.keys())
    cts = [reference_tracks[k] for k in keys]
    a = len(keys)
    return pd.DataFrame({
        "cell_type": keys * 3,
        "Assay_type": ["ATAC"] * a + ["RNA"] * 2 * a,
        "strand": ["."] * a + ["plus"] * a + ["minus"] * a,
        "filename": [f"ATAC_{ct}_100M_norm.bw" for ct in cts]
                    + [f"RNA_{ct}_plus_100M.bw" for ct in cts]
                    + [f"RNA_{ct}_minus_100M.bw" for ct in cts],
    })


# --- RNA track matching -----------------------------------------------------


def filter_multiome_genes(umi_counts, gene_pct, n_top_var_genes=5000):
    """Selects expressed, variable genes from pseudobulk RNA counts.

    Removes ribosomal and mitochondrial genes, genes expressed in at most 5% of
    the cells of every pseudobulk and genes with ``log1p(total counts) <= 6.5``,
    then keeps the ``n_top_var_genes`` genes with the highest variance across
    pseudobulks.

    Args:
        umi_counts (pd.DataFrame): Counts (genes x pseudobulks).
        gene_pct (pd.DataFrame): Percentage of expressing cells, same shape.
        n_top_var_genes (int, optional): Number of genes kept. Defaults to 5000.

    Returns:
        anndata.AnnData: Genes (``obs``) x pseudobulks (``var``), layer ``pct``.
    """
    adata = ad.AnnData(X = umi_counts)
    adata.layers["pct"] = gene_pct

    starting_genes = len(adata.obs)
    adata = adata[~adata.obs.index.str.startswith(("RPS", "RPL", "MT-"))].copy() #remove ribosome and mitochondiral genes
    max_pct = adata.layers["pct"].max(axis = 1) #remove genes that are neve expressed in more than 5% of cells of a pseudobulk
    adata = adata[max_pct > 5]
    adata = adata[np.log1p(adata.X.sum(axis = 1)) > 6.5] #remove lower tal of genes with bariyl any expression

    #slice for the top 5000 most variable genes
    topvar_idx = np.argsort(np.var(adata.X, axis=1))[-n_top_var_genes:]
    adata = adata[topvar_idx].copy()

    print(f"RNA filtering complete. Genes retained: {len(adata.obs)} / {starting_genes}")
    return adata


def gene_exon_annotation(gtf_file):
    """Reads protein-coding and processed-transcript genes with their exons from a GTF.

    Keeps chromosomes 1-22 and X (no chrM, chrY).

    Args:
        gtf_file (str or Path): GENCODE GTF.

    Returns:
        pd.DataFrame: One row per gene (``gene`` features) with ``gene_name``,
        ``gene_type``, ``gene_id`` and ``exon_intervals`` (list of ``genome.Interval``).
    """
    print("Parsing GTF file...")
    gtf = pd.read_csv(gtf_file, sep="\t", comment="#", header=None)
    gtf.columns = ["chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]

    gtf["gene_name"] = gtf["attribute"].str.extract(r'gene_name\s+"([^"]+)"')
    gtf["gene_type"] = gtf["attribute"].str.extract(r'gene_type\s+"([^"]+)"')
    gtf["gene_id"] = gtf["attribute"].str.extract(r'gene_id\s+"([^"]+)"')

    gtf = gtf[
        (gtf.chrom.str.contains("chr")) &
        (~gtf.chrom.str.contains("chrM|chrY")) &
        (gtf.gene_type.str.contains("protein_coding|processed_transcript"))
        ].drop_duplicates()

    print("Extracting and mapping exons...")
    exons_df = gtf[gtf["feature"] == "exon"].copy()
    exons_df["interval"] = exons_df.apply(
        lambda row: genome.Interval(
            chromosome=row["chrom"], start=row["start"], end=row["end"],
            strand=row["strand"], name=row["gene_name"]
        ), axis=1
    )
    exon_groups = exons_df.groupby("gene_id")["interval"].apply(list)

    gtf = gtf[gtf["feature"] == "gene"].copy()
    gtf["exon_intervals"] = gtf["gene_id"].map(exon_groups).apply(lambda d: d if isinstance(d, list) else [])
    return gtf


def gene_prediction_windows(gtf, seq_len, chrom_sizes_file):
    """Places one prediction window per gene and keeps genes that fit inside it.

    The window (``seq_len``) is centred on the TSS and shifted by a sixth of its
    width towards the gene body (one third upstream, two thirds downstream).
    Windows that cross a chromosome start or end are moved 10 kb inside it.
    Genes not fully inside their window are removed.

    Args:
        gtf (pd.DataFrame): Genes from `gene_exon_annotation`.
        seq_len (int): Window width (an AlphaGenome sequence length).
        chrom_sizes_file (str or Path): Two-column file ``chrom<TAB>size``.

    Returns:
        pd.DataFrame: ``gtf`` with ``TSS``, ``gene_interval``, ``input_interval``,
        ``adj_input_interval`` and ``chrom_end_overlap``.
    """
    print("Calculating TSS and resizing intervals...")
    gtf["TSS"] = np.where(gtf["strand"] == "+", gtf["start"], gtf["end"])
    gtf["gene_interval"] = [genome.Interval(chromosome=row["chrom"], start=row["TSS"], end=row["TSS"], strand=row["strand"], name=row["gene_name"]) for _, row in gtf.iterrows()]

    intervals = [interval.resize(seq_len) for interval in gtf["gene_interval"]]
    shift = int(np.round(intervals[0].width / 6, 0))
    gtf["input_interval"] = [interval.shift(shift, use_strand=True) for interval in intervals]

    # Load chrom sizes
    chrom_size = pd.read_csv(chrom_sizes_file, sep="\t", names=["chrom", "end"])
    chrom_size_dict = dict(zip(chrom_size["chrom"], chrom_size["end"]))

    print("Adjusting intervals for chromosome boundaries...")
    adj_intervals, overlaps = [], []
    for interval in gtf["input_interval"]:
        chr_end = chrom_size_dict.get(interval.chromosome)

        if interval.start <= 0:
            ret = interval.shift(-interval.start + 10000, use_strand=False)
            overlap = "chrom_start"
        elif chr_end is not None and interval.end >= chr_end:
            # FIXED: Shift left using a negative value
            ret = interval.shift(chr_end - interval.end - 10000, use_strand=False)
            overlap = "chrom_end"
        else:
            ret = interval
            overlap = "no_overlap"

        adj_intervals.append(ret)
        overlaps.append(overlap)

    gtf["adj_input_interval"] = adj_intervals
    gtf["chrom_end_overlap"] = overlaps

    # Filter out genes not fully within the adjusted interval
    start_mask = gtf["start"] >= [interval.start for interval in gtf["adj_input_interval"]]
    end_mask = gtf["end"] <= [interval.end for interval in gtf["adj_input_interval"]]
    gtf = gtf.loc[start_mask & end_mask, :]
    return gtf


def attach_gene_windows(adata, gtf):
    """Restricts the genes to those with a unique prediction window and adds the windows to ``obs``.

    Args:
        adata (anndata.AnnData): Genes from `filter_multiome_genes` (``obs`` index = gene names).
        gtf (pd.DataFrame): Genes with windows from `gene_prediction_windows`.

    Returns:
        anndata.AnnData: Genes present exactly once in ``gtf``, with its columns joined to ``obs``.
    """
    gtf_filtered = gtf[gtf.gene_name.isin(adata.obs.index)]
    double = gtf_filtered.gene_name.value_counts()[gtf_filtered.gene_name.value_counts() > 1].index
    gtf_filtered = gtf_filtered[~gtf_filtered.gene_name.isin(double)].set_index("gene_name")

    adata = adata[adata.obs.index.isin(gtf_filtered.index)]
    adata.obs = adata.obs.join(gtf_filtered, how="left").copy()
    return adata


def sum_exon_signal(pos_counts, interval_start, exon_intervals, exon_length_norm=False):
    """Sums predicted signal over a gene's exons (cumulative-sum lookup).

    Exon coordinates are GTF-style (1-based, inclusive end).

    Args:
        pos_counts (np.ndarray): Predictions of shape ``(seq_len, n_tracks)``.
        interval_start (int): Genomic start of the prediction window.
        exon_intervals (list): The gene's exons (``start``, ``end`` attributes).
        exon_length_norm (bool, optional): Divide by the total exon length (mean
            instead of sum). Defaults to False.

    Returns:
        np.ndarray: One value per track.
    """
    cumsum = np.vstack([np.zeros((1, pos_counts.shape[1])), np.cumsum(pos_counts, axis=0)])
    starts = np.array([exon.start - interval_start for exon in exon_intervals])
    ends = np.array([exon.end - interval_start + 1 for exon in exon_intervals])  #+1 to handle 1-index based GTF coordinates in python
    exon_sums = cumsum[ends] - cumsum[starts]
    total_exon_counts = exon_sums.sum(axis=0)

    #run exon length norm or return total counts
    if exon_length_norm:
        total_exon_length = max(np.sum(ends - starts), 1) # max(sum, 1) to prevent div-by-zero in case of no exon found
        return total_exon_counts / total_exon_length
    return total_exon_counts


def assemble_rna_track_match(pred_counts, metadata, adata):
    """Combines predicted and observed exon counts and correlates every track pair.

    Both sides get the layers ``raw_sum_counts``, ``log_counts`` (log1p) and
    ``log_norm_counts`` (quantile-normalised, gene-centred). Pearson correlations
    across genes of all track pairs are stored in ``varm``.

    Args:
        pred_counts (dict): ``{gene: pd.Series}`` of predicted exon sums, indexed
            by ``(name, strand_type)`` of the AlphaGenome RNA tracks.
        metadata (pd.DataFrame): AlphaGenome RNA-seq track metadata with ``strand_type``.
        adata (anndata.AnnData): Observed counts with gene windows (`attach_gene_windows`).

    Returns:
        anndata.AnnData: Genes x (AlphaGenome + multiome) tracks; ``var["data_source"]``
        is ``"AG"`` or ``"multiome"``; ``varm["pear_r_log_norm_across_genes"]`` and
        ``varm["pear_r_log_norm"]`` hold the correlations of the normalised and log
        counts. Object columns of ``obs`` are converted to strings (exons dropped) so
        the result can be written to ``.h5ad``.
    """
    #make indexes for track metadata
    var = metadata.drop_duplicates(subset=['name', 'biosample_name', 'Assay title', 'ontology_curie', 'biosample_type',
            'biosample_life_stage', 'gtex_tissue', 'data_source', 'endedness',
            'genetically_modified', 'nonzero_mean', "strand_type"]).drop("strand", axis = 1).copy()
    var_multi_index = pd.MultiIndex.from_frame(var[["name", "strand_type"]])

    #take predicted counts and covnert to anndata
    pred_df = pd.DataFrame(pred_counts).T
    pred_df = pred_df.loc[:, var_multi_index] #reorder to match metadata
    flattened_index = [f"{name}_{strand}" for name, strand in var_multi_index] #collapse multiindex for adata
    var.index = flattened_index
    pred_df.columns = flattened_index

    pred_adata = ad.AnnData(X=pred_df, var=var, obs=adata.obs)
    pred_adata.layers["raw_sum_counts"] = pred_adata.X.copy()
    pred_adata.layers["log_counts"] = np.log1p(pred_df)
    pred_adata.layers["log_norm_counts"] = quantile_normalize_RNA_counts(pred_adata.layers["log_counts"])

    #add lof transform and normalization also to obs adata
    adata.layers["raw_sum_counts"] = adata.X.copy()
    adata.layers["log_counts"] = np.log1p(adata.layers["raw_sum_counts"]).copy()
    adata.layers["log_norm_counts"] = quantile_normalize_RNA_counts(adata.layers["log_counts"])

    # Merge with observed feature counts
    combined_adata = ad.concat(
        [pred_adata, adata],
        axis=1,
        join="outer",
        label="data_source",
        keys=["AG", "multiome"]
    )
    combined_adata.obs = pred_adata.obs.copy()

    #calucalete the correaltions
    print("Calculating Pearson and Spearman correlation matrices...")
    r_norm = np.corrcoef(combined_adata.layers["log_norm_counts"], rowvar=False)
    combined_adata.varm["pear_r_log_norm_across_genes"] = r_norm

    r_log = np.corrcoef(combined_adata.layers["log_counts"], rowvar=False)
    combined_adata.varm["pear_r_log_norm"] = r_log

    #save data
    #drop the exons to make them savable
    combined_adata.obs = combined_adata.obs.drop(columns=['exon_intervals'])
    #safety check
    for col in combined_adata.obs.columns:
        if combined_adata.obs[col].dtype == 'object':
            combined_adata.obs[col] = combined_adata.obs[col].astype(str)
    return combined_adata


# --- ATAC track matching ----------------------------------------------------


def group_regions_into_windows(regions_df, seq_len, max_span):
    """Groups nearby regions into shared prediction windows (greedy 1D sweep).

    Regions are sorted by position; consecutive regions on a chromosome are put
    in one group while the group spans at most ``max_span``. Each group gets a
    window of ``seq_len`` centred on its span (start clipped at 0).

    Args:
        regions_df (pd.DataFrame): Regions with ``seqnames``, ``start`` and ``end``.
        seq_len (int): Window width.
        max_span (int): Maximum width covered by the regions of one group.

    Returns:
        list of tuple: ``(chrom, start, end)`` per window.
    """
    regions_df = regions_df.sort_values(by=["seqnames", "start", "end"]).reset_index(drop=True)

    windows = []
    current_chrom = None
    group_start = -1
    group_end = -1

    def close_group():
        if current_chrom is None: return
        # Calculate the true center of the group's span
        center = (group_start + group_end) // 2
        # Prevent negative start coordinates near the start of a chromosome
        api_start = max(0, center - (seq_len // 2))
        windows.append((current_chrom, api_start, api_start + seq_len))

    for _, row in tqdm(regions_df.iterrows(), total=len(regions_df), desc="Grouping regions"):
        chrom = row["seqnames"]
        start = row["start"]
        end = row["end"]
        # If we switch chromosomes OR the group becomes wider than max_span, close it and start a new one
        if chrom != current_chrom or (end - group_start) > max_span:
            close_group()
            current_chrom = chrom
            group_start = start
            group_end = end
        else:
            # Region fits within the max_span limit; extend the group's right boundary
            group_end = max(group_end, end)

    # Close the final pending group
    close_group()
    return windows


def list_atac_bigwigs(bw_dir):
    """Lists the 100M-normalised ATAC BigWigs of a folder.

    Args:
        bw_dir (str or Path): Folder with ``ATAC_<cell type>_100M_norm.bw`` files.

    Returns:
        pd.DataFrame: ``cell_type`` and ``path`` per BigWig.
    """
    bw_paths_list = []
    for bw_path in tqdm(list(Path(bw_dir).glob("*_100M_norm.bw")), desc="Parsing BigWig paths"):
        filename = bw_path.name
        # Strip suffix and "ATAC_" prefix
        ct = filename.replace("_100M_norm.bw", "").replace("ATAC_", "")
        bw_paths_list.append({
            "cell_type": ct,
            "path": str(bw_path),
        })
    return pd.DataFrame(bw_paths_list)


def read_bigwig_windows(bw_paths, intervals_data):
    """Reads the base-resolution BigWig signal of all windows.

    Windows outside a chromosome (or on chromosomes missing from a BigWig) are
    filled with NaN, as are positions without signal.

    Args:
        bw_paths (pd.DataFrame): ``cell_type`` and ``path`` (`list_atac_bigwigs`).
        intervals_data (list of tuple): ``(chrom, start, end)`` per window.

    Returns:
        tuple: ``(T, bw_track_names)``; ``T`` is float32 of shape
        ``(n_windows, window_len, n_bigwigs)``.
    """
    target_list = [[] for _ in range(len(intervals_data))]
    bw_track_names = []

    for _, bw_row in tqdm(bw_paths.iterrows(), total=len(bw_paths), desc="Extracting BigWig tracks"):
        bw_name = f'{bw_row["cell_type"]}'
        bw_track_names.append(bw_name)

        # Open each BigWig exactly once
        bw_file = pyBigWig.open(bw_row["path"])
        valid_chroms = bw_file.chroms() # Dictionary of {chrom: length}

        for i, (chrom, start, end) in enumerate(intervals_data):
            expected_length = end - start

            # SAFETY CHECK: Does the chromosome exist AND are the coordinates within bounds?
            if chrom not in valid_chroms or start < 0 or end > valid_chroms[chrom]:
                # Interval is out of bounds. Fill with pure NaNs.
                # masked_pearson_per_window will correctly ignore these.
                target_list[i].append(np.full(expected_length, np.nan, dtype=np.float32))
                continue

            # Leave them as NaNs so the downstream masking function detects them.
            vals = bw_file.values(chrom, start, end)
            target_list[i].append(np.array(vals, dtype=np.float32))

        bw_file.close()

    # T shape: (num_intervals, seq_len, num_bw_tracks)
    # We swap axes so the shape is (Interval, seq Length, Tracks) to match P
    T = np.array(target_list, dtype=np.float32).transpose(0, 2, 1)
    return T, bw_track_names


def track_match_long_table(R_indiv, T_counts, P_counts, api_labels, pred_track_names, bw_track_names):
    """Flattens per-window correlations and signal sums into a long table.

    Args:
        R_indiv (np.ndarray): Correlations ``(n_windows, n_pred, n_obs)`` from `masked_pearson_per_window`.
        T_counts (np.ndarray): Summed observed signal, same shape.
        P_counts (np.ndarray): Summed predicted signal, same shape.
        api_labels (list of str): Window labels (``chrom:start-end``).
        pred_track_names (list of str): AlphaGenome track names.
        bw_track_names (list of str): Observed BigWig track names.

    Returns:
        pd.DataFrame: One row per window and track pair with ``Interval``,
        ``alphagenome_track``, ``BigWig_track``, ``pearson_r``, ``sum_pred_count``,
        ``sum_obs_count`` and their ``log_*`` (log1p); rows without a correlation are dropped.
    """
    idx = pd.MultiIndex.from_product(
        [api_labels, pred_track_names, bw_track_names],
        names=["Interval", "alphagenome_track", "BigWig_track"]
    )

    df_indiv_long = pd.DataFrame({
        "pearson_r": R_indiv.flatten(),
        "sum_pred_count": P_counts.flatten(),
        "sum_obs_count": T_counts.flatten()
    }, index=idx).reset_index()

    df_indiv_long["log_sum_pred_count"] = np.log1p(df_indiv_long["sum_pred_count"])
    df_indiv_long["log_sum_obs_count"] = np.log1p(df_indiv_long["sum_obs_count"])

    # Drop rows where correlation couldnt be computed (failed min_valid mask)
    initial_rows = len(df_indiv_long)
    df_indiv_long.dropna(subset=["pearson_r"], inplace=True) #inplace to save RAM
    final_rows = len(df_indiv_long)
    print(f"Dropped {initial_rows - final_rows} NaN intervals in tracks. Final long-format rows: {final_rows}")
    return df_indiv_long
