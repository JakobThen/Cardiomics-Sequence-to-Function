# import os
# import sys
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import h5py

# parent_dir = os.path.abspath("/home/then")
# if parent_dir not in sys.path:
#     sys.path.append(parent_dir)

# # Import our new streaming modules
# from utils.eval.accumulators import (
#     StreamingPearsonAccumulator, 
#     StreamingGeneAccumulator, 
#     BestIntervalTracker, 
#     ReservoirSampler
# )
# from utils.io.batch_bw_loader import BatchBigWigLoader

# # Assume your custom plotting functions are still available in your path
# from utils.eval.track_correlation import (
#     quantile_normalize_RNA_counts,
#     density_scatter,
#     plot_correlation_beeswarm,
#     compare_track_coverage,
#     plot_strand_expression_heatmaps
# )
# from utils.eval.track_prediction import get_gtf

# """
#     Pipeline to run comprehensive model evaluation on test intervals from h5 saved mdoel predictions or streaming from GPU inference batches during test inference.
#     Model tracks need to be matched to individual .bw files.
#     Assumes Intervals do not overlap and are the same size.
#     Computes:
#     -   Pearson r across all concatenated intervals per tracks.
#     -   Gene centric metrics for RNA tracks:
#         -   Pearson r across all genes, raw summed log1p exon counts
#         -   Pearson r across all genes, quantile norm raw across genes, gene mean subtrackted
#         -   Pearson r across all tracks, quantile norm raw across genes, gene mean subtrackted
#     - Basic QC metrics:
#         -   distribution of counts across intervals per track
#         -   percentage of non NaN bins per interval and track (NaN bins have no coverage in .bw at any position in the bin).
        
#     Returns:
#     -   csv file of pearson metrics and coverage per track.
#     -   summary txt file
#     -   figures of the metrics above stored in ./fig
#     Code by Jakob Then
# """

# # ---------------------------------------------------------------------------
# # CORE LOOP (The Engine)
# # ---------------------------------------------------------------------------
# def _run_analysis_loop(
#     batch_iterator, 
#     num_intervals,
#     intervals_df, 
#     tracks_df, 
#     bw_paths_df,
#     bin_size, 
#     out_dir, 
#     model_name, 
#     analysis_name,
#     coverage_cutoff=0.1,
#     gtf_file=None,
#     seq_len=None,    
#     label_len=None,
#     is_squashed_scale = True
# ):
#     """
#     Consumes batches of predictions, loads BigWigs on the fly, 
#     and updates streaming accumulators.
#     """
#     num_tracks = len(tracks_df)
#     track_names = tracks_df.index.tolist()
    
#     # Determine default lengths if not explicitly passed
#     if seq_len is None:
#         seq_len = intervals_df['end'].iloc[0] - intervals_df['start'].iloc[0]
#     if label_len is None:
#         label_len = seq_len
    
#     # Check modalities
#     tracks_df['Assay_type'] = tracks_df['Assay_type'].replace(
#         to_replace=r'(?i)^(cnt|c&t|cut&tag|cutntag)$', 
#         value='CnT', 
#         regex=True
#     ) #cas isnensitve cnt standardization
#     rna_mask = tracks_df['Assay_type'] == 'RNA'
#     atac_mask = tracks_df['Assay_type'] == 'ATAC'
#     cnt_mask = tracks_df['Assay_type'] == 'CnT'
    
#     # 1. Initialize Accumulators
#     print("Initializing Streaming Accumulators...", flush = True)
#     num_bins = (intervals_df['end'].iloc[0] - intervals_df['start'].iloc[0]) // bin_size
    
#     pearson_acc = StreamingPearsonAccumulator(num_tracks, track_names)
#     best_tracker = BestIntervalTracker(tracks_df)
#     reservoir = ReservoirSampler(num_tracks, k=10000)
    
#     # Initialize Gene Accumulator ONLY if we have RNA tracks and a GTF
#     gene_acc = None
#     if rna_mask.any() and gtf_file is not None:
#         gtf = get_gtf(gtf_file, chrom_filter="autosomesXY",
#                       gene_type_filter=["protein_coding", "processed_transcript"],
#                       feature=None)
#         exons_df = gtf[gtf["feature"] == "exon"].copy()
#         gtf_df   = gtf[gtf["feature"] == "gene"].copy()
              
#         exons_df["interval"] = list(zip(exons_df["start"], exons_df["end"]))
#         exon_groups = exons_df.groupby("gene_id")["interval"].apply(list)
#         gtf_df["exon_intervals"] = gtf_df["gene_id"].map(exon_groups).apply(lambda d: d if isinstance(d, list) else [])
#         gtf_df = gtf_df.set_index("gene_name")
        
#         # Determine strands for RNA tracks
#         track_strands = [1 if "plus" in str(t).lower() else -1 if "minus" in str(t).lower() else 0 for t in track_names]
        
#         gene_acc = StreamingGeneAccumulator(intervals_df, gtf_df, bin_size, track_names, track_strands)

#     # 2. Main Processing Loop
#     print(f"Starting Streaming Evaluation over {num_intervals} intervals...", flush = True)
#     with BatchBigWigLoader(bw_paths_df, num_bins, bin_size) as bw_loader:
        
#         # batch_iterator must yield: (start_idx, P_batch, coords_batch)
#         for start_idx, P_batch, coords_batch in batch_iterator:
            
#             # Load BigWig data concurrently for this batch
#             T_batch = bw_loader.load_batch(coords_batch)
            
#             # Feed accumulators
#             pearson_acc.update(P_batch, T_batch)
#             best_tracker.update(P_batch, T_batch, coords_batch)
#             reservoir.update(P_batch, T_batch)
            
#             if gene_acc is not None:
#                 # We only need to compute gene sums for RNA tracks, but the accumulator 
#                 # handles masking internally if set up properly. 
#                 gene_acc.update(start_idx, P_batch, T_batch)
                
#             # if (start_idx + len(P_batch)) % 1000 == 0 or (start_idx + len(P_batch)) == num_intervals:
#             #     print(f"Processed {start_idx + len(P_batch)} / {num_intervals} intervals", flush = True)
#             current_processed = start_idx + len(P_batch)
#             if current_processed // 1000 > start_idx // 1000 or current_processed == num_intervals:
#                 print(f"Processed {current_processed} / {num_intervals} intervals", flush = True)

#     # 3. Finalize and Generate Outputs
#     print("Evaluation Complete. Generating outputs...", flush = True)
#     _generate_outputs(
#         out_dir=out_dir, model_name=model_name, analysis_name=analysis_name,
#         tracks_df=tracks_df, pearson_acc=pearson_acc, best_tracker=best_tracker, 
#         reservoir=reservoir, gene_acc=gene_acc, coverage_cutoff=coverage_cutoff,
#         num_intervals=num_intervals, seq_len=seq_len, label_len=label_len, bin_size=bin_size
#     )

# # ---------------------------------------------------------------------------
# # OUTPUT GENERATION (Plots & Summaries)
# # ---------------------------------------------------------------------------
# def _generate_outputs(
#     out_dir, model_name, analysis_name, tracks_df, 
#     pearson_acc, best_tracker, reservoir, gene_acc, coverage_cutoff,
#     num_intervals, seq_len, label_len, bin_size
# ):
#     fig_dir = Path(out_dir) / model_name / analysis_name / "fig"
#     cor_dir = Path(out_dir) / model_name / analysis_name
#     fig_dir.mkdir(parents=True, exist_ok=True)
#     cor_dir.mkdir(parents=True, exist_ok=True)

#     # Base Metrics
#     concat_r, coverage = pearson_acc.compute(coverage_cutoff)
#     tracks_df['concat_cor'] = concat_r
#     tracks_df['coverage'] = coverage
    
#     # Gene Metrics
#     raw_gene_cor, norm_gene_cor, norm_track_cor = None, None, None
#     if gene_acc is not None:
#         gene_P, gene_T = gene_acc.compute(return_log1p=True)
        
#         dup_mask = ~gene_P.index.duplicated(keep='first') # Drop duplicate gene names
#         gene_P, gene_T = gene_P[dup_mask], gene_T[dup_mask]
#         valid_mask = gene_P.notna().any(axis=1) & gene_T.notna().any(axis=1) #Drop genes that are completely NaN across all tracks
#         gene_P, gene_T = gene_P[valid_mask], gene_T[valid_mask]
        
#         # Only compute correlation for actual RNA tracks to avoid NaN pollution
#         rna_cols = tracks_df[tracks_df['Assay_type'] == 'RNA'].index
#         gP_rna, gT_rna = gene_P[rna_cols], gene_T[rna_cols]
        
#         raw_gene_cor = gT_rna.corrwith(gP_rna, method='pearson')
 
#         gene_norm_P = quantile_normalize_RNA_counts(gP_rna)
#         gene_norm_T = quantile_normalize_RNA_counts(gT_rna)
#         norm_gene_cor = gene_norm_T.corrwith(gene_norm_P, method='pearson')
#         norm_track_cor = gene_norm_T.T.corrwith(gene_norm_P.T, method='pearson')
        
#         # Save to tracks metadata
#         tracks_df.loc[rna_cols, 'raw_gene_cor'] = raw_gene_cor
#         tracks_df.loc[rna_cols, 'norm_gene_cor'] = norm_gene_cor
    
#         # GENE PLOTS
#         print("Generating Gene-level plots...", flush = True)
#         # Gene Expression Heatmap       
#         fig = plot_strand_expression_heatmaps(
#             gene_P=gP_rna,
#             gene_T=gT_rna,
#             gene_cors={'raw gene r': raw_gene_cor, 'norm gene r': norm_gene_cor},
#             gene_cors_by_gene={'norm track r': norm_track_cor},   # gene-indexed series
#             figsize=(26, 15)
#         )
#         fig.savefig(fig_dir / "gene_expression_heatmap.pdf", bbox_inches='tight', dpi=300) 
#         plt.close(fig)

#         # Gene Counts Scatter
#         idx_max = {'raw': raw_gene_cor.idxmax(),'gene': norm_gene_cor.idxmax()}
#         idx_min = {'raw': raw_gene_cor.idxmin(),'gene': norm_gene_cor.idxmin()}
#         fig, axes = plt.subplots(2, 2, figsize=(8, 8))
#         fig.suptitle(f"Gene Counts Scatter (Max & Min {model_name} Tracks)", fontsize=14, fontweight='bold')
#         density_scatter(gP_rna[idx_max['raw']], gT_rna[idx_max['raw']], 
#                         ax=axes[0,0], title=f"Raw Max\n{idx_max['raw']}\n(r={raw_gene_cor[idx_max['raw']]:.3f})")
#         density_scatter(gene_norm_P[idx_max['gene']], gene_norm_T[idx_max['gene']], 
#                         ax=axes[0,1], title=f"Norm Gene Max\n{idx_max['gene']}\n(r={norm_gene_cor[idx_max['gene']]:.3f})")
#         density_scatter(gP_rna[idx_min['raw']], gT_rna[idx_min['raw']], 
#                         ax=axes[1,0], title=f"Raw Min\n{idx_min['raw']}\n(r={raw_gene_cor[idx_min['raw']]:.3f})")
#         density_scatter(gene_norm_P[idx_min['gene']], gene_norm_T[idx_min['gene']], 
#                         ax=axes[1,1], title=f"Norm Gene Min\n{idx_min['gene']}\n(r={norm_gene_cor[idx_min['gene']]:.3f})")
#         plt.tight_layout()
#         plt.savefig(fig_dir / "gene_counts_scatter.pdf", bbox_inches='tight', dpi=300) 
#         plt.close(fig)
                
#         #Gene correlation figure
#         rna_track_names = raw_gene_cor.index.tolist()
#         fig, axes = plt.subplots(1, 2, figsize=(10, 3+(len(raw_gene_cor)/30)*5), sharey=True)
#         fig.suptitle(f'{model_name} Gene Correlations Across Normalization Methods', fontsize=17, fontweight='bold', y=0.98)
#         # Raw Gene Correlations
#         axes[0].barh(rna_track_names, raw_gene_cor.values, color='skyblue', edgecolor='black')
#         axes[0].set_title('Raw Gene Correlations', fontsize=14)
#         axes[0].set_xlabel('Pearson R')
#         # Normalized Across Genes
#         axes[1].barh(rna_track_names, norm_gene_cor.values, color='lightgreen', edgecolor='black')
#         axes[1].set_title('Normalized Across Genes', fontsize=14)
#         axes[1].set_xlabel('Pearson R')
#         plt.tight_layout()
#         plt.savefig(fig_dir / "gene_corr.pdf", bbox_inches='tight', dpi=300) 
#         plt.close(fig)

#     # Save metrics CSV
#     tracks_df.to_csv(cor_dir / "track_correlations.csv")

#     # Text Summary
#     _write_summary(cor_dir,
#                    model_name, analysis_name, tracks_df, coverage,
#                    concat_r, raw_gene_cor, norm_gene_cor, norm_track_cor,
#                   num_intervals=num_intervals, num_tracks=len(tracks_df), seq_len=seq_len, label_len=label_len,  bin_size=bin_size)      

#     # Plots 
#     # Concatenated Scatter (using reservoir sample)
#     fig, axes = plt.subplots(1, 3, figsize=(15, 4))
#     for ax_idx, mod in enumerate(['ATAC', 'RNA', 'CnT']):
#         mask = (tracks_df['Assay_type'] == mod).values
#         if not mask.any(): continue
        
#         mod_r = concat_r[mask]
#         if mod_r.isna().all(): continue
            
#         best_track_name = mod_r.idxmax()
#         t_idx = tracks_df.index.get_loc(best_track_name)
        
#         p_samp = np.log1p(reservoir.reservoir_P[t_idx])
#         t_samp = np.log1p(reservoir.reservoir_T[t_idx])
        
#         valid = ~np.isnan(p_samp) & ~np.isnan(t_samp)
#         if valid.sum() > 0:
#             density_scatter(p_samp[valid], t_samp[valid], ax=axes[ax_idx], 
#                             title=f"Best {mod}: {best_track_name}\n(r={mod_r.max():.3f})")
#     plt.tight_layout()
#     plt.savefig(fig_dir / "concatenated_counts_scatter.pdf", dpi=300)
#     plt.close()

#     # Best Interval Coverage Tracks
#     fig, axes = plt.subplots(3, 1, figsize=(10, 8))
#     for ax_idx, mod in enumerate(['ATAC', 'RNA', 'CnT']):
#         data = best_tracker.best_data[mod]
#         if data is None: continue
            
#         # We plot the first track of that modality for the best interval
#         compare_track_coverage(
#             data["P"][0].flatten(), 
#             data["T"][0].flatten(),
#             pos=data["coord"],
#             ax=axes[ax_idx],
#             title=f"Best {mod} Interval: {data['coord']}"
#         )
#     plt.tight_layout()
#     plt.savefig(fig_dir / "individual_track_coverage.pdf", dpi=300)
#     plt.close()

#     # Summary Beeswarm
#     # (Adapted to use just the concat correlation since we dropped the heavy individual distribution)
#     fig, ax = plt.subplots(figsize=(6, 4))
#     plot_data = [concat_r[tracks_df['Assay_type'] == mod].dropna() for mod in ['ATAC', 'RNA', 'CnT']]
#     plot_correlation_beeswarm(plot_data, labels=['ATAC', 'RNA', 'CnT'], ax=ax, title=f"{model_name} Concat Correlations")
#     plt.tight_layout()
#     plt.savefig(fig_dir / "summary_beeswarm.pdf", dpi=300)
#     plt.close()

# def _write_summary(cor_dir, model_name, analysis_name, tracks_df, coverage, concat_r, 
#                    raw_gene_cor, norm_gene_cor, norm_track_cor, 
#                    num_intervals, num_tracks, seq_len, label_len, bin_size):
#     summary_path = cor_dir / f"{analysis_name}_summary_metrics.txt"
#     with open(summary_path, "w") as f:
#         f.write("====================================================\n")
#         f.write(f"Seq2Fun Streaming Evaluation Summary - {model_name}\n")
#         f.write(f"Analysis Name: {analysis_name}\n")
#         f.write("====================================================\n\n")
        
#         f.write("GLOBAL SETUP & QC\n")
#         f.write(f"Total Intervals:      {num_intervals}\n")
#         f.write(f"Total Tracks:         {num_tracks}\n")
#         f.write(f"Input Seq Length:     {seq_len}\n")
#         f.write(f"Label Seq  Length:    {label_len}\n")
#         f.write(f"Bin Size:             {bin_size}\n")      
#         f.write(f"Average Valid Bins (non-NaN): {coverage.mean() * 100:.2f}%\n")
        
#         f.write("BIN-RESOLUTION (BY MODALITY)\n")
#         for mod in ["ATAC", "RNA", "CnT"]:
#             mask = tracks_df['Assay_type'] == mod
#             if mask.sum() > 0:
#                 f.write(f"[{mod}] Concatenated Pearson r | Median: {concat_r[mask].median():.4f} | Max: {concat_r[mask].max():.4f} | Min: {concat_r[mask].min():.4f}\n")

#         if raw_gene_cor is not None:
#             f.write("\nGENE-RESOLUTION (RNA ONLY)\n")
#             f.write(f"Raw Gene Median r:        {raw_gene_cor.median():.4f}\n")
#             f.write(f"Norm-Gene Median r:       {norm_gene_cor.median():.4f}\n")
#             f.write(f"Norm-Track Median r:      {norm_track_cor.median():.4f}\n")
#         else:
#             f.write("GENE-RESOLUTION\n")
#             f.write("No RNA tracks detected. Gene metrics skipped.\n")
#     print(f"Summary metrics written to: {summary_path}")
            

# # ---------------------------------------------------------------------------
# # ENTRY POINTS
# # ---------------------------------------------------------------------------
# def evaluate_from_inference(
#     data_module, model_forward_fn, intervals_df, tracks_df, bw_paths_df, 
#     bin_size, out_dir, model_name, analysis_name, gtf_file=None, batch_size=16
# ):
#     """Entry point to run evaluation live during model inference."""
    
#     # Create a generator that yields batches
#     def batch_generator():
#         for batch_idx, batch in enumerate(data_module.iter_batches(split="test")):
#             start_idx = batch_idx * batch_size

#             # Pass to your forward function
#             P_batch = model_forward_fn(batch, start_idx)
            
#             # Extract coordinates for this batch
#             # Assuming batch has a way to get the interval coordinates
#             coords_batch = intervals_df.iloc[start_idx : start_idx + len(P_batch)][['chrom', 'start', 'end']].apply(tuple, axis=1).tolist()
            
#             yield start_idx, P_batch, coords_batch

#     _run_analysis_loop(
#         batch_iterator=batch_generator(),
#         num_intervals=len(intervals_df),
#         intervals_df=intervals_df,
#         tracks_df=tracks_df,
#         bw_paths_df=bw_paths_df,
#         bin_size=bin_size,
#         out_dir=out_dir,
#         model_name=model_name,
#         analysis_name=analysis_name,
#         gtf_file=gtf_file
#     )

# # ---------------------------------------------------------------------------
# # H5 STREAMING HELPERS
# # ---------------------------------------------------------------------------
# def _extract_h5_group_to_df(group):
#     """
#     Helper function to convert an HDF5 group of 1D datasets into a pandas DataFrame.
#     """
#     if group is None:
#         return None
        
#     data_dict = {}
#     for key in group.keys():
#         arr = group[key][()]
#         if len(arr) > 0 and isinstance(arr[0], bytes):
#             arr = [val.decode('utf-8') for val in arr]
#         data_dict[key] = arr
        
#     if not data_dict:
#         return None
        
#     return pd.DataFrame(data_dict)

# def _get_alignment_axes(shape, num_intervals=None, num_tracks=None, head_name="Single Head"):
#     """
#     Infers the axes mapping from the dataset shape on disk without loading it into RAM.
#     Returns the tuple (ax_intervals, ax_tracks, ax_bins) so slices can be transposed properly.
#     """
#     if len(shape) != 3:
#         raise ValueError(f"[{head_name}] Expected 3D prediction array, got {len(shape)}D.")

#     ax_intervals, ax_tracks, ax_bins = 0, 1, 2
    
#     # find the intervals Axis
#     if num_intervals is not None:
#         if shape[0] == num_intervals:
#             ax_intervals = 0
#         elif num_intervals in shape:
#             ax_intervals = shape.index(num_intervals)
#         else:
#             raise ValueError(f"[{head_name}] Shape {shape} does not contain an axis matching {num_intervals} intervals.")
            
#     # find the Tracks Axis
#     if num_tracks is not None:
#         remaining_axes = [i for i in range(3) if i != ax_intervals]
        
#         if shape[remaining_axes[0]] == num_tracks and shape[remaining_axes[1]] != num_tracks:
#             ax_tracks, ax_bins = remaining_axes[0], remaining_axes[1]
#         elif shape[remaining_axes[1]] == num_tracks and shape[remaining_axes[0]] != num_tracks:
#             ax_tracks, ax_bins = remaining_axes[1], remaining_axes[0]
#         elif shape[remaining_axes[0]] == num_tracks and shape[remaining_axes[1]] == num_tracks:
#             ax_tracks, ax_bins = remaining_axes[0], remaining_axes[1]
#         else:
#             raise ValueError(f"[{head_name}] Shape {shape} does not contain an axis matching {num_tracks} tracks.")

#     elif num_intervals is not None:
#         remaining_axes = [i for i in range(3) if i != ax_intervals]
#         ax_tracks, ax_bins = remaining_axes[0], remaining_axes[1]

#     if (ax_intervals, ax_tracks, ax_bins) != (0, 1, 2):
#         print(f"[{head_name}] Auto-aligning axes: Mapping intervals->axis{ax_intervals}, "
#               f"tracks->axis{ax_tracks}, bins->axis{ax_bins}.", flush = True)

#     return ax_intervals, ax_tracks, ax_bins

# # ---------------------------------------------------------------------------
# # HDF5 ENTRY POINT
# # ---------------------------------------------------------------------------
# def evaluate_from_h5(
#     h5_path, bw_paths_df, bin_size, out_dir, model_name, analysis_name, 
#     gtf_file=None, batch_size=16
# ):
#     """Entry point to run streaming evaluation on an already-saved HDF5 file."""
    
#     h5_path = Path(h5_path)
#     if not h5_path.exists():
#         raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

#     # First, let's open the file just to extract metadata
#     with h5py.File(h5_path, 'r') as f:
#         intervals_df = None
#         num_intervals = None
#         if "intervals" in f:
#             intervals_df = _extract_h5_group_to_df(f["intervals"])
#             # Format columns to standard names just in case
#             intervals_df = intervals_df.rename(columns={
#                 "CHROM": "chrom", "#CHROM": "chrom", "Chrom": "chrom", "Chromosome": "chrom", "chromosome": "chrom",
#                 "Start": "start", "START": "start", "End": "end", "END": "end"
#             })
#             # Ensure coordinates are integers
#             intervals_df['start'] = intervals_df['start'].astype(int)
#             intervals_df['end'] = intervals_df['end'].astype(int)
#             num_intervals = len(intervals_df)
#         else:
#             raise ValueError(f"'intervals' group missing in {h5_path.name}")

#         # Map datasets and figure out alignments
#         datasets_info = []
#         tracks_list = []
        
#         if "predictions" in f:
#             # Single-Head (Borzoi)
#             ds = f["predictions"]
#             t_df = _extract_h5_group_to_df(f.get("tracks"))
#             num_tracks = len(t_df) if t_df is not None else None
#             ax_map = _get_alignment_axes(ds.shape, num_intervals, num_tracks, "SingleHead")
            
#             datasets_info.append({"dataset": ds.name, "shape": ds.shape, "ax_map": ax_map})
#             if t_df is not None: tracks_list.append(t_df)
#         else:
#             # Multi-Head (AG)
#             for head_name in f.keys():
#                 if head_name == "intervals": continue 
#                 head_grp = f[head_name]
#                 if "predictions" not in head_grp: continue 
                
#                 ds = head_grp["predictions"]
#                 t_df = _extract_h5_group_to_df(head_grp.get("tracks"))
#                 num_tracks = len(t_df) if t_df is not None else None
#                 ax_map = _get_alignment_axes(ds.shape, num_intervals, num_tracks, head_name)
                
#                 datasets_info.append({"dataset": ds.name, "shape": ds.shape, "ax_map": ax_map})
#                 if t_df is not None: tracks_list.append(t_df)
                
#         if not datasets_info:
#             raise ValueError(f"Could not find any 'predictions' datasets in {h5_path.name}")

#         # Concatenate tracks DataFrames row-wise
#         if tracks_list and len(tracks_list) == len(datasets_info):
#             tracks_df = pd.concat(tracks_list, ignore_index=True)
#             from utils.eval.track_prediction import process_track_metadata
#             tracks_df = process_track_metadata(tracks_df)
#             tracks_df = tracks_df.set_index("id")
#         else:
#             raise ValueError(f"Missing track metadata in {h5_path.name}")
            
#     # HANDLE BORZOI INTERVAL CROPPING
#     # Get the number of bins directly from the model predictions on disk
#     ax_intervals, ax_tracks, ax_bins = datasets_info[0]["ax_map"]
#     num_bins_p = datasets_info[0]["shape"][ax_bins]
#     SEQ_LEN = intervals_df['end'].iloc[0] - intervals_df['start'].iloc[0]
#     LABEL_LEN = num_bins_p * bin_size
#     if SEQ_LEN != LABEL_LEN:
#         print(f"WARNING: Resizing intervals from {SEQ_LEN} to {LABEL_LEN} to match prediction bins.", flush = True)
#         CROP = (SEQ_LEN - LABEL_LEN) // 2
#         intervals_df['start'] = intervals_df['start'] + CROP
#         intervals_df['end'] = intervals_df['end'] - CROP
#     print(f"Using SEQ_LEN: {SEQ_LEN}, LABEL_LEN: {LABEL_LEN}, BIN_SIZE: {bin_size}, N_BINS: {num_bins_p} as inferred from h5 file.",
#           flush = True) 

#     # Create the generator that slices lazily
#     def h5_generator():
#         # Keep file open during streaming
#         with h5py.File(h5_path, 'r') as f:
#             # Re-fetch dataset pointers for the open file session
#             active_datasets = []
#             for info in datasets_info:
#                 ds_path = info["dataset"]
#                 active_datasets.append({
#                     "dataset": f[ds_path],
#                     "ax_map": info["ax_map"]
#                 })

#             for start_idx in range(0, num_intervals, batch_size):
#                 end_idx = min(start_idx + batch_size, num_intervals)
                
#                 batch_slices = []
#                 for info in active_datasets:
#                     ds = info["dataset"]
#                     ax_intervals, ax_tracks, ax_bins = info["ax_map"]
                    
#                     # Create a slicing tuple: slice(None) means ':'
#                     slices = [slice(None)] * 3
#                     slices[ax_intervals] = slice(start_idx, end_idx)
                    
#                     # Read only the batch from disk!
#                     p_chunk = ds[tuple(slices)]
                    
#                     # Transpose to (Intervals, Tracks, Bins) if needed
#                     if (ax_intervals, ax_tracks, ax_bins) != (0, 1, 2):
#                         p_chunk = np.transpose(p_chunk, (ax_intervals, ax_tracks, ax_bins))
                        
#                     batch_slices.append(p_chunk)
                
#                 # Concatenate along the tracks axis
#                 P_batch = np.concatenate(batch_slices, axis=1)
                
#                 # Get coordinates
#                 coords_batch = intervals_df.iloc[start_idx:end_idx][['chrom', 'start', 'end']].apply(tuple, axis=1).tolist()
                
#                 yield start_idx, P_batch, coords_batch

#     # Hand off to the core loop
#     _run_analysis_loop(
#         batch_iterator=h5_generator(),
#         num_intervals=num_intervals,
#         intervals_df=intervals_df,
#         tracks_df=tracks_df,
#         bw_paths_df=bw_paths_df,
#         bin_size=bin_size,
#         out_dir=out_dir,
#         model_name=model_name,
#         analysis_name=analysis_name,
#         gtf_file=gtf_file,
#         seq_len=SEQ_LEN,
#         label_len=LABEL_LEN
#     )

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import seaborn as sns
import h5py

parent_dir = os.path.abspath("/home/then")
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils.eval.accumulators import (
    StreamingPearsonAccumulator, 
    StreamingGeneAccumulator, 
    BestIntervalTracker, 
    StreamingCoverageAccumulator
   # ReservoirSampler
)
from utils.io.batch_bw_loader import BatchBigWigLoader
from utils.io.data_input import SelectiveSquashTransform, get_streaming_non_zero_track_means
from utils.eval.track_correlation import (
    quantile_normalize_RNA_counts,
    density_scatter,
    plot_correlation_beeswarm,
    compare_track_coverage,
    plot_strand_expression_heatmaps
)
from utils.eval.track_prediction import get_gtf

"""
    Pipeline to run comprehensive model evaluation on test intervals from h5 saved mdoel predictions or streaming from GPU inference batches during test inference.
    Model tracks need to be matched to individual .bw files.
    Assumes Intervals do not overlap and are the same size.
    Computes:
    -   Pearson r across all concatenated intervals per tracks.
    -   Gene centric metrics for RNA tracks:
        -   Pearson r across all genes, raw summed log1p exon counts
        -   Pearson r across all genes, quantile norm raw across genes, gene mean subtrackted
        -   Pearson r across all tracks, quantile norm raw across genes, gene mean subtrackted
    - Basic QC metrics:
        -   distribution of counts across intervals per track
        -   percentage of non NaN bins per interval and track (NaN bins have no coverage in .bw at any position in the bin).
        
    Returns:
    -   csv file of pearson metrics and coverage per track.
    -   summary txt file
    -   figures of the metrics above stored in ./fig
    Code by Jakob Then
"""

# ---------------------------------------------------------------------------
# CORE LOOP (The Engine)
# ---------------------------------------------------------------------------
def _run_analysis_loop(
    batch_iterator, 
    num_intervals,
    intervals_df, 
    tracks_df, 
    bw_paths_df,
    bin_size, 
    out_dir, 
    model_name, 
    analysis_name,
    gtf_file=None,
    seq_len=None,    
    label_len=None,
    is_squashed_scale = True
):
    """
    Consumes batches of predictions, loads BigWigs on the fly, 
    and updates streaming accumulators. Defaults to squahsed scale predcition otherwise transforms accordingly.
    """
    num_tracks = len(tracks_df)
    track_names = tracks_df.index.tolist()
    
    # Determine default lengths if not explicitly passed
    if seq_len is None:
        seq_len = intervals_df['end'].iloc[0] - intervals_df['start'].iloc[0]
    if label_len is None:
        label_len = seq_len
    
    # Check modalities
    tracks_df['Assay_type'] = tracks_df['Assay_type'].replace(
        to_replace=r'(?i)^(cnt|c&t|cut&tag|cutntag)$', 
        value='CnT', 
        regex=True
    ) #cas isnensitve cnt standardization
    rna_mask = tracks_df['Assay_type'] == 'RNA'
    atac_mask = tracks_df['Assay_type'] == 'ATAC'
    cnt_mask = tracks_df['Assay_type'] == 'CnT'
    
    # 1. Initialize Accumulators
    print("Initializing Streaming Accumulators...", flush = True)
    num_bins = (intervals_df['end'].iloc[0] - intervals_df['start'].iloc[0]) // bin_size
    
    pearson_acc = StreamingPearsonAccumulator(num_tracks, track_names)
    best_tracker = BestIntervalTracker(tracks_df)
    coverage_acc = StreamingCoverageAccumulator()
   # reservoir = ReservoirSampler(num_tracks, k=10000)
    
    # Initialize Gene Accumulator ONLY if we have RNA tracks and a GTF
    gene_acc = None
    if rna_mask.any() and gtf_file is not None:
        gtf = get_gtf(gtf_file, chrom_filter="autosomesXY",
                      gene_type_filter=["protein_coding", "processed_transcript"],
                      feature=None)
        exons_df = gtf[gtf["feature"] == "exon"].copy()
        gtf_df   = gtf[gtf["feature"] == "gene"].copy()
              
        exons_df["interval"] = list(zip(exons_df["start"], exons_df["end"]))
        exon_groups = exons_df.groupby("gene_id")["interval"].apply(list)
        gtf_df["exon_intervals"] = gtf_df["gene_id"].map(exon_groups).apply(lambda d: d if isinstance(d, list) else [])
        gtf_df = gtf_df.set_index("gene_name")
        
        # Determine strands for RNA tracks
        track_strands = [1 if "plus" in str(t).lower() else -1 if "minus" in str(t).lower() else 0 for t in track_names]
        
        gene_acc = StreamingGeneAccumulator(intervals_df, gtf_df, bin_size, track_names, track_strands)
        
    #Initilaize correct sqush transforms for current model. 
    rna_indices = np.where(rna_mask)[0].tolist()
    if "BORZOI" in model_name.upper():
        squash = SelectiveSquashTransform(rna_task_indices=rna_indices, task_axis=1, transform="Borzoi", inverse=False)  
        unsquash = SelectiveSquashTransform(rna_task_indices=rna_indices, task_axis=1, transform="Borzoi", inverse=True)  
    elif ("ALPHAGENOME" in model_name.upper()) or ("AG" in model_name.upper()):
        #track_means = get_streaming_non_zero_track_means(bw_paths_df, intervals_df, bin_size) #make a quick pass over all bw files to compute the track mean
        if 'nonzero_means' not in bw_paths_df.columns:
            raise ValueError("AlphaGenome requires 'nonzero_mean' in bw_paths_df, but it was not found.")
        track_means = bw_paths_df['nonzero_means'].values
        squash = SelectiveSquashTransform(rna_task_indices=rna_indices, task_axis=1,transform="AlphaGenome",track_means=track_means,inverse=False)  
        unsquash = SelectiveSquashTransform(rna_task_indices=rna_indices, task_axis=1,transform="AlphaGenome",track_means=track_means,inverse=True)  
    else:
        print("WARNING: model_name was not recognized as Borzoi or AlphaGenome, defaulting to no squash transformations.", flush = True)
        def squash(X):   return X
        def unsquash(X): return X

    # 2. Main Processing Loop
    print(f"Starting Streaming Evaluation over {num_intervals} intervals...", flush = True)
    with BatchBigWigLoader(bw_paths_df, num_bins, bin_size) as bw_loader:
        
        # batch_iterator must yield: (start_idx, P_batch, coords_batch)
        for start_idx, P_batch, coords_batch in batch_iterator:
            
            # Load BigWig data concurrently for this batch
            T_batch_unsq = bw_loader.load_batch(coords_batch)
            T_batch = squash(T_batch_unsq)
            
            #apply squash scale transforms
            if is_squashed_scale:
                P_batch_unsq = unsquash(P_batch)
            else:
                P_batch_unsq = P_batch
                P_batch = squash(P_batch)
                           
            # Feed accumulators
            pearson_acc.update(P_batch, T_batch) #cumpute pearson in squahed scale
            best_tracker.update(P_batch, T_batch, coords_batch) #compute pearson for best interval also in squashed scale
            coverage_acc.update(T_batch_unsq) #get the nan coverage percentage for the 
         #   reservoir.update(P_batch_unsq, T_batch_unsq) #keep unsquahed points in reservoir to later plot in log scale instead of squashed
            
            if gene_acc is not None:
                # We only need to compute gene sums for RNA tracks, but the accumulator 
                # handles masking internally if set up properly. 
                gene_acc.update(start_idx, P_batch_unsq, T_batch_unsq) #pass unsquahed to gene accumulator to get log1p coutns for genes instead of squashed
                
            current_processed = start_idx + len(P_batch)
            if current_processed // 1000 > start_idx // 1000 or current_processed == num_intervals:
                print(f"Processed {current_processed} / {num_intervals} intervals", flush = True)

    # 3. Finalize and Generate Outputs
    print("Evaluation Complete. Generating outputs...", flush = True)
    
    #define nan coverage cutoffs for bw fiels based on binsize (the larger the bin the less sparse bins we get thus increase cutoff
    #use pporbabalisitc saturation model
    # baseline_nan_cutoff=0.01
    # adj_nan_cutoff = 1 - (1 - baseline_nan_cutoff) ** (bin_size ** 0.678) #0.678 adjusted so that this 0.01 at binszie 1 and 0.1 at binsize 32 
    # print(f"Dynamic NaN BigWig Coverage Cutoff for Bin Size {bin_size}: {adj_nan_cutoff*100:.2f}%", flush=True)
    
    adj_nan_cutoff = 0 #USE NO CUTOFF FOR NOW
    
    interval_coverage_matrix = coverage_acc.compute()
    
    _generate_outputs(
        out_dir=out_dir, model_name=model_name, analysis_name=analysis_name,
        tracks_df=tracks_df, pearson_acc=pearson_acc, best_tracker=best_tracker, 
      #  reservoir=reservoir,
        gene_acc=gene_acc, coverage_cutoff=adj_nan_cutoff, interval_coverage_matrix=interval_coverage_matrix, 
        num_intervals=num_intervals, seq_len=seq_len, label_len=label_len, bin_size=bin_size
    )

# ---------------------------------------------------------------------------
# OUTPUT GENERATION (Plots & Summaries)
# ---------------------------------------------------------------------------
def _generate_outputs(
    out_dir, model_name, analysis_name, tracks_df, 
    pearson_acc, best_tracker, #reservoir,
    gene_acc, coverage_cutoff, interval_coverage_matrix,
    num_intervals, seq_len, label_len, bin_size
):
    fig_dir = Path(out_dir) / model_name / analysis_name / "fig"
    cor_dir = Path(out_dir) / model_name / analysis_name
    fig_dir.mkdir(parents=True, exist_ok=True)
    cor_dir.mkdir(parents=True, exist_ok=True)

    # Base Metrics
    concat_r, coverage = pearson_acc.compute(coverage_cutoff)
    tracks_df['concat_cor'] = concat_r
    tracks_df['coverage'] = coverage
    
    # Gene Metrics
    raw_gene_cor, norm_gene_cor, norm_track_cor = None, None, None
    if gene_acc is not None:
        gene_P, gene_T = gene_acc.compute(return_log1p=True)
        
        dup_mask = ~gene_P.index.duplicated(keep='first') # Drop duplicate gene names
        gene_P, gene_T = gene_P[dup_mask], gene_T[dup_mask]
        valid_mask = gene_P.notna().any(axis=1) & gene_T.notna().any(axis=1) #Drop genes that are completely NaN across all tracks
        gene_P, gene_T = gene_P[valid_mask], gene_T[valid_mask]
        
        # Only compute correlation for actual RNA tracks to avoid NaN pollution
        rna_cols = tracks_df[tracks_df['Assay_type'] == 'RNA'].index
        gP_rna, gT_rna = gene_P[rna_cols], gene_T[rna_cols]
        
        raw_gene_cor = gT_rna.corrwith(gP_rna, method='pearson')
 
        gene_norm_P = quantile_normalize_RNA_counts(gP_rna)
        gene_norm_T = quantile_normalize_RNA_counts(gT_rna)
        norm_gene_cor = gene_norm_T.corrwith(gene_norm_P, method='pearson')
        norm_track_cor = gene_norm_T.T.corrwith(gene_norm_P.T, method='pearson')
        
        # Save to tracks metadata
        tracks_df.loc[rna_cols, 'raw_gene_cor'] = raw_gene_cor
        tracks_df.loc[rna_cols, 'norm_gene_cor'] = norm_gene_cor
    
        # GENE PLOTS
        print("Generating Gene-level plots...", flush = True)
        # Gene Expression Heatmap       
        fig = plot_strand_expression_heatmaps(
            gene_P=gP_rna,
            gene_T=gT_rna,
            gene_cors={'raw gene r': raw_gene_cor, 'norm gene r': norm_gene_cor},
            gene_cors_by_gene={'norm track r': norm_track_cor},   # gene-indexed series
            figsize=(26, 15)
        )
        fig.savefig(fig_dir / "gene_expression_heatmap.pdf", bbox_inches='tight', dpi=300) 
        plt.close(fig)

        # Gene Counts Scatter
        idx_max = {'raw': raw_gene_cor.idxmax(),'gene': norm_gene_cor.idxmax()}
        idx_min = {'raw': raw_gene_cor.idxmin(),'gene': norm_gene_cor.idxmin()}
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        fig.suptitle(f"Gene Counts Scatter (Max & Min {model_name} Tracks)", fontsize=14, fontweight='bold')
        density_scatter(gP_rna[idx_max['raw']], gT_rna[idx_max['raw']], 
                        ax=axes[0,0], title=f"Raw Max\n{idx_max['raw']}\n(r={raw_gene_cor[idx_max['raw']]:.3f})")
        density_scatter(gene_norm_P[idx_max['gene']], gene_norm_T[idx_max['gene']], 
                        ax=axes[0,1], title=f"Norm Gene Max\n{idx_max['gene']}\n(r={norm_gene_cor[idx_max['gene']]:.3f})")
        density_scatter(gP_rna[idx_min['raw']], gT_rna[idx_min['raw']], 
                        ax=axes[1,0], title=f"Raw Min\n{idx_min['raw']}\n(r={raw_gene_cor[idx_min['raw']]:.3f})")
        density_scatter(gene_norm_P[idx_min['gene']], gene_norm_T[idx_min['gene']], 
                        ax=axes[1,1], title=f"Norm Gene Min\n{idx_min['gene']}\n(r={norm_gene_cor[idx_min['gene']]:.3f})")
        plt.tight_layout()
        plt.savefig(fig_dir / "gene_counts_scatter.pdf", bbox_inches='tight', dpi=300) 
        plt.close(fig)
                
        #Gene correlation figure
        rna_track_names = raw_gene_cor.index.tolist()
        fig, axes = plt.subplots(1, 2, figsize=(10, 3+(len(raw_gene_cor)/30)*5), sharey=True)
        fig.suptitle(f'{model_name} Gene Correlations Across Normalization Methods', fontsize=17, fontweight='bold', y=0.98)
        # Raw Gene Correlations
        axes[0].barh(rna_track_names, raw_gene_cor.values, color='skyblue', edgecolor='black')
        axes[0].set_title('Raw Gene Correlations', fontsize=14)
        axes[0].set_xlabel('Pearson R')
        # Normalized Across Genes
        axes[1].barh(rna_track_names, norm_gene_cor.values, color='lightgreen', edgecolor='black')
        axes[1].set_title('Normalized Across Genes', fontsize=14)
        axes[1].set_xlabel('Pearson R')
        plt.tight_layout()
        plt.savefig(fig_dir / "gene_corr.pdf", bbox_inches='tight', dpi=300) 
        plt.close(fig)

    # Save metrics CSV
    tracks_df.to_csv(cor_dir / "track_correlations.csv")

    # Text Summary
    _write_summary(cor_dir,
                   model_name, analysis_name, tracks_df, coverage, coverage_cutoff,
                   concat_r, raw_gene_cor, norm_gene_cor, norm_track_cor,
                  num_intervals=num_intervals, num_tracks=len(tracks_df), seq_len=seq_len, label_len=label_len,  bin_size=bin_size)      

    # Plots 
    track_names = tracks_df.index.tolist()
    num_tracks = len(track_names)
    
    # CONCATENATED CORRELATION PER TRACK
    fig_width = max(12, 12 * (num_tracks / 30))
    plt.figure(figsize=(fig_width, 5))
    modality_colors = {'ATAC': 'red','RNA': 'orange','CnT': 'green'}
    rgba_colors = []
    edge_colors = []
    for idx, track_id in enumerate(track_names):
        mod = tracks_df['Assay_type'].iloc[idx]
        cov = coverage.iloc[idx]
        base_color = modality_colors.get(mod, 'gray') # fallback to gray if unknown modality
        # Check if it passes the coverage cutoff
        if pd.isna(cov) or cov < coverage_cutoff:
            rgba_colors.append(mcolors.to_rgba(base_color, alpha=0.3)) # Faded
            edge_colors.append(mcolors.to_rgba('gray', alpha=0.5))
        else:
            rgba_colors.append(mcolors.to_rgba(base_color, alpha=1.0)) # Solid
            edge_colors.append('black')

    bar_positions = np.arange(1, num_tracks + 1)
    plt.bar(bar_positions, concat_r.values, color=rgba_colors, edgecolor=edge_colors, linewidth=1)
    plt.axhline(y=0, color='black', linewidth=0.8, zorder=0) # Add a baseline 
    plt.xlabel('')
    plt.ylabel('Concatenated Pearson r')
    plt.xticks(ticks=bar_positions, labels=track_names, rotation=45, ha='right')
    plt.title(f'Concatenated Pearson r per Track ({model_name})')
    legend_elements = [
        Patch(facecolor='red', edgecolor='black', label='ATAC'),
        Patch(facecolor='orange', edgecolor='black', label='RNA'),
        Patch(facecolor='green', edgecolor='black', label='CnT'),
        Patch(facecolor='white', edgecolor='gray', hatch='///', alpha=0.5, label=f'< {coverage_cutoff*100:.1f}% Valid Bins') 
    ]
    plt.legend(handles=legend_elements, loc='best')
    plt.tight_layout()
    plt.savefig(fig_dir / "concatenated_correlation_barplot.pdf", bbox_inches='tight', dpi=300)
    plt.close()
    
    # QC: COVERAGE DISTRIBUTION PLOT   
    filtered_interval_coverage_matrix = [col[~np.isnan(col)] for col in interval_coverage_matrix.T]     # Filter out NaNs (if any intervals were totally skipped/failed) for the boxplot
    fig_width = max(12, 12 * (num_tracks / 30))
    plt.figure(figsize=(fig_width, 5))
    plt.axhline(y=coverage_cutoff, color='red', linewidth=1.5, zorder=0, alpha=0.8, 
                label=f'Track Cutoff ({coverage_cutoff*100:.1f}%)')
    plt.boxplot(filtered_interval_coverage_matrix, flierprops=dict(marker='o', markerfacecolor='black', markeredgecolor='black', markersize=2, alpha=1))
    plt.xlabel('')
    plt.ylabel('Covered Bins per Interval')
    plt.xticks(ticks=np.arange(1, num_tracks + 1), labels=track_names, rotation=45, ha='right')
    plt.title(f'Internal Coverage: Percentage of non-NaN Bins per Interval ({model_name})')    
    plt.tight_layout()
    plt.savefig(fig_dir / "QC_coverage_distribution_across_intervals.pdf", bbox_inches='tight', dpi=300) 
    plt.close()
#     # Concatenated Scatter (using reservoir sample)
#     fig, axes = plt.subplots(1, 3, figsize=(15, 4))
#     for ax_idx, mod in enumerate(['ATAC', 'RNA', 'CnT']):
#         mask = (tracks_df['Assay_type'] == mod).values
#         if not mask.any(): continue
        
#         mod_r = concat_r[mask]
#         if mod_r.isna().all(): continue
            
#         best_track_name = mod_r.idxmax()
#         t_idx = tracks_df.index.get_loc(best_track_name)
        
#         p_samp = np.log1p(reservoir.reservoir_P[t_idx])
#         t_samp = np.log1p(reservoir.reservoir_T[t_idx])
        
#         valid = ~np.isnan(p_samp) & ~np.isnan(t_samp)
#         if valid.sum() > 0:
#             density_scatter(p_samp[valid], t_samp[valid], ax=axes[ax_idx], 
#                             title=f"Best {mod}: {best_track_name}\n(r={mod_r.max():.3f})")
#     plt.tight_layout()
#     plt.savefig(fig_dir / "concatenated_counts_scatter.pdf", dpi=300)
#     plt.close()

    # Best Interval Coverage Tracks
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    for ax_idx, mod in enumerate(['ATAC', 'RNA', 'CnT']):
        data = best_tracker.best_data[mod]
        if data is None: continue
            
        # We plot the first track of that modality for the best interval
        compare_track_coverage(
            data["P"][0].flatten(), 
            data["T"][0].flatten(),
            pos=data["coord"],
            ax=axes[ax_idx],
            log_transform = False, #because we pass squashed scale counts we dont do log tranform  
            title=f"Best {mod} Interval: {data['coord']}"
        )
    plt.tight_layout()
    plt.savefig(fig_dir / "individual_track_coverage.pdf", dpi=300)
    plt.close()

    # Summary Beeswarm
    plot_data = [concat_r[tracks_df['Assay_type'] == mod].dropna() for mod in ['ATAC', 'RNA', 'CnT']]
    if gene_acc is not None:
        fig, axes = plt.subplots(1, 2, figsize=(7, 4), gridspec_kw={'width_ratios': [1, 1]})
        plot_correlation_beeswarm(plot_data, labels=['ATAC', 'RNA', 'CnT'], ax= axes[0], title=f"{model_name} Concatenated Correlations")
        plot_correlation_beeswarm([raw_gene_cor,norm_gene_cor, norm_track_cor],
                                  labels=["raw gene", "norm gene", "norm track"],
                                  ax= axes[1],
                                  title=f"{model_name} Resolution Correlations")
        plt.tight_layout()
        plt.savefig(fig_dir / "summary_beeswarm.pdf", dpi=300)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(4, 4))
        plot_correlation_beeswarm(plot_data, labels=['ATAC', 'RNA', 'CnT'], ax=ax, title=f"{model_name} Concat Correlations")
        plt.tight_layout()
        plt.savefig(fig_dir / "summary_beeswarm.pdf", dpi=300)
        plt.close(fig)


def _write_summary(cor_dir, model_name, analysis_name, tracks_df, coverage, coverage_cutoff, concat_r, 
                   raw_gene_cor, norm_gene_cor, norm_track_cor, 
                   num_intervals, num_tracks, seq_len, label_len, bin_size):
    summary_path = cor_dir / f"{analysis_name}_summary_metrics.txt"
    with open(summary_path, "w") as f:
        f.write("====================================================\n")
        f.write(f"Seq2Fun Streaming Evaluation Summary - {model_name}\n")
        f.write(f"Analysis Name: {analysis_name}\n")
        f.write("====================================================\n\n")
        
        f.write("GLOBAL SETUP & QC\n")
        f.write(f"Total Intervals:      {num_intervals}\n")
        f.write(f"Total Tracks:         {num_tracks}\n")
        f.write(f"Input Seq Length:     {seq_len}\n")
        f.write(f"Label Seq  Length:    {label_len}\n")
        f.write(f"Bin Size:             {bin_size}\n")      
        f.write(f"Average Valid Bins (non-NaN): {coverage.mean() * 100:.2f}%\n")
        f.write(f"Dynamic NaN Coverage Cutoff): {coverage_cutoff * 100:.2f}%\n")
        
        f.write("BIN-RESOLUTION (BY MODALITY)\n")
        for mod in ["ATAC", "RNA", "CnT"]:
            mask = tracks_df['Assay_type'] == mod
            if mask.sum() > 0:
                f.write(f"[{mod}] Concatenated Pearson r | Median: {concat_r[mask].median():.4f} | Max: {concat_r[mask].max():.4f} | Min: {concat_r[mask].min():.4f}\n")

        if raw_gene_cor is not None:
            f.write("\nGENE-RESOLUTION (RNA ONLY)\n")
            f.write(f"Raw Gene Median r:        {raw_gene_cor.median():.4f}\n")
            f.write(f"Norm-Gene Median r:       {norm_gene_cor.median():.4f}\n")
            f.write(f"Norm-Track Median r:      {norm_track_cor.median():.4f}\n")
        else:
            f.write("GENE-RESOLUTION\n")
            f.write("No RNA tracks detected. Gene metrics skipped.\n")
    print(f"Summary metrics written to: {summary_path}")
            

# ---------------------------------------------------------------------------
# ENTRY POINTS
# ---------------------------------------------------------------------------
def evaluate_from_inference(
    data_module, model_forward_fn, intervals_df, tracks_df, bw_paths_df, 
    bin_size, out_dir, model_name, analysis_name, is_squashed_scale, gtf_file=None, batch_size=16
):
    """Entry point to run evaluation live during model inference."""
    
    # Create a generator that yields batches
    def batch_generator():
        for batch_idx, batch in enumerate(data_module.iter_batches(split="test")):
            start_idx = batch_idx * batch_size

            # Pass to your forward function
            P_batch = model_forward_fn(batch, start_idx)
            
            # Extract coordinates for this batch
            # Assuming batch has a way to get the interval coordinates
            coords_batch = intervals_df.iloc[start_idx : start_idx + len(P_batch)][['chrom', 'start', 'end']].apply(tuple, axis=1).tolist()
            
            yield start_idx, P_batch, coords_batch

    _run_analysis_loop(
        batch_iterator=batch_generator(),
        num_intervals=len(intervals_df),
        intervals_df=intervals_df,
        tracks_df=tracks_df,
        bw_paths_df=bw_paths_df,
        bin_size=bin_size,
        out_dir=out_dir,
        model_name=model_name,
        analysis_name=analysis_name,
        gtf_file=gtf_file,
        is_squashed_scale = is_squashed_scale
    )

# ---------------------------------------------------------------------------
# H5 STREAMING HELPERS
# ---------------------------------------------------------------------------
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

def _get_alignment_axes(shape, num_intervals=None, num_tracks=None, head_name="Single Head"):
    """
    Infers the axes mapping from the dataset shape on disk without loading it into RAM.
    Returns the tuple (ax_intervals, ax_tracks, ax_bins) so slices can be transposed properly.
    """
    if len(shape) != 3:
        raise ValueError(f"[{head_name}] Expected 3D prediction array, got {len(shape)}D.")

    ax_intervals, ax_tracks, ax_bins = 0, 1, 2
    
    # find the intervals Axis
    if num_intervals is not None:
        if shape[0] == num_intervals:
            ax_intervals = 0
        elif num_intervals in shape:
            ax_intervals = shape.index(num_intervals)
        else:
            raise ValueError(f"[{head_name}] Shape {shape} does not contain an axis matching {num_intervals} intervals.")
            
    # find the Tracks Axis
    if num_tracks is not None:
        remaining_axes = [i for i in range(3) if i != ax_intervals]
        
        if shape[remaining_axes[0]] == num_tracks and shape[remaining_axes[1]] != num_tracks:
            ax_tracks, ax_bins = remaining_axes[0], remaining_axes[1]
        elif shape[remaining_axes[1]] == num_tracks and shape[remaining_axes[0]] != num_tracks:
            ax_tracks, ax_bins = remaining_axes[1], remaining_axes[0]
        elif shape[remaining_axes[0]] == num_tracks and shape[remaining_axes[1]] == num_tracks:
            ax_tracks, ax_bins = remaining_axes[0], remaining_axes[1]
        else:
            raise ValueError(f"[{head_name}] Shape {shape} does not contain an axis matching {num_tracks} tracks.")

    elif num_intervals is not None:
        remaining_axes = [i for i in range(3) if i != ax_intervals]
        ax_tracks, ax_bins = remaining_axes[0], remaining_axes[1]

    if (ax_intervals, ax_tracks, ax_bins) != (0, 1, 2):
        print(f"[{head_name}] Auto-aligning axes: Mapping intervals->axis{ax_intervals}, "
              f"tracks->axis{ax_tracks}, bins->axis{ax_bins}.", flush = True)

    return ax_intervals, ax_tracks, ax_bins

# ---------------------------------------------------------------------------
# HDF5 ENTRY POINT
# ---------------------------------------------------------------------------
def evaluate_from_h5(
    h5_path, bw_paths_df, bin_size, out_dir, model_name, analysis_name, is_squashed_scale,
    gtf_file=None, batch_size=16
):
    """Entry point to run streaming evaluation on an already-saved HDF5 file."""
    
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    # First, let's open the file just to extract metadata
    with h5py.File(h5_path, 'r') as f:
        intervals_df = None
        num_intervals = None
        if "intervals" in f:
            intervals_df = _extract_h5_group_to_df(f["intervals"])
            # Format columns to standard names just in case
            intervals_df = intervals_df.rename(columns={
                "CHROM": "chrom", "#CHROM": "chrom", "Chrom": "chrom", "Chromosome": "chrom", "chromosome": "chrom",
                "Start": "start", "START": "start", "End": "end", "END": "end"
            })
            # Ensure coordinates are integers
            intervals_df['start'] = intervals_df['start'].astype(int)
            intervals_df['end'] = intervals_df['end'].astype(int)
            num_intervals = len(intervals_df)
        else:
            raise ValueError(f"'intervals' group missing in {h5_path.name}")

        # Map datasets and figure out alignments
        datasets_info = []
        tracks_list = []
        
        if "predictions" in f:
            # Single-Head (Borzoi)
            ds = f["predictions"]
            t_df = _extract_h5_group_to_df(f.get("tracks"))
            num_tracks = len(t_df) if t_df is not None else None
            ax_map = _get_alignment_axes(ds.shape, num_intervals, num_tracks, "SingleHead")
            
            datasets_info.append({"dataset": ds.name, "shape": ds.shape, "ax_map": ax_map})
            if t_df is not None: tracks_list.append(t_df)
        else:
            # Multi-Head (AG)
            for head_name in f.keys():
                if head_name == "intervals": continue 
                head_grp = f[head_name]
                if "predictions" not in head_grp: continue 
                
                ds = head_grp["predictions"]
                t_df = _extract_h5_group_to_df(head_grp.get("tracks"))
                num_tracks = len(t_df) if t_df is not None else None
                ax_map = _get_alignment_axes(ds.shape, num_intervals, num_tracks, head_name)
                
                datasets_info.append({"dataset": ds.name, "shape": ds.shape, "ax_map": ax_map})
                if t_df is not None: tracks_list.append(t_df)
                
        if not datasets_info:
            raise ValueError(f"Could not find any 'predictions' datasets in {h5_path.name}")

        # Concatenate tracks DataFrames row-wise
        if tracks_list and len(tracks_list) == len(datasets_info):
            tracks_df = pd.concat(tracks_list, ignore_index=True)
            from utils.eval.track_prediction import process_track_metadata
            tracks_df = process_track_metadata(tracks_df)
            tracks_df = tracks_df.set_index("id")
        else:
            raise ValueError(f"Missing track metadata in {h5_path.name}")
            
    # HANDLE BORZOI INTERVAL CROPPING
    # Get the number of bins directly from the model predictions on disk
    ax_intervals, ax_tracks, ax_bins = datasets_info[0]["ax_map"]
    num_bins_p = datasets_info[0]["shape"][ax_bins]
    SEQ_LEN = intervals_df['end'].iloc[0] - intervals_df['start'].iloc[0]
    LABEL_LEN = num_bins_p * bin_size
    if SEQ_LEN != LABEL_LEN:
        print(f"WARNING: Resizing intervals from {SEQ_LEN} to {LABEL_LEN} to match prediction bins.", flush = True)
        CROP = (SEQ_LEN - LABEL_LEN) // 2
        intervals_df['start'] = intervals_df['start'] + CROP
        intervals_df['end'] = intervals_df['end'] - CROP
    print(f"Using SEQ_LEN: {SEQ_LEN}, LABEL_LEN: {LABEL_LEN}, BIN_SIZE: {bin_size}, N_BINS: {num_bins_p} as inferred from h5 file.",
          flush = True) 

    # Create the generator that slices lazily
    def h5_generator():
        # Keep file open during streaming
        with h5py.File(h5_path, 'r') as f:
            # Re-fetch dataset pointers for the open file session
            active_datasets = []
            for info in datasets_info:
                ds_path = info["dataset"]
                active_datasets.append({
                    "dataset": f[ds_path],
                    "ax_map": info["ax_map"]
                })

            for start_idx in range(0, num_intervals, batch_size):
                end_idx = min(start_idx + batch_size, num_intervals)
                
                batch_slices = []
                for info in active_datasets:
                    ds = info["dataset"]
                    ax_intervals, ax_tracks, ax_bins = info["ax_map"]
                    
                    # Create a slicing tuple: slice(None) means ':'
                    slices = [slice(None)] * 3
                    slices[ax_intervals] = slice(start_idx, end_idx)
                    
                    # Read only the batch from disk!
                    p_chunk = ds[tuple(slices)]
                    
                    # Transpose to (Intervals, Tracks, Bins) if needed
                    if (ax_intervals, ax_tracks, ax_bins) != (0, 1, 2):
                        p_chunk = np.transpose(p_chunk, (ax_intervals, ax_tracks, ax_bins))
                        
                    batch_slices.append(p_chunk)
                
                # Concatenate along the tracks axis
                P_batch = np.concatenate(batch_slices, axis=1)
                
                # Get coordinates
                coords_batch = intervals_df.iloc[start_idx:end_idx][['chrom', 'start', 'end']].apply(tuple, axis=1).tolist()
                
                yield start_idx, P_batch, coords_batch

    # Hand off to the core loop
    _run_analysis_loop(
        batch_iterator=h5_generator(),
        num_intervals=num_intervals,
        intervals_df=intervals_df,
        tracks_df=tracks_df,
        bw_paths_df=bw_paths_df,
        bin_size=bin_size,
        out_dir=out_dir,
        model_name=model_name,
        analysis_name=analysis_name,
        gtf_file=gtf_file,
        seq_len=SEQ_LEN,
        label_len=LABEL_LEN,
        is_squashed_scale = is_squashed_scale
    )