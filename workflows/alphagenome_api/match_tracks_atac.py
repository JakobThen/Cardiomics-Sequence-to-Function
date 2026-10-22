"""Matches pretrained AlphaGenome ATAC tracks to the multiome ATAC BigWigs (workflow step 01, ATAC).

Groups nearby candidate enhancer regions into shared prediction windows, predicts
all AlphaGenome ATAC tracks at base resolution and correlates every predicted with
every observed (BigWig) track per window, masking positions without signal. The
logic lives in `cardiomics_s2f.eval.alphagenome_api`. Needs ~2 TB of memory for
the full enhancer set (all windows are held in memory).

Usage (normally via 01_track_matching.sh):
    python workflows/alphagenome_api/match_tracks_atac.py --regions TSV --bw_dir DIR --out_dir DIR

Output:
    <out_dir>/AG_Multiome_ATAC_stats_enhancer_regions_<seq_len>.parquet
"""
import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from alphagenome.data import genome
from alphagenome.models import dna_client

from cardiomics_s2f.config import get_alphagenome_api_key
from cardiomics_s2f.eval.alphagenome_api import (
    group_regions_into_windows,
    list_atac_bigwigs,
    masked_pearson_per_window,
    read_bigwig_windows,
    track_match_long_table,
)
from cardiomics_s2f.vep.alphagenome_api import sequence_length_from_label


def get_parser() -> argparse.ArgumentParser:
    """Creates the argument parser (separate so Sphinx can document it).

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Match pretrained AlphaGenome ATAC tracks to multiome ATAC BigWigs")
    parser.add_argument("--regions", type=str, required=True,
                        help="TSV of candidate enhancer regions (columns seqnames, start, end)")
    parser.add_argument("--bw_dir", type=str, required=True,
                        help="Folder with the observed ATAC_<cell type>_100M_norm.bw files")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--seq_len", type=str, choices=["1MB", "500KB", "100KB", "16KB"], default="500KB",
                        help="Prediction window (default: 500KB)")
    parser.add_argument("--buffer_fraction", type=float, default=0.10,
                        help="Part of each window kept free of regions as flank (default: 0.10)")
    parser.add_argument("--min_valid_fraction", type=float, default=0.10,
                        help="Part of a window that must have BigWig signal to compute a correlation (default: 0.10)")
    parser.add_argument("--max_workers", type=int, default=1,
                        help="Parallel API requests (default: 1; more can break the API)")
    return parser


def main(args: argparse.Namespace) -> None:
    """Predicts, reads and correlates the ATAC tracks and writes the long table.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = dna_client.create(get_alphagenome_api_key())

    seq_len = sequence_length_from_label(args.seq_len)
    max_span = seq_len - int(seq_len * args.buffer_fraction)
    min_valid_bins = int(seq_len * args.min_valid_fraction)

    # 1. Prediction windows around the regions
    regions = pd.read_csv(args.regions, sep="\t")
    print("Grouping close regions to minimize API calls...")
    windows = group_regions_into_windows(regions, seq_len, max_span)
    api_intervals = [genome.Interval(chromosome=c, start=s, end=e) for c, s, e in windows]
    print(f"Success! Grouped {len(regions)} core regions into {len(api_intervals)} API requests.")

    # 2. Predictions
    print("Requesting predictions from API...")
    output = model.predict_intervals(
        intervals=api_intervals,
        requested_outputs=[dna_client.OutputType.ATAC],
        max_workers=args.max_workers,
        ontology_terms=None  # all tracks
    )

    print("Extracting prediction arrays...")
    pred_list = []
    intervals_data = []
    for tdata in output:
        intervals_data.append((tdata.atac.interval.chromosome, tdata.atac.interval.start, tdata.atac.interval.end))
        pred_list.append(tdata.atac.values)  # Shape: (seq_len, num_pred_tracks)

    # P shape: (num_intervals, seq_len, num_pred_tracks)
    P = np.stack(pred_list, axis=0, dtype=np.float32)
    pred_track_names = output[0].atac.metadata["biosample_name"].tolist()
    del output, pred_list; gc.collect()

    # 3. Observed signal
    print("Extracting BigWig arrays...")
    bw_paths = list_atac_bigwigs(args.bw_dir)
    T, bw_track_names = read_bigwig_windows(bw_paths, intervals_data)

    # 4. Correlations
    print("Computing Interval Pearson Corrs...")
    R_indiv, T_counts, P_counts = masked_pearson_per_window(P, T, min_valid=min_valid_bins)
    del P, T; gc.collect()

    print("Converting distributions and counts to Long-Format DataFrame...")
    api_labels = [f"{iv.chromosome}:{iv.start}-{iv.end}" for iv in api_intervals]
    df_indiv_long = track_match_long_table(R_indiv, T_counts, P_counts, api_labels, pred_track_names, bw_track_names)
    del R_indiv, P_counts, T_counts; gc.collect()

    out_file = out_dir / f"AG_Multiome_ATAC_stats_enhancer_regions_{args.seq_len}.parquet"
    df_indiv_long.to_parquet(out_file, index=False)
    print(f"Saved {out_file}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
