"""Predicts aggregated pretrained AlphaGenome tracks for genomic intervals (workflow step 02).

Predicts ATAC and stranded total RNA-seq for every interval of a BED file (e.g.
the test fold), averages the tracks of each aggregation group (ontology CURIEs
from the track aggregations config) and saves the predictions with a track table
that names, per group, the BigWig of the reference cell type it is evaluated
against. The logic lives in `cardiomics_s2f.eval.alphagenome_api`.

Usage (normally via 02_track_prediction.sh):
    python workflows/alphagenome_api/predict_tracks.py --intervals BED --config YAML
        --reference_tracks CM_and_earlyCM_tracks=Cardiomyocytes naive_heart_tracks=Cardiomyocytes
        --out_dir DIR

Output:
    <out_dir>/<intervals name>_<config name>.h5 (``predictions`` of shape
    (intervals, tracks, bins), ``intervals`` and ``tracks`` groups;
    ``.npy`` only if writing the HDF5 fails)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from alphagenome.data import genome
from alphagenome.models import dna_client

from cardiomics_s2f.config import get_alphagenome_api_key
from cardiomics_s2f.eval.alphagenome_api import aggregate_tracks_by_curies, prediction_track_table
from cardiomics_s2f.io.data_output import save_borzoi_predictions_with_fallback


def parse_reference_tracks(pairs):
    """Parses ``GROUP=CELL_TYPE`` pairs into an ordered dict."""
    mapping = {}
    for pair in pairs:
        group, sep, cell_type = pair.partition("=")
        if not sep or not group or not cell_type:
            raise ValueError(f"Expected GROUP=CELL_TYPE, got '{pair}'")
        mapping[group] = cell_type
    return mapping


def get_parser() -> argparse.ArgumentParser:
    """Creates the argument parser (separate so Sphinx can document it).

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Predict aggregated pretrained AlphaGenome tracks")
    parser.add_argument("--intervals", type=str, required=True,
                        help="BED with header chrom/start/end, all intervals of one supported length")
    parser.add_argument("--config", type=str, required=True,
                        help="Track aggregations YAML ({group: [ontology CURIE, ...]})")
    parser.add_argument("--reference_tracks", type=str, nargs="+", required=True,
                        help="GROUP=CELL_TYPE pairs: groups to predict (in this order) and the cell type "
                             "whose BigWigs each is evaluated against")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--max_workers", type=int, default=1, help="Parallel API requests (default: 1)")
    return parser


def main(args: argparse.Namespace) -> None:
    """Predicts, aggregates and saves the tracks.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    bed_file, config_file = Path(args.intervals), Path(args.config)
    prefix = f"{bed_file.stem}_{config_file.stem}"
    reference_tracks = parse_reference_tracks(args.reference_tracks)
    model = dna_client.create(get_alphagenome_api_key())

    # 1. Intervals
    print("\n[1/6] Loading BED file and constructing intervals...", flush=True)
    bed = pd.read_csv(bed_file, sep="\t")
    assert (bed.end - bed.start).nunique() == 1, "Multiple Interval lengths in .bed file"
    seq_len = bed.iloc[0].end - bed.iloc[0].start
    assert seq_len in dna_client.SUPPORTED_SEQUENCE_LENGTHS.values(), "bed file sequence length not supported by AG API"
    ivs = [genome.Interval(chromosome=c, start=s, end=e, strand=".", name="") for (c, s, e) in bed.itertuples(index=False)]
    print(f"    -> {len(ivs)} intervals loaded, sequence length: {seq_len} bp", flush=True)

    # 2. Aggregation groups
    print("\n[2/6] Loading track aggregation config...", flush=True)
    with open(config_file, 'r') as file:
        config_dict = yaml.safe_load(file)
    missing = [k for k in reference_tracks if k not in config_dict]
    if missing:
        raise KeyError(f"Groups not in {config_file}: {missing}")
    keys = list(reference_tracks)
    curies = list({c for key in keys for c in config_dict[key]})
    print(f"    -> {len(keys)} aggregation keys: {keys}", flush=True)
    print(f"    -> {len(curies)} unique ontology curies", flush=True)

    # 3. ATAC
    print(f"\n[3/6] Requesting ATAC predictions for {len(ivs)} intervals...", flush=True)
    output = model.predict_intervals(
        intervals=ivs, requested_outputs=[dna_client.OutputType.ATAC],
        max_workers=args.max_workers, ontology_terms=curies,
    )
    A_meta = output[0].atac.metadata
    A = np.array([aggregate_tracks_by_curies(o.atac.values.T, A_meta, config_dict, keys) for o in output])
    print(f"    -> ATAC array shape: {A.shape}", flush=True)
    del output, A_meta

    # 4. Stranded total RNA-seq
    print(f"\n[4/6] Requesting RNA-seq predictions for {len(ivs)} intervals...", flush=True)
    output = model.predict_intervals(
        intervals=ivs, requested_outputs=[dna_client.OutputType.RNA_SEQ],
        max_workers=args.max_workers, ontology_terms=curies,
    )
    R_meta = output[0].rna_seq.metadata
    is_stranded_rna_seq = (R_meta.strand.isin(["+", "-"])) & (R_meta["Assay title"] == "total RNA-seq")
    R_meta = R_meta[is_stranded_rna_seq]
    print(f"    -> {is_stranded_rna_seq.sum()} stranded RNA-seq tracks retained. Aggregating...", flush=True)
    R_pos = np.array([aggregate_tracks_by_curies(o.rna_seq.values.T[is_stranded_rna_seq, :], R_meta, config_dict, keys, strand="+") for o in output])
    R_neg = np.array([aggregate_tracks_by_curies(o.rna_seq.values.T[is_stranded_rna_seq, :], R_meta, config_dict, keys, strand="-") for o in output])
    del output, R_meta

    # 5. Combine: ATAC, RNA+, RNA- per group
    print("\n[5/6] Concatenating ATAC + RNA+/- arrays...", flush=True)
    P = np.concatenate([A, R_pos, R_neg], axis=1)
    print(f"    -> Final prediction array shape: {P.shape}", flush=True)
    del A, R_pos, R_neg
    P_meta = prediction_track_table(reference_tracks)

    # 6. Save
    print("\n[6/6] Saving predictions...", flush=True)
    save_borzoi_predictions_with_fallback(preds=P, out_dir=args.out_dir, file_prefix=prefix,
                                          tracks_df=P_meta, intervals_df=bed)
    print(f"\n[Done] All outputs written to {args.out_dir}", flush=True)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
