"""Exports the Borzoi train/validation/test windows as BED files.

Loads the pretrained Borzoi ``human_rep0`` model through gReLU, takes its own data split
(524 kbp windows; not the AlphaGenome fold intervals), keeps autosomes and chrX, drops
windows past chromosome ends and windows within ``--blacklist_window`` bp of the
blacklist (:func:`~cardiomics_s2f.finetuning.borzoi.intervals.build_borzoi_fold_intervals`).
Runs on the login node (downloads the model on first use); needs the Borzoi environment
(``CS2F_ENV_BORZOI``).

Usage:
    python workflows/borzoi/export_borzoi_intervals.py
    python workflows/borzoi/export_borzoi_intervals.py --out_dir DIR --overwrite

Output:
    <out_dir>/Borzoi_human_rep0_{train,val,test}_intervals.bed (header ``chrom start end``;
    default out_dir: ``intervals_dir`` from config/paths.yaml)
"""
import argparse

import pandas as pd

import grelu.resources

from cardiomics_s2f.config import load_paths
from cardiomics_s2f.finetuning.borzoi.intervals import (
    borzoi_interval_bed_paths,
    build_borzoi_fold_intervals,
    write_borzoi_fold_intervals,
)


def get_parser() -> argparse.ArgumentParser:
    """Creates the argument parser (separate so Sphinx can document it).

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Export the Borzoi human_rep0 intervals as BED files")
    parser.add_argument("--chrom_sizes", type=str, default=None,
                        help="genome.chrom.sizes file (default: chrom_sizes from config/paths.yaml)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output folder (default: intervals_dir from config/paths.yaml)")
    parser.add_argument("--blacklist_window", type=int, default=1000,
                        help="Drop windows closer than this many bp to a blacklist region (default: 1000)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing BED files")
    return parser


def main(args: argparse.Namespace) -> None:
    """Builds the windows and writes one BED file per split.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    paths = None
    if args.chrom_sizes is None or args.out_dir is None:
        paths = load_paths()
    chrom_sizes_file = args.chrom_sizes or paths.chrom_sizes
    out_dir = args.out_dir or paths.intervals_dir

    # Fail before the model download if the files exist
    existing = [str(p) for p in borzoi_interval_bed_paths(out_dir).values() if p.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"Error: BED files already exist (use --overwrite): {existing}")

    chrom_sizes = pd.read_csv(chrom_sizes_file, sep="\t", names=["chrom", "end"])
    model = grelu.resources.load_model(project="borzoi", model_name="human_rep0")
    splits = build_borzoi_fold_intervals(model, chrom_sizes, blacklist_window=args.blacklist_window)
    written = write_borzoi_fold_intervals(splits, out_dir, overwrite=args.overwrite)
    for split, path in written.items():
        print(f"{split}: {len(splits[split])} intervals -> {path}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
