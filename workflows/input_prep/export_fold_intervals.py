"""Exports AlphaGenome's train/valid/test fold intervals as BED files.

The intervals define the data split of AlphaGenome fine-tuning and are reused by
the other models (e.g. Borzoi) and by the pretrained-AlphaGenome benchmarks. Runs
in seconds on the login node; needs the AlphaGenome environment
(``CS2F_ENV_ALPHAGENOME_FT``), which provides ``alphagenome_ft``.

Usage:
    python workflows/input_prep/export_fold_intervals.py
    python workflows/input_prep/export_fold_intervals.py --fold 2 --out_dir DIR --overwrite

Output:
    <out_dir>/AG_{train,valid,test}_intervals_fold<fold>_<organism>.bed
    (default out_dir: ``intervals_dir`` from config/paths.yaml)
"""
from __future__ import annotations

import argparse

from alphagenome_ft.finetune.data import prepare_intervals_from_fold

from cardiomics_s2f.config import load_paths
from cardiomics_s2f.preprocessing.fold_intervals import write_fold_intervals


def get_parser() -> argparse.ArgumentParser:
    """Creates the argument parser (separate so Sphinx can document it).

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Export AlphaGenome fold intervals as BED files")
    parser.add_argument("--fold", type=str, default="1", help="Data fold (0, 1, 2, 3; default: 1)")
    parser.add_argument("--window_size", type=int, default=1_048_576,
                        help="Interval width in bp (default: 1 Mbp, as in fine-tuning)")
    parser.add_argument("--organism", type=str, default="HOMO_SAPIENS", help="Organism (default: HOMO_SAPIENS)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output folder (default: intervals_dir from config/paths.yaml)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing BED files")
    return parser


def main(args: argparse.Namespace) -> None:
    """Builds the fold intervals and writes one BED file per split.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    out_dir = args.out_dir or load_paths().intervals_dir
    intervals = prepare_intervals_from_fold(
        fold=args.fold,
        window_size=args.window_size,
        organism=args.organism,
    )
    paths = write_fold_intervals(intervals, out_dir, args.fold, args.organism, overwrite=args.overwrite)
    for split, path in paths.items():
        print(f"{split}: {len(intervals[split])} intervals -> {path}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
