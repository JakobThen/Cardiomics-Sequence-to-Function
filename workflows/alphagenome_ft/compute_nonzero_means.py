"""Computes the non-zero mean of every BigWig file in a directory (workflow step 00).

The mean is restricted to canonical chromosomes (chr1-22, X, Y), see
`cardiomics_s2f.preprocessing.bigwig_stats`. Results are appended file by file,
so partial results survive if the job times out.

Output CSV columns: ``basename,nonzero_mean_canonical,full_path``. Files without
non-zero values get ``NaN``; unreadable files get ``Error_Reading_File``.

Usage (normally via 00_compute_nonzero_means.sh):
    python workflows/alphagenome_ft/compute_nonzero_means.py --input_dir DIR --output_csv OUT.csv
"""
from __future__ import annotations

import argparse
import math
import os
from datetime import datetime
from pathlib import Path

from cardiomics_s2f.preprocessing.bigwig_stats import bigwig_nonzero_mean


def get_parser() -> argparse.ArgumentParser:
    """Creates and configures the argument parser for the non-zero mean computation.

    Returns:
        argparse.ArgumentParser: The configured argument parser object.
    """
    parser = argparse.ArgumentParser(description="Compute non-zero means of BigWig files")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory searched recursively (following symlinks) for .bw files")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Output CSV file (overwritten)")
    return parser


def find_bigwigs(input_dir) -> list[str]:
    """Lists all ``.bw`` files below ``input_dir``, following symlinked folders.

    Args:
        input_dir (str or Path): Directory to search.

    Returns:
        list of str: Sorted file paths, built from ``input_dir`` (not resolved).
    """
    found = []
    for root, _, files in os.walk(input_dir, followlinks=True):
        found.extend(os.path.join(root, f) for f in files if f.endswith(".bw"))
    return sorted(found)


def main(args: argparse.Namespace) -> None:
    """Computes the non-zero mean per BigWig file and writes the CSV.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting on {datetime.now()}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output file: {output_csv}")

    with open(output_csv, "w") as out:
        out.write("basename,nonzero_mean_canonical,full_path\n")
        for bw_file in find_bigwigs(args.input_dir):
            try:
                mean_val = bigwig_nonzero_mean(bw_file)
                value = "NaN" if math.isnan(mean_val) else str(mean_val)
            except Exception:
                value = "Error_Reading_File"
            out.write(f"{os.path.basename(bw_file)},{value},{bw_file}\n")
            out.flush()
            print(f"Processed: {os.path.basename(bw_file)}", flush=True)

    print(f"Finished on {datetime.now()}. Results saved to {output_csv}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
