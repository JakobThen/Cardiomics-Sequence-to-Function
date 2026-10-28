"""Converts a variant CSV/TSV into a variant-scorer input file (used by step 03).

Normalises the columns to the variant-scorer ``chrombpnet`` schema
(``chr, pos, allele1, allele2, variant_id``), optionally removes variants whose REF does
not match the genome, and trims the shared leading bases of indels
(:func:`~cardiomics_s2f.vep.chrombpnet.write_chrombpnet_variants`).

Usage:
    python workflows/chrombpnet/convert_variants.py --input_file variants.csv \\
        --output_file variants_for_chrombpnet.tsv --genome genome.fa
"""
import argparse

from cardiomics_s2f.vep.chrombpnet import write_chrombpnet_variants


def get_parser() -> argparse.ArgumentParser:
    """Creates the argument parser (separate so Sphinx can document it).

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Convert CSV/TSV variant tables to ChromBPNet-compatible TSV")
    parser.add_argument("--input_file", required=True,
                        help="Input CSV or TSV file (comma or tab separated) with CHROM, POS, REF, ALT (+ variant_id)")
    parser.add_argument("--output_file", default="variants_for_chrombpnet.tsv",
                        help="Output TSV (no header; default: variants_for_chrombpnet.tsv)")
    parser.add_argument("--genome", default=None,
                        help="Reference genome FASTA; if given, variants with a mismatching REF are removed")
    return parser


def main(args: argparse.Namespace) -> None:
    """Runs the conversion.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    write_chrombpnet_variants(args.input_file, args.output_file, args.genome)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
