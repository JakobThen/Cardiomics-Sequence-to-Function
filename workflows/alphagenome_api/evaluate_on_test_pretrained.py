"""Evaluates the pretrained AlphaGenome track predictions on the test intervals (workflow step 03).

Streams the aggregated predictions of step 02 (``<intervals>_<config>.h5``) against the
cardioid BigWigs with
:func:`~cardiomics_s2f.eval.streamed_correlation_analysis.evaluate_from_h5`, exactly as the
fine-tuned AlphaGenome test evaluation: the reference BigWigs and their non-zero means come
from the fine-tuned AlphaGenome targets config
(:func:`~cardiomics_s2f.eval.alphagenome_api.reference_bigwigs_from_targets_config`), and
correlations are computed on the AlphaGenome squashed scale (the linear predictions are
scaled by the track means and RNA tracks squashed). Gene-level RNA correlations use the
unsquashed exon counts (needs ``--gtf``).

Output: ``<out_dir>/<model_name>/<analysis_name>/`` with ``track_correlations.csv``, a
summary ``.txt`` and ``fig/``.
"""

import argparse
from pathlib import Path

from cardiomics_s2f.eval.alphagenome_api import reference_bigwigs_from_targets_config
from cardiomics_s2f.eval.streamed_correlation_analysis import evaluate_from_h5
from cardiomics_s2f.eval.track_prediction import read_h5_track_table


def get_parser() -> argparse.ArgumentParser:
    """Creates the argument parser (separate so Sphinx can document it).

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Streamed test-set evaluation of pretrained AlphaGenome predictions")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Predictions HDF5 written by predict_tracks.py (step 02)")
    parser.add_argument("--targets_config", type=str, required=True,
                        help="Fine-tuned AlphaGenome targets config (reference BigWigs and non-zero means)")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output folder (results in <out_dir>/<model_name>/<analysis_name>/)")
    parser.add_argument("--gtf", type=str, default=None, help="GTF annotation for the gene-level RNA correlations")
    parser.add_argument("--bin_size", type=int, default=1, help="Bin size of the predictions in bp (default: 1)")
    parser.add_argument("--model_name", type=str, default="AlphaGenome_pretrained",
                        help="Model name used for the output folder (default: AlphaGenome_pretrained)")
    parser.add_argument("--analysis_name", type=str, default=None,
                        help="Name of the result subfolder (default: the predictions file name without extension)")
    parser.add_argument("--batch_size", type=int, default=16, help="Intervals read from the HDF5 per batch")
    return parser


def main(args: argparse.Namespace) -> None:
    """Resolves the reference BigWigs and runs the streamed evaluation.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    predictions = Path(args.predictions)
    analysis_name = args.analysis_name or predictions.stem

    tracks = read_h5_track_table(predictions)
    bw_paths_df = reference_bigwigs_from_targets_config(tracks, args.targets_config)
    print(f"Resolved {len(bw_paths_df)} reference BigWigs from {args.targets_config}.", flush=True)

    evaluate_from_h5(
        h5_path=predictions,
        bw_paths_df=bw_paths_df,
        bin_size=args.bin_size,
        out_dir=args.out_dir,
        model_name=args.model_name,
        analysis_name=analysis_name,
        is_squashed_scale=False,
        gtf_file=args.gtf,
        batch_size=args.batch_size,
        transform="alphagenome",
    )
    print(f"Done. Results in {Path(args.out_dir) / args.model_name / analysis_name}", flush=True)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
