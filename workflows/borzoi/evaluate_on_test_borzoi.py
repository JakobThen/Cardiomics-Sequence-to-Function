"""Evaluates a fine-tuned Borzoi model on its test intervals (workflow step 03).

Streams the test predictions that ``finetune_borzoi.py`` saved
(``<prefix>_squashed_predictions.h5``) against the observed BigWig tracks with
:func:`~cardiomics_s2f.eval.streamed_correlation_analysis.evaluate_from_h5`: Pearson r per
track over the concatenated test intervals, coverage QC and, for RNA tracks, gene-level
correlations of summed exon counts (needs ``--gtf``). Correlations are computed on the
squashed scale, gene-level counts on the unsquashed scale.

The BigWigs are found from the staged paths stored in the H5's track table
(:func:`~cardiomics_s2f.finetuning.borzoi.training.resolve_staged_track_paths`), so each
track maps to the exact folder it was trained on.

Output: ``<out_dir>/<model_name>/<analysis_name>/`` with ``track_correlations.csv``, a
summary ``.txt`` and ``fig/``.
"""

import argparse
from pathlib import Path

from cardiomics_s2f.eval.streamed_correlation_analysis import evaluate_from_h5
from cardiomics_s2f.eval.track_prediction import read_h5_track_table
from cardiomics_s2f.finetuning.borzoi.training import resolve_staged_track_paths


def get_parser() -> argparse.ArgumentParser:
    """Creates the argument parser (separate so Sphinx can document it).

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Streamed test-set evaluation of a fine-tuned Borzoi model")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Test predictions HDF5 written by finetune_borzoi.py (<prefix>_squashed_predictions.h5)")
    parser.add_argument("--bigwig_dir", type=str, required=True,
                        help="Folder containing the *bigwig* folders the model was trained on")
    parser.add_argument("--out_dir", type=str, required=True, help="Output folder (results in <out_dir>/<model_name>/<analysis_name>/)")
    parser.add_argument("--gtf", type=str, default=None, help="GTF annotation for the gene-level RNA correlations")
    parser.add_argument("--bin_size", type=int, default=32, help="Bin size of the predictions in bp (default: 32)")
    parser.add_argument("--model_name", type=str, default="Borzoi",
                        help="Model name used for the output folder (default: Borzoi)")
    parser.add_argument("--analysis_name", type=str, default=None,
                        help="Name of the result subfolder (default: the predictions file name without extension)")
    parser.add_argument("--rna_scale", choices=["squashed", "linear"], default="squashed",
                        help="Scale of the saved RNA predictions (default: squashed)")
    parser.add_argument("--batch_size", type=int, default=16, help="Intervals read from the HDF5 per batch")
    return parser


def main(args: argparse.Namespace) -> None:
    """Resolves the BigWigs of the saved tracks and runs the streamed evaluation.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    predictions = Path(args.predictions)
    analysis_name = args.analysis_name or predictions.stem

    tracks = read_h5_track_table(predictions)
    bw_paths_df = resolve_staged_track_paths(tracks, args.bigwig_dir)
    print(f"Resolved {len(bw_paths_df)} BigWig files under {args.bigwig_dir}.", flush=True)

    evaluate_from_h5(
        h5_path=predictions,
        bw_paths_df=bw_paths_df[["path"]],
        bin_size=args.bin_size,
        out_dir=args.out_dir,
        model_name=args.model_name,
        analysis_name=analysis_name,
        is_squashed_scale=args.rna_scale == "squashed",
        gtf_file=args.gtf,
        batch_size=args.batch_size,
        transform="borzoi",
    )
    print(f"Done. Results in {Path(args.out_dir) / args.model_name / analysis_name}", flush=True)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
