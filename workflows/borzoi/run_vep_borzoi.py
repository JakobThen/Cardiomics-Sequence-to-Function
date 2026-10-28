"""Variant effect prediction (VEP) with a fine-tuned Borzoi model (workflow step 04).

Variants (CSV with ``variant_id, CHROM, POS, REF, ALT``) are restricted to autosomes and
chrX, filtered against the blacklist (1 kb window) and against chromosome ends. SNVs are
predicted with gReLU's ``VariantDataset``; indels and MNVs with the stitching strategy of
:class:`~cardiomics_s2f.vep.borzoi.StitchedIndelDataset`. Both use reverse-complement
augmentation.

With ``--rna_scale squashed`` (default, for models trained on squashed RNA targets such as
``heads_only_deduplicated_squash``) the RNA predictions are unsquashed before scoring; use
``linear`` for models trained without the squash transform.

Output: ``VEP_Borzoi_<vcf name>.csv`` in ``--out_dir``, one row per variant and track
(and gene, for RNA): masked ``L2ED_score``/``logSED_score`` and whole-window
``SUM_score``/``L2_score``/``logSUM_score`` (see :mod:`cardiomics_s2f.vep.borzoi`).
"""

import argparse
from pathlib import Path

import pandas as pd

import grelu.lightning
from grelu.data.dataset import VariantDataset
from grelu.data.preprocess import filter_blacklist, filter_chromosomes

from cardiomics_s2f.finetuning.borzoi.training import find_checkpoint_file
from cardiomics_s2f.vep.borzoi import (
    StitchedIndelDataset,
    average_indel_augmentations,
    filter_variants_near_chrom_ends,
    read_gtf_exons,
    score_variants_masked_windows,
    score_variants_whole_window,
    stitch_indel_predictions,
    unsquash_rna_predictions,
)

GENOME = "hg38"


def parse_devices(value):
    """Parses ``--devices``: ``cpu`` or comma-separated GPU indices (e.g. ``0`` or ``0,1``)."""
    if value == "cpu":
        return "cpu"
    return [int(d) for d in value.split(",")]


def get_parser() -> argparse.ArgumentParser:
    """Creates the argument parser (separate so Sphinx can document it).

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Borzoi Variant Effect Prediction (VEP) with Stitching Strategy.")

    parser.add_argument("--input_vcf", required=True, type=str,
                        help="CSV with columns variant_id, CHROM, POS, REF, ALT (other columns are ignored)")
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Checkpoint (.ckpt file, or a run folder with exactly one epoch*.ckpt)")
    parser.add_argument("--out_dir", required=True, type=str, help="Output directory for the VEP CSV")
    parser.add_argument("--gtf", required=True, type=str, help="GTF annotation (exons for RNA scoring)")
    parser.add_argument("--chrom_sizes", required=True, type=str, help="Path to the genome.chrom.sizes file")
    parser.add_argument("--rna_scale", choices=["squashed", "linear"], default="squashed",
                        help="Scale of the model's RNA outputs; 'squashed' predictions are unsquashed before scoring "
                             "(default: squashed)")
    parser.add_argument("--devices", type=parse_devices, default=[0],
                        help="'cpu' or comma-separated GPU indices (default: 0)")
    parser.add_argument("--workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for prediction")
    return parser


def _predict_snvs(model, snvs, seq_len, args):
    snv_ds = VariantDataset(
        snvs,
        seq_len=seq_len,
        genome=GENOME,
        rc=True,
        max_seq_shift=0,
    )
    return model.predict_on_dataset(
        snv_ds,
        devices=args.devices,
        num_workers=args.workers,
        batch_size=args.batch_size,
        augment_aggfunc="mean",
    )  # b a t l: b = number of variants, a (alleles) = 2


def _predict_indels(model, indels, seq_len, args):
    indel_ds = StitchedIndelDataset(
        indels,
        seq_len=seq_len,
        genome=GENOME,
        rc=True,
        max_seq_shift=0,
    )
    indels_pred = model.predict_on_dataset(
        indel_ds,
        devices=args.devices,
        num_workers=args.workers,
        batch_size=args.batch_size,
        augment_aggfunc="mean",
    )  # (b n a) t l: gReLU does not reshape custom datasets
    indels_pred = average_indel_augmentations(
        indels_pred, n_augmented=indel_ds.n_augmented, n_alleles=indel_ds.n_alleles, rc=indel_ds.rc
    )
    # Stitch left and right matched alleles
    return stitch_indel_predictions(indels_pred, indels)


def main(args: argparse.Namespace) -> None:
    """Predicts, scores and writes the variant effects.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    vcf_file = Path(args.input_vcf)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    checkpoint = find_checkpoint_file(args.checkpoint)
    print(f"Loading checkpoint: {checkpoint}", flush=True)
    model = grelu.lightning.LightningModel.load_from_checkpoint(checkpoint, weights_only=False)
    seq_len = model.data_params["train"]["seq_len"]
    label_len = model.data_params["train"]["label_len"]
    track_names = model.data_params["tasks"]["name"]

    chrom_size_df = pd.read_csv(args.chrom_sizes, sep="\t", names=["chrom", "end"])
    chrom_sizes = dict(zip(chrom_size_df['chrom'], chrom_size_df['end']))

    # Load VCF
    var = pd.read_csv(vcf_file)
    var = var[["variant_id", "CHROM", "POS", "REF", "ALT"]].drop_duplicates()
    var = var.rename(columns={"CHROM": "chrom", "REF": "ref", "ALT": "alt", "POS": "pos"})

    # Filter to chromosomes of the model and remove blacklisted areas
    var = filter_chromosomes(var, include='autosomesX')
    var = filter_blacklist(var, genome=GENOME, window=1000)
    # Prevent extraction errors by removing variants whose windows reach the chromosome ends
    var = filter_variants_near_chrom_ends(var, chrom_sizes, seq_len)

    # SNVs go through gReLU's VariantDataset (2 sequences per variant instead of 3);
    # indels and MNVs (e.g. AGA -> CGC) through the stitching dataset
    is_snv = (var.ref.str.len() == 1) & (var.alt.str.len() == 1)
    snvs = var[is_snv].reset_index(drop=True)
    indels = var[~is_snv].reset_index(drop=True)
    print(f"{len(snvs)} SNVs and {len(indels)} indels/MNVs to score.", flush=True)

    exons_df = read_gtf_exons(args.gtf)
    print("Model and Data loaded.", flush=True)

    scores = []
    for name, variants, predict in (("SNV", snvs, _predict_snvs), ("Indel", indels, _predict_indels)):
        if variants.empty:
            print(f"No {name} variants, skipping.", flush=True)
            continue
        print(f"Starting {name} prediction...", flush=True)
        preds = predict(model, variants, seq_len, args)
        if args.rna_scale == "squashed":
            preds = unsquash_rna_predictions(preds, track_names)
        print(f"Finished {name} prediction, computing scores...", flush=True)

        masked = score_variants_masked_windows(preds, variants, exons_df, track_names, label_len)
        unmasked = score_variants_whole_window(preds, variants, track_names)
        scores.append(pd.merge(masked, unmasked, how="outer", on=["variant_id", "track"]))

    vep = pd.concat(scores, axis=0).sort_values(by=["variant_id", "assay_type", "track"])

    out_file = out_dir / f'VEP_Borzoi_{vcf_file.stem}.csv'
    vep.to_csv(out_file, index=False)
    print(f"Output scores saved at: {out_file}", flush=True)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
