"""DeepSHAP attributions of the fine-tuned AlphaGenome model for genomic regions (workflow step 05).

Loads the fine-tuned checkpoint, centres a window of ``--window_size`` on each
region of a BED file and computes DeepSHAP-style nucleotide attributions for each
requested track of one head (summed 1 bp output, optionally only over a centre
window). The logic lives in `cardiomics_s2f.finetuning.alphagenome.attributions`.

Usage (normally via 05_attributions.sh):
    python workflows/alphagenome_ft/compute_attributions.py --regions BED --config YAML
        --checkpoint_dir DIR --fasta_path FA --head multiome_ATAC
        --tracks ATAC_Cardiomyocytes ATAC_Fibroblasts --out_dir DIR

Output:
    <out_dir>/<out_prefix>_<track>.npy (shape (regions, window_size, 4), BED order)
    <out_dir>/<out_prefix>_regions.tsv (region names and window coordinates)
"""
import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from alphagenome.data import genome
from alphagenome_ft import load_checkpoint
from alphagenome_ft.finetune.config import load_targets_config, prepare_head_specs
from alphagenome_ft.finetune.train import register_predefined_heads
from pyfaidx import Fasta

from cardiomics_s2f.finetuning.alphagenome.attributions import (
    compute_deepshap_attributions,
    one_hot_encode_sequences,
)


def get_parser() -> argparse.ArgumentParser:
    """Creates the argument parser (separate so Sphinx can document it).

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(description="DeepSHAP attributions of the fine-tuned AlphaGenome model")
    # I/O
    parser.add_argument("--regions", type=str, required=True,
                        help="BED without header (chrom, start, end[, name]); a window is centred on each region")
    parser.add_argument("--config", type=str, required=True, help="Targets config of the fine-tuned model")
    parser.add_argument("--base_dir", type=str, default=None,
                        help="Base directory for relative paths in the config (defaults to the config's folder)")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Fine-tuning checkpoint directory (its 'best' checkpoint is used)")
    parser.add_argument("--fasta_path", type=str, required=True, help="Path to the genome.fa file")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--out_prefix", type=str, default="attributions", help="Output file prefix")
    # Target
    parser.add_argument("--head", type=str, required=True, help="Head to attribute (e.g. multiome_ATAC)")
    parser.add_argument("--tracks", type=str, nargs="+", required=True,
                        help="Track labels of the head (e.g. ATAC_Cardiomyocytes); one output file each")
    parser.add_argument("--unscaled", action="store_true",
                        help="Attribute predictions_1bp instead of scaled_predictions_1bp")
    # Method
    parser.add_argument("--window_size", type=int, default=16_384, help="Input window in bp (default: 16384)")
    parser.add_argument("--center_mask_size", type=int, default=500,
                        help="Width (bp) of the centre window whose output is attributed; 0 = whole window (default: 500)")
    parser.add_argument("--n_references", type=int, default=30, help="Reference sequences per region (default: 30)")
    parser.add_argument("--reference_type", type=str, default="dinucleotide_shuffle",
                        choices=["dinucleotide_shuffle", "shuffle", "uniform", "gc_match"],
                        help="How reference sequences are generated (default: dinucleotide_shuffle)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed (default: 42)")
    # Model
    parser.add_argument("--organism", type=str, default="HOMO_SAPIENS", choices=["HOMO_SAPIENS", "MUS_MUSCULUS"],
                        help="Organism (default: HOMO_SAPIENS)")
    parser.add_argument("--model_version", type=str, default="fold_1", help="Base model version (default: fold_1)")
    return parser


def read_regions(path) -> pd.DataFrame:
    """Reads a BED file (no header) with optional name column."""
    bed = pd.read_csv(path, sep="\t", header=None)
    regions = bed.iloc[:, :3].copy()
    regions.columns = ["chrom", "start", "end"]
    regions["name"] = bed.iloc[:, 3].astype(str) if bed.shape[1] >= 4 else [f"region_{i}" for i in range(len(bed))]
    return regions


def main(args: argparse.Namespace) -> None:
    """Computes and saves the attributions of every requested track.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Raises:
        ValueError: If the head or a track is not part of the targets config.
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config)
    base_dir = Path(args.base_dir) if args.base_dir else config_path.parent

    # 1. Heads and tracks
    config_dict = load_targets_config(config_path, base_dir=base_dir)
    head_specs = prepare_head_specs(config_dict)
    register_predefined_heads(head_specs)

    avail_heads = {spec.head_id: spec for spec in head_specs}
    if args.head not in avail_heads:
        raise ValueError(f"Head '{args.head}' not in config. Available: {list(avail_heads)}")
    track_names = [info.name for info in avail_heads[args.head].tracks]
    missing = [t for t in args.tracks if t not in track_names]
    if missing:
        raise ValueError(f"Tracks {missing} not in head {args.head}: {track_names}")

    # 2. Sequences
    regions = read_regions(args.regions)
    hg38 = Fasta(args.fasta_path)
    windows = [
        genome.Interval(chromosome=r.chrom, start=int(r.start), end=int(r.end)).resize(args.window_size)
        for r in regions.itertuples()
    ]
    seqs = [str(hg38[iv.chromosome][iv.start:iv.end]) for iv in windows]
    seq_1hot = one_hot_encode_sequences(seqs)  # (regions, window_size, 4)
    organism_index = (jnp.zeros if args.organism == "HOMO_SAPIENS" else jnp.ones)(seq_1hot.shape[0], dtype=jnp.int32)

    regions["window_start"] = [iv.start for iv in windows]
    regions["window_end"] = [iv.end for iv in windows]
    regions.to_csv(out_dir / f"{args.out_prefix}_regions.tsv", sep="\t", index=False)
    print(f"{len(regions)} regions, window {args.window_size} bp", flush=True)

    # 3. Model and attributions
    model = load_checkpoint(Path(args.checkpoint_dir) / "best", base_model_version=args.model_version,
                            init_seq_len=args.window_size)
    for track in args.tracks:
        print(f"Computing attributions for {args.head} / {track} ...", flush=True)
        attributions = compute_deepshap_attributions(
            model=model,
            sequence=jnp.array(seq_1hot),
            organism_index=organism_index,
            head_name=args.head,
            output_key=track,
            track_names=track_names,
            return_scaled=not args.unscaled,
            n_references=args.n_references,
            reference_type=args.reference_type,
            random_state=args.random_state,
            center_mask_size=args.center_mask_size or None,
        )
        out_file = out_dir / f"{args.out_prefix}_{track}.npy"
        np.save(out_file, np.asarray(attributions))
        print(f"  -> {out_file} {attributions.shape}", flush=True)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
