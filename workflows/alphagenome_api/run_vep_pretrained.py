"""Variant effect prediction with the pretrained AlphaGenome model (workflow step 04).

Scores variants in batches with the AlphaGenome API (recommended scorers of the
chosen modalities) and aggregates the scores per variant over each track group of
the track aggregations config. Finished batches are checkpointed, so an
interrupted run continues where it stopped. The logic lives in
`cardiomics_s2f.vep.alphagenome_api`.

Usage (normally via 04_vep.sh):
    python workflows/alphagenome_api/run_vep_pretrained.py --input_file CSV --config YAML
        --output_dir DIR --modalities atac rna_seq chip_histone chip_tf

Output:
    <output_dir>/<input name>_<group>_aggr.csv per track group
    (appended batch by batch; checkpoint file removed on completion)
"""
import argparse
import json
import os

import yaml
from alphagenome.data import genome
from alphagenome.models import dna_client
from alphagenome.models import variant_scorers

from cardiomics_s2f.config import get_alphagenome_api_key
from cardiomics_s2f.vep.alphagenome_api import (
    aggregate_scores_by_curies,
    read_variant_table,
    sequence_length_from_label,
    tidy_variant_scores,
)

VALID_MODALITIES = [
    "rna_seq", "cage", "procap", "atac", "dnase",
    "chip_histone", "chip_tf", "polyadenylation",
    "splice_sites", "splice_site_usage", "splice_junctions", "contact_maps"
]


def get_parser() -> argparse.ArgumentParser:
    """Creates the argument parser (separate so Sphinx can document it).

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Pretrained AlphaGenome VEP - batched prediction + aggregation")
    # Prediction args
    parser.add_argument("--input_file", type=str, required=True,
                        help="CSV/TSV with variants (CHROM, POS, REF, ALT; optional variant_id)")
    parser.add_argument("--interval_size", type=str, choices=["1MB", "500KB", "100KB", "16KB"], default="1MB",
                        help="Context window size (default: 1MB)")
    parser.add_argument("--organism", type=str, default="HOMO_SAPIENS", help="Organism enum string")
    parser.add_argument("--max_api_workers", type=int, default=8,
                        help="Parallel workers for API query batch scoring")
    parser.add_argument("--max_agg_workers", type=int, default=8,
                        help="Parallel workers for aggregation")
    parser.add_argument("--modalities", nargs="+", choices=VALID_MODALITIES, default=[],
                        help="Modalities to score")
    parser.add_argument("--batch_size", type=int, default=10_000,
                        help="Number of variants per prediction batch (default: 10000)")
    # Aggregation args
    parser.add_argument("--config", type=str, required=True,
                        help="Track aggregations YAML ({group: [ontology CURIE, ...]}; empty list = all tracks)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for the output CSV files")
    return parser


def checkpoint_path(output_dir, base_name):
    """Path of the checkpoint file of a run."""
    return os.path.join(output_dir, f".{base_name}_checkpoint.json")


def load_checkpoint(ckpt_path):
    """Return set of completed batch indices from a checkpoint file."""
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r") as f:
            data = json.load(f)
        completed = set(data.get("completed_batches", []))
        print(f"  Checkpoint found — {len(completed)} batch(es) already done, skipping them.")
        return completed
    return set()


def save_checkpoint(ckpt_path, completed_batches):
    """Writes the indices of the completed batches."""
    with open(ckpt_path, "w") as f:
        json.dump({"completed_batches": sorted(completed_batches)}, f)


def append_to_csv(df, path, write_header):
    """Append a DataFrame to a CSV, writing the header only on the first batch."""
    df.to_csv(path, mode="a", header=write_header, index=False)


def main(args: argparse.Namespace) -> None:
    """Scores the variants batch by batch and writes the aggregated tables.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Raises:
        ValueError: If no scorer is supported for the chosen modalities and organism.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input_file))[0]
    ckpt_path = checkpoint_path(args.output_dir, base_name)

    # Load YAML config
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    print(f"Loaded {len(config_dict)} aggregation config(s): {list(config_dict.keys())}")

    # Build output paths once; we need them for header tracking
    out_paths = {
        list_name: os.path.join(args.output_dir, f"{base_name}_{list_name}_aggr.csv")
        for list_name in config_dict
    }

    # Initialise AlphaGenome model
    print("Initialising AlphaGenome client ...")
    model = dna_client.create(get_alphagenome_api_key())
    organism = getattr(dna_client.Organism, args.organism)

    scorer_selections = {mod: (mod in args.modalities) for mod in VALID_MODALITIES}
    all_scorers = variant_scorers.RECOMMENDED_VARIANT_SCORERS
    selected_scorers = [
        all_scorers[k] for k in all_scorers if scorer_selections.get(k.lower(), False)
    ]
    selected_scorers = [
        s for s in selected_scorers
        if (organism.value in variant_scorers.SUPPORTED_ORGANISMS[s.base_variant_scorer])
        and not (
            s.requested_output == dna_client.OutputType.PROCAP
            and organism == dna_client.Organism.MUS_MUSCULUS
        )
    ]
    if not selected_scorers:
        raise ValueError("No valid scorers selected, check --modalities and --organism.")

    # Load variants
    print(f"Loading variants from {args.input_file}...")
    df = read_variant_table(args.input_file)
    if "variant_id" not in df.columns:
        df["variant_id"] = (
            df["CHROM"].astype(str) + "-" + df["POS"].astype(str) + "-"
            + df["REF"].astype(str) + "-" + df["ALT"].astype(str)
        )
    print(f"  {len(df):,} variants loaded.")

    sequence_length_const = sequence_length_from_label(args.interval_size)

    # Build Variant + Interval objects for every row
    all_variants = []
    all_intervals = []
    for _, row in df.iterrows():
        v = genome.Variant(
            chromosome=str(row["CHROM"]),
            position=int(row["POS"]),
            reference_bases=str(row["REF"]),
            alternate_bases=str(row["ALT"]),
            name=str(row["variant_id"]),
        )
        all_variants.append(v)
        all_intervals.append(v.reference_interval.resize(sequence_length_const))

    # Checkpoint to which batches are already done?
    completed_batches = load_checkpoint(ckpt_path)

    # Split into batches
    batch_size = args.batch_size
    total = len(all_variants)
    n_batches = (total + batch_size - 1) // batch_size
    print(f"  {n_batches} batch(es) of up to {batch_size:,} variants each.\n")

    # Output files already present (from a resumed run) already have their header
    headers_written = {list_name: os.path.exists(out_paths[list_name]) for list_name in config_dict}

    for batch_idx in range(n_batches):
        if batch_idx in completed_batches:
            print(f"[Batch {batch_idx + 1}/{n_batches}] Already completed, skipping.")
            continue

        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        print(f"[Batch {batch_idx + 1}/{n_batches}] Predicting variants {start}–{end - 1} ...")
        batch_variants = all_variants[start:end]
        batch_intervals = all_intervals[start:end]

        batch_results = model.score_variants(
            intervals=batch_intervals,
            variants=batch_variants,
            variant_scorers=selected_scorers,
            organism=organism,
            progress_bar=True,
            max_workers=args.max_api_workers,
        )

        print(f"[Batch {batch_idx + 1}/{n_batches}] Tidying scores...")
        vep_batch = tidy_variant_scores(batch_results, batch_variants)

        # Aggregate and append for every config entry
        for list_name, curie_list in config_dict.items():
            curie_count = len(curie_list) if curie_list else "ALL"
            print(f"[Batch {batch_idx + 1}/{n_batches}] Aggregating '{list_name}' ({curie_count} tracks)...")
            df_aggr = aggregate_scores_by_curies(vep_batch, curies=curie_list, max_workers=args.max_agg_workers)
            if df_aggr is not None and not df_aggr.empty:
                append_to_csv(df_aggr, out_paths[list_name], write_header=not headers_written[list_name])
                headers_written[list_name] = True
                print(f"  -> Appended to {out_paths[list_name]}")
            else:
                print(f"  -> Warning: no matches for '{list_name}' in this batch.")

        completed_batches.add(batch_idx)
        save_checkpoint(ckpt_path, completed_batches)
        print(f"[Batch {batch_idx + 1}/{n_batches}] Done. Checkpoint saved.\n")

    print("=" * 60)
    print("All batches completed successfully.")
    print("Output files:")
    for list_name, path in out_paths.items():
        if os.path.exists(path):
            size_gb = os.path.getsize(path) / (1024 ** 3)
            print(f"  {list_name:30s}  {path}  ({size_gb:.2f} GB)")
    print("=" * 60)

    # Clean up checkpoint file on clean completion
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print("Checkpoint file removed.")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
