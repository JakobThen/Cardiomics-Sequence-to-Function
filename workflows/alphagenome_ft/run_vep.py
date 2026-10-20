"""
Variant Effect Prediction (VEP) Script (workflow step 04)

This script loads a fine-tuned AlphaGenome model and runs batch Variant Effect
Prediction (VEP) on variants from an input VCF-style CSV. It uses center mask
scoring for CutnTag and ATAC-seq tracks, and exon mask scoring for RNA-seq
tracks (see `cardiomics_s2f.vep.scoring`).

Output: a long CSV ``VEP_AG_ft_<model_version>_<vcf name>.csv`` in ``--out_dir``
with one row per variant, track (and gene, for RNA heads). It is written in
chunks to bound memory and overwritten if it already exists.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from alphagenome_ft import load_checkpoint
from alphagenome_ft.finetune.config import load_targets_config, prepare_head_specs
from alphagenome_ft.finetune.train import register_predefined_heads
from alphagenome.models import dna_model as ag_dna_model
from alphagenome_research.model.variant_scoring.variant_scoring import align_alternate
from alphagenome_research.model.variant_scoring.gene_mask_extractor import (
    GeneMaskExtractor, GeneMaskType, GeneQueryType
)

from cardiomics_s2f.finetuning.alphagenome.inference import make_parallel_predict, setup_data_parallel_mesh
from cardiomics_s2f.vep.alphagenome_ft import load_chrom_sizes, variant_batch_generator
from cardiomics_s2f.vep.scoring import center_window_half_width, score_center_window, score_rna_exons

DEFAULT_GTF_URL = "https://storage.googleapis.com/alphagenome/reference/gencode/hg38/gencode.v46.annotation.gtf.gz.feather"
REQUIRED_COLUMNS = ["variant_id", "CHROM", "POS", "REF", "ALT"]
CHUNK_BATCHES = 25  # write the CSV every N batches to save memory


def get_parser() -> argparse.ArgumentParser:
    """Creates and configures the argument parser for the VEP script.

    Returns:
        argparse.ArgumentParser: The configured argument parser object.
    """
    parser = argparse.ArgumentParser(description="Fine-tuned AlphaGenome Variant Effect Prediction")

    # Core I/O Arguments
    parser.add_argument("--input_vcf", type=str, required=True,
                        help="CSV with columns variant_id, CHROM, POS, REF, ALT (other columns are ignored)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for the VEP CSV")
    parser.add_argument("--fasta_path", type=str, required=True, help="Path to the genome.fa file")
    parser.add_argument("--chrom_sizes", type=str, required=True, help="Path to the genome.chrom.sizes file")
    parser.add_argument("--config", type=str, required=True, help="Path to the target YAML config file")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Fine-tuning checkpoint directory (its 'best' checkpoint is used)")
    parser.add_argument("--base_dir", type=str, default=None,
                        help="Base directory for resolving relative paths in the config (defaults to the config's folder)")
    parser.add_argument("--gtf_feather", type=str, default=DEFAULT_GTF_URL,
                        help="GENCODE annotation (.feather path or URL) used for RNA exon masks")

    # Data & Model Options
    parser.add_argument("--window_size", type=int, default=1_048_576, help="Window size in bp")
    parser.add_argument("--organism", type=str, default="HOMO_SAPIENS", help="Target organism")
    parser.add_argument("--model_version", type=str, default="fold_1", help="Base model version (e.g., fold_1, all_folds)")
    parser.add_argument("--batch_size", type=int, default=24, help="Global batch size (must be divisible by N_GPUs)")

    return parser


def load_variants(path) -> pd.DataFrame:
    """Loads the variant table and keeps unique variants.

    Args:
        path (str or Path): CSV with at least ``variant_id, CHROM, POS, REF, ALT``.

    Returns:
        pd.DataFrame: Unique variants with exactly the required columns.

    Raises:
        ValueError: If a required column is missing.
    """
    variants_df = pd.read_csv(path)
    missing_cols = set(REQUIRED_COLUMNS) - set(variants_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return variants_df[REQUIRED_COLUMNS].drop_duplicates()  # remove redundant celltype info etc


def track_names(head_metadata: pd.DataFrame, num_tracks: int) -> list:
    """Track labels from the config, falling back to ``Track_<i>``."""
    if head_metadata.empty:
        return [f"Track_{trk}" for trk in range(num_tracks)]
    return [head_metadata.iloc[trk].get('label', f"Track_{trk}") for trk in range(num_tracks)]


def main(args: argparse.Namespace) -> None:
    """Runs variant effect prediction and writes the long-format score table.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    input_vcf = Path(args.input_vcf)
    out_dir = Path(args.out_dir)
    targets_config_path = Path(args.config)
    checkpoint_dir = Path(args.checkpoint_dir)
    base_dir = Path(args.base_dir) if args.base_dir else targets_config_path.parent
    organism_idx = 0 if args.organism == "HOMO_SAPIENS" else 1

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"VEP_AG_ft_{args.model_version}_{input_vcf.stem}.csv"
    if csv_path.exists():
        print(f"Overwriting existing output: {csv_path}", flush=True)
        csv_path.unlink()

    # 1. Load variants and chromosome sizes
    variants_df = load_variants(input_vcf)
    print("VCF file loaded!", flush=True)
    chrom_sizes = load_chrom_sizes(args.chrom_sizes)

    # 2. Setup RNA Exon Extractor
    print(f"Loading GTF annotations from {args.gtf_feather}...", flush=True)
    gtf_df = pd.read_feather(args.gtf_feather)
    exon_extractor = GeneMaskExtractor(
        gtf=gtf_df,
        gene_mask_type=GeneMaskType.EXONS,
        gene_query_type=GeneQueryType.INTERVAL_CONTAINED,
        filter_protein_coding=False
    )

    # 3. Model & Mesh Setup
    config_dict = load_targets_config(targets_config_path, base_dir=base_dir)
    head_specs = prepare_head_specs(config_dict)
    register_predefined_heads(head_specs)

    model = load_checkpoint(checkpoint_dir / "best", base_model_version=args.model_version, init_seq_len=args.window_size)
    print("Loaded model onto GPU", flush=True)

    # Prepare Track Metadata from config_dict
    tracks_metadata = {}
    for head in config_dict.get('heads', []):
        tracks_metadata[head['id']] = pd.DataFrame(head.get('targets', []))

    organism_enum = getattr(ag_dna_model.Organism, args.organism)
    strand_reindexing = jax.device_put(model._metadata[organism_enum].strand_reindexing, model._device_context._device)

    mesh, data_sharding, replicated_sharding = setup_data_parallel_mesh()
    parallel_predict = make_parallel_predict(model, head_specs, data_sharding, replicated_sharding)

    batched_align_alternate = jax.jit(jax.vmap(align_alternate, in_axes=(0, 0))) #make align alternate function batch callable

    # 4. VEP Evaluation, Aggregation & Long DataFrame Generation
    # We will collect a list of dictionaries to build our long DataFrame
    vep_records = []
    first_write = True # Flag to track whether we need to write the CSV headers

    def flush_records():
        nonlocal first_write
        if vep_records:
            pd.DataFrame(vep_records).to_csv(csv_path, mode='a', header=first_write, index=False)
            first_write = False
            vep_records.clear()

    print("Starting Variant Effect Prediction and Aggregation...", flush=True)
    batches = variant_batch_generator(
        variants_df, args.fasta_path, args.batch_size, args.window_size,
        organism=args.organism, chrom_sizes=chrom_sizes
    )

    with model._device_context, jax.set_mesh(mesh):
        for batch_idx, (ref_seq, alt_seq, batch_masks, batch_variants, batch_intervals, var_ids, actual_size) in enumerate(batches):

            current_bsz = ref_seq.shape[0]

            org_idx = jnp.full((current_bsz,), organism_idx, dtype=jnp.int32)
            neg_mask = jnp.zeros((current_bsz,), dtype=jnp.bool_)

            org_idx_sharded = jax.device_put(org_idx, data_sharding)
            neg_mask_sharded = jax.device_put(neg_mask, data_sharding)

            ref_sharded = jax.device_put(ref_seq, data_sharding)
            preds_ref = parallel_predict(model._params, model._state, ref_sharded, org_idx_sharded, neg_mask_sharded, strand_reindexing)

            alt_sharded = jax.device_put(alt_seq, data_sharding)
            preds_alt = parallel_predict(model._params, model._state, alt_sharded, org_idx_sharded, neg_mask_sharded, strand_reindexing)

            # stack the masks for GPU processing
            batched_masks = jtu.tree_map(lambda *x: jnp.stack(x), *batch_masks)

            if batch_idx == 0:
                print(f"Batched_masks shape: {jax.tree_util.tree_map(lambda x: x.shape, batched_masks)}", flush = True)

            # ALIGN, MASK, AGGREGATE
            for head_spec in head_specs:
                head_name = head_spec.head_id

                p_ref_batch_jax = preds_ref[head_name]['predictions_1bp']
                p_alt_batch_jax = preds_alt[head_name]['predictions_1bp']

                # batched alignment on GPU
                p_alt_aligned_batch_jax = batched_align_alternate(p_alt_batch_jax, batched_masks)
                p_ref_batch = np.array(p_ref_batch_jax)
                p_alt_aligned_batch = np.array(p_alt_aligned_batch_jax)

                num_tracks = p_ref_batch.shape[2]
                trk_names = track_names(tracks_metadata.get(head_name, pd.DataFrame()), num_tracks)
                is_rna = "RNA" in head_name.upper()
                half_width = None if is_rna else center_window_half_width(head_name)

                # Iterate through actual batch size (skips padding)
                for i in range(actual_size):
                    p_ref_i = p_ref_batch[i]
                    p_alt_aligned_i = p_alt_aligned_batch[i] # Already aligned!
                    variant_id = var_ids[i]
                    interval = batch_intervals[i]
                    interval_str = f"{interval.chromosome}:{interval.start}-{interval.end}"

                    # RNA LOGIC (Exon Mask, Mean, Natural Log)
                    if is_rna:
                        gene_mask_2d, annotations = exon_extractor.extract(
                            interval=interval,
                            variant=batch_variants[i]
                        )

                        if batch_idx == 0:
                            assert gene_mask_2d.shape[0] == p_ref_i.shape[0], \
                                f"Mask/prediction length mismatch: {gene_mask_2d.shape[0]} vs {p_ref_i.shape[0]}"

                        if gene_mask_2d.shape[1] == 0:
                            continue # Skip entirely if no genes are found in this window

                        gene_scores, has_exons = score_rna_exons(p_ref_i, p_alt_aligned_i, gene_mask_2d)
                        for g in np.flatnonzero(has_exons): # skip genes with no exons or invalid sequence in window
                            g_id = annotations['gene_id'].iloc[g]
                            g_name = annotations['gene_name'].iloc[g]
                            for trk in range(num_tracks):
                                vep_records.append({
                                    'variant_id': variant_id,
                                    'scored_interval': interval_str,
                                    'gene_id': g_id,
                                    'gene_name': g_name,
                                    'output_type': head_name,
                                    'track_name': trk_names[trk],
                                    'score': gene_scores[g, trk]
                                })

                    # ATAC / CUT&TAG LOGIC (Variant-Centric, Sum, Log2)
                    else:
                        variant_center = batch_variants[i].position - 1 - interval.start
                        scores = score_center_window(p_ref_i, p_alt_aligned_i, variant_center, half_width)

                        for trk in range(num_tracks):
                            vep_records.append({
                                'variant_id': variant_id,
                                'scored_interval': interval_str,
                                'gene_id': None,
                                'gene_name': None,
                                'output_type': head_name,
                                'track_name': trk_names[trk],
                                'score': scores[trk]
                            })

            if (batch_idx + 1) % CHUNK_BATCHES == 0:
                flush_records()
                print(f"Processed {batch_idx + 1} batches...", flush=True)

    # write remaining batches
    flush_records()

    print("Saved all records to:")
    print(f" -> {csv_path}")
    print("Variant Effect Prediction complete!", flush=True)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
