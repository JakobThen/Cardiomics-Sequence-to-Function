"""In silico mutagenesis (ISM) with the pretrained AlphaGenome model (workflow step 05).

Scores every possible single-nucleotide change in each region of a BED file with
the AlphaGenome API (ATAC, and optionally ChIP-TF for given transcription factors
and RNA-seq for given genes), averages the tracks of the given ontology CURIEs
and saves one ISM matrix per region and modality. The logic lives in
`cardiomics_s2f.ism`.

Usage (normally via 05_ism.sh):
    python workflows/alphagenome_api/run_ism.py --bed BED --curies UBERON:0002078 UBERON:0002079
        --seq_len 1MB --out_prefix DIR/AG_pretrained_ISM [--tfs CTCF GATA4 | --tfs tfs.txt] [--genes genes.csv]

Output:
    <out_prefix>_<bed name>.h5
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from alphagenome.data import genome
from alphagenome.models import dna_client, variant_scorers
from joblib import Parallel, delayed

from cardiomics_s2f.config import get_alphagenome_api_key
from cardiomics_s2f.ism import collect_region_ism_matrices, save_ism_dict
from cardiomics_s2f.vep.alphagenome_api import sequence_length_from_label


def get_parser() -> argparse.ArgumentParser:
    """Creates the argument parser (separate so Sphinx can document it).

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Run AlphaGenome ISM pipeline for targeted regions.")
    parser.add_argument("--bed", required=True, type=str,
                        help="BED file without header. If the 4th (name) column is missing, regions are numbered.")
    parser.add_argument("--curies", required=True, nargs='+', type=str,
                        help="Ontology CURIEs of the tracks to average (e.g. EFO:0002067).")
    parser.add_argument("--seq_len", required=True, type=str, choices=["16KB", "100KB", "500KB", "1MB"],
                        help="Sequence length context for the model.")
    parser.add_argument("--out_prefix", required=True, type=str,
                        help="Output path prefix; the BED name and .h5 are appended.")
    parser.add_argument("--tfs", nargs='+', type=str, default=None,
                        help="TF names, or a single .txt file with one TF per line.")
    parser.add_argument("--genes", type=str, default=None,
                        help="CSV with target genes, one row per BED region (several columns allowed, no header).")
    return parser


def main(args: argparse.Namespace) -> None:
    """Runs ISM for all regions and writes the HDF5 file.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    model = dna_client.create(get_alphagenome_api_key())

    # Setup Output File
    bed_basename = os.path.basename(args.bed).replace(".bed", "")
    out_file = f"{args.out_prefix}_{bed_basename}.h5"
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {out_file}")

    # Parse BED File
    bed_raw = pd.read_csv(args.bed, sep="\t", header=None)
    if bed_raw.shape[1] >= 4:
        enhancers = bed_raw.iloc[:, :4]
        enhancers.columns = ["chrom", "start", "end", "name"]
    else:
        enhancers = bed_raw.iloc[:, :3]
        enhancers.columns = ["chrom", "start", "end"]
        enhancers["name"] = [f"region_{i}" for i in range(len(enhancers))]
    print(f"Loaded {len(enhancers)} regions from {args.bed}")

    # Parse Genes CSV (if provided)
    if args.genes:
        genes_df = pd.read_csv(args.genes, header=None)
        if len(genes_df) != len(enhancers):
            sys.exit(f"Error: Target genes file has {len(genes_df)} rows, but BED file has {len(enhancers)} rows.")
        # Collect non-NaN values across columns into a list per row
        enhancers["target_gene"] = [row.dropna().astype(str).tolist() for _, row in genes_df.iterrows()]
    else:
        enhancers["target_gene"] = None

    # Parse TFs (if provided)
    tfs = []
    if args.tfs:
        if len(args.tfs) == 1 and os.path.isfile(args.tfs[0]):
            with open(args.tfs[0], 'r') as f:
                tfs = [line.strip() for line in f if line.strip()]
        else:
            tfs = args.tfs

    # Filter TFs against available metadata
    track_metadata = model.output_metadata(dna_client.Organism.HOMO_SAPIENS).concatenate()
    metadata = track_metadata[
        (track_metadata["ontology_curie"].isin(args.curies)) &
        (track_metadata["output_type"].isin([
            dna_client.OutputType.ATAC,
            dna_client.OutputType.CHIP_TF,
            dna_client.OutputType.RNA_SEQ
        ]))
    ].dropna(how="all", axis=1)

    if tfs:
        valid_tfs = metadata.get("transcription_factor", pd.Series(dtype=str)).str.upper().tolist()
        tfs = [tf.upper() for tf in tfs if tf.upper() in valid_tfs]
        print(f"Filtered to {len(tfs)} valid transcription factors for given curies.")
        if len(tfs) == 0:
            tfs = None

    # Dynamically build Scorers and Modality Indices
    scorers = [variant_scorers.RECOMMENDED_VARIANT_SCORERS['ATAC']]
    modality_indices = {"ATAC": 0}
    current_idx = 1
    if tfs:
        scorers.append(variant_scorers.RECOMMENDED_VARIANT_SCORERS['CHIP_TF'])
        modality_indices["TF"] = current_idx
        current_idx += 1
    if args.genes:
        scorers.append(variant_scorers.RECOMMENDED_VARIANT_SCORERS['RNA_SEQ'])
        modality_indices["RNA"] = current_idx

    # Generate model intervals
    seq_len = sequence_length_from_label(args.seq_len)
    enhancers["ivs"] = [
        genome.Interval(chromosome=row.chrom, start=row.start, end=row.end)
        for row in enhancers.itertuples()
    ]
    enhancers["ivs_big"] = [iv.resize(seq_len) for iv in enhancers["ivs"]]

    # Run AlphaGenome ISM
    print("Running Model ISM...")
    ism_result = {
        row.name: model.score_ism_variants(
            interval=row.ivs_big,
            ism_interval=row.ivs,
            variant_scorers=scorers,
            interval_variant=None,
        ) for row in enhancers.itertuples()
    }

    # Process results in parallel
    print("Processing ISM scores...")
    results = Parallel(n_jobs=-3, backend="loky", verbose=10)(
        delayed(collect_region_ism_matrices)(
            name=row.name,
            ism_data=ism_result[row.name],
            curie=args.curies,
            indices=modality_indices,
            genes=row.target_gene,
            TFs=tfs
        )
        for row in enhancers.itertuples()
    )

    # Save output
    print("Saving HDF5 file...")
    save_ism_dict(dict(results), out_file)
    print("Done!")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
