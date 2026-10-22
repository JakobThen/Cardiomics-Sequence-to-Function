"""Matches pretrained AlphaGenome RNA-seq tracks to the multiome RNA pseudobulks (workflow step 01, RNA).

Selects expressed, variable genes, predicts all AlphaGenome RNA-seq tracks in a
TSS-anchored window per gene (strand-matched tracks), sums the predicted signal
over each gene's exons and correlates predicted and observed exon counts across
genes for every track pair. The logic lives in `cardiomics_s2f.eval.alphagenome_api`.

Usage (normally via 01_track_matching.sh):
    python workflows/alphagenome_api/match_tracks_rna.py --counts_csv CSV --pct_csv CSV
        --gtf GTF --chrom_sizes SIZES --out_dir DIR

Output:
    <out_dir>/AG_Multiome_RNA_feature_{sum,mean}_exon_counts_corr_<seq_len>.h5ad
"""
import argparse
from pathlib import Path

import pandas as pd
from alphagenome.models import dna_client
from tqdm import tqdm

from cardiomics_s2f.config import get_alphagenome_api_key
from cardiomics_s2f.eval.alphagenome_api import (
    assemble_rna_track_match,
    attach_gene_windows,
    filter_multiome_genes,
    gene_exon_annotation,
    gene_prediction_windows,
    sum_exon_signal,
)
from cardiomics_s2f.vep.alphagenome_api import sequence_length_from_label


def get_parser() -> argparse.ArgumentParser:
    """Creates the argument parser (separate so Sphinx can document it).

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Match pretrained AlphaGenome RNA-seq tracks to multiome RNA")
    parser.add_argument("--counts_csv", type=str, required=True,
                        help="Pseudobulk counts per gene (genes x cell types, gene names as index)")
    parser.add_argument("--pct_csv", type=str, required=True,
                        help="Percentage of expressing cells per gene and pseudobulk, same layout")
    parser.add_argument("--gtf", type=str, required=True, help="GENCODE GTF (gene and exon features)")
    parser.add_argument("--chrom_sizes", type=str, required=True, help="genome.chrom.sizes")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--seq_len", type=str, choices=["1MB", "500KB", "100KB", "16KB"], default="500KB",
                        help="Prediction window per gene (default: 500KB)")
    parser.add_argument("--n_top_var_genes", type=int, default=5000,
                        help="Number of most variable genes to match (default: 5000)")
    parser.add_argument("--exon_length_norm", action="store_true",
                        help="Use the mean instead of the sum over exons")
    return parser


def main(args: argparse.Namespace) -> None:
    """Predicts, sums and correlates the RNA tracks and writes the AnnData.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = dna_client.create(get_alphagenome_api_key())
    seq_len = sequence_length_from_label(args.seq_len)

    # 1. Genes and their prediction windows
    print("Loading RNA feature counts and building AnnData...")
    adata = filter_multiome_genes(
        pd.read_csv(args.counts_csv, index_col=0), pd.read_csv(args.pct_csv, index_col=0),
        n_top_var_genes=args.n_top_var_genes,
    )
    gtf = gene_prediction_windows(gene_exon_annotation(args.gtf), seq_len, args.chrom_sizes)
    adata = attach_gene_windows(adata, gtf)
    print(f"Ready for prediction! Using {len(adata.obs)} matched genes/intervals.")

    # 2. Prediction (tracks on the gene's strand plus unstranded tracks)
    metadata = model.output_metadata(organism=dna_client.Organism.HOMO_SAPIENS).rna_seq
    metadata['strand_type'] = metadata['strand'].replace({'+': 'stranded', '-': 'stranded', '.': 'unstranded'})

    track_names_plus_mask = metadata.strand.isin(["+", "."])
    track_names_minus_mask = metadata.strand.isin(["-", "."])

    # Create MultiIndex arrays for labels (biosample_name, strand_type)
    labels_plus = pd.MultiIndex.from_frame(metadata[track_names_plus_mask][["name", "strand_type"]])
    labels_minus = pd.MultiIndex.from_frame(metadata[track_names_minus_mask][["name", "strand_type"]])

    pred_counts = {}
    print("Starting AlphaGenome inference loop...")
    for gene, row in tqdm(adata.obs.iterrows(), total=len(adata.obs), desc="AG Prediction"):
        strand = row["adj_input_interval"].strand
        tracks_mask = track_names_plus_mask if strand == "+" else track_names_minus_mask
        track_labels = labels_plus if strand == "+" else labels_minus

        output = model.predict_interval(
            interval=row["adj_input_interval"],
            requested_outputs=[dna_client.OutputType.RNA_SEQ],
            ontology_terms=None
        )
        tdata = output.rna_seq
        pos_counts = tdata.values[:, tracks_mask.to_numpy()]
        exon_signal = sum_exon_signal(pos_counts, tdata.interval.start, row["exon_intervals"],
                                      exon_length_norm=args.exon_length_norm)
        pred_counts[gene] = pd.Series(exon_signal, index=track_labels)

    # 3. Combine with the observed counts and correlate
    print("Inference complete. Assembling final AnnData...")
    combined_adata = assemble_rna_track_match(pred_counts, metadata, adata)

    kind = "mean" if args.exon_length_norm else "sum"
    out_file = out_dir / f"AG_Multiome_RNA_feature_{kind}_exon_counts_corr_{args.seq_len}.h5ad"
    print(f"Writing output file to disk: {out_file}")
    combined_adata.write(out_file)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
