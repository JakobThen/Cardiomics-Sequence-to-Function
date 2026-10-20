"""
Fine-tuned AlphaGenome VEP Helpers
==================================

Sequence preparation for variant effect prediction (VEP) with a fine-tuned
AlphaGenome model: chromosome sizes and batched, one-hot encoded REF/ALT windows
centered on each variant (with AlphaGenome's indel handling).
"""

import numpy as np
import pandas as pd

from alphagenome.data import genome
from alphagenome_research.io import fasta as fasta_lib
from alphagenome_research.io import genome as genome_io
from alphagenome_research.model import one_hot_encoder
from alphagenome_research.model.variant_scoring.variant_scoring import IndelMask


def load_chrom_sizes(chrom_sizes_path):
    """Reads a two-column ``chrom<TAB>size`` file into a dictionary.

    Args:
        chrom_sizes_path (str or Path): Path to the ``.chrom.sizes`` file.

    Returns:
        dict: Mapping of chromosome name to length in bp.
    """
    chrom_size_df = pd.read_csv(chrom_sizes_path, sep="\t", names=["chrom", "end"])
    return dict(zip(chrom_size_df['chrom'], chrom_size_df['end']))


def variant_batch_generator(df, fasta_path, batch_size, window_size, organism="HOMO_SAPIENS", chrom_sizes=None):
    """Yields batches of encoded reference and alternate sequences for variant effect scoring.

    Extracts genomic sequences centered on variants, applies indel shifting and padding
    if necessary, and one-hot encodes the sequences. Windows that would cross a
    chromosome end are shifted inwards so that their size stays constant. The last
    batch is padded with copies of its first variant (ID ``"__PAD__"``) so every batch
    has ``batch_size`` entries, as required by the device mesh.

    Args:
        df (pandas.DataFrame): DataFrame containing variant information (CHROM, POS, REF, ALT, variant_id).
        fasta_path (str or Path): Path to the reference genome FASTA file.
        batch_size (int): Number of variants to yield in each batch.
        window_size (int): Size of the genomic window centered on the variant.
        organism (str, optional): Target organism name. Defaults to "HOMO_SAPIENS".
        chrom_sizes (dict, optional): Chromosome lengths (see `load_chrom_sizes`) used to
            keep windows inside chromosomes. Chromosomes not listed are treated as unbounded.

    Yields:
        tuple:
            - np.ndarray: One-hot encoded reference sequences.
            - np.ndarray: One-hot encoded alternate sequences.
            - list of IndelMask: Indel alignment masks.
            - list of Variant: AlphaGenome variant objects.
            - list of Interval: Target interval coordinates.
            - list of str: Variant identifiers.
            - int: The actual size of the batch (excluding padding).
    """
    chrom_sizes = chrom_sizes or {}
    extractor = fasta_lib.FastaExtractor(str(fasta_path))
    encoder = one_hot_encoder.DNAOneHotEncoder(dtype=np.float32)

    half_window = window_size // 2

    batch_ref, batch_alt, batch_masks, batch_ids = [], [], [], []
    batch_variants, batch_intervals = [], []

    for row in df.itertuples(index=False):
        chrom = str(row.CHROM)
        pos = int(row.POS)

        # Create var objects
        variant = genome.Variant(
            chromosome=chrom,
            position=pos,
            reference_bases=str(row.REF).upper(),
            alternate_bases=str(row.ALT).upper()
        )

        # 0-based interval centered on the variant
        ideal_start = pos - 1 - half_window

        # BOUNDARY LOGIC: Shift window to stay within chromosome while keeping size constant
        max_chrom_len = chrom_sizes.get(chrom, float('inf'))
        if ideal_start < 0:
            start = 0
        elif ideal_start + window_size > max_chrom_len:
            start = max(0, max_chrom_len - window_size)
        else:
            start = ideal_start

        interval = genome.Interval(start=start, end=start + window_size, chromosome=chrom)

        # AG Sequence Extraction (handles all indel shifting & padding)
        ref_seq_string, alt_seq_string = genome_io.extract_variant_sequences(
            interval, variant, extractor
        )

        # AG Indel Masks (for post-hoc alignment)
        indel_mask = IndelMask.from_variant(variant, interval)

        # Encode
        ref_onehot = encoder.encode(ref_seq_string)
        alt_onehot = encoder.encode(alt_seq_string)

        # Store
        batch_ref.append(ref_onehot)
        batch_alt.append(alt_onehot)
        batch_masks.append(indel_mask)
        batch_ids.append(row.variant_id)
        batch_variants.append(variant)
        batch_intervals.append(interval)

        if len(batch_ref) == batch_size:
            yield np.stack(batch_ref), np.stack(batch_alt), batch_masks, batch_variants, batch_intervals, batch_ids, batch_size
            batch_ref, batch_alt, batch_masks, batch_variants, batch_intervals, batch_ids = [], [], [], [], [], []

    # Pad the last batch to batch_size so it does not crash the device mesh
    if len(batch_ref) > 0:
        actual_size = len(batch_ref)
        while len(batch_ref) < batch_size:
            batch_ref.append(batch_ref[0])
            batch_alt.append(batch_alt[0])
            batch_masks.append(batch_masks[0])
            batch_variants.append(batch_variants[0])
            batch_intervals.append(batch_intervals[0])
            batch_ids.append("__PAD__")
        yield np.stack(batch_ref), np.stack(batch_alt), batch_masks, batch_variants, batch_intervals, batch_ids, actual_size
