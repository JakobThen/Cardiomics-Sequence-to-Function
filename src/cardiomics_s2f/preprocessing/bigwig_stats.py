"""
BigWig Track Statistics
=======================

Summary statistics of BigWig tracks used to prepare AlphaGenome fine-tuning,
in particular the mean of all non-zero values on canonical chromosomes, which
``alphagenome_ft`` uses to scale each track.
"""

from typing import Iterable

import numpy as np

# Restrict analysis strictly to canonical chromosomes to prevent skewed means
# from alt-contigs or unplaced scaffolds
CANONICAL_CHROMS = frozenset([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"])


def nonzero_mean_of_arrays(arrays: Iterable[np.ndarray]) -> float:
    """Mean of all values that are neither zero nor NaN across several arrays.

    Args:
        arrays (iterable of np.ndarray): Signal arrays, e.g. one per chromosome.

    Returns:
        float: The non-zero mean, or ``nan`` if there are no non-zero values.
    """
    total_sum = 0.0
    total_count = 0
    for vals in arrays:
        # Filter out zeroes and NaN values (unmapped regions)
        nz = vals[(vals != 0) & (~np.isnan(vals))]
        total_sum += nz.sum()
        total_count += len(nz)
    return total_sum / total_count if total_count > 0 else float("nan")


def bigwig_nonzero_mean(bw_path, chroms=CANONICAL_CHROMS) -> float:
    """Mean of the non-zero values of a BigWig file, restricted to ``chroms``.

    Args:
        bw_path (str or Path): Path to the BigWig file.
        chroms (set of str, optional): Chromosomes to include. Defaults to
            chr1-22, chrX and chrY.

    Returns:
        float: The non-zero mean, or ``nan`` if the selected chromosomes hold no
        non-zero values.
    """
    import pyBigWig

    bw = pyBigWig.open(str(bw_path))
    try:
        arrays = (
            bw.values(chrom, 0, length, numpy=True)
            for chrom, length in bw.chroms().items()
            if chrom in chroms
        )
        return nonzero_mean_of_arrays(arrays)
    finally:
        bw.close()
