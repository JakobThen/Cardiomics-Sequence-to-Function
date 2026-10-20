"""
Variant Effect Scoring
======================

Per-variant scores computed from aligned REF/ALT predictions of one output head
(arrays of shape ``(sequence_length, num_tracks)``):

* **ATAC / Cut&Tag**: signal summed in a window centred on the variant,
  score = ``log2(alt_sum + 1) - log2(ref_sum + 1)``.
* **RNA**: mean signal over each gene's exons,
  score = ``ln(alt_mean + 0.001) - ln(ref_mean + 0.001)``.
"""

import numpy as np

CUTNTAG_TAGS = ("CUTNTAG", "CNT", "CUT&TAG", "C&T")


def center_window_half_width(head_name: str) -> int:
    """Half width of the scoring window for a variant-centred head.

    Args:
        head_name (str): Output head identifier.

    Returns:
        int: 1000 for Cut&Tag heads (2001 bp window), otherwise 250 (501 bp,
        used for ATAC and as the default).
    """
    head_upper = head_name.upper()
    if "ATAC" in head_upper:
        return 250
    if any(tag in head_upper for tag in CUTNTAG_TAGS):
        return 1000
    return 250


def score_center_window(p_ref: np.ndarray, p_alt: np.ndarray, variant_center: int, half_width: int) -> np.ndarray:
    """Scores a variant by the log2 fold change of signal summed around it.

    Args:
        p_ref (np.ndarray): REF predictions, shape ``(seq_len, num_tracks)``.
        p_alt (np.ndarray): ALT predictions aligned to REF, same shape.
        variant_center (int): 0-based variant position within the window.
        half_width (int): Half width of the summed window (clipped to the sequence).

    Returns:
        np.ndarray: One score per track, ``log2(alt_sum + 1) - log2(ref_sum + 1)``.
    """
    seq_len = p_ref.shape[0]
    start_idx = max(0, variant_center - half_width)
    end_idx = min(seq_len, variant_center + half_width + 1)

    spatial_mask = np.zeros(seq_len, dtype=bool)
    spatial_mask[start_idx:end_idx] = True

    ref_sum = np.sum(p_ref[spatial_mask, :], axis=0)
    alt_sum = np.sum(p_alt[spatial_mask, :], axis=0)
    return np.log2(alt_sum + 1) - np.log2(ref_sum + 1)


def score_rna_exons(p_ref: np.ndarray, p_alt: np.ndarray, gene_mask: np.ndarray, pseudocount: float = 0.001):
    """Scores a variant per gene by the log fold change of mean exon signal.

    Args:
        p_ref (np.ndarray): REF predictions, shape ``(seq_len, num_tracks)``.
        p_alt (np.ndarray): ALT predictions aligned to REF, same shape.
        gene_mask (np.ndarray): Boolean exon mask, shape ``(seq_len, num_genes)``.
        pseudocount (float, optional): Added before taking the log. Defaults to 0.001.

    Returns:
        tuple:
            - np.ndarray: Scores of shape ``(num_genes, num_tracks)``; rows of genes
              without exons in the window are NaN.
            - np.ndarray: Boolean array ``(num_genes,)``, True where a gene has exons
              in the window.
    """
    num_genes = gene_mask.shape[1]
    mask_sums = gene_mask.sum(axis=0)
    has_exons = mask_sums > 0
    scores = np.full((num_genes, p_ref.shape[1]), np.nan)

    for g in np.flatnonzero(has_exons):
        g_mask = gene_mask[:, g]
        ref_mean = np.sum(p_ref[g_mask, :], axis=0) / mask_sums[g]
        alt_mean = np.sum(p_alt[g_mask, :], axis=0) / mask_sums[g]
        scores[g] = np.log(alt_mean + pseudocount) - np.log(ref_mean + pseudocount)

    return scores, has_exons
