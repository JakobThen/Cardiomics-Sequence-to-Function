"""Tests for cardiomics_s2f.vep.scoring."""

import numpy as np
import pytest

from cardiomics_s2f.vep.scoring import center_window_half_width, score_center_window, score_rna_exons


@pytest.mark.parametrize("head, expected", [
    ("multiome_ATAC", 250),
    ("CutnTag", 1000),
    ("cut&tag_H3K27ac", 1000),
    ("other_head", 250),
])
def test_center_window_half_width(head, expected):
    assert center_window_half_width(head) == expected


def test_center_window_sums_only_the_window():
    p_ref = np.zeros((20, 2))
    p_alt = np.zeros((20, 2))
    p_alt[8:13, 0] = 1.0   # inside window (center 10, half width 2 -> 8..12)
    p_alt[0, 1] = 100.0     # outside window, must be ignored
    scores = score_center_window(p_ref, p_alt, variant_center=10, half_width=2)
    np.testing.assert_allclose(scores, [np.log2(6), 0.0])


def test_center_window_is_clipped_at_sequence_edges():
    p_ref = np.ones((10, 1))
    p_alt = np.full((10, 1), 3.0)
    scores = score_center_window(p_ref, p_alt, variant_center=1, half_width=5)  # window 0..6
    np.testing.assert_allclose(scores, [np.log2(7 * 3 + 1) - np.log2(7 + 1)])


def test_rna_exon_scores_use_mean_over_exons():
    p_ref = np.ones((6, 2))
    p_alt = np.ones((6, 2))
    p_alt[:3, 1] = 2.0
    gene_mask = np.zeros((6, 2), dtype=bool)
    gene_mask[:3, 0] = True  # gene 0 covers positions 0..2
    scores, has_exons = score_rna_exons(p_ref, p_alt, gene_mask)

    np.testing.assert_array_equal(has_exons, [True, False])
    np.testing.assert_allclose(scores[0], [0.0, np.log(2.001) - np.log(1.001)])
    assert np.isnan(scores[1]).all()


def test_rna_exon_scores_without_genes():
    scores, has_exons = score_rna_exons(np.ones((4, 3)), np.ones((4, 3)), np.zeros((4, 0), dtype=bool))
    assert scores.shape == (0, 3) and has_exons.shape == (0,)
