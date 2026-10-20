"""Tests for cardiomics_s2f.preprocessing.bigwig_stats (pure numpy part)."""

import math

import numpy as np

from cardiomics_s2f.preprocessing.bigwig_stats import CANONICAL_CHROMS, nonzero_mean_of_arrays


def test_nonzero_mean_ignores_zeros_and_nans():
    arrays = [np.array([0.0, 1.0, np.nan, 3.0]), np.array([0.0, 0.0, 2.0])]
    assert nonzero_mean_of_arrays(arrays) == 2.0


def test_nonzero_mean_without_signal_is_nan():
    assert math.isnan(nonzero_mean_of_arrays([np.zeros(5), np.array([np.nan])]))


def test_canonical_chromosomes():
    assert {"chr1", "chr22", "chrX", "chrY"} <= CANONICAL_CHROMS
    assert "chrM" not in CANONICAL_CHROMS and "chr1_KI270706v1_random" not in CANONICAL_CHROMS
