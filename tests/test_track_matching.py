"""Tests for cardiomics_s2f.eval.track_matching (selection helpers)."""

import pandas as pd
import pytest

pytest.importorskip("adjustText")
pytest.importorskip("matplotlib_venn")

from cardiomics_s2f.eval.track_matching import (  # noqa: E402
    extreme_correlation_pairs,
    top_matching_tracks_per_target,
)


@pytest.fixture
def corr():
    # targets (rows) x candidate tracks (columns)
    return pd.DataFrame(
        {"heart": [0.9, 0.1], "liver": [0.2, 0.8], "lung": [0.5, 0.3]},
        index=["Cardiomyocytes", "Hepatocytes"],
    )


def test_top_matching_tracks_per_target(corr):
    top = top_matching_tracks_per_target(corr, top_n=2)
    assert top["Cardiomyocytes"] == [("heart", 0.9), ("lung", 0.5)]
    assert top["Hepatocytes"] == [("liver", 0.8), ("lung", 0.3)]


def test_extreme_correlation_pairs(corr):
    pairs = extreme_correlation_pairs(corr)
    # two highest (ascending), then the two lowest
    assert [p["Correlation"] for p in pairs] == [0.8, 0.9, 0.1, 0.2]
    assert pairs[1] == {"Target_Track": "heart", "AG_Track": "Cardiomyocytes", "Correlation": 0.9}
