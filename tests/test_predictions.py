"""Tests for cardiomics_s2f.finetuning.alphagenome.predictions."""

import numpy as np
import pandas as pd
import pytest

from cardiomics_s2f.finetuning.alphagenome.predictions import (
    get_aligned_nonzero_means,
    get_prediction_key,
)


def test_prediction_key_prefers_requested_resolution():
    raw = {"scaled_predictions_1bp": 0, "scaled_predictions_128bp": 0}
    assert get_prediction_key(raw, bin_size=128) == "scaled_predictions_128bp"
    assert get_prediction_key(raw, bin_size=1) == "scaled_predictions_1bp"


def test_prediction_key_falls_back_to_other_resolution():
    assert get_prediction_key({"scaled_predictions_1bp": 0}, bin_size=128) == "scaled_predictions_1bp"
    assert get_prediction_key({"scaled_predictions_128bp": 0}, bin_size=1) == "scaled_predictions_128bp"


def test_prediction_key_unscaled():
    assert get_prediction_key({"predictions_128bp": 0}, bin_size=128, return_scaled=False) == "predictions_128bp"


def test_prediction_key_missing_raises():
    with pytest.raises(KeyError, match="No matching prediction key"):
        get_prediction_key({"other": 0})


CONFIG = {"heads": [
    {"targets": [{"path": "/bw/a.bw", "nonzero_mean": 0.5}, {"path": "/bw/b.bw", "nonzero_mean": 1.5}]},
    {"targets": [{"path": "/bw/c.bw", "nonzero_mean": 2.5}, {"path": "/bw/no_mean.bw"}]},
]}


def test_nonzero_means_follow_tracks_order():
    tracks = pd.DataFrame({"path": ["/bw/c.bw", "/bw/a.bw", "/bw/b.bw"]})
    np.testing.assert_array_equal(get_aligned_nonzero_means(tracks, CONFIG), [2.5, 0.5, 1.5])


def test_nonzero_means_missing_path_raises():
    with pytest.raises(KeyError, match="no_mean.bw"):
        get_aligned_nonzero_means(pd.DataFrame({"path": ["/bw/no_mean.bw"]}), CONFIG)
