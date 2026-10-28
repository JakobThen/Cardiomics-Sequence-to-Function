"""Tests for cardiomics_s2f.eval.training_curves."""

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cardiomics_s2f.eval.training_curves import plot_training_curves  # noqa: E402


def test_plot_training_curves_one_row_per_epoch():
    # Lightning logs train and validation metrics on separate rows (NaN elsewhere)
    log = pd.DataFrame({
        "epoch": [0, 0, 0, 1, 1, 2],
        "step": [10, 19, 19, 29, 29, 39],
        "train_loss_step": [5.0, np.nan, np.nan, np.nan, np.nan, 3.0],
        "train_loss_epoch": [np.nan, 4.0, np.nan, 3.5, np.nan, np.nan],
        "val_loss": [np.nan, np.nan, 4.5, np.nan, 3.8, np.nan],
    })
    fig, losses = plot_training_curves(log)
    assert losses.index.tolist() == [0, 1]  # epoch 2 has no validation loss yet
    assert losses["train_loss_epoch"].tolist() == [4.0, 3.5]
    assert losses["val_loss"].tolist() == [4.5, 3.8]
    assert len(fig.axes[0].lines) == 2
