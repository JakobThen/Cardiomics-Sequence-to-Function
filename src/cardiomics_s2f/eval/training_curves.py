"""
Training Curves
===============

Plots of the per-epoch losses logged during gReLU (PyTorch Lightning) training, read
from the ``metrics.csv`` that the CSV logger writes to ``<run>/version_0/``.
"""

import matplotlib.pyplot as plt


def plot_training_curves(training_log_df):
    """Plots the training and validation loss per epoch.

    Args:
        training_log_df (pd.DataFrame): The ``metrics.csv`` of a gReLU training run, with
            the columns ``epoch``, ``train_loss_epoch`` and ``val_loss``.

    Returns:
        tuple:
            - matplotlib.figure.Figure: The loss plot.
            - pd.DataFrame: Training and validation loss per epoch (index ``epoch``;
              epochs with both losses only).
    """
    train_epoch = (
        training_log_df[["epoch", "train_loss_epoch"]]
        .dropna()
        .drop_duplicates(subset="epoch")
        .set_index("epoch")
    )

    val_epoch = (
        training_log_df[["epoch", "val_loss"]]
        .dropna()
        .drop_duplicates(subset="epoch")
        .set_index("epoch")
    )

    losses = train_epoch.join(val_epoch, how="inner")

    fig, axes = plt.subplots(1, 1, figsize=(5, 4))

    # Loss Plot
    axes.plot(losses.index, losses["train_loss_epoch"], label="Train Loss")
    axes.plot(losses.index, losses["val_loss"], label="Validation Loss")
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Loss")
    axes.set_title("Train vs Val Loss")
    axes.legend()
    axes.grid(True)

    return fig, losses
