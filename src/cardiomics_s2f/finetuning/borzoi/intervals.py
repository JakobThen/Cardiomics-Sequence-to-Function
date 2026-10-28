"""
Borzoi Fold Intervals
=====================

Train/validation/test windows for Borzoi fine-tuning. Borzoi uses its own
``human_rep0`` split (524 kbp windows stored in the pretrained model's
``data_params``), not the AlphaGenome fold intervals of
``workflows/alphagenome_ft/export_fold_intervals.py``.

The windows are restricted to autosomes and chrX, windows that run past a
chromosome end are dropped, and windows within ``blacklist_window`` bp of an
ENCODE blacklist region are removed.
"""

import pandas as pd


def drop_intervals_beyond_chrom_ends(df, chrom_size_df,
                                     chrom_col="chrom",
                                     start_col="start",
                                     end_col="end"):
    """Keeps only intervals that lie fully inside their chromosome.

    Intervals on chromosomes missing from ``chrom_size_df`` are dropped as well.

    Args:
        df (pd.DataFrame): Intervals with chromosome, start and end columns.
        chrom_size_df (pd.DataFrame): Chromosome sizes with columns ``chrom`` and ``end``.
        chrom_col (str, optional): Chromosome column of ``df``. Defaults to "chrom".
        start_col (str, optional): Start column of ``df``. Defaults to "start".
        end_col (str, optional): End column of ``df``. Defaults to "end".

    Returns:
        pd.DataFrame: The intervals with ``start >= 0`` and ``end <= chromosome length``.
    """
    chrom_size_dict = dict(zip(chrom_size_df["chrom"], chrom_size_df["end"]))
    df = df.copy()
    df["_chrom_end"] = df[chrom_col].map(chrom_size_dict)
    ret = df[
        (df[start_col] >= 0) &
        (df[end_col] <= df["_chrom_end"])
    ]
    print(f"Keeping {len(ret)} of {len(df)} intervals after chromosome end overlap.")
    return ret.drop(columns="_chrom_end")


def build_borzoi_fold_intervals(model, chrom_sizes, genome=None, blacklist_window=1000):
    """Builds the filtered train/validation/test windows of a pretrained Borzoi model.

    Reproduces the interval preparation of the ``gReLu_Borzoi_test`` notebook that
    wrote ``Borzoi_human_rep0_{train,val,test}_intervals.bed``.

    Args:
        model: gReLU ``LightningModel`` of the pretrained Borzoi (``human_rep0``); its
            ``data_params[split]["intervals"]`` hold the original split.
        chrom_sizes (pd.DataFrame): Chromosome sizes with columns ``chrom`` and ``end``.
        genome (str, optional): Genome name for the blacklist. Defaults to the model's
            training genome (``hg38``).
        blacklist_window (int, optional): Windows closer than this many bp to a blacklist
            region are removed. Defaults to 1000.

    Returns:
        dict: ``{"train": df, "val": df, "test": df}`` with columns ``chrom``, ``start``, ``end``.
    """
    import grelu.data.preprocess
    import grelu.data.utils

    if genome is None:
        genome = model.data_params["train"]["genome"]

    chrom_size = chrom_sizes[
        chrom_sizes.chrom.isin(grelu.data.utils.get_chromosomes("autosomesX"))
    ]

    splits = {}
    for split in ("train", "val", "test"):
        intervals = pd.DataFrame(model.data_params[split]["intervals"])
        intervals = drop_intervals_beyond_chrom_ends(intervals, chrom_size)
        splits[split] = grelu.data.preprocess.filter_blacklist(
            intervals,
            genome=genome,
            window=blacklist_window,
        )
    return splits
