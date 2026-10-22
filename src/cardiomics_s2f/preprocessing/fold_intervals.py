"""
Fold Intervals
==============

Writes AlphaGenome's genomic fold split (train/valid/test intervals of 1 Mbp) as BED
files, so that the pretrained-AlphaGenome benchmarks use exactly the intervals of
AlphaGenome fine-tuning. (Borzoi has its own 524 kbp split.)

The intervals themselves come from ``alphagenome_ft.finetune.data.prepare_intervals_from_fold``;
the command-line script is ``workflows/alphagenome_ft/export_fold_intervals.py``.
"""

from pathlib import Path
from typing import Dict, Iterable, Mapping, Union

import pandas as pd

FOLD_SPLITS = ("train", "valid", "test")


def fold_interval_bed_name(split: str, fold: str, organism: str) -> str:
    """Returns the file name of one split's BED file.

    Args:
        split (str): ``"train"``, ``"valid"`` or ``"test"``.
        fold (str): Fold identifier (e.g. ``"1"``).
        organism (str): Organism name (e.g. ``"HOMO_SAPIENS"``).

    Returns:
        str: ``AG_<split>_intervals_fold<fold>_<organism>.bed``.
    """
    return f"AG_{split}_intervals_fold{fold}_{organism}.bed"


def intervals_to_frame(intervals: Iterable) -> pd.DataFrame:
    """Converts genomic intervals to a BED-style table.

    Args:
        intervals (iterable): Objects with ``chromosome``, ``start`` and ``end``
            attributes (e.g. ``alphagenome.data.genome.Interval``).

    Returns:
        pd.DataFrame: Columns ``chrom``, ``start``, ``end`` (0-based, half-open).
    """
    return pd.DataFrame(
        [(iv.chromosome, iv.start, iv.end) for iv in intervals],
        columns=["chrom", "start", "end"],
    )


def write_fold_intervals(
    intervals: Mapping[str, Iterable],
    out_dir: Union[str, Path],
    fold: str,
    organism: str,
    overwrite: bool = False,
) -> Dict[str, Path]:
    """Writes the train, valid and test intervals of a fold as BED files.

    The files are tab-separated with a ``chrom start end`` header line and LF
    line endings on every platform.

    Args:
        intervals (Mapping[str, Iterable]): Intervals per split, as returned by
            ``prepare_intervals_from_fold``; must contain ``train``, ``valid`` and ``test``.
        out_dir (str or Path): Output folder (created if missing).
        fold (str): Fold identifier, used in the file names.
        organism (str): Organism name, used in the file names.
        overwrite (bool, optional): Replace existing files. Defaults to False.

    Returns:
        dict: ``{split: Path}`` of the written files.

    Raises:
        ValueError: If a split is missing from ``intervals``.
        FileExistsError: If an output file exists and ``overwrite`` is False.
    """
    missing = [s for s in FOLD_SPLITS if s not in intervals]
    if missing:
        raise ValueError(f"Missing splits in intervals: {missing}")

    out_dir = Path(out_dir)
    paths = {s: out_dir / fold_interval_bed_name(s, fold, organism) for s in FOLD_SPLITS}
    existing = [str(p) for p in paths.values() if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite (use overwrite=True): {', '.join(existing)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, path in paths.items():
        intervals_to_frame(intervals[split]).to_csv(path, sep="\t", index=False, header=True, lineterminator="\n")
    return paths
