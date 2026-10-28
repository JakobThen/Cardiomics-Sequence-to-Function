"""
Borzoi Training Datasets
========================

:class:`StreamingBigWigSeqDataset` is a drop-in replacement for gReLU's
``BigWigSeqDataset`` that reads the labels of each window from the BigWig files on
demand instead of loading all labels into memory. 
* interval coordinates are kept as plain NumPy arrays (no pandas lookups per item),
  so DataLoader workers share them copy-on-write;
* labels are written into one pre-allocated ``float32`` buffer per item;
* ``pyBigWig`` handles are opened lazily inside each worker (they cannot be pickled).
"""

from typing import Callable, List, Optional, Tuple, Union
import os

import numpy as np
import pandas as pd
import pyBigWig
import torch
from einops import rearrange
from grelu.data.augment import _split_overall_idx
from grelu.data.dataset import LabeledSeqDataset
from grelu.sequence.format import indices_to_one_hot
from grelu.sequence.utils import resize


class StreamingBigWigSeqDataset(LabeledSeqDataset):
    """Dataset of genomic intervals with labels streamed from BigWig files.

    Takes the same arguments as gReLU's ``BigWigSeqDataset``.

    Args:
        intervals (pd.DataFrame): Genomic intervals (``chrom``, ``start``, ``end``).
        bw_files (str or list of str): BigWig files, one per task.
        tasks (list of str or pd.DataFrame, optional): Task names. Defaults to the
            BigWig file names without extension.
        seq_len (int, optional): Length of the output sequences.
        genome (str, optional): Genome to read the sequences from.
        end (str, optional): Which end to resize ("left", "right" or "both").
        rc (bool, optional): Augment by reverse complementation.
        max_seq_shift (int, optional): Maximum sequence shift for augmentation.
        label_len (int, optional): Length of the label window.
        max_pair_shift (int, optional): Maximum joint shift of sequence and label.
        label_aggfunc (str or callable, optional): Aggregation of labels over ``bin_size``.
        bin_size (int, optional): Number of bases per label bin.
        min_label_clip (int, optional): Minimum label value.
        max_label_clip (int, optional): Maximum label value.
        label_transform_func (str or callable, optional): Transform applied to the labels
            (e.g. :class:`~cardiomics_s2f.io.data_input.SelectiveSquashTransform`).
        seed (int, optional): Random seed for augmentation.
        augment_mode (str, optional): "serial" or "random".
    """

    def __init__(
        self,
        intervals: pd.DataFrame,
        bw_files: Union[str, List[str]],
        tasks: Optional[Union[List[str], pd.DataFrame]] = None,
        seq_len: Optional[int] = None,
        genome: Optional[str] = None,
        end: str = "both",
        rc: bool = False,
        max_seq_shift: int = 0,
        label_len: Optional[int] = None,
        max_pair_shift: int = 0,
        label_aggfunc: Optional[Union[str, Callable]] = np.sum,
        bin_size: Optional[int] = None,
        min_label_clip: Optional[int] = None,
        max_label_clip: Optional[int] = None,
        label_transform_func: Optional[Union[str, Callable]] = None,
        seed: Optional[int] = None,
        augment_mode: str = "serial",
    ) -> None:

        tasks = tasks or [os.path.splitext(os.path.basename(f))[0] for f in (bw_files if isinstance(bw_files, list) else [bw_files])]

        super().__init__(
            seqs=intervals,
            labels=bw_files,
            tasks=tasks,
            seq_len=seq_len,
            genome=genome,
            end=end,
            rc=rc,
            max_seq_shift=max_seq_shift,
            max_pair_shift=max_pair_shift,
            label_len=label_len,
            label_aggfunc=label_aggfunc,
            bin_size=bin_size,
            min_label_clip=min_label_clip,
            max_label_clip=max_label_clip,
            label_transform_func=label_transform_func,
            seed=seed,
            augment_mode=augment_mode,
        )

    def _load_labels(self, bw_files: Union[str, List[str]]) -> None:
        # Overrides LabeledSeqDataset._load_labels: store paths and label windows only.
        self.bw_files = [bw_files] if isinstance(bw_files, str) else bw_files

        # Pre-calculate the padded intervals
        label_intervals = resize(
            self.intervals, self.padded_label_len, input_type="intervals"
        )

        # Strip pandas away completely. Convert to primitive NumPy arrays
        # to ensure perfect copy-on-write sharing across DataLoader workers.
        self._chroms = label_intervals["chrom"].to_numpy().astype(str)
        self._starts = label_intervals["start"].to_numpy().astype(int)
        self._ends = label_intervals["end"].to_numpy().astype(int)

        self.bws = None

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Open the file handles once per worker. The cluster patch also tried to reopen
        # them at every new epoch (idx // (n_seqs * n_augmented) changing), but idx is
        # always smaller than len(self), so that check never fired; it is left out.
        if getattr(self, "bws", None) is None:
            self.bws = [pyBigWig.open(bw, "r") for bw in self.bw_files]

        seq_idx, augment_idx = _split_overall_idx(idx, (self.n_seqs, self.n_augmented))
        seq = self.seqs[seq_idx]

        # Fetch coordinates directly from the primitive arrays (no .iloc)
        chrom = self._chroms[seq_idx]
        start = self._starts[seq_idx]
        end = self._ends[seq_idx]

        # Pre-allocate a contiguous NumPy buffer to stop shared memory fragmentation
        n_tasks = len(self.bws)
        label = np.empty((n_tasks, end - start), dtype=np.float32)

        for i, bw in enumerate(self.bws):
            signal = bw.values(chrom, start, end, numpy=True)
            label[i, :] = np.nan_to_num(signal).astype(np.float32)

        seq, label = self.augmenter(seq=seq, label=label, idx=augment_idx)
        seq = indices_to_one_hot(seq)

        if getattr(self, "predict", False):
            return seq

        if self.label_aggfunc is not None:
            label = rearrange(label, "t (l b) -> t l b", b=self.bin_size)
            label = self.label_aggfunc(label, axis=-1)

        if getattr(self, "label_transform", None) is not None:
            label = self.label_transform(label)

        # Convert the contiguous buffer directly to a tensor
        return seq, torch.as_tensor(label)

    def __del__(self):
        if getattr(self, "bws", None) is not None:
            for bw in self.bws:
                try:
                    bw.close()
                except Exception:
                    pass
