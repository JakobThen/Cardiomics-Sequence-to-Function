"""Tests for cardiomics_s2f.preprocessing.fold_intervals."""

from types import SimpleNamespace

import pandas as pd
import pytest

from cardiomics_s2f.preprocessing.fold_intervals import (
    fold_interval_bed_name,
    intervals_to_frame,
    write_fold_intervals,
)


def _iv(chrom, start, end):
    return SimpleNamespace(chromosome=chrom, start=start, end=end)


@pytest.fixture
def intervals():
    return {
        "train": [_iv("chr1", 0, 1_048_576), _iv("chr2", 10, 1_048_586)],
        "valid": [_iv("chr3", 5, 1_048_581)],
        "test": [_iv("chr17", 59390033, 60438609)],
    }


def test_bed_name():
    assert fold_interval_bed_name("test", "1", "HOMO_SAPIENS") == "AG_test_intervals_fold1_HOMO_SAPIENS.bed"


def test_intervals_to_frame(intervals):
    df = intervals_to_frame(intervals["train"])
    assert list(df.columns) == ["chrom", "start", "end"]
    assert df.iloc[1].tolist() == ["chr2", 10, 1_048_586]


def test_write_matches_existing_format(tmp_path, intervals):
    paths = write_fold_intervals(intervals, tmp_path, "1", "HOMO_SAPIENS")
    assert set(paths) == {"train", "valid", "test"}
    # same layout as the BEDs used so far: tab-separated, header line
    assert paths["test"].read_text().splitlines() == ["chrom\tstart\tend", "chr17\t59390033\t60438609"]
    assert len(pd.read_csv(paths["train"], sep="\t")) == 2


def test_refuses_overwrite(tmp_path, intervals):
    write_fold_intervals(intervals, tmp_path, "1", "HOMO_SAPIENS")
    with pytest.raises(FileExistsError):
        write_fold_intervals(intervals, tmp_path, "1", "HOMO_SAPIENS")
    write_fold_intervals(intervals, tmp_path, "1", "HOMO_SAPIENS", overwrite=True)


def test_missing_split(tmp_path, intervals):
    del intervals["valid"]
    with pytest.raises(ValueError, match="valid"):
        write_fold_intervals(intervals, tmp_path, "1", "HOMO_SAPIENS")
