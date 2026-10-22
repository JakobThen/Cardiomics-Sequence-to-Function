"""Tests for the variant ID, peak and strand helpers in cardiomics_s2f.io.variant_annotation."""

import importlib
import sys
from unittest import mock

import pandas as pd

# The module imports these at module level; the helpers tested here do not use them.
for _name in ("pyranges", "ncls", "pyBigWig"):
    try:
        importlib.import_module(_name)
    except ImportError:
        sys.modules[_name] = mock.MagicMock()

from cardiomics_s2f.io.variant_annotation import (  # noqa: E402
    add_variant_id,
    align_variants_to_reference_strand,
    celltype_from_peak_file,
    flag_variants_in_peaks,
    read_narrowpeak,
)


def test_add_variant_id():
    df = pd.DataFrame({"Chromosome": ["chr1"], "Pos": [10], "Ref": ["A"], "Alt": ["G"]})
    out = add_variant_id(df)
    assert out["variant_id"].tolist() == ["chr1-10-A-G"]
    assert out is not df


def test_read_narrowpeak(tmp_path):
    f = tmp_path / "CM.filtered_peaks.narrowPeak"
    f.write_text("chr1\t100\t200\tp1\t10\t.\t5.0\t3.0\t2.0\t50\n")
    peaks = read_narrowpeak(f)
    assert list(peaks.columns) == ["chrom", "chromStart", "chromEnd"]
    assert peaks.iloc[0].tolist() == ["chr1", 100, 200]


def test_flag_variants_in_peaks_half_open():
    peaks = pd.DataFrame({"chrom": ["chr1"], "chromStart": [100], "chromEnd": [200]})
    var = pd.DataFrame({
        "Chromosome": ["chr1", "chr1", "chr1", "chr2"],
        "Start": [99, 199, 200, 150],
        "End": [100, 200, 201, 151],
    }, index=[10, 11, 12, 13])
    mask = flag_variants_in_peaks(var, peaks)
    # [99,100) touches but does not overlap [100,200); chr2 has no peaks
    assert mask.to_dict() == {10: False, 11: True, 12: False, 13: False}


def test_celltype_from_peak_file():
    assert celltype_from_peak_file("/x/Cardiac_mesoderm.filtered.dedup_peaks.narrowPeak") == "Cardiac_mesoderm"


def test_align_variants_to_reference_strand():
    hg38 = {"chr1": "AACGTT"}  # 1-based: A A C G T T
    df = pd.DataFrame({
        "Chromosome": ["chr1", "chr1", "chr1"],
        "Pos": [3, 3, 4],
        "Ref": ["C", "G", "A"],      # matches; reverse complement of C; mismatch
        "Alt": ["T", "A", "C"],
        "variant_id": ["chr1-3-C-T", "chr1-3-G-A", "chr1-4-A-C"],
    })
    out = align_variants_to_reference_strand(df, hg38)
    assert out["variant_id"].tolist() == ["chr1-3-C-T", "chr1-3-C-T"]
    assert out["Alt"].tolist() == ["T", "T"]
    assert "Ref_hg38" not in out.columns
