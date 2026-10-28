"""Tests for cardiomics_s2f.vep.chrombpnet (pysam replaced by a stand-in)."""

import sys
import types

import pandas as pd
import pytest

from cardiomics_s2f.vep.chrombpnet import (
    drop_variants_with_ref_mismatch,
    read_variant_table,
    to_chrombpnet_variants,
    trim_shared_allele_prefix,
    write_chrombpnet_variants,
)


@pytest.mark.parametrize("pos, ref, alt, expected", [
    (100, "C", "T", (100, "C", "T")),              # SNV unchanged
    (100, "AGA", "CGC", (100, "AGA", "CGC")),      # MNV unchanged
    (100, "AG", "A", (101, "G", "-")),             # deletion
    (100, "A", "AT", (101, "-", "T")),             # insertion
    (100, "ACG", "AT", (101, "CG", "T")),          # complex indel
    (100, "ACG", "T", (100, "ACG", "T")),          # complex, no shared base
    (100, "ACAC", "AC", (102, "AC", "-")),         # non-normalised deletion
])
def test_trim_shared_allele_prefix(pos, ref, alt, expected):
    df = pd.DataFrame({"pos": [pos], "allele1": [ref], "allele2": [alt]})
    out = trim_shared_allele_prefix(df)
    assert (out["pos"].iloc[0], out["allele1"].iloc[0], out["allele2"].iloc[0]) == expected


def test_to_chrombpnet_variants_columns_and_ids():
    df = pd.DataFrame({"CHROM": ["chr1", "chr2"], "POS": [10, 20], "REF": ["ag", "C"], "ALT": ["a", "T"],
                       "extra": [1, 2]})
    out = to_chrombpnet_variants(df)
    assert list(out.columns) == ["chr", "pos", "allele1", "allele2", "variant_id"]
    assert out.values.tolist() == [["chr1", 11, "G", "-", "chr1-10-AG-A"], ["chr2", 20, "C", "T", "chr2-20-C-T"]]


def test_to_chrombpnet_variants_keeps_dash_format_and_ids():
    df = pd.DataFrame({"chr": ["chr1"], "pos": [11], "allele1": ["G"], "allele2": ["-"], "variant_id": ["v1"]})
    assert to_chrombpnet_variants(df).values.tolist() == [["chr1", 11, "G", "-", "v1"]]


def test_to_chrombpnet_variants_missing_column():
    with pytest.raises(ValueError, match="allele2"):
        to_chrombpnet_variants(pd.DataFrame({"CHROM": ["chr1"], "POS": [1], "REF": ["A"]}))


class _FakeFasta:
    SEQ = {"chr1": "ACGTACGTAC"}

    def __init__(self, path):
        self.references = list(self.SEQ)

    def fetch(self, chrom, start, end):
        if end > len(self.SEQ[chrom]):
            raise ValueError("outside")
        return self.SEQ[chrom][start:end]

    def close(self):
        pass


@pytest.fixture
def fake_pysam(monkeypatch):
    monkeypatch.setitem(sys.modules, "pysam", types.SimpleNamespace(FastaFile=_FakeFasta))


def test_drop_variants_with_ref_mismatch(fake_pysam):
    df = pd.DataFrame({
        "chr": ["chr1", "1", "chr1", "chr2", "chr1"],
        "pos": [1, 2, 3, 1, 10],
        "allele1": ["A", "c", "T", "A", "CA"],   # ok, ok (no prefix, lower case), mismatch, unknown chrom, past end
        "allele2": ["G", "T", "A", "G", "C"],
    })
    out = drop_variants_with_ref_mismatch(df, "genome.fa")
    assert out.index.tolist() == [0, 1]


def test_write_chrombpnet_variants(tmp_path, fake_pysam):
    src = tmp_path / "vars.csv"
    pd.DataFrame({"variant_id": ["v1", "v2", "v3"], "CHROM": ["chr1"] * 3, "POS": [2, 3, 5],
                  "REF": ["CG", "G", "A"], "ALT": ["C", "C", "AGG"]}).to_csv(src, index=False)
    out_file = tmp_path / "out.tsv"
    write_chrombpnet_variants(str(src), str(out_file), genome_path="genome.fa")
    assert out_file.read_text().splitlines() == ["chr1\t3\tG\t-\tv1", "chr1\t3\tG\tC\tv2", "chr1\t6\t-\tGG\tv3"]
    assert read_variant_table(str(src)).shape == (3, 5)
