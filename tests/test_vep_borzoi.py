"""Tests for cardiomics_s2f.vep.borzoi (numpy/pandas parts; the dataset needs gReLU)."""

import importlib
import sys
from unittest import mock

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

# io.data_input imports pyBigWig (via batch_bw_loader); it does not build on Windows
try:
    importlib.import_module("pyBigWig")
except ImportError:
    sys.modules["pyBigWig"] = mock.MagicMock()

from cardiomics_s2f.io.data_input import SelectiveSquashTransform  # noqa: E402
from cardiomics_s2f.vep.borzoi import (  # noqa: E402
    _split_overall_idx,
    average_indel_augmentations,
    filter_variants_near_chrom_ends,
    read_gtf_exons,
    score_variants_masked_windows,
    score_variants_whole_window,
    stitch_indel_predictions,
    unsquash_rna_predictions,
)


def test_split_overall_idx_roundtrip():
    dims = (4, 2, 3)
    decoded = [_split_overall_idx(i, dims) for i in range(np.prod(dims))]
    assert decoded == [(s, a, l) for s in range(4) for a in range(2) for l in range(3)]


def test_stitch_insertion_and_deletion():
    preds3 = np.stack([np.full((2, 1, 4), k, dtype=float) for k in range(3)], axis=1)  # (2 vars, 3 alleles, 1, 4)
    var_df = pd.DataFrame({"ref": ["A", "ACG"], "alt": ["AT", "A"]})
    out = stitch_indel_predictions(preds3, var_df)
    assert out.shape == (2, 2, 1, 4)
    # insertion: REF = allele 0; ALT = left of allele 1, right of allele 2
    np.testing.assert_array_equal(out[0, 0, 0], [0, 0, 0, 0])
    np.testing.assert_array_equal(out[0, 1, 0], [1, 1, 2, 2])
    # deletion: REF = left of allele 0, right of allele 1; ALT = allele 2
    np.testing.assert_array_equal(out[1, 0, 0], [0, 0, 1, 1])
    np.testing.assert_array_equal(out[1, 1, 0], [2, 2, 2, 2])


def test_average_indel_augmentations_flips_rc_half():
    pytest.importorskip("einops")
    bins = np.arange(4, dtype=float)
    # 1 variant, 2 augmentations (fwd, rc), 3 alleles, 1 track, 4 bins
    fwd = [bins + 10 * a for a in range(3)]
    rc = [(bins + 10 * a)[::-1] for a in range(3)]
    flat = np.array(fwd + rc)[:, None, :]
    out = average_indel_augmentations(flat, n_augmented=2, n_alleles=3, rc=True)
    assert out.shape == (1, 3, 1, 4)
    for a in range(3):
        np.testing.assert_allclose(out[0, a, 0], bins + 10 * a)


def test_filter_variants_near_chrom_ends():
    var = pd.DataFrame({"chrom": ["chr1", "chr1", "chr1"], "pos": [10, 500, 990],
                        "ref": ["A", "A", "A"], "alt": ["C", "CT", "G"]})
    kept = filter_variants_near_chrom_ends(var, {"chr1": 1000}, seq_len=100)
    assert kept["pos"].tolist() == [500]  # safe distance = 50 + 2 + 10


def test_unsquash_only_rna_tracks():
    names = ["ATAC_CM_100M_norm.bw", "RNA_CM_plus_100M.bw"]
    raw = np.abs(np.random.default_rng(0).normal(size=(3, 2, 2, 5))) * 1000
    squashed = SelectiveSquashTransform(rna_task_indices=[1], transform="Borzoi")(raw)
    out = unsquash_rna_predictions(squashed, names)
    np.testing.assert_allclose(out[:, :, 1], raw[:, :, 1], rtol=1e-4)
    np.testing.assert_array_equal(out[:, :, 0], squashed[:, :, 0])


def test_whole_window_scores():
    preds = np.zeros((1, 2, 2, 3))
    preds[0, 1, 0] = 1.0  # ALT +1 in every bin of track 0
    var_df = pd.DataFrame({"variant_id": ["v1"]})
    out = score_variants_whole_window(preds, var_df, ["t0", "t1"])
    assert out["track"].tolist() == ["t0", "t1"]
    np.testing.assert_allclose(out["SUM_score"], [3.0, 0.0])
    np.testing.assert_allclose(out["logSUM_score"], [3.0, 0.0])
    np.testing.assert_allclose(out["L2_score"], [np.sqrt(3), 0.0])


def _gtf(tmp_path):
    rows = [
        ["chr1", "HAVANA", "gene", 1, 400, ".", "+", ".", 'gene_id "G1"; gene_type "protein_coding"; gene_name "GENE1";'],
        ["chr1", "HAVANA", "exon", 101, 164, ".", "+", ".", 'gene_id "G1"; transcript_id "T1"; gene_type "protein_coding"; gene_name "GENE1";'],
        ["chr1", "HAVANA", "exon", 101, 164, ".", "+", ".", 'gene_id "G2"; transcript_id "T2"; gene_type "lncRNA"; gene_name "LNC";'],
        ["chrM", "HAVANA", "exon", 1, 50, ".", "+", ".", 'gene_id "G3"; transcript_id "T3"; gene_type "protein_coding"; gene_name "MT1";'],
    ]
    path = tmp_path / "genes.gtf"
    pd.DataFrame(rows).to_csv(path, sep="\t", header=False, index=False, quoting=3)
    return path


def test_read_gtf_exons(tmp_path):
    exons = read_gtf_exons(_gtf(tmp_path))
    assert exons["gene_name"].tolist() == ["GENE1"]
    assert exons["transcript_id"].tolist() == ["T1"]


def test_masked_window_scores(tmp_path):
    exons = read_gtf_exons(_gtf(tmp_path))
    label_len, n_bins = 320, 10  # bin size 32
    names = ["ATAC_CM", "RNA_CM_plus", "RNA_CM_minus", "K27acDay1"]
    preds = np.ones((1, 2, 4, n_bins))
    preds[0, 1, 0, 4:6] = 3.0   # ATAC change in the centre bins
    preds[0, 1, 1, :] = 3.0     # plus-strand RNA change everywhere
    var_df = pd.DataFrame({"variant_id": ["v1"], "chrom": ["chr1"], "pos": [161]})
    out = score_variants_masked_windows(preds, var_df, exons, names, label_len)

    atac = out[out["assay_type"] == "ATAC"].iloc[0]
    # 501 bp > label_len: the window covers all bins, two of which change by log2(4/2) = 1
    assert atac["L2ED_score"] == pytest.approx(np.sqrt(2))
    assert atac["logSED_score"] == pytest.approx(np.log2(14 / 10))

    rna = out[out["assay_type"] == "RNA"]
    assert rna["track"].tolist() == ["RNA_CM_plus"]  # only the gene's strand
    assert rna["gene"].tolist() == ["GENE1"]
    # window starts at 161 - 1 - 160 = 0; exon 101-164 -> bins 3..5 (3 bins)
    assert rna.iloc[0]["L2ED_score"] == pytest.approx(np.sqrt(3))
    assert rna.iloc[0]["logSED_score"] == pytest.approx(np.log2(3.0))
    assert set(out["assay_type"]) == {"ATAC", "Cut&Tag", "RNA"}
