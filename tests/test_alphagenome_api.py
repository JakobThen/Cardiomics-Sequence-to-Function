"""Tests for cardiomics_s2f.vep.alphagenome_api and cardiomics_s2f.eval.alphagenome_api.

alphagenome (and pyBigWig) are replaced by stand-ins if they are not installed;
the tested functions only use pandas and numpy.
"""

import importlib.abc
import importlib.machinery
import importlib.util
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest

_STUBBED = {n for n in ("alphagenome", "pyBigWig", "pyranges", "ncls") if importlib.util.find_spec(n) is None}


class _Stub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".")[0] in _STUBBED:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

    def create_module(self, spec):
        m = mock.MagicMock(name=spec.name)
        m.__path__ = []
        return m

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _Stub())

from cardiomics_s2f.eval.alphagenome_api import (  # noqa: E402
    aggregate_tracks_by_curies,
    group_regions_into_windows,
    masked_pearson_per_window,
    prediction_track_table,
    sum_exon_signal,
)
from cardiomics_s2f.vep.alphagenome_api import aggregate_modality_scores, read_variant_table  # noqa: E402


# ------------------------------------------------------------------ VEP ---
def test_read_variant_table_sniffs_separator(tmp_path):
    f = tmp_path / "variants.txt"
    f.write_text("CHROM\tPOS\tREF\tALT\nchr1\t10\tA\tG\n")
    df = read_variant_table(str(f))
    assert list(df.columns) == ["CHROM", "POS", "REF", "ALT"]
    with pytest.raises(FileNotFoundError):
        read_variant_table(str(tmp_path / "missing.csv"))


def test_aggregate_modality_scores():
    atac = pd.DataFrame({"variant_name": ["v1", "v1", "v2"], "raw_score": [1.0, 3.0, 5.0], "quantile_score": [0.1, 0.3, 0.5]})
    out, merge_cols, keep_genes = aggregate_modality_scores("ATAC", atac)
    assert out.set_index("variant_name")["ATAC_raw_score"].to_dict() == {"v1": 2.0, "v2": 5.0}
    assert merge_cols == ["variant_name"] and keep_genes is False

    rna = pd.DataFrame({
        "variant_name": ["v1"] * 3, "gene_id": ["g1", "g1", "g2"], "gene_name": ["A", "A", "B"],
        "gene_type": ["protein_coding", "protein_coding", "lncRNA"],
        "raw_score": [1.0, 2.0, 9.0], "quantile_score": [0.0, 1.0, 1.0],
    })
    out, merge_cols, keep_genes = aggregate_modality_scores("RNA_SEQ", rna)
    assert out["RNA_raw_score"].tolist() == [1.5] and keep_genes  # lncRNA dropped

    hist = pd.DataFrame({"variant_name": ["v1", "v1"], "histone_mark": ["H3K27ac", "H3K4me3"],
                         "raw_score": [1.0, 2.0], "quantile_score": [0.5, 0.6]})
    out, _, _ = aggregate_modality_scores("CHIP_HISTONE", hist)
    assert out.loc[0, "H3K27ac_HIST_raw_score"] == 1.0 and out.loc[0, "H3K4me3_HIST_quantile_score"] == 0.6

    assert aggregate_modality_scores("UNKNOWN", atac) == (None, [], False)


# --------------------------------------------------------------- tracks ---
def test_aggregate_tracks_by_curies():
    P = np.array([[1, 1], [3, 3], [10, 10], [20, 20]], dtype=float)
    meta = pd.DataFrame({"ontology_curie": ["A", "B", "A", "C"], "strand": ["+", "+", "-", "+"]})
    config = {"g1": ["A", "B"], "g2": ["C"]}
    out = aggregate_tracks_by_curies(P, meta, config, ["g1", "g2"])
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [[14 / 3, 14 / 3], [20, 20]], rtol=1e-6)
    np.testing.assert_array_equal(aggregate_tracks_by_curies(P, meta, config, ["g1"], strand="+"), [[2, 2]])


def test_prediction_track_table():
    t = prediction_track_table({"cm": "Cardiomyocytes", "naive": "Cardiomyocytes", "prog": "Cardiac_progenitors"})
    assert len(t) == 9
    assert t["filename"].tolist()[:3] == ["ATAC_Cardiomyocytes_100M_norm.bw"] * 2 + ["ATAC_Cardiac_progenitors_100M_norm.bw"]
    assert t["filename"].iloc[3] == "RNA_Cardiomyocytes_plus_100M.bw" and t["strand"].iloc[-1] == "minus"
    assert t["cell_type"].tolist() == ["cm", "naive", "prog"] * 3


def test_sum_exon_signal():
    pos = np.arange(20, dtype=float).reshape(10, 2)  # positions 100..109
    exons = [SimpleNamespace(start=101, end=102), SimpleNamespace(start=105, end=105)]  # 1-based inclusive
    expected = pos[[1, 2, 5]].sum(axis=0)
    np.testing.assert_array_equal(sum_exon_signal(pos, 100, exons), expected)
    np.testing.assert_array_equal(sum_exon_signal(pos, 100, exons, exon_length_norm=True), expected / 3)


def test_group_regions_into_windows():
    regions = pd.DataFrame({"seqnames": ["chr1", "chr1", "chr1", "chr2"],
                            "start": [1000, 3000, 50_000, 500], "end": [1200, 3500, 50_500, 900]})
    windows = group_regions_into_windows(regions, seq_len=20_000, max_span=10_000)
    # first two regions share a window (span 2500), third starts a new one, chr2 its own
    assert windows == [("chr1", 0, 20_000), ("chr1", 40_250, 60_250), ("chr2", 0, 20_000)]


def test_masked_pearson_per_window():
    rng = np.random.default_rng(1)
    P = rng.normal(size=(2, 200, 3)).astype(np.float32)
    T = rng.normal(size=(2, 200, 2)).astype(np.float32)
    T[0, :50, 0] = np.nan
    T_orig = T.copy()
    R, sum_T, sum_P = masked_pearson_per_window(P, T, min_valid=100)
    valid = ~np.isnan(T_orig[0, :, 0])
    expected = np.corrcoef(P[0, valid, 1], T_orig[0, valid, 0])[0, 1]
    assert R.shape == (2, 3, 2)
    assert R[0, 1, 0] == pytest.approx(expected, abs=1e-4)
    assert sum_P[0, 1, 0] == pytest.approx(P[0, valid, 1].sum(), rel=1e-4)
    # too few valid positions -> NaN
    R2, _, _ = masked_pearson_per_window(P, T_orig.copy(), min_valid=190)
    assert np.isnan(R2[0, 0, 0]) and not np.isnan(R2[1, 0, 0])


def _write_targets_config(path, targets):
    import yaml

    path.write_text(yaml.safe_dump({"heads": [{"id": "h", "targets": targets}]}))
    return path


def test_reference_bigwigs_from_targets_config(tmp_path):
    from cardiomics_s2f.eval.alphagenome_api import prediction_track_table, reference_bigwigs_from_targets_config

    config = _write_targets_config(tmp_path / "targets.yaml", [
        {"path": "/g/data/multiome_ATAC_bigwig/ATAC_CM_100M_norm.bw", "nonzero_mean": 0.7},
        {"path": "rna/RNA_CM_plus_100M.bw", "nonzero_mean": 1.8},
        {"path": "rna/RNA_CM_minus_100M.bw", "nonzero_mean": 1.6},
    ])
    tracks = prediction_track_table({"CM_group": "CM", "naive_group": "CM"})
    out = reference_bigwigs_from_targets_config(tracks, config)
    assert out["path"].tolist()[:2] == ["/g/data/multiome_ATAC_bigwig/ATAC_CM_100M_norm.bw"] * 2  # absolute kept
    assert out["path"].iloc[2] == str(tmp_path / "rna" / "RNA_CM_plus_100M.bw")              # relative resolved
    assert out["nonzero_means"].tolist() == [0.7, 0.7, 1.8, 1.8, 1.6, 1.6]


def test_reference_bigwigs_missing_duplicate_or_without_mean(tmp_path):
    from cardiomics_s2f.eval.alphagenome_api import reference_bigwigs_from_targets_config

    tracks = pd.DataFrame({"filename": ["ATAC_CM_100M_norm.bw"]})
    missing = _write_targets_config(tmp_path / "a.yaml", [{"path": "/x/ATAC_FB_100M_norm.bw", "nonzero_mean": 1}])
    with pytest.raises(KeyError, match="0 times"):
        reference_bigwigs_from_targets_config(tracks, missing)
    duplicate = _write_targets_config(tmp_path / "b.yaml", [{"path": "/x/ATAC_CM_100M_norm.bw", "nonzero_mean": 1},
                                                            {"path": "/y/ATAC_CM_100M_norm.bw", "nonzero_mean": 2}])
    with pytest.raises(KeyError, match="2 times"):
        reference_bigwigs_from_targets_config(tracks, duplicate)
    no_mean = _write_targets_config(tmp_path / "c.yaml", [{"path": "/x/ATAC_CM_100M_norm.bw"}])
    with pytest.raises(ValueError, match="nonzero_mean"):
        reference_bigwigs_from_targets_config(tracks, no_mean)
