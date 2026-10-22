"""Tests for cardiomics_s2f.vep.cohort_plotting (helpers and figure smoke tests)."""

import importlib
import sys
from unittest import mock

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

for _name in ("pyranges", "ncls", "pyBigWig"):
    try:
        importlib.import_module(_name)
    except ImportError:
        sys.modules[_name] = mock.MagicMock()

from cardiomics_s2f.vep.cohort_plotting import (  # noqa: E402
    CRE_ORDER,
    _significance_stars,
    format_p_value,
    label_cardioid_enhancer_state,
    plot_richter_cohort_overview,
    plot_richter_cohort_qc,
)


@pytest.mark.parametrize("p, expected", [(np.nan, ""), (0.0004, "P < 0.001"), (0.0123, "P = 0.012")])
def test_format_p_value(p, expected):
    assert format_p_value(p) == expected


@pytest.mark.parametrize("p, expected", [(np.nan, ""), (0.0001, "***"), (0.005, "**"), (0.03, "*"), (0.2, "ns")])
def test_significance_stars(p, expected):
    assert _significance_stars(p) == expected


@pytest.mark.parametrize("row, expected", [
    ({"in_cardioid_enhancer": False, "enhancer_regulatory_position": None, "is_cardioid_H3K27ac": None}, "Not Active in Cardioid"),
    ({"in_cardioid_enhancer": np.nan, "enhancer_regulatory_position": None, "is_cardioid_H3K27ac": None}, "Not Active in Cardioid"),
    ({"in_cardioid_enhancer": True, "enhancer_regulatory_position": "distal", "is_cardioid_H3K27ac": True}, "Distal H3K27ac+"),
    ({"in_cardioid_enhancer": True, "enhancer_regulatory_position": "proximal", "is_cardioid_H3K27ac": False}, "Proximal H3K27ac-"),
])
def test_label_cardioid_enhancer_state(row, expected):
    assert label_cardioid_enhancer_state(pd.Series(row)) == expected


def _cohort(n, seed):
    """Synthetic annotated variants with the columns the figures use."""
    rng = np.random.default_rng(seed)
    bases = np.array(list("ACGT"))
    return pd.DataFrame({
        "Blinded ID": rng.choice([f"p{i}" for i in range(20)], n),
        "Ref": rng.choice(bases, n),
        "Alt": rng.choice(np.array(["A", "C", "G", "T", "AT", "C"]), n),
        "cCRE_class": rng.choice(np.array(CRE_ORDER + [None], dtype=object), n),
        "gtf_feature": rng.choice(np.array(["exon", "intron", "UTR", None], dtype=object), n),
        "coding_effect": rng.choice(np.array(["missense_variant", "synonymous_variant", None], dtype=object), n),
        "closest_TSS": rng.integers(0, 1_000_000, n),
        "phyloP_100way": rng.normal(size=n),
        "CADD_PHRED_score": rng.uniform(0, 30, n),
        "DNase_maxZ": rng.normal(size=n), "H3K4me3_maxZ": rng.normal(size=n),
        "H3K27ac_maxZ": rng.normal(size=n), "CTCF_maxZ": rng.normal(size=n),
        "in_cardioid_enhancer": rng.choice([True, False], n),
        "celltypes_in_enhancer": rng.choice(["Cardiomyocytes", "Cardiomyocytes, Fibroblasts"], n),
        "enhancer_regulatory_position": rng.choice(["proximal", "distal"], n),
        "is_cardioid_H3K27ac": rng.choice([True, False], n),
    })


def test_plot_richter_cohort_figures_run(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    chd, ctrl = _cohort(300, 0), _cohort(300, 1)
    ca = {}
    for ct in ("Cardiomyocytes", "Fibroblasts"):
        ca[ct] = {}
        for name, df in (("CHD", chd), ("Control", ctrl)):
            mask = df.index % 2 == 0
            ca[ct][name] = {"in": df[mask], "out": df[~mask]}
    plot_richter_cohort_overview(chd, ctrl, CA_results_dict=ca, celltype_filter=["Cardiomyocytes", "Fibroblasts"])
    plot_richter_cohort_qc(chd, ctrl)
    plt.close("all")
