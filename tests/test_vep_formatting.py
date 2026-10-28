"""Tests for the VEP table formatting of the fine-tuned models and the power analysis."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from cardiomics_s2f.vep.processing import (
    format_alphagenome_ft_vep_per_gene,
    format_borzoi_vep_per_gene,
    strip_phasing_suffix,
)


def test_strip_phasing_suffix():
    ids = pd.Series(["chr1-10-A-G-(for)", "chr1-10-A-G-(rev)", "chr2-5-C-T-phased", "chr3-7-G-A"])
    assert strip_phasing_suffix(ids).tolist() == ["chr1-10-A-G", "chr1-10-A-G", "chr2-5-C-T", "chr3-7-G-A"]


def _ag_ft_row(variant_id, output_type, track_name, score, gene=None):
    return dict(variant_id=variant_id, scored_interval="chr1:0-1", gene_id=gene and f"ENSG_{gene}",
                gene_name=gene, output_type=output_type, track_name=track_name, score=score)


def test_format_alphagenome_ft_vep_per_gene():
    rows = [
        # Cut&Tag: days and haplotypes averaged, marks harmonised
        _ag_ft_row("v1-(for)", "CutnTag", "K4Day4", 1.0),
        _ag_ft_row("v1-(rev)", "CutnTag", "K4Day8", 3.0),
        _ag_ft_row("v1-(for)", "CutnTag", "K27acDay4", 4.0),
        # ATAC per cell type; other cell types dropped
        _ag_ft_row("v1-(for)", "multiome_ATAC", "Cardiomyocytes_ATAC", 0.5),
        _ag_ft_row("v1-(for)", "multiome_ATAC", "Endoderm_ATAC", 9.0),
        # RNA: only the track on the gene strand, haplotypes averaged
        _ag_ft_row("v1-(for)", "multiome_RNA", "Cardiomyocytes_RNA_plus", 2.0, gene="GP"),
        _ag_ft_row("v1-(rev)", "multiome_RNA", "Cardiomyocytes_RNA_plus", 4.0, gene="GP"),
        _ag_ft_row("v1-(for)", "multiome_RNA", "Cardiomyocytes_RNA_minus", 7.0, gene="GP"),
        _ag_ft_row("v1-(for)", "multiome_RNA", "Cardiomyocytes_RNA_minus", -1.0, gene="GM"),
    ]
    gtf = pd.DataFrame({"gene_id": ["ENSG_GP", "ENSG_GM"], "gene_name": ["GP", "GM"], "strand": ["+", "-"]})

    out = format_alphagenome_ft_vep_per_gene(pd.DataFrame(rows), gtf).set_index("Gene")

    assert set(out.index) == {"GP", "GM"}
    assert (out.cell_type == "CM").all() and (out.variant_id == "v1").all()
    assert out.loc["GP", "AG_ft_RNA_score"] == 3.0
    assert out.loc["GM", "AG_ft_RNA_score"] == -1.0
    assert (out.AG_ft_ATAC_score == 0.5).all()
    assert (out.AG_ft_H3K4me3_score == 2.0).all()
    assert (out.AG_ft_H3K27ac_score == 4.0).all()


def test_format_borzoi_vep_per_gene():
    scores = ("L2ED_score", "logSED_score", "SUM_score", "L2_score", "logSUM_score")
    rows = [
        dict(variant_id="v1-(for)", track="K4Day4_100M_norm.bw", assay_type="Cut&Tag", gene=None),
        dict(variant_id="v1-(rev)", track="K4Day8_100M_norm.bw", assay_type="Cut&Tag", gene=None),
        dict(variant_id="v1-(for)", track="Fibroblasts_ATAC.bw", assay_type="ATAC", gene=None),
        dict(variant_id="v1-(for)", track="Fibroblasts_RNA.bw", assay_type="RNA", gene="G1"),
        dict(variant_id="v1-(rev)", track="Fibroblasts_RNA.bw", assay_type="RNA", gene="G1"),
    ]
    vep = pd.DataFrame(rows)
    for s in scores:
        vep[s] = [1.0, 3.0, 5.0, 2.0, 4.0]
    vep.loc[vep.assay_type != "RNA", ["L2ED_score", "logSED_score"]] = np.nan

    out = format_borzoi_vep_per_gene(vep)

    assert len(out) == 1
    row = out.iloc[0]
    assert (row.variant_id, row.cell_type, row.Gene) == ("v1", "FB", "G1")
    assert row.logSED_score == 3.0 and row.SUM_score_RNA == 3.0
    assert row.SUM_score_ATAC == 5.0
    assert row.SUM_score_H3K4me3 == 2.0


def test_mannwhitney_power_grid_reaches_power_for_separated_groups():
    from cardiomics_s2f.vep.stats import mannwhitney_power_grid

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"has_effect": [True] * 50 + [False] * 50,
                       "score": np.r_[rng.normal(3, 1, 50), rng.normal(0, 1, 50)]})
    results, required_n = mannwhitney_power_grid(
        df, x_col="has_effect", y_cols="score", alternative="greater",
        n_range=(5, 10, 20), n_sims=50, random_state=0)

    assert list(results["score"].columns) == ["n", "power", "power_ci_lower", "power_ci_upper"]
    assert required_n["score"]["estimate"] in (5, 10, 20)
    with pytest.raises(ValueError):
        mannwhitney_power_grid(df.assign(has_effect="a"), x_col="has_effect", y_cols="score")
