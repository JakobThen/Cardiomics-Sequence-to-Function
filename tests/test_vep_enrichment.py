"""Tests for the CHD enrichment analysis, the RF causality classifier and the Richter VEP formatting."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from scipy.stats import fisher_exact

from cardiomics_s2f.vep import causality_classifier as cc
from cardiomics_s2f.vep import enrichment as en
from cardiomics_s2f.vep.plotting import plot_stratified_auprc_heatmap
from cardiomics_s2f.vep.processing import format_pretrained_alphagenome_vep_top_gene


def _patients(n_pat=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_pat):
        is_chd = i < n_pat // 2
        for _ in range(4):
            rows.append(dict(pid=f"P{i}", is_CHD=is_chd, score=rng.normal() + 1.5 * is_chd,
                             cls=rng.choice(["Prom", "Enh"])))
    return pd.DataFrame(rows)


def test_woolf_ci_contains_or_and_handles_zero_cells():
    lo, hi = en._woolf_ci(20, 10, 10, 20)
    assert lo < 4.0 < hi
    lo, hi = en._woolf_ci(0, 5, 50, 40)  # Haldane-Anscombe correction instead of log(0)
    assert np.isfinite(lo) and np.isfinite(hi)


def test_add_bh_fdr_is_monotone():
    res = en._add_bh_fdr(pd.DataFrame({"p_value": [0.04, 0.001, 0.03, 0.5]}), fdr_alpha=0.05)
    assert res["p_value"].is_monotonic_increasing
    assert res["fdr_q"].is_monotonic_increasing
    # p * m / rank = [0.004, 0.06, 0.0533, 0.5], then step-up minimum from the end
    assert res["fdr_q"].tolist() == pytest.approx([0.004, 0.16 / 3, 0.16 / 3, 0.5])
    assert res["significant"].tolist() == [True, False, False, False]


def test_fisher_odds_ratio_ci_matches_scipy():
    y = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    pred = np.array([1, 1, 0, 0, 0, 1, 1, 0])
    res = en.fisher_odds_ratio_ci(y, pred)
    assert res["table"] == [[3, 1], [1, 3]]
    assert res["OR"] == pytest.approx(fisher_exact([[3, 1], [1, 3]])[0])


def test_build_stratum_grid_combines_with_and():
    base = {"a": pd.Series([True, True, False]), "b": pd.Series([True, False, False])}
    grid = en.build_stratum_grid(base, max_depth=2)
    assert list(grid) == ["a", "b", "a & b"]
    assert grid["a & b"].tolist() == [True, False, False]


def test_systematic_stratum_search_finds_enriched_score():
    df = _patients()
    res = en.systematic_stratum_search(df, ["score"], {"All": df.index}, group_col="pid",
                                       min_n=50, n_splits=3, cutoff_methods=("youden",))
    assert len(res) == 1
    assert res.loc[0, "OR"] > 1 and res.loc[0, "significant"]
    # strata below min_n are skipped
    assert en.systematic_stratum_search(df, ["score"], {"All": df.index}, group_col="pid", min_n=10_000).empty


def test_patient_level_enrichment_counts_carriers_once():
    df = pd.DataFrame({"pid": ["A", "A", "B", "C", "D"], "is_CHD": [True, True, True, False, False],
                       "hit": [True, True, False, False, True]})
    res = en.patient_level_enrichment(df, {"hit": df["hit"]}, group_col="pid", min_carriers=1)
    row = res.iloc[0]
    assert (row.carrier_chd, row.carrier_control, row.total_chd, row.total_control) == (1, 1, 2, 2)


def test_add_variant_type():
    df = pd.DataFrame({"REF": ["A", "A", "AT", "AC"], "ALT": ["G", "AT", "A", "GT"]})
    assert cc.add_variant_type(df)["variant_type"].tolist() == ["SNV", "insertion", "deletion", "mnv"]


def test_rf_oof_predictions_and_importances():
    rng = np.random.default_rng(1)
    df = _patients(n_pat=40)
    df["variant_id"] = [f"v{i}" for i in range(len(df))]
    df["CHROM"] = rng.choice(["chr1", "chr2"], len(df))
    df["REF"], df["ALT"] = "A", "G"
    oof, imp, params = cc.train_grouped_oof_rf_classifier(
        df, label_col="is_CHD", group_col="pid", id_col="variant_id", numeric_cols=["score"],
        categorical_cols=["cls"], conditional_cols=[], balance_cols=["CHROM"],
        outer_splits=3, inner_splits=2, n_iter=2, random_state=0)

    assert oof["variant_id"].tolist() == df["variant_id"].tolist()
    assert oof["oof_predicted_proba"].between(0, 1).all()
    assert sorted(imp["fold"].unique()) == [0, 1, 2] and len(params) == 3

    fig, imp_df = cc.plot_rf_feature_importances(imp, ["score"], ["cls"])
    assert set(imp_df["feature"]) == {"score", "cls", "variant_type"}
    assert imp_df["importance"].sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        cc.train_grouped_oof_rf_classifier(df, "is_CHD", "pid", "variant_id", ["score"], ["cls"], [], ["pid"])


def test_stratified_heatmap_auroc():
    df = pd.DataFrame({"y": [0, 0, 1, 1] * 5, "s": [0.1, 0.2, 0.8, 0.9] * 5})
    res = plot_stratified_auprc_heatmap(df, "y", ["s"], {"All": df.index}, metric="AUROC")
    assert res.loc["All", "s"] == 1.0
    with pytest.raises(ValueError):
        plot_stratified_auprc_heatmap(df, "y", ["s"], {"All": df.index}, metric="F1")


def test_format_pretrained_alphagenome_vep_top_gene_keeps_largest_rna_gene():
    marks = ["RNA", "ATAC", "H3K27ac_HIST", "H3K27me3_HIST", "H3K36me3_HIST", "H3K4me1_HIST",
             "H3K4me3_HIST", "H3K9ac_HIST", "H3K9me3_HIST"]
    rows = []
    for gene, rna in (("G1", 0.1), ("G2", -0.9)):
        base = dict(variant_name="chr1-5-A-G", gene_id=f"E_{gene}", gene_name=gene, CHROM="chr1", POS=5,
                    REF="A", ALT="G", **{f"{m}_raw_score": 1.0 for m in marks},
                    **{f"{m}_quantile_score": 0.5 for m in marks})
        base["RNA_raw_score"] = rna
        for tf, score in (("CTCF", 0.2), ("POLR2A", 0.3)):
            rows.append({**base, "transcription_factor": tf, "TF_raw_score": score, "TF_quantile_score": 0.1})

    out = format_pretrained_alphagenome_vep_top_gene(pd.DataFrame(rows))

    assert len(out) == 1
    row = out.iloc[0]
    assert (row.variant_id, row.AG_pre_Gene, row.AG_pre_RNA_raw_score) == ("chr1-5-A-G", "G2", -0.9)
    assert (row.AG_pre_CTCF_TF_raw_score, row.AG_pre_POLR2A_TF_raw_score) == (0.2, 0.3)
    assert not any("quantile" in c for c in out.columns)


def test_plot_top_strata_forest_skips_or_outside_ci(capsys):
    res = pd.DataFrame({"stratum": ["a", "b", "c"], "score": "s", "cutoff_method": "youden",
                        "OR": [2.0, 0.0, np.inf], "ci_low": [1.2, 0.004, 0.5], "ci_high": [3.0, 1.4, 160.0],
                        "p_value": [0.01, 0.02, 0.07], "fdr_q": [0.03, 0.03, 0.07],
                        "stratum_n": [100, 100, 100], "n_pos": [50, 0, 5]})
    fig, ax = en.plot_top_strata_forest(res)
    assert len(ax.get_yticks()) == 1
    assert "Skipped 2 row(s)" in capsys.readouterr().out
