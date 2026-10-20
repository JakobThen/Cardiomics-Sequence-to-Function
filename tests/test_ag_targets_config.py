"""Tests for cardiomics_s2f.preprocessing.ag_targets_config."""

import pandas as pd
import pytest
import yaml

from cardiomics_s2f.preprocessing.ag_targets_config import (
    build_targets_config,
    infer_modality,
    write_targets_config,
)


@pytest.mark.parametrize("folder, expected", [
    ("multiome_ATAC_bigwig", ("multiome_ATAC", "atac")),
    ("multiome_RNA_3prime_bigwig", ("multiome_RNA", "rna_seq")),
    ("CutnTag_bigwig", ("CutnTag", "atac")),
    ("cut&tag", ("CutnTag", "atac")),
])
def test_infer_modality(folder, expected):
    assert infer_modality(folder) == expected


def test_infer_modality_rejects_unknown():
    with pytest.raises(ValueError, match="ChIP"):
        infer_modality("ChIP_bigwig")


@pytest.fixture
def bigwig_dir(tmp_path):
    """Two modalities with empty .bw files and a matching non-zero means table."""
    layout = {
        "multiome_ATAC_bigwig": ["ATAC_FB_100M_norm.bw", "ATAC_CM_100M_norm.bw", "ATAC_CM_raw.bw"],
        "multiome_RNA_3prime_bigwig": ["RNA_CM_100M_norm.bw"],
    }
    rows = []
    for folder, files in layout.items():
        (tmp_path / folder).mkdir()
        for i, name in enumerate(files):
            (tmp_path / folder / name).write_bytes(b"")
            rows.append({"full_path": f"{tmp_path / folder}/{name}", "nonzero_mean_canonical": 0.12345 + i})
    return tmp_path, pd.DataFrame(rows)


def test_build_targets_config(bigwig_dir):
    data_dir, means = bigwig_dir
    config = build_targets_config(data_dir, ["multiome_ATAC_bigwig", "multiome_RNA_3prime_bigwig"], means)

    atac, rna = config["heads"]
    assert (atac["id"], atac["kind"], atac["dir_name"]) == ("multiome_ATAC", "atac", "multiome_ATAC_bigwig")
    assert [t["label"] for t in atac["targets"]] == ["ATAC_CM_100M_norm", "ATAC_CM_raw", "ATAC_FB_100M_norm"]
    assert atac["targets"][0]["nonzero_mean"] == round(0.12345 + 1, 3)
    assert rna["kind"] == "rna_seq" and len(rna["targets"]) == 1


def test_norm_tag_filters_and_strips_labels(bigwig_dir):
    data_dir, means = bigwig_dir
    config = build_targets_config(data_dir, ["multiome_ATAC_bigwig"], means, norm_tag="_100M_norm")
    assert [t["label"] for t in config["heads"][0]["targets"]] == ["ATAC_CM", "ATAC_FB"]


def test_missing_nonzero_mean_raises(bigwig_dir):
    data_dir, means = bigwig_dir
    with pytest.raises(AssertionError, match="not found in nonzero track means"):
        build_targets_config(data_dir, ["multiome_ATAC_bigwig"], means.iloc[1:])


def test_missing_folder_gives_empty_head(bigwig_dir):
    data_dir, means = bigwig_dir
    config = build_targets_config(data_dir, ["CutnTag_bigwig"], means)
    assert config["heads"][0]["targets"] == []


def test_write_targets_config_roundtrip(tmp_path, bigwig_dir):
    data_dir, means = bigwig_dir
    config = build_targets_config(data_dir, ["multiome_RNA_3prime_bigwig"], means)
    out = tmp_path / "nested" / "config.yaml"
    write_targets_config(config, out)
    assert yaml.safe_load(out.read_text()) == config
    # Key order is preserved (sort_keys=False)
    assert list(yaml.safe_load(out.read_text())["heads"][0]) == ["dir_name", "id", "kind", "targets"]
