"""Tests for cardiomics_s2f.finetuning.borzoi.training and .intervals."""

import warnings
from types import SimpleNamespace

import pandas as pd
import pytest

from cardiomics_s2f.finetuning.borzoi.intervals import drop_intervals_beyond_chrom_ends
from cardiomics_s2f.finetuning.borzoi.training import (
    find_normalized_bigwig_tracks,
    tracks_from_checkpoint,
    unfreeze_embedding_layers,
    warn_on_deduplicated_folder_pairs,
)


def _touch(folder, *names):
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"")


@pytest.fixture
def staged(tmp_path):
    _touch(tmp_path / "multiome_RNA_3prime_bigwig", "RNA_b_plus_100M.bw", "RNA_a_minus_100M.bw", "RNA_a_minus_raw.bw")
    _touch(tmp_path / "CutnTag_bigwig", "K4Day1_100M_norm.bw")
    _touch(tmp_path / "metadata", "ignored_100M.bw")
    return tmp_path


def test_find_tracks_sorted_and_filtered(staged):
    tracks = find_normalized_bigwig_tracks(staged)
    assert tracks["file_name"].tolist() == ["K4Day1_100M_norm.bw", "RNA_a_minus_100M.bw", "RNA_b_plus_100M.bw"]
    assert tracks["modality"].tolist() == ["CutnTag", "multiome_RNA_3prime", "multiome_RNA_3prime"]


def test_find_tracks_selected_folders(staged):
    tracks = find_normalized_bigwig_tracks(staged, folders=["CutnTag_bigwig"])
    assert tracks["file_name"].tolist() == ["K4Day1_100M_norm.bw"]
    with pytest.raises(FileNotFoundError):
        find_normalized_bigwig_tracks(staged, folders=["missing_bigwig"])


def test_find_tracks_empty_raises(tmp_path):
    _touch(tmp_path / "ATAC_bigwig", "raw.bw")
    with pytest.raises(ValueError):
        find_normalized_bigwig_tracks(tmp_path)


def test_deduplicated_pair_warning():
    with pytest.warns(UserWarning, match="multiome_ATAC_bigwig_deduplicated"):
        pairs = warn_on_deduplicated_folder_pairs(
            ["multiome_ATAC_bigwig", "multiome_ATAC_bigwig_deduplicated", "CutnTag_bigwig"]
        )
    assert pairs == [("multiome_ATAC_bigwig", "multiome_ATAC_bigwig_deduplicated")]


def test_no_warning_for_single_versions():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert warn_on_deduplicated_folder_pairs(
            ["multiome_ATAC_bigwig_deduplicated", "multiome_RNA_3prime_bigwig"]
        ) == []


def _model(task_names):
    return SimpleNamespace(data_params={"tasks": {"name": task_names}})


def test_tracks_from_checkpoint_keeps_checkpoint_order(staged):
    order = ["RNA_b_plus_100M.bw", "K4Day1_100M_norm.bw", "RNA_a_minus_100M.bw"]
    tracks = tracks_from_checkpoint(_model(order), staged)
    assert tracks["file_name"].tolist() == order
    assert tracks["modality"].tolist() == ["multiome_RNA_3prime", "CutnTag", "multiome_RNA_3prime"]


def test_tracks_from_checkpoint_missing(staged):
    with pytest.raises(ValueError, match="Missing: \\['gone.bw'\\]"):
        tracks_from_checkpoint(_model(["gone.bw"]), staged)


def test_tracks_from_checkpoint_ambiguous_warns_and_raises(tmp_path):
    _touch(tmp_path / "multiome_ATAC_bigwig", "ATAC_CM_100M_norm.bw")
    _touch(tmp_path / "multiome_ATAC_bigwig_deduplicated", "ATAC_CM_100M_norm.bw")
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match="several folders"):
            tracks_from_checkpoint(_model(["ATAC_CM_100M_norm.bw"]), tmp_path)
    # restricting the search to one folder resolves it
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tracks =tracks_from_checkpoint(_model(["ATAC_CM_100M_norm.bw"]), tmp_path,
                                        folders=["multiome_ATAC_bigwig_deduplicated"])
    assert tracks["modality"].tolist() == ["multiome_ATAC"]


def test_unfreeze_embedding_layers():
    torch = pytest.importorskip("torch")
    embedding = torch.nn.ModuleDict({
        "unet_tower": torch.nn.Linear(2, 2),
        "transformer_tower": torch.nn.Linear(2, 2),
    })
    model = SimpleNamespace(model=SimpleNamespace(embedding=embedding))
    unfreeze_embedding_layers(model, ["unet_tower"])
    grads = {n: p.requires_grad for n, p in embedding.named_parameters()}
    assert grads == {
        "unet_tower.weight": True, "unet_tower.bias": True,
        "transformer_tower.weight": False, "transformer_tower.bias": False,
    }
    # empty list leaves the flags untouched
    unfreeze_embedding_layers(model, [])
    assert {n: p.requires_grad for n, p in embedding.named_parameters()} == grads


def test_drop_intervals_beyond_chrom_ends():
    intervals = pd.DataFrame({
        "chrom": ["chr1", "chr1", "chr1", "chrY"],
        "start": [0, 90, -5, 10],
        "end": [100, 101, 50, 20],
    })
    sizes = pd.DataFrame({"chrom": ["chr1"], "end": [100]})
    kept = drop_intervals_beyond_chrom_ends(intervals, sizes)
    assert kept.index.tolist() == [0]
    assert list(kept.columns) == ["chrom", "start", "end"]


def test_find_checkpoint_file(tmp_path):
    from cardiomics_s2f.finetuning.borzoi.training import find_checkpoint_file

    ckpt_dir = tmp_path / "run" / "2026_01_04_17_59" / "version_0" / "checkpoints"
    _touch(ckpt_dir, "epoch=7-step=100.ckpt", "last.ckpt")
    best = ckpt_dir / "epoch=7-step=100.ckpt"
    assert find_checkpoint_file(tmp_path / "run") == best
    assert find_checkpoint_file(best) == best
    _touch(tmp_path / "run" / "2026_02_01_10_00" / "version_0" / "checkpoints", "epoch=3-step=40.ckpt")
    with pytest.raises(ValueError, match="Several"):
        find_checkpoint_file(tmp_path / "run")
    with pytest.raises(FileNotFoundError):
        find_checkpoint_file(tmp_path / "empty_or_missing")


def test_write_borzoi_fold_intervals(tmp_path):
    from cardiomics_s2f.finetuning.borzoi.intervals import write_borzoi_fold_intervals

    df = pd.DataFrame({"chrom": ["chr5"], "start": [41462305], "end": [41986593]})
    paths = write_borzoi_fold_intervals({"train": df, "val": df, "test": df}, tmp_path)
    assert paths["test"].name == "Borzoi_human_rep0_test_intervals.bed"
    assert paths["test"].read_bytes() == b"chrom\tstart\tend\nchr5\t41462305\t41986593\n"
    with pytest.raises(FileExistsError):
        write_borzoi_fold_intervals({"train": df, "val": df, "test": df}, tmp_path)
    write_borzoi_fold_intervals({"train": df, "val": df, "test": df}, tmp_path, overwrite=True)


def test_resolve_staged_track_paths(tmp_path):
    from cardiomics_s2f.finetuning.borzoi.training import resolve_staged_track_paths

    _touch(tmp_path / "multiome_ATAC_bigwig", "ATAC_CM_100M_norm.bw")
    _touch(tmp_path / "multiome_ATAC_bigwig_deduplicated", "ATAC_CM_100M_norm.bw")
    tracks = pd.DataFrame({
        "modality": ["multiome_ATAC"],
        "file_name": ["ATAC_CM_100M_norm.bw"],
        "path": ["/tmp/borzoi_bws_123/multiome_ATAC_bigwig_deduplicated/ATAC_CM_100M_norm.bw"],
    })
    out = resolve_staged_track_paths(tracks, tmp_path)
    assert out["path"].tolist() == [str(tmp_path / "multiome_ATAC_bigwig_deduplicated" / "ATAC_CM_100M_norm.bw")]
    assert out["staged_path"].tolist() == tracks["path"].tolist()
    with pytest.raises(FileNotFoundError):
        resolve_staged_track_paths(tracks.assign(file_name=["gone.bw"]), tmp_path)
