"""Tests for cardiomics_s2f.io.data_output."""

import h5py
import numpy as np
import pandas as pd

from cardiomics_s2f.io.data_output import HDF5BatchWriter, save_borzoi_predictions_with_fallback

INTERVALS = pd.DataFrame({"chrom": ["chr1", "chr2", "chr3"], "start": [0, 10, 20], "end": [5, 15, 25]})
TRACKS = pd.DataFrame({"label": ["ATAC_CM", "ATAC_FB"], "nonzero_mean": [0.5, 1.5]})


def test_batch_writer_assembles_batches(tmp_path):
    preds = np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4)
    writer = HDF5BatchWriter(tmp_path / "out", "eval", total_samples=3)
    writer.write_batch("ATAC", preds[:2], start_idx=0)
    writer.write_batch("ATAC", preds[2:], start_idx=2)
    writer.write_metadata(INTERVALS, TRACKS)
    writer.close()

    with h5py.File(tmp_path / "out" / "eval.h5", "r") as f:
        np.testing.assert_array_equal(f["ATAC/predictions"][()], preds)
        assert f["intervals/chrom"][()].tolist() == [b"chr1", b"chr2", b"chr3"]
        np.testing.assert_array_equal(f["intervals/start"][()], [0, 10, 20])
        assert f["tracks/label"][()].tolist() == [b"ATAC_CM", b"ATAC_FB"]


def test_save_predictions_writes_annotated_h5(tmp_path):
    preds = np.ones((3, 2, 4), dtype=np.float32)
    save_borzoi_predictions_with_fallback(preds, tmp_path, "borzoi", INTERVALS, TRACKS)
    with h5py.File(tmp_path / "borzoi.h5", "r") as f:
        np.testing.assert_array_equal(f["predictions"][()], preds)
        np.testing.assert_array_equal(f["tracks/nonzero_mean"][()], [0.5, 1.5])
    assert not (tmp_path / "borzoi.npy").exists()
