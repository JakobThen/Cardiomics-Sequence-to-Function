"""Tests for the transform selection of cardiomics_s2f.eval.streamed_correlation_analysis
and the PyTorch-free use of cardiomics_s2f.io.data_input."""

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("h5py")


@pytest.fixture(scope="module")
def resolve_transform():
    import importlib
    from unittest import mock

    for name in ("pyBigWig",):
        try:
            importlib.import_module(name)
        except ImportError:
            sys.modules[name] = mock.MagicMock()
    from cardiomics_s2f.eval.streamed_correlation_analysis import resolve_transform as fn
    return fn


@pytest.mark.parametrize("model_name, transform, expected", [
    ("Borzoi", None, "borzoi"),
    ("AG_ft_fold1", None, "alphagenome"),
    ("AlphaGenome_pretrained", None, "alphagenome"),
    ("ChromBPNet", None, "none"),
    ("anything", "Borzoi", "borzoi"),
    ("AlphaGenome_pretrained", "none", "none"),   # explicit beats the name
])
def test_resolve_transform(resolve_transform, model_name, transform, expected):
    assert resolve_transform(model_name, transform) == expected


def test_resolve_transform_rejects_unknown(resolve_transform):
    with pytest.raises(ValueError):
        resolve_transform("Borzoi", "log")


def test_data_input_works_without_torch():
    code = textwrap.dedent("""
        import sys, types
        sys.modules["torch"] = None            # makes `import torch` fail
        sys.modules.setdefault("pyBigWig", types.ModuleType("pyBigWig"))
        import numpy as np
        from cardiomics_s2f.io import data_input as di
        assert di.torch is None and not di._is_tensor(np.zeros(2))
        y = np.abs(np.random.default_rng(0).normal(size=(2, 3, 4))) * 50
        t = di.SelectiveSquashTransform(rna_task_indices=[1], task_axis=1, transform="AlphaGenome",
                                        track_means=np.array([1.0, 2.0, 3.0]))
        back = di.SelectiveSquashTransform(rna_task_indices=[1], task_axis=1, transform="AlphaGenome",
                                           track_means=np.array([1.0, 2.0, 3.0]), inverse=True)
        np.testing.assert_allclose(back(t(y)), y, rtol=1e-4)
        np.testing.assert_allclose(di.inverse_borzoi_squash(di.borzoi_squash(y)), y, rtol=1e-4)
        print("ok")
    """)
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "ok"
