"""Tests for cardiomics_s2f.utils."""

import time

import pytest

from cardiomics_s2f.utils import log_stage_resources


def test_log_stage_resources(capsys):
    pytest.importorskip("psutil")
    start = time.time() - 3725
    t = log_stage_resources("stage", start_time=start)
    assert isinstance(t, float)
    out = capsys.readouterr().out
    assert "[PROFILE - stage]" in out
    assert "Time Elapsed: 01:02:05" in out
