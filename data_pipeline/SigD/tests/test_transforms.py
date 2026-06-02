from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from transforms import PerWindowZScore  # noqa: E402


def test_zscore_output_shape_dtype_mean_std() -> None:
    x = np.linspace(-2, 2, 1250, dtype=np.float32)
    y = PerWindowZScore()(x)
    assert tuple(y.shape) == (1, 1250)
    assert y.dtype == torch.float32
    assert abs(float(y.mean())) < 1.0e-6
    assert abs(float(y.std(unbiased=False)) - 1.0) < 1.0e-5


def test_source_array_not_modified() -> None:
    x = np.random.default_rng(42).normal(size=1250).astype(np.float32)
    original = x.copy()
    _ = PerWindowZScore()(x)
    assert np.array_equal(x, original)


def test_nonfinite_input_raises() -> None:
    x = np.ones(1250, dtype=np.float32)
    x[10] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        PerWindowZScore()(x)
