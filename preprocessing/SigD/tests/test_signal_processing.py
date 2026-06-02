from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from signal_processing import (  # noqa: E402
    filter_ppg,
    interpolate_nonfinite,
    iter_window_slices,
    model_input_validity,
    resample_to_target,
)


def config() -> dict:
    return {
        "papagei_alignment": {
            "filter_backend": "scipy_test",
            "filter_lowcut_hz": 0.5,
            "filter_highcut_hz": 12,
            "filter_order": 4,
        },
        "nonfinite_handling": {
            "max_window_original_nonfinite_ratio_for_model_input": 0.05,
        },
        "hard_validity": {
            "minimum_finite_std": 1.0e-8,
        },
    }


def test_125hz_10s_signal_keeps_1250_samples() -> None:
    fs = 125
    x = np.sin(2 * np.pi * 1.2 * np.arange(1250) / fs).astype(np.float32)
    y = filter_ppg(x, fs, config())
    out, resampled = resample_to_target(y, fs, 125, 10)
    assert out.shape == (1250,)
    assert resampled is False


def test_raw_ranges_are_not_concatenated_and_remainder_is_recorded() -> None:
    slices, remainder = iter_window_slices(total_samples=1250 + 300, fs=125, window_seconds=10)
    assert slices == [(0, 1250)]
    assert remainder == 300


def test_fs_mismatch_resamples_to_125hz() -> None:
    fs = 100
    x = np.sin(2 * np.pi * 1.2 * np.arange(1000) / fs)
    out, resampled = resample_to_target(x, fs, 125, 10)
    assert out.shape == (1250,)
    assert resampled is True


def test_partial_nan_input_is_interpolated() -> None:
    x = np.arange(10, dtype=np.float32)
    x[2:4] = np.nan
    repaired, mask, changed = interpolate_nonfinite(x)
    assert changed is True
    assert mask.sum() == 2
    assert np.all(np.isfinite(repaired))


def test_all_nan_input_fails() -> None:
    try:
        interpolate_nonfinite(np.full(10, np.nan, dtype=np.float32))
    except ValueError as exc:
        assert str(exc) == "all_nonfinite"
    else:
        raise AssertionError("all-NaN input should fail")


def test_low_sqi_or_flatline_is_not_a_hard_rejection_reason() -> None:
    x = np.sin(2 * np.pi * 1.2 * np.arange(1250) / 125).astype(np.float32)
    valid, reason = model_input_validity(x, 0.0, 1250, config())
    assert valid is True
    assert reason == ""


def test_nonfinite_filtered_output_is_rejected() -> None:
    x = np.ones(1250, dtype=np.float32)
    x[4] = np.nan
    valid, reason = model_input_validity(x, 0.0, 1250, config())
    assert valid is False
    assert reason == "filtered_output_nonfinite"

