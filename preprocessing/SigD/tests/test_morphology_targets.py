from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from morphology_targets import compute_morphology_targets, compute_ipa, extract_svri, skewness_sqi  # noqa: E402


def synthetic_ppg(fs: int = 125, seconds: int = 10) -> np.ndarray:
    t = np.arange(fs * seconds) / fs
    return (0.7 * np.sin(2 * np.pi * 1.2 * t) + 0.2 * np.sin(2 * np.pi * 2.4 * t)).astype(np.float32)


def test_normal_synthetic_ppg_produces_svri_and_sqi() -> None:
    targets = compute_morphology_targets(synthetic_ppg(), 125)
    assert targets["svri_valid_mask"] is True
    assert targets["sqi_valid_mask"] is True
    assert np.isfinite(targets["svri"])
    assert np.isfinite(targets["sqi_skewness"])


def test_constant_signal_marks_svri_invalid() -> None:
    result = extract_svri(np.ones(1250, dtype=np.float32))
    assert result["svri_valid_mask"] is False
    assert result["svri_failure_reason"] == "constant_signal"


def test_constant_signal_marks_sqi_invalid() -> None:
    result = skewness_sqi(np.ones(1250, dtype=np.float32))
    assert result["sqi_valid_mask"] is False


def test_ipa_extrema_failure_is_masked() -> None:
    result = compute_ipa(np.linspace(0, 1, 1250, dtype=np.float32), 125)
    assert result["ipa_valid_mask"] is False
    assert result["ipa_failure_reason"]


def test_ipa_failure_does_not_imply_window_removal() -> None:
    targets = compute_morphology_targets(np.linspace(0, 1, 1250, dtype=np.float32), 125)
    assert targets["ipa_valid_mask"] is False
    assert "model_input_available" not in targets

