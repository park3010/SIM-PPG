from __future__ import annotations

from pathlib import Path
import sys

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from thresholds import apply_threshold, compute_eer_threshold, compute_far_target_threshold  # noqa: E402


def test_validation_eer_threshold_computable() -> None:
    result = compute_eer_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert result["eer"] == 0.0
    assert result["far"] == 0.0
    assert result["frr"] == 0.0


def test_far_target_threshold_computable() -> None:
    result = compute_far_target_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], target_far=0.01)
    assert result["validation_far"] <= 0.01
    assert result["validation_tar"] == 1.0


def test_apply_provided_threshold_without_selection() -> None:
    metrics = apply_threshold([0, 1], [0.2, 0.8], threshold=0.5)
    assert metrics["far"] == 0.0
    assert metrics["frr"] == 0.0
    assert metrics["tar"] == 1.0


def test_single_class_threshold_selection_raises() -> None:
    with pytest.raises(ValueError):
        compute_eer_threshold([1, 1], [0.8, 0.9])
