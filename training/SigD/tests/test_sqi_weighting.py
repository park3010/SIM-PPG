from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from sqi_weighting import compute_sqi_weights  # noqa: E402


def sample():
    sqi = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, float("nan")])
    mask = torch.tensor([True, True, True, True, True, False])
    return sqi, mask


def test_mild_linear_range() -> None:
    sqi, mask = sample()
    weights, _ = compute_sqi_weights(sqi, mask, "mild_linear")
    assert float(weights.min()) >= 0.5
    assert float(weights.max()) <= 1.0


def test_clipped_linear_range() -> None:
    sqi, mask = sample()
    weights, _ = compute_sqi_weights(sqi, mask, "clipped_linear")
    assert float(weights.min()) >= 0.5
    assert float(weights.max()) <= 1.0


def test_strong_linear_range() -> None:
    sqi, mask = sample()
    weights, _ = compute_sqi_weights(sqi, mask, "strong_linear")
    assert float(weights.min()) >= 0.25
    assert float(weights.max()) <= 1.0


def test_rank_bottom20_downweights_expected_sample() -> None:
    sqi, mask = sample()
    weights, _ = compute_sqi_weights(sqi, mask, "rank_bottom20_downweight")
    assert weights[0].item() == 0.5
    assert weights[1:5].eq(1.0).all().item()


def test_invalid_samples_neutral() -> None:
    sqi, mask = sample()
    weights, _ = compute_sqi_weights(sqi, mask, "mild_linear")
    assert weights[-1].item() == 1.0


def test_all_invalid_returns_ones() -> None:
    sqi, mask = sample()
    weights, diagnostics = compute_sqi_weights(sqi, torch.zeros_like(mask), "mild_linear")
    assert torch.allclose(weights, torch.ones_like(weights))
    assert diagnostics["valid_count"] == 0


def test_constant_sqi_stable() -> None:
    sqi = torch.ones(8)
    mask = torch.ones(8, dtype=torch.bool)
    weights, _ = compute_sqi_weights(sqi, mask, "strong_linear")
    assert torch.allclose(weights, torch.ones_like(weights))


def test_unsupported_mode_error() -> None:
    sqi, mask = sample()
    with pytest.raises(ValueError, match="Unsupported"):
        compute_sqi_weights(sqi, mask, "not_a_mode")

