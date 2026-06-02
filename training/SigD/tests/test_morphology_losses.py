from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from morphology_losses import compute_morphology_losses, masked_mse_loss  # noqa: E402


def test_masked_mse_uses_only_valid_samples() -> None:
    pred = torch.tensor([1.0, 10.0, 3.0])
    target = torch.tensor([2.0, 999.0, 1.0])
    mask = torch.tensor([True, False, True])
    loss, count = masked_mse_loss(pred, target, mask)
    assert count == 2
    assert float(loss) == pytest.approx(((1 - 2) ** 2 + (3 - 1) ** 2) / 2)


def test_valid_count_zero_returns_zero_loss() -> None:
    pred = torch.tensor([1.0, 2.0], requires_grad=True)
    loss, count = masked_mse_loss(pred, torch.tensor([float("nan"), float("nan")]), torch.tensor([False, False]))
    loss.backward()
    assert count == 0
    assert float(loss.detach()) == 0.0
    assert pred.grad is not None


def test_nonfinite_valid_target_raises() -> None:
    with pytest.raises(ValueError, match="nonfinite"):
        masked_mse_loss(torch.tensor([1.0]), torch.tensor([float("nan")]), torch.tensor([True]))


def test_compute_morphology_losses_ignores_ipa() -> None:
    pred = {"svri_pred": torch.tensor([0.0, 1.0]), "sqi_pred": torch.tensor([1.0, 2.0])}
    batch = {
        "svri": torch.tensor([0.0, 2.0]),
        "sqi": torch.tensor([1.0, 4.0]),
        "svri_valid_mask": torch.tensor([True, False]),
        "sqi_valid_mask": torch.tensor([True, True]),
        "ipa": torch.tensor([float("nan"), float("nan")]),
        "ipa_valid_mask": torch.tensor([False, False]),
    }
    cfg = {"loss_components": {"lambda_svri": 0.1, "lambda_sqi": 0.2, "use_ipa": False}}
    total, diagnostics = compute_morphology_losses(pred, batch, cfg)
    assert torch.isfinite(total).item()
    assert diagnostics["svri_valid_count"] == 1
    assert diagnostics["sqi_valid_count"] == 2
    assert diagnostics["ipa_used"] is False

