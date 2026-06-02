from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from encoder_interface import freeze_encoder  # noqa: E402
from mock_encoder import DeterministicMockEncoder  # noqa: E402


def test_mock_encoder_shape_and_finite_output() -> None:
    encoder = DeterministicMockEncoder()
    waveforms = torch.randn(4, 1, 1250)
    embeddings = encoder.encode(waveforms)
    assert embeddings.shape == (4, 64)
    assert torch.isfinite(embeddings).all()


def test_mock_encoder_invalid_input_shape_error() -> None:
    encoder = DeterministicMockEncoder()
    with pytest.raises(ValueError):
        encoder.encode(torch.randn(4, 1250))


def test_freeze_encoder_sets_requires_grad_false() -> None:
    module = torch.nn.Linear(4, 2)
    freeze_encoder(module)
    assert module.training is False
    assert all(not parameter.requires_grad for parameter in module.parameters())
