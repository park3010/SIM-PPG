from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from positive_masks import build_generic_supcon_positive_mask  # noqa: E402
from supervised_contrastive_loss import supervised_contrastive_loss  # noqa: E402


def test_supcon_loss_finite() -> None:
    embeddings = F.normalize(torch.randn(8, 128), dim=1)
    subjects = ["a"] * 4 + ["b"] * 4
    sessions = ["s1", "s1", "s2", "s2"] * 2
    mask = build_generic_supcon_positive_mask(subjects, sessions)
    loss = supervised_contrastive_loss(embeddings, mask)
    assert torch.isfinite(loss).item()


def test_separated_negatives_reduce_loss() -> None:
    subjects = ["a", "a", "b", "b"]
    sessions = ["s1", "s2", "s1", "s2"]
    mask = build_generic_supcon_positive_mask(subjects, sessions)
    good = F.normalize(torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]]), dim=1)
    bad = F.normalize(torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.9, 0.1], [-0.9, -0.1]]), dim=1)
    assert supervised_contrastive_loss(good, mask) < supervised_contrastive_loss(bad, mask)


def test_positive_less_anchor_raises() -> None:
    embeddings = F.normalize(torch.randn(4, 128), dim=1)
    mask = torch.zeros((4, 4), dtype=torch.bool)
    with pytest.raises(ValueError, match="at least one positive"):
        supervised_contrastive_loss(embeddings, mask)


def test_temperature_validation() -> None:
    embeddings = F.normalize(torch.randn(4, 128), dim=1)
    subjects = ["a", "a", "b", "b"]
    sessions = ["s1", "s2", "s1", "s2"]
    mask = build_generic_supcon_positive_mask(subjects, sessions)
    with pytest.raises(ValueError, match="temperature"):
        supervised_contrastive_loss(embeddings, mask, temperature=0.0)


def test_backward_possible() -> None:
    embeddings = F.normalize(torch.randn(8, 128), dim=1).requires_grad_(True)
    subjects = ["a"] * 4 + ["b"] * 4
    sessions = ["s1", "s1", "s2", "s2"] * 2
    mask = build_generic_supcon_positive_mask(subjects, sessions)
    loss = supervised_contrastive_loss(embeddings, mask)
    loss.backward()
    assert embeddings.grad is not None
