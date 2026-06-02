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
from weighted_supcon_loss import weighted_supervised_contrastive_loss  # noqa: E402


def ids():
    subjects = ["a", "a", "b", "b"]
    sessions = ["s1", "s2", "s1", "s2"]
    return subjects, sessions


def test_all_ones_equals_unweighted() -> None:
    subjects, sessions = ids()
    embeddings = F.normalize(torch.randn(4, 8), dim=1)
    mask = build_generic_supcon_positive_mask(subjects, sessions)
    weighted = weighted_supervised_contrastive_loss(embeddings, mask, torch.ones(4))
    unweighted = supervised_contrastive_loss(embeddings, mask)
    assert float(weighted.detach()) == pytest.approx(float(unweighted.detach()), abs=1.0e-6)


def test_nonuniform_weights_change_loss() -> None:
    subjects, sessions = ids()
    embeddings = F.normalize(torch.randn(4, 8), dim=1)
    mask = build_generic_supcon_positive_mask(subjects, sessions)
    weighted = weighted_supervised_contrastive_loss(embeddings, mask, torch.tensor([1.0, 0.1, 1.0, 0.1]))
    unweighted = supervised_contrastive_loss(embeddings, mask)
    assert abs(float(weighted.detach()) - float(unweighted.detach())) > 1.0e-8


def test_negative_weights_error() -> None:
    subjects, sessions = ids()
    embeddings = F.normalize(torch.randn(4, 8), dim=1)
    mask = build_generic_supcon_positive_mask(subjects, sessions)
    with pytest.raises(ValueError, match="nonnegative"):
        weighted_supervised_contrastive_loss(embeddings, mask, torch.tensor([1.0, -1.0, 1.0, 1.0]))


def test_nonfinite_weights_error() -> None:
    subjects, sessions = ids()
    embeddings = F.normalize(torch.randn(4, 8), dim=1)
    mask = build_generic_supcon_positive_mask(subjects, sessions)
    with pytest.raises(ValueError, match="nonfinite"):
        weighted_supervised_contrastive_loss(embeddings, mask, torch.tensor([1.0, float("nan"), 1.0, 1.0]))


def test_positive_less_anchor_error() -> None:
    embeddings = F.normalize(torch.randn(4, 8), dim=1)
    mask = torch.zeros((4, 4), dtype=torch.bool)
    with pytest.raises(ValueError, match="at least one positive"):
        weighted_supervised_contrastive_loss(embeddings, mask, torch.ones(4))

