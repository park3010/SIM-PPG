from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from session_alignment_loss import SessionCentroidAlignmentLoss  # noqa: E402


def ids(subjects: int = 8):
    subject_ids = []
    session_ids = []
    for subject in range(subjects):
        for session in ("s1", "s2"):
            for _ in range(2):
                subject_ids.append(f"p{subject}")
                session_ids.append(session)
    return subject_ids, session_ids


def test_cross_session_batch_alignment_loss_finite() -> None:
    subject_ids, session_ids = ids()
    embeddings = F.normalize(torch.randn(32, 128), dim=1)
    loss, diagnostics = SessionCentroidAlignmentLoss()(embeddings, subject_ids, session_ids, return_diagnostics=True)
    assert torch.isfinite(loss).item()
    assert diagnostics["subject_count"] == 8
    assert diagnostics["centroid_pair_count"] == 8


def test_missing_second_session_raises() -> None:
    embeddings = F.normalize(torch.randn(4, 128), dim=1)
    subject_ids = ["p0"] * 4
    session_ids = ["s1"] * 4
    with pytest.raises(ValueError, match="exactly 2 sessions"):
        SessionCentroidAlignmentLoss()(embeddings, subject_ids, session_ids)


def test_unexpected_session_sample_count_raises() -> None:
    embeddings = F.normalize(torch.randn(5, 128), dim=1)
    subject_ids = ["p0"] * 5
    session_ids = ["s1", "s1", "s1", "s2", "s2"]
    with pytest.raises(ValueError, match="exactly 2 samples"):
        SessionCentroidAlignmentLoss()(embeddings, subject_ids, session_ids)


def test_zero_centroid_raises() -> None:
    embeddings = torch.zeros(4, 2)
    subject_ids = ["p0"] * 4
    session_ids = ["s1", "s1", "s2", "s2"]
    with pytest.raises(ValueError, match="near-zero"):
        SessionCentroidAlignmentLoss()(embeddings, subject_ids, session_ids)


def test_identical_centroids_loss_near_zero() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    subject_ids = ["p0"] * 4
    session_ids = ["s1", "s1", "s2", "s2"]
    loss = SessionCentroidAlignmentLoss()(embeddings, subject_ids, session_ids)
    assert float(loss) == pytest.approx(0.0, abs=1.0e-6)


def test_orthogonal_centroids_loss_near_one() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    subject_ids = ["p0"] * 4
    session_ids = ["s1", "s1", "s2", "s2"]
    loss = SessionCentroidAlignmentLoss()(embeddings, subject_ids, session_ids)
    assert float(loss) == pytest.approx(1.0, abs=1.0e-6)

