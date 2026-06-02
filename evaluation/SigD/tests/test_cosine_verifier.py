from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import cosine_verifier  # noqa: E402
from cosine_verifier import aggregate_template, cosine_score, l2_normalize  # noqa: E402


def test_identical_embedding_cosine_is_one() -> None:
    vector = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    assert abs(cosine_score(vector, vector) - 1.0) < 1.0e-6


def test_orthogonal_embedding_cosine_is_zero() -> None:
    a = np.asarray([1.0, 0.0], dtype=np.float32)
    b = np.asarray([0.0, 1.0], dtype=np.float32)
    assert abs(cosine_score(a, b)) < 1.0e-6


def test_k5_aggregation_shape_and_normalization() -> None:
    embeddings = np.eye(5, 8, dtype=np.float32)
    template = aggregate_template(embeddings)
    assert template.shape == (8,)
    assert abs(float(np.linalg.norm(template)) - 1.0) < 1.0e-6


def test_score_range() -> None:
    score = cosine_score(np.asarray([1.0, 1.0]), np.asarray([-1.0, -1.0]))
    assert -1.0 <= score <= 1.0


def test_cosine_verifier_zero_norm_embedding_rejected() -> None:
    with pytest.raises(ValueError, match="zero or near-zero"):
        l2_normalize(np.zeros(4, dtype=np.float32))


def test_cosine_verifier_nonfinite_embedding_rejected() -> None:
    with pytest.raises(ValueError, match="nonfinite"):
        l2_normalize(np.asarray([1.0, np.nan], dtype=np.float32))


def test_cosine_verifier_score_clipped_only_within_tolerance(monkeypatch: pytest.MonkeyPatch) -> None:
    values = [
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.asarray([1.0 + 5.0e-7, 0.0], dtype=np.float32),
    ]

    def fake_normalize(vector, eps=1.0e-8):
        return values.pop(0)

    monkeypatch.setattr(cosine_verifier, "l2_normalize", fake_normalize)
    assert cosine_score(np.asarray([1.0]), np.asarray([1.0])) == 1.0


def test_cosine_verifier_excessive_out_of_range_score_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    values = [
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.asarray([1.0001, 0.0], dtype=np.float32),
    ]

    def fake_normalize(vector, eps=1.0e-8):
        return values.pop(0)

    monkeypatch.setattr(cosine_verifier, "l2_normalize", fake_normalize)
    with pytest.raises(ValueError, match="outside valid range"):
        cosine_score(np.asarray([1.0]), np.asarray([1.0]))
