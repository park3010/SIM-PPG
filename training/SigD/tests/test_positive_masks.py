from __future__ import annotations

from pathlib import Path
import sys

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from positive_masks import (  # noqa: E402
    build_cross_session_positive_mask,
    build_generic_supcon_positive_mask,
    build_negative_mask,
    validate_positive_mask,
)


def batch_ids():
    subjects = []
    sessions = []
    for subject in range(8):
        for session in ("s1", "s2"):
            for _ in range(2):
                subjects.append(f"p{subject}")
                sessions.append(session)
    return subjects, sessions


def test_generic_positive_count_is_three_per_anchor() -> None:
    subjects, sessions = batch_ids()
    mask = build_generic_supcon_positive_mask(subjects, sessions)
    assert set(mask.sum(dim=1).tolist()) == {3}
    assert validate_positive_mask(mask)["valid"] is True


def test_cross_session_positive_count_is_two_per_anchor() -> None:
    subjects, sessions = batch_ids()
    mask = build_cross_session_positive_mask(subjects, sessions)
    assert set(mask.sum(dim=1).tolist()) == {2}
    assert validate_positive_mask(mask)["valid"] is True


def test_cross_session_excludes_same_session_positive() -> None:
    subjects, sessions = batch_ids()
    mask = build_cross_session_positive_mask(subjects, sessions)
    assert mask[0, 1].item() is False
    assert mask[0, 2].item() is True
    assert mask[0, 3].item() is True


def test_negative_mask_is_different_subject_only() -> None:
    subjects, _ = batch_ids()
    negative = build_negative_mask(subjects)
    assert negative[0, 4].item() is True
    assert negative[0, 1].item() is False


def test_diagonal_always_false() -> None:
    subjects, sessions = batch_ids()
    for mask in (
        build_generic_supcon_positive_mask(subjects, sessions),
        build_cross_session_positive_mask(subjects, sessions),
        build_negative_mask(subjects),
    ):
        assert not torch.diagonal(mask).any().item()
