"""Positive and negative pair masks for controlled SupCon objectives."""

from __future__ import annotations

from typing import Sequence

import torch


def _validate_inputs(subject_ids: Sequence[str], session_ids: Sequence[str] | None = None) -> int:
    n = len(subject_ids)
    if session_ids is not None and len(session_ids) != n:
        raise ValueError("subject_ids and session_ids must have the same length.")
    if n == 0:
        raise ValueError("Cannot build masks for an empty batch.")
    return n


def build_generic_supcon_positive_mask(subject_ids: Sequence[str], session_ids: Sequence[str]) -> torch.Tensor:
    """Positive iff same subject and different sample; session is ignored."""

    n = _validate_inputs(subject_ids, session_ids)
    mask = torch.zeros((n, n), dtype=torch.bool)
    for i in range(n):
        for j in range(n):
            mask[i, j] = i != j and subject_ids[i] == subject_ids[j]
    return mask


def build_cross_session_positive_mask(subject_ids: Sequence[str], session_ids: Sequence[str]) -> torch.Tensor:
    """Positive iff same subject, different sample, and different session."""

    n = _validate_inputs(subject_ids, session_ids)
    mask = torch.zeros((n, n), dtype=torch.bool)
    for i in range(n):
        for j in range(n):
            mask[i, j] = i != j and subject_ids[i] == subject_ids[j] and session_ids[i] != session_ids[j]
    return mask


def build_negative_mask(subject_ids: Sequence[str]) -> torch.Tensor:
    """Negative iff different subject."""

    n = _validate_inputs(subject_ids)
    mask = torch.zeros((n, n), dtype=torch.bool)
    for i in range(n):
        for j in range(n):
            mask[i, j] = subject_ids[i] != subject_ids[j]
    return mask


def build_positive_mask(mode: str, subject_ids: Sequence[str], session_ids: Sequence[str]) -> torch.Tensor:
    """Dispatch positive mask construction by config mode."""

    if mode == "same_subject_different_sample":
        return build_generic_supcon_positive_mask(subject_ids, session_ids)
    if mode == "same_subject_different_session":
        return build_cross_session_positive_mask(subject_ids, session_ids)
    raise ValueError(f"Unsupported positive_mask_mode: {mode}")


def validate_positive_mask(mask: torch.Tensor) -> dict[str, object]:
    """Validate diagonal, symmetry, and positive availability."""

    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError("Positive mask must be square [B, B].")
    if mask.dtype is not torch.bool:
        raise ValueError("Positive mask must be boolean.")
    diagonal_false = not torch.diagonal(mask).any().item()
    symmetric = torch.equal(mask, mask.T)
    positive_counts = mask.sum(dim=1)
    anchors_without_positive = torch.where(positive_counts == 0)[0].cpu().tolist()
    return {
        "valid": diagonal_false and symmetric and len(anchors_without_positive) == 0,
        "diagonal_false": diagonal_false,
        "symmetric": symmetric,
        "positive_counts": positive_counts.cpu().tolist(),
        "anchors_without_positive": anchors_without_positive,
    }
