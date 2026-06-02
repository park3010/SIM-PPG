"""Validation-only threshold selection utilities."""

from __future__ import annotations

from typing import Any

import numpy as np


def validate_binary_labels(labels: np.ndarray) -> None:
    """Require genuine=1 and impostor=0 labels with both classes present."""

    unique = set(int(x) for x in np.asarray(labels).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"Labels must be binary 0/1, got {sorted(unique)}")
    if unique != {0, 1}:
        raise ValueError("Both genuine=1 and impostor=0 classes are required.")


def apply_threshold(labels: Any, scores: Any, threshold: float) -> dict[str, float | int]:
    """Apply a fixed threshold and compute FAR/FRR/TAR."""

    labels_arr = np.asarray(labels, dtype=np.int64)
    scores_arr = np.asarray(scores, dtype=np.float64)
    if labels_arr.shape != scores_arr.shape:
        raise ValueError("labels and scores must have the same shape.")
    genuine = labels_arr == 1
    impostor = labels_arr == 0
    accept = scores_arr >= float(threshold)
    genuine_count = int(genuine.sum())
    impostor_count = int(impostor.sum())
    far = float((accept & impostor).sum() / impostor_count) if impostor_count else float("nan")
    frr = float(((~accept) & genuine).sum() / genuine_count) if genuine_count else float("nan")
    tar = float((accept & genuine).sum() / genuine_count) if genuine_count else float("nan")
    accuracy = float((accept == genuine).mean()) if labels_arr.size else float("nan")
    return {
        "threshold": float(threshold),
        "far": far,
        "frr": frr,
        "tar": tar,
        "accuracy": accuracy,
        "genuine_count": genuine_count,
        "impostor_count": impostor_count,
    }


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    """Return deterministic threshold candidates for score distributions."""

    unique = np.unique(np.asarray(scores, dtype=np.float64))
    if unique.size == 0:
        raise ValueError("No scores available.")
    eps = 1.0e-12
    mids = (unique[:-1] + unique[1:]) / 2.0 if unique.size > 1 else np.asarray([], dtype=np.float64)
    return np.asarray([unique[0] - eps, *mids.tolist(), unique[-1] + eps], dtype=np.float64)


def compute_eer_threshold(labels: Any, scores: Any) -> dict[str, float]:
    """Select threshold where FAR and FRR are closest."""

    labels_arr = np.asarray(labels, dtype=np.int64)
    scores_arr = np.asarray(scores, dtype=np.float64)
    validate_binary_labels(labels_arr)
    best: dict[str, float] | None = None
    for threshold in candidate_thresholds(scores_arr):
        metrics = apply_threshold(labels_arr, scores_arr, float(threshold))
        diff = abs(float(metrics["far"]) - float(metrics["frr"]))
        eer = (float(metrics["far"]) + float(metrics["frr"])) / 2.0
        if best is None or (diff, eer) < (best["diff"], best["eer"]):
            best = {
                "threshold": float(threshold),
                "eer": eer,
                "far": float(metrics["far"]),
                "frr": float(metrics["frr"]),
                "diff": diff,
            }
    assert best is not None
    best.pop("diff")
    return best


def compute_far_target_threshold(labels: Any, scores: Any, target_far: float = 0.01) -> dict[str, float]:
    """Select the lowest threshold satisfying FAR <= target_far on validation scores."""

    labels_arr = np.asarray(labels, dtype=np.int64)
    scores_arr = np.asarray(scores, dtype=np.float64)
    validate_binary_labels(labels_arr)
    feasible: list[tuple[float, dict[str, float | int]]] = []
    for threshold in candidate_thresholds(scores_arr):
        metrics = apply_threshold(labels_arr, scores_arr, float(threshold))
        if float(metrics["far"]) <= float(target_far) + 1.0e-12:
            feasible.append((float(threshold), metrics))
    if not feasible:
        threshold = float(np.max(scores_arr) + 1.0e-12)
        metrics = apply_threshold(labels_arr, scores_arr, threshold)
    else:
        threshold, metrics = min(feasible, key=lambda item: (item[0], -float(item[1]["tar"])))
    return {
        "threshold": float(threshold),
        "validation_far": float(metrics["far"]),
        "validation_tar": float(metrics["tar"]),
        "target_far": float(target_far),
    }
