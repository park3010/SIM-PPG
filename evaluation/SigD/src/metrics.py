"""Verification metrics for SigD evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from thresholds import apply_threshold, compute_eer_threshold, validate_binary_labels


def validate_binary_trials(labels: Any) -> dict[str, Any]:
    """Validate binary labels, returning a safe status object."""

    arr = np.asarray(labels, dtype=np.int64)
    unique = sorted(set(int(x) for x in arr.tolist()))
    valid = set(unique).issubset({0, 1}) and set(unique) == {0, 1}
    return {
        "valid": valid,
        "unique_labels": unique,
        "genuine_count": int((arr == 1).sum()),
        "impostor_count": int((arr == 0).sum()),
    }


def compute_roc_auc(labels: Any, scores: Any) -> float | None:
    """Compute ROC-AUC with tie-aware ranks; return None for unavailable subsets."""

    labels_arr = np.asarray(labels, dtype=np.int64)
    scores_arr = np.asarray(scores, dtype=np.float64)
    status = validate_binary_trials(labels_arr)
    if not status["valid"]:
        return None
    order = np.argsort(scores_arr)
    sorted_scores = scores_arr[order]
    ranks = np.empty_like(scores_arr, dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    positives = labels_arr == 1
    p = int(positives.sum())
    n = int((labels_arr == 0).sum())
    return float((ranks[positives].sum() - p * (p + 1) / 2.0) / (p * n))


def compute_eer(labels: Any, scores: Any) -> dict[str, Any]:
    """Compute diagnostic EER if both classes are available."""

    try:
        return compute_eer_threshold(labels, scores)
    except ValueError as exc:
        return {"threshold": None, "eer": None, "far": None, "frr": None, "unavailable_reason": str(exc)}


def compute_threshold_metrics(labels: Any, scores: Any, threshold: float) -> dict[str, Any]:
    """Compute fixed-threshold metrics with safe single-class support."""

    metrics = apply_threshold(labels, scores, threshold)
    return dict(metrics)


def compute_split_metrics(
    *,
    split: str,
    labels: Any,
    scores: Any,
    validation_eer_threshold: float | None = None,
    validation_far_threshold: float | None = None,
) -> dict[str, Any]:
    """Build a split metrics JSON object."""

    labels_arr = np.asarray(labels, dtype=np.int64)
    scores_arr = np.asarray(scores, dtype=np.float64)
    output: dict[str, Any] = {
        "split": split,
        "trial_count": int(labels_arr.size),
        "label_status": validate_binary_trials(labels_arr),
        "roc_auc": compute_roc_auc(labels_arr, scores_arr),
        "diagnostic_eer": compute_eer(labels_arr, scores_arr),
    }
    if validation_eer_threshold is not None:
        output["validation_fixed_eer_threshold"] = compute_threshold_metrics(
            labels_arr, scores_arr, validation_eer_threshold
        )
    if validation_far_threshold is not None:
        output["validation_fixed_far_1pct_threshold"] = compute_threshold_metrics(
            labels_arr, scores_arr, validation_far_threshold
        )
    return output
