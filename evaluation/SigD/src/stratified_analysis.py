"""Stratified and macro analyses for SigD verification scores."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from common import numeric_summary
from metrics import compute_eer, compute_roc_auc, compute_threshold_metrics


def time_gap_metrics(
    score_rows: list[dict[str, Any]],
    threshold: float,
    buckets: list[str],
    threshold_name: str = "validation_eer_threshold",
) -> list[dict[str, Any]]:
    """Compute fixed-threshold metrics by time-gap bucket."""

    rows: list[dict[str, Any]] = []
    for bucket in buckets:
        subset = [row for row in score_rows if row.get("probe_time_gap_bucket") == bucket]
        labels = np.asarray([int(row["label"]) for row in subset], dtype=np.int64)
        scores = np.asarray([float(row["score"]) for row in subset], dtype=np.float64)
        metrics = compute_threshold_metrics(labels, scores, threshold) if subset else {}
        rows.append(
            {
                "threshold_name": threshold_name,
                "probe_time_gap_bucket": bucket,
                "trial_count": len(subset),
                "genuine_count": int((labels == 1).sum()) if subset else 0,
                "impostor_count": int((labels == 0).sum()) if subset else 0,
                "roc_auc": compute_roc_auc(labels, scores) if subset else None,
                "diagnostic_eer_optional": compute_eer(labels, scores).get("eer") if subset else None,
                "far": metrics.get("far"),
                "frr": metrics.get("frr"),
                "tar": metrics.get("tar"),
                "threshold": threshold if subset else None,
            }
        )
    return rows


def subject_macro_metrics(
    score_rows: list[dict[str, Any]],
    threshold: float,
    threshold_name: str = "validation_eer_threshold",
) -> list[dict[str, Any]]:
    """Compute threshold metrics per enrollment subject."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        grouped[str(row["enroll_subject_id"])].append(row)
    output: list[dict[str, Any]] = []
    for subject_id, rows in sorted(grouped.items()):
        labels = [int(row["label"]) for row in rows]
        scores = [float(row["score"]) for row in rows]
        metrics = compute_threshold_metrics(labels, scores, threshold)
        output.append({"threshold_name": threshold_name, "enroll_subject_id": subject_id, "trial_count": len(rows), **metrics})
    return output


def session_pair_macro_table(
    score_rows: list[dict[str, Any]],
    threshold: float,
    threshold_name: str = "validation_eer_threshold",
) -> list[dict[str, Any]]:
    """Average scores within session-pair groups before fixed-threshold metrics."""

    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        key = (
            str(row["enroll_subject_id"]),
            str(row["enroll_session_id"]),
            str(row["probe_subject_id"]),
            str(row["probe_session_id"]),
            str(row["trial_type"]),
        )
        grouped[key].append(row)
    output: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        labels = {int(row["label"]) for row in rows}
        label = labels.pop() if len(labels) == 1 else None
        mean_score = float(np.mean([float(row["score"]) for row in rows]))
        metrics = compute_threshold_metrics([label], [mean_score], threshold) if label is not None else {}
        output.append(
            {
                "threshold_name": threshold_name,
                "enroll_subject_id": key[0],
                "enroll_session_id": key[1],
                "probe_subject_id": key[2],
                "probe_session_id": key[3],
                "trial_type": key[4],
                "label": label,
                "window_trial_count": len(rows),
                "mean_score": mean_score,
                "accepted_at_threshold": bool(mean_score >= threshold),
                "far": metrics.get("far"),
                "frr": metrics.get("frr"),
                "tar": metrics.get("tar"),
            }
        )
    return output


def macro_summary(rows: list[dict[str, Any]], fields: tuple[str, ...] = ("far", "frr", "tar")) -> dict[str, Any]:
    """Summarize macro metric rows."""

    return {field: numeric_summary(row.get(field) for row in rows) for field in fields}
