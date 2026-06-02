"""K-window template aggregation and cosine trial scoring."""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def l2_normalize(vector: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    """Return L2-normalized vector."""

    if not np.isfinite(vector).all():
        raise ValueError("Cannot normalize embedding vector containing nonfinite values.")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm):
        raise ValueError("Cannot normalize nonfinite vector.")
    if norm <= eps:
        raise ValueError("Cannot normalize zero or near-zero embedding vector.")
    return vector / norm


def aggregate_template(embeddings: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    """Normalize K embeddings, average, then normalize the template."""

    if embeddings.ndim != 2:
        raise ValueError("Enrollment embeddings must be [K, D].")
    normalized = np.stack([l2_normalize(row.astype(np.float32), eps) for row in embeddings], axis=0)
    return l2_normalize(normalized.mean(axis=0), eps).astype(np.float32)


def cosine_score(template_embedding: np.ndarray, probe_embedding: np.ndarray, eps: float = 1.0e-8) -> float:
    """Compute cosine similarity between normalized template and probe."""

    template = l2_normalize(template_embedding.astype(np.float32), eps)
    probe = l2_normalize(probe_embedding.astype(np.float32), eps)
    score = float(np.dot(template, probe))
    if not np.isfinite(score):
        raise ValueError("Nonfinite cosine score.")
    tolerance = 1.0e-6
    if score < -1.0 - tolerance or score > 1.0 + tolerance:
        raise ValueError(f"Cosine score outside valid range beyond tolerance: {score}")
    return float(np.clip(score, -1.0, 1.0))


def score_trial(
    *,
    trial: dict[str, str],
    template: dict[str, str],
    embedding_cache: dict[int, np.ndarray],
    encoder_id: str,
    eps: float = 1.0e-8,
) -> dict[str, Any]:
    """Score one canonical verification trial."""

    enrollment_indices = [int(item) for item in json.loads(template["enrollment_window_indices"])]
    enroll_embeddings = np.stack([embedding_cache[index] for index in enrollment_indices], axis=0)
    template_embedding = aggregate_template(enroll_embeddings, eps)
    probe_index = int(float(trial["probe_window_index"]))
    score = cosine_score(template_embedding, embedding_cache[probe_index], eps)
    return {
        "trial_id": trial["trial_id"],
        "split": trial["split"],
        "label": int(trial["label"]),
        "trial_type": trial["trial_type"],
        "template_id": trial["template_id"],
        "enroll_subject_id": trial["enroll_subject_id"],
        "probe_subject_id": trial["probe_subject_id"],
        "enroll_session_id": trial["enroll_session_id"],
        "probe_session_id": trial["probe_session_id"],
        "probe_time_gap_days": float(trial["probe_time_gap_days"]),
        "probe_time_gap_bucket": trial["probe_time_gap_bucket"],
        "score": score,
        "encoder_id": encoder_id,
        "input_protocol_id": trial["input_protocol_id"],
        "protocol_id": trial["protocol_id"],
    }


def score_trials(
    *,
    trial_rows: list[dict[str, str]],
    template_rows: list[dict[str, str]],
    embedding_cache: dict[int, np.ndarray],
    encoder_id: str,
    eps: float = 1.0e-8,
) -> list[dict[str, Any]]:
    """Score a list of trials."""

    templates = {row["template_id"]: row for row in template_rows}
    return [
        score_trial(
            trial=trial,
            template=templates[trial["template_id"]],
            embedding_cache=embedding_cache,
            encoder_id=encoder_id,
            eps=eps,
        )
        for trial in trial_rows
    ]
