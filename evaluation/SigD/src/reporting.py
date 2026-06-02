"""Reporting helpers for SigD evaluation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from common import ensure_dir, write_csv_rows, write_json


SCORE_COLUMNS = [
    "trial_id",
    "split",
    "label",
    "trial_type",
    "template_id",
    "enroll_subject_id",
    "probe_subject_id",
    "enroll_session_id",
    "probe_session_id",
    "probe_time_gap_days",
    "probe_time_gap_bucket",
    "score",
    "encoder_id",
    "input_protocol_id",
    "protocol_id",
]


def write_score_csv(path: Path, score_rows: list[dict[str, Any]]) -> None:
    """Write score rows."""

    write_csv_rows(path, score_rows, SCORE_COLUMNS)


def write_metrics_json(path: Path, metrics: dict[str, Any]) -> None:
    """Write metrics JSON."""

    write_json(path, metrics)


def write_run_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Write run manifest JSON."""

    write_json(path, payload)


def prepare_result_root(path: Path) -> Path:
    """Create a result root directory."""

    return ensure_dir(path)
