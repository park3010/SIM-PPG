"""Reporting helpers for SigD adaptation training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from common import write_csv_rows, write_json


def write_training_summary(path: Path, payload: dict[str, Any]) -> None:
    """Write training summary JSON."""

    write_json(path, payload)


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a history CSV."""

    write_csv_rows(path, rows)
