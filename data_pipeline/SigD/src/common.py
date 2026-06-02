"""Common helpers for the SigD data pipeline."""

from __future__ import annotations

import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


DEFAULT_CONFIG_PATH = Path("data_pipeline/SigD/config/sigd_data_pipeline_10s_k5m1.yaml")


def utc_now_iso() -> str:
    """Return a UTC ISO-8601 timestamp."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def detect_project_root(root: str | Path | None = None) -> Path:
    """Resolve the SIM_PPG project root."""

    if root is not None:
        resolved = Path(root).expanduser().resolve()
        if not (resolved / "preprocessing" / "SigD").exists():
            raise FileNotFoundError(f"Not a SIM_PPG root: {resolved}")
        return resolved
    for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (candidate / "preprocessing" / "SigD").exists() and (candidate / "protocol" / "SigD").exists():
            return candidate
    raise FileNotFoundError("Could not detect SIM_PPG root. Pass --root PATH.")


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_pipeline_config(root: Path, config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the data-pipeline YAML config."""

    path = Path(config_path).expanduser().resolve() if config_path else root / DEFAULT_CONFIG_PATH
    return load_yaml_config(path)


def resolve_from_root(root: Path, relative_path: str | Path) -> Path:
    """Resolve project-relative paths."""

    path = Path(relative_path)
    return path if path.is_absolute() else root / path


def require_file(path: Path) -> Path:
    """Require a file to exist and return it."""

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")
    return path


def require_dir(path: Path) -> Path:
    """Require a directory to exist and return it."""

    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Required directory missing: {path}")
    return path


def set_random_seed(seed: int) -> None:
    """Set Python, NumPy, and torch random seeds when available."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load CSV rows as dictionaries."""

    require_file(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""

    require_file(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON summary under data_pipeline/SigD/metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def bool_from_any(value: Any) -> bool:
    """Parse common bool-like values."""

    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def optional_float(value: Any) -> float:
    """Parse a float, returning NaN on blank/nonfinite values."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if np.isfinite(parsed) else float("nan")


def numeric_summary(values: Iterable[Any]) -> dict[str, Any]:
    """Return JSON-friendly count/min/median/max/mean for finite numbers."""

    clean = sorted(float(v) for v in values if _is_finite_number(v))
    if not clean:
        return {"count": 0, "min": None, "median": None, "max": None, "mean": None}
    mid = len(clean) // 2
    median = clean[mid] if len(clean) % 2 else (clean[mid - 1] + clean[mid]) / 2.0
    return {
        "count": len(clean),
        "min": clean[0],
        "median": median,
        "max": clean[-1],
        "mean": sum(clean) / len(clean),
    }


def distribution(values: Iterable[Any]) -> dict[str, int]:
    """Count values as strings."""

    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _is_finite_number(value: Any) -> bool:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(parsed))
