"""Common helpers for SigD evaluation scripts."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


DEFAULT_EVAL_CONFIG = Path("evaluation/SigD/config/papagei_s_frozen_cosine_eval.yaml")


def utc_now_iso() -> str:
    """Return a UTC ISO-8601 timestamp."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def detect_project_root(root: str | Path | None = None) -> Path:
    """Resolve the SIM_PPG project root."""

    if root is not None:
        resolved = Path(root).expanduser().resolve()
        if not (resolved / "data_pipeline" / "SigD").exists():
            raise FileNotFoundError(f"Not a SIM_PPG root: {resolved}")
        return resolved
    for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (candidate / "data_pipeline" / "SigD").exists() and (candidate / "protocol" / "SigD").exists():
            return candidate
    raise FileNotFoundError("Could not detect SIM_PPG root. Pass --root PATH.")


def add_data_pipeline_src(root: Path) -> None:
    """Make data_pipeline/SigD/src importable."""

    src = root / "data_pipeline" / "SigD" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_eval_config(root: Path, config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the evaluation YAML config."""

    path = Path(config_path).expanduser().resolve() if config_path else root / DEFAULT_EVAL_CONFIG
    return load_yaml_config(path)


def resolve_from_root(root: Path, relative_path: str | Path | None) -> Path | None:
    """Resolve a project-relative path."""

    if relative_path is None:
        return None
    path = Path(relative_path)
    return path if path.is_absolute() else root / path


def require_file(path: Path) -> Path:
    """Require a file to exist."""

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")
    return path


def ensure_dir(path: Path) -> Path:
    """Create an output directory and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path) -> str:
    """Compute SHA256 for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    """Compute MD5 for a file when an upstream source publishes MD5."""

    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_random_seed(seed: int) -> None:
    """Set Python, NumPy, and torch seeds when available."""

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


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    """Write CSV rows with stable field ordering."""

    ensure_dir(path.parent)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""

    require_file(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON object."""

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def numeric_summary(values: Iterable[Any]) -> dict[str, Any]:
    """Return count/min/median/max/mean/std for finite numeric values."""

    clean = np.asarray([float(v) for v in values if _is_finite(v)], dtype=np.float64)
    if clean.size == 0:
        return {"count": 0, "min": None, "median": None, "max": None, "mean": None, "std": None}
    return {
        "count": int(clean.size),
        "min": float(np.min(clean)),
        "median": float(np.median(clean)),
        "max": float(np.max(clean)),
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean)),
    }


def distribution(values: Iterable[Any]) -> dict[str, int]:
    """Count values as strings."""

    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


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


def _is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(_jsonable(value), ensure_ascii=False)
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
