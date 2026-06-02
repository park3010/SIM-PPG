"""Common helpers for SigD-Core preprocessing."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


PREPROCESSING_DIR_PARTS = ("preprocessing", "SigD")
DEFAULT_CONFIG_PATH = Path("preprocessing/SigD/config/sigd_preprocess_10s.yaml")


def utc_now_iso() -> str:
    """Return a UTC ISO-8601 timestamp with seconds precision."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def detect_root(root_arg: str | None = None) -> Path:
    """Resolve the SIM_PPG project root."""

    if root_arg:
        return Path(root_arg).expanduser().resolve()
    candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    script_root = Path(__file__).resolve().parents[3]
    candidates.extend([script_root, *script_root.parents])
    for candidate in candidates:
        if (candidate / "dataset" / "SigD").exists() and (
            candidate / "preprocessing" / "SigD"
        ).exists():
            return candidate
    raise SystemExit("Could not detect SIM_PPG root. Pass --root PATH.")


def preprocessing_dir(root: Path) -> Path:
    """Return preprocessing/SigD under the project root."""

    return root / "preprocessing" / "SigD"


def setup_logging(root: Path, filename: str, verbose: bool = False) -> None:
    """Configure UTC console/file logging."""

    log_dir = preprocessing_dir(root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)sZ %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[
            logging.FileHandler(log_dir / filename, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )

    import time

    logging.Formatter.converter = time.gmtime


def load_config(root: Path, config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML preprocessing config."""

    path = (
        Path(config_path).expanduser().resolve()
        if config_path
        else root / DEFAULT_CONFIG_PATH
    )
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_path(root: Path, value: str | Path) -> Path:
    """Resolve project-relative paths."""

    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def sha256_file(path: Path) -> str:
    """Compute SHA256 for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_jsonable(payload: Any) -> str:
    """Hash a JSON-serializable object with stable formatting."""

    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file as dictionaries."""

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Write fixed-column UTF-8 CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def read_json(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON file."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write readable UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def as_float(value: Any) -> float | None:
    """Parse a finite float from a CSV/string value."""

    if value in {"", None}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def bool_from_any(value: Any) -> bool:
    """Parse common string/boolean values."""

    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def distribution(values: Iterable[Any]) -> dict[str, int]:
    """Return a JSON-friendly distribution."""

    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def numeric_summary(values: Iterable[Any]) -> dict[str, Any]:
    """Return count/min/median/max/mean for finite numeric values."""

    clean = sorted(
        float(value)
        for value in values
        if value not in {"", None} and math.isfinite(float(value))
    )
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
