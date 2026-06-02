"""Common helpers for SigD adaptation training."""

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


DEFAULT_GENERIC_CONFIG = Path("training/SigD/config/papagei_s_generic_supcon_head_only_seed42.yaml")
DEFAULT_CS_CONFIG = Path("training/SigD/config/papagei_s_cs_supcon_head_only_seed42.yaml")
DEFAULT_E6_BASE_CONFIG = Path("training/SigD/config/papagei_s_e6_base_generic_cs_batch_noalign_seed42.yaml")
DEFAULT_E6_A_CONFIG = Path("training/SigD/config/papagei_s_e6_a_cs_supcon_alignment_seed42.yaml")
DEFAULT_E6_B_CONFIG = Path("training/SigD/config/papagei_s_e6_b_generic_supcon_alignment_cs_batch_seed42.yaml")
DEFAULT_E7_A_CONFIG = Path("training/SigD/config/papagei_s_e7_a_generic_supcon_morph_e4_branch_seed42.yaml")
DEFAULT_E7_B_CONFIG = Path("training/SigD/config/papagei_s_e7_b_generic_supcon_morph_cs_batch_branch_seed42.yaml")
DEFAULT_E8_CONFIG = Path("training/SigD/config/papagei_s_e8_sqi_weighted_morph_e7a_seed42.yaml")


def utc_now_iso() -> str:
    """Return UTC ISO-8601 timestamp."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def detect_project_root(root: str | Path | None = None) -> Path:
    """Resolve the SIM_PPG project root."""

    if root is not None:
        resolved = Path(root).expanduser().resolve()
        if not (resolved / "training" / "SigD").exists():
            raise FileNotFoundError(f"Not a SIM_PPG root: {resolved}")
        return resolved
    for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (candidate / "training" / "SigD").exists() and (candidate / "data_pipeline" / "SigD").exists():
            return candidate
    raise FileNotFoundError("Could not detect SIM_PPG root. Pass --root PATH.")


def add_data_pipeline_src(root: Path) -> None:
    """Make data_pipeline/SigD/src importable."""

    src = root / "data_pipeline" / "SigD" / "src"
    if str(src) not in sys.path:
        sys.path.append(str(src))


def add_evaluation_src(root: Path) -> None:
    """Make evaluation/SigD/src importable."""

    src = root / "evaluation" / "SigD" / "src"
    if str(src) not in sys.path:
        sys.path.append(str(src))


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_training_config(root: Path, config_path: str | Path | None = None) -> dict[str, Any]:
    """Load a training config."""

    path = Path(config_path).expanduser().resolve() if config_path else root / DEFAULT_GENERIC_CONFIG
    return load_yaml_config(path)


def resolve_from_root(root: Path, relative_path: str | Path | None) -> Path | None:
    """Resolve project-relative paths."""

    if relative_path is None:
        return None
    path = Path(relative_path)
    return path if path.is_absolute() else root / path


def rewrite_result_root_seed(result_root: str | Path, seed: int) -> str:
    """Rewrite the final seed directory of a result root."""

    path = Path(result_root)
    parts = list(path.parts)
    if not parts:
        raise ValueError("Result root cannot be empty.")
    seed_part = f"seed{int(seed)}"
    if parts[-1].startswith("seed"):
        parts[-1] = seed_part
    else:
        parts.append(seed_part)
    return str(Path(*parts))


def require_file(path: Path) -> Path:
    """Require a file."""

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")
    return path


def ensure_dir(path: Path) -> Path:
    """Create a directory and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path) -> str:
    """Compute SHA256 for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load CSV rows."""

    require_file(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    """Write CSV rows with stable field ordering."""

    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON object."""

    require_file(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON object."""

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def set_random_seed(seed: int) -> None:
    """Set Python, NumPy, and torch seeds."""

    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


def bool_from_any(value: Any) -> bool:
    """Parse bool-like values."""

    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def optional_float(value: Any) -> float:
    """Parse optional float."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if np.isfinite(parsed) else float("nan")


def numeric_summary(values: Iterable[Any]) -> dict[str, Any]:
    """Summarize finite numeric values."""

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
    """Return value counts as a sorted dict."""

    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


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
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value
