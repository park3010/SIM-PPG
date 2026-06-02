"""Index the SigD common preprocessing manifest."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from common import bool_from_any, load_csv_rows, optional_float, require_file, resolve_from_root


class ManifestIndex:
    """Lookup layer for common-input windows and their metadata."""

    def __init__(self, root: Path, config: dict[str, Any]) -> None:
        self.root = root
        self.config = config
        self.manifest_path = resolve_from_root(root, config["input"]["preprocessing_manifest_path"])
        self.array_path = resolve_from_root(root, config["input"]["common_array_path"])
        require_file(self.manifest_path)
        require_file(self.array_path)
        self.array = np.load(self.array_path, mmap_mode="r")
        self.rows_all = load_csv_rows(self.manifest_path)
        self.rows = [self._normalize_row(row) for row in self.rows_all if bool_from_any(row.get("common_input_available"))]
        self.by_array_index = {int(row["array_index"]): row for row in self.rows}
        self.by_subject_session: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.sessions_by_subject: dict[str, set[str]] = defaultdict(set)
        for row in self.rows:
            array_index = int(row["array_index"])
            subject = row["subject_id"]
            session = row["session_id"]
            self.by_subject_session[(subject, session)].append(array_index)
            self.sessions_by_subject[subject].add(session)
        for key in self.by_subject_session:
            self.by_subject_session[key].sort()

    def _normalize_row(self, row: dict[str, str]) -> dict[str, Any]:
        """Normalize one CSV row while preserving metadata fields."""

        out: dict[str, Any] = dict(row)
        if out.get("array_index", "") == "":
            raise ValueError(f"Missing array_index for common window: {out.get('window_id')}")
        out["array_index"] = int(float(out["array_index"]))
        out["session_id"] = out.get("session_timestamp", "")
        out["raw_range_id"] = out.get("parent_raw_range_id", "")
        out["common_input_available"] = bool_from_any(out.get("common_input_available"))
        for key in ("sqi_valid_mask", "svri_valid_mask", "ipa_valid_mask"):
            out[key] = bool_from_any(out.get(key))
        out["sqi"] = optional_float(out.get("sqi_skewness"))
        out["svri"] = optional_float(out.get("svri"))
        out["ipa"] = optional_float(out.get("ipa"))
        return out

    def validate(self) -> dict[str, Any]:
        """Validate array/manifest consistency."""

        expected_shape = tuple(int(x) for x in self.config["input"]["expected_array_shape"])
        expected_dtype = str(self.config["input"]["expected_dtype"])
        indices = sorted(self.by_array_index)
        errors: list[str] = []
        if tuple(self.array.shape) != expected_shape:
            errors.append(f"array_shape_mismatch:{tuple(self.array.shape)}!={expected_shape}")
        if str(self.array.dtype) != expected_dtype:
            errors.append(f"array_dtype_mismatch:{self.array.dtype}!={expected_dtype}")
        if len(self.rows) != expected_shape[0]:
            errors.append(f"common_row_count_mismatch:{len(self.rows)}!={expected_shape[0]}")
        if len(indices) != len(set(indices)):
            errors.append("array_index_not_unique")
        if indices and (indices[0] != 0 or indices[-1] != expected_shape[0] - 1):
            errors.append(f"array_index_range_mismatch:{indices[0]}..{indices[-1]}")
        if indices != list(range(expected_shape[0])):
            errors.append("array_index_not_contiguous")
        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "common_available_count": len(self.rows),
            "array_shape": list(self.array.shape),
            "array_dtype": str(self.array.dtype),
            "array_index_min": indices[0] if indices else None,
            "array_index_max": indices[-1] if indices else None,
            "subject_count": len(self.sessions_by_subject),
            "session_count": len(self.by_subject_session),
        }

    def get_metadata(self, array_index: int) -> dict[str, Any]:
        """Return metadata for an array row."""

        try:
            return self.by_array_index[int(array_index)]
        except KeyError as exc:
            raise KeyError(f"array_index not in manifest: {array_index}") from exc

    def get_waveform(self, array_index: int) -> np.ndarray:
        """Return a read-only memmap row by array index."""

        self.get_metadata(array_index)
        return self.array[int(array_index)]

    def subject_ids(self) -> list[str]:
        """Return all subjects with common windows."""

        return sorted(self.sessions_by_subject)

    def available_sessions(self, subject_id: str) -> list[str]:
        """Return sorted sessions for a subject."""

        return sorted(self.sessions_by_subject.get(subject_id, set()))

    def array_indices_for_session(self, subject_id: str, session_id: str) -> list[int]:
        """Return array indices for one subject/session."""

        return list(self.by_subject_session.get((subject_id, session_id), []))

    def array_indices_for_subject(self, subject_id: str) -> list[int]:
        """Return all array indices for a subject."""

        indices: list[int] = []
        for session in self.available_sessions(subject_id):
            indices.extend(self.array_indices_for_session(subject_id, session))
        return sorted(indices)

    def split_window_indices(self, subject_ids: set[str]) -> list[int]:
        """Return all common-input array indices for a subject set."""

        indices: list[int] = []
        for subject_id in subject_ids:
            indices.extend(self.array_indices_for_subject(subject_id))
        return sorted(indices)
