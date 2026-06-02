"""Train-subject common-window pool for dynamic adaptation samplers."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from common import bool_from_any, load_csv_rows, numeric_summary, resolve_from_root
from manifest_index import ManifestIndex


class TrainSubjectPool:
    """Lookup layer restricted to train split subjects and common-input windows."""

    def __init__(self, root: Path, config: dict[str, Any], manifest_index: ManifestIndex) -> None:
        self.root = root
        self.config = config
        self.manifest_index = manifest_index
        split_rows = load_csv_rows(resolve_from_root(root, config["protocol"]["subject_split_path"]))
        self.subject_split = {row["subject_id"]: row["split"] for row in split_rows}
        self.train_subject_ids = sorted(
            subject for subject, split in self.subject_split.items() if split in set(config["training_pool"]["allowed_split"])
        )
        self.train_subject_set = set(self.train_subject_ids)
        self.subject_sessions: dict[str, list[str]] = {}
        self.session_indices: dict[tuple[str, str], list[int]] = {}
        for subject in self.train_subject_ids:
            sessions = manifest_index.available_sessions(subject)
            self.subject_sessions[subject] = sessions
            for session in sessions:
                self.session_indices[(subject, session)] = manifest_index.array_indices_for_session(subject, session)
        self.cross_session_subject_ids = sorted(
            subject for subject, sessions in self.subject_sessions.items() if len(sessions) >= 2
        )

    def validate(self) -> dict[str, Any]:
        """Validate no validation/test leakage and common-input-only inclusion."""

        errors: list[str] = []
        expected_train = int(self.config["protocol"]["expected_subject_counts"]["train"])
        if len(self.train_subject_ids) != expected_train:
            errors.append(f"train_subject_count_mismatch:{len(self.train_subject_ids)}!={expected_train}")
        leakage = [
            subject
            for subject in self.subject_sessions
            if self.subject_split.get(subject) not in set(self.config["training_pool"]["allowed_split"])
        ]
        if leakage:
            errors.append(f"non_train_subject_in_pool:{leakage[:5]}")
        for (subject, session), indices in self.session_indices.items():
            for index in indices:
                row = self.manifest_index.get_metadata(index)
                if not bool_from_any(row.get("common_input_available")):
                    errors.append(f"non_common_window_in_pool:{index}")
                if row["subject_id"] != subject or row["session_id"] != session:
                    errors.append(f"metadata_lookup_mismatch:{index}")
        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "train_subject_count": len(self.train_subject_ids),
            "train_session_count": sum(len(sessions) for sessions in self.subject_sessions.values()),
            "train_common_window_count": sum(len(indices) for indices in self.session_indices.values()),
            "cross_session_eligible_train_subject_count": len(self.cross_session_subject_ids),
            "val_test_leakage_count": len(leakage),
            "morphology_validity_used_for_sampling": False,
        }

    def sessions_for_subject(self, subject_id: str) -> list[str]:
        return list(self.subject_sessions.get(subject_id, []))

    def indices_for_session(self, subject_id: str, session_id: str) -> list[int]:
        return list(self.session_indices.get((subject_id, session_id), []))

    def indices_for_subject(self, subject_id: str) -> list[int]:
        indices: list[int] = []
        for session in self.sessions_for_subject(subject_id):
            indices.extend(self.indices_for_session(subject_id, session))
        return indices

    def summary(self) -> dict[str, Any]:
        """Return train-pool summary statistics."""

        session_counts = [len(sessions) for sessions in self.subject_sessions.values()]
        window_counts = [len(self.indices_for_subject(subject)) for subject in self.train_subject_ids]
        all_indices = [idx for subject in self.train_subject_ids for idx in self.indices_for_subject(subject)]
        return {
            **self.validate(),
            "subject_session_count_summary": numeric_summary(session_counts),
            "subject_window_count_summary": numeric_summary(window_counts),
            "morphology_reference_counts": {
                "sqi_valid": sum(1 for idx in all_indices if self.manifest_index.get_metadata(idx)["sqi_valid_mask"]),
                "svri_valid": sum(1 for idx in all_indices if self.manifest_index.get_metadata(idx)["svri_valid_mask"]),
                "ipa_valid": sum(1 for idx in all_indices if self.manifest_index.get_metadata(idx)["ipa_valid_mask"]),
            },
        }
