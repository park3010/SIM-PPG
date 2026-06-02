"""Fixed canonical verification-trial Dataset for SigD."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from common import load_csv_rows, resolve_from_root
from manifest_index import ManifestIndex
from transforms import PerWindowZScore


class VerificationTrialDataset(Dataset):
    """Return K=5 enrollment windows and one M=1 probe window per trial."""

    def __init__(
        self,
        root: Path,
        config: dict[str, Any],
        manifest_index: ManifestIndex,
        split: str,
        transform: Any | None = None,
    ) -> None:
        allowed = set(config["evaluation_dataset"]["allowed_splits"])
        if split not in allowed:
            raise ValueError(f"Unsupported split: {split}")
        self.root = root
        self.config = config
        self.manifest_index = manifest_index
        self.split = split
        self.transform = transform if transform is not None else PerWindowZScore(
            eps=float(config["normalization"]["epsilon"]),
            output_channel_first=bool(config["normalization"]["output_channel_first"]),
        )
        template_rows = load_csv_rows(resolve_from_root(root, config["protocol"]["enrollment_templates_path"]))
        self.templates = {row["template_id"]: row for row in template_rows}
        split_paths = config["protocol"].get("verification_trial_paths_by_split")
        if split_paths:
            if split not in split_paths:
                raise ValueError(f"No verification trial CSV configured for split: {split}")
            all_trials = load_csv_rows(resolve_from_root(root, split_paths[split]))
        else:
            all_trials = load_csv_rows(resolve_from_root(root, config["protocol"]["verification_trials_path"]))
        self.trials = [row for row in all_trials if row["split"] == split]

    def __len__(self) -> int:
        return len(self.trials)

    def _load_window(self, array_index: int) -> torch.Tensor:
        self.manifest_index.get_metadata(array_index)
        return self.transform(self.manifest_index.get_waveform(array_index))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.trials[int(index)]
        template = self.templates[row["template_id"]]
        enrollment_indices = [int(item) for item in json.loads(template["enrollment_window_indices"])]
        enrollment = torch.stack([self._load_window(idx) for idx in enrollment_indices], dim=0)
        probe_index = int(float(row["probe_window_index"]))
        probe = self._load_window(probe_index)
        return {
            "trial_id": row["trial_id"],
            "split": row["split"],
            "label": int(row["label"]),
            "trial_type": row["trial_type"],
            "template_id": row["template_id"],
            "enrollment_windows": enrollment,
            "probe_window": probe,
            "enroll_subject_id": row["enroll_subject_id"],
            "probe_subject_id": row["probe_subject_id"],
            "enroll_session_id": row["enroll_session_id"],
            "probe_session_id": row["probe_session_id"],
            "probe_time_gap_days": float(row["probe_time_gap_days"]),
            "probe_time_gap_bucket": row["probe_time_gap_bucket"],
            "k": int(row["k"]),
            "m": int(row["m"]),
            "input_protocol_id": row["input_protocol_id"],
            "protocol_id": row["protocol_id"],
        }
