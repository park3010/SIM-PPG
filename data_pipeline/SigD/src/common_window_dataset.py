"""Single-window Dataset for SigD common PPG windows."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from manifest_index import ManifestIndex
from transforms import PerWindowZScore


class CommonPPGWindowDataset(Dataset):
    """Dataset returning one normalized common-input PPG window at a time."""

    def __init__(
        self,
        manifest_index: ManifestIndex,
        array_indices: Sequence[int] | None = None,
        transform: Any | None = None,
        index_mode: str = "position",
    ) -> None:
        self.manifest_index = manifest_index
        self.array_indices = list(array_indices) if array_indices is not None else sorted(manifest_index.by_array_index)
        self.transform = transform if transform is not None else PerWindowZScore()
        if index_mode not in {"position", "array_index"}:
            raise ValueError(f"Unsupported index_mode: {index_mode}")
        self.index_mode = index_mode

    def __len__(self) -> int:
        return len(self.array_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        array_index = int(index) if self.index_mode == "array_index" else int(self.array_indices[int(index)])
        metadata = self.manifest_index.get_metadata(array_index)
        waveform = self.transform(self.manifest_index.get_waveform(array_index))
        return {
            "waveform": waveform,
            "array_index": array_index,
            "subject_id": metadata["subject_id"],
            "session_id": metadata["session_id"],
            "raw_range_id": metadata["raw_range_id"],
            "window_id": metadata.get("window_id", ""),
            "sqi": float(metadata["sqi"]),
            "svri": float(metadata["svri"]),
            "ipa": float(metadata["ipa"]),
            "sqi_valid_mask": bool(metadata["sqi_valid_mask"]),
            "svri_valid_mask": bool(metadata["svri_valid_mask"]),
            "ipa_valid_mask": bool(metadata["ipa_valid_mask"]),
        }
