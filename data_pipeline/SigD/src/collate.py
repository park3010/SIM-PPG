"""Collate helpers for SigD train batches."""

from __future__ import annotations

from typing import Any

import torch


def train_collate_fn(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate CommonPPGWindowDataset items for training."""

    return {
        "waveforms": torch.stack([item["waveform"] for item in items], dim=0),
        "array_indices": torch.tensor([int(item["array_index"]) for item in items], dtype=torch.long),
        "subject_ids": [item["subject_id"] for item in items],
        "session_ids": [item["session_id"] for item in items],
        "sqi": torch.tensor([float(item["sqi"]) for item in items], dtype=torch.float32),
        "svri": torch.tensor([float(item["svri"]) for item in items], dtype=torch.float32),
        "ipa": torch.tensor([float(item["ipa"]) for item in items], dtype=torch.float32),
        "sqi_valid_mask": torch.tensor([bool(item["sqi_valid_mask"]) for item in items], dtype=torch.bool),
        "svri_valid_mask": torch.tensor([bool(item["svri_valid_mask"]) for item in items], dtype=torch.bool),
        "ipa_valid_mask": torch.tensor([bool(item["ipa_valid_mask"]) for item in items], dtype=torch.bool),
    }
