"""Embedding cache helpers for fixed SigD verification trials."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import ensure_dir, utc_now_iso, write_json
from encoder_interface import PPGEncoderInterface


def parse_enrollment_indices(value: str) -> list[int]:
    """Parse JSON enrollment window index list."""

    return [int(item) for item in json.loads(value)]


def collect_unique_array_indices(template_rows: list[dict[str, str]], trial_rows: list[dict[str, str]]) -> list[int]:
    """Collect all enrollment/probe array indices needed by a split."""

    template_ids = {row["template_id"] for row in trial_rows}
    indices: set[int] = set()
    for row in template_rows:
        if row["template_id"] in template_ids:
            indices.update(parse_enrollment_indices(row["enrollment_window_indices"]))
    for row in trial_rows:
        indices.add(int(float(row["probe_window_index"])))
    return sorted(indices)


def compute_embedding_cache(
    *,
    manifest_index: Any,
    transform: Any,
    encoder: PPGEncoderInterface,
    array_indices: list[int],
    batch_size: int,
    device: torch.device,
) -> dict[int, np.ndarray]:
    """Compute array_index -> embedding using frozen inference."""

    encoder.eval()
    encoder.to(device)
    cache: dict[int, np.ndarray] = {}
    with torch.inference_mode():
        for start in range(0, len(array_indices), batch_size):
            batch_indices = array_indices[start : start + batch_size]
            waveforms = torch.stack(
                [transform(manifest_index.get_waveform(index)) for index in batch_indices],
                dim=0,
            ).to(device)
            embeddings = encoder.encode(waveforms).detach().cpu().numpy().astype(np.float32)
            for index, embedding in zip(batch_indices, embeddings):
                cache[int(index)] = embedding
    return cache


def save_embedding_cache(
    *,
    path: Path,
    manifest_path: Path,
    embeddings: dict[int, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    """Save embeddings and cache manifest."""

    ensure_dir(path.parent)
    array_indices = np.asarray(sorted(embeddings), dtype=np.int64)
    matrix = np.stack([embeddings[int(index)] for index in array_indices], axis=0).astype(np.float32)
    np.savez_compressed(path, array_indices=array_indices, embeddings=matrix)
    write_json(
        manifest_path,
        {
            **metadata,
            "unique_window_count": int(len(array_indices)),
            "embedding_dim": int(matrix.shape[1]) if matrix.ndim == 2 else None,
            "created_datetime": utc_now_iso(),
        },
    )


def load_embedding_cache(path: Path, expected_metadata: dict[str, Any], manifest_path: Path) -> dict[int, np.ndarray]:
    """Load an embedding cache after metadata key validation."""

    from common import load_json

    manifest = load_json(manifest_path)
    for key, value in expected_metadata.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Embedding cache metadata mismatch for {key}: {manifest.get(key)} != {value}")
    payload = np.load(path)
    return {
        int(index): embedding.astype(np.float32)
        for index, embedding in zip(payload["array_indices"], payload["embeddings"])
    }
