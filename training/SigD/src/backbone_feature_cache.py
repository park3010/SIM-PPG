"""Frozen PaPaGei-S backbone feature cache for projection-head adaptation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from common import (
    add_data_pipeline_src,
    add_evaluation_src,
    ensure_dir,
    load_csv_rows,
    load_json,
    load_yaml_config,
    resolve_from_root,
    sha256_file,
    utc_now_iso,
    write_json,
)


CACHE_DIR = Path("training/SigD/cache/papagei_s_frozen_common_input_v1")
ROLE_FILENAMES = {
    "train": ("train_backbone_embeddings.npz", "train_cache_manifest.json"),
    "validation_exhaustive": (
        "validation_exhaustive_backbone_embeddings.npz",
        "validation_cache_manifest.json",
    ),
    "test_exhaustive": ("test_exhaustive_backbone_embeddings.npz", "test_cache_manifest.json"),
}
EXPECTED_ROLE_COUNTS = {
    "train": 12219,
    "validation_exhaustive": 2740,
    "test_exhaustive": 2625,
}


def role_paths(root: Path, role: str) -> tuple[Path, Path]:
    """Return cache data and manifest paths for a role."""

    if role not in ROLE_FILENAMES:
        raise ValueError(f"Unsupported cache role: {role}")
    data_name, manifest_name = ROLE_FILENAMES[role]
    base = root / CACHE_DIR
    return base / data_name, base / manifest_name


def array_indices_sha256(array_indices: Sequence[int]) -> str:
    """Hash sorted int64 array indices."""

    arr = np.asarray(sorted(int(x) for x in array_indices), dtype=np.int64)
    import hashlib

    return hashlib.sha256(arr.tobytes()).hexdigest()


def _load_train_pipeline(root: Path, train_config: dict[str, Any]):
    add_data_pipeline_src(root)
    from manifest_index import ManifestIndex
    from train_subject_pool import TrainSubjectPool
    from transforms import PerWindowZScore

    dp_path = resolve_from_root(root, train_config["input"]["common_data_pipeline_config"])
    assert dp_path is not None
    dp_config = load_yaml_config(dp_path)
    manifest_index = ManifestIndex(root, dp_config)
    train_pool = TrainSubjectPool(root, dp_config, manifest_index)
    transform = PerWindowZScore(
        eps=float(dp_config["normalization"]["epsilon"]),
        output_channel_first=bool(dp_config["normalization"]["output_channel_first"]),
    )
    return dp_config, manifest_index, train_pool, transform


def _load_final_eval_pipeline(root: Path, train_config: dict[str, Any]):
    add_data_pipeline_src(root)
    from manifest_index import ManifestIndex
    from transforms import PerWindowZScore

    dp_path = resolve_from_root(root, train_config["input"]["final_evaluation_data_pipeline_config"])
    assert dp_path is not None
    dp_config = load_yaml_config(dp_path)
    manifest_index = ManifestIndex(root, dp_config)
    transform = PerWindowZScore(
        eps=float(dp_config["normalization"]["epsilon"]),
        output_channel_first=bool(dp_config["normalization"]["output_channel_first"]),
    )
    return dp_config, manifest_index, transform


def _load_templates_and_trials(root: Path, dp_config: dict[str, Any], split: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    templates = load_csv_rows(resolve_from_root(root, dp_config["protocol"]["enrollment_templates_path"]))
    split_paths = dp_config["protocol"].get("verification_trial_paths_by_split")
    if split_paths:
        rows = load_csv_rows(resolve_from_root(root, split_paths[split]))
    else:
        rows = [row for row in load_csv_rows(resolve_from_root(root, dp_config["protocol"]["verification_trials_path"])) if row["split"] == split]
    return templates, [row for row in rows if row["split"] == split]


def collect_trial_unique_indices(template_rows: list[dict[str, str]], trial_rows: list[dict[str, str]]) -> list[int]:
    """Collect enrollment/probe array indices used by a trial list."""

    template_ids = {row["template_id"] for row in trial_rows}
    indices: set[int] = set()
    for row in template_rows:
        if row["template_id"] in template_ids:
            indices.update(int(item) for item in json.loads(row["enrollment_window_indices"]))
    for row in trial_rows:
        indices.add(int(float(row["probe_window_index"])))
    return sorted(indices)


def collect_role_indices(root: Path, train_config: dict[str, Any], role: str) -> tuple[list[int], Any, Any, dict[str, Any]]:
    """Collect array indices and loaders for one cache role."""

    if role == "train":
        dp_config, manifest_index, train_pool, transform = _load_train_pipeline(root, train_config)
        indices = sorted({idx for subject in train_pool.train_subject_ids for idx in train_pool.indices_for_subject(subject)})
        return indices, manifest_index, transform, dp_config
    if role in {"validation_exhaustive", "test_exhaustive"}:
        dp_config, manifest_index, transform = _load_final_eval_pipeline(root, train_config)
        split = "val" if role == "validation_exhaustive" else "test"
        templates, trials = _load_templates_and_trials(root, dp_config, split)
        indices = collect_trial_unique_indices(templates, trials)
        return indices, manifest_index, transform, dp_config
    raise ValueError(f"Unsupported cache role: {role}")


def expected_cache_provenance(root: Path, train_config: dict[str, Any], dp_config: dict[str, Any], role: str, array_indices: Sequence[int]) -> dict[str, Any]:
    """Return provenance fields used for cache invalidation."""

    reference = load_json(root / "evaluation" / "SigD" / "metadata" / "papagei_model_reference_manifest.json")
    source_manifest_path = root / "evaluation" / "SigD" / "metadata" / "papagei_source_snapshot_manifest.json"
    source_manifest = load_json(source_manifest_path) if source_manifest_path.exists() else {}
    adapter_source = root / "evaluation" / "SigD" / "src" / "papagei_s_adapter.py"
    return {
        "cache_role": role,
        "encoder_id": "papagei_s_frozen_official_common_input_v1",
        "backbone_embedding_dim": 512,
        "backbone_checkpoint_md5": reference.get("checkpoint_md5"),
        "backbone_checkpoint_sha256": reference.get("checkpoint_sha256"),
        "official_source_git_commit_sha": source_manifest.get("git_commit_sha"),
        "official_source_archive_sha256": source_manifest.get("archive_sha256"),
        "adapter_source_sha256": sha256_file(adapter_source),
        "input_protocol_id": train_config["input"]["input_protocol_id"],
        "preprocessing_snapshot_reference": dp_config["input"].get("preprocessing_snapshot_dir"),
        "normalization_policy": dp_config["normalization"]["policy"],
        "final_protocol_id": train_config["input"]["final_protocol_id"],
        "subject_split_seed": 42,
        "array_index_count": int(len(array_indices)),
        "array_indices_sha256": array_indices_sha256(array_indices),
    }


def compute_backbone_embeddings(
    *,
    root: Path,
    train_config: dict[str, Any],
    role: str,
    batch_size: int = 128,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute frozen PaPaGei-S backbone embeddings for one role."""

    add_evaluation_src(root)
    from papagei_s_adapter import PaPaGeiSFrozenAdapter

    indices, manifest_index, transform, dp_config = collect_role_indices(root, train_config, role)
    eval_config = load_yaml_config(resolve_from_root(root, train_config["input"]["frozen_eval_config_reference"]))
    encoder = PaPaGeiSFrozenAdapter(root, eval_config)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    encoder.eval()
    embeddings: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            waveforms = torch.stack([transform(manifest_index.get_waveform(index)) for index in batch_indices], dim=0).to(device)
            encoded = encoder.encode(waveforms).detach().cpu().numpy().astype(np.float32)
            embeddings.append(encoded)
    matrix = np.concatenate(embeddings, axis=0).astype(np.float32) if embeddings else np.empty((0, 512), dtype=np.float32)
    array_indices = np.asarray(indices, dtype=np.int64)
    provenance = expected_cache_provenance(root, train_config, dp_config, role, indices)
    return array_indices, matrix, provenance


def save_backbone_cache(root: Path, role: str, array_indices: np.ndarray, embeddings: np.ndarray, provenance: dict[str, Any], overwrite: bool = False) -> dict[str, Any]:
    """Save cache NPZ and manifest."""

    data_path, manifest_path = role_paths(root, role)
    if (data_path.exists() or manifest_path.exists()) and not overwrite:
        raise FileExistsError(f"Cache exists for {role}; pass --overwrite to replace it.")
    ensure_dir(data_path.parent)
    if len(array_indices) != len(set(int(x) for x in array_indices.tolist())):
        raise ValueError("Duplicate array indices in cache.")
    if embeddings.shape != (len(array_indices), 512):
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")
    finite = bool(np.isfinite(embeddings).all())
    if not finite:
        raise ValueError("Nonfinite backbone embeddings detected.")
    np.savez_compressed(data_path, array_indices=array_indices.astype(np.int64), embeddings=embeddings.astype(np.float32))
    manifest = {
        **provenance,
        "embedding_shape": [int(x) for x in embeddings.shape],
        "finite": finite,
        "duplicate_array_index_count": int(len(array_indices) - len(set(int(x) for x in array_indices.tolist()))),
        "cache_path": str(data_path.relative_to(root)),
        "created_datetime": utc_now_iso(),
    }
    write_json(manifest_path, manifest)
    return manifest


def load_backbone_cache(root: Path, role: str, expected_provenance: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load and validate a cache."""

    data_path, manifest_path = role_paths(root, role)
    manifest = load_json(manifest_path)
    if expected_provenance:
        mismatches = [
            key
            for key, value in expected_provenance.items()
            if key in manifest and manifest.get(key) != value
        ]
        if mismatches:
            raise RuntimeError(f"Backbone cache provenance mismatch: {mismatches}")
    payload = np.load(data_path)
    array_indices = payload["array_indices"].astype(np.int64)
    embeddings = payload["embeddings"].astype(np.float32)
    verify_cache_payload(role, array_indices, embeddings, manifest)
    return array_indices, embeddings, manifest


def verify_cache_payload(role: str, array_indices: np.ndarray, embeddings: np.ndarray, manifest: dict[str, Any]) -> dict[str, Any]:
    """Verify cache shape, uniqueness, finite values, and expected counts."""

    errors: list[str] = []
    expected_count = EXPECTED_ROLE_COUNTS.get(role)
    if expected_count is not None and len(array_indices) != expected_count:
        errors.append(f"array_index_count_mismatch:{len(array_indices)}!={expected_count}")
    if len(array_indices) != len(set(int(x) for x in array_indices.tolist())):
        errors.append("duplicate_array_index")
    if embeddings.shape != (len(array_indices), 512):
        errors.append(f"embedding_shape_mismatch:{embeddings.shape}")
    if not np.isfinite(embeddings).all():
        errors.append("embedding_nonfinite")
    if manifest.get("array_indices_sha256") != array_indices_sha256(array_indices):
        errors.append("array_indices_sha256_mismatch")
    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "cache_role": role,
        "array_index_count": int(len(array_indices)),
        "embedding_shape": [int(x) for x in embeddings.shape],
        "finite": bool(np.isfinite(embeddings).all()),
    }


class CachedBackboneEmbeddingDataset(Dataset):
    """Dataset returning cached 512-d backbone embeddings plus manifest metadata."""

    def __init__(self, root: Path, role: str, manifest_index: Any, index_mode: str = "array_index") -> None:
        self.root = root
        self.role = role
        self.manifest_index = manifest_index
        self.array_indices, self.embeddings, self.cache_manifest = load_backbone_cache(root, role)
        self.position_by_array_index = {int(index): pos for pos, index in enumerate(self.array_indices.tolist())}
        if index_mode not in {"position", "array_index"}:
            raise ValueError(f"Unsupported index_mode: {index_mode}")
        self.index_mode = index_mode

    def __len__(self) -> int:
        return int(len(self.array_indices))

    def __getitem__(self, index: int) -> dict[str, Any]:
        array_index = int(index) if self.index_mode == "array_index" else int(self.array_indices[int(index)])
        pos = self.position_by_array_index[array_index]
        metadata = self.manifest_index.get_metadata(array_index)
        return {
            "backbone_embedding": torch.from_numpy(self.embeddings[pos].copy()).to(dtype=torch.float32),
            "array_index": array_index,
            "subject_id": metadata["subject_id"],
            "session_id": metadata["session_id"],
            "sqi": float(metadata["sqi"]),
            "svri": float(metadata["svri"]),
            "ipa": float(metadata["ipa"]),
            "sqi_valid_mask": bool(metadata["sqi_valid_mask"]),
            "svri_valid_mask": bool(metadata["svri_valid_mask"]),
            "ipa_valid_mask": bool(metadata["ipa_valid_mask"]),
        }


def cached_backbone_collate_fn(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate cached backbone embedding batches for projection-head training."""

    return {
        "backbone_embeddings": torch.stack([item["backbone_embedding"] for item in items], dim=0),
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


def ensure_backbone_cache(
    *,
    root: Path,
    train_config: dict[str, Any],
    role: str,
    overwrite: bool = False,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Build cache if needed, then verify and return manifest."""

    data_path, manifest_path = role_paths(root, role)
    if not data_path.exists() or not manifest_path.exists() or overwrite:
        indices, embeddings, provenance = compute_backbone_embeddings(
            root=root,
            train_config=train_config,
            role=role,
            device=device,
        )
        return save_backbone_cache(root, role, indices, embeddings, provenance, overwrite=True)
    indices, embeddings, manifest = load_backbone_cache(root, role)
    verification = verify_cache_payload(role, indices, embeddings, manifest)
    if not verification["passed"]:
        raise RuntimeError(f"Backbone cache verification failed for {role}: {verification['errors']}")
    return manifest
