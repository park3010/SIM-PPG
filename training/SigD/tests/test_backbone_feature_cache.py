from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

from backbone_feature_cache import (  # noqa: E402
    array_indices_sha256,
    collect_role_indices,
    expected_cache_provenance,
    load_backbone_cache,
    role_paths,
    save_backbone_cache,
)
from common import load_yaml_config, write_json  # noqa: E402
from helpers import FakeBackbone, minimal_config  # noqa: E402
from papagei_projection_model import PaPaGeiProjectionModel  # noqa: E402


def real_root() -> Path:
    return Path(__file__).resolve().parents[3]


def real_config() -> dict:
    return load_yaml_config(real_root() / "training/SigD/config/papagei_s_generic_supcon_head_only_seed42.yaml")


def test_train_backbone_cache_expected_count_12219() -> None:
    indices, _, _, _ = collect_role_indices(real_root(), real_config(), "train")
    assert len(indices) == 12219


def test_validation_backbone_cache_expected_unique_count_2740() -> None:
    indices, _, _, _ = collect_role_indices(real_root(), real_config(), "validation_exhaustive")
    assert len(indices) == 2740


def test_cache_manifest_contains_checkpoint_and_input_provenance() -> None:
    indices, _, _, dp_config = collect_role_indices(real_root(), real_config(), "validation_exhaustive")
    provenance = expected_cache_provenance(real_root(), real_config(), dp_config, "validation_exhaustive", indices)
    for key in (
        "backbone_checkpoint_md5",
        "backbone_checkpoint_sha256",
        "adapter_source_sha256",
        "input_protocol_id",
        "normalization_policy",
        "final_protocol_id",
        "array_indices_sha256",
    ):
        assert provenance.get(key)


def test_cache_invalidated_when_checkpoint_hash_changes(tmp_path: Path) -> None:
    data_path, manifest_path = role_paths(tmp_path, "train")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    indices = np.asarray([0, 1], dtype=np.int64)
    embeddings = np.zeros((2, 512), dtype=np.float32)
    np.savez_compressed(data_path, array_indices=indices, embeddings=embeddings)
    write_json(
        manifest_path,
        {
            "cache_role": "train",
            "array_index_count": 2,
            "array_indices_sha256": array_indices_sha256(indices),
            "embedding_shape": [2, 512],
            "finite": True,
            "backbone_checkpoint_sha256": "expected",
        },
    )
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        load_backbone_cache(tmp_path, "train", {"backbone_checkpoint_sha256": "changed"})


def test_direct_vs_cached_projection_output_equivalent() -> None:
    root = real_root()
    model = PaPaGeiProjectionModel(root, minimal_config(), backbone_adapter=FakeBackbone())
    model.eval()
    waveforms = torch.randn(4, 1, 1250)
    direct = model.encode(waveforms)
    cached_backbone = model.encode_backbone(waveforms)
    cached = model.project(cached_backbone)
    assert torch.allclose(direct, cached, atol=1.0e-6)


def test_e4_e5_share_same_backbone_cache_paths() -> None:
    root = real_root()
    generic = load_yaml_config(root / "training/SigD/config/papagei_s_generic_supcon_head_only_seed42.yaml")
    cs = load_yaml_config(root / "training/SigD/config/papagei_s_cs_supcon_head_only_seed42.yaml")
    assert generic["model"]["backbone_checkpoint"] == cs["model"]["backbone_checkpoint"]
    assert generic["input"]["final_protocol_id"] == cs["input"]["final_protocol_id"]
    assert role_paths(root, "train") == role_paths(root, "train")
    assert role_paths(root, "validation_exhaustive") == role_paths(root, "validation_exhaustive")
