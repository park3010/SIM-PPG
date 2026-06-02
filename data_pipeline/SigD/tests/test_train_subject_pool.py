from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from common import detect_project_root, load_pipeline_config  # noqa: E402
from manifest_index import ManifestIndex  # noqa: E402
from train_subject_pool import TrainSubjectPool  # noqa: E402


def build_pool() -> TrainSubjectPool:
    root = detect_project_root(Path(__file__).resolve().parents[3])
    config = load_pipeline_config(root)
    index = ManifestIndex(root, config)
    return TrainSubjectPool(root, config, index)


def test_train_subject_count_and_no_leakage() -> None:
    pool = build_pool()
    validation = pool.validate()
    assert validation["passed"] is True
    assert validation["train_subject_count"] == 139
    assert validation["val_test_leakage_count"] == 0


def test_common_input_windows_only() -> None:
    pool = build_pool()
    for subject in pool.train_subject_ids[:10]:
        for index in pool.indices_for_subject(subject):
            assert pool.manifest_index.get_metadata(index)["common_input_available"] is True


def test_morphology_invalid_window_can_remain_in_pool() -> None:
    pool = build_pool()
    train_indices = [idx for subject in pool.train_subject_ids for idx in pool.indices_for_subject(subject)]
    assert any(not pool.manifest_index.get_metadata(idx)["ipa_valid_mask"] for idx in train_indices)
