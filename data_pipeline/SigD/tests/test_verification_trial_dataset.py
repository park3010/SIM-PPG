from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from common import detect_project_root, load_pipeline_config  # noqa: E402
from manifest_index import ManifestIndex  # noqa: E402
from transforms import PerWindowZScore  # noqa: E402
from verification_trial_dataset import VerificationTrialDataset  # noqa: E402


def datasets():
    root = detect_project_root(Path(__file__).resolve().parents[3])
    config = load_pipeline_config(root)
    index = ManifestIndex(root, config)
    transform = PerWindowZScore(float(config["normalization"]["epsilon"]))
    return {
        split: VerificationTrialDataset(root, config, index, split, transform)
        for split in ("train", "val", "test")
    }


def test_split_lengths() -> None:
    ds = datasets()
    assert len(ds["train"]) == 14532
    assert len(ds["val"]) == 15060
    assert len(ds["test"]) == 14370


def test_item_shapes_and_protocol() -> None:
    sample = datasets()["val"][0]
    assert tuple(sample["enrollment_windows"].shape) == (5, 1, 1250)
    assert tuple(sample["probe_window"].shape) == (1, 1250)
    assert sample["protocol_id"] == "SIGD_COMMON_10S_K5M1_EARLIEST_ENROLL_LATER_PROBE_V2"
    assert sample["probe_time_gap_days"] > 0


def test_labels_match_trial_type() -> None:
    val = datasets()["val"]
    for idx in (0, len(val) - 1):
        sample = val[idx]
        if sample["trial_type"] == "genuine":
            assert sample["label"] == 1
        else:
            assert sample["trial_type"] == "impostor"
            assert sample["label"] == 0
