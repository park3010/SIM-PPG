from __future__ import annotations

from pathlib import Path
import sys

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

from adaptation_evaluator import evaluate_validation_only  # noqa: E402
from common import load_yaml_config  # noqa: E402
from helpers import FakeBackbone  # noqa: E402
from papagei_projection_model import PaPaGeiProjectionModel  # noqa: E402


def load_real_config() -> dict:
    root = Path(__file__).resolve().parents[3]
    return load_yaml_config(root / "training/SigD/config/papagei_s_generic_supcon_head_only_seed42.yaml")


def test_adapted_embedding_dim_128_validation_scoring() -> None:
    root = Path(__file__).resolve().parents[3]
    config = load_real_config()
    model = PaPaGeiProjectionModel(root, config, backbone_adapter=FakeBackbone())
    result = evaluate_validation_only(
        root=root,
        train_config=config,
        model=model,
        device=torch.device("cpu"),
        max_trials=8,
    )
    assert result["split_summary"]["trial_count"] == 8
    assert result["test_data_read"] is False
    assert result["metrics"]["split"] == "val"


def test_threshold_selection_source_validation_only() -> None:
    root = Path(__file__).resolve().parents[3]
    config = load_real_config()
    assert config["evaluation"]["threshold_source_split"] == "val"
    assert config["evaluation"]["validation_threshold_only"] is True
    assert config["evaluation"]["checkpoint_selection_uses_test"] is False


def test_evaluation_protocol_id_is_exhaustive_v2() -> None:
    config = load_real_config()
    assert config["input"]["final_protocol_id"] == "SIGD_COMMON_10S_K5M1_EARLIEST_ENROLL_LATER_PROBE_EXHAUSTIVE_EVAL_V2"


def test_result_path_separate_from_frozen_result_path() -> None:
    config = load_real_config()
    assert config["output"]["result_root"].startswith("training/SigD/results/")
    assert "evaluation/SigD/results/papagei_s_frozen" not in config["output"]["result_root"]


def test_source_overlap_limitation_inherited() -> None:
    config = load_real_config()
    assert config["fairness"]["source_overlap_limitation_inherited"] is True
