from __future__ import annotations

from pathlib import Path
import sys

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from common import rewrite_result_root_seed  # noqa: E402
from trainer import protect_scientific_result_root, resolve_training_result_root  # noqa: E402


def config_with_root(result_root: str) -> dict:
    return {"output": {"result_root": result_root}}


def test_smoke_results_use_separate_output_root(tmp_path: Path) -> None:
    config = config_with_root("training/SigD/results/papagei_s_generic_supcon_head_only/seed42")
    root = tmp_path
    smoke_root = resolve_training_result_root(root, config, smoke_test=True)
    assert smoke_root == root / "training/SigD/results/smoke_runs/papagei_s_generic_supcon_head_only/seed42"


def test_e6_smoke_results_use_lambda_output_root(tmp_path: Path) -> None:
    config = {
        "output": {"result_root": "training/SigD/results/papagei_s_e6_a_cs_supcon_alignment/seed42"},
        "loss_components": {
            "session_centroid_alignment_weight": 0.10,
            "session_centroid_alignment_weight_candidates": [0.01, 0.05, 0.10, 0.20],
        },
    }
    smoke_root = resolve_training_result_root(tmp_path, config, smoke_test=True)
    assert smoke_root == tmp_path / "training/SigD/results/smoke_runs/papagei_s_e6_a_cs_supcon_alignment/lambda_0p10/seed42"


def test_e7_smoke_results_use_candidate_output_root(tmp_path: Path) -> None:
    config = {
        "output": {"result_root": "training/SigD/results/papagei_s_e7_a_generic_supcon_morph_e4_branch/seed42"},
        "loss_components": {
            "lambda_svri": 0.05,
            "lambda_sqi": 0.05,
            "candidate_name": "svri0p05_sqi0p05",
            "morphology_loss_candidates": [{"name": "svri0p05_sqi0p05", "lambda_svri": 0.05, "lambda_sqi": 0.05}],
        },
    }
    smoke_root = resolve_training_result_root(tmp_path, config, smoke_test=True)
    assert smoke_root == tmp_path / "training/SigD/results/smoke_runs/papagei_s_e7_a_generic_supcon_morph_e4_branch/svri0p05_sqi0p05/seed42"


def test_e8_smoke_results_use_sqi_candidate_output_root(tmp_path: Path) -> None:
    config = {
        "output": {"result_root": "training/SigD/results/papagei_s_e8_sqi_weighted_morph_e7a/seed42"},
        "loss_components": {
            "candidate_name": "sqi_mild_linear",
            "sqi_weighting_mode": "mild_linear",
            "sqi_weighting_enabled": True,
            "sqi_weighting_candidates": [{"name": "sqi_mild_linear", "mode": "mild_linear"}],
        },
    }
    smoke_root = resolve_training_result_root(tmp_path, config, smoke_test=True)
    assert smoke_root == tmp_path / "training/SigD/results/smoke_runs/papagei_s_e8_sqi_weighted_morph_e7a/sqi_mild_linear/seed42"


def test_seed_override_rewrites_final_seed_directory() -> None:
    root = "training/SigD/results/papagei_s_generic_supcon_head_only/seed42"
    assert rewrite_result_root_seed(root, 52) == "training/SigD/results/papagei_s_generic_supcon_head_only/seed52"


def test_e7a_multiseed_candidate_result_root(tmp_path: Path) -> None:
    config = {
        "output": {"result_root": rewrite_result_root_seed("training/SigD/results/papagei_s_e7_a_generic_supcon_morph_e4_branch/seed42", 52)},
        "loss_components": {
            "lambda_svri": 0.05,
            "lambda_sqi": 0.05,
            "candidate_name": "svri0p05_sqi0p05",
            "morphology_loss_candidates": [{"name": "svri0p05_sqi0p05", "lambda_svri": 0.05, "lambda_sqi": 0.05}],
        },
    }
    result_root = resolve_training_result_root(tmp_path, config, smoke_test=False)
    assert result_root == tmp_path / "training/SigD/results/papagei_s_e7_a_generic_supcon_morph_e4_branch/svri0p05_sqi0p05/seed52"


def test_full_training_refuses_existing_scientific_result_without_overwrite(tmp_path: Path) -> None:
    result_root = tmp_path / "training/SigD/results/papagei_s_generic_supcon_head_only/seed42"
    result_root.mkdir(parents=True)
    (result_root / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Scientific training output exists"):
        protect_scientific_result_root(result_root, smoke_test=False, overwrite=False)


def test_smoke_result_guard_allows_existing_manifest(tmp_path: Path) -> None:
    result_root = tmp_path / "training/SigD/results/smoke_runs/papagei_s_generic_supcon_head_only/seed42"
    result_root.mkdir(parents=True)
    (result_root / "manifest.json").write_text("{}", encoding="utf-8")
    protect_scientific_result_root(result_root, smoke_test=True, overwrite=False)


def test_e4_e5_result_snapshot_policy_file_exists() -> None:
    root = Path(__file__).resolve().parents[3]
    assert (root / "training/SigD/metadata/post_e4_e5_development_policy.json").exists()


def test_e7_morphology_policy_file_exists() -> None:
    root = Path(__file__).resolve().parents[3]
    assert (root / "training/SigD/metadata/e7_morphology_development_policy.json").exists()


def test_e8_sqi_weighting_policy_file_exists() -> None:
    root = Path(__file__).resolve().parents[3]
    assert (root / "training/SigD/metadata/e8_sqi_weighting_development_policy.json").exists()
