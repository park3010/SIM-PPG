from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_multiseed_results import audit_model_seed  # noqa: E402
from summarize_multiseed_results import E4_MODEL, E7A_MODEL, paired_deltas, read_seed_result  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_result(root: Path, model: str, seed: int, *, sqi_weighting: bool = False) -> Path:
    if model == E4_MODEL:
        result_root = root / f"training/SigD/results/papagei_s_generic_supcon_head_only/seed{seed}"
        manifest = {
            "experiment_id": "PAPAGEI_S_GENERIC_SUPCON_HEAD_ONLY_SIGD_V1",
            "test_accessed_during_training": False,
            "morphology_heads_enabled": False,
            "model_metadata": {"backbone_checkpoint_sha256": "hash"},
        }
    else:
        result_root = root / f"training/SigD/results/papagei_s_e7_a_generic_supcon_morph_e4_branch/svri0p05_sqi0p05/seed{seed}"
        manifest = {
            "experiment_id": "PAPAGEI_S_GENERIC_SUPCON_MORPH_E4_BRANCH_SIGD_V1",
            "test_accessed_during_training": False,
            "candidate_name": "svri0p05_sqi0p05",
            "lambda_svri": 0.05,
            "lambda_sqi": 0.05,
            "sqi_weighting_enabled": sqi_weighting,
            "use_ipa": False,
            "model_metadata": {"backbone_checkpoint_sha256": "hash"},
        }
    _write_json(result_root / "manifest.json", manifest)
    _write_csv(
        result_root / "validation_history.csv",
        [{"epoch": 1, "validation_exhaustive_eer": 0.35, "validation_tar_at_far_1pct": 0.05}],
    )
    (result_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (result_root / "checkpoints/best_projection_head.pt").write_bytes(b"checkpoint")
    eval_root = result_root / "final_exhaustive_evaluation"
    _write_json(
        eval_root / "adapted_model_run_manifest.json",
        {
            "validation_threshold_only": True,
            "test_threshold_tuning_performed": False,
            "final_protocol_id": "SIGD_COMMON_10S_K5M1_EARLIEST_ENROLL_LATER_PROBE_EXHAUSTIVE_EVAL_V2",
            "input_protocol_id": "COMMON_PPG_10S_125HZ_V1",
            "model_metadata": {"backbone_checkpoint_sha256": "hash"},
        },
    )
    _write_json(
        eval_root / "test_metrics.json",
        {
            "roc_auc": 0.7 if model == E4_MODEL else 0.72,
            "diagnostic_eer": {"eer": 0.36 if model == E4_MODEL else 0.34},
            "validation_fixed_far_1pct_threshold": {"tar": 0.06 if model == E4_MODEL else 0.07, "far": 0.01},
            "validation_fixed_eer_threshold": {"tar": 0.64, "far": 0.35},
        },
    )
    _write_csv(eval_root / "test_scores.csv", [{"trial_id": "t0", "score": 0.5}])
    return result_root


def test_multiseed_audit_accepts_fixed_e7a_candidate(tmp_path: Path) -> None:
    _make_result(tmp_path, E7A_MODEL, 52)
    errors: list[str] = []
    warnings: list[str] = []
    entry = audit_model_seed(tmp_path, "E7_A_SIM_PPG", 52, False, errors, warnings)
    assert entry["present"] is True
    assert entry["candidate_name"] == "svri0p05_sqi0p05"
    assert errors == []


def test_multiseed_audit_flags_forbidden_e7a_sqi_weighting(tmp_path: Path) -> None:
    _make_result(tmp_path, E7A_MODEL, 52, sqi_weighting=True)
    errors: list[str] = []
    warnings: list[str] = []
    audit_model_seed(tmp_path, "E7_A_SIM_PPG", 52, False, errors, warnings)
    assert any("forbidden option" in error for error in errors)


def test_multiseed_summary_reads_metrics_and_paired_delta(tmp_path: Path) -> None:
    e4_root = _make_result(tmp_path, E4_MODEL, 52)
    e7a_root = _make_result(tmp_path, E7A_MODEL, 52)
    e4 = read_seed_result(E4_MODEL, 52, e4_root, None)
    e7a = read_seed_result(E7A_MODEL, 52, e7a_root, None)
    deltas = paired_deltas([e4, e7a])
    assert e4["test_diagnostic_eer"] == 0.36
    assert e7a["test_roc_auc"] == 0.72
    assert deltas["e7a_improves_eer_count"] == 1
    assert deltas["e7a_improves_tar_far1_count"] == 1
