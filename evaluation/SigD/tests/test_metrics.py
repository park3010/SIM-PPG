from __future__ import annotations

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from metrics import compute_eer, compute_roc_auc, compute_threshold_metrics, validate_binary_trials  # noqa: E402


def test_perfect_score_auc_and_eer() -> None:
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert compute_roc_auc(labels, scores) == 1.0
    assert compute_eer(labels, scores)["eer"] == 0.0


def test_label_mapping_genuine_one() -> None:
    status = validate_binary_trials([0, 1, 1])
    assert status["valid"] is True
    assert status["genuine_count"] == 2
    assert status["impostor_count"] == 1


def test_threshold_metrics_far_frr_tar() -> None:
    metrics = compute_threshold_metrics([0, 0, 1, 1], [0.7, 0.1, 0.4, 0.9], threshold=0.5)
    assert metrics["far"] == 0.5
    assert metrics["frr"] == 0.5
    assert metrics["tar"] == 0.5


def test_unavailable_metric_safe_return() -> None:
    assert compute_roc_auc([1, 1], [0.8, 0.9]) is None
    assert compute_eer([1, 1], [0.8, 0.9])["eer"] is None
