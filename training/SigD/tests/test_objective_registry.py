from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from objective_registry import compute_total_objective, infer_objective_name  # noqa: E402
from positive_masks import build_cross_session_positive_mask, build_generic_supcon_positive_mask  # noqa: E402
from session_alignment_loss import SessionCentroidAlignmentLoss  # noqa: E402
from supervised_contrastive_loss import supervised_contrastive_loss  # noqa: E402


def batch_ids():
    subjects = []
    sessions = []
    for subject in range(8):
        for session in ("s1", "s2"):
            for _ in range(2):
                subjects.append(f"p{subject}")
                sessions.append(session)
    return subjects, sessions


def config(mode: str, weight: float) -> dict:
    return {
        "training": {"positive_mask_mode": mode, "temperature": 0.07, "loss": "supervised_contrastive"},
        "loss_components": {
            "supervised_contrastive_weight": 1.0,
            "session_centroid_alignment_weight": weight,
            "morphology_weight": 0.0,
            "sqi_weighting_enabled": False,
            "ipa_weight": 0.0,
        },
    }


def test_e6_base_total_equals_generic_supcon_when_lambda_zero() -> None:
    subjects, sessions = batch_ids()
    embeddings = F.normalize(torch.randn(32, 128), dim=1)
    cfg = config("same_subject_different_sample", 0.0)
    total, diagnostics = compute_total_objective(embeddings, subjects, sessions, cfg)
    expected = supervised_contrastive_loss(embeddings, build_generic_supcon_positive_mask(subjects, sessions))
    assert infer_objective_name(cfg) == "generic_supcon_noalign"
    assert float(total.detach()) == pytest.approx(float(expected.detach()), abs=1.0e-6)
    assert diagnostics["alignment_loss"] == 0.0


def test_e6_a_total_is_cs_supcon_plus_alignment() -> None:
    subjects, sessions = batch_ids()
    embeddings = F.normalize(torch.randn(32, 128), dim=1)
    cfg = config("same_subject_different_session", 0.1)
    total, diagnostics = compute_total_objective(embeddings, subjects, sessions, cfg)
    supcon = supervised_contrastive_loss(embeddings, build_cross_session_positive_mask(subjects, sessions))
    align = SessionCentroidAlignmentLoss()(embeddings, subjects, sessions)
    assert infer_objective_name(cfg) == "cs_supcon_with_alignment"
    assert float(total.detach()) == pytest.approx(float((supcon + 0.1 * align).detach()), abs=1.0e-6)
    assert diagnostics["morphology_loss_active"] is False


def test_e6_b_total_is_generic_supcon_plus_alignment() -> None:
    subjects, sessions = batch_ids()
    embeddings = F.normalize(torch.randn(32, 128), dim=1)
    cfg = config("same_subject_different_sample", 0.2)
    total, _ = compute_total_objective(embeddings, subjects, sessions, cfg)
    supcon = supervised_contrastive_loss(embeddings, build_generic_supcon_positive_mask(subjects, sessions))
    align = SessionCentroidAlignmentLoss()(embeddings, subjects, sessions)
    assert infer_objective_name(cfg) == "generic_supcon_with_alignment"
    assert float(total.detach()) == pytest.approx(float((supcon + 0.2 * align).detach()), abs=1.0e-6)


def test_morphology_sqi_ipa_are_inactive() -> None:
    subjects, sessions = batch_ids()
    embeddings = F.normalize(torch.randn(32, 128), dim=1)
    _, diagnostics = compute_total_objective(embeddings, subjects, sessions, config("same_subject_different_sample", 0.0))
    assert diagnostics["morphology_loss_active"] is False
    assert diagnostics["sqi_weighting_active"] is False
    assert diagnostics["ipa_loss_active"] is False


def test_unsupported_objective_configuration_raises() -> None:
    subjects, sessions = batch_ids()
    embeddings = F.normalize(torch.randn(32, 128), dim=1)
    with pytest.raises(ValueError, match="Unsupported"):
        compute_total_objective(embeddings, subjects, sessions, config("unknown_mask", 0.0))

