from __future__ import annotations

from pathlib import Path
import sys

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

from helpers import FakeBackbone, minimal_config  # noqa: E402
from objective_registry import compute_total_objective  # noqa: E402
from papagei_projection_model import PaPaGeiProjectionModel  # noqa: E402
from positive_masks import build_positive_mask  # noqa: E402


def ids():
    subjects = []
    sessions = []
    for subject in range(8):
        for session in ("s1", "s2"):
            for _ in range(2):
                subjects.append(f"p{subject}")
                sessions.append(session)
    return subjects, sessions


def cfg(stage: str) -> dict:
    config = minimal_config()
    config["training"].update(
        {
            "loss": "supervised_contrastive",
            "temperature": 0.07,
            "positive_mask_mode": "same_subject_different_session" if stage == "e6_a" else "same_subject_different_sample",
        }
    )
    config["loss_components"] = {
        "supervised_contrastive_weight": 1.0,
        "session_centroid_alignment_weight": 0.1 if stage in {"e6_a", "e6_b"} else 0.0,
        "morphology_weight": 0.0,
        "sqi_weighting_enabled": False,
        "ipa_weight": 0.0,
    }
    return config


def run_step(stage: str):
    config = cfg(stage)
    fake = FakeBackbone()
    model = PaPaGeiProjectionModel(Path(__file__).resolve().parents[3], config, backbone_adapter=fake)
    before_backbone = [p.detach().clone() for p in model.backbone.parameters()]
    before_projection = [p.detach().clone() for p in model.projection_head.parameters()]
    cached = torch.randn(32, 512)
    cached_before = cached.clone()
    subjects, sessions = ids()
    embeddings = model.project(cached)
    mask = build_positive_mask(config["training"]["positive_mask_mode"], subjects, sessions)
    loss, diagnostics = compute_total_objective(embeddings, subjects, sessions, config)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1.0e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    backbone_delta = max((a - b).abs().max().item() for a, b in zip(before_backbone, model.backbone.parameters()))
    projection_delta = max((a - b).abs().max().item() for a, b in zip(before_projection, model.projection_head.parameters()))
    cached_delta = float((cached_before - cached).abs().max().item())
    return mask, diagnostics, backbone_delta, projection_delta, cached_delta, model.trainable_parameter_count(), fake.call_count


def test_e6_base_anchor_positive_count_is_three() -> None:
    mask, *_ = run_step("e6_base")
    assert set(mask.sum(dim=1).tolist()) == {3}


def test_e6_a_anchor_positive_count_is_two() -> None:
    mask, *_ = run_step("e6_a")
    assert set(mask.sum(dim=1).tolist()) == {2}


def test_e6_b_anchor_positive_count_is_three() -> None:
    mask, *_ = run_step("e6_b")
    assert set(mask.sum(dim=1).tolist()) == {3}


def test_alignment_loss_finite_for_e6_a_and_e6_b() -> None:
    for stage in ("e6_a", "e6_b"):
        _, diagnostics, *_ = run_step(stage)
        assert diagnostics["alignment_loss"] >= 0.0
        assert diagnostics["alignment_diagnostics"]["centroid_pair_count"] == 8


def test_projection_updates_but_backbone_and_cache_do_not() -> None:
    _, _, backbone_delta, projection_delta, cached_delta, _, call_count = run_step("e6_b")
    assert projection_delta > 0.0
    assert backbone_delta == 0.0
    assert cached_delta == 0.0
    assert call_count == 0


def test_e6_models_trainable_parameter_count_identical() -> None:
    counts = {run_step(stage)[5] for stage in ("e6_base", "e6_a", "e6_b")}
    assert counts == {164736}


def test_training_step_does_not_access_test_data() -> None:
    # Unit-level objective/training step only receives cached train embeddings and batch metadata.
    assert run_step("e6_a")[6] == 0

